/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * Paged Attention Orchestration - 4D input shapes, N_UNROLL=64, 4 Tasks Per Group
 *
 * Batches up to N_UNROLL blocks per group. Each group submits exactly 4 tasks:
 *   1. QK matmul:  qi @ K^T for n_blocks → sij_buf (1, 1, q_tile, n_blocks * block_size)
 *   2. Softmax:    two-pass over sij_buf → pij_buf, mi, li
 *   3. PV matmul:  SplitK accumulated P @ V → oi_new (1, 1, q_tile, head_dim)
 *   4. Update:     online softmax accumulation with group-level mi, li, oi_new
 *
 * Memory Layout (4D throughout):
 *   Query: (batch, seq_len=1, num_heads, head_dim) bf16
 *   Key:   (total_blocks, block_size, kv_head_num, head_dim) bf16
 *   Value: (total_blocks, block_size, kv_head_num, head_dim) bf16
 *   Out:   (batch, seq_len=1, num_heads, head_dim) fp32
 */

#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"

#define N_UNROLL 64

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3

extern "C" {
/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // Read dimensions from tensor metadata
    // query: shape=[batch, seq_len, num_heads, head_dim]
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[2];
    uint64_t head_dim = orch_args.tensor(0).shapes[3];
    DataType data_type = orch_args.tensor(0).dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size = orch_args.tensor(1).shapes[1];

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args.scalar(0);
    uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;

    // External 4D tensors inherit shape/dtype from TaskArg (golden provides 4D).
    Tensor query = from_tensor_arg(orch_args.tensor(0));
    Tensor key_cache = from_tensor_arg(orch_args.tensor(1));
    Tensor value_cache = from_tensor_arg(orch_args.tensor(2));
    Tensor block_table = from_tensor_arg(orch_args.tensor(3));
    Tensor out = from_tensor_arg(orch_args.tensor(5));

    int *host_context_lens = orch_args.tensor(4).data_as<int>();

    // Loop-invariant shape descriptors: 4D data tiles (1, 1, q_tile, head_dim),
    // 3D scalar vectors (1, 1, q_tile).
    uint32_t tile4d_shapes[4] = {1, 1, (uint32_t)q_tile, (uint32_t)head_dim};
    uint32_t scalar_shapes[3] = {1, 1, (uint32_t)q_tile};
    TensorCreateInfo tile4d_ci(tile4d_shapes, 4, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(scalar_shapes, 3, DataType::FLOAT32);

    // Prefetch first block host_context_lens data into cache
    __builtin_prefetch(&host_context_lens[0], 0, 3);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        // Prefetch next block host_context_lens data while processing current batch
        if (b_idx + 1 < batch) {
            __builtin_prefetch(&host_context_lens[b_idx + 1], 0, 3);
        }
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE() {
                // 4D views into query/out, matching (1, 1, q_tile, head_dim).
                uint32_t view_shapes[4] = {1, 1, (uint32_t)q_tile, (uint32_t)head_dim};
                uint32_t view_offsets[4] = {(uint32_t)b_idx, 0, (uint32_t)(q_idx * q_tile), 0};
                Tensor qi = query.view(view_shapes, view_offsets);
                Tensor out_view = out.view(view_shapes, view_offsets, true);

                // Per-group accumulators: oi (4D data), mi_update/li_update (3D scalars).
                TaskOutputTensors alloc_outs = alloc_tensors(tile4d_ci, scalar_ci, scalar_ci);
                const Tensor &oi = alloc_outs.get_ref(0);
                const Tensor &li_update = alloc_outs.get_ref(1);
                const Tensor &mi_update = alloc_outs.get_ref(2);

                // Reusable Arg objects — reset() before each use avoids
                // repeated stack-frame construction in the inner loop.
                Arg params_qk, params_sf, params_pv, params_up;

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min((uint64_t)N_UNROLL, bn_this_batch - bn);

                    // Valid length for last block in this group
                    uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last = std::min(block_size, cur_seq - last_block_seq_start);

                    // === Task 1: Batched QK matmul — produces 4D sij_buf ===
                    uint32_t sij_buf_shapes[4] = {1, 1, (uint32_t)q_tile, (uint32_t)(n_blocks * block_size)};
                    TensorCreateInfo sij_buf_ci(sij_buf_shapes, 4, DataType::FLOAT32);

                    params_qk.reset();
                    params_qk.add_input(qi);
                    params_qk.add_input(key_cache);
                    params_qk.add_input(block_table);
                    params_qk.add_output(sij_buf_ci);
                    params_qk.add_scalar(n_blocks);
                    params_qk.add_scalar(b_idx * block_num + bn);
                    TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij_buf = qk_outs.get_ref(0);

                    // === Task 2: Two-pass softmax — produces 4D pij_buf, 3D mi, li ===
                    uint32_t pij_buf_shapes[4] = {1, 1, (uint32_t)q_tile, (uint32_t)(n_blocks * block_size)};
                    TensorCreateInfo pij_buf_ci(pij_buf_shapes, 4, data_type);

                    params_sf.reset();
                    params_sf.add_input(sij_buf);
                    params_sf.add_output(pij_buf_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_scalar(scale_value);
                    params_sf.add_scalar(n_blocks);
                    params_sf.add_scalar(valid_len_last);
                    TaskOutputTensors sf_outs = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_buf = sf_outs.get_ref(0);
                    const Tensor &mi = sf_outs.get_ref(1);
                    const Tensor &li = sf_outs.get_ref(2);

                    // === Task 3: SplitK PV matmul — produces 4D oi_new ===
                    params_pv.reset();
                    params_pv.add_input(pij_buf);
                    params_pv.add_input(value_cache);
                    params_pv.add_input(block_table);
                    params_pv.add_output(tile4d_ci);
                    params_pv.add_scalar(n_blocks);
                    params_pv.add_scalar(b_idx * block_num + bn);
                    TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_new = pv_outs.get_ref(0);

                    // === Task 4: Online update (per-group) ===
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;

                    params_up.reset();
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_new);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_inout(out_view);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                }
            }
        }
    }
}

}  // extern "C"
