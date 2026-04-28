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
 * Paged Attention Orchestration — Per-Block Version
 * (aicpu_build_graph variant: explicit add_dependency, no TensorMap)
 *
 * For each batch, for each head tile, for each KV block:
 *   1. QK matmul:  qi @ kj^T → sij (q_tile, block_size)
 *   2. Softmax:    sij → pij, mi, li
 *   3. PV matmul:  pij @ vj → oi_tmp (q_tile, head_dim)
 *   4. Update:     online softmax accumulation
 *
 * Dependency graph per block:
 *   QK → Softmax → PV → Update
 *              └──────────→ Update
 *   Update(prev block) ──→ Update(this block)
 *   Hub(init) ────────────→ Update(first block)
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(PTO2Runtime *rt, const ChipStorageTaskArgs &orch_args) {
    // Read dimensions from tensor metadata
    // query: shape=[batch, num_heads, head_dim]
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size = orch_args.tensor(1).shapes[1];

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;

    // Reshape tensors for kernel consumption (2D flattened)
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    uint64_t total_blocks_count = orch_args.tensor(1).shapes[0];

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t value_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type, false);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type, false);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type, false);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    int *host_block_table = orch_args.tensor(3).data_as<int>();
    int *host_context_lens = orch_args.tensor(4).data_as<int>();

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
                uint32_t mi_shapes[1] = {static_cast<uint32_t>(q_tile)};
                uint32_t qi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);

                // Hub task: zero-initialize accumulators
                Arg args_inplace;
                args_inplace.add_output(TensorCreateInfo(oi_shapes, 2, DataType::FLOAT32));
                args_inplace.add_output(TensorCreateInfo(li_shapes, 1, DataType::FLOAT32));
                args_inplace.add_output(TensorCreateInfo(mi_shapes, 1, DataType::FLOAT32));
                SubmitResult r_hub = rt_submit_aiv_task(rt, FUNC_AIV_HUB, args_inplace);
                const Tensor &oi = r_hub.outputs.get_ref(0);
                const Tensor &li_update = r_hub.outputs.get_ref(1);
                const Tensor &mi_update = r_hub.outputs.get_ref(2);

                PTO2TaskId prev_update_task = r_hub.task_id;

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);

                    // KV views for this block
                    uint32_t kv_shapes[2] = {static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim)};
                    uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                    // === Task 1: QK matmul ===
                    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(block_size)};

                    Arg args_qk;
                    args_qk.add_input(qi);
                    args_qk.add_input(kj);
                    args_qk.add_output(TensorCreateInfo(sij_shapes, 2, DataType::FLOAT32));
                    SubmitResult r_qk = rt_submit_aic_task(rt, FUNC_QK_MATMUL, args_qk);

                    // === Task 2: Softmax ===
                    uint32_t sij_valid_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(valid_len)};
                    uint32_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = r_qk.outputs.get_ref(0).view(sij_valid_shapes, sij_valid_offsets);

                    Arg args_sf;
                    args_sf.add_input(sij_valid);
                    args_sf.add_output(TensorCreateInfo(sij_shapes, 2, data_type));
                    args_sf.add_output(TensorCreateInfo(mi_shapes, 1, DataType::FLOAT32));
                    args_sf.add_output(TensorCreateInfo(li_shapes, 1, DataType::FLOAT32));
                    args_sf.add_scalar(scale_value);
                    SubmitResult r_sf = rt_submit_aiv_task(rt, FUNC_SOFTMAX_PREPARE, args_sf);
                    rt_add_dependency(rt, r_qk.task_id, r_sf.task_id);

                    // === Task 3: PV matmul ===
                    uint32_t oi_tmp_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};

                    Arg args_pv;
                    args_pv.add_input(r_sf.outputs.get_ref(0));
                    args_pv.add_input(vj);
                    args_pv.add_output(TensorCreateInfo(oi_tmp_shapes, 2, DataType::FLOAT32));
                    SubmitResult r_pv = rt_submit_aic_task(rt, FUNC_PV_MATMUL, args_pv);
                    rt_add_dependency(rt, r_sf.task_id, r_pv.task_id);

                    // === Task 4: Online update ===
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    Arg args_up;
                    args_up.add_input(r_sf.outputs.get_ref(1));
                    args_up.add_input(r_sf.outputs.get_ref(2));
                    args_up.add_input(r_pv.outputs.get_ref(0));
                    args_up.add_inout(mi_update);
                    args_up.add_inout(li_update);
                    args_up.add_inout(oi);
                    args_up.add_inout(out_view);
                    args_up.add_scalar(is_first);
                    args_up.add_scalar(is_last);
                    SubmitResult r_up = rt_submit_aiv_task(rt, FUNC_ONLINE_UPDATE, args_up);
                    rt_add_dependency(rt, r_sf.task_id, r_up.task_id);
                    rt_add_dependency(rt, r_pv.task_id, r_up.task_id);
                    rt_add_dependency(rt, prev_update_task, r_up.task_id);

                    prev_update_task = r_up.task_id;
                }
            }
        }
    }
}

}  // extern "C"
