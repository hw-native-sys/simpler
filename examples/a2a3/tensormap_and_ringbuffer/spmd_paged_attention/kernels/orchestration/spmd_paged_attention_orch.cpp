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
 * SPMD Paged Attention Orchestration (dual-vector subvector partitioning)
 *
 * Uses SPMD parallelism: block_num = batch * q_loop, where each logical
 * block handles one (batch_idx, q_tile_idx) position. Kernels use
 * get_block_idx() to compute their data offsets.
 *
 * QK and PV matmuls are AIC-only SPMD tasks. Softmax and online-update are
 * submitted as MIX tasks (AIC hub + AIV0 + AIV1) so the two AIV lanes within
 * a cluster each process one half of the 16 query rows, using
 * get_sub_block_id() to pick their 8-row slice. This mirrors the AscendC
 * reference (paged_attention_antiquantkv.h) subvector partitioning strategy.
 *
 * Memory Layout:
 *   Query:       (batch, num_heads, head_dim) - bfloat16
 *   Key/Value:   (total_blocks, block_size, kv_head_num, head_dim) - bfloat16
 *   Block Table: (batch, max_num_blocks_per_req) - int32
 *   Context Lens: (batch,) - int32
 *   Output:      (batch, num_heads, head_dim) - float32
 *
 * Scratch layout (runtime-allocated, indexed by block_idx * Q_TILE):
 *   sij:     (spmd_blocks * Q_TILE, block_size) float32
 *   pij:     (spmd_blocks * Q_TILE, block_size) data_type
 *   oi_new:  (spmd_blocks * Q_TILE, head_dim) float32
 *   mij/lij: (spmd_blocks * Q_TILE,) float32
 *   oi_acc/mi_acc/li_acc: persistent accumulators across bn loop
 */

#include <stddef.h>
#include <stdint.h>

#include <cinttypes>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_PV_MATMUL 1
#define FUNC_AIC_HUB 2
#define FUNC_SOFTMAX_PREPARE 3
#define FUNC_ONLINE_UPDATE 4
#define FUNC_AIV_HUB 5

static constexpr uint64_t Q_TILE = 16;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // query: shape=[batch, num_heads, head_dim]
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size = orch_args.tensor(1).shapes[1];

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t max_num_blocks_per_req = orch_args.tensor(3).shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_loop = (num_heads + Q_TILE - 1) / Q_TILE;
    int16_t spmd_block_num = static_cast<int16_t>(batch * q_loop);

    LOG_INFO(
        "SPMD PA: batch=%" PRIu64 " heads=%" PRIu64 " hd=%" PRIu64 " bs=%" PRIu64 " q_loop=%" PRIu64 " blocks=%d",
        batch, num_heads, head_dim, block_size, q_loop, spmd_block_num
    );

    // Wrap host-provided tensors
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    uint64_t total_kv_blocks = orch_args.tensor(1).shapes[0];
    uint64_t kv_total_rows = total_kv_blocks * block_size;

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t kv_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};

    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, kv_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, kv_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(max_num_blocks_per_req)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

    // Find max context_len for KV block loop bound
    uint64_t max_ctx = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint32_t idx[1] = {static_cast<uint32_t>(b)};
        uint64_t ctx = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, idx));
        if (ctx > max_ctx) max_ctx = ctx;
    }
    uint64_t max_bn = (max_ctx + block_size - 1) / block_size;

    // Scratch tensor create infos (sized for all SPMD blocks)
    uint32_t n_rows = static_cast<uint32_t>(spmd_block_num) * static_cast<uint32_t>(Q_TILE);
    uint32_t sij_shapes[2] = {n_rows, static_cast<uint32_t>(block_size)};
    uint32_t pij_shapes[2] = {n_rows, static_cast<uint32_t>(block_size)};
    uint32_t oi_new_shapes[2] = {n_rows, static_cast<uint32_t>(head_dim)};
    uint32_t scalar_shapes[1] = {n_rows};

    TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo pij_ci(pij_shapes, 2, data_type);
    TensorCreateInfo oi_new_ci(oi_new_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo mij_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo lij_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo acc_oi_ci(oi_new_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo acc_mi_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo acc_li_ci(scalar_shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE() {
        // Allocate persistent accumulators via no-op AIV hub
        Arg hub_args;
        hub_args.add_output(acc_oi_ci);
        hub_args.add_output(acc_mi_ci);
        hub_args.add_output(acc_li_ci);
        TaskOutputTensors hub_outs = pto2_rt_submit_aiv_task(FUNC_AIV_HUB, hub_args);
        const Tensor &oi_acc = hub_outs.get_ref(0);
        const Tensor &mi_acc = hub_outs.get_ref(1);
        const Tensor &li_acc = hub_outs.get_ref(2);

        for (uint64_t bn = 0; bn < max_bn; bn++) {
            uint64_t is_first = (bn == 0) ? 1 : 0;
            uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;

            // -- QK Matmul (AIC, SPMD) --
            Arg qk_args;
            qk_args.add_input(query);
            qk_args.add_input(key_cache);
            qk_args.add_input(block_table);
            qk_args.add_input(context_lens);
            qk_args.add_output(sij_ci);
            qk_args.add_scalar(static_cast<int64_t>(bn));
            qk_args.add_scalar(static_cast<int64_t>(num_heads));
            qk_args.add_scalar(static_cast<int64_t>(head_dim));
            qk_args.add_scalar(static_cast<int64_t>(block_size));
            qk_args.add_scalar(static_cast<int64_t>(max_num_blocks_per_req));
            qk_args.add_scalar(static_cast<int64_t>(q_loop));
            qk_args.launch_spec.set_block_num(spmd_block_num);
            TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, qk_args);
            const Tensor &sij = qk_outs.get_ref(0);

            // -- Softmax Prepare (MIX: AIC hub + AIV0 + AIV1, SPMD) --
            // AIV0 processes rows 0..7, AIV1 processes rows 8..15 of the Q_TILE
            // slice, discriminated via get_sub_block_id() inside the kernel.
            Arg sf_args;
            sf_args.add_input(sij);
            sf_args.add_input(context_lens);
            sf_args.add_output(pij_ci);
            sf_args.add_output(mij_ci);
            sf_args.add_output(lij_ci);
            sf_args.add_scalar(scale_value);
            sf_args.add_scalar(static_cast<int64_t>(bn));
            sf_args.add_scalar(static_cast<int64_t>(block_size));
            sf_args.add_scalar(static_cast<int64_t>(q_loop));
            sf_args.launch_spec.set_block_num(spmd_block_num);
            MixedKernels sf_mk;
            sf_mk.aic_kernel_id = FUNC_AIC_HUB;
            sf_mk.aiv0_kernel_id = FUNC_SOFTMAX_PREPARE;
            sf_mk.aiv1_kernel_id = FUNC_SOFTMAX_PREPARE;
            TaskOutputTensors sf_outs = pto2_rt_submit_task(sf_mk, sf_args);
            const Tensor &pij = sf_outs.get_ref(0);
            const Tensor &mij = sf_outs.get_ref(1);
            const Tensor &lij = sf_outs.get_ref(2);

            // -- PV Matmul (AIC, SPMD) --
            Arg pv_args;
            pv_args.add_input(pij);
            pv_args.add_input(value_cache);
            pv_args.add_input(block_table);
            pv_args.add_input(context_lens);
            pv_args.add_output(oi_new_ci);
            pv_args.add_scalar(static_cast<int64_t>(bn));
            pv_args.add_scalar(static_cast<int64_t>(num_heads));
            pv_args.add_scalar(static_cast<int64_t>(head_dim));
            pv_args.add_scalar(static_cast<int64_t>(block_size));
            pv_args.add_scalar(static_cast<int64_t>(max_num_blocks_per_req));
            pv_args.add_scalar(static_cast<int64_t>(q_loop));
            pv_args.launch_spec.set_block_num(spmd_block_num);
            TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, pv_args);
            const Tensor &oi_new = pv_outs.get_ref(0);

            // -- Online Update (MIX: AIC hub + AIV0 + AIV1, SPMD) --
            // Row-independent online softmax update: AIV0 updates rows 0..7 of
            // the Q_TILE accumulator slice, AIV1 updates rows 8..15.
            Arg up_args;
            up_args.add_input(mij);
            up_args.add_input(lij);
            up_args.add_input(oi_new);
            up_args.add_inout(mi_acc);
            up_args.add_inout(li_acc);
            up_args.add_inout(oi_acc);
            up_args.add_inout(out);
            up_args.add_scalar(is_first);
            up_args.add_scalar(is_last);
            up_args.add_scalar(static_cast<int64_t>(num_heads));
            up_args.add_scalar(static_cast<int64_t>(head_dim));
            up_args.add_scalar(static_cast<int64_t>(q_loop));
            up_args.launch_spec.set_block_num(spmd_block_num);
            MixedKernels up_mk;
            up_mk.aic_kernel_id = FUNC_AIC_HUB;
            up_mk.aiv0_kernel_id = FUNC_ONLINE_UPDATE;
            up_mk.aiv1_kernel_id = FUNC_ONLINE_UPDATE;
            pto2_rt_submit_task(up_mk, up_args);
        }
    }

    LOG_INFO("SPMD PA: %" PRIu64 " KV iters x 4 tasks, blocks=%d", max_bn, static_cast<int>(spmd_block_num));
}

}  // extern "C"
