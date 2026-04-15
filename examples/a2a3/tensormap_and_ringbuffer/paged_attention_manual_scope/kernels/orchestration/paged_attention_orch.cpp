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
 * Paged Attention Orchestration Function - 16x16 Version
 *
 * Simplified for 16x16 framework-generated matmul kernels.
 * Each block processes a single 16x16 matmul operation.
 *
 * Memory Layout:
 *   Query: (batch, 16, 16) - one 16x16 tile per batch bf16
 *   Key:   (total_blocks, 16, 16) - stored as K^T for direct matmul bf16
 *   Value: (total_blocks, 16, 16) - direct format bf16
 *
 * This file compiles as a standalone .so with zero runtime link dependencies.
 * All runtime calls go through the PTO2RuntimeOps function-pointer table.
 */

#include <stddef.h>
#include <stdint.h>

#include <cinttypes>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3

inline bool validate_paged_attention_shape(uint64_t num_heads, uint64_t block_size, uint64_t head_dim) {
    if (head_dim == 16 && block_size == 16 && num_heads != 0 && (num_heads % 16) == 0) {
        return true;
    }

    pto2_rt_report_fatal(
        PTO2_ERROR_INVALID_ARGS,
        "example paged_attention supports only block_size=16, head_dim=16, and num_heads multiple of 16; got "
        "num_heads=%" PRIu64 ", block_size=%" PRIu64 ", head_dim=%" PRIu64,
        num_heads, block_size, head_dim
    );
    return false;
}

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
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
    uint64_t q_tile = 16;
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    uint64_t elem_size = get_element_size(data_type);
    if (!validate_paged_attention_shape(num_heads, block_size, head_dim)) {
        return;
    }

    // Reshape tensors for kernel consumption (2D flattened)
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    // Compute kv_total_rows from key_cache tensor metadata
    uint64_t total_blocks_count = orch_args.tensor(1).shapes[0];
    uint64_t kv_total_rows = total_blocks_count * block_size;

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);
    LOG_DEBUG("query=%s", query.dump().c_str());
    LOG_DEBUG("key_cache=%s", key_cache.dump().c_str());
    LOG_DEBUG("value_cache=%s", value_cache.dump().c_str());
    LOG_DEBUG("out=%s", out.dump().c_str());

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

    // Create infos are loop-invariant — shapes depend only on q_tile/head_dim/block_size
    uint32_t tile2d_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t scalar_shapes[1] = {static_cast<uint32_t>(q_tile)};
    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(block_size)};
    TensorCreateInfo tile2d_ci(tile2d_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo pij_bf16_ci(sij_shapes, 2, data_type);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b_idx)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                uint32_t cur_offset = static_cast<uint32_t>(b_idx * q_head_num + q_idx * q_tile);

                uint32_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(tile2d_shapes, qi_offsets);
                uint32_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = out.view(tile2d_shapes, out_view_offsets);

                TaskOutputTensors alloc_outs = alloc_tensors(tile2d_ci, scalar_ci, scalar_ci);
                const Tensor &oi = alloc_outs.get_ref(0);
                const Tensor &li_update = alloc_outs.get_ref(1);
                const Tensor &mi_update = alloc_outs.get_ref(2);
                PTO2TaskId alloc_task = alloc_outs.task_id();
                PTO2TaskId prev_update_task = PTO2TaskId::invalid();

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint32_t bt_idx[2] = {static_cast<uint32_t>(b_idx), static_cast<uint32_t>(bn)};
                    uint64_t cur_block_idx = static_cast<uint64_t>(get_tensor_data<int32_t>(block_table, 2, bt_idx));
                    uint64_t valid_len =
                        block_size < (cur_seq - bn * block_size) ? block_size : (cur_seq - bn * block_size);
                    uint32_t kv_shapes[2] = {static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim)};
                    uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                    Arg params_qk;
                    params_qk.add_input(qi);
                    params_qk.add_input(kj);
                    params_qk.add_output(sij_ci);
                    TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij = qk_outs.get_ref(0);

                    uint32_t sij_valid_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(valid_len)};
                    uint32_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    Arg params_sf;
                    params_sf.add_input(sij_valid);
                    params_sf.add_output(pij_bf16_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_dep(qk_outs.task_id());
                    params_sf.add_scalar(scale_value);
                    TaskOutputTensors sf_outs = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_bf16 = sf_outs.get_ref(0);
                    const Tensor &mi = sf_outs.get_ref(1);
                    const Tensor &li = sf_outs.get_ref(2);

                    Arg params_pv;
                    params_pv.add_input(pij_bf16);
                    params_pv.add_input(vj);
                    params_pv.add_output(tile2d_ci);
                    params_pv.add_dep(sf_outs.task_id());
                    TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_tmp = pv_outs.get_ref(0);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    Arg params_up;
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_tmp);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_inout(out_view);
                    params_up.add_dep(alloc_task);
                    params_up.add_dep(sf_outs.task_id());
                    params_up.add_dep(pv_outs.task_id());
                    if (prev_update_task.is_valid()) {
                        params_up.add_dep(prev_update_task);
                    }
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    TaskOutputTensors update_outs = pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                    prev_update_task = update_outs.task_id();
                }
            }
        }
    }

    LOG_INFO("tasks submitted for batch=%" PRIu64 ", num_heads=%" PRIu64, batch, num_heads);
}

}  // extern "C"
