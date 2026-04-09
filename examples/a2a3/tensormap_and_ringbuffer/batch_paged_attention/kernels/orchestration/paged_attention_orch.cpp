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
 * Batch Paged Attention Orchestration Function - 16x16 Version
 *
 * Chunked batched architecture: the full batch is split into chunks of
 * IN_CORE_BATCH size. Each chunk's QK/SF/PV/UP tasks are independent
 * and can be scheduled to different cores in parallel.
 *
 * Memory Layout:
 *   Query: (batch * num_heads, head_dim) fp16
 *   Key:   (total_blocks, block_size, head_dim) fp16 (stored as K^T for QK)
 *   Value: (total_blocks, block_size, head_dim) fp16
 */

#include <stddef.h>
#include <stdint.h>

#include <cinttypes>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
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
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;

    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t block_num = orch_args.tensor(3).shapes[1];

    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_tile = 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    uint64_t elem_size = get_element_size(data_type);

    LOG_INFO("batch_paged_attention: batch=%" PRIu64 ", num_heads=%" PRIu64, batch, num_heads);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).data_as<void>(), bt_shapes, 2, DataType::INT32, false);

    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).data_as<void>(), cl_shapes, 1, DataType::INT32, false);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    // Reshape tensors for kernel consumption (2D flattened)
    void *query_ptr = orch_args.tensor(0).data_as<void>();
    void *kc_ptr = orch_args.tensor(1).data_as<void>();
    void *vc_ptr = orch_args.tensor(2).data_as<void>();
    void *out_ptr = orch_args.tensor(5).data_as<void>();

    uint64_t total_blocks_count = orch_args.tensor(1).shapes[0];
    uint64_t kv_total_rows = total_blocks_count * block_size;

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};

    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32, true);

    uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;
            uint64_t batch_start = chunk_idx * IN_CORE_BATCH;

            PTO2_SCOPE() {
                uint32_t tile2d_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t scalar_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                uint32_t sij_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(block_size)};
                TensorCreateInfo tile2d_ci(tile2d_shapes, 2, DataType::FLOAT32);
                TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32);
                TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
                TensorCreateInfo pij_ci(sij_shapes, 2, data_type);

                TaskOutputTensors alloc_outs = alloc_tensors(tile2d_ci, scalar_ci, scalar_ci);
                const Tensor &oi_batch = alloc_outs.get_ref(0);
                const Tensor &li_batch = alloc_outs.get_ref(1);
                const Tensor &mi_batch = alloc_outs.get_ref(2);

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    Arg params_qk;
                    params_qk.add_input(query);
                    params_qk.add_input(key_cache);
                    params_qk.add_input(block_table);
                    params_qk.add_output(sij_ci);
                    params_qk.add_scalar(chunk_bc);
                    params_qk.add_scalar(bn);
                    params_qk.add_scalar(q_offset);
                    params_qk.add_scalar(block_num);
                    params_qk.add_scalar(num_heads);
                    params_qk.add_scalar(batch_start);
                    TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij_b = qk_outs.get_ref(0);

                    Arg params_sf;
                    params_sf.add_input(sij_b);
                    params_sf.add_input(context_lens);
                    params_sf.add_output(pij_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_output(scalar_ci);
                    params_sf.add_scalar(scale_value);
                    params_sf.add_scalar(chunk_bc);
                    params_sf.add_scalar(bn);
                    params_sf.add_scalar(batch_start);
                    TaskOutputTensors sf_outs = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_b = sf_outs.get_ref(0);
                    const Tensor &mij_b = sf_outs.get_ref(1);
                    const Tensor &lij_b = sf_outs.get_ref(2);

                    Arg params_pv;
                    params_pv.add_input(pij_b);
                    params_pv.add_input(value_cache);
                    params_pv.add_input(block_table);
                    params_pv.add_output(tile2d_ci);
                    params_pv.add_scalar(chunk_bc);
                    params_pv.add_scalar(bn);
                    params_pv.add_scalar(block_num);
                    params_pv.add_scalar(batch_start);
                    TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_new_b = pv_outs.get_ref(0);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    Arg params_up;
                    params_up.add_input(mij_b);
                    params_up.add_input(lij_b);
                    params_up.add_input(oi_new_b);
                    params_up.add_inout(mi_batch);
                    params_up.add_inout(li_batch);
                    params_up.add_inout(oi_batch);
                    params_up.add_inout(out);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    params_up.add_scalar(chunk_bc);
                    params_up.add_scalar(q_offset);
                    params_up.add_scalar(num_heads);
                    params_up.add_scalar(batch_start);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                }
            }
        }
    }

    LOG_INFO(
        "batch_paged_attention: %" PRIu64 " tasks (batch=%" PRIu64 ", max_bn=%" PRIu64 ", chunks=%" PRIu64
        ", IN_CORE_BATCH=%" PRIu64 ")",
        static_cast<uint64_t>(num_chunks * (1 + max_bn * 4)), batch, max_bn, num_chunks, IN_CORE_BATCH
    );
}

}  // extern "C"
