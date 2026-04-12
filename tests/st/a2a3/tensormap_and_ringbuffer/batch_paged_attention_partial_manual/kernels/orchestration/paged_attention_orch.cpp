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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;
    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t block_num = orch_args.tensor(3).shapes[1];
    uint64_t scale_value = orch_args.scalar(0);
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;

    Tensor query = from_tensor_arg(orch_args.tensor(0), true);
    Tensor key_cache = from_tensor_arg(orch_args.tensor(1), true);
    Tensor value_cache = from_tensor_arg(orch_args.tensor(2), true);
    Tensor block_table = from_tensor_arg(orch_args.tensor(3), true);
    Tensor context_lens = from_tensor_arg(orch_args.tensor(4), true);
    Tensor out = from_tensor_arg(orch_args.tensor(5), true);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) {
            max_bn = bn_b;
        }
    }

    constexpr uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
            if (chunk_bc > IN_CORE_BATCH) {
                chunk_bc = IN_CORE_BATCH;
            }
            uint64_t batch_start = chunk_idx * IN_CORE_BATCH;

            PTO2_SCOPE() {
                uint32_t oi_batch_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t scalar_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                TensorCreateInfo oi_batch_ci(oi_batch_shapes, 2, DataType::FLOAT32, true);
                TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32, true);
                TaskOutputTensors accumulators = alloc_tensors(oi_batch_ci, scalar_ci, scalar_ci);
                const Tensor &oi_batch = accumulators.get_ref(0);
                const Tensor &li_batch = accumulators.get_ref(1);
                const Tensor &mi_batch = accumulators.get_ref(2);

                uint32_t sij_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(block_size)};
                uint32_t vec_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                uint32_t oi_new_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
                TensorCreateInfo pij_ci(sij_shapes, 2, data_type);
                TensorCreateInfo vec_ci(vec_shapes, 1, DataType::FLOAT32);
                TensorCreateInfo oi_new_ci(oi_new_shapes, 2, DataType::FLOAT32);
                PTO2TaskId prev_update = PTO2TaskId::invalid();

                PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
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
                        PTO2ManualSubmitResult qk = pto2_rt_submit_aic_task_manual(FUNC_QK_MATMUL, params_qk);

                        Arg params_sf;
                        params_sf.add_input(qk.outputs.get_ref(0));
                        params_sf.add_input(context_lens);
                        params_sf.add_output(pij_ci);
                        params_sf.add_output(vec_ci);
                        params_sf.add_output(vec_ci);
                        params_sf.add_scalar(scale_value);
                        params_sf.add_scalar(chunk_bc);
                        params_sf.add_scalar(bn);
                        params_sf.add_scalar(batch_start);
                        PTO2ManualSubmitResult sf =
                            pto2_rt_submit_aiv_task_manual_with_deps(FUNC_SOFTMAX_PREPARE, params_sf, {qk.task_id});

                        Arg params_pv;
                        params_pv.add_input(sf.outputs.get_ref(0));
                        params_pv.add_input(value_cache);
                        params_pv.add_input(block_table);
                        params_pv.add_output(oi_new_ci);
                        params_pv.add_scalar(chunk_bc);
                        params_pv.add_scalar(bn);
                        params_pv.add_scalar(block_num);
                        params_pv.add_scalar(batch_start);
                        PTO2ManualSubmitResult pv =
                            pto2_rt_submit_aic_task_manual_with_deps(FUNC_PV_MATMUL, params_pv, {sf.task_id});

                        uint64_t is_first = (bn == 0) ? 1 : 0;
                        uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                        Arg params_up;
                        params_up.add_input(sf.outputs.get_ref(1));
                        params_up.add_input(sf.outputs.get_ref(2));
                        params_up.add_input(pv.outputs.get_ref(0));
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
                        PTO2ManualSubmitResult up = prev_update.is_valid()
                            ? pto2_rt_submit_aiv_task_manual_with_deps(
                                  FUNC_ONLINE_UPDATE, params_up, {sf.task_id, pv.task_id, prev_update}
                              )
                            : pto2_rt_submit_aiv_task_manual_with_deps(
                                  FUNC_ONLINE_UPDATE, params_up, {sf.task_id, pv.task_id}
                              );
                        prev_update = up.task_id;
                    }
                }
            }
        }
    }
}

}  // extern "C"
