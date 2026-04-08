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

#include <algorithm>
#include <cstdint>

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
aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;    // NOLINT(readability/casting)
    (void)orch_thread_index;  // NOLINT(readability/casting)

    uint64_t batch = orch_args.tensor(0).shapes[0];
    uint64_t num_heads = orch_args.tensor(0).shapes[1];
    uint64_t head_dim = orch_args.tensor(0).shapes[2];
    DataType data_type = orch_args.tensor(0).dtype;
    uint64_t block_size = orch_args.tensor(1).shapes[1];
    uint64_t block_num = orch_args.tensor(3).shapes[1];
    uint64_t scale_value = orch_args.scalar(0);

    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;

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
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    int *host_block_table = orch_args.tensor(3).data_as<int>();
    int *host_context_lens = orch_args.tensor(4).data_as<int>();

    uint32_t tile2d_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t scalar_shapes[1] = {static_cast<uint32_t>(q_tile)};
    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(block_size)};
    TensorCreateInfo tile2d_ci(tile2d_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32);
    TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo pij_f16_ci(sij_shapes, 2, data_type);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE() {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = query.view(tile2d_shapes, qi_offsets);
                Tensor out_view = out.view(tile2d_shapes, out_view_offsets);

                PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                    Arg params_inplace;
                    params_inplace.add_output(tile2d_ci);
                    params_inplace.add_output(scalar_ci);
                    params_inplace.add_output(scalar_ci);
                    PTO2ManualSubmitResult hub_outs = pto2_rt_submit_aiv_task_manual(FUNC_AIV_HUB, params_inplace);
                    const Tensor &oi = hub_outs.outputs.get_ref(0);
                    const Tensor &li_update = hub_outs.outputs.get_ref(1);
                    const Tensor &mi_update = hub_outs.outputs.get_ref(2);
                    PTO2TaskId prev_update_task = hub_outs.task_id;

                    for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                        uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                        uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);

                        uint32_t kv_shapes[2] = {
                            static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim)
                        };
                        uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
                        Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                        Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                        Arg params_qk;
                        params_qk.add_input(qi);
                        params_qk.add_input(kj);
                        params_qk.add_output(sij_ci);
                        PTO2ManualSubmitResult qk_outs = pto2_rt_submit_aic_task_manual(FUNC_QK_MATMUL, params_qk);
                        const Tensor &sij = qk_outs.outputs.get_ref(0);

                        uint32_t sij_valid_shapes[2] = {
                            static_cast<uint32_t>(q_tile), static_cast<uint32_t>(valid_len)
                        };
                        uint32_t sij_valid_offsets[2] = {0, 0};
                        Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);

                        Arg params_sf;
                        params_sf.add_input(sij_valid);
                        params_sf.add_output(pij_f16_ci);
                        params_sf.add_output(scalar_ci);
                        params_sf.add_output(scalar_ci);
                        params_sf.add_scalar(scale_value);
                        PTO2ManualSubmitResult sf_outs =
                            pto2_rt_submit_aiv_task_manual(FUNC_SOFTMAX_PREPARE, params_sf);
                        const Tensor &pij_f16 = sf_outs.outputs.get_ref(0);
                        const Tensor &mi = sf_outs.outputs.get_ref(1);
                        const Tensor &li = sf_outs.outputs.get_ref(2);

                        Arg params_pv;
                        params_pv.add_input(pij_f16);
                        params_pv.add_input(vj);
                        params_pv.add_output(tile2d_ci);
                        PTO2ManualSubmitResult pv_outs = pto2_rt_submit_aic_task_manual(FUNC_PV_MATMUL, params_pv);
                        const Tensor &oi_tmp = pv_outs.outputs.get_ref(0);

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
                        params_up.add_scalar(is_first);
                        params_up.add_scalar(is_last);
                        PTO2ManualSubmitResult up_outs =
                            pto2_rt_submit_aiv_task_manual(FUNC_ONLINE_UPDATE, params_up);

                        pto2_rt_add_dependency(qk_outs.task_id, sf_outs.task_id);
                        pto2_rt_add_dependency(sf_outs.task_id, pv_outs.task_id);
                        pto2_rt_add_dependency(sf_outs.task_id, up_outs.task_id);
                        pto2_rt_add_dependency(pv_outs.task_id, up_outs.task_id);
                        pto2_rt_add_dependency(prev_update_task, up_outs.task_id);
                        prev_update_task = up_outs.task_id;
                    }
                }
            }
        }
    }
}

}  // extern "C"
