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

#define N_UNROLL 64

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
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type, false);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type, false);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type, false);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    int *host_block_table = orch_args.tensor(3).data_as<int>();
    int *host_context_lens = orch_args.tensor(4).data_as<int>();

    uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
    TensorCreateInfo tile2d_ci(oi_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_noinit_ci(li_shapes, 1, DataType::FLOAT32, false);
    TensorCreateInfo scalar_ci(li_shapes, 1, DataType::FLOAT32);

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        int *bt_base = host_block_table + b_idx * block_num;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE() {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t qi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets, true);

                Arg params_inplace;
                params_inplace.add_output(tile2d_ci);
                params_inplace.add_output(scalar_noinit_ci);
                params_inplace.add_output(scalar_noinit_ci);
                TaskOutputTensors hub_outs = pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_inplace);
                const Tensor &oi = hub_outs.get_ref(0);
                const Tensor &li_update = hub_outs.get_ref(1);
                const Tensor &mi_update = hub_outs.get_ref(2);

                Arg params_qk;
                Arg params_sf;
                Arg params_pv;
                Arg params_up;

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);
                    uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last = std::min(block_size, cur_seq - last_block_seq_start);

                    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                        uint32_t sij_buf_shapes[2] = {
                            static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                        };
                        TensorCreateInfo sij_buf_ci(sij_buf_shapes, 2, DataType::FLOAT32);

                        params_qk.reset();
                        params_qk.add_input(qi);
                        params_qk.add_input(key_cache);
                        params_qk.add_output(sij_buf_ci);
                        params_qk.add_scalar(n_blocks);
                        params_qk.add_scalar(reinterpret_cast<uint64_t>(bt_base + bn));
                        PTO2ManualSubmitResult qk_outs = pto2_rt_submit_aic_task_manual(FUNC_QK_MATMUL, params_qk);

                        uint32_t pij_buf_shapes[2] = {
                            static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                        };
                        TensorCreateInfo pij_buf_ci(pij_buf_shapes, 2, data_type);

                        params_sf.reset();
                        params_sf.add_input(qk_outs.outputs.get_ref(0));
                        params_sf.add_output(pij_buf_ci);
                        params_sf.add_output(scalar_ci);
                        params_sf.add_output(scalar_ci);
                        params_sf.add_scalar(scale_value);
                        params_sf.add_scalar(n_blocks);
                        params_sf.add_scalar(valid_len_last);
                        PTO2ManualSubmitResult sf_outs =
                            pto2_rt_submit_aiv_task_manual(FUNC_SOFTMAX_PREPARE, params_sf);

                        params_pv.reset();
                        params_pv.add_input(sf_outs.outputs.get_ref(0));
                        params_pv.add_input(value_cache);
                        params_pv.add_output(tile2d_ci);
                        params_pv.add_scalar(n_blocks);
                        params_pv.add_scalar(reinterpret_cast<uint64_t>(bt_base + bn));
                        PTO2ManualSubmitResult pv_outs = pto2_rt_submit_aic_task_manual(FUNC_PV_MATMUL, params_pv);

                        uint64_t is_first = (bn == 0) ? 1 : 0;
                        uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;

                        params_up.reset();
                        params_up.add_input(sf_outs.outputs.get_ref(1));
                        params_up.add_input(sf_outs.outputs.get_ref(2));
                        params_up.add_input(pv_outs.outputs.get_ref(0));
                        params_up.add_inout(mi_update);
                        params_up.add_inout(li_update);
                        params_up.add_inout(oi);
                        params_up.add_inout(out_view);
                        params_up.add_scalar(is_first);
                        params_up.add_scalar(is_last);
                        PTO2ManualSubmitResult up_outs = pto2_rt_submit_aiv_task_manual(FUNC_ONLINE_UPDATE, params_up);

                        pto2_rt_add_dependency(qk_outs.task_id, sf_outs.task_id);
                        pto2_rt_add_dependency(sf_outs.task_id, pv_outs.task_id);
                        pto2_rt_add_dependency(sf_outs.task_id, up_outs.task_id);
                        pto2_rt_add_dependency(pv_outs.task_id, up_outs.task_id);
                    }
                }
            }
        }
    }
}

}  // extern "C"
