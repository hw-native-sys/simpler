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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0), true);
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1), true);
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2), true);
    Tensor ext_config = from_tensor_arg(orch_args.tensor(3), true);

    int64_t *host_config = orch_args.tensor(3).data_as<int64_t>();
    int grid_k = static_cast<int>(host_config[1]);
    int num_groups = static_cast<int>(host_config[2]);
    int incore_loop = static_cast<int>(host_config[3]);
    int tile_size = static_cast<int>(host_config[0]);
    uint64_t group_tile_elems = static_cast<uint64_t>(incore_loop) * tile_size * tile_size;

    uint32_t group_shapes[1] = {static_cast<uint32_t>(group_tile_elems)};
    TensorCreateInfo group_ci(group_shapes, 1, DataType::FLOAT32);

    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        PTO2_SCOPE() {
            uint32_t c_view_offsets[1] = {static_cast<uint32_t>(static_cast<uint64_t>(group_idx) * group_tile_elems)};
            Tensor c_view = ext_C.view(group_shapes, c_view_offsets, true);
            PTO2TaskId prev_add = PTO2TaskId::invalid();

            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                for (int k_idx = 0; k_idx < grid_k; k_idx++) {
                    uint64_t ab_offset =
                        (static_cast<uint64_t>(group_idx) * grid_k + static_cast<uint64_t>(k_idx)) * group_tile_elems;
                    uint32_t ab_offsets[1] = {static_cast<uint32_t>(ab_offset)};
                    Tensor a_view = ext_A.view(group_shapes, ab_offsets, true);
                    Tensor b_view = ext_B.view(group_shapes, ab_offsets, true);

                    Arg params_gemm;
                    params_gemm.add_input(a_view);
                    params_gemm.add_input(b_view);
                    params_gemm.add_output(group_ci);
                    params_gemm.add_input(ext_config);
                    PTO2ManualSubmitResult gemm =
                        pto2_rt_submit_aic_task_manual(FUNC_GEMM_TILE, params_gemm);

                    Arg params_add;
                    params_add.add_inout(c_view);
                    params_add.add_input(gemm.outputs.get_ref(0));
                    params_add.add_input(ext_config);
                    PTO2ManualSubmitResult add = prev_add.is_valid()
                        ? pto2_rt_submit_aiv_task_manual_with_deps(FUNC_TILE_ADD, params_add, {gemm.task_id, prev_add})
                        : pto2_rt_submit_aiv_task_manual_with_deps(FUNC_TILE_ADD, params_add, {gemm.task_id});
                    prev_add = add.task_id;
                }
            }
        }
    }
}

}  // extern "C"
