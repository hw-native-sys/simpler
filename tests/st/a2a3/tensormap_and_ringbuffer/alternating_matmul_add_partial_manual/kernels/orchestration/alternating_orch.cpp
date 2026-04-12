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

#define FUNC_MATMUL 0
#define FUNC_ADD 1

static constexpr uint64_t MATMUL_ELEMS = 128 * 128;
static constexpr uint64_t ADD_ELEMS = 128 * 128;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 11,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_A = from_tensor_arg(orch_args.tensor(0), true);
    Tensor ext_B = from_tensor_arg(orch_args.tensor(1), true);
    Tensor ext_C = from_tensor_arg(orch_args.tensor(2), true);
    Tensor ext_X = from_tensor_arg(orch_args.tensor(3), true);
    Tensor ext_Y = from_tensor_arg(orch_args.tensor(4), true);
    Tensor ext_Z = from_tensor_arg(orch_args.tensor(5), true);

    int batch = static_cast<int>(orch_args.scalar(0));
    int m_tasks = static_cast<int>(orch_args.scalar(1));
    int n_tasks = static_cast<int>(orch_args.scalar(2));
    int matmul_batch = static_cast<int>(orch_args.scalar(3));
    int add_batch = static_cast<int>(orch_args.scalar(4));

    int total_matmul_tasks = batch * m_tasks;
    int total_add_tasks = batch * n_tasks;
    int num_matmul_groups = total_matmul_tasks / matmul_batch;
    int num_add_groups = total_add_tasks / add_batch;
    int max_groups = (num_matmul_groups > num_add_groups) ? num_matmul_groups : num_add_groups;

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        for (int group_idx = 0; group_idx < max_groups; group_idx++) {
            if (group_idx < num_matmul_groups) {
                uint64_t offset = static_cast<uint64_t>(group_idx * matmul_batch) * MATMUL_ELEMS;
                uint64_t group_size = static_cast<uint64_t>(matmul_batch) * MATMUL_ELEMS;
                uint32_t group_shapes[1] = {static_cast<uint32_t>(group_size)};
                uint32_t offsets[1] = {static_cast<uint32_t>(offset)};

                Tensor a_view = ext_A.view(group_shapes, offsets, true);
                Tensor b_view = ext_B.view(group_shapes, offsets, true);
                Tensor c_view = ext_C.view(group_shapes, offsets, true);

                Arg params_matmul;
                params_matmul.add_input(a_view);
                params_matmul.add_input(b_view);
                params_matmul.add_output(c_view);
                (void)pto2_rt_submit_aic_task_manual(FUNC_MATMUL, params_matmul);  // NOLINT(readability/casting)
            }

            if (group_idx < num_add_groups) {
                uint64_t offset = static_cast<uint64_t>(group_idx * add_batch) * ADD_ELEMS;
                uint64_t group_size = static_cast<uint64_t>(add_batch) * ADD_ELEMS;
                uint32_t group_shapes[1] = {static_cast<uint32_t>(group_size)};
                uint32_t offsets[1] = {static_cast<uint32_t>(offset)};

                Tensor x_view = ext_X.view(group_shapes, offsets, true);
                Tensor y_view = ext_Y.view(group_shapes, offsets, true);
                Tensor z_view = ext_Z.view(group_shapes, offsets, true);

                Arg params_add;
                params_add.add_input(x_view);
                params_add.add_input(y_view);
                params_add.add_output(z_view);
                (void)pto2_rt_submit_aiv_task_manual(FUNC_ADD, params_add);  // NOLINT(readability/casting)
            }
        }
    }
}

}  // extern "C"
