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

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_NOOP 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;    // NOLINT(readability/casting)
    (void)orch_thread_index;  // NOLINT(readability/casting)

    Tensor tensor = from_tensor_arg(orch_args.tensor(0));
    uint64_t mode = orch_args.scalar(0);
    uint32_t idx[1] = {0};

    switch (mode) {
        case 1:
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                PTO2_SCOPE(PTO2ScopeMode::MANUAL) {}
            }
            break;
        case 2:
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                (void)get_tensor_data<float>(tensor, 1, idx);  // NOLINT(readability/casting)
            }
            break;
        case 3:
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) { set_tensor_data(tensor, 1, idx, 1.0f); }
            break;
        case 4:
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
                PTO2TaskId invalid = PTO2TaskId::invalid();
                pto2_rt_add_dependency(invalid, invalid);
            }
            break;
        case 5:
            PTO2_SCOPE(PTO2ScopeMode::MANUAL) { PTO2_SCOPE() {} }
            break;
        default:
            PTO2_SCOPE() {}
            break;
    }
}
}
