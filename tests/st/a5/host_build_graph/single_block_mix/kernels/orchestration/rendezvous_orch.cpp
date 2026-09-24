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

#include <cstdint>
#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 3};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const auto mask = static_cast<uint32_t>(orch_args.scalar<int64_t>(0));
    const uint32_t shape[1] = {48};
    for (uint32_t task = 0; task < static_cast<uint32_t>(orch_args.scalar<int64_t>(1)); ++task) {
        const uint32_t offset[1] = {task * 48};
        auto state = orch_args.tensor(0).ref().view(shape, offset);
        CoreTaskArgs args;
        args.add_inout(state);
        args.add_scalar(static_cast<int64_t>(mask));
        MixedKernels kernels;
        kernels.aic_kernel_id = (mask & 1U) != 0 ? 0 : INVALID_KERNEL_ID;
        kernels.aiv0_kernel_id = 1;
        kernels.aiv1_kernel_id = 2;
        rt_submit_task(kernels, args);
    }
}
