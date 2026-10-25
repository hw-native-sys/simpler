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
    return OrchestrationConfig{.expected_arg_count = 8};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &inputs) {
    CoreTaskArgs args;
    args.add_inout(inputs.tensor(0).ref());
    args.add_inout(inputs.tensor(1).ref());
    const auto blocks = inputs.scalar<int64_t>(0);
    args.add_scalar(blocks);
    args.add_scalar(inputs.scalar<int64_t>(2));
    args.add_scalar(inputs.scalar<int64_t>(3));
    args.add_scalar(inputs.scalar<int64_t>(4));
    args.add_scalar(inputs.scalar<int64_t>(5));
    const auto kind = inputs.scalar<int64_t>(1);
    args.launch_spec.set_block_num(static_cast<int16_t>(kind == 2 ? 1 : blocks));
    if (kind == 2) rt_submit_task(MixedKernels{0, 1, INVALID_KERNEL_ID}, args);
    else if (kind == 0) rt_submit_aic_task(0, args);
    else rt_submit_aiv_task(1, args);
}
