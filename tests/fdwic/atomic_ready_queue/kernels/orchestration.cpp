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

#include "orchestration_api.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 5};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &inputs) {
    const int64_t blocks = inputs.scalar<int64_t>(0);
    const int64_t mask = inputs.scalar<int64_t>(1);
    const int64_t epochs = inputs.scalar<int64_t>(2);
    if (blocks <= 0 || epochs <= 0 || epochs > 8 ||
        !((mask == 1 && blocks <= 32) || (mask == 2 && blocks <= 64) || (mask == 7 && blocks <= 32))) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "invalid atomic ready queue probe shape");
        return;
    }
    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        CoreTaskArgs args;
        args.add_inout(inputs.tensor(0).ref());
        args.add_inout(inputs.tensor(1).ref());
        args.add_scalar(blocks);
        args.add_scalar(mask);
        args.add_scalar(epoch);
        args.launch_spec.set_block_num(static_cast<int16_t>(blocks));
        args.launch_spec.set_require_sync_start(true);
        if (mask == 1) rt_submit_aic_task(0, args);
        else if (mask == 2) rt_submit_aiv_task(1, args);
        else rt_submit_task(MixedKernels{0, 1, 2}, args);
    }
}
