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

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t mask = args.scalar<int64_t>(0);
    const bool partial = args.scalar<int64_t>(1) != 0;
    const int64_t spin_limit = args.scalar<int64_t>(2);
    const int32_t available = mask == 2 ? rt_available_aiv_count() : rt_available_cluster_count();
    const int32_t blocks = partial && available > 1 ? available / 2 : available;
    const int32_t lanes = __builtin_popcount(static_cast<uint32_t>(mask));
    if (blocks < 1 || blocks * lanes > 320 || mask < 1 || mask > 7 || mask == 4) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "invalid mask cohort shape");
        return;
    }
    for (int i = 0; i < 3; ++i) {
        // Independent outputs avoid tensor dependencies serializing the three
        // cohorts: simultaneous entry obligations must be resolved by runtime.
        CoreTaskArgs task;
        task.add_inout(args.tensor(i).ref());
        task.add_scalar(int64_t{0});
        task.add_scalar(int64_t{320});
        task.add_scalar(spin_limit);
        task.add_scalar(static_cast<int64_t>(lanes));
        task.launch_spec.set_block_num(blocks);
        task.launch_spec.set_require_sync_start(true);
        if (mask == 1) rt_submit_aic_task(0, task);
        else if (mask == 2) rt_submit_aiv_task(1, task);
        else {
            MixedKernels kernels;
            if (mask & 1) kernels.aic_kernel_id = 0;
            if (mask & 2) kernels.aiv0_kernel_id = mask & 1 ? 2 : 1;
            if (mask & 4) kernels.aiv1_kernel_id = mask == 7 ? 3 : 2;
            rt_submit_task(kernels, task);
        }
    }
}
}
