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
    return OrchestrationConfig{.expected_arg_count = 5};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const bool partial = args.scalar<int64_t>(0) != 0;
    const int64_t spin_limit = args.scalar<int64_t>(1);
    for (int i = 0; i < 3; ++i) {
        const int32_t available = i == 2 ? rt_available_aiv_count() : rt_available_cluster_count();
        const int32_t blocks = partial && available > 1 ? available / 2 : available;
        const int64_t lanes = i == 1 ? 3 : 1;
        if (blocks < 1 || blocks * lanes > 320) {
            rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "invalid independent cohort width");
            return;
        }
        CoreTaskArgs task;
        task.add_inout(args.tensor(i).ref());
        task.add_scalar(int64_t{0});
        task.add_scalar(int64_t{320});
        task.add_scalar(spin_limit);
        task.add_scalar(lanes);
        task.launch_spec.set_block_num(blocks);
        task.launch_spec.set_require_sync_start(true);
        if (i == 0) rt_submit_aic_task(0, task);
        else if (i == 2) rt_submit_aiv_task(1, task);
        else {
            MixedKernels kernels;
            kernels.aic_kernel_id = 0;
            kernels.aiv0_kernel_id = 2;
            kernels.aiv1_kernel_id = 3;
            rt_submit_task(kernels, task);
        }
    }
}
}
