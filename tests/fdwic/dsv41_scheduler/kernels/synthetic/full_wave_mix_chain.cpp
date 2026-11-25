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

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t clusters = rt_available_cluster_count();
    const int32_t vectors = rt_available_aiv_count();
    if (clusters != 32 || vectors != 64) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "full-wave chain needs 32 clusters and 64 AIVs");
        return;
    }
    struct Stage {
        int32_t blocks;
        MixedKernels kernels;
        int32_t members;
    };
    const MixedKernels cube{0, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    const MixedKernels vector{INVALID_KERNEL_ID, 1, INVALID_KERNEL_ID};
    const MixedKernels full{0, 1, 1};
    const MixedKernels pair{0, 1, INVALID_KERNEL_ID};
    const Stage stages[] = {{clusters, cube, 1}, {clusters, full, 3}, {vectors, vector, 1},
                            {clusters, full, 3}, {1, pair, 2},        {clusters, full, 3}};

    int32_t previous_members = clusters;
    for (int32_t i = 0; i < 6; ++i) {
        const auto &stage = stages[i];
        CoreTaskArgs task;
        task.add_input(args.tensor(i).ref());
        task.add_inout(args.tensor(i + 1).ref());
        task.add_scalar(stage.members);
        task.add_scalar(previous_members);
        task.add_scalar((i + 1) * args.scalar<int64_t>(0));
        task.launch_spec.set_block_num(stage.blocks);
        task.launch_spec.set_require_sync_start(stage.blocks > 1);
        rt_submit_task(stage.kernels, task);
        previous_members = stage.blocks * stage.members;
    }
}
