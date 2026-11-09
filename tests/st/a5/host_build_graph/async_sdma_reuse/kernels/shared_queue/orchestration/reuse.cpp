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
#include "../layout.h"

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}
extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const uint32_t lanes = rt_available_aiv_count();
    const uint64_t posts = args.scalar<uint64_t>(2);
    always_assert(lanes > 0 && lanes <= sdma_reuse::kMaxLanes && lanes % 2 == 0);
    always_assert(posts == 1 || posts == sdma_reuse::kPostsPerLane);
    CoreTaskArgs task;
    for (int i = 0; i < 4; ++i)
        task.add_no_dep(args.tensor(i).ref());
    task.add_scalar(args.scalar<uint64_t>(0), args.scalar<uint64_t>(1), posts, lanes, args.scalar<uint64_t>(3));
    task.launch_spec.set_block_num(lanes);
    task.launch_spec.set_require_sync_start(true);
    const TaskId submitted = rt_submit_aiv_task(0, task).task_id();
    task.set_dependencies(&submitted, 1);
    task.launch_spec.set_require_sync_start(false);
    (void)rt_submit_aiv_task(1, task);
}
