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

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 8};
}
__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    CoreTaskArgs task;
    for (int32_t i = 0; i < 4; ++i)
        task.add_no_dep(args.tensor(i).ref());
    task.add_scalar(args.scalar<int64_t>(0), args.scalar<int64_t>(1), args.scalar<int64_t>(2), args.scalar<int64_t>(3));
    if (args.scalar<int64_t>(3) == 2) {
        rt_submit_aiv_task(3, task);
        return;
    }
    const TaskId transfer = rt_submit_aiv_task(0, task).task_id();
    CoreTaskArgs probe;
    for (int32_t i = 0; i < 4; ++i)
        probe.add_no_dep(args.tensor(i).ref());
    probe.add_scalar(
        args.scalar<int64_t>(0), args.scalar<int64_t>(1), args.scalar<int64_t>(2), args.scalar<int64_t>(3)
    );
    probe.dump(args.tensor(3).ref());
    rt_submit_aiv_task(2, probe);
    task.set_dependencies(&transfer, 1);
    rt_submit_aiv_task(1, task);
}
}
