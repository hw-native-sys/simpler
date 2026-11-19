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
#include <algorithm>
#include <cstdint>
#include <vector>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int32_t waiters = rt_available_aiv_count() + static_cast<int32_t>(args.scalar<int64_t>(1));
    always_assert(waiters > 0);
    const auto submit = [&](int32_t role, const std::vector<TaskId> &dependencies) {
        CoreTaskArgs task;
        task.add_no_dep(args.tensor(0).ref(), args.tensor(1).ref(), args.tensor(2).ref());
        if (role == 2) task.add_output(args.tensor(3).ref());
        else task.add_no_dep(args.tensor(3).ref());
        task.add_scalar(role, args.scalar<uint64_t>(0));
        task.set_dependencies(dependencies.data(), static_cast<uint32_t>(dependencies.size()));
        return rt_submit_aiv_task(0, task).task_id();
    };
    std::vector<TaskId> waiting;
    for (int32_t i = 0; i < waiters; ++i)
        waiting.push_back(submit(0, {}));
    // This publisher shares the waiters' AIV pool and has no waiter dependency.
    (void)submit(1, {});
    std::vector<TaskId> joins;
    for (int32_t first = 0; first < waiters; first += 64) {
        CoreTaskArgs join;
        join.set_dependencies(waiting.data() + first, static_cast<uint32_t>(std::min(64, waiters - first)));
        joins.push_back(rt_submit_dummy_task(join).task_id());
    }
    (void)submit(2, joins);
}
