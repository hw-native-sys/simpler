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
    return OrchestrationConfig{.expected_arg_count = 2};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    constexpr uint32_t line_words = 32;
    const int32_t cores = rt_available_aiv_count();
    const int32_t waiters = cores + static_cast<int32_t>(args.scalar<int64_t>(0));
    always_assert(cores > 1 && waiters > 0);
    const int32_t chunks = (waiters + 63) / 64;
    const uint32_t shape[]{static_cast<uint32_t>(2 * waiters + 3 + chunks) * line_words};
    TensorCreateInfo state_info(shape, 1, DataType::INT32);
    const auto &out = args.tensor(0).ref();
    CoreTaskArgs initialize;
    initialize.add_output(state_info);
    initialize.add_no_dep(out);
    initialize.add_scalar(0, 0, waiters);
    const auto initialized = rt_submit_aiv_task(0, initialize);
    const auto state = initialized.get_ref(0);
    const TaskId init = initialized.task_id();
    const auto submit = [&](int32_t role, int32_t index, const std::vector<TaskId> &dependencies) {
        CoreTaskArgs task;
        task.add_no_dep(state, out);
        task.add_scalar(role, index, waiters);
        task.set_dependencies(dependencies.data(), static_cast<uint32_t>(dependencies.size()));
        return rt_submit_aiv_task(0, task).task_id();
    };
    std::vector<TaskId> waiting;
    for (int32_t i = 0; i < waiters; ++i)
        waiting.push_back(submit(1, i, {init}));
    // Both ordinary work and the notifier are submitted after every waiter,
    // use the same AIV pool, and depend only on initialization.
    const auto probe = submit(2, 0, {init});
    const auto publisher = submit(3, 0, {init});
    std::vector<TaskId> consumers;
    for (int32_t i = 0; i < waiters; ++i)
        consumers.push_back(submit(4, i, {waiting[i]}));
    std::vector<TaskId> summaries{probe, publisher};
    for (int32_t i = 0; i < waiters; i += 64) {
        const auto end = std::min(i + 64, waiters);
        summaries.push_back(submit(5, i, {consumers.begin() + i, consumers.begin() + end}));
    }
    (void)submit(6, 0, summaries);
}
