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
    const uint32_t count = static_cast<uint32_t>(args.scalar<int64_t>(0));
    const uint32_t chain_width = std::min(count, UINT32_C(256));
    always_assert(chain_width != 0);
    const auto &out = args.tensor(0).ref();
    CoreTaskArgs waiter_args;
    waiter_args.add_no_dep(out);
    waiter_args.add_scalar(0, 0, chain_width);
    const TaskId waiter = rt_submit_aiv_task(0, waiter_args).task_id();
    always_assert(waiter.local_id() == 0);
    CoreTaskArgs started_args;
    started_args.add_no_dep(out);
    started_args.add_scalar(3, 0, chain_width);
    const TaskId started = rt_submit_aiv_task(0, started_args).task_id();
    // Independent chains keep ready work available without allocating a
    // completion-counter chain for every background invocation.
    std::vector<TaskId> tails(chain_width, started);
    for (uint32_t index = 0; index < count; ++index) {
        CoreTaskArgs work;
        work.add_no_dep(out);
        work.add_scalar(1, index, chain_width);
        const uint32_t chain = index % chain_width;
        work.set_dependencies(&tails[chain], 1);
        const TaskId task = rt_submit_aiv_task(0, work).task_id();
        tails[chain] = task;
    }
    CoreTaskArgs consumer;
    consumer.add_no_dep(out);
    consumer.add_scalar(2, 0, chain_width);
    consumer.set_dependencies(&waiter, 1);
    (void)rt_submit_aiv_task(0, consumer);
}
