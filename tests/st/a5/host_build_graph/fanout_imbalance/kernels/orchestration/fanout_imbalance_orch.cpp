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

/**
 * fanout_imbalance orchestration: one root, then `task_count - 1` independent
 * consumers that all depend on it and on nothing else.
 *
 * The shape concentrates work on purpose. Every consumer becomes ready at the
 * same instant, inside the wake-list walk of whichever Scheduler observed the
 * root's completion, and that Scheduler publishes all of them to the inbox it
 * owns. A Scheduler steals only inside its own directory shard, so the work is
 * reachable by a fraction of the cores rather than all of them — which is the
 * load imbalance this scene test exists to create.
 *
 * Consumers alternate AIC / AIV by task id parity so the spread also crosses
 * core types.
 *
 * Args layout: [state] + {task_count, repeats} scalars.
 */

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;

TaskId submit(
    const simpler::hbg::Tensor &state, int64_t task_id, int64_t producer, int64_t repeats, const TaskId *deps,
    int32_t dep_count
) {
    CoreTaskArgs task;
    task.add_no_dep(state);
    task.add_scalar(task_id);
    task.add_scalar(producer);
    task.add_scalar(repeats);
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    const bool use_aic = (task_id & 1) == 0;
    return (use_aic ? rt_submit_aic_task(kAicFuncId, task) : rt_submit_aiv_task(kAivFuncId, task)).task_id();
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &state = orch_args.tensor(0).ref();
    const int64_t task_count = orch_args.scalar<int64_t>(0);
    const int64_t repeats = orch_args.scalar<int64_t>(1);

    if (task_count < 2) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "fanout needs a root and at least one consumer, got %ld", task_count
        );
        return;
    }
    if (repeats < 0) {
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "negative repeats %ld", repeats);
        return;
    }

    const TaskId root = submit(state, 0, -1, repeats, nullptr, 0);
    for (int64_t task_id = 1; task_id < task_count; ++task_id) {
        (void)submit(state, task_id, 0, repeats, &root, 1);
    }
}

}  // extern "C"
