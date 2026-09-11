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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;
constexpr int32_t kTaskCapacity = 4096;
constexpr int32_t kKernelCheckedFanin = 4;

TaskId submit_task(
    const simpler::hbg::Tensor &task_state, int64_t graph_case, int64_t logical_id, const TaskId *deps,
    int32_t dep_count
) {
    CoreTaskArgs task;
    task.add_no_dep(task_state);
    task.add_scalar(logical_id);
    task.add_scalar(graph_case);
    task.add_scalar(static_cast<int64_t>(dep_count));
    for (int32_t i = 0; i < kKernelCheckedFanin; ++i) {
        task.add_scalar(i < dep_count ? static_cast<int64_t>(deps[i].local_id()) : -1);
    }
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    const bool use_aic = (logical_id & 1) == 0;
    return (use_aic ? rt_submit_aic_task(kAicFuncId, task) : rt_submit_aiv_task(kAivFuncId, task)).task_id();
}

void build_chain(const simpler::hbg::Tensor &task_state, int64_t task_count) {
    TaskId ids[kTaskCapacity];
    ids[0] = submit_task(task_state, 0, 0, nullptr, 0);
    for (int64_t task_id = 1; task_id < task_count; ++task_id) {
        ids[task_id] = submit_task(task_state, 0, task_id, &ids[task_id - 1], 1);
    }
}

void build_fanin32(const simpler::hbg::Tensor &task_state, int64_t task_count) {
    TaskId roots[32];
    const int32_t root_count = static_cast<int32_t>(std::min<int64_t>(task_count, 32));
    for (int32_t task_id = 0; task_id < root_count; ++task_id) {
        roots[task_id] = submit_task(task_state, 1, task_id, nullptr, 0);
    }
    for (int64_t task_id = root_count; task_id < task_count; ++task_id) {
        (void)submit_task(task_state, 1, task_id, roots, root_count);
    }
}

void build_multi_root(const simpler::hbg::Tensor &task_state, int64_t task_count) {
    for (int64_t task_id = 0; task_id < task_count; ++task_id) {
        (void)submit_task(task_state, 2, task_id, nullptr, 0);
    }
}

void build_random(const simpler::hbg::Tensor &task_state, int64_t task_count) {
    TaskId ids[kTaskCapacity];
    ids[0] = submit_task(task_state, 3, 0, nullptr, 0);
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t task_id = 1; task_id < task_count; ++task_id) {
        random = random * UINT64_C(6364136223846793005) + 1;
        const int32_t requested = static_cast<int32_t>(random % 5);
        TaskId deps[kKernelCheckedFanin];
        int32_t count = 0;
        for (int32_t edge = 0; edge < requested; ++edge) {
            random = random * UINT64_C(6364136223846793005) + 1;
            const int64_t producer = static_cast<int64_t>(random % static_cast<uint64_t>(task_id));
            bool duplicate = false;
            for (int32_t i = 0; i < count; ++i)
                duplicate = duplicate || deps[i] == ids[producer];
            if (!duplicate) deps[count++] = ids[producer];
        }
        ids[task_id] = submit_task(task_state, 3, task_id, deps, count);
    }
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{.expected_arg_count = 3};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const simpler::hbg::Tensor &task_state = args.tensor(0).ref();
    const int64_t graph_case = args.scalar<int64_t>(0);
    const int64_t task_count = args.scalar<int64_t>(1);
    if (task_count < 1 || task_count > kTaskCapacity) {
        rt_report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, "task_count must be in [1, %d], got %ld", kTaskCapacity, task_count
        );
        return;
    }
    switch (graph_case) {
    case 0:
        build_chain(task_state, task_count);
        break;
    case 1:
        build_fanin32(task_state, task_count);
        break;
    case 2:
        build_multi_root(task_state, task_count);
        break;
    case 3:
        build_random(task_state, task_count);
        break;
    default:
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported graph case %ld", graph_case);
        break;
    }
}

}  // extern "C"
