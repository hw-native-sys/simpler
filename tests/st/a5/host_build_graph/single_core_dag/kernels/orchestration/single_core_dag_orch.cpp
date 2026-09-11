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

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int32_t kAicFuncId = 0;
constexpr int32_t kAivFuncId = 1;
constexpr int32_t kTaskCapacity = 64;

TaskId submit_task(
    const simpler::hbg::Tensor &task_state, int64_t core_mode, int64_t logical_id, const TaskId *deps, int32_t dep_count
) {
    CoreTaskArgs task;
    task.add_no_dep(task_state);
    task.add_scalar(logical_id);
    uint64_t producer_mask = 0;
    for (int32_t i = 0; i < dep_count; ++i)
        producer_mask |= UINT64_C(1) << deps[i].local_id();
    task.add_scalar(producer_mask);
    task.set_dependencies(deps, static_cast<uint32_t>(dep_count));
    const bool use_aic = core_mode == 0 || (core_mode == 2 && (logical_id & 1) == 0);
    return (use_aic ? rt_submit_aic_task(kAicFuncId, task) : rt_submit_aiv_task(kAivFuncId, task)).task_id();
}

void build_chain(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    TaskId ids[kTaskCapacity];
    ids[0] = submit_task(task_state, core_type, 0, nullptr, 0);
    for (int64_t task_id = 1; task_id < kTaskCapacity; ++task_id)
        ids[task_id] = submit_task(task_state, core_type, task_id, &ids[task_id - 1], 1);
}

void build_diamonds(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    TaskId ids[25];
    ids[0] = submit_task(task_state, core_type, 0, nullptr, 0);
    int32_t next = 1;
    int32_t prior = 0;
    for (int32_t layer = 0; layer < 8; ++layer) {
        int32_t left = next++;
        int32_t right = next++;
        ids[left] = submit_task(task_state, core_type, left, &ids[prior], 1);
        ids[right] = submit_task(task_state, core_type, right, &ids[prior], 1);
        TaskId deps[2] = {ids[left], ids[right]};
        int32_t join = next++;
        ids[join] = submit_task(task_state, core_type, join, deps, 2);
        prior = join;
    }
}

void build_fanout(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    TaskId root = submit_task(task_state, core_type, 0, nullptr, 0);
    for (int64_t task_id = 1; task_id < kTaskCapacity; ++task_id)
        (void)submit_task(task_state, core_type, task_id, &root, 1);
}

void build_dense_fanin(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    TaskId roots[32];
    for (int64_t task_id = 0; task_id < 32; ++task_id)
        roots[task_id] = submit_task(task_state, core_type, task_id, nullptr, 0);
    for (int64_t task_id = 32; task_id < kTaskCapacity; ++task_id)
        (void)submit_task(task_state, core_type, task_id, roots, 32);
}

void build_multi_root(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    for (int64_t task_id = 0; task_id < kTaskCapacity; ++task_id)
        (void)submit_task(task_state, core_type, task_id, nullptr, 0);
}

void build_random(const simpler::hbg::Tensor &task_state, int64_t core_type) {
    TaskId ids[kTaskCapacity];
    ids[0] = submit_task(task_state, core_type, 0, nullptr, 0);
    uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (int64_t task_id = 1; task_id < kTaskCapacity; ++task_id) {
        random = random * UINT64_C(6364136223846793005) + 1;
        int32_t requested = static_cast<int32_t>(random % 5);
        TaskId deps[4];
        int32_t count = 0;
        for (int32_t edge = 0; edge < requested; ++edge) {
            random = random * UINT64_C(6364136223846793005) + 1;
            int64_t producer = static_cast<int64_t>(random % static_cast<uint64_t>(task_id));
            bool duplicate = false;
            for (int32_t i = 0; i < count; ++i)
                duplicate = duplicate || deps[i] == ids[producer];
            if (!duplicate) deps[count++] = ids[producer];
        }
        ids[task_id] = submit_task(task_state, core_type, task_id, deps, count);
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
    int64_t graph_case = args.scalar<int64_t>(0);
    int64_t core_type = args.scalar<int64_t>(1);
    switch (graph_case) {
    case 0:
        build_chain(task_state, core_type);
        break;
    case 1:
        build_diamonds(task_state, core_type);
        break;
    case 2:
        build_fanout(task_state, core_type);
        break;
    case 3:
        build_dense_fanin(task_state, core_type);
        break;
    case 4:
        build_multi_root(task_state, core_type);
        break;
    case 5:
        build_random(task_state, core_type);
        break;
    default:
        rt_report_fatal(SIMPLER_ERROR_INVALID_ARGS, "unsupported graph case %ld", graph_case);
        break;
    }
}

}  // extern "C"
