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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "scheduler/scheduler_async_layout.h"
#include "core_async_capability.h"

struct AsyncTaskPlanInput {
    int64_t task_ref{-1};
    uint32_t block_count{0};
    LaneMask active_mask{};
    std::array<uint16_t, 3> capacities{};
};

// Every active leaf lane requires an explicit valid registration contract.
// Zero capacity is the projection of SYNCHRONOUS, never a missing-capability fallback.
inline bool
async_task_apply_capabilities(AsyncTaskPlanInput *task, const std::array<CoreAsyncCapability, 3> &capabilities) {
    if (task == nullptr) return false;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!task->active_mask.has(static_cast<SubtaskSlot>(lane))) {
            task->capacities[lane] = 0;
            continue;
        }
        const auto &capability = capabilities[lane];
        if (!core_async_capability_valid(capability) ||
            (capability.mode != CoreAsyncMode::SYNCHRONOUS && capability.mode != CoreAsyncMode::DEFERRED))
            return false;
        task->capacities[lane] = capability.mode == CoreAsyncMode::SYNCHRONOUS ? 0 : capability.max_conditions;
    }
    return true;
}

// This is a separate data-visibility proof, not the task's own capability.
// Every executable node receives this proof from the actual expanded DAG.
inline bool async_mark_input_visibility(
    const std::vector<std::vector<int64_t>> &fanin, const std::vector<int64_t> &order,
    const std::vector<LaneMask> &producers, std::vector<SchedulerTaskMetadata> *metadata
) {
    if (metadata == nullptr || metadata->size() != fanin.size() || order.size() != fanin.size() ||
        producers.size() != fanin.size())
        return false;
    std::vector<bool> seen(fanin.size(), false), may_publish_async(fanin.size(), false);
    for (const int64_t ref : order) {
        if (ref < 0 || static_cast<uint64_t>(ref) >= fanin.size() || seen[ref]) return false;
        bool incoming = false;
        for (const int64_t predecessor : fanin[ref]) {
            if (predecessor < 0 || static_cast<uint64_t>(predecessor) >= fanin.size() || !seen[predecessor])
                return false;
            incoming = incoming || may_publish_async[predecessor];
        }
        auto &task = (*metadata)[ref];
        task.flags &= ~SCHEDULER_TASK_ASYNC_INPUT;
        if (scheduler_task_is_executable(task.flags) && !scheduler_task_is_inline(task.flags) && incoming)
            task.flags |= SCHEDULER_TASK_ASYNC_INPUT;
        may_publish_async[ref] = incoming || !producers[ref].is_empty();
        seen[ref] = true;
    }
    return true;
}

struct AsyncTaskHostPlan {
    SchedulerAsyncTaskData data;
    uint64_t entries_offset{0};
    std::array<uint16_t, 3> capacities{};
};

struct AsyncCompletionHostPlan {
    uint64_t begin_offset{0};
    uint64_t layout_offset{0};
    uint64_t total_bytes{0};
    uint64_t initialized_bytes{0};
    SchedulerAsyncLayout layout{};
    std::vector<AsyncTaskHostPlan> tasks;
};

// Inputs already refer to the expanded resident task namespace. The run epoch
// belongs to the final commit, not an earlier preparation attempt.
// Failure leaves out untouched. Without nonzero-capacity groups, no bytes are appended.
bool plan_async_completion(
    uint64_t begin_offset, uint64_t run_epoch, uint32_t task_count, const std::vector<AsyncTaskPlanInput> &tasks,
    AsyncCompletionHostPlan *out, std::string *error, uint32_t sdma_guard_count = 0
);

// Initializes only the appended prefix; existing scheduler/native Graph data
// and the reserved suffix remain intact. All references are offsets from image.
bool initialize_async_completion(void *image, uint64_t image_bytes, const AsyncCompletionHostPlan &plan);
