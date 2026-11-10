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

struct AsyncTaskPlanInput {
    int64_t task_ref{-1};
    uint32_t block_count{0};
    LaneMask active_mask{};
    std::array<uint16_t, 3> capacities{MAX_COMPLETIONS_PER_TASK, MAX_COMPLETIONS_PER_TASK, MAX_COMPLETIONS_PER_TASK};
};

struct AsyncTaskHostPlan {
    SchedulerAsyncTaskData data;
    uint64_t entries_offset{0};
    uint32_t block_entry_bytes{0};
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
// belongs to the final commit, not an earlier parked prepare attempt.
// Failure leaves out untouched. Without nonzero-capacity groups, no bytes are appended.
bool plan_async_completion(
    uint64_t begin_offset, uint64_t run_epoch, uint32_t task_count, const std::vector<AsyncTaskPlanInput> &tasks,
    AsyncCompletionHostPlan *out, std::string *error, uint32_t sdma_guard_count = 0
);

// Initializes only the appended prefix; existing scheduler/native Graph data
// and the reserved suffix remain intact. All references are offsets from image.
bool initialize_async_completion(void *image, uint64_t image_bytes, const AsyncCompletionHostPlan &plan);
