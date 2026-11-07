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

#include "scheduler_resolve.h"

enum class SchedulerCompletionPhase : uint64_t {
    INITIALIZE,
    SELECT,
    RESOLVE,
    ENQUEUE,
    PREPARE,
    PUBLISH,
    PUBLISH_DIRECTORY,
    DRAINED,
    FAILED
};

enum class SchedulerCompletionResult : uint8_t { PENDING, DONE, ERROR };

// A bounded publisher owns at most eight references, or an already-linked
// bootstrap chain. Targets and counts are bounded by the ready-pool contract.
struct SchedulerReadyPublication {
    int64_t pending_tasks[8]{};
    int64_t pending_first{SCHEDULER_TASK_REF_INVALID};
    int64_t pending_last{SCHEDULER_TASK_REF_INVALID};
    uint32_t pending_count{0};
    uint32_t pending_target{0};
    uint32_t pending_link_count{0};
    uint8_t pending_kind{0};
    uint8_t pending_queue{0};
    uint8_t phase{0};
    uint8_t resume_phase{0};
};
static_assert(sizeof(SchedulerReadyPublication) == 96);

// A resolver owns its detached chain and unpublished batch until fully drained.
struct SchedulerCompletionCursor {
    SchedulerResolveCursor resolve{};
    int64_t current_waiter{SCHEDULER_TASK_REF_INVALID};
    int64_t detached_next{SCHEDULER_TASK_REF_INVALID};
    uint64_t visited_count{0};
    SchedulerReadyPublication publication{};
};

static_assert(sizeof(SchedulerCompletionCursor) == 152);
static_assert(
    std::is_standard_layout_v<SchedulerCompletionCursor> && std::is_trivially_copyable_v<SchedulerCompletionCursor>
);
