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

#include <type_traits>

#include "scheduler_types.h"

enum class SchedulerRouteResult : uint64_t {
    READY_TO_ENQUEUE = 1,
    WAITING = 2,
    COMPLETED = 3,
    ERROR = 4,
    YIELD = 5,
};

enum class SchedulerResolvePhase : uint8_t { INITIALIZE, FANIN, LINK, READY, WAITING, COMPLETED, ERROR };

// This cursor holds only values, so a FINISH owner can publish it before handing
// resolution to another core. It never retains a pointer into a waiter's links.
struct SchedulerResolveCursor {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    int64_t producer_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t next_fanin{0};
    uint32_t fanin_count{0};
    SchedulerResolvePhase phase{SchedulerResolvePhase::INITIALIZE};
    bool validate_current_state{true};
    uint8_t reserved[6]{};
};

static_assert(sizeof(SchedulerResolveCursor) == 32, "resolve cursor must remain compact");
static_assert(
    std::is_standard_layout_v<SchedulerResolveCursor> && std::is_trivially_copyable_v<SchedulerResolveCursor>
);

inline __aicore__ SchedulerResolveCursor scheduler_resolve_begin(int64_t task_ref, bool validate_state = true) {
    SchedulerResolveCursor cursor{};
    cursor.task_ref = task_ref;
    cursor.validate_current_state = validate_state;
    return cursor;
}
