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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scheduler_types.h"

enum class SchedulerRouteResult : uint64_t {
    READY_TO_ENQUEUE = 1,
    WAITING = 2,
    COMPLETED = 3,
    ERROR = 4,
    YIELD = 5,
};

// The exclusive resolver owns this value. A handoff persists it under the
// resolver lease; it never borrows a core's stack or a waiter's mutable link.
struct SchedulerRouteCursor {
    int64_t task_ref{-1};
    int64_t producer_ref{-1};
    int32_t next_fanin{-1};
    uint32_t fanin_count{0};
    SchedulerRouteResult result{SchedulerRouteResult::YIELD};
};

struct SchedulerRouteProgress {
    SchedulerRouteResult result;
    uint32_t work_units;
};

static_assert(sizeof(SchedulerRouteCursor) == 32 && alignof(SchedulerRouteCursor) == 8);
static_assert(offsetof(SchedulerRouteCursor, task_ref) == 0 && offsetof(SchedulerRouteCursor, producer_ref) == 8);
static_assert(offsetof(SchedulerRouteCursor, next_fanin) == 16 && offsetof(SchedulerRouteCursor, result) == 24);
static_assert(std::is_standard_layout_v<SchedulerRouteCursor> && std::is_trivially_copyable_v<SchedulerRouteCursor>);

enum class SchedulerReadyPublishPhase : uint32_t { COLLECTING, PREPARED, DONE, FAILED };
enum class SchedulerReadyAppendResult : uint8_t { ADDED, DIRECT, FULL, ERROR };
enum class SchedulerReadyPublishResult : uint8_t { DONE, YIELD, ERROR };
inline constexpr uint32_t SCHEDULER_READY_PUBLICATION_CAPACITY = 8;
enum class SchedulerReadyDestination : uint32_t { AIC, AIV, CONTROL, INVALID = UINT32_MAX };

inline __aicore__ SchedulerReadyDestination scheduler_ready_destination(uint32_t queue_class) {
    if (queue_class == 0) return SchedulerReadyDestination::AIC;
    if (queue_class == 1) return SchedulerReadyDestination::AIV;
    return SchedulerReadyDestination::INVALID;
}

// The resolver owns this entire record. No part of its continuation belongs
// to SchedulerLocalState.
struct SchedulerReadyPublication {
    int64_t refs[SCHEDULER_READY_PUBLICATION_CAPACITY]{};
    uint64_t inbox_offset{0};
    uint32_t count{0};
    SchedulerReadyDestination destination{SchedulerReadyDestination::INVALID};
    uint32_t target{0};
    SchedulerReadyPublishPhase phase{SchedulerReadyPublishPhase::COLLECTING};
};

static_assert(sizeof(SchedulerReadyPublication) == 88 && alignof(SchedulerReadyPublication) == 8);
static_assert(offsetof(SchedulerReadyPublication, inbox_offset) == 64);
static_assert(offsetof(SchedulerReadyPublication, count) == 72 && offsetof(SchedulerReadyPublication, phase) == 84);
static_assert(
    std::is_standard_layout_v<SchedulerReadyPublication> && std::is_trivially_copyable_v<SchedulerReadyPublication>
);

enum class SchedulerResolvePhase : uint32_t { START, SELECT, ROUTE, APPEND, PUBLISH, DONE, FAILED };
enum class SchedulerResolveResult : uint8_t { DONE, YIELD, ERROR };

struct SchedulerResolveCursor {
    int64_t producer_ref{-1};
    int64_t next_waiter{-1};
    int64_t waiter_ref{-1};
    int64_t saved_next{-1};
    SchedulerRouteCursor route{};
    SchedulerReadyPublication publication{};
    SchedulerResolvePhase phase{SchedulerResolvePhase::START};
    uint32_t visited_count{0};
};

struct SchedulerResolveProgress {
    SchedulerResolveResult result;
    uint32_t work_units;
};

static_assert(sizeof(SchedulerResolveCursor) == 160 && alignof(SchedulerResolveCursor) == 8);
static_assert(offsetof(SchedulerResolveCursor, route) == 32 && offsetof(SchedulerResolveCursor, publication) == 64);
static_assert(offsetof(SchedulerResolveCursor, phase) == 152);
static_assert(
    std::is_standard_layout_v<SchedulerResolveCursor> && std::is_trivially_copyable_v<SchedulerResolveCursor>
);
