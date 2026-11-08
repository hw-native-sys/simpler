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

#include <cstdint>

namespace async_case {

constexpr int64_t kConditionLimit = 64;  // Public registration contract, not a queue capacity.
constexpr int64_t kSignalStride = 32;
constexpr int64_t kConditionsPerExecution = kConditionLimit + 1;
constexpr int64_t kPayloadWords = 32;
constexpr int64_t kObservationWords = 32;
constexpr int64_t kControlStride = 16;
constexpr int64_t kControlCells = 16;
constexpr int32_t kPoison = -1000000007;

enum class Scenario : int64_t {
    BASIC,
    TWO_CONDITIONS,
    FANIN,
    PROGRESS,
    VISIBILITY,
    GRAPH,
    NEVER,
    FULL_SYNC_START,
    SPARSE_READY
};
enum class Action : int64_t { PUBLISH, CONSUME, PREREQUISITE, WARM, RESCUE };
enum class Control : int64_t {
    ARRIVED,
    PUBLISHED,
    PREREQUISITE,
    WARMED,
    RESCUED,
    CAPACITY,
    EXECUTIONS,
    PENDING,
    BUSINESS_AT_CONSUMER,
    BUSINESS_WHILE_PENDING
};
enum class Observation : int64_t { VISITS = 0, BODY_DONE = 1, MISSING_CONTEXT = 2, EARLY = 3, TIMEOUT = 4 };

struct Plan {
    int64_t scenario;
    int64_t conditions;
    int64_t waiters;
    int64_t blocks;
    int64_t mask;
    int64_t expected;
    int64_t epoch;
    int64_t delay_ms;
    int64_t tail_ms;
    int64_t sync_start;
    int64_t instances;
    int64_t skip;
    int64_t capacity_delta;
    int64_t slots;
    int64_t registration_epoch{1};
};

}  // namespace async_case
