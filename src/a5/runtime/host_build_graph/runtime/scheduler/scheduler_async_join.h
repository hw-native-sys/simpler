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

#include "scheduler_memory.h"

inline constexpr uint32_t SCHEDULER_ASYNC_JOIN_LIMIT = INT16_MAX;
inline constexpr uint64_t SCHEDULER_ASYNC_JOIN_FAILED = UINT64_C(1) << 63;
inline constexpr uint64_t SCHEDULER_ASYNC_JOIN_COUNTS = UINT64_C(0xffffffff);

struct alignas(128) SchedulerAsyncTaskJoin {
    volatile uint64_t word{0};
};

static_assert(sizeof(SchedulerAsyncTaskJoin) == 128);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncTaskJoin> && std::is_trivially_copyable_v<SchedulerAsyncTaskJoin>
);

enum class SchedulerAsyncJoinResult : uint8_t { PENDING, FINISH_OWNER, FAILED, INVALID_STATE, OVERFLOW, UNDERFLOW };

struct SchedulerAsyncJoinOutcome {
    SchedulerAsyncJoinResult result;
    uint64_t previous;
};

constexpr inline __aicore__ uint32_t scheduler_async_join_groups(uint64_t word) {
    return static_cast<uint32_t>(word & UINT64_C(0xffff));
}

constexpr inline __aicore__ uint32_t scheduler_async_join_blocks(uint64_t word) {
    return static_cast<uint32_t>((word >> 16) & UINT64_C(0xffff));
}

constexpr inline __aicore__ bool scheduler_async_join_valid(uint64_t word) {
    return (word & ~(SCHEDULER_ASYNC_JOIN_FAILED | SCHEDULER_ASYNC_JOIN_COUNTS)) == 0 &&
           ((word & SCHEDULER_ASYNC_JOIN_FAILED) != 0 ||
            (scheduler_async_join_groups(word) <= SCHEDULER_ASYNC_JOIN_LIMIT &&
             scheduler_async_join_blocks(word) <= SCHEDULER_ASYNC_JOIN_LIMIT));
}

// Initialization is Host-only before publication, never a reset of a live join.
// Groups are registered dynamically, before releasing their owning block.
inline __aicore__ bool scheduler_async_join_initial_word(uint32_t blocks, uint64_t &word) {
    if (blocks == 0 || blocks > SCHEDULER_ASYNC_JOIN_LIMIT) return false;
    word = static_cast<uint64_t>(blocks) << 16;
    return true;
}

inline __aicore__ uint64_t scheduler_async_join_load(__gm__ SchedulerAsyncTaskJoin &join) {
    return scheduler_gm_fetch_add(join.word, UINT64_C(0));
}

// Failure poisons both count fields with a midpoint. At most INT16_MAX total
// physical executions/groups and blocks belong to a task, so even all remaining
// legal RMWs racing with abort cannot carry/borrow out of either 16-bit field.
// FAILED therefore stays sticky without a CAS retry loop or an error lock.
// Counts after failure are poison, not outstanding-work diagnostics. Runtime
// error retirement uses stop/last-use proofs, never these poisoned counters.
inline __aicore__ void scheduler_async_join_abort(__gm__ SchedulerAsyncTaskJoin &join) {
    constexpr uint64_t poison = (UINT64_C(0x8000) << 16) | UINT64_C(0x8000);
    scheduler_gm_store(join.word, SCHEDULER_ASYNC_JOIN_FAILED | poison);
}

inline __aicore__ SchedulerAsyncJoinOutcome scheduler_async_join_add_group(__gm__ SchedulerAsyncTaskJoin &join) {
    const uint64_t previous = scheduler_gm_fetch_add(join.word, UINT64_C(1));
    if (!scheduler_async_join_valid(previous)) return {SchedulerAsyncJoinResult::INVALID_STATE, previous};
    if ((previous & SCHEDULER_ASYNC_JOIN_FAILED) != 0) return {SchedulerAsyncJoinResult::FAILED, previous};
    if (scheduler_async_join_blocks(previous) == 0) return {SchedulerAsyncJoinResult::INVALID_STATE, previous};
    if (scheduler_async_join_groups(previous) == SCHEDULER_ASYNC_JOIN_LIMIT)
        return {SchedulerAsyncJoinResult::OVERFLOW, previous};
    return {SchedulerAsyncJoinResult::PENDING, previous};
}

namespace scheduler_async_join_detail {
template <bool Group>
inline __aicore__ SchedulerAsyncJoinOutcome settle(__gm__ SchedulerAsyncTaskJoin &join) {
    constexpr uint64_t unit = Group ? UINT64_C(1) : (UINT64_C(1) << 16);
    const uint64_t previous = scheduler_gm_fetch_add(join.word, UINT64_C(0) - unit);
    if (!scheduler_async_join_valid(previous)) return {SchedulerAsyncJoinResult::INVALID_STATE, previous};
    if ((previous & SCHEDULER_ASYNC_JOIN_FAILED) != 0) return {SchedulerAsyncJoinResult::FAILED, previous};
    const uint32_t count = Group ? scheduler_async_join_groups(previous) : scheduler_async_join_blocks(previous);
    if (count == 0) return {SchedulerAsyncJoinResult::UNDERFLOW, previous};
    return {previous == unit ? SchedulerAsyncJoinResult::FINISH_OWNER : SchedulerAsyncJoinResult::PENDING, previous};
}
}  // namespace scheduler_async_join_detail

// A unique execution-release or group-terminal owner authorizes each RMW.
// Only the decrement that first makes both fields zero owns FINISH publication;
// merely observing zero, registering a group or aborting never grants ownership.
// Invalid contributions require run failure, not compensation/retry. FINISH
// must also observe the run error and join state before publishing TaskDone.
inline __aicore__ SchedulerAsyncJoinOutcome scheduler_async_join_settle_group(__gm__ SchedulerAsyncTaskJoin &join) {
    return scheduler_async_join_detail::settle<true>(join);
}

inline __aicore__ SchedulerAsyncJoinOutcome scheduler_async_join_release_block(__gm__ SchedulerAsyncTaskJoin &join) {
    return scheduler_async_join_detail::settle<false>(join);
}
