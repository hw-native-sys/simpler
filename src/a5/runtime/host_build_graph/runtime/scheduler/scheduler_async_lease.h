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

enum class SchedulerAsyncPhase : uint8_t { DORMANT, RUNNABLE, OWNED, TERMINAL, FAILED };

inline constexpr uint64_t SCHEDULER_ASYNC_GENERATION_MAX = UINT64_MAX >> 3;

constexpr inline __aicore__ uint64_t scheduler_async_lease_word(uint64_t generation, SchedulerAsyncPhase phase) {
    if (generation == 0 || generation > SCHEDULER_ASYNC_GENERATION_MAX ||
        static_cast<uint8_t>(phase) > static_cast<uint8_t>(SchedulerAsyncPhase::FAILED) ||
        (phase == SchedulerAsyncPhase::DORMANT && generation != 1))
        return 0;
    return (generation << 3) | static_cast<uint64_t>(phase);
}

constexpr inline __aicore__ SchedulerAsyncPhase scheduler_async_lease_phase(uint64_t word) {
    return static_cast<SchedulerAsyncPhase>(word & UINT64_C(7));
}

constexpr inline __aicore__ uint64_t scheduler_async_lease_generation(uint64_t word) { return word >> 3; }

constexpr inline __aicore__ bool scheduler_async_lease_valid(uint64_t word) {
    return word != 0 &&
           scheduler_async_lease_word(scheduler_async_lease_generation(word), scheduler_async_lease_phase(word)) ==
               word;
}

struct alignas(128) SchedulerAsyncLease {
    volatile uint64_t word{scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT)};
};

static_assert(sizeof(SchedulerAsyncLease) == 128);
static_assert(std::is_standard_layout_v<SchedulerAsyncLease> && std::is_trivially_copyable_v<SchedulerAsyncLease>);

enum class SchedulerAsyncLeaseResult : uint8_t {
    SUCCESS,
    UNAVAILABLE,
    CONTENDED,
    OWNERSHIP_LOST,
    INVALID_STATE,
    INVALID_TRANSITION,
    GENERATION_EXHAUSTED,
};

struct SchedulerAsyncLeaseOutcome {
    SchedulerAsyncLeaseResult result;
    // On success this is the granted word; otherwise it is the observed/input word.
    uint64_t word;
};

inline __aicore__ uint64_t scheduler_async_lease_load(__gm__ SchedulerAsyncLease &lease) {
    return scheduler_gm_fetch_add(lease.word, UINT64_C(0));
}

// Each operation performs at most one CAS. There is no local retry loop: an
// unsuccessful claimant returns to ordinary scheduling without touching progress.
// The caller publishes payload/progress before handoff and acquires it after a
// successful claim with actual AICore barriers; atomic order arguments alone
// do not provide those device barriers. No live lease is reset between retries.
inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_try_transition(__gm__ SchedulerAsyncLease &lease, uint64_t expected, SchedulerAsyncPhase next) {
    if (!scheduler_async_lease_valid(expected)) return {SchedulerAsyncLeaseResult::INVALID_STATE, expected};
    const auto previous_phase = scheduler_async_lease_phase(expected);
    uint64_t generation = scheduler_async_lease_generation(expected);
    bool permitted = false;
    switch (previous_phase) {
    case SchedulerAsyncPhase::DORMANT:
        permitted = next == SchedulerAsyncPhase::RUNNABLE || next == SchedulerAsyncPhase::FAILED;
        break;
    case SchedulerAsyncPhase::RUNNABLE:
        permitted = next == SchedulerAsyncPhase::OWNED;
        break;
    case SchedulerAsyncPhase::OWNED:
        permitted = next == SchedulerAsyncPhase::RUNNABLE || next == SchedulerAsyncPhase::TERMINAL ||
                    next == SchedulerAsyncPhase::FAILED;
        if (next == SchedulerAsyncPhase::RUNNABLE) {
            if (generation == SCHEDULER_ASYNC_GENERATION_MAX)
                return {SchedulerAsyncLeaseResult::GENERATION_EXHAUSTED, expected};
            ++generation;
        }
        break;
    default:
        break;
    }
    if (!permitted) return {SchedulerAsyncLeaseResult::INVALID_TRANSITION, expected};
    const uint64_t desired = scheduler_async_lease_word(generation, next);
    const uint64_t actual = scheduler_gm_compare_exchange(lease.word, expected, desired);
    if (actual == expected) return {SchedulerAsyncLeaseResult::SUCCESS, desired};
    if (!scheduler_async_lease_valid(actual)) return {SchedulerAsyncLeaseResult::INVALID_STATE, actual};
    return {
        previous_phase == SchedulerAsyncPhase::RUNNABLE ? SchedulerAsyncLeaseResult::CONTENDED :
                                                          SchedulerAsyncLeaseResult::OWNERSHIP_LOST,
        actual
    };
}

inline __aicore__ SchedulerAsyncLeaseOutcome scheduler_async_lease_publish(__gm__ SchedulerAsyncLease &lease) {
    return scheduler_async_lease_try_transition(
        lease, scheduler_async_lease_word(1, SchedulerAsyncPhase::DORMANT), SchedulerAsyncPhase::RUNNABLE
    );
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_try_claim(__gm__ SchedulerAsyncLease &lease, uint64_t observed) {
    if (!scheduler_async_lease_valid(observed)) return {SchedulerAsyncLeaseResult::INVALID_STATE, observed};
    if (scheduler_async_lease_phase(observed) != SchedulerAsyncPhase::RUNNABLE)
        return {SchedulerAsyncLeaseResult::UNAVAILABLE, observed};
    return scheduler_async_lease_try_transition(lease, observed, SchedulerAsyncPhase::OWNED);
}

inline __aicore__ SchedulerAsyncLeaseOutcome
scheduler_async_lease_handoff(__gm__ SchedulerAsyncLease &lease, uint64_t owned, SchedulerAsyncPhase next) {
    if (!scheduler_async_lease_valid(owned)) return {SchedulerAsyncLeaseResult::INVALID_STATE, owned};
    if (scheduler_async_lease_phase(owned) != SchedulerAsyncPhase::OWNED)
        return {SchedulerAsyncLeaseResult::INVALID_TRANSITION, owned};
    return scheduler_async_lease_try_transition(lease, owned, next);
}
