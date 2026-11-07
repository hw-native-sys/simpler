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

#include "async_capability.h"
#include "scheduler_async_layout.h"

inline constexpr uint32_t SCHEDULER_ASYNC_CHECK_WORK_LIMIT = 8;

enum class SchedulerAsyncCheckResult : uint8_t { PENDING, DONE, ERROR };
enum class SchedulerAsyncCheckPhase : uint64_t { UNINITIALIZED, ACTIVE, DRAINED, FAILED };
enum class SchedulerAsyncConditionState : uint8_t { PENDING, READY, FAILED };
enum class SchedulerAsyncCheckError : int64_t {
    INVALID_STATE = -1,
    INVALID_GROUP = -2,
    INVALID_ENTRY = -3,
    INVALID_BINDING = -4,
    BACKEND_REJECTED = -5,
    BACKEND_FAILED = -6,
    INVALID_BACKEND_RESULT = -7,
};

struct SchedulerAsyncConditionOutcome {
    SchedulerAsyncConditionState state{SchedulerAsyncConditionState::PENDING};
    int64_t error_code{0};
};

// Sealing validates registration once. CHECK observes only immutable tokens.
struct SchedulerAsyncCheckView {
    __gm__ const DeferredCompletionEntry *entries{nullptr};
    uint64_t entry_capacity{0};
};

namespace scheduler_async_check_detail {
inline __aicore__ SchedulerAsyncCheckPhase phase(const SchedulerAsyncGroupProgress &progress) {
    return static_cast<SchedulerAsyncCheckPhase>(progress.phase);
}

inline __aicore__ void increment(uint64_t &counter) {
    if (counter != UINT64_MAX) ++counter;
}

inline __aicore__ SchedulerAsyncCheckResult fail(SchedulerAsyncGroupProgress &progress, int64_t error) {
    if (phase(progress) != SchedulerAsyncCheckPhase::FAILED) {
        progress.failure_code = error != 0 ? error : static_cast<int64_t>(SchedulerAsyncCheckError::BACKEND_FAILED);
        progress.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED);
    }
    return SchedulerAsyncCheckResult::ERROR;
}

inline __aicore__ SchedulerAsyncCheckResult
fail(SchedulerAsyncGroupProgress &progress, SchedulerAsyncCheckError error) {
    return fail(progress, static_cast<int64_t>(error));
}

inline __aicore__ bool
group_valid(const SchedulerAsyncGroupIdentity &identity, const SchedulerAsyncGroupWriter &writer) {
    return identity.run_epoch != 0 && identity.task_ref >= 0 && identity.finish_index != SCHEDULER_ASYNC_INDEX_NONE &&
           identity.subtask_slot < 3 && identity.capacity >= 1 && identity.capacity <= CORE_ASYNC_MAX_CONDITIONS &&
           identity.condition_types != 0 && (identity.condition_types & ~CORE_ASYNC_CONDITION_TYPES) == 0 &&
           writer.reservation.sealed == 1 && writer.reservation.limit == 0 && writer.count <= identity.capacity;
}

inline __aicore__ uint64_t count_mask(uint32_t count) { return count == 64 ? UINT64_MAX : (UINT64_C(1) << count) - 1; }

inline __aicore__ DeferredCompletionEntry entry(const SchedulerAsyncCheckView &view, uint32_t index) {
    static_assert(offsetof(DeferredCompletionEntry, addr) == 0);
    static_assert(offsetof(DeferredCompletionEntry, backend_cookie) == 8);
    static_assert(offsetof(DeferredCompletionEntry, expected_value) == 16);
    static_assert(offsetof(DeferredCompletionEntry, engine) == 20);
    static_assert(offsetof(DeferredCompletionEntry, completion_type) == 24);
    static_assert(offsetof(DeferredCompletionEntry, _pad) == 28);
    const auto &source = view.entries[index];
    const uint64_t expected_engine =
        scheduler_gm_query_u32_pair(const_cast<__gm__ volatile uint32_t *>(&source.expected_value));
    const uint64_t type_padding = scheduler_gm_query_u32_pair(
        reinterpret_cast<__gm__ volatile uint32_t *>(const_cast<__gm__ int32_t *>(&source.completion_type))
    );
    // Sealed metadata is published before the lease. Bypass reads avoid keeping
    // an old observer's cached snapshot; never use an RMW on these fields.
    return {
        scheduler_gm_query(const_cast<__gm__ uint64_t &>(source.addr)),
        scheduler_gm_query(const_cast<__gm__ uint64_t &>(source.backend_cookie)),
        static_cast<uint32_t>(expected_engine),
        static_cast<uint32_t>(expected_engine >> 32),
        static_cast<int32_t>(static_cast<uint32_t>(type_padding)),
        static_cast<uint32_t>(type_padding >> 32)
    };
}

inline __aicore__ bool entry_valid(const SchedulerAsyncGroupIdentity &identity, const DeferredCompletionEntry &entry) {
    uint32_t type = 0;
    if (entry.completion_type == COMPLETION_TYPE_COUNTER) type = CORE_ASYNC_CONDITION_COUNTER;
    else if (entry.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) type = CORE_ASYNC_CONDITION_SDMA_POST_DONE;
    return type != 0 && (identity.condition_types & type) != 0 && entry._pad == 0;
}

inline __aicore__ bool outcome_valid(const SchedulerAsyncConditionOutcome &outcome) {
    return (outcome.state == SchedulerAsyncConditionState::PENDING ||
            outcome.state == SchedulerAsyncConditionState::READY) ?
               outcome.error_code == 0 :
               outcome.state == SchedulerAsyncConditionState::FAILED;
}
}  // namespace scheduler_async_check_detail

// The sealing owner initializes progress once, before publishing its CHECK lease
// and directory bit. Reinitializing a live/drained record is an error, not a way
// to rebuild its mask from count and accidentally observe completed entries again.
inline __aicore__ SchedulerAsyncCheckResult scheduler_async_check_initialize(
    const SchedulerAsyncGroupIdentity &identity, const SchedulerAsyncGroupWriter &writer,
    SchedulerAsyncGroupProgress *progress
) {
    using namespace scheduler_async_check_detail;
    if (progress == nullptr) return SchedulerAsyncCheckResult::ERROR;
    if (phase(*progress) != SchedulerAsyncCheckPhase::UNINITIALIZED || progress->remaining_mask != 0 ||
        progress->next_condition != 0 || progress->failure_code != 0)
        return fail(*progress, SchedulerAsyncCheckError::INVALID_STATE);
    if (writer.error_code != 0) return fail(*progress, writer.error_code);
    if (!group_valid(identity, writer)) return fail(*progress, SchedulerAsyncCheckError::INVALID_GROUP);
    progress->remaining_mask = count_mask(writer.count);
    progress->phase =
        static_cast<uint64_t>(writer.count == 0 ? SchedulerAsyncCheckPhase::DRAINED : SchedulerAsyncCheckPhase::ACTIVE);
    return writer.count == 0 ? SchedulerAsyncCheckResult::DONE : SchedulerAsyncCheckResult::PENDING;
}

// One operation observes one condition, including the backend's observation
// ordering. READY is monotonic for the run and is never polled again. The backend
// returns immediately; it cannot retain a continuation, a core, or a payload.
// The caller publishes this cursor before releasing the CHECK lease.
template <typename Backend>
inline __aicore__ SchedulerAsyncCheckResult scheduler_async_check_step(
    const SchedulerAsyncGroupIdentity &identity, const SchedulerAsyncGroupWriter &writer,
    const SchedulerAsyncCheckView &view, Backend &backend, SchedulerAsyncGroupProgress *progress, uint32_t *budget
) {
    using namespace scheduler_async_check_detail;
    if (progress == nullptr || budget == nullptr) return SchedulerAsyncCheckResult::ERROR;
    if (phase(*progress) == SchedulerAsyncCheckPhase::FAILED) return SchedulerAsyncCheckResult::ERROR;
    if (writer.error_code != 0) return fail(*progress, writer.error_code);
    if (!group_valid(identity, writer)) return fail(*progress, SchedulerAsyncCheckError::INVALID_GROUP);
    const uint32_t count = writer.count;
    if (phase(*progress) == SchedulerAsyncCheckPhase::DRAINED) {
        if (progress->remaining_mask != 0 || progress->failure_code != 0 ||
            (count == 0 ? progress->next_condition != 0 : progress->next_condition >= count))
            return fail(*progress, SchedulerAsyncCheckError::INVALID_STATE);
        return SchedulerAsyncCheckResult::DONE;
    }
    if (phase(*progress) != SchedulerAsyncCheckPhase::ACTIVE || progress->failure_code != 0 || count == 0 ||
        progress->remaining_mask == 0 || (progress->remaining_mask & ~count_mask(count)) != 0 ||
        progress->next_condition >= count)
        return fail(*progress, SchedulerAsyncCheckError::INVALID_STATE);
    if (view.entries == nullptr || view.entry_capacity < identity.capacity)
        return fail(*progress, SchedulerAsyncCheckError::INVALID_ENTRY);
    increment(progress->slice_count);
    uint32_t units = *budget < SCHEDULER_ASYNC_CHECK_WORK_LIMIT ? *budget : SCHEDULER_ASYNC_CHECK_WORK_LIMIT;
    uint64_t observed = 0;
    while (units != 0) {
        const uint64_t eligible = progress->remaining_mask & ~observed;
        if (eligible == 0) break;
        const uint64_t suffix = eligible & (UINT64_MAX << progress->next_condition);
        const uint32_t index = static_cast<uint32_t>(__builtin_ctzll(suffix != 0 ? suffix : eligible));
        --units;
        --*budget;
        increment(progress->work_units);
        const auto condition = entry(view, index);
        if (!entry_valid(identity, condition)) return fail(*progress, SchedulerAsyncCheckError::INVALID_ENTRY);
        observed |= UINT64_C(1) << index;
        progress->next_condition = index + 1 == count ? 0 : index + 1;
        increment(progress->observation_count);
        const auto outcome = backend.observe_once(identity, index, condition);
        if (!outcome_valid(outcome)) return fail(*progress, SchedulerAsyncCheckError::INVALID_BACKEND_RESULT);
        if (outcome.state == SchedulerAsyncConditionState::FAILED) return fail(*progress, outcome.error_code);
        if (outcome.state == SchedulerAsyncConditionState::READY) {
            progress->remaining_mask &= ~(UINT64_C(1) << index);
            if (progress->remaining_mask == 0) {
                progress->phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED);
                return SchedulerAsyncCheckResult::DONE;
            }
        }
    }
    return SchedulerAsyncCheckResult::PENDING;
}
