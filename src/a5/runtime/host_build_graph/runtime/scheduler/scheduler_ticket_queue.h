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

// Cursors occupy separate atomic-only lines. Entries live in a different,
// bypass-published region and are never overwritten within a run.
struct alignas(128) SchedulerTicketQueue {
    volatile uint64_t head{0};
    uint8_t head_padding[120]{};
    volatile uint64_t reserved_tail{0};
    uint8_t tail_padding[120]{};
};

static_assert(sizeof(SchedulerTicketQueue) == 256);
static_assert(offsetof(SchedulerTicketQueue, reserved_tail) == 128);
static_assert(std::is_standard_layout_v<SchedulerTicketQueue>);
static_assert(std::is_trivially_copyable_v<SchedulerTicketQueue>);

// Core-local access descriptor; this is not a wire structure. The backing
// owns the control and all entries until every worker crosses the run fence.
struct SchedulerTicketQueueView {
    __gm__ SchedulerTicketQueue *control{nullptr};
    __gm__ volatile int64_t *entries{nullptr};
    uint64_t capacity{0};
    uint64_t task_count{0};
};

struct SchedulerTicketRange {
    uint64_t begin{0};
    uint64_t count{0};
};

enum class SchedulerTicketResult : uint8_t {
    SUCCESS,
    NOT_READY,
    RETRY,
    INVALID,
};

inline __aicore__ bool scheduler_ticket_view_valid(const SchedulerTicketQueueView &queue) {
    return queue.control != nullptr && queue.capacity <= queue.task_count && queue.task_count <= INT32_MAX &&
           (queue.capacity == 0 || queue.entries != nullptr);
}

// Each logical task may reserve at most one entry. Capacity overflow is a
// terminal caller/protocol error: the failed reservation must never be retried.
// No entry is accessed on overflow, even when concurrent reservations caused it.
inline __aicore__ SchedulerTicketResult
scheduler_ticket_reserve(const SchedulerTicketQueueView &queue, uint64_t count, SchedulerTicketRange *range) {
    if (range == nullptr) return SchedulerTicketResult::INVALID;
    *range = {};
    if (!scheduler_ticket_view_valid(queue) || count > queue.capacity) return SchedulerTicketResult::INVALID;
    if (count == 0) return SchedulerTicketResult::SUCCESS;
    const uint64_t begin = scheduler_gm_fetch_add(queue.control->reserved_tail, count);
    if (begin > queue.capacity || count > queue.capacity - begin) return SchedulerTicketResult::INVALID;
    *range = {begin, count};
    return SchedulerTicketResult::SUCCESS;
}

// The successful reservation is the sole writer grant for these entries.
// Each range is published once; entries are immutable after publication.
// An earlier unpublished range may delay later ranges, so publication must
// neither execute a kernel nor wait for another worker.
inline __aicore__ SchedulerTicketResult scheduler_ticket_publish(
    const SchedulerTicketQueueView &queue, const SchedulerTicketRange &range, const int64_t *refs
) {
    if (!scheduler_ticket_view_valid(queue) || range.begin > queue.capacity ||
        range.count > queue.capacity - range.begin || (range.count != 0 && refs == nullptr))
        return SchedulerTicketResult::INVALID;
    for (uint64_t index = 0; index < range.count; ++index)
        if (refs[index] < 0 || static_cast<uint64_t>(refs[index]) >= queue.task_count)
            return SchedulerTicketResult::INVALID;
    if (range.count == 0) return SchedulerTicketResult::SUCCESS;
    scheduler_cache_barrier();
    for (uint64_t index = 0; index < range.count; ++index) {
#if defined(__CCE_AICORE__)
        auto *entry =
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&queue.entries[range.begin + index]));
        st_dev(static_cast<uint64_t>(refs[index]), entry, 0);
#else
        __atomic_store_n(&queue.entries[range.begin + index], refs[index], __ATOMIC_RELEASE);
#endif
    }
    scheduler_cache_barrier();
    return SchedulerTicketResult::SUCCESS;
}

// NOT_READY is a poll result, not a linearizable empty-queue observation or a
// run-completion proof. An earlier reservation may still be unpublished.
// A stale head can cause only a failed CAS: entries never change after publish.
template <uint32_t Capacity>
inline __aicore__ SchedulerTicketResult scheduler_ticket_try_pop_batch(
    const SchedulerTicketQueueView &queue, int64_t (&refs)[Capacity], uint32_t credit, uint32_t *count,
    uint32_t retry_budget = 8, uint64_t *cas_retries = nullptr
) {
    static_assert(Capacity > 0 && Capacity <= SCHEDULER_PENDING_SLOT_COUNT);
    if (count == nullptr) return SchedulerTicketResult::INVALID;
    *count = 0;
    for (auto &ref : refs)
        ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_ticket_view_valid(queue) || credit > Capacity) return SchedulerTicketResult::INVALID;
    if (credit == 0) return SchedulerTicketResult::NOT_READY;
    for (uint32_t attempt = 0; attempt < retry_budget; ++attempt) {
        const uint64_t head = scheduler_gm_query(queue.control->head);
        if (head > queue.capacity) return SchedulerTicketResult::INVALID;
        if (head == queue.capacity) return SchedulerTicketResult::NOT_READY;
        int64_t prefix[Capacity]{};
        uint32_t available = 0;
        while (available < credit && available < queue.capacity - head) {
            const int64_t ref = scheduler_gm_query(queue.entries[head + available]);
            if (ref == SCHEDULER_TASK_REF_INVALID) break;
            if (ref < 0 || static_cast<uint64_t>(ref) >= queue.task_count) return SchedulerTicketResult::INVALID;
            prefix[available++] = ref;
        }
        if (available == 0) return SchedulerTicketResult::NOT_READY;
        if (scheduler_gm_compare_exchange(queue.control->head, head, head + available) == head) {
            scheduler_cache_barrier();
            for (uint32_t index = 0; index < available; ++index)
                refs[index] = prefix[index];
            *count = available;
            return SchedulerTicketResult::SUCCESS;
        }
        if (cas_retries != nullptr) ++*cas_retries;
    }
    return SchedulerTicketResult::RETRY;
}

inline __aicore__ SchedulerTicketResult scheduler_ticket_try_pop(
    const SchedulerTicketQueueView &queue, int64_t *task_ref, uint32_t retry_budget = 8, uint64_t *cas_retries = nullptr
) {
    if (task_ref == nullptr) return SchedulerTicketResult::INVALID;
    int64_t refs[1]{};
    uint32_t count = 0;
    const auto result = scheduler_ticket_try_pop_batch(queue, refs, 1, &count, retry_budget, cas_retries);
    *task_ref = refs[0];
    return result;
}
