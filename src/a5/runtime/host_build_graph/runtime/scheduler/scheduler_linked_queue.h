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

// Head names a persistent sentinel or the last claimed task. Tail is a hint
// along the same immutable chain and may lag behind head. No node is reused
// before the run fence; an index therefore cannot return to either cursor.
struct alignas(128) SchedulerLinkedQueue {
    volatile int64_t head{0};
    uint8_t head_padding[120]{};
    volatile int64_t tail{0};
    uint8_t tail_padding[120]{};
};

static_assert(sizeof(SchedulerLinkedQueue) == 256);
static_assert(offsetof(SchedulerLinkedQueue, tail) == 128);
static_assert(std::is_standard_layout_v<SchedulerLinkedQueue>);
static_assert(std::is_trivially_copyable_v<SchedulerLinkedQueue>);

// All links, including the sentinel, occupy atomic-only storage separate from
// task-control links and payload. Task refs index [0, task_count); each queue
// has its own sentinel in [task_count, node_count). This view is core-local.
struct SchedulerLinkedQueueView {
    __gm__ SchedulerLinkedQueue *control{nullptr};
    __gm__ volatile int64_t *links{nullptr};
    uint64_t task_count{0};
    uint64_t node_count{0};
    int64_t sentinel{SCHEDULER_TASK_REF_INVALID};
};

struct SchedulerLinkedBatch {
    int64_t first{SCHEDULER_TASK_REF_INVALID};
    int64_t last{SCHEDULER_TASK_REF_INVALID};
    uint32_t count{0};
};

enum class SchedulerLinkedResult : uint8_t { SUCCESS, NOT_READY, RETRY, INVALID };

inline __aicore__ bool scheduler_linked_view_valid(const SchedulerLinkedQueueView &queue) {
    return queue.control != nullptr && queue.links != nullptr && queue.task_count <= INT32_MAX &&
           queue.node_count > queue.task_count && queue.node_count <= UINT32_MAX && queue.sentinel >= 0 &&
           static_cast<uint64_t>(queue.sentinel) >= queue.task_count &&
           static_cast<uint64_t>(queue.sentinel) < queue.node_count;
}

inline __aicore__ bool scheduler_linked_ref_valid(const SchedulerLinkedQueueView &queue, int64_t ref) {
    return ref >= 0 && static_cast<uint64_t>(ref) < queue.task_count;
}

inline __aicore__ bool scheduler_linked_cursor_valid(const SchedulerLinkedQueueView &queue, int64_t cursor) {
    return cursor == queue.sentinel || scheduler_linked_ref_valid(queue, cursor);
}

// Every input task is exclusively owned by the caller and has never been
// published or prepared in another batch. Preparation does not reserve any
// shared queue position. RETRY from try_push retains this exact prepared batch.
inline __aicore__ SchedulerLinkedResult scheduler_linked_prepare(
    const SchedulerLinkedQueueView &queue, const int64_t *refs, uint32_t count, SchedulerLinkedBatch *batch
) {
    if (batch == nullptr) return SchedulerLinkedResult::INVALID;
    *batch = {};
    if (!scheduler_linked_view_valid(queue) || count == 0 || count > 8 || refs == nullptr)
        return SchedulerLinkedResult::INVALID;
    for (uint32_t index = 0; index < count; ++index) {
        if (!scheduler_linked_ref_valid(queue, refs[index])) return SchedulerLinkedResult::INVALID;
        for (uint32_t prior = 0; prior < index; ++prior)
            if (refs[index] == refs[prior]) return SchedulerLinkedResult::INVALID;
    }
    for (uint32_t index = 0; index < count; ++index)
        scheduler_gm_store(queue.links[refs[index]], index + 1 == count ? SCHEDULER_TASK_REF_INVALID : refs[index + 1]);
    scheduler_cache_barrier();
    *batch = {refs[0], refs[count - 1], count};
    return SchedulerLinkedResult::SUCCESS;
}

// Both chains belong exclusively to this publisher and have no shared nodes.
// Appending transfers the incoming chain into pending without publishing it;
// pending survives service calls, incoming does not retain a second owner.
inline __aicore__ SchedulerLinkedResult scheduler_linked_append_private(
    const SchedulerLinkedQueueView &queue, SchedulerLinkedBatch *pending, SchedulerLinkedBatch *incoming
) {
    if (!scheduler_linked_view_valid(queue) || pending == nullptr || incoming == nullptr || pending == incoming ||
        incoming->count == 0 || incoming->count > queue.task_count ||
        pending->count > queue.task_count - incoming->count || !scheduler_linked_ref_valid(queue, incoming->first) ||
        !scheduler_linked_ref_valid(queue, incoming->last))
        return SchedulerLinkedResult::INVALID;
    if (pending->count == 0) {
        if (pending->first != SCHEDULER_TASK_REF_INVALID || pending->last != SCHEDULER_TASK_REF_INVALID)
            return SchedulerLinkedResult::INVALID;
        *pending = *incoming;
    } else {
        if (!scheduler_linked_ref_valid(queue, pending->first) || !scheduler_linked_ref_valid(queue, pending->last))
            return SchedulerLinkedResult::INVALID;
        scheduler_gm_store(queue.links[pending->last], incoming->first);
        scheduler_cache_barrier();
        pending->last = incoming->last;
        pending->count += incoming->count;
    }
    *incoming = {};
    return SchedulerLinkedResult::SUCCESS;
}

// The successful empty-link CAS publishes the whole prepared batch. Tail
// helping is optional after that point: failure cannot revoke publication.
// An exhausted budget leaves the batch private and must not abort a run.
inline __aicore__ SchedulerLinkedResult scheduler_linked_try_push(
    const SchedulerLinkedQueueView &queue, SchedulerLinkedBatch *batch, uint32_t retry_budget = 2
) {
    if (!scheduler_linked_view_valid(queue) || batch == nullptr || batch->count == 0 ||
        batch->count > queue.task_count || !scheduler_linked_ref_valid(queue, batch->first) ||
        !scheduler_linked_ref_valid(queue, batch->last))
        return SchedulerLinkedResult::INVALID;
    scheduler_cache_barrier();
    for (uint32_t attempt = 0; attempt < retry_budget; ++attempt) {
        const int64_t tail = scheduler_gm_query(queue.control->tail);
        if (!scheduler_linked_cursor_valid(queue, tail)) return SchedulerLinkedResult::INVALID;
        int64_t next = scheduler_gm_query(queue.links[tail]);
        if (next == SCHEDULER_TASK_REF_INVALID) {
            next = scheduler_gm_compare_exchange(queue.links[tail], SCHEDULER_TASK_REF_INVALID, batch->first);
            if (next == SCHEDULER_TASK_REF_INVALID) {
                scheduler_cache_barrier();
                (void)scheduler_gm_compare_exchange(queue.control->tail, tail, batch->last);
                *batch = {};
                return SchedulerLinkedResult::SUCCESS;
            }
        }
        if (!scheduler_linked_ref_valid(queue, next) || next == tail) return SchedulerLinkedResult::INVALID;
        (void)scheduler_gm_compare_exchange(queue.control->tail, tail, next);
    }
    return SchedulerLinkedResult::RETRY;
}

// The immutable prefix remains valid even if another consumer wins. Only
// the successful head CAS grants ownership; NOT_READY is merely a poll result.
template <uint32_t Capacity>
inline __aicore__ SchedulerLinkedResult scheduler_linked_try_pop_batch(
    const SchedulerLinkedQueueView &queue, int64_t (&refs)[Capacity], uint32_t credit, uint32_t *count,
    uint32_t retry_budget = 2
) {
    static_assert(Capacity > 0 && Capacity <= SCHEDULER_PENDING_SLOT_COUNT);
    if (count == nullptr) return SchedulerLinkedResult::INVALID;
    *count = 0;
    for (auto &ref : refs)
        ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_linked_view_valid(queue) || credit > Capacity) return SchedulerLinkedResult::INVALID;
    if (credit == 0) return SchedulerLinkedResult::NOT_READY;
    for (uint32_t attempt = 0; attempt < retry_budget; ++attempt) {
        const int64_t head = scheduler_gm_query(queue.control->head);
        if (!scheduler_linked_cursor_valid(queue, head)) return SchedulerLinkedResult::INVALID;
        int64_t prefix[Capacity]{};
        int64_t last = head;
        uint32_t available = 0;
        while (available < credit) {
            const int64_t next = scheduler_gm_query(queue.links[last]);
            if (next == SCHEDULER_TASK_REF_INVALID) break;
            if (!scheduler_linked_ref_valid(queue, next) || next == head || next == last)
                return SchedulerLinkedResult::INVALID;
            prefix[available++] = next;
            last = next;
        }
        if (available == 0) return SchedulerLinkedResult::NOT_READY;
        if (scheduler_gm_compare_exchange(queue.control->head, head, last) == head) {
            scheduler_cache_barrier();
            for (uint32_t index = 0; index < available; ++index)
                refs[index] = prefix[index];
            *count = available;
            return SchedulerLinkedResult::SUCCESS;
        }
    }
    return SchedulerLinkedResult::RETRY;
}
