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

#include "scheduler_linked_queue.h"

// Cursor is atomic-only. Payload has one publisher, uses bypass stores, and
// stays immutable after the descriptor enters the persistent chunk chain.
struct alignas(128) SchedulerReadyChunk {
    volatile uint64_t cursor{0};
    uint8_t cursor_padding[120]{};
    volatile uint64_t count{0};
    volatile int64_t refs[8]{};
    uint8_t payload_padding[56]{};
};

static_assert(sizeof(SchedulerReadyChunk) == 256);
static_assert(offsetof(SchedulerReadyChunk, count) == 128);
static_assert(std::is_standard_layout_v<SchedulerReadyChunk>);
static_assert(std::is_trivially_copyable_v<SchedulerReadyChunk>);

struct SchedulerChunkQueueView {
    SchedulerLinkedQueueView chain{};
    __gm__ SchedulerReadyChunk *chunks{nullptr};
};

struct SchedulerChunkCandidate {
    int64_t chunk{SCHEDULER_TASK_REF_INVALID};
    uint64_t position{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
};

inline __aicore__ bool scheduler_chunk_view_valid(const SchedulerChunkQueueView &queue) {
    return scheduler_linked_view_valid(queue.chain) && (queue.chain.task_count == 0 || queue.chunks != nullptr);
}

// Each task belongs to exactly one batch. The first ref is therefore a unique
// descriptor ID. Host initializes every cursor/count to zero before the run;
// no descriptor or task can be recycled until the run fence.
inline __aicore__ SchedulerLinkedResult scheduler_chunk_prepare(
    const SchedulerChunkQueueView &queue, const int64_t *refs, uint32_t count, SchedulerLinkedBatch *batch
) {
    if (batch == nullptr) return SchedulerLinkedResult::INVALID;
    *batch = {};
    if (!scheduler_chunk_view_valid(queue) || refs == nullptr || count == 0 || count > 8)
        return SchedulerLinkedResult::INVALID;
    for (uint32_t index = 0; index < count; ++index) {
        if (!scheduler_linked_ref_valid(queue.chain, refs[index])) return SchedulerLinkedResult::INVALID;
        for (uint32_t prior = 0; prior < index; ++prior)
            if (refs[index] == refs[prior]) return SchedulerLinkedResult::INVALID;
    }
    auto &chunk = queue.chunks[refs[0]];
    for (uint32_t index = 0; index < count; ++index) {
#if defined(__CCE_AICORE__)
        auto *word = reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&chunk.refs[index]));
        st_dev(static_cast<uint64_t>(refs[index]), word, 0);
#else
        __atomic_store_n(&chunk.refs[index], refs[index], __ATOMIC_RELEASE);
#endif
    }
    scheduler_cache_barrier();
    scheduler_publish_signal(chunk.count, static_cast<uint64_t>(count));
    return scheduler_linked_prepare(queue.chain, refs, 1, batch);
}

// A window grants no ownership. Head skips only exhausted descriptors, and
// every descriptor stays readable after that advance. A stale window can
// lose a cursor CAS but cannot grant a task twice.
template <uint32_t Window>
inline __aicore__ SchedulerLinkedResult scheduler_chunk_window(
    const SchedulerChunkQueueView &queue, int64_t (&chunks)[Window], uint32_t *count, uint32_t skip_budget = 8,
    uint32_t window_size = Window
) {
    static_assert(Window > 0 && Window <= 8);
    if (count == nullptr) return SchedulerLinkedResult::INVALID;
    *count = 0;
    if (!scheduler_chunk_view_valid(queue) || window_size == 0 || window_size > Window)
        return SchedulerLinkedResult::INVALID;
    int64_t head = scheduler_gm_query(queue.chain.control->head);
    bool found = false;
    for (uint32_t step = 0; step < skip_budget; ++step) {
        if (!scheduler_linked_cursor_valid(queue.chain, head)) return SchedulerLinkedResult::INVALID;
        if (head != queue.chain.sentinel) {
            const auto size = scheduler_gm_query(queue.chunks[head].count);
            if (size > 8) return SchedulerLinkedResult::INVALID;
            if (size == 0) return SchedulerLinkedResult::NOT_READY;
            const auto position = scheduler_gm_query(queue.chunks[head].cursor);
            if (position > size) return SchedulerLinkedResult::INVALID;
            if (position < size) {
                found = true;
                break;
            }
        }
        const auto next = scheduler_gm_query(queue.chain.links[head]);
        if (next == SCHEDULER_TASK_REF_INVALID) return SchedulerLinkedResult::NOT_READY;
        if (!scheduler_linked_ref_valid(queue.chain, next) || next == head) return SchedulerLinkedResult::INVALID;
        const auto observed = scheduler_gm_compare_exchange(queue.chain.control->head, head, next);
        head = observed == head ? next : observed;
    }
    if (!found) return SchedulerLinkedResult::RETRY;
    scheduler_cache_barrier();
    chunks[(*count)++] = head;
    while (*count < window_size) {
        const auto next = scheduler_gm_query(queue.chain.links[head]);
        if (next == SCHEDULER_TASK_REF_INVALID) break;
        if (!scheduler_linked_ref_valid(queue.chain, next)) return SchedulerLinkedResult::INVALID;
        for (uint32_t prior = 0; prior < *count; ++prior)
            if (chunks[prior] == next) return SchedulerLinkedResult::INVALID;
        chunks[(*count)++] = next;
        head = next;
    }
    return SchedulerLinkedResult::SUCCESS;
}

// chunk_id comes from a published window in this run. Placement can inspect
// this candidate before attempting the cursor CAS.
// The candidate never owns a task; exhaustion and CAS loss require a new peek.
inline __aicore__ SchedulerLinkedResult
scheduler_chunk_peek(const SchedulerChunkQueueView &queue, int64_t chunk_id, SchedulerChunkCandidate *candidate) {
    if (candidate == nullptr) return SchedulerLinkedResult::INVALID;
    *candidate = {};
    if (!scheduler_chunk_view_valid(queue) || !scheduler_linked_ref_valid(queue.chain, chunk_id))
        return SchedulerLinkedResult::INVALID;
    auto &chunk = queue.chunks[chunk_id];
    const auto size = scheduler_gm_query(chunk.count);
    if (size == 0) return SchedulerLinkedResult::NOT_READY;
    const auto position = scheduler_gm_query(chunk.cursor);
    if (size > 8 || position > size) return SchedulerLinkedResult::INVALID;
    if (position == size) return SchedulerLinkedResult::NOT_READY;
    const auto ref = scheduler_gm_query(chunk.refs[position]);
    if (ref == SCHEDULER_TASK_REF_INVALID) return SchedulerLinkedResult::NOT_READY;
    if (!scheduler_linked_ref_valid(queue.chain, ref)) return SchedulerLinkedResult::INVALID;
    *candidate = {chunk_id, position, ref};
    return SchedulerLinkedResult::SUCCESS;
}

template <uint32_t Capacity>
inline __aicore__ SchedulerLinkedResult scheduler_chunk_try_claim(
    const SchedulerChunkQueueView &queue, const SchedulerChunkCandidate &candidate, uint32_t credit,
    int64_t (&refs)[Capacity], uint32_t *count
) {
    static_assert(Capacity > 0 && Capacity <= SCHEDULER_PENDING_SLOT_COUNT);
    if (count == nullptr) return SchedulerLinkedResult::INVALID;
    *count = 0;
    for (auto &ref : refs)
        ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_chunk_view_valid(queue) || !scheduler_linked_ref_valid(queue.chain, candidate.chunk) ||
        credit > Capacity)
        return SchedulerLinkedResult::INVALID;
    if (credit == 0) return SchedulerLinkedResult::NOT_READY;
    auto &chunk = queue.chunks[candidate.chunk];
    const auto size = scheduler_gm_query(chunk.count);
    if (size == 0) return SchedulerLinkedResult::NOT_READY;
    if (size > 8 || candidate.position >= size) return SchedulerLinkedResult::INVALID;
    const uint32_t available =
        static_cast<uint32_t>(size - candidate.position < credit ? size - candidate.position : credit);
    int64_t prefix[Capacity]{};
    for (uint32_t index = 0; index < available; ++index) {
        prefix[index] = scheduler_gm_query(chunk.refs[candidate.position + index]);
        if (prefix[index] == SCHEDULER_TASK_REF_INVALID) return SchedulerLinkedResult::NOT_READY;
        if (!scheduler_linked_ref_valid(queue.chain, prefix[index])) return SchedulerLinkedResult::INVALID;
    }
    if (prefix[0] != candidate.task_ref) return SchedulerLinkedResult::INVALID;
    if (scheduler_gm_compare_exchange(chunk.cursor, candidate.position, candidate.position + available) !=
        candidate.position)
        return SchedulerLinkedResult::RETRY;
    scheduler_cache_barrier();
    for (uint32_t index = 0; index < available; ++index)
        refs[index] = prefix[index];
    *count = available;
    return SchedulerLinkedResult::SUCCESS;
}
