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

#include "scheduler_completion_state.h"

enum class SchedulerCompletionQueue : uint64_t { INBOX, CONTROL, TICKET, LINKED, CHUNK };

// The production policy never hides a retry loop. Tests can place a competing
// publisher at the reservation/CAS boundary without a device runtime hook.
struct SchedulerCompletionQueueOps {
    inline __aicore__ static SchedulerTicketResult
    reserve(const SchedulerTicketQueueView &queue, uint64_t count, SchedulerTicketRange *range) {
        return scheduler_ticket_reserve(queue, count, range);
    }
    inline __aicore__ static SchedulerLinkedResult
    push(const SchedulerLinkedQueueView &queue, SchedulerLinkedBatch *batch) {
        return scheduler_linked_try_push(queue, batch, 1);
    }
    inline __aicore__ static uint64_t
    compare_exchange(__gm__ volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

namespace scheduler_completion_detail {
inline __aicore__ SchedulerCompletionPhase phase(const SchedulerReadyPublication &progress) {
    return static_cast<SchedulerCompletionPhase>(progress.phase);
}
inline __aicore__ void phase(SchedulerReadyPublication &progress, SchedulerCompletionPhase next) {
    progress.phase = static_cast<uint64_t>(next);
}

inline __aicore__ SchedulerCompletionPhase phase(const SchedulerCompletionCursor &cursor) {
    return phase(cursor.publication);
}
inline __aicore__ void phase(SchedulerCompletionCursor &cursor, SchedulerCompletionPhase next) {
    phase(cursor.publication, next);
}

inline __aicore__ SchedulerCompletionResult fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task,
    SchedulerCompletionCursor &progress, SchedulerErrorSite site = SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED,
    SchedulerGraphResult status = SchedulerGraphResult::INVALID_ARGUMENTS
) {
    phase(progress, SchedulerCompletionPhase::FAILED);
    scheduler_record_error(run, task, status, &graph, local, site);
    return SchedulerCompletionResult::ERROR;
}

struct Destination {
    SchedulerCompletionQueue kind{SchedulerCompletionQueue::INBOX};
    uint64_t queue{0};
    uint64_t target{0};
};

inline __aicore__ bool destination(
    const SchedulerGraphAccess &graph, __gm__ void *base, const SchedulerLocalState *local, int64_t task,
    Destination *out
) {
    SchedulerNodeKind kind{};
    if (scheduler_node_kind(graph, task, &kind) != SchedulerGraphResult::OK) return false;
    const auto *metadata = scheduler_task_metadata_at(base, local, task);
    const auto lanes = scheduler_task_lanes(*metadata);
    if (scheduler_node_is_control(kind)) {
        if (!lanes.is_empty() || scheduler_task_is_executable(metadata->flags) ||
            local->config.control_inboxes_offset == 0 || local->worker_id() >= local->config.runtime_worker_count)
            return false;
        *out = {SchedulerCompletionQueue::CONTROL, SCHEDULER_READY_QUEUE_COUNT, local->worker_id()};
        return true;
    }
    const uint32_t type = scheduler_task_ready_queue(metadata->flags, lanes);
    if (!scheduler_task_is_executable(metadata->flags) || type >= SCHEDULER_READY_QUEUE_COUNT) return false;
    out->queue = type;
    if (local->config.ready_pool_offset == 0) {
        out->kind = SchedulerCompletionQueue::INBOX;
        out->target = scheduler_ready_push_target(local, type, task);
        return out->target < scheduler_ready_queue_owner_count(local, type) &&
               out->target < scheduler_ready_queue_capacity(type);
    }
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
    if (pool.group_counts[type] == 0) return false;
    const uint32_t group = scheduler_ready_pool_ref_group(task, pool.group_counts[type]);
    if (pool.queue_begins[type] >= pool.queue_count || group >= pool.queue_count - pool.queue_begins[type])
        return false;
    out->target = pool.queue_begins[type] + group;
    switch (pool.kind) {
    case SchedulerReadyPoolKind::TICKET:
        out->kind = SchedulerCompletionQueue::TICKET;
        break;
    case SchedulerReadyPoolKind::LINKED:
        out->kind = SchedulerCompletionQueue::LINKED;
        break;
    case SchedulerReadyPoolKind::CHUNK:
        out->kind = SchedulerCompletionQueue::CHUNK;
        break;
    default:
        return false;
    }
    return true;
}

inline __aicore__ void begin_publication(SchedulerReadyPublication &progress, SchedulerCompletionPhase resume) {
    progress.resume_phase = static_cast<uint64_t>(resume);
    phase(progress, SchedulerCompletionPhase::PREPARE);
}

// Preparation owns at most eight tasks and does not reserve shared queue space.
// Prepared links and chunk identity persist across retry and owner handoff.
inline __aicore__ bool prepare(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerReadyPublication &progress
) {
    const uint32_t count = static_cast<uint32_t>(progress.pending_count);
    if (count == 0 || progress.pending_count > 8) return false;
    for (uint32_t index = 0; index < count; ++index) {
        if (!scheduler_task_ref_valid(graph, progress.pending_tasks[index])) return false;
        for (uint32_t prior = 0; prior < index; ++prior)
            if (progress.pending_tasks[prior] == progress.pending_tasks[index]) return false;
    }
    const auto kind = static_cast<SchedulerCompletionQueue>(progress.pending_kind);
    SchedulerLinkedBatch batch{};
    if (kind == SchedulerCompletionQueue::LINKED || kind == SchedulerCompletionQueue::CHUNK) {
        const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
        const auto result = kind == SchedulerCompletionQueue::CHUNK ?
                                scheduler_chunk_prepare(
                                    scheduler_ready_pool_chunk_queue(base, pool, progress.pending_target),
                                    progress.pending_tasks, count, &batch
                                ) :
                                scheduler_linked_prepare(
                                    scheduler_ready_pool_linked_queue(base, pool, progress.pending_target),
                                    progress.pending_tasks, count, &batch
                                );
        if (result != SchedulerLinkedResult::SUCCESS) return false;
    } else if (kind == SchedulerCompletionQueue::INBOX || kind == SchedulerCompletionQueue::CONTROL) {
        for (uint32_t index = 0; index < count; ++index) {
            auto *task = scheduler_task_control_at(base, local, progress.pending_tasks[index]);
            scheduler_gm_store(
                task->next_waiter, index + 1 == count ? SCHEDULER_INBOX_EMPTY : progress.pending_tasks[index + 1]
            );
        }
        scheduler_cache_barrier();
        batch.first = progress.pending_tasks[0];
        batch.last = progress.pending_tasks[count - 1];
        batch.count = count;
    } else if (kind != SchedulerCompletionQueue::TICKET) {
        return false;
    }
    progress.pending_first = batch.first;
    progress.pending_last = batch.last;
    progress.pending_link_count = batch.count;
    phase(progress, SchedulerCompletionPhase::PUBLISH);
    return true;
}

enum class PublishResult : uint8_t { PUBLISHED, ADVANCED, RETRY, ERROR };

inline __aicore__ void published(SchedulerReadyPublication &progress) {
    progress.pending_count = 0;
    progress.pending_first = progress.pending_last = SCHEDULER_TASK_REF_INVALID;
    progress.pending_link_count = 0;
    progress.phase = progress.resume_phase;
}

template <typename QueueOps>
inline __aicore__ PublishResult publish(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerReadyPublication &progress
) {
    const auto kind = static_cast<SchedulerCompletionQueue>(progress.pending_kind);
    if (progress.pending_count == 0 || (kind == SchedulerCompletionQueue::TICKET && progress.pending_count > 8))
        return PublishResult::ERROR;
    if (kind == SchedulerCompletionQueue::TICKET) {
        const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
        const auto queue = scheduler_ready_pool_queue(base, pool, progress.pending_target);
        SchedulerTicketRange range{};
        const auto result = QueueOps::reserve(queue, progress.pending_count, &range);
        if (result == SchedulerTicketResult::RETRY) return PublishResult::RETRY;
        if (result != SchedulerTicketResult::SUCCESS) return PublishResult::ERROR;
        // There is no budget check or resumable phase between reservation and
        // publishing this fixed-size range. Failure here aborts; never reserve it again.
        if (scheduler_ticket_publish(queue, range, progress.pending_tasks) != SchedulerTicketResult::SUCCESS)
            return PublishResult::ERROR;
    } else if (kind == SchedulerCompletionQueue::LINKED || kind == SchedulerCompletionQueue::CHUNK) {
        const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
        if (progress.pending_link_count == 0) return PublishResult::ERROR;
        SchedulerLinkedBatch batch{
            progress.pending_first, progress.pending_last, static_cast<uint32_t>(progress.pending_link_count)
        };
        const auto result =
            QueueOps::push(scheduler_ready_pool_linked_queue(base, pool, progress.pending_target), &batch);
        if (result == SchedulerLinkedResult::RETRY) return PublishResult::RETRY;
        if (result != SchedulerLinkedResult::SUCCESS) return PublishResult::ERROR;
    } else if (kind == SchedulerCompletionQueue::INBOX || kind == SchedulerCompletionQueue::CONTROL) {
        auto *inbox = kind == SchedulerCompletionQueue::CONTROL ?
                          scheduler_state_at<SchedulerReadyInbox>(base, local->config.control_inboxes_offset) +
                              progress.pending_target :
                          scheduler_ready_inbox_at(base, local, progress.pending_queue, progress.pending_target);
        const uint64_t packed = scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) return PublishResult::ERROR;
        const int64_t head = scheduler_ready_endpoints_head(packed);
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        const bool empty = scheduler_ready_endpoints_empty(packed);
        if (!empty && (!scheduler_task_ref_valid(graph, head) || !scheduler_task_ref_valid(graph, tail)))
            return PublishResult::ERROR;
        const int64_t first = progress.pending_first;
        const int64_t last = progress.pending_last;
        if (!scheduler_task_ref_valid(graph, first) || !scheduler_task_ref_valid(graph, last) ||
            !scheduler_ready_endpoint_fits(first) || !scheduler_ready_endpoint_fits(last))
            return PublishResult::ERROR;
        const uint64_t desired = scheduler_ready_endpoints_pack(empty ? first : head, last);
        if (QueueOps::compare_exchange(inbox->endpoints, packed, desired) != packed) return PublishResult::RETRY;
        if (!empty) scheduler_gm_store(scheduler_task_control_at(base, local, tail)->next_waiter, first);
        scheduler_cache_barrier();
        if (kind == SchedulerCompletionQueue::INBOX) {
            // The endpoint CAS already transferred all nodes. The directory's
            // shared OR may contend with other publishers, so retain only its
            // outstanding responsibility and never publish this batch again.
            phase(progress, SchedulerCompletionPhase::PUBLISH_DIRECTORY);
            return PublishResult::ADVANCED;
        }
    } else {
        return PublishResult::ERROR;
    }
    // Ownership has moved. Do not inspect any newly public task metadata here.
    published(progress);
    return PublishResult::PUBLISHED;
}

template <typename QueueOps>
inline __aicore__ PublishResult
publish_directory(__gm__ void *base, SchedulerLocalState *local, SchedulerReadyPublication &progress) {
    if (progress.pending_kind != static_cast<uint64_t>(SchedulerCompletionQueue::INBOX) ||
        progress.pending_queue >= SCHEDULER_READY_QUEUE_COUNT ||
        progress.pending_target >= scheduler_ready_queue_owner_count(local, progress.pending_queue) ||
        progress.pending_target >= scheduler_ready_queue_capacity(progress.pending_queue))
        return PublishResult::ERROR;
    auto *directory = scheduler_ready_directory_at(base, local);
    const uint64_t shard = progress.pending_target / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    const uint64_t bit = UINT64_C(1) << (progress.pending_target % SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD);
    auto &word = directory->core_types[progress.pending_queue][shard].bits;
    const uint64_t observed = scheduler_gm_query(word);
    // ld_dev can still show a one that a consumer's atomic already cleared.
    // Even an apparently set bit needs atomic confirmation of this snapshot.
    if (QueueOps::compare_exchange(word, observed, observed | bit) != observed) return PublishResult::RETRY;
    published(progress);
    return PublishResult::PUBLISHED;
}
template <typename QueueOps = SchedulerCompletionQueueOps>
inline __aicore__ PublishResult advance_publication(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerReadyPublication &progress
) {
    switch (phase(progress)) {
    case SchedulerCompletionPhase::PREPARE:
        return prepare(graph, base, local, progress) ? PublishResult::ADVANCED : PublishResult::ERROR;
    case SchedulerCompletionPhase::PUBLISH:
        return publish<QueueOps>(graph, base, local, progress);
    case SchedulerCompletionPhase::PUBLISH_DIRECTORY:
        return publish_directory<QueueOps>(base, local, progress);
    default:
        return PublishResult::ERROR;
    }
}
}  // namespace scheduler_completion_detail
