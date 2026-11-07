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
inline __aicore__ SchedulerCompletionPhase phase(const SchedulerCompletionCursor &progress) {
    return static_cast<SchedulerCompletionPhase>(progress.phase);
}
inline __aicore__ void phase(SchedulerCompletionCursor &progress, SchedulerCompletionPhase next) {
    progress.phase = static_cast<uint64_t>(next);
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

inline __aicore__ void begin_publication(SchedulerCompletionCursor &progress, SchedulerCompletionPhase resume) {
    progress.resume_phase = static_cast<uint64_t>(resume);
    phase(progress, SchedulerCompletionPhase::PREPARE);
}

// Preparation owns at most eight tasks and does not reserve shared queue space.
// Prepared links and chunk identity persist across retry and owner handoff.
inline __aicore__ bool prepare(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerCompletionCursor &progress
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
    } else if (kind != SchedulerCompletionQueue::TICKET) {
        return false;
    }
    progress.pending_first = batch.first;
    progress.pending_last = batch.last;
    progress.pending_link_count = batch.count;
    phase(progress, SchedulerCompletionPhase::PUBLISH);
    return true;
}

enum class PublishResult : uint8_t { PUBLISHED, DIRECTORY_PENDING, RETRY, ERROR };

inline __aicore__ void published(SchedulerCompletionCursor &progress) {
    progress.pending_count = 0;
    progress.pending_first = progress.pending_last = SCHEDULER_TASK_REF_INVALID;
    progress.pending_link_count = 0;
    progress.phase = progress.resume_phase;
}

template <typename QueueOps>
inline __aicore__ PublishResult publish(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerCompletionCursor &progress
) {
    const auto kind = static_cast<SchedulerCompletionQueue>(progress.pending_kind);
    if (progress.pending_count == 0 || progress.pending_count > 8) return PublishResult::ERROR;
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
        if (progress.pending_link_count == 0 || progress.pending_link_count > 8) return PublishResult::ERROR;
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
        const int64_t first = progress.pending_tasks[0];
        const int64_t last = progress.pending_tasks[progress.pending_count - 1];
        const uint64_t desired = scheduler_ready_endpoints_pack(empty ? first : head, last);
        if (QueueOps::compare_exchange(inbox->endpoints, packed, desired) != packed) return PublishResult::RETRY;
        if (!empty) scheduler_gm_store(scheduler_task_control_at(base, local, tail)->next_waiter, first);
        scheduler_cache_barrier();
        if (kind == SchedulerCompletionQueue::INBOX) {
            // The endpoint CAS already transferred all nodes. The directory's
            // shared OR may contend with other publishers, so retain only its
            // outstanding responsibility and never publish this batch again.
            phase(progress, SchedulerCompletionPhase::PUBLISH_DIRECTORY);
            return PublishResult::DIRECTORY_PENDING;
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
publish_directory(__gm__ void *base, SchedulerLocalState *local, SchedulerCompletionCursor &progress) {
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
}  // namespace scheduler_completion_detail

// Direct refill candidates remain local to a synchronous completion call.
struct SchedulerCompletionDirect {
    SchedulerReadyClaim *ready{nullptr};
    SchedulerReadyClaim *mix{nullptr};
    uint32_t core_type{0};
    bool mix_published{false};
};

// Each slice performs bounded traversal and publication attempts. DONE means
// all successors have transferred ownership; the caller contributes completion.
template <typename QueueOps = SchedulerCompletionQueueOps, typename Link = SchedulerResolveLink>
inline __aicore__ SchedulerCompletionResult scheduler_resolve_completion_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerCompletionCursor *progress, uint32_t *budget,
    SchedulerWakeStats *wake_stats = nullptr, SchedulerReadyStats *ready_stats = nullptr,
    SchedulerCompletionDirect *direct = nullptr, bool validate_done_state = true
) {
    using namespace scheduler_completion_detail;
    if (progress == nullptr || budget == nullptr || base == nullptr || local == nullptr || run == nullptr)
        return SchedulerCompletionResult::ERROR;
    if (phase(*progress) == SchedulerCompletionPhase::FAILED) return SchedulerCompletionResult::ERROR;
    if (progress->phase > static_cast<uint64_t>(SchedulerCompletionPhase::FAILED))
        return fail(graph, local, run, producer_ref, *progress);
    if (scheduler_read_error(run) != 0) {
        phase(*progress, SchedulerCompletionPhase::FAILED);
        return SchedulerCompletionResult::ERROR;
    }
    for (;;) {
        if (scheduler_read_error(run) != 0) {
            phase(*progress, SchedulerCompletionPhase::FAILED);
            return SchedulerCompletionResult::ERROR;
        }
        const auto current_phase = phase(*progress);
        if (current_phase == SchedulerCompletionPhase::DRAINED) return SchedulerCompletionResult::DONE;
        if (*budget == 0) return SchedulerCompletionResult::PENDING;
        if (current_phase == SchedulerCompletionPhase::RESOLVE) {
            if (!scheduler_task_ref_valid(graph, progress->current_waiter) ||
                progress->resolve.task_ref != progress->current_waiter)
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::ROUTE_INVALID_STATE);
            const auto result =
                scheduler_resolve_step<Link>(graph, base, local, run, &progress->resolve, budget, wake_stats);
            if (result == SchedulerRouteResult::ERROR) {
                phase(*progress, SchedulerCompletionPhase::FAILED);
                return SchedulerCompletionResult::ERROR;
            }
            if (result == SchedulerRouteResult::YIELD) return SchedulerCompletionResult::PENDING;
            if (result == SchedulerRouteResult::READY_TO_ENQUEUE) {
                phase(*progress, SchedulerCompletionPhase::ENQUEUE);
            } else if (result == SchedulerRouteResult::WAITING || result == SchedulerRouteResult::COMPLETED) {
                progress->current_waiter = progress->detached_next;
                phase(*progress, SchedulerCompletionPhase::SELECT);
            } else {
                // Only an explicit transfer/already-completed result releases
                // this successor. Unknown results cannot silently drain FINISH.
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::ROUTE_INVALID_STATE);
            }
            continue;
        }
        --*budget;
        if (current_phase == SchedulerCompletionPhase::INITIALIZE) {
            if (!scheduler_task_ref_valid(graph, producer_ref)) return fail(graph, local, run, producer_ref, *progress);
            auto *control = scheduler_task_control_at(base, local, producer_ref);
            if (validate_done_state &&
                scheduler_gm_query(control->state) != static_cast<int64_t>(SchedulerTaskState::DONE))
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_TASK_NOT_DONE);
            const int64_t head = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (head < SCHEDULER_WAKE_LIST_OPEN)
                return fail(
                    graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
                );
            progress->current_waiter = head;
            phase(*progress, SchedulerCompletionPhase::SELECT);
            if (wake_stats != nullptr) ++wake_stats->wake_close_count;
        } else if (current_phase == SchedulerCompletionPhase::SELECT) {
            if (progress->pending_count == 8 || progress->current_waiter == SCHEDULER_TASK_REF_INVALID) {
                if (progress->pending_count != 0) {
                    begin_publication(*progress, SchedulerCompletionPhase::SELECT);
                } else {
                    phase(*progress, SchedulerCompletionPhase::DRAINED);
                    return SchedulerCompletionResult::DONE;
                }
                continue;
            }
            if (!scheduler_task_ref_valid(graph, progress->current_waiter) ||
                progress->visited_count >= graph.task_count)
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            if (progress->pending_count != 0) {
                uint32_t fanin_count = 0;
                if (scheduler_node_fanin_count(graph, progress->current_waiter, &fanin_count) !=
                    SchedulerGraphResult::OK)
                    return fail(graph, local, run, producer_ref, *progress);
                // A long fanin must not keep an already-ready prefix private
                // while routing yields. Publish it before entering that cursor.
                if (fanin_count > 1) {
                    begin_publication(*progress, SchedulerCompletionPhase::SELECT);
                    continue;
                }
            }
            auto *waiter = scheduler_task_control_at(base, local, progress->current_waiter);
            progress->detached_next = scheduler_observe_next_waiter(waiter);
            if (progress->detached_next < SCHEDULER_WAKE_LIST_OPEN ||
                progress->detached_next == progress->current_waiter)
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            ++progress->visited_count;
            progress->resolve = scheduler_resolve_begin(progress->current_waiter);
            phase(*progress, SchedulerCompletionPhase::RESOLVE);
            if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        } else if (current_phase == SchedulerCompletionPhase::ENQUEUE) {
            int64_t candidate = progress->current_waiter;
            const auto *metadata = scheduler_task_metadata_at(base, local, candidate);
            const auto lanes = scheduler_task_lanes(*metadata);
            if (scheduler_task_is_executable(metadata->flags) &&
                scheduler_metadata_single_subtask_slot(lanes) == UINT8_MAX && !scheduler_task_is_mix(metadata->flags))
                return fail(
                    graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_SHAPE,
                    SchedulerGraphResult::UNSUPPORTED_SHAPE
                );
            Destination target{};
            if (!destination(graph, base, local, candidate, &target))
                return fail(graph, local, run, producer_ref, *progress);
            // Only a synchronous caller supplies direct sinks. A yielded FINISH
            // keeps every unpublished reference in the persisted cursor.
            bool return_mix = false;
            if (direct != nullptr && target.kind != SchedulerCompletionQueue::CONTROL) {
                if (target.queue == SCHEDULER_MIX_QUEUE && direct->mix != nullptr &&
                    !scheduler_task_ref_is_null(direct->mix->task_ref)) {
                    candidate = direct->mix->task_ref;
                    return_mix = true;
                    if (!destination(graph, base, local, candidate, &target))
                        return fail(graph, local, run, producer_ref, *progress);
                } else {
                    SchedulerReadyClaim *sink = nullptr;
                    if (target.queue == SCHEDULER_MIX_QUEUE && direct->mix != nullptr && !direct->mix_published &&
                        (progress->pending_count == 0 || progress->pending_queue != SCHEDULER_MIX_QUEUE)) {
                        auto *inbox =
                            scheduler_ready_inbox_at(base, local, SCHEDULER_MIX_QUEUE, local->self_type_rank());
                        if (scheduler_ready_endpoints_empty(scheduler_gm_query(inbox->endpoints)) &&
                            !scheduler_ready_directory_nonempty(
                                base, local, local->config.scheduler_count, SCHEDULER_MIX_QUEUE
                            ))
                            sink = direct->mix;
                    }
                    if (sink == nullptr && target.queue == direct->core_type && direct->ready != nullptr &&
                        scheduler_task_ref_is_null(direct->ready->task_ref))
                        sink = direct->ready;
                    if (sink != nullptr) {
                        sink->task_ref = candidate;
                        sink->source = SchedulerReadySource::DIRECT_RESOLVE;
                        sink->publication_mode = SchedulerPublicationMode::REFILL;
                        progress->current_waiter = progress->detached_next;
                        phase(*progress, SchedulerCompletionPhase::SELECT);
                        continue;
                    }
                }
            }
            if (progress->pending_count != 0 &&
                (progress->pending_count == 8 || progress->pending_kind != static_cast<uint64_t>(target.kind) ||
                 progress->pending_queue != target.queue || progress->pending_target != target.target)) {
                begin_publication(*progress, SchedulerCompletionPhase::ENQUEUE);
                continue;
            }
            progress->pending_kind = static_cast<uint64_t>(target.kind);
            progress->pending_queue = target.queue;
            progress->pending_target = target.target;
            progress->pending_tasks[progress->pending_count++] = candidate;
            if (return_mix) {
                *direct->mix = {};
            } else {
                progress->current_waiter = progress->detached_next;
                phase(*progress, SchedulerCompletionPhase::SELECT);
            }
            if (ready_stats != nullptr) ++ready_stats->enqueue_count;
            if (direct != nullptr && target.queue == SCHEDULER_MIX_QUEUE && !direct->mix_published &&
                progress->pending_count >= 2 && progress->current_waiter >= 0 && !return_mix) {
                direct->mix_published = true;
                begin_publication(*progress, SchedulerCompletionPhase::SELECT);
            }
        } else if (current_phase == SchedulerCompletionPhase::PREPARE) {
            if (!prepare(graph, base, local, *progress)) return fail(graph, local, run, producer_ref, *progress);
        } else if (current_phase == SchedulerCompletionPhase::PUBLISH) {
            if (direct != nullptr && progress->pending_queue == SCHEDULER_MIX_QUEUE) direct->mix_published = true;
            const auto result = publish<QueueOps>(graph, base, local, *progress);
            if (result == PublishResult::ERROR) return fail(graph, local, run, producer_ref, *progress);
            if (result == PublishResult::RETRY) {
                if (ready_stats != nullptr) ++ready_stats->contention_giveup_count;
                return SchedulerCompletionResult::PENDING;
            }
            if (result == PublishResult::PUBLISHED && ready_stats != nullptr) ++ready_stats->batch_count;
        } else if (current_phase == SchedulerCompletionPhase::PUBLISH_DIRECTORY) {
            const auto result = publish_directory<QueueOps>(base, local, *progress);
            if (result == PublishResult::ERROR) return fail(graph, local, run, producer_ref, *progress);
            if (result == PublishResult::RETRY) {
                if (ready_stats != nullptr) ++ready_stats->contention_giveup_count;
                return SchedulerCompletionResult::PENDING;
            }
            if (ready_stats != nullptr) ++ready_stats->batch_count;
        } else {
            return fail(graph, local, run, producer_ref, *progress);
        }
    }
}
