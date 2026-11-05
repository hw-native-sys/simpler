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

#include "scheduler_async_layout.h"
#include "scheduler_ready.h"

enum class SchedulerAsyncFinishResult : uint8_t { PENDING, DONE, ERROR };
enum class SchedulerAsyncFinishPhase : uint64_t {
    INITIALIZE,
    SELECT,
    RESOLVE,
    ENQUEUE,
    PREPARE,
    PUBLISH,
    PUBLISH_DIRECTORY,
    DRAINED,
    CONTRIBUTED,
    FAILED
};
enum class SchedulerAsyncFinishQueue : uint64_t { INBOX, CONTROL, TICKET, LINKED, CHUNK };

// The production policy never hides a retry loop. Tests can place a competing
// publisher at the reservation/CAS boundary without a device runtime hook.
struct SchedulerAsyncFinishQueueOps {
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

namespace scheduler_async_finish_detail {
inline __aicore__ SchedulerAsyncFinishPhase phase(const SchedulerAsyncFinishProgress &progress) {
    return static_cast<SchedulerAsyncFinishPhase>(progress.phase);
}
inline __aicore__ void phase(SchedulerAsyncFinishProgress &progress, SchedulerAsyncFinishPhase next) {
    progress.phase = static_cast<uint64_t>(next);
}

inline __aicore__ SchedulerAsyncFinishResult fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task,
    SchedulerAsyncFinishProgress &progress,
    SchedulerErrorSite site = SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED
) {
    phase(progress, SchedulerAsyncFinishPhase::FAILED);
    scheduler_record_error(run, task, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site);
    return SchedulerAsyncFinishResult::ERROR;
}

// A finishing core need not still participate in the task's admission. The
// immutable run record, rather than this core's admission frontier, owns order.
inline __aicore__ bool protocol_order(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint64_t epoch,
    const SchedulerTaskMetadata &metadata, uint64_t *order
) {
    *order = UINT64_MAX;
    if (!scheduler_protocol_enabled(local) || !scheduler_task_requires_admission(metadata)) return true;
    const auto *admission = local->admission;
    if (admission == nullptr || admission->records_offset == 0 ||
        admission->record_stride < sizeof(SchedulerAdmissionRecord) || admission->epoch != epoch)
        return false;
    auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
        base, admission->records_offset + static_cast<uint64_t>(task) * admission->record_stride
    );
    scheduler_observe_cache_line(record);
    if (record->run_epoch != epoch || record->task_ref != task || record->order_key >= admission->capacity)
        return false;
    *order = record->order_key;
    return true;
}

struct Destination {
    SchedulerAsyncFinishQueue kind{SchedulerAsyncFinishQueue::INBOX};
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
        *out = {SchedulerAsyncFinishQueue::CONTROL, SCHEDULER_READY_QUEUE_COUNT, local->worker_id()};
        return true;
    }
    const uint32_t type = scheduler_task_ready_queue(metadata->flags, lanes);
    if (!scheduler_task_is_executable(metadata->flags) || type >= SCHEDULER_READY_QUEUE_COUNT) return false;
    out->queue = type;
    if (local->config.ready_pool_offset == 0) {
        out->kind = SchedulerAsyncFinishQueue::INBOX;
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
        out->kind = SchedulerAsyncFinishQueue::TICKET;
        break;
    case SchedulerReadyPoolKind::LINKED:
        out->kind = SchedulerAsyncFinishQueue::LINKED;
        break;
    case SchedulerReadyPoolKind::CHUNK:
        out->kind = SchedulerAsyncFinishQueue::CHUNK;
        break;
    default:
        return false;
    }
    return true;
}

inline __aicore__ void begin_publication(SchedulerAsyncFinishProgress &progress, SchedulerAsyncFinishPhase resume) {
    progress.resume_phase = static_cast<uint64_t>(resume);
    phase(progress, SchedulerAsyncFinishPhase::PREPARE);
}

// Preparation owns at most eight tasks and does not reserve shared queue space.
// Prepared links and chunk identity persist across retry and owner handoff.
inline __aicore__ bool prepare(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerAsyncFinishProgress &progress
) {
    const uint32_t count = static_cast<uint32_t>(progress.pending_count);
    if (count == 0 || progress.pending_count > 8) return false;
    for (uint32_t index = 0; index < count; ++index) {
        if (!scheduler_task_ref_valid(graph, progress.pending_tasks[index])) return false;
        for (uint32_t prior = 0; prior < index; ++prior)
            if (progress.pending_tasks[prior] == progress.pending_tasks[index]) return false;
    }
    const auto kind = static_cast<SchedulerAsyncFinishQueue>(progress.pending_kind);
    SchedulerLinkedBatch batch{};
    if (kind == SchedulerAsyncFinishQueue::LINKED || kind == SchedulerAsyncFinishQueue::CHUNK) {
        const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
        const auto result = kind == SchedulerAsyncFinishQueue::CHUNK ?
                                scheduler_chunk_prepare(
                                    scheduler_ready_pool_chunk_queue(base, pool, progress.pending_target),
                                    progress.pending_tasks, count, &batch
                                ) :
                                scheduler_linked_prepare(
                                    scheduler_ready_pool_linked_queue(base, pool, progress.pending_target),
                                    progress.pending_tasks, count, &batch
                                );
        if (result != SchedulerLinkedResult::SUCCESS) return false;
    } else if (kind == SchedulerAsyncFinishQueue::INBOX || kind == SchedulerAsyncFinishQueue::CONTROL) {
        for (uint32_t index = 0; index < count; ++index) {
            auto *task = scheduler_task_control_at(base, local, progress.pending_tasks[index]);
            scheduler_gm_store(
                task->next_waiter, index + 1 == count ? SCHEDULER_INBOX_EMPTY : progress.pending_tasks[index + 1]
            );
        }
        scheduler_cache_barrier();
    } else if (kind != SchedulerAsyncFinishQueue::TICKET) {
        return false;
    }
    progress.pending_first = batch.first;
    progress.pending_last = batch.last;
    progress.pending_link_count = batch.count;
    phase(progress, SchedulerAsyncFinishPhase::PUBLISH);
    return true;
}

enum class PublishResult : uint8_t { PUBLISHED, DIRECTORY_PENDING, RETRY, ERROR };

inline __aicore__ void published(SchedulerAsyncFinishProgress &progress) {
    progress.pending_count = 0;
    progress.pending_first = progress.pending_last = SCHEDULER_TASK_REF_INVALID;
    progress.pending_link_count = 0;
    progress.phase = progress.resume_phase;
}

template <typename QueueOps>
inline __aicore__ PublishResult publish(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerAsyncFinishProgress &progress
) {
    const auto kind = static_cast<SchedulerAsyncFinishQueue>(progress.pending_kind);
    if (progress.pending_count == 0 || progress.pending_count > 8) return PublishResult::ERROR;
    if (kind == SchedulerAsyncFinishQueue::TICKET) {
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
    } else if (kind == SchedulerAsyncFinishQueue::LINKED || kind == SchedulerAsyncFinishQueue::CHUNK) {
        const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
        if (progress.pending_link_count == 0 || progress.pending_link_count > 8) return PublishResult::ERROR;
        SchedulerLinkedBatch batch{
            progress.pending_first, progress.pending_last, static_cast<uint32_t>(progress.pending_link_count)
        };
        const auto result =
            QueueOps::push(scheduler_ready_pool_linked_queue(base, pool, progress.pending_target), &batch);
        if (result == SchedulerLinkedResult::RETRY) return PublishResult::RETRY;
        if (result != SchedulerLinkedResult::SUCCESS) return PublishResult::ERROR;
    } else if (kind == SchedulerAsyncFinishQueue::INBOX || kind == SchedulerAsyncFinishQueue::CONTROL) {
        auto *inbox = kind == SchedulerAsyncFinishQueue::CONTROL ?
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
        if (kind == SchedulerAsyncFinishQueue::INBOX) {
            // The endpoint CAS already transferred all nodes. The directory's
            // shared OR may contend with other publishers, so retain only its
            // outstanding responsibility and never publish this batch again.
            phase(progress, SchedulerAsyncFinishPhase::PUBLISH_DIRECTORY);
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
publish_directory(__gm__ void *base, SchedulerLocalState *local, SchedulerAsyncFinishProgress &progress) {
    if (progress.pending_kind != static_cast<uint64_t>(SchedulerAsyncFinishQueue::INBOX) ||
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
}  // namespace scheduler_async_finish_detail

// The caller owns the FINISH lease and passes a local copy of its GM progress.
// Before relinquishing that lease it must publish this copy back to the record.
// Each unit performs a fixed transition, one real resolver operation, one <=8
// preparation, or one publication attempt. RETRY always hands control back.
// DONE means all successors are public/drained, not that completion was counted.
template <typename QueueOps = SchedulerAsyncFinishQueueOps, typename Link = SchedulerResolveLink>
inline __aicore__ SchedulerAsyncFinishResult scheduler_async_finish_step_impl(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAsyncFinishIdentity &identity, __gm__ SchedulerAsyncTaskJoin *join,
    SchedulerAsyncFinishProgress *progress, uint32_t *budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr
) {
    using namespace scheduler_async_finish_detail;
    if (progress == nullptr || budget == nullptr || base == nullptr || local == nullptr || run == nullptr ||
        join == nullptr)
        return SchedulerAsyncFinishResult::ERROR;
    if (phase(*progress) == SchedulerAsyncFinishPhase::CONTRIBUTED) return SchedulerAsyncFinishResult::DONE;
    if (phase(*progress) == SchedulerAsyncFinishPhase::FAILED) return SchedulerAsyncFinishResult::ERROR;
    if (progress->phase > static_cast<uint64_t>(SchedulerAsyncFinishPhase::FAILED))
        return fail(graph, local, run, identity.task_ref, *progress);
    if (scheduler_read_error(run) != 0 || scheduler_async_join_load(*join) != 0) {
        phase(*progress, SchedulerAsyncFinishPhase::FAILED);
        if (scheduler_read_error(run) == 0) return fail(graph, local, run, identity.task_ref, *progress);
        return SchedulerAsyncFinishResult::ERROR;
    }
    for (;;) {
        if (scheduler_read_error(run) != 0) {
            phase(*progress, SchedulerAsyncFinishPhase::FAILED);
            return SchedulerAsyncFinishResult::ERROR;
        }
        const auto current_phase = phase(*progress);
        if (current_phase == SchedulerAsyncFinishPhase::DRAINED) return SchedulerAsyncFinishResult::DONE;
        if (*budget == 0) return SchedulerAsyncFinishResult::PENDING;
        if (current_phase == SchedulerAsyncFinishPhase::RESOLVE) {
            const auto result =
                scheduler_resolve_step<Link>(graph, base, local, run, &progress->resolve, budget, wake_stats);
            if (result == SchedulerRouteResult::ERROR) {
                phase(*progress, SchedulerAsyncFinishPhase::FAILED);
                return SchedulerAsyncFinishResult::ERROR;
            }
            if (result == SchedulerRouteResult::YIELD) return SchedulerAsyncFinishResult::PENDING;
            if (result == SchedulerRouteResult::READY_TO_ENQUEUE) {
                phase(*progress, SchedulerAsyncFinishPhase::ENQUEUE);
            } else {
                progress->current_waiter = progress->detached_next;
                phase(*progress, SchedulerAsyncFinishPhase::SELECT);
            }
            continue;
        }
        --*budget;
        if (current_phase == SchedulerAsyncFinishPhase::INITIALIZE) {
            if (!scheduler_task_ref_valid(graph, identity.task_ref) || identity.run_epoch == 0 ||
                local->config.graph_summary_offset == 0 ||
                scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset)->run_epoch !=
                    identity.run_epoch)
                return fail(graph, local, run, identity.task_ref, *progress);
            const auto metadata = scheduler_load_dispatch_metadata(base, local, identity.task_ref, 0);
            uint64_t order = UINT64_MAX;
            if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags) ||
                !protocol_order(base, local, identity.task_ref, identity.run_epoch, metadata, &order))
                return fail(graph, local, run, identity.task_ref, *progress);
            auto *control = scheduler_task_control_at(base, local, identity.task_ref);
            scheduler_protocol_task_actor(base, local, identity.task_ref, order, metadata.flags, true, true);
            if (scheduler_read_error(run) != 0) return fail(graph, local, run, identity.task_ref, *progress);
            if (scheduler_gm_compare_exchange(
                    control->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED),
                    static_cast<int64_t>(SchedulerTaskState::DONE)
                ) != static_cast<int64_t>(SchedulerTaskState::BLOCKED))
                return fail(graph, local, run, identity.task_ref, *progress, SchedulerErrorSite::ROUTE_INVALID_STATE);
            scheduler_protocol_task_actor(base, local, identity.task_ref, order, metadata.flags, true, false);
            const int64_t head = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (head < SCHEDULER_WAKE_LIST_OPEN)
                return fail(
                    graph, local, run, identity.task_ref, *progress, SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
                );
            progress->current_waiter = head;
            phase(*progress, SchedulerAsyncFinishPhase::SELECT);
            if (wake_stats != nullptr) ++wake_stats->wake_close_count;
        } else if (current_phase == SchedulerAsyncFinishPhase::SELECT) {
            if (progress->pending_count == 8 || progress->current_waiter == SCHEDULER_TASK_REF_INVALID) {
                if (progress->pending_count != 0) {
                    begin_publication(*progress, SchedulerAsyncFinishPhase::SELECT);
                } else {
                    phase(*progress, SchedulerAsyncFinishPhase::DRAINED);
                    return SchedulerAsyncFinishResult::DONE;
                }
                continue;
            }
            if (!scheduler_task_ref_valid(graph, progress->current_waiter) ||
                progress->visited_count >= graph.task_count)
                return fail(
                    graph, local, run, identity.task_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_WAITER
                );
            auto *waiter = scheduler_task_control_at(base, local, progress->current_waiter);
            progress->detached_next = scheduler_observe_next_waiter(waiter);
            if (progress->detached_next < SCHEDULER_WAKE_LIST_OPEN ||
                progress->detached_next == progress->current_waiter)
                return fail(
                    graph, local, run, identity.task_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_WAITER
                );
            ++progress->visited_count;
            progress->resolve = scheduler_resolve_begin(progress->current_waiter);
            phase(*progress, SchedulerAsyncFinishPhase::RESOLVE);
            if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        } else if (current_phase == SchedulerAsyncFinishPhase::ENQUEUE) {
            Destination target{};
            if (!destination(graph, base, local, progress->current_waiter, &target))
                return fail(graph, local, run, identity.task_ref, *progress);
            if (progress->pending_count != 0 &&
                (progress->pending_kind != static_cast<uint64_t>(target.kind) ||
                 progress->pending_queue != target.queue || progress->pending_target != target.target)) {
                begin_publication(*progress, SchedulerAsyncFinishPhase::ENQUEUE);
                continue;
            }
            if (progress->pending_count >= 8) return fail(graph, local, run, identity.task_ref, *progress);
            progress->pending_kind = static_cast<uint64_t>(target.kind);
            progress->pending_queue = target.queue;
            progress->pending_target = target.target;
            progress->pending_tasks[progress->pending_count++] = progress->current_waiter;
            progress->current_waiter = progress->detached_next;
            phase(*progress, SchedulerAsyncFinishPhase::SELECT);
            if (ready_stats != nullptr) ++ready_stats->enqueue_count;
        } else if (current_phase == SchedulerAsyncFinishPhase::PREPARE) {
            if (!prepare(graph, base, local, *progress)) return fail(graph, local, run, identity.task_ref, *progress);
        } else if (current_phase == SchedulerAsyncFinishPhase::PUBLISH) {
            const auto result = publish<QueueOps>(graph, base, local, *progress);
            if (result == PublishResult::ERROR) return fail(graph, local, run, identity.task_ref, *progress);
            if (result == PublishResult::RETRY) {
                if (ready_stats != nullptr) ++ready_stats->contention_giveup_count;
                return SchedulerAsyncFinishResult::PENDING;
            }
            if (result == PublishResult::PUBLISHED && ready_stats != nullptr) ++ready_stats->batch_count;
        } else if (current_phase == SchedulerAsyncFinishPhase::PUBLISH_DIRECTORY) {
            const auto result = publish_directory<QueueOps>(base, local, *progress);
            if (result == PublishResult::ERROR) return fail(graph, local, run, identity.task_ref, *progress);
            if (result == PublishResult::RETRY) {
                if (ready_stats != nullptr) ++ready_stats->contention_giveup_count;
                return SchedulerAsyncFinishResult::PENDING;
            }
            if (ready_stats != nullptr) ++ready_stats->batch_count;
        } else {
            return fail(graph, local, run, identity.task_ref, *progress);
        }
    }
}

struct SchedulerAsyncFinishClock {
    inline __aicore__ static uint64_t now() { return scheduler_cycles(); }
};

inline __aicore__ void scheduler_async_finish_accumulate(uint64_t &total, uint64_t added) {
    total = added > UINT64_MAX - total ? UINT64_MAX : total + added;
}

// Every measured interval starts and ends on the same core. The old synchronous
// completion_resolve span would attribute all handoffs and gaps to its first
// worker, so async FINISH never emits that pair. Host export can consume these
// owned counters independently; no common clock origin across cores is assumed.
template <
    typename QueueOps = SchedulerAsyncFinishQueueOps, typename Link = SchedulerResolveLink,
    typename Clock = SchedulerAsyncFinishClock>
inline __aicore__ SchedulerAsyncFinishResult scheduler_async_finish_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAsyncFinishIdentity &identity, __gm__ SchedulerAsyncTaskJoin *join,
    SchedulerAsyncFinishProgress *progress, uint32_t *budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr, uint64_t profiling_level = 0
) {
    const bool measured = scheduler_phase_timing_enabled(profiling_level) && progress != nullptr;
    const uint64_t start = measured ? Clock::now() : 0;
    const uint32_t initial_budget = budget == nullptr ? 0 : *budget;
    const auto result = scheduler_async_finish_step_impl<QueueOps, Link>(
        graph, base, local, run, identity, join, progress, budget, wake_stats, ready_stats
    );
    if (measured) {
        scheduler_async_finish_accumulate(progress->active_slice_cycles, Clock::now() - start);
        scheduler_async_finish_accumulate(progress->slice_count, 1);
        scheduler_async_finish_accumulate(progress->work_units, budget == nullptr ? 0 : initial_budget - *budget);
    }
    return result;
}

// The terminal owner calls this in the same non-yielding tail after publishing
// progress, transitioning the lease to TERMINAL, and clearing its directory bit.
// It then publishes CONTRIBUTED before returning. No terminal record is claimed
// again, and no local pending-publication descriptor participates in this tail.
inline __aicore__ bool scheduler_async_finish_contribute(
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, __gm__ SchedulerAsyncTaskJoin &join,
    __gm__ SchedulerAsyncLease &lease, __gm__ SchedulerAsyncDirectoryWord &directory, uint32_t bit,
    SchedulerAsyncFinishProgress *progress, SchedulerCompletionStats *stats = nullptr
) {
    using namespace scheduler_async_finish_detail;
    if (local == nullptr || run == nullptr || progress == nullptr || bit >= 64) return false;
    if (phase(*progress) == SchedulerAsyncFinishPhase::CONTRIBUTED) return true;
    if (phase(*progress) != SchedulerAsyncFinishPhase::DRAINED || scheduler_read_error(run) != 0 ||
        scheduler_async_join_load(join) != 0)
        return false;
    const auto word = scheduler_async_lease_load(lease);
    if (!scheduler_async_lease_valid(word) || scheduler_async_lease_phase(word) != SchedulerAsyncPhase::TERMINAL ||
        (scheduler_async_directory_load(directory) & (UINT64_C(1) << bit)) != 0)
        return false;
    ++local->pending_completed;
    if (stats != nullptr) ++stats->resolve_count;
    phase(*progress, SchedulerAsyncFinishPhase::CONTRIBUTED);
    return true;
}
