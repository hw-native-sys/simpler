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

#include "scheduler_ready_publication.h"

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
    if (progress->publication.phase > static_cast<uint64_t>(SchedulerCompletionPhase::FAILED))
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
            if (progress->publication.pending_count == 8 || progress->current_waiter == SCHEDULER_TASK_REF_INVALID) {
                if (progress->publication.pending_count != 0) {
                    begin_publication(progress->publication, SchedulerCompletionPhase::SELECT);
                } else {
                    phase(*progress, SchedulerCompletionPhase::DRAINED);
                    return SchedulerCompletionResult::DONE;
                }
                continue;
            }
            if (!scheduler_task_ref_valid(graph, progress->current_waiter) ||
                progress->visited_count >= graph.task_count)
                return fail(graph, local, run, producer_ref, *progress, SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            if (progress->publication.pending_count != 0) {
                uint32_t fanin_count = 0;
                if (scheduler_node_fanin_count(graph, progress->current_waiter, &fanin_count) !=
                    SchedulerGraphResult::OK)
                    return fail(graph, local, run, producer_ref, *progress);
                // A long fanin must not keep an already-ready prefix private
                // while routing yields. Publish it before entering that cursor.
                if (fanin_count > 1) {
                    begin_publication(progress->publication, SchedulerCompletionPhase::SELECT);
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
                        (progress->publication.pending_count == 0 ||
                         progress->publication.pending_queue != SCHEDULER_MIX_QUEUE)) {
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
            if (progress->publication.pending_count != 0 &&
                (progress->publication.pending_count == 8 ||
                 progress->publication.pending_kind != static_cast<uint64_t>(target.kind) ||
                 progress->publication.pending_queue != target.queue ||
                 progress->publication.pending_target != target.target)) {
                begin_publication(progress->publication, SchedulerCompletionPhase::ENQUEUE);
                continue;
            }
            progress->publication.pending_kind = static_cast<uint64_t>(target.kind);
            progress->publication.pending_queue = target.queue;
            progress->publication.pending_target = target.target;
            progress->publication.pending_tasks[progress->publication.pending_count++] = candidate;
            if (return_mix) {
                *direct->mix = {};
            } else {
                progress->current_waiter = progress->detached_next;
                phase(*progress, SchedulerCompletionPhase::SELECT);
            }
            if (ready_stats != nullptr) ++ready_stats->enqueue_count;
            if (direct != nullptr && target.queue == SCHEDULER_MIX_QUEUE && !direct->mix_published &&
                progress->publication.pending_count >= 2 && progress->current_waiter >= 0 && !return_mix) {
                direct->mix_published = true;
                begin_publication(progress->publication, SchedulerCompletionPhase::SELECT);
            }
        } else if (current_phase == SchedulerCompletionPhase::PREPARE ||
                   current_phase == SchedulerCompletionPhase::PUBLISH ||
                   current_phase == SchedulerCompletionPhase::PUBLISH_DIRECTORY) {
            if (current_phase == SchedulerCompletionPhase::PUBLISH && direct != nullptr &&
                progress->publication.pending_queue == SCHEDULER_MIX_QUEUE)
                direct->mix_published = true;
            const auto result = advance_publication<QueueOps>(graph, base, local, progress->publication);
            if (result == PublishResult::ERROR) return fail(graph, local, run, producer_ref, *progress);
            if (result == PublishResult::RETRY) {
                if (ready_stats != nullptr) ++ready_stats->contention_giveup_count;
                return SchedulerCompletionResult::PENDING;
            }
            if (result == PublishResult::PUBLISHED && ready_stats != nullptr) ++ready_stats->batch_count;
        } else {
            return fail(graph, local, run, producer_ref, *progress);
        }
    }
}
