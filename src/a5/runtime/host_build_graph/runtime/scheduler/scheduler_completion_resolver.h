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

#include "scheduler_compile.h"

#include "scheduler_ready_publication.h"

// The producer is already DONE. This cursor owns its detached chain until all
// successors are handed off and every ready publication has finished. Returning
// DONE does not itself contribute to the run's completion counter.
SCHEDULER_HEAVY_INLINE __aicore__ SchedulerResolveProgress scheduler_resolve_completion_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerResolveCursor &cursor, uint32_t budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr, SchedulerCompletionStats *completion_stats = nullptr,
    uint64_t profiling_level = 0, SchedulerCompletionDirect *direct = nullptr, bool validate_done_state = true
) {
    uint32_t used = 0;
    auto fail = [&](SchedulerErrorSite site = SchedulerErrorSite::UNKNOWN,
                    SchedulerGraphResult error = SchedulerGraphResult::INVALID_ARGUMENTS) __aicore__ {
        cursor.phase = SchedulerResolvePhase::FAILED;
        if (run != nullptr) scheduler_record_error(run, producer_ref, error, &graph, local, site);
        return SchedulerResolveProgress{SchedulerResolveResult::ERROR, used};
    };
    if (base == nullptr || local == nullptr || run == nullptr || !scheduler_task_ref_valid(graph, producer_ref) ||
        (cursor.producer_ref != SCHEDULER_TASK_REF_INVALID && cursor.producer_ref != producer_ref))
        return fail();
    if (cursor.phase != SchedulerResolvePhase::START && cursor.producer_ref != producer_ref) return fail();
    if (cursor.phase == SchedulerResolvePhase::FAILED) return {SchedulerResolveResult::ERROR, 0};
    if (cursor.phase == SchedulerResolvePhase::DONE) return {SchedulerResolveResult::DONE, 0};
    if (scheduler_read_error(run) != 0) {
        cursor.phase = SchedulerResolvePhase::FAILED;
        return {SchedulerResolveResult::ERROR, 0};
    }
    const bool timing = scheduler_phase_timing_enabled(profiling_level);
    if (timing && local->profiling == nullptr) return fail();
    while (used < budget) {
        // Check at successor and publication boundaries; local cursor-only
        // transitions retain the error observation made at this slice's entry.
        if ((cursor.phase == SchedulerResolvePhase::SELECT || cursor.phase == SchedulerResolvePhase::PUBLISH) &&
            scheduler_read_error(run) != 0) {
            cursor.phase = SchedulerResolvePhase::FAILED;
            return {SchedulerResolveResult::ERROR, used};
        }
        switch (cursor.phase) {
        case SchedulerResolvePhase::START: {
            ++used;
            auto *control = scheduler_task_control_at(base, local, producer_ref);
            if (validate_done_state &&
                scheduler_gm_query(control->state) != static_cast<int64_t>(SchedulerTaskState::DONE))
                return fail(SchedulerErrorSite::COMPLETION_TASK_NOT_DONE);
            if (timing) {
                scheduler_observe_cache_line(&control->next_fanin_index);
                control->completion_resolve_start_cycles = scheduler_cycles();
                control->scheduler_worker_id = local->worker_id();
                control->completion_resolve_loop_iter = local->profiling->loop_iter;
                scheduler_publish_cache_line(&control->next_fanin_index);
            }
            cursor.producer_ref = producer_ref;
            cursor.next_waiter = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (cursor.next_waiter < SCHEDULER_WAKE_LIST_OPEN)
                return fail(SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED);
            if (wake_stats != nullptr) ++wake_stats->wake_close_count;
            cursor.phase = SchedulerResolvePhase::SELECT;
            break;
        }
        case SchedulerResolvePhase::SELECT: {
            ++used;
            if (cursor.next_waiter == SCHEDULER_WAKE_LIST_OPEN) {
                if (cursor.publication.count != 0) {
                    cursor.phase = SchedulerResolvePhase::PUBLISH;
                    break;
                }
                if (timing) {
                    auto *control = scheduler_task_control_at(base, local, producer_ref);
                    scheduler_observe_cache_line(&control->next_fanin_index);
                    control->completion_resolve_end_cycles = scheduler_cycles();
                    scheduler_publish_cache_line(&control->next_fanin_index);
                }
                cursor.phase = SchedulerResolvePhase::DONE;
                if (completion_stats != nullptr) ++completion_stats->resolve_count;
                return {SchedulerResolveResult::DONE, used};
            }
            if (!scheduler_task_ref_valid(graph, cursor.next_waiter) || cursor.visited_count >= graph.task_count)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            uint32_t fanin_count = 0;
            if ((direct != nullptr || cursor.publication.count != 0) &&
                scheduler_node_fanin_count(graph, cursor.next_waiter, &fanin_count) != SchedulerGraphResult::OK)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            // A ready prefix must become visible before a later fanin traversal can yield.
            if (cursor.publication.count != 0 && fanin_count > 1) {
                cursor.phase = SchedulerResolvePhase::PUBLISH;
                break;
            }
            cursor.waiter_ref = cursor.next_waiter;
            cursor.saved_next =
                scheduler_observe_next_waiter(scheduler_task_control_at(base, local, cursor.waiter_ref));
            if (cursor.saved_next < SCHEDULER_WAKE_LIST_OPEN || cursor.saved_next == cursor.waiter_ref)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            ++cursor.visited_count;
            cursor.route = {};
            if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
            // A detached single-fanin waiter is ready when its unique producer completes.
            cursor.phase =
                direct != nullptr && fanin_count == 1 ? SchedulerResolvePhase::APPEND : SchedulerResolvePhase::ROUTE;
            break;
        }
        case SchedulerResolvePhase::ROUTE: {
            const auto progress = scheduler_route_task_step(
                graph, base, local, run, cursor.waiter_ref, cursor.route, budget - used, wake_stats
            );
            used += progress.work_units;
            if (progress.result == SchedulerRouteResult::ERROR) return fail();
            if (progress.result == SchedulerRouteResult::YIELD) return {SchedulerResolveResult::YIELD, used};
            if (progress.result == SchedulerRouteResult::READY_TO_ENQUEUE) {
                cursor.phase = SchedulerResolvePhase::APPEND;
            } else {
                // The old next was captured before routing transferred the waiter.
                cursor.next_waiter = cursor.saved_next;
                cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
                cursor.phase = SchedulerResolvePhase::SELECT;
            }
            break;
        }
        case SchedulerResolvePhase::APPEND: {
            ++used;
            int64_t candidate = cursor.waiter_ref;
            uint32_t type = SCHEDULER_CONTROL_PUBLICATION_QUEUE;
            bool return_mix = false;
            if (direct != nullptr) {
                bool control = false;
                if (graph.native_header_offset != 0) {
                    SchedulerNodeKind kind{};
                    if (scheduler_node_kind(graph, candidate, &kind) != SchedulerGraphResult::OK)
                        return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
                    control = scheduler_node_is_control(kind);
                }
                if (!control) {
                    const auto &metadata = *scheduler_task_metadata_at(base, local, candidate);
                    const LaneMask lanes = scheduler_task_lanes(metadata);
                    if (!scheduler_task_is_executable(metadata.flags))
                        return fail(SchedulerErrorSite::COMPLETION_WAITER_NOT_EXECUTABLE);
                    if (scheduler_metadata_single_subtask_slot(lanes) == UINT8_MAX &&
                        !scheduler_task_is_mix(metadata.flags))
                        return fail(
                            SchedulerErrorSite::COMPLETION_INVALID_SHAPE, SchedulerGraphResult::UNSUPPORTED_SHAPE
                        );
                    type = scheduler_task_ready_queue(metadata.flags, lanes);
                    if (type == SCHEDULER_MIX_QUEUE && direct->mix != nullptr &&
                        !scheduler_task_ref_is_null(direct->mix->task_ref)) {
                        // Two MIX successors require ordered publication of the earlier candidate.
                        candidate = direct->mix->task_ref;
                        return_mix = true;
                    } else {
                        SchedulerReadyClaim *sink = nullptr;
                        if (type == SCHEDULER_MIX_QUEUE && direct->mix != nullptr && !direct->mix_prefix_published &&
                            (cursor.publication.count == 0 || cursor.publication.type != SCHEDULER_MIX_QUEUE)) {
                            auto *inbox = scheduler_ready_inbox_at(base, local, type, local->self_type_rank());
                            if (scheduler_ready_endpoints_empty(scheduler_gm_query(inbox->endpoints)) &&
                                !scheduler_ready_directory_nonempty(base, local, local->config.scheduler_count, type))
                                sink = direct->mix;
                        }
                        if (sink == nullptr && type == direct->core_type && direct->ready != nullptr &&
                            scheduler_task_ref_is_null(direct->ready->task_ref))
                            sink = direct->ready;
                        if (sink != nullptr) {
                            sink->task_ref = candidate;
                            sink->source = SchedulerReadySource::DIRECT_RESOLVE;
                            sink->publication_mode = SchedulerPublicationMode::REFILL;
                            cursor.next_waiter = cursor.saved_next;
                            cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
                            cursor.phase = SchedulerResolvePhase::SELECT;
                            break;
                        }
                    }
                }
            }
            const auto appended = scheduler_ready_publication_append(graph, base, local, candidate, cursor.publication);
            if (appended == SchedulerReadyAppendResult::ERROR)
                return fail(SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED);
            if (appended == SchedulerReadyAppendResult::FULL) {
                cursor.phase = SchedulerResolvePhase::PUBLISH;
                break;
            }
            if (ready_stats != nullptr) ++ready_stats->enqueue_count;
            if (return_mix) {
                *direct->mix = {};
            } else {
                cursor.next_waiter = cursor.saved_next;
                cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
            }
            const bool mix_prefix = direct != nullptr && type == SCHEDULER_MIX_QUEUE && !direct->mix_prefix_published &&
                                    !return_mix && cursor.next_waiter >= 0 && cursor.publication.count >= 2;
            cursor.phase = cursor.publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY || mix_prefix ?
                               SchedulerResolvePhase::PUBLISH :
                           return_mix ? SchedulerResolvePhase::APPEND :
                                        SchedulerResolvePhase::SELECT;
            break;
        }
        case SchedulerResolvePhase::PUBLISH: {
            ++used;
            const auto published =
                scheduler_ready_publication_step(graph, base, local, cursor.publication, ready_stats);
            if (published == SchedulerReadyPublishResult::ERROR)
                return fail(SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED);
            if (published == SchedulerReadyPublishResult::DONE) {
                if (direct != nullptr && cursor.publication.type == SCHEDULER_MIX_QUEUE)
                    direct->mix_prefix_published = true;
                cursor.publication = {};
                cursor.phase = cursor.waiter_ref == SCHEDULER_TASK_REF_INVALID ? SchedulerResolvePhase::SELECT :
                                                                                 SchedulerResolvePhase::APPEND;
            }
            break;
        }
        default:
            return fail();
        }
    }
    return {SchedulerResolveResult::YIELD, used};
}
