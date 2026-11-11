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

// CCEC has failed when over-aligned stack objects escape through an outlined
// call (#2973). Local arguments at this boundary use natural alignment; shared
// scheduler records are passed through their explicitly qualified GM pointers.
static_assert(
    alignof(SchedulerGraphAccess) <= 8 && alignof(SchedulerLocalState) <= 8 && alignof(SchedulerResolveCursor) <= 8 &&
        alignof(SchedulerCompletionDirect) <= 8 && alignof(SchedulerReadyClaim) <= 8,
    "outlined resolver local arguments must not require over-aligned stack storage"
);

// The producer is already DONE. This cursor owns its successor traversal until
// every ready publication and direct handoff has finished. Returning
// DONE does not itself contribute to the run's completion counter.
static __aicore__ __attribute__((noinline)) SchedulerResolveProgress scheduler_resolve_completion_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerResolveCursor &cursor, uint32_t budget, uint64_t profiling_level = 0,
    SchedulerCompletionDirect *direct = nullptr, bool validate_done_state = true
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarResolve, producer_ref);
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
    const auto *dependencies = scheduler_dependencies(base, local->config.ready_pool_offset);
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
            if (dependencies != nullptr &&
                (cursor.next_waiter != SCHEDULER_WAKE_LIST_OPEN ||
                 !scheduler_dependency_successor(base, *dependencies, producer_ref, 0, &cursor.next_waiter)))
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);

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

                return {SchedulerResolveResult::DONE, used};
            }
            if (!scheduler_task_ref_valid(graph, cursor.next_waiter) || cursor.visited_count >= graph.task_count)
                return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
            if (dependencies != nullptr) {
                cursor.waiter_ref = cursor.next_waiter;
                if (!scheduler_dependency_successor(
                        base, *dependencies, producer_ref, cursor.visited_count + 1, &cursor.saved_next
                    ))
                    return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
                ++cursor.visited_count;
                const auto result = scheduler_dependency_notify(base, *dependencies, cursor.waiter_ref);
                if (result == SchedulerDependencyResult::ERROR)
                    return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
                if (result == SchedulerDependencyResult::READY) {
                    cursor.phase = SchedulerResolvePhase::APPEND;
                } else {
                    cursor.next_waiter = cursor.saved_next;
                    cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
                }
                break;
            }
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

            // A detached single-fanin waiter is ready when its unique producer completes.
            cursor.phase =
                direct != nullptr && fanin_count == 1 ? SchedulerResolvePhase::APPEND : SchedulerResolvePhase::ROUTE;
            break;
        }
        case SchedulerResolvePhase::ROUTE: {
            const auto progress =
                scheduler_route_task_step(graph, base, local, run, cursor.waiter_ref, cursor.route, budget - used);
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
                    const uint32_t type = scheduler_task_ready_queue(metadata.flags, lanes);
                    if (!scheduler_task_is_mix(metadata.flags) && type == direct->core_type &&
                        direct->ready != nullptr && scheduler_task_ref_is_null(direct->ready->task_ref)) {
                        direct->ready->task_ref = candidate;
                        direct->ready->source = SchedulerReadySource::DIRECT_RESOLVE;
                        direct->ready->publication_mode = SchedulerPublicationMode::REFILL;
                        cursor.next_waiter = cursor.saved_next;
                        cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
                        cursor.phase = SchedulerResolvePhase::SELECT;
                        break;
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

            cursor.next_waiter = cursor.saved_next;
            cursor.waiter_ref = SCHEDULER_TASK_REF_INVALID;
            cursor.phase = cursor.publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY ?
                               SchedulerResolvePhase::PUBLISH :
                               SchedulerResolvePhase::SELECT;
            break;
        }
        case SchedulerResolvePhase::PUBLISH: {
            ++used;
            const auto published = scheduler_ready_publication_step(graph, base, local, cursor.publication);
            if (published == SchedulerReadyPublishResult::ERROR)
                return fail(SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED);
            if (published == SchedulerReadyPublishResult::DONE) {
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
