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

#include "scheduler_ready.h"

inline constexpr uint32_t SCHEDULER_CONTROL_NODE_BUDGET = 8;
inline constexpr uint32_t SCHEDULER_CONTROL_RESOLVE_BUDGET = 64;

// A detached wake chain belongs to one worker until fully routed. No reference
// is recycled during a run; keeping the remaining head needs no epoch protocol.
inline __aicore__ bool scheduler_service_controls(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    *progress = false;
    if (local->config.control_inboxes_offset == 0) return true;
    if (scheduler_read_error(run) != 0) return false;
    if (scheduler_gm_query(run->bootstrap_scan_complete) == 0) return true;
    uint32_t started = 0, completed = 0;
    uint32_t budget = SCHEDULER_CONTROL_RESOLVE_BUDGET;
    SchedulerReadyBatch batches[SCHEDULER_READY_QUEUE_COUNT]{};
    auto &continuation = local->control_resolve;
    while (budget != 0) {
        if (continuation.producer_ref < 0) {
            if (started == SCHEDULER_CONTROL_NODE_BUDGET) break;
            int64_t ref = SCHEDULER_TASK_REF_INVALID;
            auto *own = scheduler_state_at<SchedulerReadyInbox>(base, local->config.control_inboxes_offset) +
                        local->worker_id();
            if (!scheduler_ready_pop_from_queue(graph, base, local, run, own, &ref, nullptr)) return false;
            if (ref < 0 && local->config.runtime_worker_count > 1) {
                const uint32_t peer = local->control_peer_cursor++ % local->config.runtime_worker_count;
                if (peer != local->worker_id()) {
                    auto *inbox =
                        scheduler_state_at<SchedulerReadyInbox>(base, local->config.control_inboxes_offset) + peer;
                    if (!scheduler_ready_pop_from_queue(graph, base, local, run, inbox, &ref, nullptr)) return false;
                }
            }
            if (ref < 0) break;
            SchedulerNodeKind kind{};
            if (scheduler_node_kind(graph, ref, &kind) != SchedulerGraphResult::OK ||
                !scheduler_node_is_control(kind)) {
                scheduler_record_error(run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                return false;
            }
            const auto *metadata = scheduler_task_metadata_at(base, local, ref);
            if (!scheduler_task_lanes(*metadata).is_empty() || scheduler_task_is_executable(metadata->flags)) {
                scheduler_record_error(run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                return false;
            }
            auto *control = scheduler_task_control_at(base, local, ref);
            scheduler_protocol_task_done(base, local, ref, true);
            if (scheduler_gm_compare_exchange(
                    control->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED),
                    static_cast<int64_t>(SchedulerTaskState::DONE)
                ) != static_cast<int64_t>(SchedulerTaskState::BLOCKED)) {
                scheduler_record_error(
                    run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                    SchedulerErrorSite::ROUTE_INVALID_STATE
                );
                return false;
            }
            scheduler_protocol_task_done(base, local, ref, false);
            const int64_t head = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (head < SCHEDULER_WAKE_LIST_OPEN) {
                scheduler_record_error(
                    run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                    SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
                );
                return false;
            }
            continuation.producer_ref = ref;
            continuation.next_waiter = head;
            ++started;
            *progress = true;
        }
        while (continuation.next_waiter >= 0 && budget != 0) {
            const int64_t waiter = continuation.next_waiter;
            if (continuation.route.task_ref == SCHEDULER_TASK_REF_INVALID) {
                if (!scheduler_task_ref_valid(graph, waiter)) {
                    scheduler_record_error(
                        run, continuation.producer_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, local
                    );
                    return false;
                }
                auto *control = scheduler_task_control_at(base, local, waiter);
                continuation.detached_next = scheduler_observe_next_waiter(control);
                if (continuation.detached_next < SCHEDULER_WAKE_LIST_OPEN) {
                    scheduler_record_error(run, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                    return false;
                }
                continuation.route = scheduler_resolve_begin(waiter);
            }
            const auto route = scheduler_resolve_step(graph, base, local, run, &continuation.route, &budget);
            if (route == SchedulerRouteResult::ERROR) return false;
            *progress = true;
            if (route == SchedulerRouteResult::YIELD) break;
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                SchedulerNodeKind kind{};
                if (scheduler_node_kind(graph, waiter, &kind) != SchedulerGraphResult::OK) {
                    scheduler_record_error(run, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                    return false;
                }
                if (scheduler_node_is_control(kind)) {
                    if (!scheduler_control_enqueue(graph, base, local, run, waiter)) return false;
                } else {
                    auto *metadata = scheduler_task_metadata_at(base, local, waiter);
                    const auto lanes = scheduler_task_lanes(*metadata);
                    const uint32_t queue = scheduler_task_ready_queue(metadata->flags, lanes);
                    if (!scheduler_task_is_executable(metadata->flags) || queue >= SCHEDULER_READY_QUEUE_COUNT ||
                        !scheduler_ready_batch_append(base, local, waiter, &batches[queue], nullptr, profiling_level)) {
                        scheduler_record_error(run, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                        return false;
                    }
                }
            }
            // A successful link may already have been detached by its new
            // producer. Advance using our saved link, never the waiter's fields.
            continuation.next_waiter = continuation.detached_next;
            continuation.route = {};
        }
        if (continuation.next_waiter >= 0) break;
        ++completed;
        continuation = {};
    }
    // Successors become visible before their producer contributes to the exit
    // count. A peer may already have resolved EXIT while this tail was owned here.
    if (scheduler_read_error(run) != 0) return false;
    for (uint32_t queue = 0; queue < SCHEDULER_READY_QUEUE_COUNT; ++queue)
        if (!scheduler_ready_batch_push(base, local, queue, &batches[queue], nullptr)) {
            scheduler_record_error(
                run, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local
            );
            return false;
        }
    local->pending_completed += completed;
    return true;
}
