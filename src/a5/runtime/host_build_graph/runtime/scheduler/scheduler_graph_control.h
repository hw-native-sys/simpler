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

#include "scheduler_completion_resolver.h"

inline constexpr uint32_t SCHEDULER_CONTROL_NODE_BUDGET = 8;
inline constexpr uint32_t SCHEDULER_CONTROL_WORK_BUDGET = 64;

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
    uint32_t started = 0, work = 0, completed = 0;
    auto &continuation = local->control_resolve;
    while (work < SCHEDULER_CONTROL_WORK_BUDGET) {
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
            continuation.producer_ref = ref;
            ++work;
            ++started;
            *progress = true;
        }
        if (work == SCHEDULER_CONTROL_WORK_BUDGET) break;
        const auto outcome = scheduler_resolve_completion_step(
            graph, base, local, run, continuation.producer_ref, continuation, SCHEDULER_CONTROL_WORK_BUDGET - work,
            nullptr, nullptr, nullptr, profiling_level
        );
        work += outcome.work_units;
        *progress = *progress || outcome.work_units != 0;
        if (outcome.result == SchedulerResolveResult::ERROR) return false;
        if (outcome.result == SchedulerResolveResult::YIELD) break;
        ++completed;
        continuation = {};
    }
    // A drained cursor has published every successor before its producer
    // contributes to the run's completion count.
    local->pending_completed += completed;
    return true;
}
