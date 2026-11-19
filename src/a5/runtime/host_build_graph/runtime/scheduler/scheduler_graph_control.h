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
#include "scheduler_protocol.h"

inline constexpr uint32_t SCHEDULER_CONTROL_NODE_BUDGET = 8;

// Control and inline nodes have no runtime argument construction or kernel.
// A ready node may publish an idempotent notification before the CAS that
// assigns its unique canonical accounting responsibility.
static inline __aicore__ bool scheduler_service_controls(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    (void)profiling_level;
    *progress = false;
    if (graph.task_count == 0) return true;
    const auto *catalog = scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    if (catalog->control_count == 0) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    for (uint32_t scanned = 0; scanned < SCHEDULER_CONTROL_NODE_BUDGET; ++scanned) {
        const int64_t ref = local->control_peer_cursor++ % graph.task_count;
        const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
        const bool inline_task = scheduler_task_is_inline(metadata.flags);
        if (!inline_task) {
            if (graph.native_header_offset == 0) continue;
            SchedulerNodeKind kind{};
            if (scheduler_node_kind(graph, ref, &kind) != SchedulerGraphResult::OK) return false;
            if (!scheduler_node_is_control(kind)) continue;
            if (!scheduler_task_lanes(metadata).is_empty() || scheduler_task_is_executable(metadata.flags)) {
                scheduler_record_error(run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
                return false;
            }
        }
        auto *control = scheduler_task_control_at(base, local, ref);
        if (scheduler_observe_cached_completion(base, local, ref) == SchedulerCompletionObservation::DONE) continue;
        SchedulerFaninView fanin{};
        if (scheduler_profiled_node_fanin_view(graph, ref, &fanin) != SchedulerGraphResult::OK) return false;
        const uint64_t tag = static_cast<uint64_t>(ref) + 1;
        auto &cached = local->control_fanin_tags[static_cast<uint64_t>(ref) % SCHEDULER_TASK_PROOF_CACHE_COUNT];
        const uint32_t cached_prefix = cached >> 32 == tag ? static_cast<uint32_t>(cached) : 0;
        uint32_t next_fanin = cached_prefix;
        bool ready = false;
        if (!scheduler_observe_fanin_prefix(graph, base, local, run, ref, fanin, next_fanin, &ready)) return false;
        // A zero-progress visitor cannot displace another control's positive proof.
        if (next_fanin != cached_prefix) cached = (tag << 32) | next_fanin;
        if (!ready) continue;
        const bool record = scheduler_protocol_enabled(local);
        auto *notification = scheduler_done_notification_at(base, local, ref);
        if (notification != nullptr) {
            // Only INITIAL and DONE are legal control-node states in this run.
            // Validate before any notification can release a successor.
            const uint64_t observed = scheduler_gm_query(control->completion_state);
            if (observed == scheduler_completion_completed_state(1)) continue;
            if (observed != scheduler_completion_initial_state(1)) {
                scheduler_record_error(
                    run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                    SchedulerErrorSite::ROUTE_INVALID_STATE
                );
                return false;
            }
        }
        scheduler_cache_barrier();
        if (notification != nullptr) scheduler_notify_done(*notification);
        // No yield or error-return separates publication from this accounting
        // attempt. A losing CAS means another publisher already owns the count.
        const uint64_t before = record ? scheduler_protocol_cycles(local) : 0;
        const uint64_t previous = scheduler_gm_compare_exchange(
            control->completion_state, scheduler_completion_initial_state(1), scheduler_completion_completed_state(1)
        );
        const uint64_t after = record ? scheduler_protocol_cycles(local) : 0;
        if (previous == scheduler_completion_completed_state(1)) continue;
        if (previous != scheduler_completion_initial_state(1)) {
            scheduler_record_error(
                run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::ROUTE_INVALID_STATE
            );
            return false;
        }
        if (record) {
            const uint32_t flags =
                metadata.flags | (notification != nullptr ? SCHEDULER_PROTOCOL_CONTROL_NOTIFY_ACCOUNTING : 0);
            // These timestamps describe the unique accounting CAS. They do not
            // identify the first of possibly several idempotent notifications.
            scheduler_protocol_task_actor(base, local, ref, flags, true, before);
            scheduler_protocol_task_actor(base, local, ref, flags, false, after);
        }
        scheduler_remember_task_proof(local, ref, SchedulerTaskProof::DONE);
        scheduler_report_task_completed(run);
        *progress = true;
    }
    return true;
}
