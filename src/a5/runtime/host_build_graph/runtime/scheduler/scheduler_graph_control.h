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
// Their fanin proof precedes the unique canonical completion CAS.
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
    if (scheduler_gm_query(run->bootstrap_scan_complete) == 0) return true;
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
        if (scheduler_gm_query(control->completion_state) == scheduler_completion_completed_state(1)) continue;
        SchedulerFaninView fanin{};
        if (scheduler_node_fanin_view(graph, ref, &fanin) != SchedulerGraphResult::OK) return false;
        bool ready = false;
        uint32_t next_fanin = 0;
        if (!scheduler_observe_fanin_prefix(graph, base, local, run, ref, fanin, next_fanin, &ready)) return false;
        if (!ready) continue;
        const bool record = scheduler_protocol_enabled(local);
        scheduler_cache_barrier();
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
            scheduler_protocol_task_actor(base, local, ref, UINT64_MAX, metadata.flags, true, true, before);
            scheduler_protocol_task_actor(base, local, ref, UINT64_MAX, metadata.flags, true, false, after);
        }
        ++local->pending_completed;
        *progress = true;
    }
    return true;
}
