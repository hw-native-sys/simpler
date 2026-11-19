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

#include "scheduler_admission.h"
#include "scheduler_issue.h"

inline __aicore__ const __gm__ SchedulerTaskCatalogOrder &
scheduler_order_for(__gm__ void *base, const SchedulerLocalState *local, int64_t task) {
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    return scheduler_task_catalog_order(base, catalog, static_cast<uint32_t>(task));
}

inline __aicore__ bool scheduler_cluster_owes_before(
    const SchedulerLocalState *local, uint32_t rank, uint32_t current_bank = SCHEDULER_MIX_MEMBER_COUNT
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterOwesBefore);
    const auto *signal = local->cluster_signal;
    if (signal == nullptr) return false;
    for (uint32_t bank = 0; bank < SCHEDULER_MIX_MEMBER_COUNT; ++bank) {
        if (bank == current_bank) continue;
        const auto *descriptor = &signal->groups[bank];
        const uint32_t generation = scheduler_cluster_load_acquire(&descriptor->armed_generation);
        if ((generation & 1U) == 0) continue;
        const uint32_t previous = scheduler_cluster_load_field(&descriptor->rank);
        bool owes = previous < rank;
        const auto *ack = &descriptor->accounted_generation;
        if (current_bank == SCHEDULER_MIX_MEMBER_COUNT) {
            const uint32_t bits = scheduler_cluster_load_field(&descriptor->active_mask);
            const LaneMask mask{(bits & 1U) != 0, (bits & 2U) != 0, (bits & 4U) != 0};
            const auto lane = local->config.self_lane;
            owes = owes && mask.has(static_cast<SubtaskSlot>(lane));
            if (lane != scheduler_mix_accountant_lane(mask)) ack = &descriptor->finished[lane];
        }
        if (scheduler_cluster_load_acquire(&descriptor->armed_generation) != generation) return true;
        if (owes && scheduler_cluster_load_acquire(ack) != generation) return true;
    }
    return false;
}

inline __aicore__ bool scheduler_owns_earlier_sync(const SchedulerLocalState *local, uint32_t rank) {
    for (const auto &owner : local->execution_tokens)
        if (owner.sync_start && owner.rank < rank && owner.state != SchedulerDispatchSlotState::FREE) return true;
    return false;
}

// F excludes future lower-rank publications. Acquire it before reading the
// descriptor it covers. A published descriptor can still owe this lane work.
static inline __aicore__ bool scheduler_self_order_ready(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t rank, uint32_t required, bool sync, bool *ready
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarSelfOrderReady);
    *ready = false;
    if (sync && scheduler_owns_earlier_sync(local, rank)) return true;
    for (const auto &group : local->mix_groups)
        if (group.phase == SchedulerGmMixPhase::ARRIVED || group.phase == SchedulerGmMixPhase::RUNNING) return true;
    if (required != 0) {
        if (local->cluster_signal == nullptr || required > rank)
            return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
        if (local->observed_mix_floor < required) {
            local->observed_mix_floor = scheduler_cluster_load_acquire(&local->cluster_signal->mix_floor);
            if (local->observed_mix_floor < required) return true;
        }
        if (scheduler_cluster_owes_before(local, rank)) return true;
    }
    *ready = true;
    return true;
}

// An earlier owned block can still be needed by a peer even when the catalog
// requires no sync floor on this lane.
inline __aicore__ bool scheduler_owned_token_blocks_mix_arrival(const SchedulerLocalState *local, uint32_t rank) {
    for (const auto &token : local->execution_tokens)
        if (token.state == SchedulerDispatchSlotState::EXECUTING ||
            (token.state != SchedulerDispatchSlotState::FREE && token.rank < rank))
            return true;
    return false;
}

// The private catalog cursor cannot cross unissued work or an unresolved
// Graph control predecessor. Already assigned pure cohorts remain in W.
static inline __aicore__ bool scheduler_mix_lane_order_ready(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool *ready
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarMixLaneOrderReady, ref);
    *ready = false;
    const auto &row = scheduler_order_for(base, local, ref);
    if (scheduler_owned_token_blocks_mix_arrival(local, row.rank)) return true;
    const auto required = row.required_floor[local->config.self_lane];
    local->observed_self_floor = 0;
    if (required != 0) {
        const auto &catalog = local->config.engine_catalog;
        if (local->issue_cursor > catalog.count || required > row.rank)
            return scheduler_issue_fail(graph, local, run, ref);
        uint32_t floor = static_cast<uint32_t>(graph.task_count);
        if (local->issue_cursor != catalog.count) {
            const auto head = catalog.refs[local->issue_cursor];
            if (!scheduler_task_ref_valid(graph, head)) return scheduler_issue_fail(graph, local, run, ref);
            floor = scheduler_order_for(base, local, head).rank;
        }
        if (floor < required) return true;
        local->observed_self_floor = floor;
    }
    *ready = true;
    return true;
}

static inline __aicore__ bool scheduler_advance_mix_floor(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarAdvanceMixFloor);
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr || local->next_mix_ref == UINT32_MAX)
        return true;
    const auto ref = local->next_mix_ref;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    const auto &row = scheduler_order_for(base, local, ref);
    uint32_t floor = row.rank;
    uint32_t prior = UINT32_MAX;
    if (floor == local->published_mix_floor) {
        // A global seal includes our own not-yet-published reservation.
        if (local->cluster_pending.task_ref == ref) return true;
        if (scheduler_task_has_fixed_owners(scheduler_task_metadata_at(base, local, ref)->flags)) {
            // A fixed group must be published by this proxy before its floor
            // can admit a conflicting self task, even if other clusters advanced.
            const auto &catalog = local->config.engine_catalog;
            if (local->issue_cursor < catalog.count &&
                scheduler_order_for(base, local, catalog.refs[local->issue_cursor]).rank <= row.rank)
                return true;
        } else if (local->last_sync_mix_ref != ref) {
            bool closed = false;
            if (!scheduler_task_supply_closed(graph, base, local, run, ref, &closed)) return false;
            if (!closed) return true;
        }
        prior = ref;
        const uint32_t next = row.next_mix_ref;
        if (next != UINT32_MAX && !scheduler_task_ref_valid(graph, next))
            return scheduler_issue_fail(graph, local, run, ref);
        floor =
            next == UINT32_MAX ? static_cast<uint32_t>(graph.task_count) : scheduler_order_for(base, local, next).rank;
    }
    if (floor <= local->published_mix_floor || floor > graph.task_count)
        return scheduler_issue_fail(graph, local, run, ref);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (prior != UINT32_MAX) local->next_mix_ref = row.next_mix_ref;
    const auto cluster = local->config.shared_context->cluster_index;
    const bool trace = scheduler_protocol_enabled(local);
    if (trace) scheduler_protocol_mix_floor(base, local, cluster, prior, floor, true);
    scheduler_cluster_store_release(&local->cluster_signal->mix_floor, floor);
    if (trace) scheduler_protocol_mix_floor(base, local, cluster, prior, floor, false);
    local->published_mix_floor = floor;
    *progress = true;
    return true;
}
