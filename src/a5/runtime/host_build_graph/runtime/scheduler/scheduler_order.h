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
scheduler_order_for(const SchedulerLocalState *local, int64_t task) {
    return local->config.engine_catalog.orders[task];
}

// Normal A5 groups publish one counter credit per returning lane. The simulator's
// GM group credit and the optional protocol trace require an accountant receipt.
static inline __aicore__ bool scheduler_mix_needs_receipt(
    const SchedulerLocalState *local, const SCHEDULER_CLUSTER_SSBUF SchedulerClusterDescriptor *descriptor
) {
    if constexpr (!SCHEDULER_REQUIRES_CNT_NOTIFY) {
        const uint64_t offset = local->config.completion_counters_offset;
        if (offset == 0) return true;
        auto *base = reinterpret_cast<__gm__ void *>(local->config.shared_context->scheduler_state_base_address);
        if (scheduler_completion_counter_at(base, offset, scheduler_cluster_load_field(&descriptor->task_ref)) ==
            nullptr)
            return true;
    }
    return scheduler_protocol_enabled(local);
}

enum class SchedulerMixFin { PENDING, COMPLETE, FAILED };

inline __aicore__ SchedulerMixFin scheduler_mix_fin_status(
    const SCHEDULER_CLUSTER_SSBUF SchedulerClusterDescriptor *descriptor, uint32_t mask, uint32_t generation
) {
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((mask & (1U << lane)) == 0) continue;
        const uint32_t fin = scheduler_cluster_load_field(&descriptor->finished[lane]);
        if (fin != generation) return fin < generation ? SchedulerMixFin::PENDING : SchedulerMixFin::FAILED;
    }
    return SchedulerMixFin::COMPLETE;
}

static inline __aicore__ bool scheduler_cluster_owes_before(const SchedulerLocalState *local, uint32_t rank) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterOwesBefore);
    static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
    if (local->cluster_signal == nullptr) return false;
    const auto *descriptor = &local->cluster_signal->groups[0];
    const uint32_t generation = scheduler_cluster_load_acquire(&descriptor->armed_generation);
    // This lane already published FIN (and its required receipt), or was absent from this generation.
    if ((generation & 1U) == 0 || generation == local->mix_groups[0].settled_generation) return false;
    const uint32_t previous = scheduler_cluster_load_field(&descriptor->rank);
    const bool receipt = scheduler_mix_needs_receipt(local, descriptor);
    const uint32_t bits = scheduler_cluster_load_field(&descriptor->active_mask);
    const LaneMask mask{(bits & 1U) != 0, (bits & 2U) != 0, (bits & 4U) != 0};
    const auto lane = local->config.self_lane;
    if (scheduler_cluster_load_acquire(&descriptor->armed_generation) != generation) return true;
    if (previous >= rank || !mask.has(static_cast<SubtaskSlot>(lane))) return false;
    const auto *ack = receipt && lane == scheduler_mix_accountant_lane(mask) ? &descriptor->accounted_generation :
                                                                               &descriptor->finished[lane];
    return scheduler_cluster_load_acquire(ack) != generation;
}

// F excludes future lower-rank publications. Acquire it before reading the
// descriptor it covers. A published descriptor can still owe this lane work.
static inline __aicore__ bool scheduler_self_order_ready(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t rank,
    uint32_t required, bool *ready
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarSelfOrderReady);
    // The ordinary slot is empty before issue and holds this rank at execution.
    static_assert(SCHEDULER_EXECUTION_TOKEN_COUNT == 1);
    *ready = false;
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
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerClusterMixLocalState &group, bool *ready
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarMixLaneOrderReady, group.task_ref);
    *ready = false;
    if (scheduler_owned_token_blocks_mix_arrival(local, group.rank)) return true;
    const auto required = group.required_floor;
    local->observed_self_floor = 0;
    if (required != 0) {
        const auto &catalog = local->config.engine_catalog;
        if (local->issue_cursor > catalog.count || required > group.rank)
            return scheduler_issue_fail(graph, local, run, group.task_ref);
        uint32_t floor = static_cast<uint32_t>(graph.task_count);
        if (local->issue_cursor != catalog.count) {
            const auto head = catalog.entries[local->issue_cursor].ref;
            if (!scheduler_task_ref_valid(graph, head)) return scheduler_issue_fail(graph, local, run, group.task_ref);
            floor = scheduler_order_for(local, head).rank;
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
    const auto &row = scheduler_order_for(local, ref);
    uint32_t floor = row.rank;
    uint32_t prior = UINT32_MAX;
    if (floor == local->published_mix_floor) {
        if (scheduler_task_has_fixed_owners(scheduler_task_metadata_at(base, local, ref)->flags)) {
            // A fixed group must be published by this proxy before its floor
            // can admit a conflicting self task, even if other clusters advanced.
            const auto &catalog = local->config.engine_catalog;
            if (local->issue_cursor < catalog.count &&
                scheduler_order_for(local, catalog.entries[local->issue_cursor].ref).rank <= row.rank)
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
        floor = next == UINT32_MAX ? static_cast<uint32_t>(graph.task_count) : scheduler_order_for(local, next).rank;
    }
    if (floor <= local->published_mix_floor || floor > graph.task_count)
        return scheduler_issue_fail(graph, local, run, ref);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (prior != UINT32_MAX) local->next_mix_ref = row.next_mix_ref;
    const auto cluster = local->config.shared_context->cluster_index;
    const bool trace = scheduler_protocol_enabled(local);
    if (__builtin_expect(trace, false)) scheduler_protocol_mix_floor(base, local, cluster, prior, floor, true);
    scheduler_cluster_store_release(&local->cluster_signal->mix_floor, floor);
    if (__builtin_expect(trace, false)) scheduler_protocol_mix_floor(base, local, cluster, prior, floor, false);
    local->published_mix_floor = floor;
    *progress = true;
    return true;
}
