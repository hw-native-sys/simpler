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

// One fill invocation owns this snapshot; INVALID is also a cached result.
struct SchedulerIssueSnapshot {
    int64_t ref{SCHEDULER_TASK_REF_INVALID};
    bool evaluated{false};
};

static inline __aicore__ void scheduler_invalidate_issue_snapshot(SchedulerIssueSnapshot *snapshot) {
    if (snapshot != nullptr) snapshot->evaluated = false;
}

static inline __aicore__ bool scheduler_issue_fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t ref,
    SchedulerGraphResult result = SchedulerGraphResult::INVALID_ARGUMENTS,
    SchedulerErrorSite site = SchedulerErrorSite::BLOCK_CURSOR_INVALID
) {
    scheduler_record_error(run, ref, result, &graph, local, site);
    return false;
}

// Only immutable catalog facts persist across fills. Supply and completion
// observations retain their independent freshness and lifetime contracts.
static inline __aicore__ bool scheduler_bind_issue_entry(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref
) {
    auto &binding = local->issue_binding;
    if (binding.task_ref == ref) return true;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    const auto metadata = scheduler_load_dispatch_metadata(base, local, ref, 0);
    if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags) ||
        scheduler_task_execution_engine(metadata.flags, scheduler_task_lanes(metadata)) != local->config.self_core_type)
        return scheduler_issue_fail(graph, local, run, ref);
    binding.metadata = metadata;
    binding.control = scheduler_task_control_at(base, local, ref);
    binding.fanin_bound = false;
    binding.task_ref = ref;
    return true;
}

// The last valid claimant publishes an immutable supply seal. Admission tasks
// publish it only after the required member registrations are complete.
static inline __aicore__ bool scheduler_task_supply_closed(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool *closed, const SchedulerIssueBinding *binding = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskSupplyClosed, ref);
    *closed = false;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    bool executable = true;
    uint64_t blocks = 0;
    auto *control = binding != nullptr ? binding->control : scheduler_task_control_at(base, local, ref);
    if (binding != nullptr) {
        if (binding->task_ref != ref) return scheduler_issue_fail(graph, local, run, ref);
        blocks = binding->metadata.logical_block_num;
    } else {
        const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
        if (scheduler_predecessor_is_host_done(graph, ref, metadata)) {
            *closed = true;
            return true;
        }
        executable = scheduler_task_is_executable(metadata.flags) && !scheduler_task_is_inline(metadata.flags);
        blocks = metadata.logical_block_num;
    }
    if (scheduler_cached_task_proof(local, ref) != SchedulerTaskProof::NONE) {
        *closed = true;
        return true;
    }
    if (executable && blocks == 0) return scheduler_issue_fail(graph, local, run, ref);
    // Canonical DONE proves both closed supply and completed dependencies.
    // Pending compute or async debt still requires the independent supply seal.
    const auto completion = scheduler_observe_cached_completion(base, local, ref);
    if (completion == SchedulerCompletionObservation::INVALID) return scheduler_issue_fail(graph, local, run, ref);
    if (completion == SchedulerCompletionObservation::DONE) {
        *closed = true;
        return true;
    }
    if (executable) {
        const uint64_t seal = scheduler_gm_query(control->claim_seal);
        if (seal != 0 && seal != blocks) return scheduler_issue_fail(graph, local, run, ref);
        if (seal == blocks) {
            scheduler_remember_task_proof(local, ref, SchedulerTaskProof::SUPPLY_CLOSED);
            *closed = true;
            return true;
        }
    }
    return true;
}

// Only a successful final valid ticket owner reaches this publication. Other
// claimants may still be receiving their Calls; every valid ticket is irrevocable.
static inline __aicore__ bool scheduler_close_ordinary_supply(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, uint32_t block, uint32_t blocks
) {
    if (block + 1 != blocks) return true;
    auto *control = scheduler_task_control_at(base, local, ref);
    scheduler_cache_barrier();
    scheduler_store_signal(control->claim_seal, static_cast<uint64_t>(blocks));
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto catalog = scheduler_task_catalog(base, pool, local->config.self_core_type);
    if (!scheduler_task_catalog_valid(catalog)) return scheduler_issue_fail(graph, local, run, ref);
    scheduler_remember_task_proof(local, ref, SchedulerTaskProof::SUPPLY_CLOSED);
    const uint64_t head = local->issue_cursor;
    if (head < catalog.count && catalog.refs[head] == ref) {
        local->issue_cursor = head + 1;
    }
    return true;
}

// Private ownership only: this chooses a recheck boundary, never grants entry.
inline __aicore__ bool scheduler_issue_needs_execution_recheck(const SchedulerLocalState *local) {
    if (local->local_ready_mask != 0 || local->cluster_mix.phase != SchedulerGmMixPhase::IDLE ||
        !scheduler_task_ref_is_null(local->cluster_pending.task_ref) ||
        (local->admission != nullptr && (!scheduler_task_ref_is_null(local->admission->current_task_ref) ||
                                         !scheduler_task_ref_is_null(local->admission->pending_task_ref))))
        return true;
    for (const auto &token : local->execution_tokens)
        if (token.phase != SchedulerExecutionTokenPhase::EMPTY) return true;
    return false;
}

// Both engine catalogs are projections of the same validated topological order.
// Cross-engine predecessors must also have owners before a consumer takes space.
static inline __aicore__ bool scheduler_issue_fanin_ready(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool *ready
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskIssueFanin, ref);
    *ready = false;
    if (local->issue_task_ref != ref) {
        local->issue_task_ref = ref;
        local->issue_next_fanin = 0;
        local->issue_fanin_confirmed = false;
        local->issue_ticket_exhausted = false;
    }
    // Topology is immutable for this run, and a predecessor's issued supply
    // cannot reopen. This certificate grants issuance, never kernel entry.
    if (local->issue_fanin_confirmed) {
        *ready = true;
        return true;
    }
    if (!scheduler_bind_issue_entry(graph, base, local, run, ref)) return false;
    auto &binding = local->issue_binding;
    if (!binding.fanin_bound) {
        const auto view_status = scheduler_node_fanin_view(graph, ref, &binding.fanin);
        if (view_status != SchedulerGraphResult::OK)
            return scheduler_issue_fail(graph, local, run, ref, view_status, SchedulerErrorSite::ROUTE_INVALID_STATE);
        binding.fanin_bound = true;
    }
    const auto &fanin = binding.fanin;
    if (local->issue_next_fanin > fanin.count)
        return scheduler_issue_fail(
            graph, local, run, ref, SchedulerGraphResult::INVALID_FANIN_COUNT, SchedulerErrorSite::ROUTE_INVALID_STATE
        );
    const bool recheck_execution = scheduler_issue_needs_execution_recheck(local);
    while (local->issue_next_fanin < fanin.count) {
        int64_t predecessor = SCHEDULER_TASK_REF_INVALID;
        const auto status = scheduler_fanin_ref(fanin, ref, local->issue_next_fanin, &predecessor);
        if (status != SchedulerGraphResult::OK)
            return scheduler_issue_fail(graph, local, run, ref, status, SchedulerErrorSite::ROUTE_INVALID_STATE);
        const auto &metadata = *scheduler_task_metadata_at(base, local, predecessor);
        // Earlier executable entries in this engine have passed this core's
        // issuance cursor, including the real member seal for sync-start tasks.
        const bool issued_in_engine = scheduler_task_is_executable(metadata.flags) &&
                                      !scheduler_task_is_inline(metadata.flags) &&
                                      scheduler_task_execution_engine(metadata.flags, scheduler_task_lanes(metadata)) ==
                                          local->config.self_core_type;
        bool closed = false;
        if (!issued_in_engine) {
            if (!scheduler_task_supply_closed(graph, base, local, run, predecessor, &closed)) return false;
            if (!closed) return true;
        }
        ++local->issue_next_fanin;
        if (local->issue_next_fanin % 32 == 0 && scheduler_read_worker_error(run, local->worker_id()) != 0)
            return false;
        if (recheck_execution && local->issue_next_fanin < fanin.count) return true;
    }
    local->issue_fanin_confirmed = true;
    *ready = true;
    return true;
}

static inline __aicore__ bool scheduler_query_issue_head(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t *ref, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskIssueHead);
    if (base == nullptr || local == nullptr || run == nullptr || ref == nullptr || progress == nullptr) return false;
    *ref = SCHEDULER_TASK_REF_INVALID;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (local->config.task_catalog_offset == 0)
        return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto catalog = scheduler_task_catalog(base, pool, local->config.self_core_type);
    if (!scheduler_task_catalog_valid(catalog))
        return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    if (catalog.count == 0) return true;
    // The private cursor only crosses atomically confirmed supply seals or
    // canonical terminal states. Every earlier catalog entry is closed.
    uint64_t head = local->issue_cursor;
    if (head > catalog.count) return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    if (head == catalog.count) return true;
    for (uint32_t visited = 0; visited < 8; ++visited) {
        if (head > catalog.count) return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
        if (head == catalog.count) return true;
        const int64_t task_ref = catalog.refs[head];
        if (!scheduler_task_ref_valid(graph, task_ref)) return scheduler_issue_fail(graph, local, run, task_ref);
        if (!scheduler_bind_issue_entry(graph, base, local, run, task_ref)) return false;
        bool closed = false;
        if (!scheduler_task_supply_closed(graph, base, local, run, task_ref, &closed, &local->issue_binding))
            return false;
        if (closed) {
            local->issue_cursor = ++head;
            *progress = true;
            if (scheduler_issue_needs_execution_recheck(local)) return true;
            continue;
        }
        const int64_t previous_task = local->issue_task_ref;
        const uint32_t previous_fanin = local->issue_next_fanin;
        bool ready = false;
        if (!scheduler_issue_fanin_ready(graph, base, local, run, task_ref, &ready)) return false;
        const uint32_t confirmed_before = local->issue_task_ref == previous_task ? previous_fanin : 0;
        *progress = *progress || local->issue_next_fanin > confirmed_before;
        if (ready) *ref = task_ref;
        return true;
    }
    return true;
}

static inline __aicore__ bool scheduler_issue_head(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t *ref, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    if (snapshot != nullptr && snapshot->evaluated) {
        if (base == nullptr || local == nullptr || run == nullptr || ref == nullptr || progress == nullptr)
            return false;
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        *ref = snapshot->ref;
        return true;
    }
    if (!scheduler_query_issue_head(graph, base, local, run, ref, progress)) return false;
    if (snapshot != nullptr) {
        snapshot->ref = *ref;
        snapshot->evaluated = true;
    }
    return true;
}
