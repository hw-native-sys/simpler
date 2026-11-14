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

// A valid ticket cannot be returned. Reaching the block count therefore proves
// every block has an owner, independently of its computation or async debt.
static inline __aicore__ bool scheduler_task_supply_closed(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool *closed
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskSupplyClosed, ref);
    *closed = false;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
    if (scheduler_predecessor_is_host_done(graph, ref, metadata)) {
        *closed = true;
        return true;
    }
    auto *control = scheduler_task_control_at(base, local, ref);
    if (scheduler_task_is_executable(metadata.flags) && !scheduler_task_is_inline(metadata.flags)) {
        const uint64_t blocks = metadata.logical_block_num;
        if (blocks == 0) return scheduler_issue_fail(graph, local, run, ref);
        if (scheduler_gm_query(control->next_block) >= blocks &&
            scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) >= blocks) {
            if (!scheduler_task_requires_admission(metadata.flags)) {
                *closed = true;
                return true;
            }
            const uint64_t seal = scheduler_gm_query(control->claim_seal) == 0 ?
                                      0 :
                                      scheduler_gm_fetch_add(control->claim_seal, UINT64_C(0));
            if (seal != 0 && seal != blocks) return scheduler_issue_fail(graph, local, run, ref);
            if (seal == blocks) {
                *closed = true;
                return true;
            }
        }
    }
    const auto completion = scheduler_observe_task_completion(control);
    if (completion == SchedulerCompletionObservation::INVALID) return scheduler_issue_fail(graph, local, run, ref);
    *closed = completion == SchedulerCompletionObservation::DONE;
    return true;
}

// Both engine catalogs are projections of the same validated topological order.
// Cross-engine predecessors must also have owners before a consumer takes space.
static inline __aicore__ bool scheduler_issue_fanin_ready(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, const __gm__ SchedulerTaskCatalogHeader &pool, bool *ready
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskIssueFanin, ref);
    *ready = false;
    if (local->issue_task_ref != ref) {
        local->issue_task_ref = ref;
        local->issue_next_fanin = 0;
    }
    SchedulerFaninView fanin{};
    const auto view_status = scheduler_node_fanin_view(graph, ref, &fanin);
    if (view_status != SchedulerGraphResult::OK)
        return scheduler_issue_fail(graph, local, run, ref, view_status, SchedulerErrorSite::ROUTE_INVALID_STATE);
    SchedulerOrderingFanin ordering{};
    if (!scheduler_task_ordering_fanin(base, pool, ref, &ordering))
        return scheduler_issue_fail(
            graph, local, run, ref, SchedulerGraphResult::INVALID_FANIN_COUNT, SchedulerErrorSite::ROUTE_INVALID_STATE
        );
    const uint64_t total = static_cast<uint64_t>(fanin.count) + ordering.count;
    if (total > UINT32_MAX || local->issue_next_fanin > total)
        return scheduler_issue_fail(
            graph, local, run, ref, SchedulerGraphResult::INVALID_FANIN_COUNT, SchedulerErrorSite::ROUTE_INVALID_STATE
        );
    while (local->issue_next_fanin < total) {
        int64_t predecessor = SCHEDULER_TASK_REF_INVALID;
        if (local->issue_next_fanin < fanin.count) {
            const auto status = scheduler_fanin_ref(fanin, ref, local->issue_next_fanin, &predecessor);
            if (status != SchedulerGraphResult::OK)
                return scheduler_issue_fail(graph, local, run, ref, status, SchedulerErrorSite::ROUTE_INVALID_STATE);
        } else {
            predecessor = ordering.refs[local->issue_next_fanin - fanin.count];
            if (!scheduler_task_ref_valid(graph, predecessor) || predecessor == ref)
                return scheduler_issue_fail(
                    graph, local, run, ref, SchedulerGraphResult::INVALID_FANIN_ID,
                    SchedulerErrorSite::ROUTE_INVALID_STATE
                );
        }
        const auto &metadata = *scheduler_task_metadata_at(base, local, predecessor);
        // Earlier executable entries in this engine have passed the shared
        // issuance head, including the real member seal for sync-start tasks.
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
    }
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
    // Zero is the Host-published initial head. Later cached values come only
    // from returned atomics; bypass observations cannot move this certificate.
    uint64_t head = local->issue_observed_head;
    if (head > catalog.count) return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    if (head == catalog.count) return true;
    const uint64_t hint = scheduler_gm_query(catalog.issue->next_task);
    if (hint > head || ++local->issue_observe_attempts >= 8) {
        head = scheduler_gm_fetch_add(catalog.issue->next_task, UINT64_C(0));
        local->issue_observed_head = head;
        local->issue_observe_attempts = 0;
    }
    for (uint32_t visited = 0; visited < 8; ++visited) {
        if (head > catalog.count) return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
        if (head == catalog.count) return true;
        const int64_t task_ref = catalog.refs[head];
        if (!scheduler_task_ref_valid(graph, task_ref)) return scheduler_issue_fail(graph, local, run, task_ref);
        const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
        if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags) ||
            scheduler_task_execution_engine(metadata.flags, scheduler_task_lanes(metadata)) !=
                local->config.self_core_type)
            return scheduler_issue_fail(graph, local, run, task_ref);
        bool closed = false;
        if (!scheduler_task_supply_closed(graph, base, local, run, task_ref, &closed)) return false;
        if (closed) {
            const uint64_t observed = scheduler_gm_compare_exchange(catalog.issue->next_task, head, head + 1);
            if (observed == head) {
                ++head;
                *progress = true;
            } else head = observed;
            local->issue_observed_head = head;
            local->issue_observe_attempts = 0;
            continue;
        }
        bool ready = false;
        if (!scheduler_issue_fanin_ready(graph, base, local, run, task_ref, pool, &ready)) return false;
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
