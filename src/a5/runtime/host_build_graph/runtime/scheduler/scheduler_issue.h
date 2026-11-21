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
#include "scheduler_protocol_block.h"

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
    binding.last_open_seal_ticks = 0;
    binding.task_ref = ref;
    return true;
}

// Fixed owners are known before dispatch. Dynamic supply closes on the last
// valid claim. Neither certificate grants kernel entry.
static inline __aicore__ bool scheduler_task_supply_closed(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool *closed, const SchedulerIssueBinding *binding = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskSupplyClosed, ref);
    *closed = false;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    bool executable = true;
    uint64_t blocks = 0;
    uint8_t flags = 0;
    auto *control = binding != nullptr ? binding->control : scheduler_task_control_at(base, local, ref);
    if (binding != nullptr) {
        if (binding->task_ref != ref) return scheduler_issue_fail(graph, local, run, ref);
        flags = binding->metadata.flags;
        blocks = binding->metadata.logical_block_num;
    } else {
        const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
        if (scheduler_predecessor_is_host_done(graph, ref, metadata)) {
            *closed = true;
            return true;
        }
        flags = metadata.flags;
        executable = scheduler_task_is_executable(flags) && !scheduler_task_is_inline(flags);
        blocks = metadata.logical_block_num;
    }
    if (scheduler_task_has_fixed_owners(flags) || scheduler_cached_task_proof(local, ref) != SchedulerTaskProof::NONE) {
        *closed = true;
        return true;
    }
    if (executable && blocks == 0) return scheduler_issue_fail(graph, local, run, ref);
    // In flat graphs, the final valid claim seals supply for both ordinary and MIX tasks without predicates.
    const bool always_seals = graph.native_header_offset == 0 && executable && !scheduler_task_has_predicate(flags);
    if (!always_seals) {
        const auto completion = scheduler_observe_cached_completion(base, local, ref);
        if (completion == SchedulerCompletionObservation::INVALID) return scheduler_issue_fail(graph, local, run, ref);
        if (completion == SchedulerCompletionObservation::DONE) {
            *closed = true;
            return true;
        }
    }
    if (executable) {
        // Delay polling an unchanged open seal. The claim operation still decides ownership.
        auto &head = local->issue_binding;
        const bool retry = local->previous_round_idle && head.task_ref == ref;
        constexpr uint64_t kSealRetryTicks = 500;
        if (retry && head.last_open_seal_ticks != 0 && scheduler_cycles() - head.last_open_seal_ticks < kSealRetryTicks)
            return true;
        const uint64_t seal = scheduler_gm_query(control->claim_seal);
        if (seal != 0 && seal != blocks) return scheduler_issue_fail(graph, local, run, ref);
        if (seal == blocks) {
            scheduler_remember_task_proof(local, ref, SchedulerTaskProof::SUPPLY_CLOSED);
            *closed = true;
            return true;
        }
        if (retry) head.last_open_seal_ticks = scheduler_cycles();
    }
    return true;
}

// Each sector contains block indices part + 4*k. Its final valid claimant
// contributes that sector's capacity once; excess claims never roll back or
// contribute. A total of N therefore proves all N blocks have fixed owners.
static inline __aicore__ bool scheduler_claim_ticket(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, uint32_t blocks, uint32_t *block, bool *closes_supply, bool execution_claim = true
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarOrdinaryBlockClaim, ref);
    SchedulerExecutionClaimLoads claim_loads(execution_claim);
    *block = blocks;
    *closes_supply = false;
    auto *control = scheduler_task_control_at(base, local, ref);
    if (blocks == 1) {
        const uint64_t claimed = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
        if (claimed != 0) {
            if (scheduler_gm_fetch_add(control->next_block, UINT64_MAX) <= blocks)
                return scheduler_issue_fail(graph, local, run, ref);
            return true;
        }
        *block = 0;
        *closes_supply = true;
        return true;
    }
    const uint32_t first = static_cast<uint32_t>(local->worker_id()) & (SCHEDULER_CLAIM_SECTOR_COUNT - 1);
    for (uint32_t offset = 0; offset < SCHEDULER_CLAIM_SECTOR_COUNT; ++offset) {
        const uint32_t part = (first + offset) & (SCHEDULER_CLAIM_SECTOR_COUNT - 1);
        const uint32_t capacity =
            blocks / SCHEDULER_CLAIM_SECTOR_COUNT + (part < blocks % SCHEDULER_CLAIM_SECTOR_COUNT ? 1 : 0);
        if (capacity == 0) continue;
        auto &next = control->claim_sectors[part].next;
        if (scheduler_gm_query(next) >= capacity) continue;
        const uint64_t ticket = scheduler_gm_fetch_add(next, UINT64_C(1));
        if (ticket >= capacity) continue;
        *block = part + static_cast<uint32_t>(ticket) * SCHEDULER_CLAIM_SECTOR_COUNT;
        if (ticket + 1 == capacity) {
            const uint64_t covered = scheduler_gm_fetch_add(control->next_block, static_cast<uint64_t>(capacity));
            if (covered > blocks - capacity) return scheduler_issue_fail(graph, local, run, ref);
            *closes_supply = covered + capacity == blocks;
        }
        return true;
    }
    return true;
}

// Only the claimant that completes the final sector reaches this publication.
// Other owners may still receive Calls; their block assignments cannot move.
static inline __aicore__ bool scheduler_close_ordinary_supply(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref, bool closes_supply, uint32_t blocks, uint32_t block
) {
    if (!closes_supply) return true;
    auto *control = scheduler_task_control_at(base, local, ref);
    const bool trace = scheduler_protocol_enabled(local);
    const bool cooperative =
        trace && (scheduler_task_is_mix(scheduler_task_metadata_at(base, local, ref)->flags) ||
                  scheduler_task_requires_sync_start(scheduler_task_metadata_at(base, local, ref)->flags));
    if (__builtin_expect(cooperative, false)) scheduler_protocol_claim_seal(base, local, ref, block, true, blocks);
    scheduler_cache_barrier();
    scheduler_store_signal(control->claim_seal, static_cast<uint64_t>(blocks));
    if (__builtin_expect(cooperative, false)) scheduler_protocol_claim_seal(base, local, ref, block, false, blocks);
    const auto &catalog = local->config.engine_catalog;
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
    if (!scheduler_task_ref_is_null(scheduler_unretired_mix_task(*local))) return true;
    for (const auto &token : local->execution_tokens)
        if (token.phase != SchedulerExecutionTokenPhase::EMPTY) return true;
    return false;
}

// Both engine catalogs are projections of the same validated topological order.
// Cross-engine predecessors must also have owners before a consumer takes space.
static inline __aicore__ bool scheduler_check_predecessor_supply(
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
        const auto view_status = scheduler_profiled_node_fanin_view(graph, ref, &binding.fanin);
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

static inline __aicore__ bool scheduler_find_issuable_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t *ref, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarTaskIssueHead);
    if (base == nullptr || local == nullptr || run == nullptr || ref == nullptr || progress == nullptr) return false;
    *ref = SCHEDULER_TASK_REF_INVALID;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &catalog = local->config.engine_catalog;
    if (!scheduler_task_catalog_valid(catalog))
        return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    // A fixed owner keeps its own entry until it binds the block or publishes
    // the MIX group. Other engines can use its static supply proof immediately.
    uint64_t head = local->issue_cursor;
    if (head > catalog.count) return scheduler_issue_fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    for (uint32_t visited = 0; visited < 8 && head < catalog.count; ++visited) {
        const int64_t task_ref = catalog.refs[head];
        if (!scheduler_task_ref_valid(graph, task_ref)) return scheduler_issue_fail(graph, local, run, task_ref);
        if (!scheduler_bind_issue_entry(graph, base, local, run, task_ref)) return false;
        bool closed = false;
        const auto flags = local->issue_binding.metadata.flags;
        if (scheduler_task_has_fixed_owners(flags)) {
            closed = scheduler_task_is_mix(flags) && !local->is_mix_coordinator();
        } else if (!scheduler_task_supply_closed(graph, base, local, run, task_ref, &closed, &local->issue_binding))
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
        if (!scheduler_check_predecessor_supply(graph, base, local, run, task_ref, &ready)) return false;
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
    if (!scheduler_find_issuable_task(graph, base, local, run, ref, progress)) return false;
    if (snapshot != nullptr) {
        snapshot->ref = *ref;
        snapshot->evaluated = true;
    }
    return true;
}
