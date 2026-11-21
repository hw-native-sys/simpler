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

#include "scheduler_completion.h"
#include "scheduler_execution_tokens.h"
#include "scheduler_cluster_mix.h"

// The returning coordinator can bind its fixed ordinary head or publish the next MIX.
// An existing MIX reservation keeps publication priority.
static inline __aicore__ bool scheduler_continue_coordinator_head(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level
) {
    bool progress = false;
    if (scheduler_task_ref_is_null(local->cluster_pending.task_ref)) {
        const auto &catalog = local->config.engine_catalog;
        if (local->issue_cursor >= catalog.count || scheduler_has_execution_tokens(local)) return true;
        const int64_t ref = catalog.refs[local->issue_cursor];
        if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
        const auto flags = scheduler_task_metadata_at(base, local, ref)->flags;
        if (!scheduler_task_has_fixed_owners(flags)) return true;
        bool ready = false;
        if (!scheduler_check_predecessor_supply(graph, base, local, run, ref, &ready)) return false;
        if (!ready) return true;
        if (scheduler_task_uses_execution_tokens(flags)) {
            SchedulerIssueSnapshot head{ref, true};
            return scheduler_refill_execution_tokens(graph, base, local, run, profiling_level, &progress, &head);
        }
        if (!scheduler_task_is_mix(flags)) return true;
        if (!scheduler_claim_for_build(graph, base, local, run, ref, &progress)) return false;
        if (!progress) return true;
    }
    return scheduler_progress_mix_publication(graph, base, local, run, &progress);
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_check_entry_permit(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t index, bool *permit, bool selection_checked = false
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarCanEnterExecution);
    *permit = false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &owner = local->execution_at(index);
    scope.set_task_ref(owner.task_ref);
    if (owner.cluster_mix) {
        if (!scheduler_cluster_mix_can_enter(graph, local, run, index, permit, selection_checked)) return false;
    } else if (selection_checked) {
        // Selection acquired the applicable MIX floor and discharged earlier obligations on this lane.
        *permit = true;
    } else if (!scheduler_self_order_ready(
                   graph, base, local, run, owner.rank, owner.required_floor, owner.sync_start, permit
               )) {
        return false;
    }
    if constexpr (RecordProtocol) {
        if (*permit && scheduler_protocol_enabled(local)) {
            const uint32_t mix = !owner.cluster_mix && owner.required_floor != 0 ? local->observed_mix_floor : 0;
            const uint32_t self = owner.cluster_mix ? local->observed_self_floor : 0;
            scheduler_protocol_execution_permit(base, local, index, mix, self);
        }
    }
    return true;
}

static inline __aicore__ bool scheduler_progress_mix_groups(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, uint32_t *selected = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarServiceCooperativeSlots);
    if (local->cluster_signal == nullptr) return true;
    const uint32_t previous_generation = local->last_mix_generation;
    const uint32_t pending_rank = local->cluster_pending.rank;
    bool advanced = false;
    if (!scheduler_progress_mix_publication(graph, base, local, run, &advanced)) return false;
    *progress = *progress || advanced;
    if (!scheduler_advance_mix_floor(graph, base, local, run, progress)) return false;
    if (selected != nullptr && previous_generation != local->last_mix_generation) {
        // A delayed publication can follow younger ordinary claims. Reuse the regular selection
        // only if every bound ordinary block precedes the descriptor just released to the peers.
        bool has_earlier = false;
        for (const auto &token : local->execution_tokens) {
            if (token.phase == SchedulerExecutionTokenPhase::EMPTY) continue;
            has_earlier = token.rank < pending_rank;
            if (!has_earlier) break;
        }
        if (has_earlier) {
            if (!scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, selected, &advanced))
                return false;
            *progress = *progress || advanced;
            if (*selected != UINT32_MAX) return true;
        }
    }
    if (!scheduler_cluster_mix_service(graph, base, local, run, profiling_level, &advanced)) return false;
    *progress = *progress || advanced;
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ __attribute__((noinline)) bool scheduler_find_local_execution(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *failed, uint32_t *out = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarFillOwnSlots);
    uint32_t ignored = UINT32_MAX;
    if (out == nullptr) out = &ignored;
    *out = UINT32_MAX;
    bool progress = false;
    *failed = !scheduler_progress_mix_groups(graph, base, local, run, profiling_level, &progress, out);
    if (*failed || *out != UINT32_MAX) return progress;
    *out = scheduler_cooperative_execution(local);
    if (*out != UINT32_MAX) return progress;
    // This generation's fixed dependent cannot execute before our member returns.
    // Its still-unclaimed head also proves the sole ordinary slot was empty.
    const auto &member = local->mix_groups[0];
    if (member.fixed_consumer_waits && member.phase == SchedulerGmMixPhase::PUBLISHED) return progress;
    bool advanced = false;
    *failed = !scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, out, &advanced);
    progress = progress || advanced;
    if (*failed || *out != UINT32_MAX) return progress;

    // Always advance owned members and tickets. Exhausting the catalog only stops new claims.
    const auto &catalog = local->config.engine_catalog;
    if (local->config.prepared != nullptr && local->issue_cursor == catalog.count) return progress;

    // A completed ordinary block proves fanin for another block in the same run.
    // A continuation still has to be the current ordered head and win a real ticket.
    // Give discovery of a new MIX group an opportunity after three successful continuations.
    constexpr uint32_t kMaxConsecutiveContinuations = 3;
    const int64_t previous = local->continuation_task_ref;
    if (!scheduler_task_ref_is_null(previous) && local->continuation_streak < kMaxConsecutiveContinuations &&
        previous == local->issue_task_ref && previous == local->issue_binding.task_ref &&
        local->issue_fanin_confirmed && !local->issue_ticket_exhausted) {
        if (local->issue_cursor < catalog.count && catalog.refs[local->issue_cursor] == previous) {
            SchedulerIssueSnapshot continuation{previous, true};
            *failed = !scheduler_refill_execution_tokens(
                graph, base, local, run, profiling_level, &advanced, &continuation, out
            );
            progress = progress || advanced;
            if (*failed) return progress;
            if (advanced) {
                ++local->continuation_streak;
                return progress;
            }
        }
    }
    local->continuation_streak = 0;
    SchedulerIssueSnapshot head{};
    if (local->is_mix_coordinator() && local->cluster_signal != nullptr) {
        if (!scheduler_cluster_mix_discover_next(graph, base, local, run, &advanced, &head)) {
            *failed = true;
            return progress;
        }
        progress = progress || advanced;
        if (advanced && !scheduler_task_ref_is_null(local->cluster_pending.task_ref)) {
            // Only a new group claim needs immediate service. Advancing the catalog cursor alone publishes no
            // descriptor.
            if (!scheduler_progress_mix_groups(graph, base, local, run, profiling_level, &progress, out)) {
                *failed = true;
                return progress;
            }
            if (*out != UINT32_MAX) return progress;
            *out = scheduler_cooperative_execution(local);
            if (*out != UINT32_MAX) return progress;
        }
    }
    if (!scheduler_refill_execution_tokens(graph, base, local, run, profiling_level, &advanced, &head, out)) {
        *failed = true;
        return progress;
    }
    return progress || advanced;
}

// Only the timeout path needs to identify the group holding up exit.
static inline __aicore__ int64_t scheduler_ordered_drain_task(const SchedulerLocalState *local) {
    const int64_t outstanding = scheduler_unretired_mix_task(*local);
    if (!scheduler_task_ref_is_null(outstanding)) return outstanding;
    if (local->cluster_signal == nullptr) return SCHEDULER_TASK_REF_INVALID;
    for (uint32_t bank = 0; bank < SCHEDULER_MIX_MEMBER_COUNT; ++bank) {
        const auto *group = &local->cluster_signal->groups[bank];
        const uint32_t generation = scheduler_cluster_load_acquire(&group->armed_generation);
        if (!scheduler_cluster_mix_armed(generation)) continue;
        const int64_t task = scheduler_cluster_load_field(&group->task_ref);
        scheduler_cluster_barrier();
        if (scheduler_cluster_load_acquire(&group->armed_generation) == generation) return task;
    }
    if (local->is_mix_coordinator() && local->next_mix_ref != UINT32_MAX) return local->next_mix_ref;
    return SCHEDULER_TASK_REF_INVALID;
}

static inline __aicore__ bool scheduler_drain_ordered_exit(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *done
) {
    *done = false;
    bool retired = false;
    if (!scheduler_cluster_mix_drain(graph, base, local, run, &retired)) return false;
    if (!retired) return true;
    bool progress = false;
    if (!scheduler_advance_mix_floor(graph, base, local, run, &progress)) return false;
    *done = !local->is_mix_coordinator() || local->cluster_signal == nullptr || local->next_mix_ref == UINT32_MAX;
    return true;
}
