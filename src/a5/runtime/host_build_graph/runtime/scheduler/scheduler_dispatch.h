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

// An idle coordinator can give a waiting MIX head its ticket before re-entering the selector.
static inline __aicore__ bool scheduler_continue_coordinator_head(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool returned_mix, uint32_t *selected
) {
    const auto &catalog = local->config.engine_catalog;
    if (local->issue_cursor >= catalog.count || scheduler_has_execution_tokens(local)) return true;
    const bool ordinary_prefix = !returned_mix && local->issue_cursor < catalog.terminal_mix_begin;
    const bool idle_proxy =
        local->mix_groups[0].phase == SchedulerGmMixPhase::IDLE && local->next_mix_ref != UINT32_MAX;
    if (ordinary_prefix && !idle_proxy) return true;
    const auto &entry = catalog.entries[local->issue_cursor];
    const int64_t ref = entry.ref;
    const bool waiting_mix = idle_proxy && ref == local->next_mix_ref;
    if (ordinary_prefix && !waiting_mix) return true;
    if (!scheduler_task_ref_valid(graph, ref)) return scheduler_issue_fail(graph, local, run, ref);
    const auto flags = scheduler_task_metadata_at(base, local, ref)->flags;
    if (!scheduler_task_has_fixed_owners(flags)) {
        if (!waiting_mix) return true;
        bool progress = false;
        return scheduler_cluster_mix_discover_next(graph, base, local, run, &progress);
    }
    if (!scheduler_task_is_mix(flags) || local->issue_cursor < catalog.terminal_mix_begin) {
        if (!scheduler_bind_issue_entry(graph, base, local, run, static_cast<uint32_t>(local->issue_cursor)))
            return false;
        bool ready = false;
        if (!scheduler_check_predecessor_supply(graph, base, local, run, local->issue_binding.supply_input, &ready))
            return false;
        if (!ready) return true;
    }
    bool progress = false;
    if (scheduler_task_uses_execution_tokens(flags)) {
        SchedulerIssueSnapshot head{ref, true};
        static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
        if (local->mix_groups[0].phase != SchedulerGmMixPhase::IDLE) selected = nullptr;
        return scheduler_refill_execution_tokens(graph, base, local, run, profiling_level, &progress, &head, selected);
    }
    if (!scheduler_task_is_mix(flags)) return true;
    return scheduler_claim_for_execute(graph, base, local, run, ref, &progress);
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_validate_selected_entry(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t index
) {
    const auto &owner = local->execution_at(index);
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarCanEnterExecution, owner.task_ref);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (index >= SCHEDULER_CLUSTER_MIX_EXECUTION && !scheduler_validate_mix_entry(graph, local, run)) return false;
    if constexpr (RecordProtocol) {
        if (scheduler_protocol_enabled(local)) {
            const uint32_t mix = !owner.cluster_mix && owner.required_floor != 0 ? local->observed_mix_floor : 0;
            const uint32_t self = owner.cluster_mix ? local->observed_self_floor : 0;
            scheduler_protocol_execution_permit(base, local, index, mix, self);
        }
    }
    return true;
}

static inline __aicore__ bool scheduler_progress_mix_groups(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarServiceCooperativeSlots);
    if (local->cluster_signal == nullptr) return true;
    bool advanced = false;
    if (local->is_mix_coordinator() &&
        ((!SCHEDULER_REQUIRES_CNT_NOTIFY && local->config.completion_counters_offset == 0) ||
         scheduler_protocol_enabled(local))) {
        for (uint32_t bank = 0; bank < SCHEDULER_MIX_MEMBER_COUNT; ++bank) {
            const uint32_t generation =
                scheduler_cluster_load_field(&local->cluster_signal->groups[bank].armed_generation);
            if (scheduler_cluster_mix_armed(generation) &&
                !scheduler_cluster_mix_recycle(graph, local, run, bank, generation, progress))
                return false;
        }
    }
    if (!scheduler_advance_mix_floor(graph, base, local, run, progress)) return false;
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
    *failed = !scheduler_progress_mix_groups(graph, base, local, run, profiling_level, &progress);
    if (*failed) return progress;
    *out = scheduler_cooperative_execution(local);
    if (*out != UINT32_MAX) return progress;
    // A fixed dependent head or an owned slot waiting on this MIX cannot
    // advance until this member contributes. Its fanin and order were serviced above.
    const auto &member = local->mix_groups[0];
    if (member.ordinary_consumer_waits && member.phase == SchedulerGmMixPhase::PUBLISHED) return progress;
    bool advanced = false;
    const auto &catalog = local->config.engine_catalog;
    // An empty ordinary slot has no fanin or order to progress. Normal head discovery
    // checks worker error before issuing work; terminal and drain paths keep the poll here.
    const bool discover_head = local->issue_cursor < catalog.count && local->issue_cursor < catalog.terminal_mix_begin;
    if (!discover_head || scheduler_has_execution_tokens(local)) {
        *failed = !scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, out, &advanced);
        progress = progress || advanced;
        if (*failed || *out != UINT32_MAX) return progress;
    }

    // Always advance owned members and tickets. Exhausting the catalog only stops new claims.
    if (local->config.prepared != nullptr && local->issue_cursor == catalog.count) return progress;

    // A completed ordinary block proves fanin for another block in the same run.
    // A continuation still has to be the current ordered head and win a real ticket.
    // Give discovery of a new MIX group an opportunity after three successful continuations.
    constexpr uint32_t kMaxConsecutiveContinuations = 3;
    const int64_t previous = local->continuation_task_ref;
    if (!scheduler_task_ref_is_null(previous) && local->continuation_streak < kMaxConsecutiveContinuations &&
        previous == local->issue_binding.task_ref && local->issue_cursor == local->issue_binding.catalog_index &&
        local->issue_binding.fanin_confirmed && !local->issue_binding.ticket_exhausted) {
        SchedulerIssueSnapshot continuation{previous, true};
        *failed =
            !scheduler_refill_execution_tokens(graph, base, local, run, profiling_level, &advanced, &continuation, out);
        progress = progress || advanced;
        if (*failed) return progress;
        if (advanced) {
            ++local->continuation_streak;
            return progress;
        }
    }
    local->continuation_streak = 0;
    SchedulerIssueSnapshot head{};
    const bool terminal = local->issue_cursor >= catalog.terminal_mix_begin && local->issue_cursor < catalog.count;
    if (terminal && !local->is_mix_coordinator()) {
        local->issue_cursor = catalog.count;
        return true;
    }
    if (local->is_mix_coordinator() && local->cluster_signal != nullptr) {
        bool published = false;
        if (terminal) {
            *failed = !scheduler_claim_for_execute(
                graph, base, local, run, catalog.entries[local->issue_cursor].ref, &published
            );
            progress = progress || published;
        } else {
            *failed = !scheduler_cluster_mix_discover_next(graph, base, local, run, &advanced, &head, &published);
            progress = progress || advanced;
        }
        if (*failed) return progress;
        if (published) {
            // Peers can receive the new descriptor immediately. This coordinator's owned ordinary
            // blocks precede the published catalog head and may have become runnable during publication.
            if (scheduler_has_execution_tokens(local)) {
                *failed =
                    !scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, out, &advanced);
                progress = progress || advanced;
                if (*failed || *out != UINT32_MAX) return progress;
            }
            if (!scheduler_progress_mix_groups(graph, base, local, run, profiling_level, &progress)) {
                *failed = true;
                return progress;
            }
            *out = scheduler_cooperative_execution(local);
            if (*out != UINT32_MAX) return progress;
        }
    }
    if (terminal) return progress;
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
