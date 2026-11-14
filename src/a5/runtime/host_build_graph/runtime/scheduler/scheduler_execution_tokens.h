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

inline __aicore__ bool scheduler_execution_token_fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task_ref
) {
    scheduler_record_error(
        run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
        SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
    );
    return false;
}

static inline __aicore__ bool scheduler_execution_token_receive(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerExecutionToken &token, uint64_t profiling_level
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionTokenReceive, token.task_ref);
    if (token.phase != SchedulerExecutionTokenPhase::TICKETED)
        return token.phase == SchedulerExecutionTokenPhase::WAIT_FANIN;
    token.phase = SchedulerExecutionTokenPhase::ACQUIRING;
    const auto result = scheduler_acquire_execution_token(
        base, local->config.prepared, local->worker_id(), token.task_ref, token.block_idx, token.subtask_slot,
        &token.prepared
    );
    if (result != PreparedAccessResult::READY) return scheduler_execution_token_fail(graph, local, run, token.task_ref);
    token.acquired_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    token.phase = SchedulerExecutionTokenPhase::WAIT_FANIN;
    return true;
}

static inline __aicore__ bool scheduler_execution_token_observe_fanin(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerExecutionToken &token, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarExecutionTokenFanin, token.task_ref);
    const uint32_t previous = token.next_fanin;
    bool ready = false;
    if (!scheduler_observe_fanin_prefix(graph, base, local, run, token.task_ref, token.fanin, token.next_fanin, &ready))
        return false;
    *progress = *progress || token.next_fanin != previous;
    return true;
}

static inline __aicore__ bool scheduler_progress_execution_tokens(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, uint32_t *direct_slot, bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionTokenProgress);
    if (local == nullptr || direct_slot == nullptr || progress == nullptr) return false;
    *progress = false;
    *direct_slot = UINT32_MAX;
    if (local->config.prepared == nullptr || local->local_ready_mask != 0) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    for (uint32_t checked = 0; checked < SCHEDULER_EXECUTION_TOKEN_COUNT; ++checked) {
        auto &token = local->execution_tokens[local->execution_check_cursor++ % SCHEDULER_EXECUTION_TOKEN_COUNT];
        if (token.phase == SchedulerExecutionTokenPhase::EMPTY) continue;
        if (token.phase == SchedulerExecutionTokenPhase::TICKETED) {
            if (!scheduler_execution_token_receive(graph, base, local, run, token, profiling_level)) return false;
            *progress = true;
        }
        if (token.phase != SchedulerExecutionTokenPhase::WAIT_FANIN || token.prepared.claimed_state == 0)
            return scheduler_execution_token_fail(graph, local, run, token.task_ref);
        if (!scheduler_execution_token_observe_fanin(graph, base, local, run, token, progress)) return false;
        if (token.next_fanin != token.fanin.count) continue;
        // Local obligations can reject preparation, but never grant entry past
        // the final shared admission and MIX checks.
        const auto *admission = local->admission;
        if (admission != nullptr && (!scheduler_task_ref_is_null(admission->pending_task_ref) ||
                                     !scheduler_task_ref_is_null(admission->current_task_ref)))
            return true;
        if (scheduler_cluster_names_lane(local->cluster_signal, local->config.self_lane)) return true;
        uint32_t index = 0;
        while (index < SCHEDULER_PENDING_SLOT_COUNT && local->slots[index].state != SchedulerDispatchSlotState::FREE)
            ++index;
        if (index == SCHEDULER_PENDING_SLOT_COUNT) return true;
        const auto metadata = scheduler_load_dispatch_metadata(base, local, token.task_ref, profiling_level);
        if (!scheduler_task_uses_execution_tokens(metadata.flags))
            return scheduler_execution_token_fail(graph, local, run, token.task_ref);
        SchedulerScalarScope transfer(local, SchedPhaseKind::ScalarExecutionTokenDispatch, token.task_ref);
        SchedulerPredicateState predicate = token.predicate;
        if (predicate == SchedulerPredicateState::UNEVALUATED && scheduler_task_has_predicate(metadata.flags)) {
            const auto result = scheduler_evaluate_task_predicate(graph, token.task_ref);
            if (result == SchedulerPredicateResult::MALFORMED)
                return scheduler_execution_token_fail(graph, local, run, token.task_ref);
            predicate = result == SchedulerPredicateResult::PASS ? SchedulerPredicateState::PASS :
                                                                   SchedulerPredicateState::FAIL;
        }
        if (predicate == SchedulerPredicateState::UNEVALUATED) predicate = SchedulerPredicateState::PASS;
        auto &slot = local->slots[index];
        slot.prepared = token.prepared;
        slot.task_ref = token.task_ref;
        slot.subtask_slot = token.subtask_slot;
        slot.block_idx = token.block_idx;
        slot.predicate = predicate;
        slot.execution_token = true;
        slot.state = SchedulerDispatchSlotState::FILLING;
        auto *binding = scheduler_execution_binding_at(base, local->config.prepared, slot.prepared.execution_index);
        binding->global_context.sub_block_id = token.subtask_slot == 2 ? 1 : 0;
        SchedulerReadyClaim ready{};
        ready.task_ref = token.task_ref;
        ready.state_probe_start_cycles = token.discovery_cycles;
        ready.state_probe_end_cycles = token.acquired_cycles;
        const uint64_t dispatch_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        token = {};
        SchedulerDirectSlotScope direct_scope(local, direct_slot);
        scheduler_stage_dispatch_slot(base, local, {index}, ready, metadata, slot.subtask_slot);
        scheduler_cache_barrier();
        scheduler_commit_dispatch_slot(
            base, local, {index}, ready, metadata, profiling_level, slot.subtask_slot, dispatch_start
        );
        *progress = true;
        return true;
    }
    return true;
}

// A successful return proves immutable fanin. The hint owns no ticket and
// can only claim while its task remains the shared issuance head.
static inline __aicore__ bool scheduler_try_execution_continuation(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    *progress = false;
    const int64_t task_ref = local->continuation_task_ref;
    local->continuation_task_ref = SCHEDULER_TASK_REF_INVALID;
    if (scheduler_task_ref_is_null(task_ref)) return true;
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionContinuation, task_ref);
    int64_t head = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &head, progress, snapshot)) return false;
    if (head != task_ref) return true;
    if (scheduler_execution_token_count(local) >= SCHEDULER_EXECUTION_TOKEN_COUNT ||
        scheduler_execution_token_has_task(local, task_ref))
        return true;
    if (!scheduler_task_ref_valid(graph, task_ref)) return scheduler_execution_token_fail(graph, local, run, task_ref);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    bool permit = !scheduler_cluster_names_lane(local->cluster_signal, local->config.self_lane);
    if (permit && local->admission != nullptr &&
        !scheduler_can_enter_ordinary(graph, base, local, run, local->admission, &permit))
        return false;
    if (!permit) return true;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const uint32_t blocks = metadata.logical_block_num;
    const uint8_t lane = scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata));
    if (!scheduler_task_uses_execution_tokens(metadata.flags) || blocks == 0 || lane >= 3 ||
        scheduler_metadata_core_type_index(lane) != local->config.self_core_type)
        return scheduler_execution_token_fail(graph, local, run, task_ref);
    auto *control = scheduler_task_control_at(base, local, task_ref);
    // A stale negative hint only defers this attempt to the authoritative scan.
    if (scheduler_gm_query(control->next_block) >= blocks) return true;
    SchedulerFaninView fanin{};
    if (scheduler_node_fanin_view(graph, task_ref, &fanin) != SchedulerGraphResult::OK)
        return scheduler_execution_token_fail(graph, local, run, task_ref);
    uint32_t empty = 0;
    while (empty < SCHEDULER_EXECUTION_TOKEN_COUNT &&
           local->execution_tokens[empty].phase != SchedulerExecutionTokenPhase::EMPTY)
        ++empty;
    if (empty == SCHEDULER_EXECUTION_TOKEN_COUNT) return scheduler_execution_token_fail(graph, local, run, task_ref);
    const uint64_t discovery_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    const uint64_t block = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
    scheduler_invalidate_issue_snapshot(snapshot);
    if (block >= blocks) {
        if (scheduler_gm_fetch_add(control->next_block, UINT64_MAX) <= blocks)
            return scheduler_execution_token_fail(graph, local, run, task_ref);
        return true;
    }
    auto &token = local->execution_tokens[empty];
    token.task_ref = task_ref;
    token.block_idx = static_cast<uint32_t>(block);
    token.subtask_slot = lane;
    token.fanin = fanin;
    token.next_fanin = fanin.count;
    token.discovery_cycles = discovery_start;
    token.phase = SchedulerExecutionTokenPhase::TICKETED;
    if (!scheduler_execution_token_receive(graph, base, local, run, token, profiling_level)) return false;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_refill_execution_tokens(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionTokenRefill);
    if (base == nullptr || local == nullptr || run == nullptr || progress == nullptr) return false;
    *progress = false;
    if (local->config.prepared == nullptr) return true;
    const uint64_t discovery_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    uint32_t count = scheduler_execution_token_count(local);
    if (count > SCHEDULER_EXECUTION_TOKEN_COUNT) return scheduler_execution_token_fail(graph, local, run, -1);
    for (uint32_t claimed = 0; claimed < SCHEDULER_EXECUTION_TOKEN_COUNT && count < SCHEDULER_EXECUTION_TOKEN_COUNT;
         ++claimed) {
        int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
        if (!scheduler_issue_head(graph, base, local, run, &task_ref, progress, snapshot)) return false;
        if (scheduler_task_ref_is_null(task_ref)) return true;
        const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
        if (!scheduler_task_uses_execution_tokens(metadata.flags) ||
            scheduler_execution_token_has_task(local, task_ref))
            return true;
        const uint32_t blocks = metadata.logical_block_num;
        const uint8_t lane = scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata));
        if (blocks == 0 || lane >= PLATFORM_CORES_PER_BLOCKDIM ||
            scheduler_metadata_core_type_index(lane) != local->config.self_core_type)
            return scheduler_execution_token_fail(graph, local, run, task_ref);
        SchedulerFaninView fanin{};
        if (scheduler_node_fanin_view(graph, task_ref, &fanin) != SchedulerGraphResult::OK)
            return scheduler_execution_token_fail(graph, local, run, task_ref);
        uint32_t empty = 0;
        while (empty < SCHEDULER_EXECUTION_TOKEN_COUNT &&
               local->execution_tokens[empty].phase != SchedulerExecutionTokenPhase::EMPTY)
            ++empty;
        if (empty == SCHEDULER_EXECUTION_TOKEN_COUNT)
            return scheduler_execution_token_fail(graph, local, run, task_ref);
        auto *control = scheduler_task_control_at(base, local, task_ref);
        const uint64_t block = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
        scheduler_invalidate_issue_snapshot(snapshot);
        if (block >= blocks) {
            if (scheduler_gm_fetch_add(control->next_block, UINT64_MAX) <= blocks)
                return scheduler_execution_token_fail(graph, local, run, task_ref);
            continue;
        }
        auto &token = local->execution_tokens[empty];
        token.task_ref = task_ref;
        token.block_idx = static_cast<uint32_t>(block);
        token.subtask_slot = lane;
        token.discovery_cycles = discovery_start;
        token.fanin = fanin;
        token.next_fanin = 0;
        token.phase = SchedulerExecutionTokenPhase::TICKETED;
        if (!scheduler_execution_token_receive(graph, base, local, run, token, profiling_level)) return false;
        *progress = true;
        ++count;
        if (!scheduler_execution_token_observe_fanin(graph, base, local, run, token, progress)) return false;
        if (token.next_fanin == token.fanin.count) return true;
    }
    return true;
}
