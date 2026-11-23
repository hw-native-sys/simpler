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

#include "scheduler_cluster_mix.h"
#include "scheduler_issue.h"

inline __aicore__ bool scheduler_execution_token_fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task_ref,
    const SchedulerLocalSlotState *owner = nullptr
) {
    scheduler_record_error(
        run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
        SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED, owner
    );
    return false;
}

static inline __aicore__ bool scheduler_bind_ticketed_execution(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerExecutionToken &token, uint64_t profiling_level
) {
    if (token.state != SchedulerDispatchSlotState::TICKETED) return token.state != SchedulerDispatchSlotState::FREE;
    if (!scheduler_task_ref_valid(graph, token.task_ref) || !scheduler_task_uses_execution_tokens(token.metadata.flags))
        return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    const auto result = scheduler_claim_for_execution(
        base, local->config.prepared, local->worker_id(), token.task_ref, token.block_idx, token.subtask_slot,
        &token.prepared, SchedulerResidentGraph<>::supports_native ? nullptr : &token.metadata
    );
    if (!result) return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    const bool supplied = scheduler_close_ordinary_supply(
        graph, base, local, run, token.task_ref, token.closes_supply, token.metadata.logical_block_num, token.block_idx,
        !token.sync_start
    );
    // Cancellation of a claimed handle inspects its Call, so payload receipt
    // completes even when supply publication reports an error.
    const auto received = scheduler_receive_payload(
        base, local->config.prepared, token.task_ref, token.block_idx, token.subtask_slot, &token.prepared
    );
    if (!supplied) return false;
    if (!received) return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    if (token.sync_start && scheduler_resident_graph_is_native(graph)) {
        const auto error = scheduler_observe_graph_arguments(graph, token.task_ref);
        if (error != SchedulerGraphBindingError::NONE) {
            scheduler_record_error(
                run, token.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                scheduler_graph_binding_error_site(error), &token
            );
            return false;
        }
    }
    token.acquired_cycles = scheduler_schedule_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    token.state = SchedulerDispatchSlotState::FILLING;
    return true;
}

static inline __aicore__ bool scheduler_advance_owned_execution(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, uint32_t *selected, bool *progress
) {
    auto &token = local->execution_tokens[0];
    if (token.next_fanin != token.fanin.count) {
        SchedulerScalarScope scope(local, SchedPhaseKind::ScalarExecutionTokenFanin, token.task_ref);
        const uint32_t previous = token.next_fanin;
        static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
        auto &member = local->mix_groups[0];
        const bool owed =
            member.state == SchedulerDispatchSlotState::FILLING || member.state == SchedulerDispatchSlotState::READY;
        const int64_t owed_mix = owed ? member.task_ref : SCHEDULER_TASK_REF_INVALID;
        if (!scheduler_observe_fanin_prefix(
                graph, base, local, run, token.task_ref, token.fanin, token.next_fanin, &token.wait, owed_mix
            ))
            return false;
        if (owed && token.wait.predecessor == owed_mix) member.ordinary_consumer_waits = true;
        *progress = *progress || token.next_fanin != previous;
        if (token.next_fanin != token.fanin.count) return true;
    }
    if (selected == nullptr) return true;
    SchedulerScalarScope selection(local, SchedPhaseKind::ScalarSelectOwnedExecution, token.task_ref);
    if (token.state == SchedulerDispatchSlotState::FILLING) {
        const uint64_t fanin_seen =
            profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL ? scheduler_cycles() : 0;
        if (scheduler_task_has_predicate(token.metadata.flags)) {
            const auto result = scheduler_evaluate_task_predicate(graph, token.task_ref);
            if (result == SchedulerPredicateResult::MALFORMED)
                return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
            token.skip_kernel = result == SchedulerPredicateResult::FAIL;
        }
        if (__builtin_expect(local->profiling == nullptr, true)) {
            token.may_observe_async_inputs = (token.metadata.flags & SCHEDULER_TASK_ASYNC_INPUT) != 0;
            token.state = SchedulerDispatchSlotState::READY;
        } else {
            const uint64_t start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : fanin_seen;
            scheduler_publish_dispatch_slot(base, local, 0, token.metadata, profiling_level, start);
        }
        *progress = true;
    }
    bool permitted = false;
    if (!scheduler_self_order_ready(graph, local, run, token.rank, token.required_floor, &permitted)) return false;
    if (permitted) *selected = 0;
    return true;
}

static inline __aicore__ bool scheduler_progress_execution_tokens(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, uint32_t *selected, bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionTokenProgress);
    if (local == nullptr || selected == nullptr || progress == nullptr) return false;
    *progress = false;
    *selected = UINT32_MAX;
    if (local->config.prepared == nullptr) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    auto &token = local->execution_tokens[0];
    if (token.state == SchedulerDispatchSlotState::FREE || token.state == SchedulerDispatchSlotState::EXECUTING)
        return true;
    if (token.state == SchedulerDispatchSlotState::TICKETED || token.prepared.claimed_state == 0)
        return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    return scheduler_advance_owned_execution(graph, base, local, run, profiling_level, selected, progress);
}

// Publication may race ordinary discovery or binding. Only a fresh SSBUF generation
// for this lane merits servicing here; an observed group keeps its normal fanin cadence.
static inline __aicore__ bool scheduler_pick_new_mix_member(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, uint32_t *selected, bool *progress
) {
    static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
    const auto &member = local->mix_groups[0];
    if (member.phase != SchedulerGmMixPhase::IDLE) return true;
    const auto &descriptor = local->cluster_signal->groups[0];
    const uint32_t generation = scheduler_cluster_load_field(&descriptor.armed_generation);
    if (!scheduler_cluster_mix_armed(generation) || generation == member.generation) return true;
    if ((scheduler_cluster_load_field(&descriptor.active_mask) & (1U << local->config.self_lane)) == 0) return true;

    bool advanced = false;
    if (!scheduler_cluster_mix_service_active(graph, base, local, run, profiling_level, &advanced, 0)) return false;
    *progress = *progress || advanced;
    *selected = scheduler_cooperative_execution(local);
    return true;
}

static inline __aicore__ __attribute__((noinline)) bool scheduler_refill_execution_tokens(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr, uint32_t *selected = nullptr
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarExecutionTokenRefill);
    if (base == nullptr || local == nullptr || run == nullptr || progress == nullptr) return false;
    *progress = false;
    if (selected != nullptr) *selected = UINT32_MAX;
    if (local->config.prepared == nullptr) return true;
    const int64_t previous = local->continuation_task_ref;
    local->continuation_task_ref = SCHEDULER_TASK_REF_INVALID;
    if (scheduler_has_execution_tokens(local)) return true;
    const uint64_t discovery_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &task_ref, progress, snapshot)) return false;
    if (scheduler_task_ref_is_null(task_ref)) return true;
    if (local->issue_binding.task_ref == task_ref && local->issue_binding.ticket_exhausted) return true;
    const auto &metadata = local->issue_binding.metadata;
    if (!scheduler_task_uses_execution_tokens(metadata.flags)) return true;
    const uint32_t blocks = metadata.logical_block_num;
    const uint8_t lane = scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata));
    if (blocks == 0 || lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        scheduler_metadata_core_type_index(lane) != local->config.self_core_type)
        return scheduler_execution_token_fail(graph, local, run, task_ref);
    const auto fanin = local->issue_binding.fanin;
    auto *control = scheduler_task_control_at(base, local, task_ref);
    // Valid tickets are irrevocable; excess rollbacks cannot reopen supply.
    // Exhaustion suppresses local claims but does not publish the supply seal.
    const bool fixed = scheduler_task_has_fixed_owners(metadata.flags);
    // A fixed consumer already has its block by identity. If its last dependency is our
    // pending MIX member, binding it here would delay observing the member's own inputs.
    static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
    auto &member = local->mix_groups[0];
    if (fixed && member.phase == SchedulerGmMixPhase::PUBLISHED && fanin.count != 0) {
        int64_t predecessor = SCHEDULER_TASK_REF_INVALID;
        const auto status = scheduler_fanin_ref(fanin, task_ref, fanin.count - 1, &predecessor);
        if (status != SchedulerGraphResult::OK)
            return scheduler_issue_fail(graph, local, run, task_ref, status, SchedulerErrorSite::ROUTE_INVALID_STATE);
        if (predecessor == member.task_ref) {
            member.ordinary_consumer_waits = true;
            return true;
        }
    }
    if (!fixed && scheduler_gm_query(control->next_block) >= blocks) {
        scheduler_invalidate_issue_snapshot(snapshot);
        if (local->issue_binding.task_ref == task_ref) local->issue_binding.ticket_exhausted = true;
        return true;
    }
    const bool sync = scheduler_task_requires_sync_start(metadata.flags);
    if (sync && local->last_sync_self_ref == task_ref) return true;
    uint32_t rank = 0;
    uint32_t required_floor = 0;
    const auto read_order = [&]() __aicore__ {
        const auto &order = scheduler_order_for(local, task_ref);
        rank = order.rank;
        required_floor = order.required_floor[local->config.self_lane];
    };
    if (sync) {
        read_order();
        bool permit = false;
        if (!scheduler_self_order_ready(graph, local, run, rank, required_floor, &permit)) return false;
        if (!permit) return true;
    }
    const bool check_publication =
        selected != nullptr && local->cluster_signal != nullptr && !local->is_mix_coordinator();
    if (check_publication) {
        if (!scheduler_pick_new_mix_member(graph, base, local, run, profiling_level, selected, progress)) return false;
        if (*selected != UINT32_MAX) return true;
    }
    const bool trace_claim = sync && scheduler_protocol_enabled(local);
    const uint64_t before = __builtin_expect(trace_claim, false) ? scheduler_protocol_cycles(local) : 0;
    const uint64_t acquire_start =
        profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL ? scheduler_cycles() : discovery_start;
    uint32_t block = blocks;
    bool closes_supply = false;
    if (fixed) {
        block = local->config.cluster_index;
        if (lane != static_cast<uint8_t>(SubtaskSlot::AIC))
            block = block * PLATFORM_AIV_CORES_PER_BLOCKDIM + local->config.self_lane - 1;
    } else if (!scheduler_claim_ticket(graph, base, local, run, task_ref, blocks, &block, &closes_supply)) {
        return false;
    }
    scheduler_invalidate_issue_snapshot(snapshot);
    if (block >= blocks) {
        if (local->issue_binding.task_ref == task_ref) local->issue_binding.ticket_exhausted = true;
        return true;
    }
    if (sync) {
        local->last_sync_self_ref = task_ref;
        if (__builtin_expect(trace_claim, false)) {
            scheduler_protocol_block_claim(
                base, local, task_ref, block, before, scheduler_protocol_cycles(local), closes_supply ? blocks : 0
            );
            scheduler_protocol_member(base, local, task_ref, false);
        }
    } else {
        read_order();
    }
    auto &token = local->execution_tokens[0];
    token.metadata = metadata;
    token.rank = rank;
    token.required_floor = required_floor;
    token.sync_start = sync;
    token.task_ref = task_ref;
    token.block_idx = block;
    token.closes_supply = closes_supply;
    token.subtask_slot = lane;
    token.discovery_cycles = acquire_start;
    token.fanin = fanin;
    // A returned ordinary block proves input readiness for later blocks of the same task.
    token.next_fanin = previous == task_ref && !sync ? fanin.count : 0;
    token.state = SchedulerDispatchSlotState::TICKETED;
    if (!scheduler_bind_ticketed_execution(graph, base, local, run, token, profiling_level)) return false;
    if (fixed) ++local->issue_cursor;
    *progress = true;
    if (check_publication) {
        if (!scheduler_pick_new_mix_member(graph, base, local, run, profiling_level, selected, progress)) return false;
        if (*selected != UINT32_MAX) return true;
    }
    return scheduler_advance_owned_execution(graph, base, local, run, profiling_level, selected, progress);
}
