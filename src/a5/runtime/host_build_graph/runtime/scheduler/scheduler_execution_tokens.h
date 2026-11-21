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

#include "scheduler_order.h"
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
    if (token.phase != SchedulerExecutionTokenPhase::TICKETED)
        return token.phase == SchedulerExecutionTokenPhase::WAIT_FANIN;
    if (!scheduler_task_ref_valid(graph, token.task_ref) ||
        !scheduler_task_uses_execution_tokens(scheduler_task_metadata_at(base, local, token.task_ref)->flags))
        return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    const auto result = scheduler_claim_for_execution(
        base, local->config.prepared, local->worker_id(), token.task_ref, token.block_idx, token.subtask_slot,
        &token.prepared
    );
    if (!result) return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    const uint32_t blocks = scheduler_task_metadata_at(base, local, token.task_ref)->logical_block_num;
    const bool supplied = scheduler_close_ordinary_supply(
        graph, base, local, run, token.task_ref, token.closes_supply, blocks, token.block_idx
    );
    // Cancellation of a claimed handle inspects its Call, so payload receipt
    // completes even when supply publication reports an error.
    const auto received = scheduler_receive_payload(
        base, local->config.prepared, token.task_ref, token.block_idx, token.subtask_slot, &token.prepared
    );
    if (!supplied) return false;
    if (!received) return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    if (token.sync_start && graph.native_header_offset != 0) {
        const auto error = scheduler_observe_graph_arguments(graph, token.task_ref);
        if (error != SchedulerGraphBindingError::NONE) {
            scheduler_record_error(
                run, token.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                scheduler_graph_binding_error_site(error), &token
            );
            return false;
        }
    }
    token.metadata = scheduler_load_dispatch_metadata(base, local, token.task_ref, profiling_level);
    if (!scheduler_task_uses_execution_tokens(token.metadata.flags))
        return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
    token.acquired_cycles = scheduler_schedule_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    token.phase = SchedulerExecutionTokenPhase::WAIT_FANIN;
    return true;
}

static inline __aicore__ bool scheduler_poll_fanin(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerExecutionToken &token, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarExecutionTokenFanin, token.task_ref);
    const uint32_t previous = token.next_fanin;
    if (!scheduler_observe_fanin_prefix(
            graph, base, local, run, token.task_ref, token.fanin, token.next_fanin, &token.wait
        ))
        return false;
    *progress = *progress || token.next_fanin != previous;
    return true;
}

static inline __aicore__ bool scheduler_select_owned_execution(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t index, uint64_t profiling_level, uint32_t *selected, bool *progress
) {
    auto &token = local->execution_tokens[index];
    if (token.state != SchedulerDispatchSlotState::FILLING && token.state != SchedulerDispatchSlotState::READY)
        return true;
    SchedulerScalarScope selection(local, SchedPhaseKind::ScalarSelectOwnedExecution, token.task_ref);
    if (token.state != SchedulerDispatchSlotState::READY) {
        const uint64_t fanin_seen =
            profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL ? scheduler_cycles() : 0;
        if (token.predicate == SchedulerPredicateState::UNEVALUATED &&
            scheduler_task_has_predicate(token.metadata.flags)) {
            const auto result = scheduler_evaluate_task_predicate(graph, token.task_ref);
            if (result == SchedulerPredicateResult::MALFORMED)
                return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
            token.predicate = result == SchedulerPredicateResult::PASS ? SchedulerPredicateState::PASS :
                                                                         SchedulerPredicateState::FAIL;
        }
        if (token.predicate == SchedulerPredicateState::UNEVALUATED) token.predicate = SchedulerPredicateState::PASS;
        const uint64_t start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : fanin_seen;
        scheduler_publish_dispatch_slot(base, local, index, token.metadata, profiling_level, start);
        *progress = true;
    }
    bool permitted = false;
    if (!scheduler_self_order_ready(
            graph, base, local, run, token.rank, token.required_floor, token.sync_start, &permitted
        ))
        return false;
    if (!permitted) return true;
    if (token.sync_start) {
        // Before entering a collective with an earlier ordinary claim, all members must own tickets and pass their
        // order checks.
        for (const auto &earlier : local->execution_tokens) {
            if (earlier.rank >= token.rank || earlier.state == SchedulerDispatchSlotState::FREE || earlier.sync_start)
                continue;
            // Fixed ownership proves assignment only. Keep this core available
            // for its earlier work until the conflicting obligation retires.
            if (scheduler_task_has_fixed_owners(token.metadata.flags)) return true;
            const uint64_t seal =
                scheduler_gm_query(scheduler_task_control_at(base, local, token.task_ref)->claim_seal);
            if (seal == 0) return true;
            if (seal != token.metadata.logical_block_num)
                return scheduler_issue_fail(graph, local, run, token.task_ref);
            break;
        }
    }
    *selected = index;
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
    for (uint32_t checked = 0; checked < SCHEDULER_EXECUTION_TOKEN_COUNT; ++checked) {
        const uint32_t token_index = local->execution_check_cursor++ % SCHEDULER_EXECUTION_TOKEN_COUNT;
        auto &token = local->execution_tokens[token_index];
        if (token.phase == SchedulerExecutionTokenPhase::EMPTY || token.state == SchedulerDispatchSlotState::EXECUTING)
            continue;
        if (token.phase != SchedulerExecutionTokenPhase::WAIT_FANIN || token.prepared.claimed_state == 0)
            return scheduler_execution_token_fail(graph, local, run, token.task_ref, &token);
        if (token.next_fanin != token.fanin.count && !scheduler_poll_fanin(graph, base, local, run, token, progress))
            return false;
        if (token.next_fanin != token.fanin.count) continue;
        if (!scheduler_select_owned_execution(
                graph, base, local, run, token_index, profiling_level, selected, progress
            ))
            return false;
        if (*selected != UINT32_MAX) return true;
    }
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
    uint32_t empty = 0;
    while (empty < SCHEDULER_EXECUTION_TOKEN_COUNT &&
           local->execution_tokens[empty].state != SchedulerDispatchSlotState::FREE)
        ++empty;
    if (empty == SCHEDULER_EXECUTION_TOKEN_COUNT) return true;
    const uint64_t discovery_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &task_ref, progress, snapshot)) return false;
    if (scheduler_task_ref_is_null(task_ref)) return true;
    if (local->issue_task_ref == task_ref && local->issue_ticket_exhausted) return true;
    const auto &metadata = local->issue_binding.metadata;
    if (!scheduler_task_uses_execution_tokens(metadata.flags) || scheduler_execution_token_has_task(local, task_ref))
        return true;
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
    if (!fixed && scheduler_gm_query(control->next_block) >= blocks) {
        scheduler_invalidate_issue_snapshot(snapshot);
        if (local->issue_task_ref == task_ref) local->issue_ticket_exhausted = true;
        return true;
    }
    const bool sync = scheduler_task_requires_sync_start(metadata.flags);
    if (sync && local->last_sync_self_ref == task_ref) return true;
    uint32_t rank = 0;
    uint32_t required_floor = 0;
    const auto read_order = [&]() __aicore__ {
        const auto &order = scheduler_order_for(base, local, task_ref);
        rank = order.rank;
        required_floor = order.required_floor[local->config.self_lane];
    };
    if (sync) {
        read_order();
        bool permit = false;
        if (!scheduler_self_order_ready(graph, base, local, run, rank, required_floor, true, &permit)) return false;
        if (!permit) return true;
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
        if (local->issue_task_ref == task_ref) local->issue_ticket_exhausted = true;
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
    auto &token = local->execution_tokens[empty];
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
    token.phase = SchedulerExecutionTokenPhase::TICKETED;
    token.state = SchedulerDispatchSlotState::FILLING;
    if (!scheduler_bind_ticketed_execution(graph, base, local, run, token, profiling_level)) return false;
    if (fixed) ++local->issue_cursor;
    *progress = true;
    if (token.next_fanin != token.fanin.count && !scheduler_poll_fanin(graph, base, local, run, token, progress))
        return false;
    if (selected != nullptr && token.next_fanin == token.fanin.count) {
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        return scheduler_select_owned_execution(graph, base, local, run, empty, profiling_level, selected, progress);
    }
    return true;
}
