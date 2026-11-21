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
#include "scheduler_async_finish.h"

enum class SchedulerSubtaskFinishResult : uint8_t { PARTIAL, TASK_LAST, TASK_LAST_DEFERRED, ERROR };

// Each credit retires one physical return and its compute debt together.
// A task with deferred conditions retains its single async marker.
static inline __aicore__ SchedulerSubtaskFinishResult scheduler_contribute_returns(
    __gm__ SchedulerTaskControl *control, uint64_t credits, uint64_t *previous, uint64_t *completed_gate = nullptr
) {
    SchedulerScalarScope scope(SchedPhaseKind::ScalarReturnsContribute);
    scheduler_cache_barrier();
    const uint64_t state =
        scheduler_gm_fetch_add(control->completion_state, UINT64_C(0) - credits * SCHEDULER_COMPLETION_RETURN_UNIT);
    *previous = scheduler_completion_count(state);
    if (!scheduler_completion_valid(state) || credits == 0 || scheduler_completion_remaining(state) < credits)
        return SchedulerSubtaskFinishResult::ERROR;
    if (scheduler_completion_remaining(state) != credits) return SchedulerSubtaskFinishResult::PARTIAL;
    if (completed_gate != nullptr) *completed_gate = state - credits * SCHEDULER_COMPLETION_RETURN_UNIT;
    return scheduler_completion_debt(state) == credits ? SchedulerSubtaskFinishResult::TASK_LAST :
                                                         SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED;
}

// Only the group's coordinator calls this after every resource returned and
// every computing lane certified a successful return. Its phase remains
// RELEASED until this contribution is complete, then becomes IDLE exactly once.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_publish_fanout(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint32_t block_idx, LaneMask active
) {
    const bool counted =
        SCHEDULER_REQUIRES_CNT_NOTIFY ||
        scheduler_completion_counter_at(base, local->config.completion_counters_offset, task_ref) != nullptr;
    auto *layout = RecordProtocol ? scheduler_protocol_layout_at(base, local) : nullptr;
    // Each counted lane publishes P on return. The group owner records only bank retirement.
    if (counted && layout == nullptr) return true;
    SchedulerScalarScope scope(counted ? nullptr : local, SchedPhaseKind::ScalarMixGroupContribute, task_ref);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    __gm__ SchedulerProtocolBlockOwner *owner = nullptr;
    if (layout != nullptr) {
        const auto *metadata = scheduler_task_metadata_at(base, local, task_ref);
        const uint64_t index = metadata->trace_index_base + static_cast<uint64_t>(block_idx) * active.count();
        if (index >= layout->trace_count) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return false;
        }
        owner = &scheduler_state_at<SchedulerProtocolBlockTrace>(base, layout->blocks_offset)[index].owner;
        scheduler_protocol_observe_record(owner);
        if (owner->completion_before.sequence != 0 || owner->task_ref != task_ref || owner->block_idx != block_idx ||
            owner->worker_index != local->config.worker_ids[1] ||
            local->config.self_lane != scheduler_mix_accountant_lane(active) ||
            scheduler_lane_mask_load(owner->active_mask) != active) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return false;
        }
        owner->completion_worker = static_cast<uint32_t>(local->worker_id());
        scheduler_protocol_stamp(base, local, layout, &owner->completion_before);
    }
    uint64_t previous = UINT64_MAX;
    uint64_t completed_gate = 0;
    const auto result =
        counted ? SchedulerSubtaskFinishResult::PARTIAL :
                  scheduler_contribute_returns(
                      scheduler_task_control_at(base, local, task_ref), active.count(), &previous, &completed_gate
                  );
    if (owner != nullptr) {
        owner->completion_ticket = previous;
        owner->completion_credits = 0;
        scheduler_protocol_stamp(base, local, layout, &owner->completion_after);
        scheduler_protocol_publish_record(owner);
    }
    if (result == SchedulerSubtaskFinishResult::ERROR) {
        scheduler_record_error(
            run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::BLOCK_COMPLETION_INVALID
        );
        return false;
    }
    if (result == SchedulerSubtaskFinishResult::TASK_LAST) {
        scheduler_publish_task_done(base, local, task_ref);
        scheduler_remember_task_proof(local, task_ref, SchedulerTaskProof::DONE);
        if (owner != nullptr) {
            auto *actor =
                &scheduler_state_at<SchedulerProtocolTaskTrace>(base, layout->tasks_offset)[task_ref].completion;
            scheduler_protocol_observe_record(actor);
            if (actor->before.sequence != 0 || actor->after.sequence != 0) {
                scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
                return false;
            }
            actor->before.cycles = owner->completion_before.cycles;
            actor->before.sequence = owner->completion_before.sequence;
            actor->after.cycles = owner->completion_after.cycles;
            actor->after.sequence = owner->completion_after.sequence;
            actor->run_epoch = layout->run_epoch;
            actor->task_ref = task_ref;
            actor->order_key = owner->order_key;
            actor->worker_index = static_cast<uint32_t>(local->worker_id());
            actor->flags =
                scheduler_task_metadata_at(base, local, task_ref)->flags | SCHEDULER_PROTOCOL_COMPLETION_FROM_GROUP;
            scheduler_protocol_publish_record(actor);
        }
    } else if (result == SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED) {
        const auto completion = scheduler_async_close_computation(local->async, task_ref, completed_gate);
        const bool complete = completion.result == SchedulerAsyncCompletionResult::FINISH_OWNER;
        if ((complete && !scheduler_async_complete_task(graph, base, local, run, completion)) ||
            (!complete && completion.result != SchedulerAsyncCompletionResult::WAITING)) {
            scheduler_record_error(
                run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::ASYNC_CONTEXT_INVALID
            );
            return false;
        }
    }
    return true;
}

// At profiling retirement each worker snapshots its own partition once. A
// failed run can leave holes, so visit later counters even if one is pending.
static inline __aicore__ void scheduler_snapshot_counter_completions(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if (local->config.completion_counters_offset == 0) return;
    SchedulerScalarScope snapshot(local, SchedPhaseKind::ScalarCounterAccounting);
    for (uint64_t task = local->worker_id(); task < graph.task_count; task += local->config.runtime_worker_count) {
        const auto *counter = scheduler_completion_counter_at(base, local->config.completion_counters_offset, task);
        if (counter == nullptr) continue;
        const int observed = scheduler_completion_counter_observe(*counter);
        if (observed == 0) continue;
        const auto &metadata = *scheduler_task_metadata_at(base, local, task);
        uint32_t total = 0;
        if (observed < 0 || !scheduler_subtask_count_checked(
                                metadata.logical_block_num, scheduler_task_lanes(metadata).count(), &total
                            )) {
            scheduler_record_error(
                run, task, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::BLOCK_COMPLETION_INVALID
            );
            return;
        }
        const uint32_t flags = metadata.flags | SCHEDULER_PROTOCOL_COUNTER_ACCOUNTING;
        scheduler_protocol_task_actor(base, local, task, flags, true);
        scheduler_store_signal(
            scheduler_task_control_at(base, local, task)->completion_state, scheduler_completion_completed_state(total)
        );
        scheduler_protocol_task_actor(base, local, task, flags, false);
    }
}
