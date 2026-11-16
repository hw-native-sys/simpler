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
// Deferred conditions retain their independently registered debt.
static inline __aicore__ SchedulerSubtaskFinishResult
scheduler_contribute_returns(__gm__ SchedulerTaskControl *control, uint64_t credits, uint64_t *previous) {
    scheduler_cache_barrier();
    const uint64_t state =
        scheduler_gm_fetch_add(control->completion_state, UINT64_C(0) - credits * SCHEDULER_COMPLETION_RETURN_UNIT);
    *previous = scheduler_completion_count(state);
    if (!scheduler_completion_valid(state) || credits == 0 || scheduler_completion_remaining(state) < credits)
        return SchedulerSubtaskFinishResult::ERROR;
    if (scheduler_completion_remaining(state) != credits) return SchedulerSubtaskFinishResult::PARTIAL;
    return scheduler_completion_debt(state) == credits ? SchedulerSubtaskFinishResult::TASK_LAST :
                                                         SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED;
}

// Only the group's coordinator calls this after every resource returned and
// every computing lane certified a successful return. Its phase remains
// RELEASED until this contribution is complete, then becomes IDLE exactly once.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_contribute_mix_group(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint32_t block_idx, LaneMask active
) {
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    __gm__ SchedulerProtocolBlockOwner *owner = nullptr;
    auto *layout = RecordProtocol ? scheduler_protocol_layout_at(base, local) : nullptr;
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
            owner->worker_index != local->worker_id() || scheduler_lane_mask_load(owner->active_mask) != active) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return false;
        }
        scheduler_protocol_stamp(base, local, layout, &owner->completion_before);
    }
    uint64_t previous = 0;
    const auto result =
        scheduler_contribute_returns(scheduler_task_control_at(base, local, task_ref), active.count(), &previous);
    if (owner != nullptr) {
        owner->completion_ticket = previous;
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
        ++local->pending_completed;
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
        const auto completion = scheduler_async_close_computation(local->async, task_ref);
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
