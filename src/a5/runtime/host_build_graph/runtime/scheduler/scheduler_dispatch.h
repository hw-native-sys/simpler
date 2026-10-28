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

// AICore-resident dispatch library; scheduler_dispatch.cpp is the independent AICPU scheduler implementation.

#include "scheduler_completion.h"

// A directory load is useful only when the corresponding work can be claimed,
// so this must admit exactly what scheduler_fill_own_slots would fill:
// rejecting a fillable position suppresses the directory query and starves its
// work, and admitting an unfillable one costs a load for nothing.
//
// `refilled` says a completion pass already put a successor in the position
// this pass would otherwise claim for.
inline __aicore__ bool scheduler_has_usable_slot(const SchedulerLocalState *local, uint32_t core_type, bool refilled) {
    if (local->config.self_core_type != core_type || refilled) return false;
    return local->running.state == SchedulerDispatchSlotState::FREE;
}

// Claim one task for this core's own account, into the position it runs from.
// It claims only when that position is free, and the mailbox has already had
// its chance by then: promotion runs earlier in the loop, so a free position
// here means there was nothing of the coordinator's to take first. That
// ordering is what keeps ordinary work from piling up in front of a Mix
// subtask -- the whole backlog, not just the task then running, because
// without it the core would go on claiming.
//
// The return value reports whether this pass made progress; `failed` separately
// reports an aborted pass.
inline __aicore__ bool scheduler_fill_own_slots(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t *ready_victim_cursors, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, bool refilled, bool *failed, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (failed != nullptr) *failed = false;
    if (scheduler == nullptr) return false;
    const uint32_t core_type = scheduler->config.self_core_type;
    // The queue this core claims from is the one its own core type runs. They
    // are separate vocabularies that happen to share the AIC/AIV numbering, so
    // the slot check below takes the core type and everything queue-side takes
    // the task type.
    const TaskType task_type = scheduler_self_task_type(core_type);
    if (!scheduler_has_usable_slot(scheduler, core_type, refilled)) return false;
    // A core polls its own inbox on every pass. The directory gates only the
    // steal scan, which reads one peer endpoint per victim. Any publisher sets
    // an owner's bit and only that owner clears it, after observing its own
    // inbox empty -- so gating the owner's own pop on that bit would strand
    // whatever is queued when a clear and a push cross, until a later push
    // sets the bit again.
    const bool allow_steal = scheduler_ready_directory_nonempty(
        scheduler_state_base, scheduler, scheduler->config.scheduler_count, task_type
    );

    const uint64_t worker_id = scheduler->worker_id();
    const uint32_t self_lane = scheduler->config.self_lane;
    const uint64_t state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;

    SchedulerLocalSlotState *local_slot = &scheduler->running;
    if (refilled || local_slot->state != SchedulerDispatchSlotState::FREE) return false;

    SchedulerReadyClaim ready{};
    if (!scheduler_claim_ready_for_slot(
            graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count, task_type,
            &ready_victim_cursors[scheduler_task_type_index(task_type)], ready_stats, &ready, nullptr, allow_steal
        )) {
        if (failed != nullptr) *failed = true;
        return false;
    }
    if (scheduler_task_ref_is_null(ready.task_ref)) return false;

    ready.state_probe_start_cycles = state_probe_start_cycles;
    ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    const SchedulerFreeSlotClaim claim{worker_id, local_slot->generation, self_lane};
    local_slot->state = SchedulerDispatchSlotState::FILLING;
    if (!scheduler_fill_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
        )) {
        if (failed != nullptr) *failed = true;
        return false;
    }
    return true;
}
