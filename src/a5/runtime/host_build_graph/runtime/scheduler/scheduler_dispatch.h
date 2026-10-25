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
// so this must admit every slot scheduler_fill_own_slots would fill: rejecting
// a fillable slot suppresses the directory query and starves its work.
inline __aicore__ bool
scheduler_has_usable_slot(const SchedulerLocalState *local, uint32_t core_type, uint64_t skip_slot_mask) {
    if (local->config.self_core_type != core_type) return false;
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if ((skip_slot_mask & (UINT64_C(1) << slot)) == 0 &&
            local->own_slots()[slot].state == SchedulerDispatchSlotState::FREE)
            return true;
    }
    return false;
}

// Fill this core's own free dispatch slots. One core, one core type, two slots:
// a Scheduler schedules only itself, so there is no peer to prefer, no reserved
// capacity to hold back, and no reason to defer a slot it is about to execute
// from. `skip_slot_mask` names slots a completion pass already refilled.
//
// The return value reports whether this pass made progress; `failed` separately
// reports an aborted pass.
inline __aicore__ bool scheduler_fill_own_slots(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t *ready_victim_cursors, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, uint64_t skip_slot_mask, bool *failed, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (failed != nullptr) *failed = false;
    if (scheduler == nullptr) return false;
    if (scheduler->admission != nullptr)
        return scheduler_fill_ordered_slots(
            graph, scheduler_state_base, scheduler, run_control, ready_victim_cursors, ready_stats, profiling_level,
            failed, ssbuf_region
        );
    const uint32_t core_type = scheduler->config.self_core_type;
    if (!scheduler_has_usable_slot(scheduler, core_type, skip_slot_mask)) return false;
    // A core polls its own inbox on every pass. The directory gates only the
    // steal scan, which reads one peer endpoint per victim. Any publisher sets
    // an owner's bit and only that owner clears it, after observing its own
    // inbox empty -- so gating the owner's own pop on that bit would strand
    // whatever is queued when a clear and a push cross, until a later push
    // sets the bit again.
    const bool allow_steal = scheduler_ready_directory_nonempty(
        scheduler_state_base, scheduler, scheduler->config.scheduler_count, core_type
    );

    const uint64_t worker_id = scheduler->worker_id();
    const uint32_t self_lane = scheduler->config.self_lane;
    uint64_t state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    bool progress = false;

    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if ((skip_slot_mask & (UINT64_C(1) << slot)) != 0) continue;
        SchedulerLocalSlotState *local_slot = &scheduler->own_slots()[slot];
        if (local_slot->state != SchedulerDispatchSlotState::FREE) continue;

        SchedulerReadyClaim ready{};
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count, core_type,
                &ready_victim_cursors[core_type], ready_stats, &ready, nullptr, allow_steal
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        // An empty inbox ends the pass: the next slot would query the same one.
        if (scheduler_task_ref_is_null(ready.task_ref)) break;

        ready.state_probe_start_cycles = state_probe_start_cycles;
        ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        const SchedulerFreeSlotClaim claim{worker_id, slot, local_slot->generation, self_lane};
        local_slot->state = SchedulerDispatchSlotState::FILLING;
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        progress = true;
        state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    }
    return progress;
}
