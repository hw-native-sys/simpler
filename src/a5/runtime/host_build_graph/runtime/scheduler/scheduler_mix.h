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

#include "scheduler_dispatch.h"

inline __aicore__ bool scheduler_dispatch_mix_ready(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerReadyClaim &ready, const SchedulerMixPlacement &placement, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf
) {
    if (scheduler_task_ref_is_null(ready.task_ref) || placement.count < 2 ||
        placement.tracker >= SCHEDULER_MIX_TRACKER_COUNT)
        return false;
    const SchedulerTaskMetadata metadata =
        scheduler_load_dispatch_metadata(base, local, ready.task_ref, profiling_level);
    SchedulerTaskPayloadArguments arguments{};
    const bool inline_task = scheduler_task_is_inline(metadata.flags);
    if (!inline_task) {
        const SchedulerGraphResult status = scheduler_parse_task_payload_arguments(graph, ready.task_ref, &arguments);
        if (status != SchedulerGraphResult::OK) {
            scheduler_record_error(
                run, ready.task_ref, status, &graph, local, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
            return false;
        }
    }
    auto &tracker = local->mix_trackers[placement.tracker];
    tracker.task_ref = ready.task_ref;
    tracker.active_mask = metadata.active_mask;
    tracker.completed_mask = 0;
    for (uint8_t i = 0; i < placement.count; ++i) {
        const auto &claim = placement.slots[i];
        auto &slot = *local->slot_state(claim);
        slot.state = SchedulerDispatchSlotState::FILLING;
        slot.mix_tracker = placement.tracker;
    }
    uint64_t prepare_start_cycles[3]{};
    __gm__ DispatchPayload *payloads[3]{};
    for (uint8_t i = 0; i < placement.count; ++i) {
        if (!scheduler_prepare_dispatch_slot(
                graph, base, local, run, placement.slots[i], ready, metadata, profiling_level, ssbuf,
                placement.subtasks[i], &prepare_start_cycles[i], inline_task ? nullptr : &arguments, false
            )) {
            for (uint8_t j = 0; j < placement.count; ++j) {
                const auto &claim = placement.slots[j];
                scheduler_initialize_free_slot(local->slot_state(claim));
            }
            tracker.task_ref = SCHEDULER_TASK_REF_INVALID;
            scheduler_record_error(run, ready.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
            return false;
        }
        payloads[i] = scheduler_state_at<DispatchPayload>(base, local->slot_payload_offset(placement.slots[i]));
    }
    if (!inline_task) scheduler_materialize_mix_arguments(arguments, payloads, placement.count);
    for (uint8_t i = 0; i < placement.count; ++i)
        scheduler_stage_dispatch_slot(base, local, placement.slots[i], ready, metadata, ssbuf, placement.subtasks[i]);
    scheduler_cache_barrier();
    for (uint8_t i = 0; i < placement.count; ++i)
        scheduler_commit_dispatch_slot(
            base, local, placement.slots[i], ready, metadata, profiling_level, ssbuf, placement.subtasks[i],
            prepare_start_cycles[i]
        );
    return true;
}

inline __aicore__ bool scheduler_fill_cluster_mix_slots(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t *victim_cursor, SchedulerReadyStats *stats, uint64_t profiling_level, bool allow_steal,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf, bool *progress
) {
    // Mix placement is the coordinator's alone, because only it writes the
    // Mix slots a placement reserves.
    if (!local->has_mix || !local->is_mix_coordinator()) return true;
    for (uint32_t attempt = 0; attempt < SCHEDULER_MIX_TRACKER_COUNT; ++attempt) {
        SchedulerMixPlacement placement{};
        SchedulerReadyClaim ready{};
        if (!scheduler_claim_ready_for_slot(
                graph, base, local, run, local->config.scheduler_count, TaskType::MIX, victim_cursor, stats, &ready,
                &placement, allow_steal, ssbuf
            ))
            return false;
        if (scheduler_task_ref_is_null(ready.task_ref)) return true;
        if (!scheduler_dispatch_mix_ready(graph, base, local, run, ready, placement, profiling_level, ssbuf))
            return false;
        if (progress != nullptr) *progress = true;
    }
    return true;
}

inline __aicore__ bool scheduler_publish_direct_mix_candidate(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerReadyClaim *candidate, SchedulerReadyStats *stats, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf, bool *progress
) {
    if (candidate == nullptr || scheduler_task_ref_is_null(candidate->task_ref)) return true;
    // The ready-queue snapshot permits a direct attempt before the next queue
    // claim. A candidate without complete placement joins the queue first.
    SchedulerMixPlacement placement{};
    if (scheduler_plan_mix_placement(base, local, candidate->task_ref, &placement, ssbuf)) {
        if (!scheduler_dispatch_mix_ready(graph, base, local, run, *candidate, placement, profiling_level, ssbuf))
            return false;
        if (progress != nullptr) *progress = true;
    } else {
        SchedulerReadyBatch batch{};
        if (!scheduler_ready_batch_append(base, local, candidate->task_ref, &batch, stats, profiling_level) ||
            !scheduler_ready_batch_push(base, local, TaskType::MIX, &batch, stats))
            return false;
    }
    *candidate = {};
    return true;
}

inline __aicore__ bool scheduler_fill_mix_after_completions(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerReadyClaim *candidate, uint64_t *victim_cursor, SchedulerReadyStats *stats, uint64_t profiling_level,
    bool allow_steal, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf, bool *progress
) {
    if (local->has_mix && local->is_mix_coordinator() &&
        !scheduler_publish_direct_mix_candidate(
            graph, base, local, run, candidate, stats, profiling_level, ssbuf, progress
        ))
        return false;
    return scheduler_fill_cluster_mix_slots(
        graph, base, local, run, victim_cursor, stats, profiling_level, allow_steal, ssbuf, progress
    );
}
