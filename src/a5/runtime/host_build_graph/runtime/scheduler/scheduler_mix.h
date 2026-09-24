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
    if (ready.task_id < 0 || placement.count < 2 || placement.tracker >= SCHEDULER_MIX_TRACKER_COUNT) return false;
    const SchedulerTaskMetadata metadata =
        scheduler_load_dispatch_metadata(base, local, ready.task_id, profiling_level);
    SchedulerTaskPayloadArguments arguments{};
    const bool inline_task = scheduler_task_is_inline(metadata.flags);
    if (!inline_task) {
        const SchedulerGraphResult status = scheduler_parse_task_payload_arguments(graph, ready.task_id, &arguments);
        if (status != SchedulerGraphResult::OK) {
            scheduler_record_error(
                run, ready.task_id, status, &graph, local, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
            return false;
        }
    }
    auto &tracker = local->mix_trackers[placement.tracker];
    tracker.task_id = ready.task_id;
    tracker.active_mask = metadata.active_mask;
    tracker.completed_mask = 0;
    for (uint8_t i = 0; i < placement.count; ++i) {
        const auto &claim = placement.slots[i];
        auto &slot = local->slots[claim.cluster_lane][claim.slot_index];
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
                scheduler_initialize_free_slot(&local->slots[claim.cluster_lane][claim.slot_index]);
            }
            tracker.task_id = SCHEDULER_TASK_ID_INVALID;
            scheduler_record_error(run, ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local);
            return false;
        }
        payloads[i] = scheduler_state_at<DispatchPayload>(
            base, local->dispatch_payload_offset(placement.slots[i].cluster_lane, placement.slots[i].slot_index)
        );
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
    if (!local->has_mix) return true;
    for (uint32_t attempt = 0; attempt < SCHEDULER_MIX_TRACKER_COUNT; ++attempt) {
        SchedulerMixPlacement placement{};
        SchedulerReadyClaim ready{};
        if (!scheduler_claim_ready_for_slot(
                graph, base, local, run, local->config.scheduler_count, SCHEDULER_MIX_QUEUE, victim_cursor, stats,
                &ready, &placement, allow_steal
            ))
            return false;
        if (ready.task_id < 0) return true;
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
    if (candidate == nullptr || candidate->task_id < 0) return true;
    // The ready-queue snapshot permits a direct attempt before the next queue
    // claim. A candidate without complete placement joins the queue first.
    SchedulerMixPlacement placement{};
    if (scheduler_plan_mix_placement(base, local, candidate->task_id, &placement)) {
        if (!scheduler_dispatch_mix_ready(graph, base, local, run, *candidate, placement, profiling_level, ssbuf))
            return false;
        if (progress != nullptr) *progress = true;
    } else {
        SchedulerReadyBatch batch{};
        if (!scheduler_ready_batch_append(base, local, candidate->task_id, &batch, stats, profiling_level) ||
            !scheduler_ready_batch_push(base, local, SCHEDULER_MIX_QUEUE, &batch, stats))
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
    if (local->has_mix && !scheduler_publish_direct_mix_candidate(
                              graph, base, local, run, candidate, stats, profiling_level, ssbuf, progress
                          ))
        return false;
    return scheduler_fill_cluster_mix_slots(
        graph, base, local, run, victim_cursor, stats, profiling_level, allow_steal, ssbuf, progress
    );
}

inline __aicore__ bool scheduler_publish_refill_candidates(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerRefillCandidates *candidates, SchedulerReadyStats *stats, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf
) {
    SchedulerReadyBatch batches[SCHEDULER_READY_QUEUE_COUNT]{};
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
            const int64_t task = candidates->tasks[lane][index];
            if (task < 0) continue;
            const auto &metadata = *scheduler_task_metadata_at(base, local, task);
            SchedulerFreeSlotClaim claim{};
            for (uint32_t offset = 0; offset < PLATFORM_CORES_PER_BLOCKDIM && claim.slot_index == UINT32_MAX;
                 ++offset) {
                const uint32_t target_lane = (lane + offset) % PLATFORM_CORES_PER_BLOCKDIM;
                const uint64_t worker = local->config.worker_ids[target_lane];
                if (worker >= local->config.runtime_worker_count || target_lane == local->config.self_lane) continue;
                const auto *target = scheduler_worker_context_at(base, local, worker);
                if (scheduler_core_type_index(target->core_type) !=
                    scheduler_task_ready_queue(metadata.flags, metadata.active_mask))
                    continue;
                for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
                    const auto &state = local->slots[target_lane][slot];
                    if (state.state == SchedulerDispatchSlotState::FREE) {
                        claim = {worker, slot, state.generation, target_lane};
                        break;
                    }
                }
            }
            if (claim.slot_index != UINT32_MAX) {
                SchedulerReadyClaim ready{};
                ready.task_id = task;
                ready.source = SchedulerReadySource::DIRECT_RESOLVE;
                ready.publication_mode = SchedulerPublicationMode::REFILL;
                const uint64_t start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                if (!scheduler_fill_dispatch_slot(graph, base, local, run, claim, ready, profiling_level, ssbuf))
                    return false;
                if (scheduler_phase_timing_enabled(profiling_level)) {
                    auto *traces = scheduler_state_at<SchedulerTaskTrace>(base, local->profiling->trace_cells_offset);
                    auto &completed = traces[candidates->completed_trace_indices[lane][index]];
                    completed.refill_scheduler_worker_id = local->worker_id();
                    completed.refill_start_cycles = start;
                    completed.refill_end_cycles = scheduler_cycles();
                    completed.refill_task_id = static_cast<uint64_t>(task);
                    completed.refill_loop_iter = local->profiling->loop_iter;
                    scheduler_publish_cache_line(&completed.refill_scheduler_worker_id);
                }
            } else {
                if (!scheduler_ready_batch_append(
                        base, local, task, &batches[scheduler_task_ready_queue(metadata.flags, metadata.active_mask)],
                        stats, 0
                    ))
                    return false;
            }
            candidates->tasks[lane][index] = SCHEDULER_TASK_ID_INVALID;
        }
    }
    for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type)
        if (!scheduler_ready_batch_push(base, local, type, &batches[type], stats)) return false;
    return true;
}
