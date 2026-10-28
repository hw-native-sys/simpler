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

// AICore-resident completion library; scheduler_completion.cpp is the independent AICPU scheduler implementation.

#include "scheduler_ready.h"

// The caller publishes only at idle or in the common run epilogue. A barrier
// keeps trace/error publication ahead of the terminal completion count.
inline __aicore__ void
scheduler_flush_completions(__gm__ SchedulerRunControl *run_control, SchedulerLocalState *local) {
    const uint64_t completed = local->pending_completed;
    if (completed == 0) return;
    scheduler_cache_barrier();
    scheduler_gm_fetch_add(run_control->resolved_task_count, completed);
    local->pending_completed = 0;
}

inline __aicore__ void scheduler_account_failed_completion(
    const SchedulerGraphView &graph, SchedulerLocalState *scheduler, __gm__ SchedulerRunControl *run_control,
    int64_t task_ref, SchedulerErrorSite error_site
) {
    // Preserve a more specific error already latched by the failing helper. The
    // fallback covers argument/invariant guards that only report false.
    scheduler_record_error(
        run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler, error_site
    );
    ++scheduler->pending_completed;
}

// A Mix task resolved on the completion path, held for the coordinator to place
// once the scan is done. An ordinary successor needs no such hand-off: the core
// that resolved it owns the slot it came from and refills that slot directly.
struct SchedulerRefillCandidates {
    SchedulerReadyClaim mix_ready{};
};

inline __aicore__ bool scheduler_service_cluster_completion_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cluster_lane, uint32_t completed_generation,
    SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats,
    uint64_t profiling_level, const SchedulerReadyClaim *replacement_ready, bool *direct_refilled,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, SchedulerRefillCandidates *candidates = nullptr,
    bool mix_slot = false, bool lane_slot = false
) {
    if (direct_refilled != nullptr) *direct_refilled = false;
    if (ssbuf_region == nullptr || scheduler == nullptr || cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        completed_generation == 0)
        return false;
    // Callers prefilter consumed tokens. Keep this mutating helper fail-fast if
    // that precondition is violated, so a completion cannot be serviced twice.
    if (!scheduler_completion_generation_is_new(scheduler, cluster_lane, completed_generation, mix_slot, lane_slot))
        return false;
    const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
    if (worker_id >= scheduler->config.runtime_worker_count) return false;
    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    const SchedulerFreeSlotClaim completed_claim{worker_id, completed_generation, cluster_lane, mix_slot, lane_slot};
    SchedulerLocalSlotState *local_slot = scheduler->slot_state(completed_claim);
    const int64_t published_task_ref = local_slot->task_ref;
    if (local_slot->state != SchedulerDispatchSlotState::READY || local_slot->generation != completed_generation) {
        scheduler_record_error(
            run_control, published_task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
        );
        return false;
    }

    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t completion_observe = schedule_timing_enabled ? scheduler_cycles() : 0;
    const int64_t task_ref = local_slot->task_ref;
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, scheduler);
        return false;
    }
    SchedulerMixTracker *mix_tracker = nullptr;
    bool valid_subtask = local_slot->subtask_slot < 3;
    if (local_slot->mix_tracker != UINT8_MAX) {
        if (local_slot->mix_tracker >= SCHEDULER_MIX_TRACKER_COUNT) valid_subtask = false;
        else {
            mix_tracker = &scheduler->mix_trackers[local_slot->mix_tracker];
            const uint8_t bit = valid_subtask ? 1U << local_slot->subtask_slot : 0;
            valid_subtask = valid_subtask && mix_tracker->task_ref == task_ref &&
                            (mix_tracker->active_mask & bit) != 0 && (mix_tracker->completed_mask & bit) == 0;
        }
    }
    if (!valid_subtask) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
        );
        return false;
    }
    const bool sampled_task_timing_enabled = scheduler->sampled_task_timing(cluster_lane);
    SchedulerExecutorTaskTrace executor_trace{};
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        if (mix_slot) {
            // A Mix subtask's executor trace is in global memory: the cluster's
            // SSBUF has room for the dispatch and completion tokens, which are
            // on the dispatch path, and not for a second set of trace cells.
            __gm__ SchedulerExecutorTaskTrace *published_trace = scheduler_state_at<SchedulerExecutorTaskTrace>(
                scheduler_state_base, scheduler->mix_trace_offset(cluster_lane, lane_slot)
            );
            scheduler_observe_cache_line(published_trace);
            executor_trace.kernel_start_cycles = published_trace->kernel_start_cycles;
            executor_trace.kernel_end_cycles = published_trace->kernel_end_cycles;
            executor_trace.ready_scan_start_cycles = published_trace->ready_scan_start_cycles;
            executor_trace.ready_observe_cycles = published_trace->ready_observe_cycles;
            executor_trace.completion_id = published_trace->completion_id;
            executor_trace.completion_inbox_index = published_trace->completion_inbox_index;
        } else {
            executor_trace = scheduler->profiling->executor_trace;
        }
    }
    __gm__ SchedulerTaskTrace *completed_trace = nullptr;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->profiling->trace_cells_offset);
        completed_trace = &traces[scheduler_trace_index_at(
            scheduler_state_base, scheduler, task_ref, local_slot->subtask_slot, local_slot->block_idx
        )];
    }
    if (chip_task_timing_enabled) {
        // Finish all trace invalidations before any cached trace write.
        scheduler_invalidate_cache_line(completed_trace);
        scheduler_invalidate_cache_line(&completed_trace->kernel_start_cycles);
        if (phase_timing_enabled) {
            scheduler_invalidate_cache_line(&completed_trace->ready_transition_cycles);
            scheduler_invalidate_cache_line(&completed_trace->dispatch_start_cycles);
            scheduler_invalidate_cache_line(&completed_trace->refill_scheduler_worker_id);
            scheduler_invalidate_cache_line(&completed_trace->descriptor_cache_observed_cycles);
            if ((scheduler->profiling->worker_trace_valid_mask & (1U << cluster_lane)) == 0) {
                scheduler_invalidate_cache_line(&target->trace_aicore_entry_cycles);
                scheduler_invalidate_cache_line(&target->trace_register_release_cycles);
            }
        }
        scheduler_cache_barrier();
        completed_trace->kernel_start_cycles = executor_trace.kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace.kernel_end_cycles;
        completed_trace->ready_observe_cycles = executor_trace.ready_observe_cycles;
        if (schedule_timing_enabled) completed_trace->complete_start_cycles = completion_observe;
        if (phase_timing_enabled) {
            completed_trace->ready_scan_start_cycles = executor_trace.ready_scan_start_cycles;
            completed_trace->completion_id = executor_trace.completion_id;
            completed_trace->completion_inbox_index = executor_trace.completion_inbox_index;
            SchedulerWorkerTraceCache *worker_trace = &scheduler->profiling->worker_traces[cluster_lane];
            if ((scheduler->profiling->worker_trace_valid_mask & (1U << cluster_lane)) == 0) {
                worker_trace->descriptor_cache_observed_cycles = target->trace_descriptor_cache_observed_cycles;
                worker_trace->aicore_entry_cycles = target->trace_aicore_entry_cycles;
                worker_trace->handshake_publish_cycles = target->trace_handshake_publish_cycles;
                worker_trace->register_release_cycles = target->trace_register_release_cycles;
                scheduler->profiling->worker_trace_valid_mask |= 1U << cluster_lane;
            }
            completed_trace->descriptor_cache_observed_cycles = worker_trace->descriptor_cache_observed_cycles;
            completed_trace->aicore_entry_cycles = worker_trace->aicore_entry_cycles;
            completed_trace->handshake_publish_cycles = worker_trace->handshake_publish_cycles;
            completed_trace->register_release_cycles = worker_trace->register_release_cycles;
            completed_trace->complete_scheduler_worker_id = scheduler->worker_id();
            completed_trace->complete_loop_iter = scheduler->profiling->loop_iter;
        }
    } else if (completed_trace != nullptr) {
        completed_trace->kernel_start_cycles = executor_trace.kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace.kernel_end_cycles;
    }
    const uint8_t completed_subtask_slot = local_slot->subtask_slot;
    // Read with the subtask slot and for the same reason: both say what this
    // dispatch was, and the clear below is about to forget them.
    const uint16_t completed_block_num = local_slot->block_num;
    scheduler_mark_completion_consumed(scheduler, cluster_lane, completed_generation, mix_slot, lane_slot);
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    bool task_complete = true;
    if (mix_tracker != nullptr) {
        mix_tracker->completed_mask |= 1U << completed_subtask_slot;
        task_complete = mix_tracker->completed_mask == mix_tracker->active_mask;
        if (task_complete) mix_tracker->task_ref = SCHEDULER_TASK_REF_INVALID;
    }
    // Aggregation is two levels for a Mix SPMD task and one for the rest. The
    // tracker above closes a block -- its subtasks, on this core, in local
    // state. This closes the task: one atomic per block rather than one per
    // subtask of every block.
    //
    // The count comes from the slot, not from the task: the host writes
    // control->block_num once and never revisits it, so the copy a dispatch
    // carried is the same value, and a task of one block pays no atomic and
    // reads no GM at all.
    if (task_complete && completed_block_num > 1) {
        __gm__ SchedulerTaskControl *block_control =
            scheduler_task_control_at(scheduler_state_base, scheduler, task_ref);
        const uint64_t done = scheduler_gm_fetch_add(block_control->completed_blocks, 1) + 1;
        if (done > completed_block_num) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::SPMD_COMPLETION_OVERRUN
            );
            return false;
        }
        task_complete = done == completed_block_num;
    }
    local_slot->task_ref = SCHEDULER_TASK_REF_INVALID;
    local_slot->state = SchedulerDispatchSlotState::FREE;
    local_slot->subtask_slot = UINT8_MAX;
    local_slot->mix_tracker = UINT8_MAX;
    local_slot->block_idx = 0;
    local_slot->block_num = 1;
    scheduler->set_timing_slot(cluster_lane, -1);
    const uint64_t completion_phase_end = phase_timing_enabled ? scheduler_cycles() : 0;
    SchedulerReadyClaim resolved_ready{};
    if (task_complete) {
        __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_ref);
        scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        // A core owns the slot it just freed, so the successor it resolved can
        // go straight back into it. Only an ordinary slot qualifies: a Mix slot
        // belongs to a peer AIV and is filled through Mix placement, which is
        // the coordinator's alone. resolve_completion hands back a candidate
        // only when the successor's type matches the subtask that just
        // completed, so what lands here is always this core's own kind of work.
        //
        // Except while a sync-start cohort this core could run holds the SPMD
        // head. A direct refill bypasses the claim pass entirely, so a core
        // that resolves a successor on every completion is never seen with a
        // free slot and the cohort assembles only once the rest of the
        // reachable graph has drained -- while its already-claimed blocks spin
        // in a barrier, each holding a core whose scheduling loop is stopped.
        // Finite, but indistinguishable from a hang on a real graph. The head
        // is read from this core's memo, so this costs no GM access.
        const bool hold_slot_for_cohort = scheduler_spmd_head_is_own_cohort(scheduler);
        SchedulerReadyClaim *direct_ready =
            replacement_ready == nullptr && !mix_slot && !hold_slot_for_cohort ? &resolved_ready : nullptr;
        if (!scheduler_resolve_completion(
                graph, scheduler_state_base, scheduler, run_control, task_ref, wake_stats, ready_stats,
                completion_stats, profiling_level, false, direct_ready,
                scheduler_self_task_type(scheduler_metadata_core_type_index(completed_subtask_slot)),
                scheduler->has_mix && candidates != nullptr ? &candidates->mix_ready : nullptr
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_ref, SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
            );
            return false;
        }
    }
    SchedulerReadyClaim ready{};
    bool ready_available = replacement_ready != nullptr;
    if (ready_available) {
        ready = *replacement_ready;
    } else if (!scheduler_task_ref_is_null(resolved_ready.task_ref)) {
        ready = resolved_ready;
        ready_available = true;
    }
    if (ready_available) {
        if (refill_start_cycles == 0) refill_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
        ready.publication_mode = SchedulerPublicationMode::REFILL;
        SchedulerFreeSlotClaim claim{
            worker_id,
            local_slot->generation,
            cluster_lane,
        };
        local_slot->state = SchedulerDispatchSlotState::FILLING;
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_ref, SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED
            );
            return false;
        }
        refilled = true;
    }
    refill_end_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
    if (refill_start_cycles == 0) refill_start_cycles = refill_end_cycles;
    if (chip_task_timing_enabled) {
        if (phase_timing_enabled) {
            completed_trace->complete_end_cycles = completion_phase_end;
        }
        if (phase_timing_enabled && refilled) {
            completed_trace->refill_scheduler_worker_id = scheduler->worker_id();
            completed_trace->refill_start_cycles = refill_start_cycles;
            completed_trace->refill_end_cycles = refill_end_cycles;
            completed_trace->refill_task_ref = static_cast<uint64_t>(ready.task_ref);
            completed_trace->refill_loop_iter = scheduler->profiling->loop_iter;
        }
        scheduler_writeback_cache_line(completed_trace);
        scheduler_writeback_cache_line(&completed_trace->kernel_start_cycles);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->ready_transition_cycles);
        if (schedule_timing_enabled) scheduler_writeback_cache_line(&completed_trace->dispatch_start_cycles);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->refill_scheduler_worker_id);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->descriptor_cache_observed_cycles);
        scheduler_cache_barrier();
        scheduler_gm_store(completed_trace->valid, UINT64_C(1));
        // The preceding barrier publishes trace lines before valid. The store
        // drains valid before pending_completed reaches resolved_task_count.
    } else if (completed_trace != nullptr) {
        scheduler_publish_cache_line(&completed_trace->kernel_start_cycles);
    }
    if (task_complete) ++scheduler->pending_completed;
    if (direct_refilled != nullptr) *direct_refilled = refilled;
    return true;
}

inline __aicore__ bool scheduler_service_cluster_completions(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t profiling_level, bool *direct_refilled_out,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, SchedulerRefillCandidates *candidates = nullptr
) {
    if (scheduler == nullptr || ssbuf_region == nullptr) return false;
    if (direct_refilled_out != nullptr) *direct_refilled_out = false;
    bool progress = false;
    // This core's own position. A core retires what it ran, whoever put it
    // there: the mailbox hands over a dispatch, not the duty to retire it.
    const uint32_t completed_generation = scheduler->local_completed_generation;
    if (scheduler_completion_generation_is_new(scheduler, scheduler->config.self_lane, completed_generation)) {
        bool direct_refilled = false;
        if (!scheduler_service_cluster_completion_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->config.self_lane, completed_generation,
                wake_stats, ready_stats, completion_stats, profiling_level, nullptr, &direct_refilled, ssbuf_region,
                candidates
            )) {
            return false;
        }
        if (direct_refilled && direct_refilled_out != nullptr) *direct_refilled_out = true;
        progress = true;
    }
    if (!scheduler->has_mix || !scheduler->is_mix_coordinator()) return progress;
    // The Mix subtasks this core placed, whose completions arrive in each
    // lane's own inbox. Its own subtask of a Mix task ran out of the position
    // above, which is why that one is not a special case there.
    for (uint32_t cluster_lane = 1; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
        if (worker_id >= scheduler->config.runtime_worker_count) continue;
        const uint64_t mix_publication =
            scheduler_ssbuf_load_relaxed(&ssbuf_region->lanes[cluster_lane].mix_completion.publication);
        bool mix_acquired = false;
        for (bool lane_slot : {false, true}) {
            const uint32_t mix_generation = static_cast<uint32_t>(mix_publication >> (lane_slot ? 32U : 0U));
            if (!scheduler_completion_generation_is_new(scheduler, cluster_lane, mix_generation, true, lane_slot))
                continue;
            // The report names the slot, and the slot names the record: the
            // coordinator wrote that record when it placed the dispatch there.
            if (scheduler->mix_placed[cluster_lane][lane_slot].generation != mix_generation) {
                scheduler_record_error(
                    run_control, scheduler->mix_placed[cluster_lane][lane_slot].task_ref,
                    SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
                );
                return false;
            }
            if (!mix_acquired) {
                scheduler_cache_barrier();
                mix_acquired = true;
            }
            if (!scheduler_service_cluster_completion_slot(
                    graph, scheduler_state_base, scheduler, run_control, cluster_lane, mix_generation, wake_stats,
                    ready_stats, completion_stats, profiling_level, nullptr, nullptr, ssbuf_region, candidates, true,
                    lane_slot
                )) {
                return false;
            }
            progress = true;
        }
    }
    return progress;
}

// Take the dispatch the coordinator left in the slot this lane is not running,
// and flip the index onto it. That one write both promotes the pending
// dispatch and tells the coordinator the other slot is now the free one.
//
// Promotion runs ahead of every claim in the loop. That ordering is the whole
// of "a Mix subtask is not queued behind ordinary work": a core that could
// promote never reaches the point where it would claim for itself instead.
inline __aicore__ bool scheduler_promote_pending_dispatch(
    const SchedulerGraphView &graph, SchedulerLocalState *scheduler, __gm__ SchedulerRunControl *run_control,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (scheduler == nullptr || ssbuf_region == nullptr || !scheduler->has_mix) return false;
    const uint32_t lane = scheduler->config.self_lane;
    // Lane 0 is the coordinator: it seats its own Mix subtask directly and has
    // nothing to take from a peer.
    if (lane == 0 || lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    if (scheduler->running.state != SchedulerDispatchSlotState::FREE) return false;
    SCHEDULER_SSBUF SchedulerSsbufLane &ssbuf_lane = ssbuf_region->lanes[lane];
    const bool pending = scheduler_lane_pending_slot(ssbuf_region, lane);
    const uint64_t publication = scheduler_ssbuf_load_relaxed(&ssbuf_lane.dispatch[pending].publication);
    const uint32_t generation = static_cast<uint32_t>(publication);
    // The slot still holds whatever was published into it last time around, so
    // what makes a dispatch new is the generation, not the slot being filled.
    if (generation == 0 || generation == scheduler->consumed_dispatch_generations[pending]) return false;
    // The publication is the last word the coordinator wrote, so everything it
    // describes is in place by the time this read returns.
    scheduler_cache_barrier();
    const int64_t task_ref = ssbuf_lane.dispatch[pending].task_ref;
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
        );
        return false;
    }
    scheduler->running.task_ref = task_ref;
    scheduler->running.generation = generation;
    scheduler->running.state = SchedulerDispatchSlotState::READY;
    scheduler->running.from_mailbox = true;
    scheduler->running.lane_slot = pending;
    scheduler->consumed_dispatch_generations[pending] = generation;
    scheduler->set_timing_slot(lane, static_cast<int32_t>(publication >> 32));
    scheduler_local_ready_publish(scheduler);
    scheduler_ssbuf_store_relaxed(&ssbuf_lane.current, static_cast<uint64_t>(pending));
    return true;
}
