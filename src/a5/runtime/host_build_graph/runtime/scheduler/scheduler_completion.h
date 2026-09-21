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

inline __aicore__ void scheduler_account_failed_completion(
    const SchedulerGraphView &graph, SchedulerLocalState *scheduler, __gm__ SchedulerRunControl *run_control,
    int64_t task_id, SchedulerErrorSite error_site
) {
    // Preserve a more specific error already latched by the failing helper. The
    // fallback covers argument/invariant guards that only report false.
    scheduler_record_error(
        run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler, error_site
    );
    // AICPU may treat the completion count as the terminal graph token. Publish
    // the error first so this failed completion can never look like success.
    scheduler_cache_barrier();
    scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
}

inline __aicore__ bool scheduler_service_cluster_completion_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cluster_lane, uint32_t pending_slot,
    uint32_t completed_generation, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors, uint64_t profiling_level,
    const SchedulerReadyClaim *replacement_ready, bool *direct_refilled,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (direct_refilled != nullptr) *direct_refilled = false;
    if (ssbuf_region == nullptr || scheduler == nullptr || cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        pending_slot >= SCHEDULER_PENDING_SLOT_COUNT || completed_generation == 0)
        return false;
    // Callers prefilter consumed tokens. Keep this mutating helper fail-fast if
    // that precondition is violated, so a completion cannot be serviced twice.
    if (!scheduler_completion_generation_is_new(scheduler, cluster_lane, pending_slot, completed_generation))
        return false;
    const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
    if (worker_id >= scheduler->config.runtime_worker_count) return false;
    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    if (target->active == 0) return false;
    SchedulerLocalSlotState *local_slot = &scheduler->slots[cluster_lane][pending_slot];
    const int64_t published_task_id = local_slot->task_id;
    if (local_slot->state != SchedulerDispatchSlotState::READY || local_slot->generation != completed_generation) {
        scheduler_record_error(
            run_control, published_task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
        );
        return false;
    }

    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t completion_observe = schedule_timing_enabled ? scheduler_cycles() : 0;
    const int64_t task_id = local_slot->task_id;
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_TASK_ID, &graph, scheduler);
        return false;
    }
    const bool sampled_task_timing_enabled = local_slot->sampled_task_timing();
    SchedulerExecutorTaskTrace executor_trace{};
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        if (worker_id == scheduler->worker_id()) {
            executor_trace = scheduler->executor_traces[pending_slot];
        } else {
            const SCHEDULER_SSBUF volatile SchedulerExecutorTaskTrace *published_trace =
                &ssbuf_region->lanes[cluster_lane].traces[pending_slot].payload;
            executor_trace.kernel_start_cycles = published_trace->kernel_start_cycles;
            executor_trace.kernel_end_cycles = published_trace->kernel_end_cycles;
            executor_trace.ready_scan_start_cycles = published_trace->ready_scan_start_cycles;
            executor_trace.ready_observe_cycles = published_trace->ready_observe_cycles;
            executor_trace.completion_id = published_trace->completion_id;
            executor_trace.completion_inbox_index = published_trace->completion_inbox_index;
        }
    }
    __gm__ SchedulerTaskTrace *completed_trace = nullptr;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->config.trace_cells_offset);
        completed_trace = &traces[task_id];
    }
    if (chip_task_timing_enabled) {
        scheduler_observe_cache_line(completed_trace);
        scheduler_observe_cache_line(&completed_trace->kernel_start_cycles);
        completed_trace->kernel_start_cycles = executor_trace.kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace.kernel_end_cycles;
        completed_trace->ready_observe_cycles = executor_trace.ready_observe_cycles;
        if (schedule_timing_enabled) completed_trace->complete_start_cycles = completion_observe;
        if (phase_timing_enabled) {
            scheduler_observe_cache_line(&completed_trace->ready_transition_cycles);
            scheduler_observe_cache_line(&completed_trace->dispatch_start_cycles);
            scheduler_observe_cache_line(&completed_trace->refill_scheduler_worker_id);
            scheduler_observe_cache_line(&completed_trace->descriptor_cache_observed_cycles);
            completed_trace->ready_scan_start_cycles = executor_trace.ready_scan_start_cycles;
            completed_trace->completion_id = executor_trace.completion_id;
            completed_trace->completion_inbox_index = executor_trace.completion_inbox_index;
            SchedulerWorkerTraceCache *worker_trace = &scheduler->worker_traces[cluster_lane];
            if ((scheduler->worker_trace_valid_mask & (1U << cluster_lane)) == 0) {
                scheduler_observe_cache_line(&target->trace_aicore_entry_cycles);
                scheduler_observe_cache_line(&target->trace_register_release_cycles);
                worker_trace->descriptor_cache_observed_cycles = target->trace_descriptor_cache_observed_cycles;
                worker_trace->aicore_entry_cycles = target->trace_aicore_entry_cycles;
                worker_trace->handshake_publish_cycles = target->trace_handshake_publish_cycles;
                worker_trace->register_release_cycles = target->trace_register_release_cycles;
                scheduler->worker_trace_valid_mask |= 1U << cluster_lane;
            }
            completed_trace->descriptor_cache_observed_cycles = worker_trace->descriptor_cache_observed_cycles;
            completed_trace->aicore_entry_cycles = worker_trace->aicore_entry_cycles;
            completed_trace->handshake_publish_cycles = worker_trace->handshake_publish_cycles;
            completed_trace->register_release_cycles = worker_trace->register_release_cycles;
            completed_trace->complete_scheduler_worker_id = scheduler->worker_id();
            completed_trace->complete_loop_iter = scheduler->loop_iter;
        }
    } else if (completed_trace != nullptr) {
        completed_trace->kernel_start_cycles = executor_trace.kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace.kernel_end_cycles;
    }
    const uint8_t completed_subtask_slot = local_slot->subtask_slot;
    scheduler_mark_completion_consumed(scheduler, cluster_lane, pending_slot, completed_generation);
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    const uint64_t completion_phase_end = phase_timing_enabled ? scheduler_cycles() : 0;
    SchedulerReadyClaim resolved_ready{};
    SchedulerReadyClaim *direct_ready =
        replacement_ready == nullptr && worker_id != scheduler->worker_id() ? &resolved_ready : nullptr;
    if (!scheduler_resolve_completion(
            graph, scheduler_state_base, scheduler, run_control, task_id, wake_stats, ready_stats, completion_stats,
            profiling_level, false, direct_ready, scheduler_metadata_core_type_index(completed_subtask_slot)
        )) {
        scheduler_account_failed_completion(
            graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
        );
        return false;
    }
    SchedulerReadyClaim ready{};
    bool ready_available = replacement_ready != nullptr;
    if (ready_available) {
        ready = *replacement_ready;
    } else if (resolved_ready.task_id >= 0) {
        ready = resolved_ready;
        ready_available = true;
    } else if (ready_victim_cursors != nullptr && worker_id != scheduler->worker_id()) {
        // A normal AIV task is never refilled directly onto the Scheduler.
        // Its completed slot becomes capacity for late binding instead.
        const uint32_t core_type = scheduler_metadata_core_type_index(completed_subtask_slot);
        const uint64_t state_probe_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count, core_type,
                &ready_victim_cursors[core_type], ready_stats, &ready
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_REFILL_CLAIM_FAILED
            );
            return false;
        }
        ready.state_probe_start_cycles = state_probe_start_cycles;
        ready_available = ready.task_id >= 0;
        if (ready_available) {
            ready.state_probe_end_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
            refill_start_cycles = ready.state_probe_end_cycles;
        }
    }
    if (ready_available) {
        if (refill_start_cycles == 0) refill_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
        ready.publication_mode = SchedulerPublicationMode::REFILL;
        SchedulerFreeSlotClaim claim{
            worker_id,
            pending_slot,
            local_slot->generation,
            cluster_lane,
        };
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED
            );
            return false;
        }
        refilled = true;
    }
    refill_end_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
    if (refill_start_cycles == 0) refill_start_cycles = refill_end_cycles;
    if (!refilled) {
        local_slot->task_id = SCHEDULER_TASK_ID_INVALID;
        local_slot->state = SchedulerDispatchSlotState::FREE;
        local_slot->subtask_slot = UINT8_MAX;
        local_slot->timing_slot = -1;
    }
    if (chip_task_timing_enabled) {
        if (phase_timing_enabled) {
            completed_trace->complete_end_cycles = completion_phase_end;
        }
        if (phase_timing_enabled && refilled) {
            completed_trace->refill_scheduler_worker_id = scheduler->worker_id();
            completed_trace->refill_start_cycles = refill_start_cycles;
            completed_trace->refill_end_cycles = refill_end_cycles;
            completed_trace->refill_task_id = static_cast<uint64_t>(ready.task_id);
            completed_trace->refill_loop_iter = scheduler->loop_iter;
        }
        scheduler_writeback_cache_line(completed_trace);
        scheduler_writeback_cache_line(&completed_trace->kernel_start_cycles);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->ready_transition_cycles);
        if (schedule_timing_enabled) scheduler_writeback_cache_line(&completed_trace->dispatch_start_cycles);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->refill_scheduler_worker_id);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->descriptor_cache_observed_cycles);
        scheduler_cache_barrier();
        scheduler_gm_publish(completed_trace->valid, UINT64_C(1));
        // AICPU treats resolved_task_count as the graph-completion token. Keep
        // valid globally ordered before that token so host collection cannot
        // race the final trace publication.
        scheduler_cache_barrier();
    } else if (completed_trace != nullptr) {
        scheduler_publish_cache_line(&completed_trace->kernel_start_cycles);
    }
    scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
    if (direct_refilled != nullptr) *direct_refilled = refilled;
    return true;
}

inline __aicore__ bool scheduler_service_cluster_completions(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors, uint64_t profiling_level,
    uint64_t *direct_refilled_slot_mask, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (scheduler == nullptr || !scheduler->is_scheduler() || ssbuf_region == nullptr) return false;
    if (direct_refilled_slot_mask != nullptr) *direct_refilled_slot_mask = 0;
    bool progress = false;
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
        if (worker_id >= scheduler->config.runtime_worker_count) continue;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        if (target->active == 0) continue;
        const bool remote = cluster_lane != scheduler->config.self_lane;
        const uint64_t publication =
            remote ? scheduler_ssbuf_load_relaxed(&ssbuf_region->lanes[cluster_lane].completion.publication) : 0;
        bool acquired = false;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            const uint32_t completed_generation = !remote ? scheduler->local_completed_generations[pending_slot] :
                                                            static_cast<uint32_t>(publication >> (pending_slot * 32));
            if (!scheduler_completion_generation_is_new(scheduler, cluster_lane, pending_slot, completed_generation))
                continue;
            if (remote && !acquired) {
                scheduler_cache_barrier();
                acquired = true;
            }
            bool direct_refilled = false;
            if (!scheduler_service_cluster_completion_slot(
                    graph, scheduler_state_base, scheduler, run_control, cluster_lane, pending_slot,
                    completed_generation, wake_stats, ready_stats, completion_stats, ready_victim_cursors,
                    profiling_level, nullptr, &direct_refilled, ssbuf_region
                )) {
                return false;
            }
            if (direct_refilled && direct_refilled_slot_mask != nullptr)
                *direct_refilled_slot_mask |= UINT64_C(1)
                                              << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot);
            progress = true;
        }
    }
    return progress;
}
