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
    const SchedulerGraphView &graph, __gm__ SchedulerWorkerContext *scheduler, __gm__ SchedulerRunControl *run_control,
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
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t cluster_lane, uint32_t pending_slot,
    uint32_t completed_generation, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors, uint64_t profiling_level,
    const SchedulerReadyClaim *replacement_ready, bool *direct_refilled,
    SchedulerCompletionServiceTiming *timing = nullptr, __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (direct_refilled != nullptr) *direct_refilled = false;
    if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT ||
        completed_generation == 0)
        return false;
    const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
    if (worker_id >= scheduler->runtime_worker_count) return false;
    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    if (target->active == 0) return false;
    __gm__ SchedulerCompletionInbox *completion_line =
        scheduler_completion_inbox_at(scheduler_state_base, scheduler, worker_id);
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
    const uint64_t publication = scheduler_gm_query(slot->publication);
    if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::READY ||
        scheduler_dispatch_generation(publication) != completed_generation) {
        scheduler_record_error(
            run_control, slot->task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
        );
        return false;
    }

    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const bool record_timeline = phase_timing_enabled;
    const uint64_t completion_start = schedule_timing_enabled ? scheduler_cycles() : 0;
    uint64_t operation_start = record_timeline ? completion_start : 0;
    scheduler_observe_cache_line(slot);
    const int64_t task_id = slot->task_id;
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_TASK_ID, &graph, scheduler);
        return false;
    }
    if (slot->gang != 0) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_UNEXPECTED_GANG_SLOT
        );
        return false;
    }
    __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, scheduler, task_id);
    scheduler_observe_cache_line(metadata);
    const bool sampled_task_timing_enabled =
        metadata->timing_slot >= 0 && metadata->timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
    __gm__ SchedulerExecutorTaskTrace *executor_trace = &slot->executor_trace;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        if (scheduler_gm_query(executor_trace->generation) != completed_generation) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH
            );
            return false;
        }
        scheduler_observe_cache_line(executor_trace);
        scheduler_observe_cache_line(&executor_trace->completion_inbox_index);
    }
    __gm__ SchedulerTaskTrace *completed_trace = nullptr;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->trace_cells_offset);
        completed_trace = &traces[task_id];
    }
    if (chip_task_timing_enabled) {
        scheduler_observe_cache_line(completed_trace);
        scheduler_observe_cache_line(&completed_trace->kernel_start_cycles);
        completed_trace->kernel_start_cycles = executor_trace->kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace->kernel_end_cycles;
        completed_trace->ready_observe_cycles = executor_trace->ready_observe_cycles;
        if (schedule_timing_enabled) completed_trace->complete_start_cycles = completion_start;
        if (phase_timing_enabled) {
            scheduler_observe_cache_line(&completed_trace->ready_transition_cycles);
            scheduler_observe_cache_line(&completed_trace->dispatch_start_cycles);
            scheduler_observe_cache_line(&completed_trace->refill_scheduler_worker_id);
            scheduler_observe_cache_line(&completed_trace->descriptor_cache_observed_cycles);
            completed_trace->completion_end_cycles = executor_trace->completion_end_cycles;
            completed_trace->ready_scan_start_cycles = executor_trace->ready_scan_start_cycles;
            completed_trace->completion_bookkeeping_end_cycles = executor_trace->completion_bookkeeping_end_cycles;
            completed_trace->completion_id = executor_trace->completion_id;
            completed_trace->completion_inbox_index = executor_trace->completion_inbox_index;
            scheduler_observe_cache_line(&target->trace_aicore_entry_cycles);
            scheduler_observe_cache_line(&target->trace_register_release_cycles);
            completed_trace->descriptor_cache_observed_cycles = target->trace_descriptor_cache_observed_cycles;
            completed_trace->aicore_entry_cycles = target->trace_aicore_entry_cycles;
            completed_trace->handshake_publish_cycles = target->trace_handshake_publish_cycles;
            completed_trace->register_release_cycles = target->trace_register_release_cycles;
            completed_trace->complete_scheduler_worker_id = scheduler->worker_index;
            completed_trace->complete_loop_iter = scheduler->profiling_loop_iter;
        }
    } else if (completed_trace != nullptr) {
        completed_trace->kernel_start_cycles = executor_trace->kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace->kernel_end_cycles;
    }
    const uint8_t completed_subtask_slot = slot->subtask_slot;
    scheduler_gm_store(completion_line->completed_generations[pending_slot], UINT32_C(0));
    uint64_t operation_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->consume_cycles += operation_end - operation_start;
    operation_start = operation_end;
    uint64_t ready_publish_cycles = 0;
    uint64_t refill_cycles = 0;
    uint64_t finalize_cycles = 0;
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    if (!scheduler_resolve_completion(
            graph, scheduler_state_base, scheduler, run_control, task_id, wake_stats, ready_stats, completion_stats,
            owner_state, profiling_level, false, timing == nullptr ? nullptr : &ready_publish_cycles
        )) {
        scheduler_account_failed_completion(
            graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
        );
        return false;
    }
    if (timing != nullptr) timing->ready_publish_cycles += ready_publish_cycles;
    refill_start_cycles = record_timeline ? scheduler_cycles() : 0;
    SchedulerReadyClaim ready{};
    bool ready_available = replacement_ready != nullptr;
    if (ready_available) {
        ready = *replacement_ready;
    } else if (ready_victim_cursors != nullptr && worker_id != scheduler->worker_index) {
        // A normal AIV task is never refilled directly onto the Scheduler.
        // Its completed slot becomes capacity for late binding instead.
        const uint32_t core_type = scheduler_metadata_core_type_index(completed_subtask_slot);
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->scheduler_count, core_type,
                &ready_victim_cursors[core_type], ready_stats, &ready, owner_state, profiling_level
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_REFILL_CLAIM_FAILED
            );
            return false;
        }
        ready_available = ready.task_id >= 0;
    }
    if (ready_available) {
        SchedulerFreeSlotClaim claim{worker_id, pending_slot, slot->generation};
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_id, SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED
            );
            return false;
        }
        refilled = true;
    }
    refill_end_cycles = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) {
        refill_cycles = refill_end_cycles - refill_start_cycles;
        timing->refill_cycles += refill_cycles;
    }
    operation_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) {
        uint64_t resolve_total = operation_end - operation_start;
        uint64_t excluded = ready_publish_cycles + refill_cycles + finalize_cycles;
        timing->resolve_cycles += resolve_total > excluded ? resolve_total - excluded : 0;
    }
    operation_start = operation_end;
    if (!refilled) {
        slot->task_id = SCHEDULER_TASK_ID_INVALID;
        scheduler_writeback_cache_line(slot);
        scheduler_gm_store(
            slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::FREE)
        );
    }
    const uint64_t completion_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->finalize_cycles += completion_end - operation_start;
    if (chip_task_timing_enabled) {
        if (phase_timing_enabled) {
            completed_trace->complete_end_cycles = completion_end;
        }
        if (phase_timing_enabled && refilled) {
            completed_trace->refill_scheduler_worker_id = scheduler->worker_index;
            completed_trace->refill_start_cycles = refill_start_cycles;
            completed_trace->refill_end_cycles = refill_end_cycles;
            completed_trace->refill_task_id = static_cast<uint64_t>(ready.task_id);
            completed_trace->refill_loop_iter = scheduler->profiling_loop_iter;
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
    const uint64_t resolved_count_start = timing == nullptr ? 0 : scheduler_cycles();
    scheduler_gm_fetch_add(run_control->resolved_task_count, UINT64_C(1));
    if (timing != nullptr) {
        finalize_cycles = scheduler_cycles() - resolved_count_start;
        timing->finalize_cycles += finalize_cycles;
    }
    if (direct_refilled != nullptr) *direct_refilled = refilled;
    return true;
}

inline __aicore__ uint32_t scheduler_completion_catchup_mask(uint32_t initial_completion_mask) {
    constexpr uint32_t all_pending_slots_mask = (1U << SCHEDULER_PENDING_SLOT_COUNT) - 1;
    return initial_completion_mask == 0 ? 0 : all_pending_slots_mask & ~initial_completion_mask;
}

inline __aicore__ bool scheduler_service_cluster_completions(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, uint64_t *ready_victim_cursors = nullptr, uint64_t profiling_level = 0,
    uint64_t *direct_refilled_slot_mask = nullptr, SchedulerCompletionServiceTiming *timing = nullptr,
    __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (scheduler->is_scheduler == 0) return false;
    if (direct_refilled_slot_mask != nullptr) *direct_refilled_slot_mask = 0;
    bool progress = false;
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
        if (worker_id >= scheduler->runtime_worker_count) continue;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        if (target->active == 0) continue;
        __gm__ SchedulerCompletionInbox *completion_line =
            scheduler_completion_inbox_at(scheduler_state_base, scheduler, worker_id);
        uint64_t completed_generations = scheduler_gm_query_u32_pair(completion_line->completed_generations);
        uint32_t initial_completion_mask = 0;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            if (static_cast<uint32_t>(completed_generations >> (pending_slot * 32)) != 0)
                initial_completion_mask |= 1U << pending_slot;
        }
        uint32_t scan_mask = initial_completion_mask;
        // Completion processing can be much slower than the sibling kernel. Refresh the packed completion line once
        // for slots that were incomplete in the initial scan, excluding newly refilled slots from the catch-up pass.
        for (uint32_t scan_pass = 0; scan_pass < 2 && scan_mask != 0; ++scan_pass) {
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((scan_mask & (1U << pending_slot)) == 0) continue;
                const uint32_t completed_generation =
                    static_cast<uint32_t>(completed_generations >> (pending_slot * 32));
                if (completed_generation == 0) continue;
                bool direct_refilled = false;
                if (!scheduler_service_cluster_completion_slot(
                        graph, scheduler_state_base, scheduler, run_control, cluster_lane, pending_slot,
                        completed_generation, wake_stats, ready_stats, completion_stats, ready_victim_cursors,
                        profiling_level, nullptr, &direct_refilled, timing, owner_state
                    ))
                    return false;
                if (direct_refilled && direct_refilled_slot_mask != nullptr)
                    *direct_refilled_slot_mask |= UINT64_C(1)
                                                  << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot);
                progress = true;
            }
            scan_mask = scan_pass == 0 ? scheduler_completion_catchup_mask(initial_completion_mask) : 0;
            if (scan_mask != 0)
                completed_generations = scheduler_gm_query_u32_pair(completion_line->completed_generations);
        }
    }
    return progress;
}
