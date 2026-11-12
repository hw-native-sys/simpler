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

#include "scheduler_gm_mix.h"

// Idle and the run epilogue flush pending counts; native runs also publish
// bounded progress batches. Trace/error publication precedes the exit count.
inline __aicore__ void
scheduler_flush_completions(__gm__ SchedulerRunControl *run_control, SchedulerLocalState *local) {
    const uint64_t completed = local->pending_completed;
    if (completed == 0) return;
    scheduler_cache_barrier();
    scheduler_gm_fetch_add_discard(run_control->resolved_task_count, completed);
    local->pending_completed = 0;
}

inline __aicore__ void scheduler_account_failed_completion(
    const SchedulerGraphAccess &graph, SchedulerLocalState *scheduler, __gm__ SchedulerRunControl *run_control,
    int64_t task_ref, SchedulerErrorSite error_site
) {
    // Preserve a more specific error already latched by the failing helper. The
    // fallback covers argument/invariant guards that only report false.
    scheduler_record_error(
        run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler, error_site
    );
    ++scheduler->pending_completed;
}

// The caller's execution trace, or all released Mix lane traces, precedes
// this logical completion. Mix successors enter the ready queue; own-type
// successors may retain one persistent admission candidate.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_complete_ordered_task(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, uint64_t profiling_level
) {
    if (scheduler == nullptr || scheduler->admission == nullptr || !scheduler_task_ref_valid(graph, task_ref))
        return false;
    bool deferred = false;
    if (!scheduler_async_close_computation(scheduler->async, task_ref, deferred)) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::ASYNC_CONTEXT_INVALID
        );
        return false;
    }
    if (deferred) return true;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_ref);
    if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, true);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, false);
    if (!scheduler_resolve_completion(
            graph, scheduler_state_base, scheduler, run_control, task_ref, profiling_level, false,
            scheduler_ordered_direct_sink(scheduler->admission), scheduler->config.self_core_type
        )) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
        );
        return false;
    }
    ++scheduler->pending_completed;
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_complete_own_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t pending_slot, uint64_t profiling_level
) {
    if (scheduler == nullptr || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT ||
        scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    const uint64_t worker_id = scheduler->worker_id();
    if (worker_id >= scheduler->config.runtime_worker_count) return false;
    __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    SchedulerLocalSlotState *local_slot = &scheduler->slots[pending_slot];
    if (local_slot->state != SchedulerDispatchSlotState::EXECUTING ||
        (scheduler->local_ready_mask & (1U << pending_slot)) != 0) {
        scheduler_record_error(
            run_control, local_slot->task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, local_slot
        );
        return false;
    }
    const SchedulerLocalSlotState completed = *local_slot;
    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t completion_observe = schedule_timing_enabled ? scheduler_cycles() : 0;
    const int64_t task_ref = local_slot->task_ref;
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, scheduler);
        return false;
    }
    const auto &metadata = *scheduler_task_metadata_at(scheduler_state_base, scheduler, task_ref);
    if (completed.subtask_slot >= 3 || completed.block_idx >= metadata.logical_block_num ||
        !scheduler_task_lanes(metadata).has(static_cast<SubtaskSlot>(completed.subtask_slot)) ||
        scheduler_metadata_core_type_index(completed.subtask_slot) != scheduler->config.self_core_type ||
        (completed.cooperative &&
         (scheduler->admission == nullptr || scheduler->admission->current_task_ref != task_ref)) ||
        (completed.gm_mix &&
         (!completed.cooperative || scheduler->admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING ||
          scheduler->admission->gm_mix.slot_index != pending_slot ||
          scheduler->admission->gm_mix.block_idx != completed.block_idx ||
          scheduler->admission->gm_mix.subtask_slot != completed.subtask_slot ||
          scheduler->admission->gm_mix.group_seq == 0))) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, &completed
        );
        return false;
    }
    const bool sampled_task_timing_enabled = scheduler->sampled_task_timing(pending_slot);
    SchedulerExecutorTaskTrace executor_trace{};
    if (chip_task_timing_enabled || sampled_task_timing_enabled)
        executor_trace = scheduler->profiling->executor_traces[pending_slot];
    __gm__ SchedulerTaskTrace *completed_trace = nullptr;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->profiling->trace_cells_offset);
        completed_trace = &traces[scheduler_trace_index_at(
            scheduler_state_base, scheduler, task_ref, static_cast<SubtaskSlot>(local_slot->subtask_slot),
            local_slot->block_idx
        )];
    }
    if (chip_task_timing_enabled) {
        // Finish all trace invalidations before any cached trace write.
        scheduler_invalidate_cache_line(completed_trace);
        scheduler_invalidate_cache_line(&completed_trace->kernel_start_cycles);
        if (phase_timing_enabled) {
            scheduler_invalidate_cache_line(&completed_trace->dispatch_start_cycles);
            scheduler_invalidate_cache_line(&completed_trace->refill_scheduler_worker_id);
            scheduler_invalidate_cache_line(&completed_trace->descriptor_cache_observed_cycles);
            if (!scheduler->profiling->worker_trace_valid) {
                scheduler_invalidate_cache_line(&target->trace_aicore_entry_cycles);
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
            SchedulerWorkerTraceCache *worker_trace = &scheduler->profiling->worker_trace;
            if (!scheduler->profiling->worker_trace_valid) {
                worker_trace->descriptor_cache_observed_cycles = target->trace_descriptor_cache_observed_cycles;
                worker_trace->aicore_entry_cycles = target->trace_aicore_entry_cycles;
                worker_trace->handshake_publish_cycles = target->trace_handshake_publish_cycles;
                worker_trace->register_release_cycles = target->trace_register_release_cycles;
                scheduler->profiling->worker_trace_valid = true;
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
    const uint32_t completed_block_idx = local_slot->block_idx;
    const bool gm_mix = local_slot->gm_mix;
    const bool ordered_block = completed.cooperative && !gm_mix;
    bool task_complete = !gm_mix && !ordered_block;
    if constexpr (RecordProtocol)
        scheduler_protocol_execution_consumed(scheduler_state_base, scheduler, pending_slot, completed);
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    if (!ordered_block && !gm_mix) scheduler_initialize_free_slot(local_slot);
    scheduler->set_timing_slot(pending_slot, -1);
    const uint64_t completion_phase_end = phase_timing_enabled ? scheduler_cycles() : 0;
    SchedulerReadyClaim resolved_ready{};
    bool deferred = false;
    if (task_complete && scheduler->admission == nullptr &&
        !scheduler_async_has_registered_groups(scheduler->async, task_ref, deferred)) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::ASYNC_CONTEXT_INVALID
        );
        return false;
    }
    if (task_complete && !deferred && scheduler->admission == nullptr) {
        __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_ref);
        if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, true);
        scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, false);
        if (!scheduler_resolve_completion(
                graph, scheduler_state_base, scheduler, run_control, task_ref, profiling_level, false, &resolved_ready,
                scheduler_metadata_core_type_index(completed_subtask_slot)
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_ref, SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
            );
            return false;
        }
    }
    SchedulerReadyClaim ready = resolved_ready;
    if (!scheduler_task_ref_is_null(ready.task_ref)) {
        if (refill_start_cycles == 0) refill_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
        ready.publication_mode = SchedulerPublicationMode::REFILL;
        const SchedulerFreeSlotClaim claim{pending_slot};
        local_slot->state = SchedulerDispatchSlotState::FILLING;
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level
            )) {
            scheduler_account_failed_completion(
                graph, scheduler, run_control, task_ref, SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED
            );
            return false;
        }
        refilled = local_slot->state == SchedulerDispatchSlotState::READY;
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
        if (schedule_timing_enabled) scheduler_writeback_cache_line(&completed_trace->dispatch_start_cycles);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->refill_scheduler_worker_id);
        if (phase_timing_enabled) scheduler_writeback_cache_line(&completed_trace->descriptor_cache_observed_cycles);
        scheduler_cache_barrier();
        scheduler_gm_store(completed_trace->valid, UINT64_C(1));
        if constexpr (RecordProtocol)
            scheduler_protocol_execution_trace_published(
                scheduler_state_base, scheduler, task_ref, completed_block_idx, completed_subtask_slot, pending_slot
            );
        // The preceding barrier publishes trace lines before valid. The store
        // drains valid before pending_completed reaches resolved_task_count.
    } else if (completed_trace != nullptr) {
        scheduler_publish_cache_line(&completed_trace->kernel_start_cycles);
    }
    if (ordered_block) {
        // Every block publishes its final trace and frees its slot before the
        // shared join can authorize another core to finish the logical task.
        const auto result = scheduler_finish_ordered_block<RecordProtocol>(
            graph, scheduler_state_base, scheduler, run_control, pending_slot
        );
        if (result == SchedulerBlockFinishResult::ERROR) return false;
        task_complete = result == SchedulerBlockFinishResult::TASK_LAST;
    }
    if (gm_mix) {
        scheduler_initialize_free_slot(local_slot);
        if (!scheduler_gm_mix_release_lane<RecordProtocol>(
                graph, scheduler_state_base, scheduler, run_control, completed, pending_slot
            ))
            return false;
    } else {
        if (task_complete) {
            if (scheduler->admission != nullptr) {
                if (!scheduler_complete_ordered_task<RecordProtocol>(
                        graph, scheduler_state_base, scheduler, run_control, task_ref, profiling_level
                    ))
                    return false;
            } else if (deferred) {
                bool held = false;
                if (!scheduler_async_close_computation(scheduler->async, task_ref, held) || !held) {
                    scheduler_record_error(
                        run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                        SchedulerErrorSite::ASYNC_BLOCK_RELEASE_FAILED
                    );
                    return false;
                }
            } else {
                ++scheduler->pending_completed;
            }
        }
    }
    return true;
}
