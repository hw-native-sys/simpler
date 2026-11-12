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
static inline __aicore__ void
scheduler_flush_completions(__gm__ SchedulerRunControl *run_control, SchedulerLocalState *local) {
    const uint64_t completed = local->pending_completed;
    if (completed == 0) return;
    scheduler_cache_barrier();
    scheduler_gm_fetch_add_discard(run_control->resolved_task_count, completed);
    local->pending_completed = 0;
}

enum class SchedulerSubtaskFinishResult : uint8_t { PARTIAL, TASK_LAST, TASK_LAST_CHECK_ASYNC, ERROR };

// The caller owns this executing slot and has sealed asynchronous condition
// registration. Trace storage and slot resources remain owned until retirement
// returns. Inactive MIX resource lanes have no executed slot to contribute.
template <bool RecordProtocol = true>
static inline __aicore__ SchedulerSubtaskFinishResult scheduler_contribute_subtask(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerLocalSlotState &completed, uint32_t pending_slot
) {
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return SchedulerSubtaskFinishResult::ERROR;
    auto *control = scheduler_task_control_at(base, local, completed.task_ref);
    const bool admitted = scheduler_slot_uses_admission(local, completed);
    const bool block_owner = admitted && !completed.gm_mix;
    auto *admission = local->admission;
    const bool single_wave = admitted && scheduler_task_uses_single_wave_claim(
                                             admission->current_flags, admission->current_active_mask,
                                             admission->current_block_num, local->config.cluster_count
                                         );
    if (single_wave) {
        auto *record = scheduler_admission_record_at(base, admission, admission->current_task_ref);
        const auto *members = reinterpret_cast<__gm__ const uint64_t *>(record + 1);
        uint32_t assigned = UINT32_MAX;
        if (!scheduler_single_wave_assignment(
                admission->current_flags, admission->current_active_mask, admission->current_block_num,
                local->config.cluster_count, members, local->config.runtime_worker_count, local->worker_id(),
                admission->gm_mix.cluster_index, &assigned
            ) ||
            assigned != completed.block_idx) {
            scheduler_record_error(
                run, completed.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::BLOCK_COMPLETION_INVALID, &completed
            );
            return SchedulerSubtaskFinishResult::ERROR;
        }
    }
    if (single_wave && block_owner &&
        (admission->single_wave_claim_confirmed == 0 ||
         admission->single_wave_claim_confirmed > admission->current_block_num ||
         admission->single_wave_claim_confirmed != static_cast<uint64_t>(completed.block_idx) + 1 ||
         admission->single_wave_retired)) {
        scheduler_record_error(
            run, completed.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::BLOCK_COMPLETION_INVALID, &completed
        );
        return SchedulerSubtaskFinishResult::ERROR;
    }
    if constexpr (RecordProtocol) {
        scheduler_protocol_subtask_complete(base, local, completed, pending_slot, true, UINT64_MAX);
        if (block_owner) scheduler_protocol_block_complete(base, local, completed.block_idx, true, UINT64_MAX);
    }
    scheduler_cache_barrier();
    // Only the executing core contributes, after sealing its registration.
    // Any registered condition has already activated that core's local flag.
    const bool check_async = local->async != nullptr && local->async->work_activated;
    const uint64_t contribution = UINT64_C(1) + (check_async ? SCHEDULER_COMPLETION_ASYNC_UNIT : 0);
    const uint64_t previous_state = scheduler_gm_fetch_add(control->completion_state, contribution);
    const uint64_t previous = scheduler_completion_count(previous_state);
    const uint64_t total = scheduler_completion_total(previous_state);
    if constexpr (RecordProtocol) {
        scheduler_protocol_subtask_complete(base, local, completed, pending_slot, false, previous);
        if (block_owner) scheduler_protocol_block_complete(base, local, completed.block_idx, false, previous);
    }
    if (previous >= total) {
        scheduler_record_error(
            run, completed.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::BLOCK_COMPLETION_INVALID, &completed
        );
        return SchedulerSubtaskFinishResult::ERROR;
    }
    // Full waves have no shared claim cursor. Each physical contributor
    // validates its assigned identity before counting; the existing packed
    // completion total still selects the last real contributor and async gate.
    if (single_wave && block_owner) admission->single_wave_retired = true;
    if (previous + 1 != total) return SchedulerSubtaskFinishResult::PARTIAL;
    return ((previous_state | contribution) >> 32) != 0 ? SchedulerSubtaskFinishResult::TASK_LAST_CHECK_ASYNC :
                                                          SchedulerSubtaskFinishResult::TASK_LAST;
}

// The unique last computation contributor owns logical completion. Trace
// publication and Mix resource release finish before the owning core can
// acknowledge EXIT; successors may run independently of those records.
// One dependency-ready cooperative successor may be admitted by this core
// regardless of its execution type; excess successors retain queue ownership.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_complete_task(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, bool check_async, uint64_t profiling_level,
    SchedulerReadyClaim *direct_ready
) {
    // The unique last subtask owns logical completion; its retirement already
    // validated the live task and the non-null run/local pointers.
    if (check_async) {
        bool deferred = false;
        if (!scheduler_async_close_computation(scheduler->async, task_ref, deferred)) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::ASYNC_CONTEXT_INVALID
            );
            return false;
        }
        if (deferred) return true;
    }
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, scheduler, task_ref);
    if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, true);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    if constexpr (RecordProtocol) scheduler_protocol_task_done(scheduler_state_base, scheduler, task_ref, false);
    auto *direct = direct_ready;
    if (direct == nullptr && scheduler->admission != nullptr)
        direct = scheduler_ordered_direct_sink(scheduler->admission);
    const bool resolved = scheduler_resolve_completion(
        graph, scheduler_state_base, scheduler, run_control, task_ref, profiling_level, false, direct,
        scheduler->config.self_core_type, true
    );
    // DONE contributes once even when successor resolution reports an error.
    ++scheduler->pending_completed;
    if (!resolved) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_RESOLVE_FAILED
        );
        return false;
    }
    if (direct != nullptr && !scheduler_task_ref_is_null(direct->task_ref) &&
        scheduler_task_requires_admission(
            scheduler_task_metadata_at(scheduler_state_base, scheduler, direct->task_ref)->flags
        )) {
        const auto claimed = scheduler_try_claim_admission(
            graph, scheduler_state_base, scheduler, run_control, scheduler->admission, *direct
        );
        if (claimed == SchedulerAdmissionResult::ERROR) return false;
        if (claimed == SchedulerAdmissionResult::PROGRESS) {
            // Pending admission owns the task even if the commit lock is busy.
            // Clear the candidate only after that ownership transfer succeeds.
            *direct = {};
            if (scheduler_ordered_publish_pending<RecordProtocol>(
                    graph, scheduler_state_base, scheduler, run_control
                ) == SchedulerAdmissionResult::ERROR)
                return false;
        }
    }
    return true;
}

// FromExecutor retains the resident's exclusive ownership of a dispatched slot:
// kernels receive only GM payload/argument contexts, never its private binding.
// The checked entry also serves tests and callers that do not hold that proof.
template <bool RecordProtocol = true, bool FromExecutor = false>
static inline __aicore__ bool scheduler_complete_own_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t pending_slot, uint64_t profiling_level
) {
    if constexpr (!FromExecutor) {
        if (scheduler == nullptr || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT ||
            scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
            return false;
    }
    const uint64_t worker_id = scheduler->worker_id();
    if constexpr (!FromExecutor)
        if (worker_id >= scheduler->config.runtime_worker_count) return false;
    auto *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
    auto *local_slot = &scheduler->slots[pending_slot];
    const int64_t task_ref = local_slot->task_ref;
    if constexpr (!FromExecutor) {
        if (local_slot->state != SchedulerDispatchSlotState::EXECUTING ||
            (scheduler->local_ready_mask & (1U << pending_slot)) != 0) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, local_slot
            );
            return false;
        }
        if (!scheduler_task_ref_valid(graph, task_ref)) {
            scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, scheduler);
            return false;
        }
        const auto &metadata = *scheduler_task_metadata_at(scheduler_state_base, scheduler, task_ref);
        const bool admitted = scheduler_slot_uses_admission(scheduler, *local_slot);
        if (local_slot->subtask_slot >= 3 || local_slot->block_idx >= metadata.logical_block_num ||
            !scheduler_task_lanes(metadata).has(static_cast<SubtaskSlot>(local_slot->subtask_slot)) ||
            scheduler_metadata_core_type_index(local_slot->subtask_slot) != scheduler->config.self_core_type ||
            (admitted && scheduler->admission->current_task_ref != task_ref) ||
            (local_slot->gm_mix && (!admitted || scheduler->admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING ||
                                    scheduler->admission->gm_mix.slot_index != pending_slot ||
                                    scheduler->admission->gm_mix.block_idx != local_slot->block_idx ||
                                    scheduler->admission->gm_mix.subtask_slot != local_slot->subtask_slot ||
                                    scheduler->admission->gm_mix.group_seq == 0))) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, local_slot
            );
            return false;
        }
    }
    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t completion_observe = schedule_timing_enabled ? scheduler_cycles() : 0;
    const bool sampled_task_timing_enabled = scheduler->sampled_task_timing(pending_slot);
    const uint8_t completed_subtask_slot = local_slot->subtask_slot;
    const uint32_t completed_block_idx = local_slot->block_idx;
    const bool gm_mix = local_slot->gm_mix;
    const SchedulerLocalSlotState completed_slot = *local_slot;
    if constexpr (RecordProtocol)
        scheduler_protocol_execution_consumed(scheduler_state_base, scheduler, pending_slot, *local_slot);
    uint64_t refill_start_cycles = 0;
    uint64_t refill_end_cycles = 0;
    bool refilled = false;
    scheduler_initialize_free_slot(local_slot);
    scheduler->set_timing_slot(pending_slot, -1);
    const uint64_t completion_phase_end = phase_timing_enabled ? scheduler_cycles() : 0;
    const auto contribution = scheduler_contribute_subtask<RecordProtocol>(
        graph, scheduler_state_base, scheduler, run_control, completed_slot, pending_slot
    );
    bool completion_ok = contribution != SchedulerSubtaskFinishResult::ERROR;
    SchedulerReadyClaim resolved_ready{};
    if (contribution == SchedulerSubtaskFinishResult::TASK_LAST ||
        contribution == SchedulerSubtaskFinishResult::TASK_LAST_CHECK_ASYNC) {
        auto *direct =
            scheduler->admission != nullptr ? scheduler_ordered_direct_sink(scheduler->admission) : &resolved_ready;
        completion_ok = scheduler_complete_task<RecordProtocol>(
            graph, scheduler_state_base, scheduler, run_control, task_ref,
            contribution == SchedulerSubtaskFinishResult::TASK_LAST_CHECK_ASYNC, profiling_level, direct
        );
    }
    // This core is the sole writer. Rebinding the slot happens after all trace
    // reads below, so retirement can read the executor's record in place.
    const SchedulerExecutorTaskTrace *executor_trace = chip_task_timing_enabled || sampled_task_timing_enabled ?
                                                           &scheduler->profiling->executor_traces[pending_slot] :
                                                           nullptr;
    __gm__ SchedulerTaskTrace *completed_trace = nullptr;
    if (chip_task_timing_enabled || sampled_task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->profiling->trace_cells_offset);
        completed_trace = &traces[scheduler_trace_index_at(
            scheduler_state_base, scheduler, task_ref, static_cast<SubtaskSlot>(completed_slot.subtask_slot),
            completed_slot.block_idx
        )];
    }
    if (chip_task_timing_enabled) {
        // An own slot's dispatch, execution identity and completion trace have
        // the same sole writer.
        if (phase_timing_enabled && !scheduler->profiling->worker_trace_valid) {
            scheduler_observe_cache_line(&target->trace_aicore_entry_cycles);
        }
        completed_trace->kernel_start_cycles = executor_trace->kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace->kernel_end_cycles;
        completed_trace->ready_observe_cycles = executor_trace->ready_observe_cycles;
        if (schedule_timing_enabled) completed_trace->complete_start_cycles = completion_observe;
        if (phase_timing_enabled) {
            completed_trace->ready_scan_start_cycles = executor_trace->ready_scan_start_cycles;
            completed_trace->completion_id = executor_trace->completion_id;
            completed_trace->completion_inbox_index = executor_trace->completion_inbox_index;
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
        completed_trace->kernel_start_cycles = executor_trace->kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace->kernel_end_cycles;
    }
    if (chip_task_timing_enabled) {
        if (phase_timing_enabled) {
            completed_trace->complete_end_cycles = completion_phase_end;
        }
        // Dispatch already published the first line; completion only updates
        // its valid word through the atomic wrapper below.
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
        // Trace lines precede valid. The owning core finishes this retirement
        // before acknowledging EXIT; Host reads traces after all acknowledgements.
    } else if (completed_trace != nullptr) {
        scheduler_publish_cache_line(&completed_trace->kernel_start_cycles);
    }
    if (gm_mix && !scheduler_gm_mix_release_lane<RecordProtocol>(
                      graph, scheduler_state_base, scheduler, run_control, completed_slot, pending_slot
                  ))
        return false;
    if (!completion_ok) return false;
    if (scheduler->admission != nullptr) {
        auto &hint = *scheduler->admission;
        hint.advance_producer = task_ref;
        hint.advance_index = 0;
        hint.advance_task = SCHEDULER_TASK_REF_INVALID;
    }

    SchedulerReadyClaim ready = resolved_ready;
    if (!scheduler_task_ref_is_null(ready.task_ref)) {
        if (refill_start_cycles == 0) refill_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
        ready.publication_mode = SchedulerPublicationMode::REFILL;
        if (scheduler->admission != nullptr) {
            if (!scheduler_ordered_retain_candidate(scheduler->admission, ready)) {
                scheduler_record_error(
                    run_control, ready.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
                );
                return false;
            }
        } else {
            const SchedulerFreeSlotClaim claim{pending_slot};
            local_slot->state = SchedulerDispatchSlotState::FILLING;
            if (!scheduler_fill_dispatch_slot(
                    graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level
                )) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED
                );
                return false;
            }
            refilled = local_slot->state == SchedulerDispatchSlotState::READY;
        }
    }
    refill_end_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
    if (phase_timing_enabled && refilled) {
        completed_trace->refill_scheduler_worker_id = scheduler->worker_id();
        completed_trace->refill_start_cycles = refill_start_cycles;
        completed_trace->refill_end_cycles = refill_end_cycles;
        completed_trace->refill_task_ref = static_cast<uint64_t>(ready.task_ref);
        completed_trace->refill_loop_iter = scheduler->profiling->loop_iter;
        scheduler_publish_cache_line(&completed_trace->refill_scheduler_worker_id);
    }
    return true;
}
