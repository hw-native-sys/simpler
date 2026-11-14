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

// AICore-resident completion and cancellation.

#include "scheduler_gm_mix.h"
#include "scheduler_cluster_mix.h"

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

enum class SchedulerSubtaskFinishResult : uint8_t { PARTIAL, TASK_LAST, TASK_LAST_DEFERRED, ERROR };

// Cancellation owns only an unstarted local handle. Armed groups retire at
// the failed run's all-core fence before their storage can be reused.
static inline __aicore__ bool
scheduler_cancel_unstarted_handle(__gm__ void *base, SchedulerLocalState *local, SchedulerPreparedHandle *handle) {
    const auto *prepared = local->config.prepared;
    if (prepared == nullptr || handle->execution_index >= prepared->execution_count ||
        scheduler_prepared_owner(handle->claimed_state) != local->worker_id())
        return false;
    const auto phase = scheduler_prepared_phase(handle->claimed_state);
    if (phase != SchedulerPreparedPhase::CLAIMED && phase != SchedulerPreparedPhase::GROUP_RESERVED &&
        phase != SchedulerPreparedPhase::FAILED)
        return false;
    if (phase != SchedulerPreparedPhase::FAILED) {
        auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
        const uint64_t failed =
            (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::FAILED);
        if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, failed) != handle->claimed_state)
            return false;
    }
    *handle = {};
    return true;
}

// The explicit pre-call failure sites may cancel EXECUTING: that local state
// reserves the slot before begin/args-dump, before any business Call runs.
template <bool BeforeKernel = false>
static inline __aicore__ bool
scheduler_cancel_unstarted_slot(__gm__ void *base, SchedulerLocalState *local, uint32_t index) {
    if (index >= SCHEDULER_PENDING_SLOT_COUNT) return false;
    auto &slot = local->slots[index];
    if (slot.state == SchedulerDispatchSlotState::FREE || slot.state == SchedulerDispatchSlotState::EMPTY) return true;
    if (!BeforeKernel && slot.state == SchedulerDispatchSlotState::EXECUTING) return false;
    if (!scheduler_cancel_unstarted_handle(base, local, &slot.prepared)) return false;
    scheduler_initialize_free_slot(&slot);
    local->local_ready_mask &= ~(UINT32_C(1) << index);
    local->set_timing_slot(index, -1);
    return true;
}

// Cold abort cleanup never manufactures a return, C/D contribution or group
// finished ACK. There are no remote claims and no scan of the shared catalog.
static inline __aicore__ bool scheduler_cancel_unstarted_work(__gm__ void *base, SchedulerLocalState *local) {
    bool cancelled = true;
    for (auto &token : local->execution_tokens) {
        if (token.phase == SchedulerExecutionTokenPhase::EMPTY) continue;
        bool released = false;
        if (token.phase == SchedulerExecutionTokenPhase::TICKETED && token.prepared.claimed_state == 0 &&
            local->config.prepared != nullptr &&
            token.prepared.execution_index < local->config.prepared->execution_count) {
            // The successful FAA uniquely owns this fresh block even if fanin
            // decoding failed before its BUILT -> CLAIMED CAS.
            auto *control = scheduler_prepared_control_at(base, local->config.prepared, token.prepared.execution_index);
            const uint64_t built = scheduler_prepared_state(SchedulerPreparedPhase::BUILT);
            released =
                scheduler_gm_compare_exchange(
                    control->state, built, scheduler_prepared_state(SchedulerPreparedPhase::FAILED, local->worker_id())
                ) == built;
        } else {
            released = scheduler_cancel_unstarted_handle(base, local, &token.prepared);
        }
        if (released) token = {};
        else cancelled = false;
    }
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index)
        if (!scheduler_cancel_unstarted_slot(base, local, index)) cancelled = false;
    auto &pending = local->cluster_pending;
    if (!scheduler_task_ref_is_null(pending.task_ref)) {
        if (scheduler_cancel_unstarted_handle(base, local, &pending.anchor)) {
            if (!scheduler_cluster_mix_publish_preparation(local, LaneMask::none())) cancelled = false;
            pending = {};
        } else {
            cancelled = false;
        }
    }
    return cancelled;
}

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
    const bool block_owner = scheduler_slot_uses_admission(local, completed) && !completed.gm_mix;
    if constexpr (RecordProtocol) {
        scheduler_protocol_subtask_complete(base, local, completed, pending_slot, true, UINT64_MAX);
        if (block_owner) scheduler_protocol_block_complete(base, local, completed.block_idx, true, UINT64_MAX);
    }
    scheduler_cache_barrier();
    // One physical return retires C and its compute debt in the same RMW.
    // Group credits remain in D until the unique FINISH releases all of them.
    const uint64_t previous_state =
        scheduler_gm_fetch_add(control->completion_state, UINT64_C(0) - SCHEDULER_COMPLETION_RETURN_UNIT);
    const uint64_t previous = scheduler_completion_count(previous_state);
    const uint64_t total = scheduler_completion_total(previous_state);
    if constexpr (RecordProtocol) {
        scheduler_protocol_subtask_complete(base, local, completed, pending_slot, false, previous);
        if (block_owner) scheduler_protocol_block_complete(base, local, completed.block_idx, false, previous);
    }
    if (!scheduler_completion_valid(previous_state) || scheduler_completion_remaining(previous_state) == 0) {
        scheduler_record_error(
            run, completed.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::BLOCK_COMPLETION_INVALID, &completed
        );
        return SchedulerSubtaskFinishResult::ERROR;
    }
    if (previous + 1 != total) return SchedulerSubtaskFinishResult::PARTIAL;
    if (scheduler_completion_debt(previous_state) != 1) return SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED;
    if constexpr (RecordProtocol)
        scheduler_protocol_task_done_from_contribution(base, local, completed, pending_slot, previous);
    return SchedulerSubtaskFinishResult::TASK_LAST;
}

// FromExecutor retains the resident's exclusive ownership of a dispatched slot:
// kernels receive only GM payload/argument contexts, never its private binding.
// The checked entry also serves tests and callers that do not hold that proof.
// PublishCompletion=false retires a real kernel return after sealing or another
// post-return operation failed. It cannot grant task completion or a successor.
template <bool RecordProtocol = true, bool FromExecutor = false, bool PublishCompletion = true>
static inline __aicore__ bool scheduler_complete_own_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint32_t pending_slot, uint64_t profiling_level
) {
    if constexpr (!FromExecutor) {
        if (scheduler == nullptr || pending_slot >= SCHEDULER_PENDING_SLOT_COUNT ||
            scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
            return false;
    }
    if constexpr (!FromExecutor)
        if (scheduler->worker_id() >= scheduler->config.runtime_worker_count) return false;
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
        if (local_slot->cluster_mix) {
            const auto &group = scheduler->cluster_mix;
            const auto *signal = scheduler->cluster_signal;
            if (signal == nullptr || local_slot->gm_mix || !scheduler_task_is_mix(metadata.flags) ||
                scheduler_task_requires_sync_start(metadata.flags) || group.phase != SchedulerGmMixPhase::RUNNING ||
                group.task_ref != task_ref || group.slot_index != pending_slot ||
                group.block_idx != local_slot->block_idx || group.subtask_slot != local_slot->subtask_slot ||
                group.active_mask != scheduler_task_lanes(metadata) || !scheduler_cluster_mix_armed(group.generation) ||
                scheduler_cluster_load_acquire(&signal->armed_generation) != group.generation ||
                scheduler_cluster_load_field(&signal->run_epoch) != scheduler->config.prepared->run_epoch) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, local_slot
                );
                return false;
            }
        }
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
    const SchedulerLocalSlotState completed_slot = *local_slot;
    if (!scheduler_return_prepared_execution(scheduler_state_base, scheduler->config.prepared, &local_slot->prepared)) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID, local_slot
        );
        return false;
    }
    const bool chip_task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t completion_observe = schedule_timing_enabled ? scheduler_cycles() : 0;
    const bool sampled_task_timing_enabled = scheduler->sampled_task_timing(pending_slot);
    const uint8_t completed_subtask_slot = local_slot->subtask_slot;
    const uint32_t completed_block_idx = local_slot->block_idx;
    const bool gm_mix = local_slot->gm_mix;
    if constexpr (RecordProtocol)
        scheduler_protocol_execution_consumed(scheduler_state_base, scheduler, pending_slot, completed_slot);
    scheduler_initialize_free_slot(local_slot);
    scheduler->set_timing_slot(pending_slot, -1);
    const uint64_t completion_phase_end = phase_timing_enabled ? scheduler_cycles() : 0;
    bool completion_ok = true;
    scheduler->continuation_task_ref = SCHEDULER_TASK_REF_INVALID;
    if constexpr (PublishCompletion) {
        const auto contribution = scheduler_contribute_subtask<RecordProtocol>(
            graph, scheduler_state_base, scheduler, run_control, completed_slot, pending_slot
        );
        completion_ok = contribution != SchedulerSubtaskFinishResult::ERROR;
        if (completed_slot.execution_token && contribution == SchedulerSubtaskFinishResult::PARTIAL)
            scheduler->continuation_task_ref = task_ref;
        if (contribution == SchedulerSubtaskFinishResult::TASK_LAST) {
            ++scheduler->pending_completed;
        } else if (contribution == SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED) {
            bool deferred = false;
            completion_ok = scheduler_async_close_computation(scheduler->async, task_ref, deferred) && deferred;
            if (!completion_ok)
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::ASYNC_CONTEXT_INVALID
                );
        }
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
        completed_trace->kernel_start_cycles = executor_trace->kernel_start_cycles;
        completed_trace->kernel_end_cycles = executor_trace->kernel_end_cycles;
        completed_trace->ready_observe_cycles = executor_trace->ready_observe_cycles;
        if (schedule_timing_enabled) completed_trace->complete_start_cycles = completion_observe;
        if (phase_timing_enabled) {
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
    // Failed completion bypasses the contribution's visibility fence. Its real
    // return must still precede a resource-lane acknowledgement.
    if (!PublishCompletion || !completion_ok) scheduler_cache_barrier();
    if (completed_slot.cluster_mix &&
        !scheduler_cluster_mix_release(
            graph, scheduler_state_base, scheduler, run_control, completed_slot, pending_slot
        ))
        return false;
    if (gm_mix && !scheduler_gm_mix_release_lane<RecordProtocol>(
                      graph, scheduler_state_base, scheduler, run_control, completed_slot, pending_slot
                  ))
        return false;
    if (!completion_ok) return false;
    return true;
}
