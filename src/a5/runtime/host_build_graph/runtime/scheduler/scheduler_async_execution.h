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

#include "scheduler_async_runtime.h"
#include "scheduler_admission.h"

// Private to one resident executor. Completion authority comes from the
// existing slot/cohort retirement, independently of the executing core.
template <typename Backend>
struct SchedulerAsyncExecutionState {
    SchedulerAsyncRuntimeView view{};
    SchedulerAsyncDiscoveryState discovery{};
    SchedulerAsyncServiceGate gate{};
    uint32_t bound_generations[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t permitted_slot{UINT32_MAX};
    uint32_t permitted_generation{0};
};

template <typename Backend>
inline __aicore__ bool scheduler_async_execution_mount(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAsyncExecutionState<Backend> *state
) {
    if (state == nullptr || !scheduler_async_runtime_mount(graph, base, local, run, &state->view)) return false;
    if (!state->view.enabled()) {
        state->discovery = {};
        return true;
    }
    state->discovery.check = scheduler_async_directory_start(state->view.layout().group_count, local->worker_id());
    state->discovery.finish = scheduler_async_directory_start(state->view.layout().finish_count, local->worker_id());
    return true;
}

// The current READY slot owns its context until kernel return. A retry of
// ordered admission reuses the binding only while that slot generation matches.
template <typename Backend>
inline __aicore__ bool scheduler_async_execution_bind(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t slot_index
) {
    if (state == nullptr || local == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT) return false;
    if (!state->view.enabled()) return true;
    const auto &slot = local->slots[slot_index];
    if (state->permitted_slot != UINT32_MAX || slot.state != SchedulerDispatchSlotState::READY ||
        slot.generation == 0 || scheduler_read_error(run) != 0)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    const auto metadata = scheduler_load_dispatch_metadata(base, local, slot.task_ref, 0);
    if (slot.block_idx >= metadata.logical_block_num || slot.subtask_slot >= 3 ||
        !metadata.active_mask.has(static_cast<SubtaskSlot>(slot.subtask_slot)))
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    if (scheduler_task_is_mix(metadata.flags) || scheduler_task_requires_sync_start(metadata.flags)) {
        const auto *admission = local->admission;
        if (!slot.cooperative || slot.gm_mix != scheduler_task_is_mix(metadata.flags) || admission == nullptr ||
            admission->current_task_ref != slot.task_ref ||
            admission->current_block_num != metadata.logical_block_num || admission->current_flags != metadata.flags ||
            admission->current_active_mask != metadata.active_mask ||
            (slot.gm_mix &&
             (admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING || admission->gm_mix.slot_index != slot_index ||
              admission->gm_mix.slot_generation != slot.generation || admission->gm_mix.block_idx != slot.block_idx ||
              admission->gm_mix.subtask_slot != slot.subtask_slot)))
            return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    }
    if (state->bound_generations[slot_index] != slot.generation) {
        auto *payload = scheduler_state_at<DispatchPayload>(
            base, local->dispatch_payload_offset(local->config.self_lane, slot_index)
        );
        bool bound = false;
        if (!scheduler_async_runtime_bind_payload(
                state->view, graph, local, run, slot.task_ref, slot.block_idx, slot.subtask_slot, payload, &bound
            ))
            return false;
        state->bound_generations[slot_index] = slot.generation;
    }
    return true;
}

// This records an already-granted ordinary/admission permit. No asynchronous
// service is allowed until the matching kernel-return seal finishes.
template <typename Backend>
inline __aicore__ bool scheduler_async_execution_permitted(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, SchedulerLocalState *local,
    __gm__ SchedulerRunControl *run, uint32_t slot_index
) {
    if (!state->view.enabled()) return true;
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT || state->permitted_slot != UINT32_MAX ||
        scheduler_read_error(run) != 0)
        return scheduler_async_runtime_detail::fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    const auto &slot = local->slots[slot_index];
    if (slot.state != SchedulerDispatchSlotState::READY || slot.generation == 0 ||
        state->bound_generations[slot_index] != slot.generation)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    state->permitted_slot = slot_index;
    state->permitted_generation = slot.generation;
    return true;
}

template <typename Backend>
inline __aicore__ bool scheduler_async_execution_returned(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t slot_index
) {
    if (!state->view.enabled()) return true;
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT || state->permitted_slot != slot_index)
        return scheduler_async_runtime_detail::fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    const auto &slot = local->slots[slot_index];
    if (slot.generation != state->permitted_generation || slot.state != SchedulerDispatchSlotState::READY)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    auto *payload =
        scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, slot_index));
    bool registered = false;
    if (!scheduler_async_runtime_seal(
            state->view, graph, base, local, run, slot.task_ref, slot.block_idx, slot.subtask_slot, payload, &registered
        ))
        return false;
    state->gate.kernel_returned();
    state->permitted_slot = UINT32_MAX;
    state->permitted_generation = 0;
    return true;
}

template <typename Backend>
inline __aicore__ bool scheduler_async_execution_service(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, Backend &backend, SchedulerAsyncAdmission admission,
    bool runnable_business, bool *progress
) {
    if (state->permitted_slot != UINT32_MAX || !state->gate.take(state->view.enabled(), admission, runnable_business))
        return true;
    SchedulerAsyncSliceBudget budget{};
    const auto result =
        scheduler_async_runtime_service(state->view, graph, base, local, run, &state->discovery, &budget, backend);
    if (result == SchedulerAsyncRuntimeServiceResult::ERROR) return false;
    *progress = *progress || result == SchedulerAsyncRuntimeServiceResult::PROGRESS;
    return true;
}
