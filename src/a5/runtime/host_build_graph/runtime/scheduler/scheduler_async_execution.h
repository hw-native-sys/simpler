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
#include "scheduler_async_cohort.h"
#include "scheduler_admission.h"

// Private to one resident executor. Neither cursors nor permits enter the wire
// image; completion still obtains its unique authority from slot/cohort retirement.
template <typename Backend>
struct SchedulerAsyncExecutionState {
    SchedulerAsyncRuntimeView view{};
    SchedulerAsyncCohortView cohorts{};
    SchedulerAsyncDiscoveryState discovery{};
    SchedulerAsyncServiceGate gate{};
    typename Backend::ConsumerCursor consumers[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t bound_generations[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t prepared_generations[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint64_t prepared_write_serials[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint64_t core_write_serial{0};
    uint32_t pinned_slot{UINT32_MAX};
    uint32_t pinned_generation{0};
    uint32_t permitted_slot{UINT32_MAX};
    uint32_t permitted_generation{0};
};

template <typename Backend>
inline __aicore__ bool scheduler_async_execution_mount(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAsyncExecutionState<Backend> *state
) {
    if (state == nullptr || !scheduler_async_runtime_mount(graph, base, local, run, &state->view)) return false;
    if (state->view.enabled() &&
        !SchedulerAsyncCohortView::mount(state->view.base, state->view.layout, &state->cohorts))
        return scheduler_async_runtime_detail::fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    state->discovery.check = scheduler_async_directory_start(state->view.layout.group_count, local->worker_id());
    state->discovery.finish = scheduler_async_directory_start(state->view.layout.finish_count, local->worker_id());
    return true;
}

// A prepared cohort contribution is monotonic. Its physical core retries only
// this slot until kernel return; CHECK/FINISH service remains available.
template <typename Backend>
inline __aicore__ bool scheduler_async_execution_select(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, SchedulerLocalState *local,
    __gm__ SchedulerRunControl *run, uint32_t *slot_index, uint64_t *publication, bool *found
) {
    if (state->pinned_slot == UINT32_MAX) {
        *found = scheduler_local_ready_pop(local, slot_index, publication);
        return true;
    }
    const uint32_t selected = state->pinned_slot;
    if (selected >= SCHEDULER_PENDING_SLOT_COUNT || (local->local_ready_mask & (1u << selected)) == 0 ||
        local->slots[selected].state != SchedulerDispatchSlotState::READY ||
        local->slots[selected].generation != state->pinned_generation)
        return scheduler_async_runtime_detail::fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    local->local_ready_mask &= ~(1u << selected);
    *slot_index = selected;
    *publication = scheduler_dispatch_publication(state->pinned_generation, SchedulerDispatchSlotState::READY);
    *found = true;
    return true;
}

// Called before obtaining the business permit, after this iteration's optional
// CHECK/FINISH slice. PENDING retains the ordinary READY slot and its generation.
template <typename Backend>
inline __aicore__ bool scheduler_async_execution_prepare(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t slot_index, Backend &backend, bool *ready
) {
    if (state == nullptr || local == nullptr || ready == nullptr || slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return false;
    *ready = false;
    if (!state->view.enabled()) {
        *ready = true;
        return true;
    }
    auto &slot = local->slots[slot_index];
    if (state->permitted_slot != UINT32_MAX || slot.state != SchedulerDispatchSlotState::READY || slot.generation == 0)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    if ((state->pinned_slot != UINT32_MAX &&
         (state->pinned_slot != slot_index || state->pinned_generation != slot.generation)) ||
        scheduler_read_error(run) != 0)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    const auto metadata = scheduler_load_dispatch_metadata(base, local, slot.task_ref, 0);
    if (!scheduler_async_cohort_execution_matches(state->cohorts, slot.task_ref, metadata))
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    if (scheduler_task_is_mix(metadata.flags) || scheduler_task_requires_sync_start(metadata.flags)) {
        const auto *admission = local->admission;
        if (!slot.cooperative || slot.gm_mix != scheduler_task_is_mix(metadata.flags) || admission == nullptr ||
            admission->current_task_ref != slot.task_ref ||
            admission->current_block_num != metadata.logical_block_num || admission->current_flags != metadata.flags ||
            admission->current_active_mask != metadata.active_mask || slot.block_idx >= admission->current_block_num ||
            (slot.gm_mix &&
             (admission->gm_mix.phase != SchedulerGmMixPhase::RUNNING || admission->gm_mix.slot_index != slot_index ||
              admission->gm_mix.slot_generation != slot.generation || admission->gm_mix.block_idx != slot.block_idx ||
              admission->gm_mix.subtask_slot != slot.subtask_slot)))
            return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    }
    auto *payload =
        scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, slot_index));
    if (state->bound_generations[slot_index] != slot.generation) {
        bool bound = false;
        if (!scheduler_async_runtime_bind_payload(
                state->view, graph, local, run, slot.task_ref, slot.block_idx, slot.subtask_slot, payload, &bound
            ))
            return false;
        state->bound_generations[slot_index] = slot.generation;
        state->consumers[slot_index] = {};
    }
    typename Backend::ConsumerIdentity identity{
        state->view.layout.run_epoch, slot.task_ref, slot.block_idx, slot.subtask_slot, slot.generation,
        state->core_write_serial
    };
    uint32_t budget = 8;
    const auto result = backend.prepare_consumer(identity, state->consumers[slot_index], budget);
    if (result == Backend::ConsumerResult::ERROR || scheduler_read_error(run) != 0)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    if (backend.consumer_pinned(state->consumers[slot_index])) {
        state->pinned_slot = slot_index;
        state->pinned_generation = slot.generation;
    }
    *ready = result == Backend::ConsumerResult::READY;
    state->prepared_generations[slot_index] = *ready ? slot.generation : 0;
    state->prepared_write_serials[slot_index] = state->core_write_serial;
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
        state->bound_generations[slot_index] != slot.generation ||
        state->prepared_generations[slot_index] != slot.generation ||
        state->prepared_write_serials[slot_index] != state->core_write_serial)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    state->permitted_slot = slot_index;
    state->permitted_generation = slot.generation;
    return true;
}

template <
    typename Backend, typename BindingPolicy,
    typename Publication = scheduler_async_runtime_detail::NoAdditionalSealPublication>
inline __aicore__ bool scheduler_async_execution_returned(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t slot_index, BindingPolicy &backend,
    Publication publication = {}
) {
    if (!state->view.enabled()) return true;
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT || state->permitted_slot != slot_index ||
        state->core_write_serial == UINT64_MAX)
        return scheduler_async_runtime_detail::fail(graph, local, run, SCHEDULER_TASK_REF_INVALID);
    const auto &slot = local->slots[slot_index];
    if (slot.generation != state->permitted_generation || slot.state != SchedulerDispatchSlotState::READY)
        return scheduler_async_runtime_detail::fail(graph, local, run, slot.task_ref);
    auto *payload =
        scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, slot_index));
    bool registered = false;
    if (!scheduler_async_runtime_seal(
            state->view, graph, base, local, run, slot.task_ref, slot.block_idx, slot.subtask_slot, payload, backend,
            &registered, publication
        ))
        return false;
    ++state->core_write_serial;
    state->gate.kernel_returned();
    state->permitted_slot = UINT32_MAX;
    state->permitted_generation = 0;
    state->pinned_slot = UINT32_MAX;
    state->pinned_generation = 0;
    return true;
}

template <typename Backend>
inline __aicore__ bool scheduler_async_execution_service(
    SchedulerAsyncExecutionState<Backend> *state, const SchedulerGraphAccess &graph, __gm__ void *base,
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, Backend &backend, SchedulerAsyncAdmission admission,
    bool runnable_business, bool *progress, uint64_t profiling_level = 0
) {
    if (state->permitted_slot != UINT32_MAX || !state->gate.take(state->view.enabled(), admission, runnable_business))
        return true;
    SchedulerAsyncSliceBudget budget{};
    const auto result = scheduler_async_runtime_service(
        state->view, graph, base, local, run, &state->discovery, &budget, backend, profiling_level
    );
    if (result == SchedulerAsyncRuntimeServiceResult::ERROR) return false;
    *progress = *progress || result == SchedulerAsyncRuntimeServiceResult::PROGRESS;
    return true;
}
