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

#include "scheduler_graph_binding.h"
#include "scheduler_prepared_layout.h"
#include "scheduler_scalar_trace.h"

struct SchedulerPreparedHandle {
    uint64_t execution_index{UINT64_MAX};
    uint64_t claimed_state{0};
};

inline __aicore__ bool scheduler_prepared_region_end(
    uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment, uint64_t *end
) {
    if (offset % alignment != 0 || offset > bytes || count > (bytes - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}

// Layout validation observes metadata only. Calls have a separate acquisition
// or read-only group-preparation boundary before their first data access.
inline __aicore__ bool scheduler_prepared_layout_valid(
    __gm__ void *base, uint64_t bytes, uint64_t layout_offset, uint64_t task_count, uint64_t run_epoch
) {
    const uint64_t address = reinterpret_cast<uint64_t>(base);
    uint64_t end = 0;
    if (base == nullptr || address % SCHEDULER_PREPARED_ALIGNMENT != 0 || bytes > UINT64_MAX - address ||
        layout_offset == 0 ||
        !scheduler_prepared_region_end(
            bytes, layout_offset, 1, sizeof(SchedulerPreparedLayout), alignof(SchedulerPreparedLayout), &end
        ))
        return false;
    const auto *layout = reinterpret_cast<__gm__ const SchedulerPreparedLayout *>(address + layout_offset);
    uint64_t controls_end = 0;
    if (layout->task_controls_offset == 0 ||
        !scheduler_prepared_region_end(
            bytes, layout->task_controls_offset, task_count, sizeof(SchedulerTaskControl),
            alignof(SchedulerTaskControl), &controls_end
        ) ||
        controls_end > layout_offset)
        return false;
    if (layout->task_count != task_count || task_count == 0 || layout->run_epoch != run_epoch ||
        layout->execution_count == 0 || layout->plans_offset < end ||
        !scheduler_prepared_region_end(
            bytes, layout->plans_offset, task_count, sizeof(SchedulerExecutionPlan), alignof(SchedulerExecutionPlan),
            &end
        ) ||
        layout->controls_offset < end ||
        !scheduler_prepared_region_end(
            bytes, layout->controls_offset, layout->execution_count, sizeof(SchedulerPreparedControl),
            alignof(SchedulerPreparedControl), &end
        ) ||
        layout->bindings_offset < end ||
        !scheduler_prepared_region_end(
            bytes, layout->bindings_offset, layout->execution_count, sizeof(SchedulerExecutionBinding),
            alignof(SchedulerExecutionBinding), &end
        ) ||
        layout->calls_offset < end ||
        !scheduler_prepared_region_end(
            bytes, layout->calls_offset, layout->execution_count, sizeof(SchedulerPreparedCall),
            alignof(SchedulerPreparedCall), &end
        ))
        return false;
    return true;
}

inline __aicore__ __gm__ SchedulerPreparedControl *
scheduler_prepared_control_at(__gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint64_t index) {
    return reinterpret_cast<__gm__ SchedulerPreparedControl *>(
               reinterpret_cast<__gm__ uint8_t *>(base) + prepared->controls_offset
           ) +
           index;
}

inline __aicore__ __gm__ SchedulerPreparedCall *
scheduler_prepared_call_at(__gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint64_t index) {
    return reinterpret_cast<__gm__ SchedulerPreparedCall *>(
               reinterpret_cast<__gm__ uint8_t *>(base) + prepared->calls_offset
           ) +
           index;
}

// Ticket ownership comes from the successful task FAA, not this diagnostic.
// Its fixed writer owns the entire isolated line through return or cancellation.
static inline __aicore__ bool
scheduler_transition_prepared_state(__gm__ SchedulerPreparedControl *control, uint64_t expected, uint64_t desired) {
    if (((expected | desired) & SCHEDULER_PREPARED_TICKET_OWNED) != 0) {
        if (control->state != expected) return false;
        control->state = desired;
        scheduler_writeback_cache_line(control);
        scheduler_cache_barrier();
        return true;
    }
    return scheduler_gm_compare_exchange(control->state, expected, desired) == expected;
}

inline __aicore__ __gm__ SchedulerExecutionBinding *
scheduler_execution_binding_at(__gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint64_t index) {
    return reinterpret_cast<__gm__ SchedulerExecutionBinding *>(
               reinterpret_cast<__gm__ uint8_t *>(base) + prepared->bindings_offset
           ) +
           index;
}

inline __aicore__ bool scheduler_prepared_execution_index(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, int64_t task_ref, uint32_t block, uint8_t lane,
    uint64_t *index, uint8_t group_mask = 0
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedIndex, task_ref);
    if (prepared == nullptr || index == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= prepared->task_count || lane >= 3)
        return false;
    const auto *plans = reinterpret_cast<__gm__ const SchedulerExecutionPlan *>(
        reinterpret_cast<__gm__ uint8_t *>(base) + prepared->plans_offset
    );
    const auto &plan = plans[task_ref];
    if (group_mask != 0 &&
        (plan.active_mask_bits != group_mask || plan.active_lane_count != __builtin_popcount(group_mask)))
        return false;
    if ((plan.active_mask_bits & (1U << lane)) == 0 || block >= plan.block_num) return false;
    const uint32_t rank = __builtin_popcount(plan.active_mask_bits & ((1U << lane) - 1));
    const uint64_t offset = static_cast<uint64_t>(block) * plan.active_lane_count + rank;
    if (offset >= plan.execution_count || plan.execution_begin >= prepared->execution_count ||
        offset >= prepared->execution_count - plan.execution_begin)
        return false;
    *index = plan.execution_begin + offset;
    return true;
}

static inline __aicore__ bool scheduler_prepared_call_valid(
    __gm__ const SchedulerPreparedCall *call, __gm__ const SchedulerExecutionBinding *binding, uint64_t run_epoch,
    uint64_t index, int64_t task_ref, uint32_t block, uint8_t lane
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedValidate, task_ref);
    return call->run_epoch == run_epoch && call->execution_index == index && call->task_ref == task_ref &&
           call->block_idx == block && call->subtask_slot == lane && call->num_args <= SPMD_LOCAL_CONTEXT_INDEX &&
           call->function_bin_addr != 0 &&
           call->args[SPMD_LOCAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&binding->local_context) &&
           call->args[SPMD_GLOBAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&binding->global_context);
}

static inline __aicore__ bool scheduler_prepared_already_returned(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, const SchedulerPreparedHandle &handle
) {
    if ((handle.claimed_state & SCHEDULER_PREPARED_MIX_OWNED) != 0 || prepared == nullptr ||
        prepared->task_controls_offset == 0 || handle.execution_index >= prepared->execution_count)
        return false;
    const auto *call = scheduler_prepared_call_at(base, prepared, handle.execution_index);
    if (call->task_ref < 0 || static_cast<uint64_t>(call->task_ref) >= prepared->task_count) return false;
    const auto &plan = scheduler_state_at<SchedulerExecutionPlan>(base, prepared->plans_offset)[call->task_ref];
    if (plan.active_lane_count != 1) return false;
    auto &completion =
        scheduler_state_at<SchedulerTaskControl>(base, prepared->task_controls_offset)[call->task_ref].completion_state;
    return scheduler_prepared_retirement_inferred(
        handle.claimed_state, *prepared, handle.execution_index, *call, plan,
        scheduler_gm_fetch_add(completion, UINT64_C(0))
    );
}

// Only this member uses ENTERED to distinguish execution from pre-entry cancellation.
static inline __aicore__ void scheduler_mark_mix_kernel_entered(SchedulerPreparedHandle &handle) {
    handle.claimed_state =
        (handle.claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::ENTERED);
}

// The executor owns the returned call. FIN or the exit fence publishes its return state.
inline __aicore__ bool scheduler_return_prepared_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedReturn);
    if (prepared == nullptr || handle == nullptr || handle->execution_index >= prepared->execution_count) return false;
    const bool mix_owned = (handle->claimed_state & SCHEDULER_PREPARED_MIX_OWNED) != 0;
    if (scheduler_prepared_phase(handle->claimed_state) !=
        (mix_owned ? SchedulerPreparedPhase::ENTERED : SchedulerPreparedPhase::CLAIMED))
        return false;
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    const uint64_t returned =
        (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::RETURNED);
    if (mix_owned) {
        scheduler_store_signal(control->state, returned);
    } else if ((handle->claimed_state & SCHEDULER_PREPARED_TICKET_OWNED) != 0) {
        if (!scheduler_transition_prepared_state(control, handle->claimed_state, returned)) return false;
    } else {
        scheduler_gm_store(control->state, returned);
    }
    *handle = {};
    return true;
}

static inline __aicore__ bool scheduler_execution_token_handle_valid(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id,
    const SchedulerPreparedHandle *handle, SchedulerPreparedPhase phase
) {
    if (base == nullptr || prepared == nullptr || handle == nullptr || worker_id >= PLATFORM_MAX_CORES ||
        handle->execution_index >= prepared->execution_count)
        return false;
    const uint64_t expected =
        scheduler_prepared_state(phase, worker_id) | (handle->claimed_state & SCHEDULER_PREPARED_TICKET_OWNED);
    return handle->claimed_state == expected;
}

// Host publishes the immutable Call array before resident startup invalidates
// this core's cache. Binding remains writable and has its own receipt boundary.
static inline __aicore__ bool scheduler_receive_payload(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, int64_t task_ref, uint32_t block, uint8_t lane,
    SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedReceive, task_ref);
    const auto index = handle->execution_index;
    const auto *call = scheduler_prepared_call_at(base, prepared, index);
    auto *binding = scheduler_execution_binding_at(base, prepared, index);
    scheduler_invalidate_cache_line(&binding->local_context);
    scheduler_invalidate_cache_line(&binding->global_context);
    scheduler_cache_barrier();
    asm volatile("" ::: "memory");
    if (!scheduler_prepared_call_valid(call, binding, prepared->run_epoch, index, task_ref, block, lane)) {
        const uint64_t failed =
            (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::FAILED);
        if ((handle->claimed_state & SCHEDULER_PREPARED_MIX_OWNED) != 0 ||
            scheduler_transition_prepared_state(
                scheduler_prepared_control_at(base, prepared, index), handle->claimed_state, failed
            ))
            handle->claimed_state = failed;
        return false;
    }
    return true;
}

// Fresh tickets identify one immutable Call; its fixed owner must receive the
// payload before using the claimed handle for execution.
static inline __aicore__ bool scheduler_claim_for_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id, int64_t task_ref,
    uint32_t block, uint8_t lane, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedClaim, task_ref);
    SchedulerExecutionClaimLoads claim_loads;
    uint64_t index = 0;
    if (base == nullptr || handle == nullptr || worker_id >= PLATFORM_MAX_CORES ||
        !scheduler_prepared_execution_index(base, prepared, task_ref, block, lane, &index) ||
        (handle->execution_index != UINT64_MAX && handle->execution_index != index))
        return false;
    if (const auto &plan = scheduler_state_at<SchedulerExecutionPlan>(base, prepared->plans_offset)[task_ref];
        plan.active_lane_count != 1 ||
        (handle->claimed_state != 0 && (handle->claimed_state & SCHEDULER_PREPARED_TICKET_OWNED) == 0))
        return false;
    if (handle->claimed_state != 0)
        return scheduler_execution_token_handle_valid(
            base, prepared, worker_id, handle, SchedulerPreparedPhase::CLAIMED
        );
    handle->execution_index = index;
    auto *control = scheduler_prepared_control_at(base, prepared, index);
    const uint64_t initial = scheduler_prepared_state(SchedulerPreparedPhase::BUILT);
    const uint64_t claimed =
        scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker_id) | SCHEDULER_PREPARED_TICKET_OWNED;
    if (!scheduler_transition_prepared_state(control, initial, claimed)) return false;
    handle->claimed_state = claimed;
    return true;
}

// One valid group ticket and a stable SSBUF generation name exactly one worker
// per active lane. Only that worker enters this path for the generation.
static inline __aicore__ bool scheduler_acquire_group_lane(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker, int64_t task_ref,
    uint32_t block, uint8_t lane, uint8_t group_mask, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupAcquire, task_ref);
    uint64_t index = UINT64_MAX;
    if (!scheduler_prepared_execution_index(base, prepared, task_ref, block, lane, &index, group_mask)) return false;
    const uint64_t claimed =
        scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker) | SCHEDULER_PREPARED_MIX_OWNED;
    handle->execution_index = index;
    handle->claimed_state = claimed;
    return scheduler_receive_payload(base, prepared, task_ref, block, lane, handle);
}
