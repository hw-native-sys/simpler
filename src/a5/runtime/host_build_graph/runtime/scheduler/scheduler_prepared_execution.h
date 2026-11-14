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

enum class PreparedAccessResult : uint8_t { READY, EXHAUSTED, ERROR, WAIT };

struct SchedulerPreparedHandle {
    uint64_t execution_index{UINT64_MAX};
    uint64_t claimed_state{0};
};

struct SchedulerParkedProbePlan {
    uint32_t blocks[2]{};
    uint32_t count{0};
};

inline constexpr uint32_t SCHEDULER_PARKED_SWEEP_INTERVAL = 4;

inline __aicore__ SchedulerParkedProbePlan
scheduler_parked_probe_plan(uint32_t worker_id, uint64_t directory_round, uint32_t block_count, bool positive_hint) {
    SchedulerParkedProbePlan plan{};
    if (block_count == 0) return plan;
    const uint64_t seed = worker_id % block_count;
    // The authoritative sweep cannot be suppressed by either a stale count or
    // a successful fast probe. Run it first and cover every cell with stride one.
    if (directory_round % SCHEDULER_PARKED_SWEEP_INTERVAL == SCHEDULER_PARKED_SWEEP_INTERVAL - 1) {
        plan.blocks[plan.count++] =
            (seed + (directory_round / SCHEDULER_PARKED_SWEEP_INTERVAL) % block_count) % block_count;
    }
    if (positive_hint) {
        const uint32_t fast = (seed + directory_round % block_count) % block_count;
        if (plan.count == 0 || plan.blocks[0] != fast) plan.blocks[plan.count++] = fast;
    }
    return plan;
}

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
    if (layout->version != SCHEDULER_PREPARED_VERSION || layout->task_count != task_count || task_count == 0 ||
        layout->run_epoch != run_epoch || layout->execution_count == 0 || layout->plans_offset < end ||
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

inline __aicore__ __gm__ SchedulerExecutionBinding *
scheduler_execution_binding_at(__gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint64_t index) {
    return reinterpret_cast<__gm__ SchedulerExecutionBinding *>(
               reinterpret_cast<__gm__ uint8_t *>(base) + prepared->bindings_offset
           ) +
           index;
}

inline __aicore__ bool scheduler_prepared_execution_index(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, int64_t task_ref, uint32_t block, uint8_t lane,
    uint64_t *index
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedIndex, task_ref);
    if (prepared == nullptr || index == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= prepared->task_count || lane >= 3)
        return false;
    const auto *plans = reinterpret_cast<__gm__ const SchedulerExecutionPlan *>(
        reinterpret_cast<__gm__ uint8_t *>(base) + prepared->plans_offset
    );
    const auto &plan = plans[task_ref];
    if ((plan.active_mask_bits & (1U << lane)) == 0 || block >= plan.block_num) return false;
    const uint32_t rank = __builtin_popcount(plan.active_mask_bits & ((1U << lane) - 1));
    const uint64_t offset = static_cast<uint64_t>(block) * plan.active_lane_count + rank;
    if (offset >= plan.execution_count || plan.execution_begin >= prepared->execution_count ||
        offset >= prepared->execution_count - plan.execution_begin)
        return false;
    *index = plan.execution_begin + offset;
    return true;
}

// A Host-published Call is immutable. Refresh only its data lines; atomic controls,
// execution bindings and the recorder's dirty lines remain untouched.
static inline __aicore__ void scheduler_observe_prepared_call(__gm__ SchedulerPreparedCall *call) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedObserve);
    for (uint64_t offset = 0; offset < sizeof(*call); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(call) + offset);
    scheduler_cache_barrier();
    asm volatile("" ::: "memory");
}

static inline __aicore__ bool scheduler_prepared_call_valid(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint64_t index, int64_t task_ref, uint32_t block,
    uint8_t lane
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedValidate, task_ref);
    const auto *call = scheduler_prepared_call_at(base, prepared, index);
    const auto *binding = scheduler_execution_binding_at(base, prepared, index);
    return call->run_epoch == prepared->run_epoch && call->execution_index == index && call->task_ref == task_ref &&
           call->block_idx == block && call->subtask_slot == lane && call->num_args <= SPMD_LOCAL_CONTEXT_INDEX &&
           call->function_bin_addr != 0 &&
           call->args[SPMD_LOCAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&binding->local_context) &&
           call->args[SPMD_GLOBAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&binding->global_context);
}

inline __aicore__ bool scheduler_return_prepared_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedReturn);
    if (prepared == nullptr || handle == nullptr || handle->execution_index >= prepared->execution_count ||
        scheduler_prepared_phase(handle->claimed_state) != SchedulerPreparedPhase::CLAIMED)
        return false;
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    const uint64_t returned =
        (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::RETURNED);
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, returned) != handle->claimed_state)
        return false;
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
        scheduler_prepared_state(phase, worker_id, scheduler_prepared_generation(handle->claimed_state)) |
        SCHEDULER_PREPARED_BINDING_READY;
    return handle->claimed_state == expected;
}

static inline __aicore__ PreparedAccessResult scheduler_receive_execution_token(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, int64_t task_ref, uint32_t block, uint8_t lane,
    SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedReceive, task_ref);
    auto *call = scheduler_prepared_call_at(base, prepared, handle->execution_index);
    auto *binding = scheduler_execution_binding_at(base, prepared, handle->execution_index);
    for (uint64_t offset = 0; offset < sizeof(*call); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(call) + offset);
    scheduler_invalidate_cache_line(&binding->local_context);
    scheduler_invalidate_cache_line(&binding->global_context);
    scheduler_cache_barrier();
    asm volatile("" ::: "memory");
    if (!scheduler_prepared_call_valid(base, prepared, handle->execution_index, task_ref, block, lane)) {
        auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
        const uint64_t failed =
            (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::FAILED);
        if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, failed) == handle->claimed_state)
            handle->claimed_state = failed;
        return PreparedAccessResult::ERROR;
    }
    return PreparedAccessResult::READY;
}

// Fresh tickets identify one immutable Call; no peer may consume the same ticket.
static inline __aicore__ PreparedAccessResult scheduler_acquire_execution_token(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id, int64_t task_ref,
    uint32_t block, uint8_t lane, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedAcquire, task_ref);
    uint64_t index = 0;
    if (base == nullptr || handle == nullptr || worker_id >= PLATFORM_MAX_CORES ||
        !scheduler_prepared_execution_index(base, prepared, task_ref, block, lane, &index) ||
        (handle->execution_index != UINT64_MAX && handle->execution_index != index))
        return PreparedAccessResult::ERROR;
    if (handle->claimed_state != 0)
        return scheduler_execution_token_handle_valid(
                   base, prepared, worker_id, handle, SchedulerPreparedPhase::CLAIMED
               ) ?
                   PreparedAccessResult::READY :
                   PreparedAccessResult::ERROR;
    handle->execution_index = index;
    auto *control = scheduler_prepared_control_at(base, prepared, index);
    const uint64_t initial = scheduler_prepared_state(SchedulerPreparedPhase::BUILT);
    const uint64_t claimed =
        scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker_id) | SCHEDULER_PREPARED_BINDING_READY;
    if (scheduler_gm_compare_exchange(control->state, initial, claimed) != initial) return PreparedAccessResult::ERROR;
    handle->claimed_state = claimed;
    return scheduler_receive_execution_token(base, prepared, task_ref, block, lane, handle);
}

// Only an unstarted owner with no mutable Binding writes may park its execution.
static inline __aicore__ bool scheduler_begin_park_prepared_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id,
    SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedParkBegin);
    if (!scheduler_execution_token_handle_valid(base, prepared, worker_id, handle, SchedulerPreparedPhase::CLAIMED))
        return false;
    const uint64_t parking =
        (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::PARKING);
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, parking) != handle->claimed_state)
        return false;
    handle->claimed_state = parking;
    return true;
}

static inline __aicore__ bool scheduler_publish_parked_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id,
    SchedulerPreparedHandle *handle, __gm__ volatile uint64_t &parked_count
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedParkPublish);
    if (!scheduler_execution_token_handle_valid(base, prepared, worker_id, handle, SchedulerPreparedPhase::PARKING))
        return false;
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    const uint64_t parked = scheduler_prepared_state(
        SchedulerPreparedPhase::PARKED, UINT32_MAX, scheduler_prepared_generation(handle->claimed_state)
    );
    scheduler_cache_barrier();
    (void)scheduler_gm_fetch_add(parked_count, UINT64_C(1));
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, parked) != handle->claimed_state) {
        (void)scheduler_gm_fetch_add(parked_count, UINT64_MAX);
        return false;
    }
    *handle = {};
    return true;
}

static inline __aicore__ bool scheduler_park_prepared_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id,
    SchedulerPreparedHandle *handle, __gm__ volatile uint64_t &parked_count
) {
    return scheduler_begin_park_prepared_execution(base, prepared, worker_id, handle) &&
           scheduler_publish_parked_execution(base, prepared, worker_id, handle, parked_count);
}

static inline __aicore__ PreparedAccessResult scheduler_resume_parked_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id, int64_t task_ref,
    uint32_t block, uint8_t lane, uint64_t observed_state, SchedulerPreparedHandle *handle,
    __gm__ volatile uint64_t &parked_count
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarPreparedResume, task_ref);
    uint64_t index = 0;
    if (base == nullptr || handle == nullptr || worker_id >= PLATFORM_MAX_CORES || handle->claimed_state != 0 ||
        !scheduler_prepared_execution_index(base, prepared, task_ref, block, lane, &index) ||
        (handle->execution_index != UINT64_MAX && handle->execution_index != index))
        return PreparedAccessResult::ERROR;
    if (scheduler_prepared_phase(observed_state) != SchedulerPreparedPhase::PARKED)
        return PreparedAccessResult::EXHAUSTED;
    const uint32_t generation = scheduler_prepared_generation(observed_state);
    if (generation == UINT32_MAX ||
        observed_state != scheduler_prepared_state(SchedulerPreparedPhase::PARKED, UINT32_MAX, generation))
        return PreparedAccessResult::ERROR;
    const uint64_t claimed = scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker_id, generation + 1) |
                             SCHEDULER_PREPARED_BINDING_READY;
    auto *control = scheduler_prepared_control_at(base, prepared, index);
    if (scheduler_gm_compare_exchange(control->state, observed_state, claimed) != observed_state)
        return PreparedAccessResult::EXHAUSTED;
    handle->execution_index = index;
    handle->claimed_state = claimed;
    const uint64_t previous_count = scheduler_gm_fetch_add(parked_count, UINT64_MAX);
    if (previous_count == 0) {
        (void)scheduler_gm_fetch_add(parked_count, UINT64_C(1));
        const uint64_t failed = (claimed & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::FAILED);
        if (scheduler_gm_compare_exchange(control->state, claimed, failed) == claimed) handle->claimed_state = failed;
        return PreparedAccessResult::ERROR;
    }
    return scheduler_receive_execution_token(base, prepared, task_ref, block, lane, handle);
}

static inline __aicore__ bool scheduler_fail_prepared_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker_id,
    SchedulerPreparedHandle *handle
) {
    if (!scheduler_execution_token_handle_valid(base, prepared, worker_id, handle, SchedulerPreparedPhase::CLAIMED) &&
        !scheduler_execution_token_handle_valid(base, prepared, worker_id, handle, SchedulerPreparedPhase::PARKING))
        return false;
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    const uint64_t failed =
        (handle->claimed_state & ~UINT64_C(255)) | static_cast<uint64_t>(SchedulerPreparedPhase::FAILED);
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, failed) != handle->claimed_state)
        return false;
    *handle = {};
    return true;
}

inline __aicore__ bool scheduler_group_anchor_index(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, int64_t task_ref, uint32_t block,
    uint64_t *index, uint8_t *anchor_lane
) {
    if (base == nullptr || prepared == nullptr || anchor_lane == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= prepared->task_count)
        return false;
    const auto &plan = scheduler_state_at<SchedulerExecutionPlan>(base, prepared->plans_offset)[task_ref];
    if (plan.active_lane_count < 2 || plan.active_mask_bits == 0) return false;
    *anchor_lane = static_cast<uint8_t>(__builtin_ctz(plan.active_mask_bits));
    return scheduler_prepared_execution_index(base, prepared, task_ref, block, *anchor_lane, index);
}

inline __aicore__ bool scheduler_group_handle_valid(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t coordinator,
    const SchedulerPreparedHandle *handle, SchedulerPreparedPhase phase
) {
    return base != nullptr && prepared != nullptr && handle != nullptr && coordinator < PLATFORM_MAX_CORES &&
           handle->execution_index < prepared->execution_count &&
           handle->claimed_state ==
               scheduler_prepared_state(phase, coordinator, scheduler_prepared_generation(handle->claimed_state));
}

// Only the first active Call is a group anchor. Reservation never executes that lane.
static inline __aicore__ PreparedAccessResult scheduler_reserve_group_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t coordinator, int64_t task_ref,
    uint32_t block, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupReserve, task_ref);
    uint64_t index = UINT64_MAX;
    uint8_t lane = UINT8_MAX;
    if (handle == nullptr || coordinator >= PLATFORM_MAX_CORES || handle->claimed_state != 0 ||
        !scheduler_group_anchor_index(base, prepared, task_ref, block, &index, &lane) ||
        (handle->execution_index != UINT64_MAX && handle->execution_index != index))
        return PreparedAccessResult::ERROR;
    handle->execution_index = index;
    const uint64_t initial = scheduler_prepared_state(SchedulerPreparedPhase::BUILT);
    const uint64_t reserved = scheduler_prepared_state(SchedulerPreparedPhase::GROUP_RESERVED, coordinator);
    auto *control = scheduler_prepared_control_at(base, prepared, index);
    if (scheduler_gm_compare_exchange(control->state, initial, reserved) != initial) return PreparedAccessResult::ERROR;
    handle->claimed_state = reserved;
    return PreparedAccessResult::READY;
}

static inline __aicore__ bool scheduler_park_group_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t coordinator,
    SchedulerPreparedHandle *handle, __gm__ volatile uint64_t &parked_count
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupPark);
    if (!scheduler_group_handle_valid(base, prepared, coordinator, handle, SchedulerPreparedPhase::GROUP_RESERVED))
        return false;
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    const uint32_t generation = scheduler_prepared_generation(handle->claimed_state);
    const uint64_t parking = scheduler_prepared_state(SchedulerPreparedPhase::PARKING, coordinator, generation);
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, parking) != handle->claimed_state)
        return false;
    handle->claimed_state = parking;
    scheduler_cache_barrier();
    (void)scheduler_gm_fetch_add(parked_count, UINT64_C(1));
    const uint64_t parked = scheduler_prepared_state(SchedulerPreparedPhase::PARKED, UINT32_MAX, generation);
    if (scheduler_gm_compare_exchange(control->state, parking, parked) != parking) {
        (void)scheduler_gm_fetch_add(parked_count, UINT64_MAX);
        return false;
    }
    *handle = {};
    return true;
}

static inline __aicore__ PreparedAccessResult scheduler_resume_group_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t coordinator, int64_t task_ref,
    uint32_t block, uint64_t observed, SchedulerPreparedHandle *handle, __gm__ volatile uint64_t &parked_count
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupResume, task_ref);
    uint64_t index = UINT64_MAX;
    uint8_t lane = UINT8_MAX;
    if (handle == nullptr || coordinator >= PLATFORM_MAX_CORES || handle->claimed_state != 0 ||
        !scheduler_group_anchor_index(base, prepared, task_ref, block, &index, &lane) ||
        (handle->execution_index != UINT64_MAX && handle->execution_index != index))
        return PreparedAccessResult::ERROR;
    if (scheduler_prepared_phase(observed) != SchedulerPreparedPhase::PARKED) return PreparedAccessResult::EXHAUSTED;
    const uint32_t generation = scheduler_prepared_generation(observed);
    if (generation == UINT32_MAX ||
        observed != scheduler_prepared_state(SchedulerPreparedPhase::PARKED, UINT32_MAX, generation))
        return PreparedAccessResult::ERROR;
    const uint64_t reserved =
        scheduler_prepared_state(SchedulerPreparedPhase::GROUP_RESERVED, coordinator, generation + 1);
    auto *control = scheduler_prepared_control_at(base, prepared, index);
    if (scheduler_gm_compare_exchange(control->state, observed, reserved) != observed)
        return PreparedAccessResult::EXHAUSTED;
    handle->execution_index = index;
    handle->claimed_state = reserved;
    if (scheduler_gm_fetch_add(parked_count, UINT64_MAX) == 0) {
        (void)scheduler_gm_fetch_add(parked_count, UINT64_C(1));
        const uint64_t failed = scheduler_prepared_state(SchedulerPreparedPhase::FAILED, coordinator, generation + 1);
        if (scheduler_gm_compare_exchange(control->state, reserved, failed) == reserved) handle->claimed_state = failed;
        return PreparedAccessResult::ERROR;
    }
    return PreparedAccessResult::READY;
}

static inline __aicore__ bool scheduler_arm_group_execution(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t coordinator,
    SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupArm);
    if (!scheduler_group_handle_valid(base, prepared, coordinator, handle, SchedulerPreparedPhase::GROUP_RESERVED))
        return false;
    const uint64_t armed = scheduler_prepared_state(
        SchedulerPreparedPhase::GROUP_ARMED, coordinator, scheduler_prepared_generation(handle->claimed_state)
    );
    auto *control = scheduler_prepared_control_at(base, prepared, handle->execution_index);
    if (scheduler_gm_compare_exchange(control->state, handle->claimed_state, armed) != handle->claimed_state)
        return false;
    handle->claimed_state = armed;
    return true;
}

// SSBUF arming authorizes the actual anchor lane; its coordinator is not an executor surrogate.
static inline __aicore__ PreparedAccessResult scheduler_acquire_group_lane(
    __gm__ void *base, __gm__ const SchedulerPreparedLayout *prepared, uint32_t worker, uint32_t coordinator,
    int64_t task_ref, uint32_t block, uint8_t lane, uint64_t armed_state, SchedulerPreparedHandle *handle
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarGroupAcquire, task_ref);
    uint64_t anchor = UINT64_MAX;
    uint8_t anchor_lane = UINT8_MAX;
    if (handle == nullptr || worker >= PLATFORM_MAX_CORES || coordinator >= PLATFORM_MAX_CORES ||
        !scheduler_group_anchor_index(base, prepared, task_ref, block, &anchor, &anchor_lane))
        return PreparedAccessResult::ERROR;
    if (lane != anchor_lane)
        return scheduler_acquire_execution_token(base, prepared, worker, task_ref, block, lane, handle);
    if (handle->execution_index != UINT64_MAX && handle->execution_index != anchor) return PreparedAccessResult::ERROR;
    if (handle->claimed_state != 0)
        return scheduler_execution_token_handle_valid(base, prepared, worker, handle, SchedulerPreparedPhase::CLAIMED) ?
                   PreparedAccessResult::READY :
                   PreparedAccessResult::ERROR;
    auto *control = scheduler_prepared_control_at(base, prepared, anchor);
    const uint64_t observed = armed_state;
    const uint32_t generation = scheduler_prepared_generation(observed);
    if (observed != scheduler_prepared_state(SchedulerPreparedPhase::GROUP_ARMED, coordinator, generation))
        return PreparedAccessResult::ERROR;
    const uint64_t claimed = scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker, generation) |
                             SCHEDULER_PREPARED_BINDING_READY;
    if (scheduler_gm_compare_exchange(control->state, observed, claimed) != observed)
        return PreparedAccessResult::ERROR;
    handle->execution_index = anchor;
    handle->claimed_state = claimed;
    return scheduler_receive_execution_token(base, prepared, task_ref, block, lane, handle);
}
