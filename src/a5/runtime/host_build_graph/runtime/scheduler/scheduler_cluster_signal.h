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

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "scheduler_scalar_record.h"

#if defined(__CCE_AICORE__)
#define SCHEDULER_CLUSTER_SSBUF __ssbuf__
#else
#define SCHEDULER_CLUSTER_SSBUF
#endif

inline constexpr uint32_t SCHEDULER_CLUSTER_REGION_OFFSET = 2 * 1024;
inline constexpr uint32_t SCHEDULER_CLUSTER_REGION_BYTES = 1024;
static_assert(SCHEDULER_CLUSTER_REGION_OFFSET + SCHEDULER_CLUSTER_REGION_BYTES == PLATFORM_SSBUF_SIZE);

// AIV0 publishes the descriptor and the cluster's monotonic order boundary.
struct SchedulerClusterSignal {
    volatile uint64_t run_epoch;
    volatile int64_t task_ref;
    volatile uint64_t anchor_state;
    volatile uint32_t rank;
    volatile uint32_t block_idx;
    volatile uint32_t active_mask;
    volatile uint32_t armed_generation;
    volatile uint32_t mix_floor;
    volatile uint32_t arrived[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t finished[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t dependencies_generation;
    volatile uint32_t accounted_generation;
    volatile uint32_t trace_generation;
};
static_assert(sizeof(SchedulerClusterSignal) <= SCHEDULER_CLUSTER_REGION_BYTES);
static_assert(alignof(SchedulerClusterSignal) <= 8 && std::is_standard_layout_v<SchedulerClusterSignal>);

inline __aicore__ uint8_t scheduler_mix_accountant_lane(LaneMask active) {
    return active.has_aiv0() ? static_cast<uint8_t>(SubtaskSlot::AIV0) : static_cast<uint8_t>(SubtaskSlot::AIC);
}

inline __aicore__ SCHEDULER_CLUSTER_SSBUF SchedulerClusterSignal *scheduler_cluster_signal_at(uint64_t hardware_base) {
    return reinterpret_cast<SCHEDULER_CLUSTER_SSBUF SchedulerClusterSignal *>(
        hardware_base + SCHEDULER_CLUSTER_REGION_OFFSET
    );
}

// The DSB remains in the SSBUF access path, without an outlined operation wrapper.
inline __aicore__ void scheduler_cluster_barrier() {
    auto *trace = scheduler_scalar_current();
    const bool record = trace != nullptr && trace->buffer != nullptr;
    const uint64_t begin = record ? scheduler_scalar_clock() : 0;
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    if (record) trace->append_memory(SchedPhaseKind::ScalarDsbAll, begin, scheduler_scalar_clock());
}

template <typename T>
inline __aicore__ T scheduler_cluster_load_field(const SCHEDULER_CLUSTER_SSBUF volatile T *field) {
#if defined(__CCE_AICORE__)
    return *field;
#else
    return __atomic_load_n(field, __ATOMIC_ACQUIRE);
#endif
}

template <typename T>
inline __aicore__ void scheduler_cluster_store_field(SCHEDULER_CLUSTER_SSBUF volatile T *field, T value) {
#if defined(__CCE_AICORE__)
    *field = value;
#else
    __atomic_store_n(field, value, __ATOMIC_RELEASE);
#endif
}

inline __aicore__ uint32_t scheduler_cluster_load_acquire(const SCHEDULER_CLUSTER_SSBUF volatile uint32_t *field) {
    const uint32_t value = scheduler_cluster_load_field(field);
    scheduler_cluster_barrier();
    return value;
}

inline __aicore__ void
scheduler_cluster_store_release(SCHEDULER_CLUSTER_SSBUF volatile uint32_t *field, uint32_t value) {
    scheduler_cluster_barrier();
    scheduler_cluster_store_field(field, value);
}

// The cluster bootstrap handshake makes these fields visible to every lane.
inline __aicore__ void
scheduler_cluster_signal_clear(SCHEDULER_CLUSTER_SSBUF SchedulerClusterSignal *signal, uint64_t run_epoch) {
    scheduler_cluster_store_field(&signal->run_epoch, run_epoch);
    scheduler_cluster_store_field(&signal->task_ref, SCHEDULER_TASK_REF_INVALID);
    scheduler_cluster_store_field(&signal->anchor_state, UINT64_C(0));
    scheduler_cluster_store_field(&signal->rank, UINT32_MAX);
    scheduler_cluster_store_field(&signal->block_idx, UINT32_MAX);
    scheduler_cluster_store_field(&signal->active_mask, UINT32_C(0));
    scheduler_cluster_store_field(&signal->mix_floor, UINT32_C(0));
    scheduler_cluster_store_field(&signal->trace_generation, UINT32_C(0));
    scheduler_cluster_store_field(&signal->dependencies_generation, UINT32_C(0));
    scheduler_cluster_store_field(&signal->accounted_generation, UINT32_C(0));
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        scheduler_cluster_store_field(&signal->arrived[lane], UINT32_C(0));
        scheduler_cluster_store_field(&signal->finished[lane], UINT32_C(0));
    }
    scheduler_cluster_store_release(&signal->armed_generation, UINT32_C(0));
}
