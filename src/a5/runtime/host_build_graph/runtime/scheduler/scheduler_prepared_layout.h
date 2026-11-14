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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel_args.h"

inline constexpr uint64_t SCHEDULER_PREPARED_ALIGNMENT = 128;

inline constexpr uint64_t SCHEDULER_PREPARED_VERSION = 4;

enum class SchedulerPreparedPhase : uint8_t {
    INVALID,
    BUILT,
    CLAIMED,
    RETURNED,
    FAILED,
    PARKING,
    PARKED,
    GROUP_RESERVED,
    GROUP_ARMED
};

inline constexpr uint64_t SCHEDULER_PREPARED_BINDING_READY = UINT64_C(1) << 24;

inline __aicore__ constexpr uint64_t
scheduler_prepared_state(SchedulerPreparedPhase phase, uint32_t executor = UINT32_MAX, uint32_t generation = 0) {
    return static_cast<uint64_t>(phase) | (executor == UINT32_MAX ? 0 : ((static_cast<uint64_t>(executor) + 1) << 8)) |
           (static_cast<uint64_t>(generation) << 32);
}

inline __aicore__ constexpr SchedulerPreparedPhase scheduler_prepared_phase(uint64_t state) {
    return static_cast<SchedulerPreparedPhase>(state & UINT64_C(255));
}

inline __aicore__ constexpr uint32_t scheduler_prepared_owner(uint64_t state) {
    return static_cast<uint32_t>((state >> 8) & UINT64_C(65535)) - 1;
}

inline __aicore__ constexpr uint32_t scheduler_prepared_generation(uint64_t state) {
    return static_cast<uint32_t>(state >> 32);
}

inline __aicore__ constexpr uint32_t scheduler_prepared_execute_owner(uint64_t state) {
    const auto phase = scheduler_prepared_phase(state);
    return phase == SchedulerPreparedPhase::CLAIMED || phase == SchedulerPreparedPhase::RETURNED ?
               scheduler_prepared_owner(state) :
               UINT32_MAX;
}

// Plans are indexed by the existing graph node ref, including non-executable nodes.
struct SchedulerExecutionPlan {
    uint64_t execution_begin;
    uint32_t execution_count;
    uint16_t block_num;
    uint8_t active_mask_bits;
    uint8_t active_lane_count;
};

// Phase, owner and transfer generation share one atomic word.
// Cached stores never touch this line during a run.
struct alignas(128) SchedulerPreparedControl {
    volatile uint64_t state;
};

struct alignas(64) SchedulerPreparedCall {
    uint64_t function_bin_addr;
    uint64_t run_epoch;
    uint64_t execution_index;
    int64_t task_ref;
    uint32_t block_idx;
    uint16_t num_args;
    uint8_t subtask_slot;
    uint8_t reserved;
    uint64_t reserved_header[3];
    uint64_t args[DISPATCH_MAX_ARGS];
};

struct alignas(64) SchedulerExecutionBinding {
    LocalContext local_context;
    alignas(64) GlobalContext global_context;
};

struct alignas(64) SchedulerPreparedLayout {
    uint64_t run_epoch;
    uint64_t task_count;
    uint64_t execution_count;
    uint64_t plans_offset;
    uint64_t controls_offset;
    uint64_t calls_offset;
    uint64_t bindings_offset;
    uint64_t version{SCHEDULER_PREPARED_VERSION};
};

static_assert(sizeof(SchedulerExecutionPlan) == 16);
static_assert(sizeof(SchedulerPreparedControl) == 128 && alignof(SchedulerPreparedControl) == 128);
static_assert(sizeof(SchedulerPreparedCall) == 512 && offsetof(SchedulerPreparedCall, args) == 64);
static_assert(sizeof(SchedulerExecutionBinding) == 128 && offsetof(SchedulerExecutionBinding, global_context) == 64);
static_assert(sizeof(SchedulerPreparedLayout) == 64);
static_assert(
    std::is_trivially_copyable_v<SchedulerExecutionPlan> && std::is_standard_layout_v<SchedulerExecutionPlan>
);
static_assert(
    std::is_trivially_copyable_v<SchedulerPreparedControl> && std::is_standard_layout_v<SchedulerPreparedControl>
);
static_assert(std::is_trivially_copyable_v<SchedulerPreparedCall> && std::is_standard_layout_v<SchedulerPreparedCall>);
static_assert(
    std::is_trivially_copyable_v<SchedulerExecutionBinding> && std::is_standard_layout_v<SchedulerExecutionBinding>
);
static_assert(
    std::is_trivially_copyable_v<SchedulerPreparedLayout> && std::is_standard_layout_v<SchedulerPreparedLayout>
);
