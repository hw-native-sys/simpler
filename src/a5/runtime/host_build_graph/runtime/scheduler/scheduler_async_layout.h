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

#include "scheduler_async_directory.h"
#include "scheduler_async_join.h"
#include "scheduler_async_lease.h"
#include "scheduler_resolve.h"
#include "aicore_completion_mailbox_types.h"

inline constexpr uint64_t SCHEDULER_ASYNC_INDEX_NONE = UINT64_MAX;
inline constexpr uint64_t SCHEDULER_ASYNC_LAYOUT_VERSION = 1;

// All offsets are relative to the async region; only external resources may
// carry absolute device addresses in their separately validated binding table.
struct SchedulerAsyncTaskMap {
    uint64_t group_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t finish_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint32_t block_count{0};
    uint32_t groups_per_block{0};
    uint32_t async_lane_mask{0};
    uint32_t group_count{0};
};

struct alignas(128) SchedulerAsyncGroupIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t finish_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t entries_offset{0};
    uint64_t condition_bindings_offset{0};
    uint64_t binding_uses_offset{0};
    uint64_t binding_use_count{0};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    uint32_t capacity{0};
    uint32_t condition_types{0};
    uint8_t reserved[56]{};
};

// This line has one kernel writer and is sealed before its lease is published.
// It is separate from the atomic-only lease and the later owner's progress.
struct alignas(128) SchedulerAsyncGroupWriter {
    volatile uint32_t count{0};
    volatile int32_t error_code{0};
    uint32_t reserved_count{0};
    uint32_t sealed{0};
    uint8_t reserved[112]{};
};

struct SchedulerAsyncConditionBinding {
    uint64_t binding_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t owner_generation{0};
};

struct alignas(128) SchedulerAsyncGroupProgress {
    uint64_t remaining_mask{0};
    uint64_t next_condition{0};
    uint64_t acquire_stage{0};
    uint64_t acquire_byte_offset{0};
    uint64_t observation_count{0};
    uint64_t phase{0};
    uint64_t acquire_condition{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t acquire_count{0};
    uint64_t work_units{0};
    uint64_t slice_count{0};
    int64_t failure_code{0};
    uint8_t reserved[40]{};
};

struct alignas(128) SchedulerAsyncFinishIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t group_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t group_count{0};
    uint8_t reserved[96]{};
};

struct alignas(128) SchedulerAsyncFinishProgress {
    SchedulerResolveCursor resolve{};
    int64_t current_waiter{SCHEDULER_TASK_REF_INVALID};
    int64_t detached_next{SCHEDULER_TASK_REF_INVALID};
    uint64_t phase{0};
    uint64_t pending_count{0};
    uint64_t pending_queue{0};
    uint64_t pending_target{0};
    uint64_t pending_kind{0};
    uint64_t resume_phase{0};
    int64_t pending_first{SCHEDULER_TASK_REF_INVALID};
    int64_t pending_last{SCHEDULER_TASK_REF_INVALID};
    uint64_t pending_link_count{0};
    uint64_t visited_count{0};
    int64_t pending_tasks[8]{};
    uint64_t active_slice_cycles{0};
    uint64_t slice_count{0};
    uint64_t work_units{0};
    uint8_t reserved[40]{};
};

struct SchedulerAsyncLayout {
    uint64_t version{SCHEDULER_ASYNC_LAYOUT_VERSION};
    uint64_t total_bytes{0};
    uint64_t run_epoch{0};
    uint64_t task_count{0};
    uint64_t group_count{0};
    uint64_t finish_count{0};
    uint64_t condition_capacity{0};
    uint64_t task_map_offset{0};
    uint64_t group_identity_offset{0};
    uint64_t group_writer_offset{0};
    uint64_t group_lease_offset{0};
    uint64_t group_progress_offset{0};
    uint64_t finish_identity_offset{0};
    uint64_t finish_lease_offset{0};
    uint64_t finish_join_offset{0};
    uint64_t finish_progress_offset{0};
    uint64_t check_directory_offset{0};
    uint64_t check_directory_words{0};
    uint64_t finish_directory_offset{0};
    uint64_t finish_directory_words{0};
    uint64_t entries_offset{0};
    uint64_t entries_bytes{0};
    uint64_t condition_bindings_offset{0};
    uint64_t condition_bindings_bytes{0};
    uint64_t resources_offset{0};
    uint64_t resources_bytes{0};
    uint64_t uses_offset{0};
    uint64_t uses_bytes{0};
    uint64_t consumers_offset{0};
    uint64_t consumers_bytes{0};
    uint64_t scratch_offset{0};
    uint64_t scratch_bytes{0};
};

static_assert(sizeof(SchedulerAsyncTaskMap) == 32);
static_assert(sizeof(SchedulerAsyncGroupIdentity) == 128 && sizeof(SchedulerAsyncGroupWriter) == 128);
static_assert(sizeof(SchedulerAsyncGroupProgress) == 128 && sizeof(SchedulerAsyncFinishIdentity) == 128);
static_assert(sizeof(SchedulerAsyncFinishProgress) == 256);
static_assert(sizeof(SchedulerAsyncConditionBinding) == 16);
static_assert(sizeof(SchedulerAsyncLayout) == 256);
static_assert(std::is_standard_layout_v<SchedulerAsyncLayout> && std::is_trivially_copyable_v<SchedulerAsyncLayout>);
static_assert(std::is_standard_layout_v<SchedulerAsyncTaskMap> && std::is_trivially_copyable_v<SchedulerAsyncTaskMap>);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupIdentity> && std::is_trivially_copyable_v<SchedulerAsyncGroupIdentity>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupWriter> && std::is_trivially_copyable_v<SchedulerAsyncGroupWriter>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncConditionBinding> &&
    std::is_trivially_copyable_v<SchedulerAsyncConditionBinding>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupProgress> && std::is_trivially_copyable_v<SchedulerAsyncGroupProgress>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncFinishIdentity> &&
    std::is_trivially_copyable_v<SchedulerAsyncFinishIdentity>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncFinishProgress> &&
    std::is_trivially_copyable_v<SchedulerAsyncFinishProgress>
);

// The slot belongs to the task, not the worker's physical lane: a single AIV0
// subtask may execute on a worker occupying physical AIV1.
inline __aicore__ uint64_t
scheduler_async_group_index(const SchedulerAsyncTaskMap &map, uint32_t block, uint32_t subtask_slot) {
    if (subtask_slot >= 3 || block >= map.block_count || map.group_base == SCHEDULER_ASYNC_INDEX_NONE ||
        (map.async_lane_mask & (1u << subtask_slot)) == 0)
        return SCHEDULER_ASYNC_INDEX_NONE;
    const uint32_t preceding =
        static_cast<uint32_t>(__builtin_popcount(map.async_lane_mask & ((1u << subtask_slot) - 1)));
    return map.group_base + static_cast<uint64_t>(block) * map.groups_per_block + preceding;
}
