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
#include "scheduler_completion_state.h"
#include "intrinsic.h"
#include "aicore_completion_mailbox_types.h"
#include "async_sdma_guard.h"

inline constexpr uint64_t SCHEDULER_ASYNC_INDEX_NONE = UINT64_MAX;
inline constexpr uint64_t SCHEDULER_ASYNC_LAYOUT_VERSION = 8;

// All offsets are relative to the retained async region.
struct SchedulerAsyncTaskMap {
    uint64_t group_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t finish_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint16_t block_count{0};
    uint8_t groups_per_block{0};
    uint8_t async_lane_mask{0};
    uint32_t group_count{0};
};

struct SchedulerAsyncGroupIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t finish_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t entries_offset{0};
    uint16_t block_idx{0};
    uint8_t subtask_slot{0};
    uint8_t capacity{0};
};

struct alignas(64) SchedulerAsyncGroupWriter {
    volatile uint32_t count{0};
    volatile int32_t error_code{0};
    AsyncCompletionReservationState reservation{};
};

// A local registration view excludes cache-line padding.
struct alignas(8) SchedulerAsyncWriterSnapshot {
    uint32_t count{0};
    int32_t error_code{0};
    AsyncCompletionReservationState reservation{};
};
static_assert(sizeof(SchedulerAsyncWriterSnapshot) == 32);
static_assert(offsetof(SchedulerAsyncGroupWriter, reservation) == offsetof(SchedulerAsyncWriterSnapshot, reservation));

// CHECK owns these words under its lease. They use bypass loads/stores only.
struct SchedulerAsyncGroupProgress {
    uint64_t remaining_mask{0};
    uint64_t next_condition{0};
    uint64_t phase{0};
    int64_t failure_code{0};
};

struct SchedulerAsyncFinishIdentity {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t group_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint64_t group_count{0};
};

struct SchedulerAsyncFinishProgress {
    SchedulerCompletionCursor completion{};
    int64_t failure_code{0};
    bool contributed{false};
    uint8_t reserved[7]{};
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
    uint64_t sdma_guard_context_offset{0};
    uint64_t sdma_guard_count{0};
};

static_assert(sizeof(SchedulerAsyncTaskMap) == 24);
static_assert(sizeof(SchedulerAsyncGroupIdentity) == 40 && sizeof(SchedulerAsyncGroupWriter) == 64);
static_assert(sizeof(SchedulerAsyncGroupProgress) == 32 && sizeof(SchedulerAsyncFinishIdentity) == 32);
static_assert(sizeof(SchedulerAsyncFinishProgress) == 168);
static_assert(sizeof(SchedulerAsyncLayout) == 192);
static_assert(std::is_standard_layout_v<SchedulerAsyncLayout> && std::is_trivially_copyable_v<SchedulerAsyncLayout>);
static_assert(std::is_standard_layout_v<SchedulerAsyncTaskMap> && std::is_trivially_copyable_v<SchedulerAsyncTaskMap>);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupIdentity> && std::is_trivially_copyable_v<SchedulerAsyncGroupIdentity>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupWriter> && std::is_trivially_copyable_v<SchedulerAsyncGroupWriter>
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
