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

#include "async_completion_types.h"
#include "scheduler_async_state.h"
#include "scheduler_resolve_state.h"

struct alignas(64) SchedulerAsyncLayout {
    uint64_t run_epoch;
    uint64_t image_bytes;
    uint64_t task_map_offset;
    uint64_t task_data_offset;
    uint64_t groups_offset;
    uint64_t group_leases_offset;
    uint64_t group_progress_offset;
    uint64_t joins_offset;
    uint64_t finish_leases_offset;
    uint64_t finish_progress_offset;
    uint64_t check_directory_offset;
    uint64_t finish_directory_offset;
    uint32_t task_count;
    uint32_t async_task_count;
    uint32_t group_count;
    uint32_t sdma_guard_count;
    uint64_t sdma_guards_offset;
    uint64_t activity_offset;
};

struct SchedulerAsyncTaskData {
    int64_t task_ref;
    uint32_t group_base;
    uint16_t block_count;
    LaneMask async_mask;
};

struct alignas(64) SchedulerAsyncGroupProgress {
    uint64_t pending_mask;
    uint32_t condition_cursor;
    uint32_t acquire_cursor;
};

struct alignas(64) SchedulerAsyncFinishProgress {
    SchedulerResolveCursor resolver;
};

struct alignas(64) SchedulerAsyncDirectoryWord {
    volatile uint64_t bits;
};

struct alignas(64) SchedulerAsyncActivity {
    volatile uint64_t activated;
};

// Explicit field copies work for private/GM address spaces and preserve the
// complete continuation, including a batch not yet published by its owner.
template <typename Destination, typename Source>
inline __aicore__ void scheduler_async_copy_resolver(Destination &destination, const Source &source) {
    destination.producer_ref = source.producer_ref;
    destination.next_waiter = source.next_waiter;
    destination.waiter_ref = source.waiter_ref;
    destination.saved_next = source.saved_next;
    destination.route.task_ref = source.route.task_ref;
    destination.route.producer_ref = source.route.producer_ref;
    destination.route.next_fanin = source.route.next_fanin;
    destination.route.fanin_count = source.route.fanin_count;
    destination.route.result = source.route.result;
    destination.publication.inbox_offset = source.publication.inbox_offset;
    destination.publication.count = source.publication.count;
    destination.publication.destination = source.publication.destination;
    destination.publication.target = source.publication.target;
    destination.publication.phase = source.publication.phase;
    destination.phase = source.phase;
    destination.visited_count = source.visited_count;
    for (uint32_t i = 0; i < SCHEDULER_READY_PUBLICATION_CAPACITY; ++i)
        destination.publication.refs[i] = source.publication.refs[i];
}

static_assert(sizeof(SchedulerAsyncLayout) == 128 && alignof(SchedulerAsyncLayout) == 64);
static_assert(offsetof(SchedulerAsyncLayout, task_count) == 96);
static_assert(offsetof(SchedulerAsyncLayout, activity_offset) == 120);
static_assert(sizeof(SchedulerAsyncActivity) == 64 && alignof(SchedulerAsyncActivity) == 64);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncActivity> && std::is_trivially_copyable_v<SchedulerAsyncActivity>
);
static_assert(std::is_standard_layout_v<SchedulerAsyncLayout> && std::is_trivially_copyable_v<SchedulerAsyncLayout>);
static_assert(sizeof(SchedulerAsyncTaskData) == 16 && alignof(SchedulerAsyncTaskData) == 8);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncTaskData> && std::is_trivially_copyable_v<SchedulerAsyncTaskData>
);
static_assert(sizeof(SchedulerAsyncGroupProgress) == 64 && alignof(SchedulerAsyncGroupProgress) == 64);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncGroupProgress> && std::is_trivially_copyable_v<SchedulerAsyncGroupProgress>
);
static_assert(sizeof(SchedulerAsyncFinishProgress) == 192 && alignof(SchedulerAsyncFinishProgress) == 64);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncFinishProgress> &&
    std::is_trivially_copyable_v<SchedulerAsyncFinishProgress>
);
static_assert(sizeof(SchedulerAsyncDirectoryWord) == 64 && alignof(SchedulerAsyncDirectoryWord) == 64);

inline __aicore__ uint64_t scheduler_async_condition_mask(uint32_t count) {
    return count == 64 ? UINT64_MAX : count < 64 ? (UINT64_C(1) << count) - 1 : 0;
}
