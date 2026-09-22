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
/**
 * The async wait list lives inside SchedulerState, which is reserved in the
 * arena's device-only zone: no constructor runs over it and no image is copied
 * onto it, so it reaches the AICPU holding whatever the pooled allocation last
 * had. init_data_from_layout is what makes it an empty list.
 *
 * host_build_graph only: the tensormap_and_ringbuffer wait list travels with
 * that runtime's uploaded image.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "scheduler/scheduler.h"

namespace {

// Scheduler state on memory holding a prior generation's bytes, which is what
// the pooled arena hands the AICPU. Deliberately no constructor and no
// placement-new: the product reaches this object through region_ptr + a cast,
// and value-initializing here would supply exactly the reset under test.
class DirtySchedulerState {
public:
    explicit DirtySchedulerState(uint8_t fill) :
        raw_(::operator new(sizeof(SchedulerState), std::align_val_t{alignof(SchedulerState)})) {
        std::memset(raw_, fill, sizeof(SchedulerState));
    }

    ~DirtySchedulerState() { ::operator delete(raw_, std::align_val_t{alignof(SchedulerState)}); }

    DirtySchedulerState(const DirtySchedulerState &) = delete;
    DirtySchedulerState &operator=(const DirtySchedulerState &) = delete;

    SchedulerState *get() { return reinterpret_cast<SchedulerState *>(raw_); }

private:
    void *raw_;
};

}  // namespace

// The window this closes: a residual `count` is a length the resolution
// thread's poll trusts. It reads entries[count - 1] before anything has written
// an entry, so a fill large enough to land past the region faults on a wild
// address instead of reporting an empty list (issue #2121).
TEST(HbgAsyncWaitInit, DirtyDeviceZoneStartsEmpty) {
    DeviceArena scheduler_arena;
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm_handle, nullptr);
    const SchedulerLayout layout = SchedulerState::reserve_layout(scheduler_arena);
    DirtySchedulerState dirty(0xAA);
    SchedulerState *sched = dirty.get();
    ASSERT_NE(sched->async_wait_list.count, 0) << "the fill must reach the wait list for this to test anything";

    ASSERT_TRUE(sched->init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    EXPECT_EQ(sched->async_wait_list.count, 0);
    EXPECT_EQ(sched->async_wait_list.busy.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(sched->async_wait_list.mpsc_skipped_count.load(std::memory_order_relaxed), 0U);
}

// A residual `busy` is the quiet half of the same miss: the poll takes the list
// only through try_lock, so a non-zero fill would make every drain a no-op and
// strand deferred completions rather than crash.
TEST(HbgAsyncWaitInit, DirtyDeviceZoneYieldsATakeableLock) {
    DeviceArena scheduler_arena;
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm_handle, nullptr);
    const SchedulerLayout layout = SchedulerState::reserve_layout(scheduler_arena);
    DirtySchedulerState dirty(0xFF);
    SchedulerState *sched = dirty.get();

    ASSERT_TRUE(sched->init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    EXPECT_TRUE(sched->async_wait_list.try_lock());
    sched->async_wait_list.unlock();
    EXPECT_TRUE(sched->async_wait_list.try_lock());
}

// Re-binding the same pooled region is the ordinary case, so the reset must
// also recover a list an earlier run left holding entries.
TEST(HbgAsyncWaitInit, RebindRecoversAListLeftPopulated) {
    DeviceArena scheduler_arena;
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm_handle, nullptr);
    const SchedulerLayout layout = SchedulerState::reserve_layout(scheduler_arena);
    DirtySchedulerState dirty(0x00);
    SchedulerState *sched = dirty.get();
    ASSERT_TRUE(sched->init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    sched->async_wait_list.count = 3;
    ASSERT_TRUE(sched->async_wait_list.try_lock());

    ASSERT_TRUE(sched->init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    EXPECT_EQ(sched->async_wait_list.count, 0);
    EXPECT_TRUE(sched->async_wait_list.try_lock());
}
