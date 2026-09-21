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
 * Unit tests for host_build_graph's TaskAllocator (ring_buffer.h).
 *
 * host_build_graph is whole-graph-resident: the device runs only after the host
 * has built the whole graph, so neither the task ring nor the graph heap is ever
 * reclaimed while allocation is in progress. Both are therefore forward-only
 * bump allocators, which gives this runtime a different contract from
 * tensormap_and_ringbuffer's reclaiming allocator (tested separately in
 * test_task_allocator.cpp):
 *
 * - There is no wrap-around and no reclaim pointer. heap_available() is simply
 *   the bytes between the top and the end of the heap.
 * - Exhaustion of either resource is terminal. alloc() reports it and returns a
 *   failed result immediately — it never spins waiting for a reclaim that
 *   cannot arrive, and there is no wall-clock deadlock backstop.
 * - Zero-size allocation is a no-op returning the current top. Two consecutive
 *   zero-size allocs return the SAME pointer.
 * - Task ids start at 0 and are capped by the capacity, so they cannot approach
 *   INT32_MAX, and each id is its own task-table slot.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <set>

#include "host_build_graph/task_allocator.h"

class HbgTaskAllocatorTest : public ::testing::Test {
protected:
    static constexpr int32_t MAX_TASKS = 16;
    static constexpr uint64_t HEAP_SIZE = 4096;

    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<int32_t> error_code{SIMPLER_ERROR_NONE};
    TaskAllocator allocator{};

    void SetUp() override {
        std::memset(heap_buf, 0, sizeof(heap_buf));
        error_code.store(SIMPLER_ERROR_NONE);
        allocator.init(MAX_TASKS, heap_buf, HEAP_SIZE, &error_code);
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(HbgTaskAllocatorTest, InitialState) {
    EXPECT_EQ(allocator.capacity(), MAX_TASKS);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_EQ(allocator.heap_top(), 0u);
    EXPECT_EQ(allocator.heap_capacity(), HEAP_SIZE);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE);
    EXPECT_EQ(allocator.heap_used_bytes(), 0u);
}

TEST_F(HbgTaskAllocatorTest, AllocNonZeroSize) {
    auto result = allocator.alloc(100);
    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_NE(result.packed_base, nullptr);
    uint64_t expected_aligned = CHIP_ALIGN_UP(100u, CHIP_ALIGN_SIZE);
    EXPECT_EQ(expected_aligned, 128u);
    EXPECT_EQ(allocator.heap_top(), expected_aligned);
    EXPECT_EQ(
        static_cast<char *>(result.packed_end) - static_cast<char *>(result.packed_base),
        static_cast<ptrdiff_t>(expected_aligned)
    );
}

TEST_F(HbgTaskAllocatorTest, SequentialTaskIds) {
    int32_t prev_id = -1;
    for (int i = 0; i < 5; i++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(result.task_id, prev_id + 1) << "Task IDs must be monotonically increasing";
        prev_id = result.task_id;
    }
    EXPECT_EQ(allocator.active_count(), 5) << "the next id is both the occupancy and the run's total";
}

TEST_F(HbgTaskAllocatorTest, OutputSizeAlignment) {
    ASSERT_FALSE(allocator.alloc(1).failed());
    EXPECT_EQ(allocator.heap_top(), 64u);

    ASSERT_FALSE(allocator.alloc(33).failed());
    EXPECT_EQ(allocator.heap_top(), 128u);

    ASSERT_FALSE(allocator.alloc(64).failed());
    EXPECT_EQ(allocator.heap_top(), 192u);
}

// A task id IS its task-table index: ids run 0..capacity-1 with no wrap, so the
// whole table is addressed exactly once and no two tasks share a slot.
TEST_F(HbgTaskAllocatorTest, TaskIdIsItsOwnSlot) {
    std::set<int32_t> ids;
    for (int i = 0; i < MAX_TASKS; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed());
        EXPECT_EQ(r.task_id, i) << "ids are handed out in order, with no wrap";
        ids.insert(r.task_id);
    }
    EXPECT_EQ(ids.size(), static_cast<size_t>(MAX_TASKS)) << "Every slot is usable exactly once";
    EXPECT_TRUE(allocator.alloc(0).failed()) << "the capacity is terminal, not a wrap point";
}

// No slot lookup masks with the capacity, so it need not be a power of two: an odd
// count hands out exactly that many ids and then reports exhaustion. This is what
// lets a bind pass any runtime_env.ring_task_window through without rounding it,
// up to the TaskId::GLOBAL_TASK_MAX_NUM ceiling resolve_graph_task_capacity holds
// it to.
TEST_F(HbgTaskAllocatorTest, NonPowerOfTwoCapacitySaturatesExactly) {
    constexpr int32_t ODD_CAPACITY = 10;
    TaskAllocator odd{};
    odd.init(ODD_CAPACITY, heap_buf, HEAP_SIZE, &error_code);
    EXPECT_EQ(odd.capacity(), ODD_CAPACITY);

    for (int32_t i = 0; i < ODD_CAPACITY; i++) {
        auto r = odd.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i) << "ids run 0..capacity-1 for an odd capacity too";
    }
    EXPECT_EQ(odd.active_count(), ODD_CAPACITY);

    auto overflow = odd.alloc(0);
    EXPECT_TRUE(overflow.failed()) << "the odd capacity is the cap, not rounded up to a power of two";
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK);
}

// Zero-size allocs return the same address and don't advance the top.
TEST_F(HbgTaskAllocatorTest, ZeroSizeAllocationAliased) {
    auto r1 = allocator.alloc(0);
    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    ASSERT_FALSE(r2.failed());

    EXPECT_EQ(r1.packed_base, r2.packed_base) << "Zero-size allocs return same address";
    EXPECT_EQ(r1.packed_base, r1.packed_end) << "packed_end == packed_base for zero-size";
    EXPECT_EQ(allocator.heap_top(), 0u) << "top doesn't advance for zero-size allocs";
}

// =============================================================================
// Forward-only heap: no reclaim, no wrap
// =============================================================================

TEST_F(HbgTaskAllocatorTest, HeapAvailableIsRemainderAboveTop) {
    ASSERT_FALSE(allocator.alloc(256).failed());
    EXPECT_EQ(allocator.heap_top(), 256u);
    EXPECT_EQ(allocator.heap_used_bytes(), 256u);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 256u);
}

TEST_F(HbgTaskAllocatorTest, HeapExactFitAtEnd) {
    ASSERT_FALSE(allocator.alloc(HEAP_SIZE - 128).failed());
    auto r = allocator.alloc(128);
    ASSERT_FALSE(r.failed()) << "An allocation filling the heap exactly must succeed";
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);
    EXPECT_EQ(allocator.heap_available(), 0u);
}

// The bytes an earlier task occupies are never handed back, so a second
// allocation that would only fit by reusing them fails rather than wrapping.
TEST_F(HbgTaskAllocatorTest, ConsumedBytesAreNotReclaimed) {
    ASSERT_FALSE(allocator.alloc(HEAP_SIZE - 64).failed());
    EXPECT_EQ(allocator.heap_available(), 64u);

    auto r = allocator.alloc(128);
    EXPECT_TRUE(r.failed()) << "No wrap-around: the heap does not recycle mid-run";
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
}

// =============================================================================
// Exhaustion is terminal and immediate
// =============================================================================

TEST_F(HbgTaskAllocatorTest, AllocExactlyHeapSize) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "No space after full allocation";
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(HbgTaskAllocatorTest, AllocLargerThanHeap) {
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed()) << "Cannot allocate more than heap size";
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(HbgTaskAllocatorTest, TaskCapacitySaturates) {
    for (int i = 0; i < MAX_TASKS; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i);
    }
    EXPECT_EQ(allocator.active_count(), MAX_TASKS);

    auto overflow = allocator.alloc(0);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(allocator.active_count(), MAX_TASKS) << "A rejected allocation must not consume a slot";
}

// A failing alloc leaves the heap pointer untouched, so the reported figures
// still describe the state the caller can act on.
TEST_F(HbgTaskAllocatorTest, FailedHeapAllocLeavesStateUnchanged) {
    ASSERT_FALSE(allocator.alloc(1024).failed());
    uint64_t top_before = allocator.heap_top();
    int32_t count_before = allocator.active_count();

    EXPECT_TRUE(allocator.alloc(HEAP_SIZE).failed());
    EXPECT_EQ(allocator.heap_top(), top_before) << "Heap pointer must not move on failure";
    EXPECT_EQ(allocator.active_count(), count_before) << "No task slot is consumed and no task id handed out";
}

// Once a fatal is latched, alloc() short-circuits without overwriting the first
// error code — the caller propagates the original cause.
TEST_F(HbgTaskAllocatorTest, LatchedFatalShortCircuitsAlloc) {
    error_code.store(SIMPLER_ERROR_INVALID_ARGS);

    auto r = allocator.alloc(64);
    EXPECT_TRUE(r.failed());
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_INVALID_ARGS) << "The first error code must survive";
    EXPECT_EQ(allocator.heap_top(), 0u);
    EXPECT_EQ(allocator.active_count(), 0);
}

// =============================================================================
// Deferred heap reservation
// =============================================================================

// An outer Graph shell is submitted before its Definition exists, so its heap
// block is carved afterwards. Nothing is reclaimed during a run, so a block
// handed out after later tasks already took theirs is still sound.
TEST_F(HbgTaskAllocatorTest, ReserveDeferredHeapCarvesAfterLaterAllocations) {
    ASSERT_FALSE(allocator.alloc(0).failed());
    ASSERT_FALSE(allocator.alloc(256).failed());
    const uint64_t top_before = allocator.heap_top();

    void *base = nullptr;
    void *end = nullptr;
    ASSERT_TRUE(allocator.reserve_deferred_heap(512, &base, &end));
    EXPECT_EQ(base, static_cast<void *>(heap_buf + top_before));
    EXPECT_EQ(end, static_cast<void *>(heap_buf + top_before + 512));
    EXPECT_EQ(allocator.heap_top(), top_before + 512);
    EXPECT_EQ(allocator.active_count(), 2) << "A deferred reservation claims no task-window slot";
}

TEST_F(HbgTaskAllocatorTest, ReserveDeferredHeapZeroSizeReturnsCurrentTop) {
    ASSERT_FALSE(allocator.alloc(128).failed());

    void *base = nullptr;
    void *end = nullptr;
    ASSERT_TRUE(allocator.reserve_deferred_heap(0, &base, &end));
    EXPECT_EQ(base, static_cast<void *>(heap_buf + 128));
    EXPECT_EQ(base, end);
    EXPECT_EQ(allocator.heap_top(), 128u);
}

TEST_F(HbgTaskAllocatorTest, ReserveDeferredHeapFailsWithoutMutatingState) {
    ASSERT_FALSE(allocator.alloc(256).failed());
    const uint64_t top_before = allocator.heap_top();

    void *base = nullptr;
    void *end = nullptr;
    EXPECT_FALSE(allocator.reserve_deferred_heap(static_cast<int32_t>(HEAP_SIZE), &base, &end));
    EXPECT_EQ(allocator.heap_top(), top_before);
    EXPECT_EQ(base, nullptr);
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_NONE) << "A rejected reservation is not a fatal";
}

TEST_F(HbgTaskAllocatorTest, LatchedFatalShortCircuitsReserveDeferredHeap) {
    error_code.store(SIMPLER_ERROR_INVALID_ARGS);

    void *base = nullptr;
    void *end = nullptr;
    EXPECT_FALSE(allocator.reserve_deferred_heap(64, &base, &end));
    EXPECT_EQ(allocator.heap_top(), 0u);
}

// What the production path passes: the graph heap is allocated out of the virtual
// window, because its device region is committed only once this allocator has
// revealed how many bytes the graph needs. The window is bounded by MAX_HEAP_CAPACITY
// rather than by a configured size, so a graph is limited by what the device can
// commit afterwards.
TEST_F(HbgTaskAllocatorTest, AcceptsTheVirtualHeapWindow) {
    TaskAllocator virtual_heap{};
    auto *base = reinterpret_cast<void *>(HEAP_VIRTUAL_BASE);
    virtual_heap.init(MAX_TASKS, base, MAX_HEAP_CAPACITY, &error_code);
    EXPECT_EQ(virtual_heap.heap_capacity(), MAX_HEAP_CAPACITY);

    auto r = virtual_heap.alloc(64);
    ASSERT_FALSE(r.failed());
    EXPECT_EQ(reinterpret_cast<uint64_t>(r.packed_base), HEAP_VIRTUAL_BASE);
    EXPECT_EQ(virtual_heap.heap_used_bytes(), 64u);
}
