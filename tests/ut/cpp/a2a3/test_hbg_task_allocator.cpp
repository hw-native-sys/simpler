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
 * Unit tests for host_build_graph's PTO2TaskAllocator (pto_ring_buffer.h).
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
 * - Task ids start at 0 and are capped by the window, so they cannot approach
 *   INT32_MAX.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <set>

#include "pto_ring_buffer.h"
#include "task_interface/assert_compat.h"

class HbgTaskAllocatorTest : public ::testing::Test {
protected:
    static constexpr int32_t TASK_CAPACITY = 16;
    static constexpr uint64_t HEAP_SIZE = 4096;

    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator{};

    void SetUp() override {
        std::memset(heap_buf, 0, sizeof(heap_buf));
        current_index.store(0);
        error_code.store(PTO2_ERROR_NONE);
        allocator.init(TASK_CAPACITY, &current_index, heap_buf, HEAP_SIZE, &error_code);
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(HbgTaskAllocatorTest, InitialState) {
    EXPECT_EQ(allocator.task_capacity(), TASK_CAPACITY);
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
    EXPECT_EQ(result.slot, 0);
    EXPECT_NE(result.packed_base, nullptr);
    uint64_t expected_aligned = PTO2_ALIGN_UP(100u, PTO2_ALIGN_SIZE);
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
        EXPECT_EQ(result.slot, result.task_id & (TASK_CAPACITY - 1));
        prev_id = result.task_id;
    }
    EXPECT_EQ(allocator.active_count(), 5);
    EXPECT_EQ(current_index.load(), 5) << "Head is published to shared memory";
}

TEST_F(HbgTaskAllocatorTest, OutputSizeAlignment) {
    ASSERT_FALSE(allocator.alloc(1).failed());
    EXPECT_EQ(allocator.heap_top(), 64u);

    ASSERT_FALSE(allocator.alloc(33).failed());
    EXPECT_EQ(allocator.heap_top(), 128u);

    ASSERT_FALSE(allocator.alloc(64).failed());
    EXPECT_EQ(allocator.heap_top(), 192u);
}

TEST_F(HbgTaskAllocatorTest, SlotMappingPowerOfTwoWindow) {
    std::set<int32_t> slots;
    for (int i = 0; i < TASK_CAPACITY; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed());
        EXPECT_EQ(r.slot, r.task_id & (TASK_CAPACITY - 1));
        slots.insert(r.slot);
    }
    EXPECT_EQ(slots.size(), static_cast<size_t>(TASK_CAPACITY)) << "Every configured slot is usable exactly once";
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
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
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
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(HbgTaskAllocatorTest, AllocLargerThanHeap) {
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed()) << "Cannot allocate more than heap size";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(HbgTaskAllocatorTest, TaskCapacitySaturates) {
    for (int i = 0; i < TASK_CAPACITY; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i);
    }
    EXPECT_EQ(allocator.active_count(), TASK_CAPACITY);
    EXPECT_EQ(current_index.load(), TASK_CAPACITY);

    auto overflow = allocator.alloc(0);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(allocator.active_count(), TASK_CAPACITY);
    EXPECT_EQ(current_index.load(), TASK_CAPACITY) << "A rejected allocation must not publish a new task";
}

// A failing alloc leaves the heap pointer untouched, so the reported figures
// still describe the state the caller can act on.
TEST_F(HbgTaskAllocatorTest, FailedHeapAllocLeavesStateUnchanged) {
    ASSERT_FALSE(allocator.alloc(1024).failed());
    uint64_t top_before = allocator.heap_top();
    int32_t count_before = allocator.active_count();
    int32_t current_index_before = current_index.load();

    EXPECT_TRUE(allocator.alloc(HEAP_SIZE).failed());
    EXPECT_EQ(allocator.heap_top(), top_before) << "Heap pointer must not move on failure";
    EXPECT_EQ(allocator.active_count(), count_before) << "No task slot is consumed on failure";
    EXPECT_EQ(current_index.load(), current_index_before) << "No task index is published on failure";
}

// Once a fatal is latched, alloc() short-circuits without overwriting the first
// error code — the caller propagates the original cause.
TEST_F(HbgTaskAllocatorTest, LatchedFatalShortCircuitsAlloc) {
    error_code.store(PTO2_ERROR_INVALID_ARGS);

    auto r = allocator.alloc(64);
    EXPECT_TRUE(r.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_INVALID_ARGS) << "The first error code must survive";
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
    EXPECT_EQ(allocator.active_count(), 2) << "A deferred reservation claims no task-capacity slot";
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
    EXPECT_EQ(error_code.load(), PTO2_ERROR_NONE) << "A rejected reservation is not a fatal";
}

TEST_F(HbgTaskAllocatorTest, LatchedFatalShortCircuitsReserveDeferredHeap) {
    error_code.store(PTO2_ERROR_INVALID_ARGS);

    void *base = nullptr;
    void *end = nullptr;
    EXPECT_FALSE(allocator.reserve_deferred_heap(64, &base, &end));
    EXPECT_EQ(allocator.heap_top(), 0u);
}

// Graph recording addresses its internal nodes' outputs from
// GRAPH_RECORD_VIRTUAL_BASE upward and classifies internal vs boundary tensor
// sources by address-range containment alone. A real heap that reached into that
// range would silently misclassify, so init() refuses it.
TEST_F(HbgTaskAllocatorTest, InitRejectsAHeapOverlappingTheRecordingVirtualRange) {
    PTO2TaskAllocator overlapping{};
    auto *base = reinterpret_cast<void *>(GRAPH_RECORD_VIRTUAL_BASE);
    EXPECT_THROW(overlapping.init(TASK_CAPACITY, &current_index, base, HEAP_SIZE, &error_code), AssertionError);

    PTO2TaskAllocator straddling{};
    auto *just_below = reinterpret_cast<void *>(GRAPH_RECORD_VIRTUAL_BASE - 64);
    EXPECT_THROW(straddling.init(TASK_CAPACITY, &current_index, just_below, HEAP_SIZE, &error_code), AssertionError);
}
