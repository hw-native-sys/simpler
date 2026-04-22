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
 * Unit tests for PTO2TaskAllocator and PTO2DepListPool from pto_ring_buffer.h
 *
 * Tests ring buffer allocation, heap bump logic, dependency list pool,
 * and known boundary conditions including a bug candidate in
 * try_bump_heap wrap-around when tail == alloc_size.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "pto_ring_buffer.h"

// =============================================================================
// Helpers
// =============================================================================

static constexpr int32_t kWindowSize = 16;   // Power of 2, small for testing
static constexpr uint64_t kHeapSize = 1024;  // Small heap for boundary testing

/**
 * Test fixture for PTO2TaskAllocator tests.
 *
 * Sets up a descriptor array, heap buffer, and atomic flow-control variables.
 * last_alive starts at 0, so tasks 0..window_size-2 can be allocated before
 * the ring is considered full (active = local_task_id - last_alive + 1 < window_size).
 */
class TaskAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        descriptors_.resize(kWindowSize);
        std::memset(descriptors_.data(), 0, sizeof(PTO2TaskDescriptor) * kWindowSize);
        heap_buf_.resize(kHeapSize, 0);

        current_index_.store(0, std::memory_order_relaxed);
        last_alive_.store(0, std::memory_order_relaxed);
        error_code_.store(0, std::memory_order_relaxed);

        allocator_.init(
            descriptors_.data(), kWindowSize, &current_index_, &last_alive_, heap_buf_.data(), kHeapSize, &error_code_
        );
    }

    // Simulate the scheduler consuming tasks up to (exclusive) task_id
    // by advancing last_alive and setting packed_buffer_end on the consumed descriptor.
    void consume_up_to(int32_t task_id, uint64_t heap_tail_offset) {
        // Set the packed_buffer_end on the descriptor that last_alive-1 maps to
        // so update_heap_tail can derive the tail.
        int32_t last_consumed = task_id - 1;
        descriptors_[last_consumed & (kWindowSize - 1)].packed_buffer_end =
            static_cast<char *>(static_cast<void *>(heap_buf_.data())) + heap_tail_offset;
        last_alive_.store(task_id, std::memory_order_release);
    }

    PTO2TaskAllocator allocator_;
    std::vector<PTO2TaskDescriptor> descriptors_;
    std::vector<char> heap_buf_;
    std::atomic<int32_t> current_index_{0};
    std::atomic<int32_t> last_alive_{0};
    std::atomic<int32_t> error_code_{0};
};

// =============================================================================
// TaskAllocator: init and state queries
// =============================================================================

TEST_F(TaskAllocatorTest, InitialState) {
    EXPECT_EQ(allocator_.window_size(), kWindowSize);
    EXPECT_EQ(allocator_.active_count(), 0);
    EXPECT_EQ(allocator_.heap_top(), 0u);
    EXPECT_EQ(allocator_.heap_capacity(), kHeapSize);
    EXPECT_EQ(allocator_.heap_available(), kHeapSize);
}

// =============================================================================
// TaskAllocator: single alloc with output_size=0
// =============================================================================

TEST_F(TaskAllocatorTest, AllocZeroOutputSize) {
    auto result = allocator_.alloc(0);
    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_EQ(result.slot, 0);
    // packed_base should be heap_base + 0 (non-null)
    EXPECT_NE(result.packed_base, nullptr);
    // packed_end == packed_base when output_size == 0
    EXPECT_EQ(result.packed_base, result.packed_end);
    // Heap top should not advance for zero-size alloc
    EXPECT_EQ(allocator_.heap_top(), 0u);
}

// =============================================================================
// TaskAllocator: single alloc with non-zero size
// =============================================================================

TEST_F(TaskAllocatorTest, AllocNonZeroSize) {
    auto result = allocator_.alloc(100);
    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_EQ(result.slot, 0);
    EXPECT_NE(result.packed_base, nullptr);
    // 100 bytes aligned up to PTO2_ALIGN_SIZE (64) = 128
    uint64_t expected_aligned = PTO2_ALIGN_UP(100u, PTO2_ALIGN_SIZE);
    EXPECT_EQ(expected_aligned, 128u);
    EXPECT_EQ(allocator_.heap_top(), expected_aligned);
    EXPECT_EQ(
        static_cast<char *>(result.packed_end) - static_cast<char *>(result.packed_base),
        static_cast<ptrdiff_t>(expected_aligned)
    );
}

// =============================================================================
// TaskAllocator: sequential allocs produce sequential task IDs
// =============================================================================

TEST_F(TaskAllocatorTest, SequentialTaskIds) {
    for (int i = 0; i < 5; i++) {
        auto result = allocator_.alloc(0);
        ASSERT_FALSE(result.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(result.task_id, i);
        EXPECT_EQ(result.slot, i & (kWindowSize - 1));
    }
    EXPECT_EQ(allocator_.active_count(), 5);
}

// =============================================================================
// TaskAllocator: alignment of output_size to PTO2_ALIGN_SIZE
// =============================================================================

TEST_F(TaskAllocatorTest, OutputSizeAlignment) {
    // 1 byte -> aligned to 64
    auto r1 = allocator_.alloc(1);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator_.heap_top(), 64u);

    // Another 33 bytes -> aligned to 64, total 128
    auto r2 = allocator_.alloc(33);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator_.heap_top(), 128u);

    // Exactly 64 bytes -> stays 64, total 192
    auto r3 = allocator_.alloc(64);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(allocator_.heap_top(), 192u);
}

// =============================================================================
// TaskAllocator: try_bump_heap exact fit at end (space_at_end == alloc_size)
// =============================================================================

TEST_F(TaskAllocatorTest, HeapExactFitAtEnd) {
    // Heap size is 1024. Allocate 960 bytes (15 * 64) to leave exactly 64 at end.
    // Then allocate exactly 64 which should succeed (space_at_end >= alloc_size).
    auto r1 = allocator_.alloc(960);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator_.heap_top(), 960u);

    auto r2 = allocator_.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator_.heap_top(), 1024u);
    // Result pointer should be at heap_base + 960
    EXPECT_EQ(static_cast<char *>(r2.packed_base), heap_buf_.data() + 960);
}

// =============================================================================
// TaskAllocator: try_bump_heap wrap guard intentionally rejects tail == alloc_size
//
// The wrap guard `tail > alloc_size` uses strict > to prevent full/empty
// ambiguity.  If the allocation were allowed, heap_top would advance to
// alloc_size == tail, making top == tail.  Because top == tail is the
// canonical "empty" state, the ring could not distinguish "completely full"
// from "completely empty", causing subsequent allocations to overwrite
// live data.  Sacrificing one aligned quantum of capacity is the standard
// circular-buffer technique to avoid this.
// =============================================================================

TEST_F(TaskAllocatorTest, HeapWrapGuardRejectsTailEqualsAllocSize) {
    // Fill heap completely: allocate 1024 bytes total
    auto r1 = allocator_.alloc(1024);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator_.heap_top(), 1024u);

    // Consume task 0, setting tail to exactly 64 (one aligned block)
    consume_up_to(1, 64);

    // top=1024 (== heap_size), tail=64, alloc_size=64
    // space_at_end = 0, wrap check: tail(64) > alloc_size(64) -> FALSE
    // Allocation is correctly rejected to preserve the top != tail invariant.
    auto r2 = allocator_.alloc(64);
    EXPECT_TRUE(r2.failed()) << "wrap guard must reject when tail == alloc_size (full/empty ambiguity)";
}

// =============================================================================
// TaskAllocator: try_bump_heap wrap-around success (tail > alloc_size)
// =============================================================================

TEST_F(TaskAllocatorTest, HeapWrapAroundSuccess) {
    // Fill heap completely: allocate 1024 bytes
    auto r1 = allocator_.alloc(1024);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator_.heap_top(), 1024u);

    // Consume task 0, setting tail to 128 (more than one block)
    consume_up_to(1, 128);

    // Now: top=1024 (== heap_size), tail=128
    // space_at_end = 0, so wrap-around check: tail(128) > alloc_size(64)? => TRUE
    // Wraps to beginning: result = heap_base, top = 64
    auto r2 = allocator_.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf_.data()));
    EXPECT_EQ(allocator_.heap_top(), 64u);
}

// =============================================================================
// TaskAllocator: try_bump_heap top < tail linear-gap guard rejects exact fit
//
// The linear-gap guard `tail - top > alloc_size` uses strict > for the same
// full/empty ambiguity reason as the wrap guard.  If exact fit were allowed,
// heap_top would advance to tail, making top == tail (looks empty).  The
// next allocation would see top >= tail with space_at_end = heap_size - top
// and allocate into the region that still contains live data from the prior
// wrap.  The strict > sacrifices one quantum to keep top != tail.
// =============================================================================

TEST_F(TaskAllocatorTest, HeapLinearGapGuardRejectsExactFit) {
    // Fill heap, then wrap around to set up top < tail.
    auto r1 = allocator_.alloc(960);
    ASSERT_FALSE(r1.failed());

    // Consume task 0, tail moves to 960
    consume_up_to(1, 960);

    // Allocate 128 bytes: space_at_end = 1024-960 = 64, not enough for 128.
    // Wrap-around: tail(960) > 128 => TRUE, wraps.
    auto r2 = allocator_.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator_.heap_top(), 128u);

    // Now top=128, tail=960 (top < tail)
    // gap = tail - top = 960 - 128 = 832
    // Allocate exactly 832 bytes: gap(832) > alloc_size(832) -> FALSE
    // Correctly rejected to preserve top != tail invariant.
    auto r3 = allocator_.alloc(832);
    EXPECT_TRUE(r3.failed()) << "linear-gap guard must reject exact fit (full/empty ambiguity)";
}

// =============================================================================
// TaskAllocator: try_bump_heap top < tail insufficient space
// =============================================================================

TEST_F(TaskAllocatorTest, HeapTopLessThanTailInsufficientSpace) {
    // Set up top < tail scenario
    auto r1 = allocator_.alloc(960);
    ASSERT_FALSE(r1.failed());
    consume_up_to(1, 960);

    auto r2 = allocator_.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator_.heap_top(), 128u);

    // Now top=128, tail=960. Available = 832.
    // Try to allocate 896 (> 832): should fail (deadlock after spin).
    auto r3 = allocator_.alloc(896);
    EXPECT_TRUE(r3.failed());
    EXPECT_NE(error_code_.load(), 0);
}

// =============================================================================
// TaskAllocator: update_heap_tail from consumed task
// =============================================================================

TEST_F(TaskAllocatorTest, UpdateHeapTailFromConsumedTask) {
    auto r1 = allocator_.alloc(256);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator_.heap_top(), 256u);

    // Before consumption, heap_available should be heap_size - top = 768
    EXPECT_EQ(allocator_.heap_available(), kHeapSize - 256u);

    // Consume task 0, tail moves to 256
    consume_up_to(1, 256);

    // Force the allocator to observe the new last_alive by doing another alloc
    auto r2 = allocator_.alloc(0);
    ASSERT_FALSE(r2.failed());

    // After update_heap_tail, full heap should be available again
    // top=256, tail=256, so available = heap_size - top = 768 (at_end)
    // Actually: top >= tail, at_end = 1024-256=768, at_begin = 256
    // heap_available returns max(at_end, at_begin) = 768
    EXPECT_EQ(allocator_.heap_available(), kHeapSize - 256u);
}

// =============================================================================
// TaskAllocator: update_heap_tail at task 0 boundary
//
// When last_alive=1, update_heap_tail reads descriptors[(1-1) & mask] = descriptors[0].
// This is task 0's descriptor, which should have valid packed_buffer_end.
// =============================================================================

TEST_F(TaskAllocatorTest, UpdateHeapTailAtTask0) {
    // Allocate task 0 with some heap
    auto r1 = allocator_.alloc(64);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.task_id, 0);

    // Set packed_buffer_end on task 0's descriptor
    descriptors_[0].packed_buffer_end = static_cast<char *>(static_cast<void *>(heap_buf_.data())) + 64;

    // Advance last_alive to 1 (meaning task 0 is consumed)
    last_alive_.store(1, std::memory_order_release);

    // The next alloc triggers update_heap_tail(1), reading descriptors[0].
    auto r2 = allocator_.alloc(0);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, 1);
}

// =============================================================================
// TaskAllocator: update_heap_tail idempotent
// =============================================================================

TEST_F(TaskAllocatorTest, UpdateHeapTailIdempotent) {
    auto r1 = allocator_.alloc(128);
    ASSERT_FALSE(r1.failed());

    consume_up_to(1, 128);

    // Multiple allocs should not cause heap_tail to drift
    auto r2 = allocator_.alloc(0);
    ASSERT_FALSE(r2.failed());
    uint64_t avail_after_first = allocator_.heap_available();

    auto r3 = allocator_.alloc(0);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(allocator_.heap_available(), avail_after_first);
}

// =============================================================================
// TaskAllocator: heap_available for top>=tail and top<tail
// =============================================================================

TEST_F(TaskAllocatorTest, HeapAvailableTopGeTail) {
    // Initially top=0, tail=0: available = heap_size - 0 = 1024
    EXPECT_EQ(allocator_.heap_available(), kHeapSize);

    auto r1 = allocator_.alloc(256);
    ASSERT_FALSE(r1.failed());
    // top=256, tail=0: at_end=768, at_begin=0, available=768
    EXPECT_EQ(allocator_.heap_available(), kHeapSize - 256u);
}

TEST_F(TaskAllocatorTest, HeapAvailableTopLtTail) {
    // Set up top < tail
    auto r1 = allocator_.alloc(960);
    ASSERT_FALSE(r1.failed());
    consume_up_to(1, 960);

    // Wrap around
    auto r2 = allocator_.alloc(128);
    ASSERT_FALSE(r2.failed());
    // top=128, tail=960: available = 960 - 128 = 832
    EXPECT_EQ(allocator_.heap_available(), 832u);
}

// =============================================================================
// DepListPool Test Fixture
// =============================================================================

class DepListPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        entries_.resize(kPoolCapacity);
        std::memset(entries_.data(), 0, sizeof(PTO2DepListEntry) * kPoolCapacity);
        error_code_.store(0, std::memory_order_relaxed);

        pool_.init(entries_.data(), kPoolCapacity, &error_code_);
    }

    static constexpr int32_t kPoolCapacity = 8;

    PTO2DepListPool pool_;
    std::vector<PTO2DepListEntry> entries_;
    std::atomic<int32_t> error_code_{0};
};

// =============================================================================
// DepListPool: init (top=1, tail=1, entry 0 is NULL)
// =============================================================================

TEST_F(DepListPoolTest, InitialState) {
    EXPECT_EQ(pool_.used(), 0);
    EXPECT_EQ(pool_.available(), kPoolCapacity);

    // Entry 0 should be NULL marker
    EXPECT_EQ(entries_[0].slot_state, nullptr);
    EXPECT_EQ(entries_[0].next, nullptr);
}

// =============================================================================
// DepListPool: single alloc
// =============================================================================

TEST_F(DepListPoolTest, SingleAlloc) {
    PTO2DepListEntry *entry = pool_.alloc();
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(pool_.used(), 1);
    EXPECT_EQ(pool_.available(), kPoolCapacity - 1);

    // The allocated entry should be at index 1 (top was 1, mod capacity)
    EXPECT_EQ(entry, &entries_[1]);
}

// =============================================================================
// DepListPool: overflow detection
// =============================================================================

TEST_F(DepListPoolTest, OverflowDetection) {
    // Allocate until full (capacity entries used)
    for (int i = 0; i < kPoolCapacity; i++) {
        PTO2DepListEntry *e = pool_.alloc();
        ASSERT_NE(e, nullptr) << "Unexpected failure at alloc " << i;
    }
    EXPECT_EQ(pool_.used(), kPoolCapacity);
    EXPECT_EQ(pool_.available(), 0);

    // Next alloc should fail (overflow)
    PTO2DepListEntry *overflow = pool_.alloc();
    EXPECT_EQ(overflow, nullptr);
    EXPECT_NE(error_code_.load(), 0);
    EXPECT_EQ(error_code_.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
}

// =============================================================================
// DepListPool: prepend chain integrity
// =============================================================================

TEST_F(DepListPoolTest, PrependChainIntegrity) {
    PTO2TaskSlotState slot_a{};
    PTO2TaskSlotState slot_b{};
    PTO2TaskSlotState slot_c{};

    // Build a chain: NULL -> slot_a -> slot_b -> slot_c (prepend order)
    PTO2DepListEntry *head = nullptr;

    head = pool_.prepend(head, &slot_a);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->slot_state, &slot_a);
    EXPECT_EQ(head->next, nullptr);

    head = pool_.prepend(head, &slot_b);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->slot_state, &slot_b);
    EXPECT_EQ(head->next->slot_state, &slot_a);
    EXPECT_EQ(head->next->next, nullptr);

    head = pool_.prepend(head, &slot_c);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->slot_state, &slot_c);
    EXPECT_EQ(head->next->slot_state, &slot_b);
    EXPECT_EQ(head->next->next->slot_state, &slot_a);
    EXPECT_EQ(head->next->next->next, nullptr);
}

// =============================================================================
// DepListPool: advance_tail
// =============================================================================

TEST_F(DepListPoolTest, AdvanceTail) {
    // Allocate 4 entries
    for (int i = 0; i < 4; i++) {
        pool_.alloc();
    }
    EXPECT_EQ(pool_.used(), 4);
    EXPECT_EQ(pool_.available(), kPoolCapacity - 4);

    // Advance tail by 3 (from 1 to 4)
    pool_.advance_tail(4);
    EXPECT_EQ(pool_.used(), 1);
    EXPECT_EQ(pool_.available(), kPoolCapacity - 1);
}

// =============================================================================
// DepListPool: advance_tail backwards (no-op)
// =============================================================================

TEST_F(DepListPoolTest, AdvanceTailBackwardsNoop) {
    pool_.alloc();
    pool_.alloc();
    pool_.advance_tail(3);
    int32_t used_after = pool_.used();

    // Trying to advance backwards should be a no-op
    pool_.advance_tail(2);
    EXPECT_EQ(pool_.used(), used_after);

    // Same value should also be a no-op
    pool_.advance_tail(3);
    EXPECT_EQ(pool_.used(), used_after);
}
