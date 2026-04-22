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
 * Edge-case tests for PTO2TaskAllocator and PTO2DepListPool.
 *
 * Each test targets a specific code path, boundary condition, or potential
 * latent bug discovered through line-by-line analysis of pto_ring_buffer.h.
 *
 * Note: the unified PTO2TaskAllocator replaces the previous separate
 *   PTO2HeapRing + PTO2TaskRing. Because the allocator is single-threaded
 *   (orchestrator thread), CAS/concurrency edge cases that applied to the
 *   old design are no longer meaningful and have been removed. The wrap /
 *   fragmentation / zero-size tests remain and exercise try_bump_heap and
 *   the task window check.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- PTO2TaskAllocator (try_bump_heap)
 * ============================================================================
 *
 * DC-1: Wrap-around guard uses `tail > alloc_size` (strict >).  When
 *   tail == alloc_size the wrap branch returns nullptr.  This is
 *   intentional: allowing the allocation would set heap_top_ =
 *   alloc_size == tail, creating the classic circular-buffer full/empty
 *   ambiguity where top == tail must mean "empty".  The strict >
 *   sacrifices one aligned quantum of capacity to keep top != tail
 *   whenever the buffer has live data.
 *
 * DC-3: `heap_available()` returns max(at_end, at_begin), not the sum.
 *   A single allocation cannot split across the wrap boundary, so max
 *   is the right semantic -- callers should treat the return value as
 *   "largest contiguous allocation possible", not "total free bytes".
 *
 * DC-9: Zero-size allocation is a no-op that returns the current top
 *   without advancing. Two consecutive zero-size allocs return the
 *   SAME pointer. Semantically correct for a bump allocator.
 *
 * DC-10: Wrap path writes new_top = alloc_size; the wasted space at
 *   the end of the heap (between old top and heap_size) is not
 *   reclaimed because tail is advanced by packed_buffer_end, not by
 *   heap_size. Inherent to ring-buffer algorithms; acceptable
 *   fragmentation cost for allocator simplicity.
 *
 * EDGE-1: top == tail == 0 (initial state). space_at_end = heap_size.
 * EDGE-2: top == heap_size (exactly at end). space_at_end = 0, must wrap.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- Task window (via PTO2TaskAllocator::alloc)
 * ============================================================================
 *
 * EDGE-5: window_size = 1. Check `local_task_id_ - last_alive + 1 < 1`
 *   is always false -> every allocation spins forever (deadlock). This
 *   is undefined/unsupported configuration.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- DepListPool
 * ============================================================================
 *
 * Note: earlier comments in this file called `base[0]` a "sentinel" that
 * must never be overwritten.  That is **not** how the current src works.
 * The list terminator is literal `nullptr` (see pto_scheduler.h fanout
 * walk and `PTO2TaskSlotState::fanout_head = nullptr` initialization in
 * pto_runtime2_types.h).  `base[0]` is a normal pool entry; the init
 * clearing in `DepListPool::init` is incidental, not an invariant.  The
 * historical `SentinelOverwrite` / `SentinelDataCorruption` /
 * `MultiCyclesSentinelIntegrity` tests have been removed; they were
 * asserting behavior the src never promised.
 *
 * DC-7: `advance_tail(new_tail)` only advances if new_tail > tail; it
 *   does not validate new_tail <= top. The caller (orchestrator) is
 *   contracted to pass monotonically advancing, top-bounded values.
 *   Documented as an API contract; not a live defect.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <set>
#include <vector>

#include "pto_ring_buffer.h"

// =============================================================================
// Helper: advance last_alive so try_bump_heap can derive heap_tail from the
// consumed task's packed_buffer_end.
// =============================================================================
static void consume_up_to(
    std::vector<PTO2TaskDescriptor> &descriptors, std::atomic<int32_t> &last_alive, void *heap_base,
    int32_t window_size, int32_t new_last_alive, uint64_t heap_tail_offset
) {
    int32_t last_consumed = new_last_alive - 1;
    descriptors[last_consumed & (window_size - 1)].packed_buffer_end =
        static_cast<char *>(heap_base) + heap_tail_offset;
    last_alive.store(new_last_alive, std::memory_order_release);
}

// =============================================================================
// TaskAllocator edge-case fixture
// =============================================================================
class TaskAllocatorEdgeTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 16;
    static constexpr uint64_t HEAP_SIZE = 4096;

    std::vector<PTO2TaskDescriptor> descriptors;
    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> last_alive{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator{};

    void SetUp() override {
        descriptors.assign(WINDOW_SIZE, PTO2TaskDescriptor{});
        std::memset(heap_buf, 0, sizeof(heap_buf));
        current_index.store(0);
        last_alive.store(0);
        error_code.store(PTO2_ERROR_NONE);
        allocator.init(descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code);
    }
};

// ---------------------------------------------------------------------------
// DESIGN: Wrap guard `tail > alloc_size` is intentionally strict.
// When tail == alloc_size, accepting the allocation would set top == tail,
// creating full/empty ambiguity.  The guard sacrifices one quantum.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, WrapGuard_TailEqualsAllocSize) {
    // Fill heap to end.
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    // Consume task 0 to advance heap_tail to exactly 64.
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 64);

    // top == HEAP_SIZE, tail == 64, request 64 bytes.
    // space_at_end = 0. Wrap: tail(64) > 64 -> FALSE -> correctly rejected.
    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "Wrap guard correctly rejects when tail == alloc_size (full/empty ambiguity)";
}

// ---------------------------------------------------------------------------
// EDGE-2: top at exact end of heap (top == heap_size). After a full
// wrap the allocation must land at the base.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, TopAtExactEnd) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    // Advance tail so the wrap path has enough room for the next alloc.
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 128);

    // top(HEAP_SIZE) >= tail(128). space_at_end = 0.
    // Wrap: tail(128) > 64 -> true -> new_top = 64, result = base.
    auto r2 = allocator.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), 64u);
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-3: heap_available() reports max(at_end, at_begin), not the sum.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, AvailableFragmentation) {
    // Create a fragmented state: top near middle/high, tail in middle.
    auto r1 = allocator.alloc(3008);  // top ~ 3008 (already aligned to 64)
    ASSERT_FALSE(r1.failed());
    uint64_t actual_top = allocator.heap_top();

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 1024);

    // Force the allocator to refresh its cached heap_tail.
    auto r_probe = allocator.alloc(0);
    ASSERT_FALSE(r_probe.failed());

    uint64_t avail = allocator.heap_available();
    uint64_t at_end = HEAP_SIZE - actual_top;
    uint64_t at_begin = 1024;
    EXPECT_EQ(avail, std::max(at_end, at_begin));

    // Total free bytes (at_end + at_begin) may exceed what a single alloc can
    // take, because allocations never split across the wrap boundary.
    EXPECT_LT(avail, at_end + at_begin);
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-9: Zero-size allocation does not advance the heap pointer.
// Two consecutive zero-size allocs return the SAME address (aliased).
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, ZeroSizeAllocation) {
    auto r1 = allocator.alloc(0);
    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    ASSERT_FALSE(r2.failed());

    EXPECT_EQ(r1.packed_base, r2.packed_base) << "Zero-size allocs return same address";
    EXPECT_EQ(r1.packed_base, r1.packed_end) << "packed_end == packed_base for zero-size";
    EXPECT_EQ(r2.packed_base, r2.packed_end);
    EXPECT_EQ(allocator.heap_top(), 0u) << "top doesn't advance for zero-size allocs";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-10: Wrap-path wasted space accumulation.
// When wrapping, space between old top and heap_size is leaked.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, WrapPathWastedSpace) {
    // Allocate 4000 bytes -> top rounds to 4032 (aligned).
    auto r1 = allocator.alloc(4000);
    ASSERT_FALSE(r1.failed());
    uint64_t top_after = allocator.heap_top();
    EXPECT_GE(top_after, 4000u);
    EXPECT_LT(top_after, HEAP_SIZE);  // Some trailing space remains unused.

    // Reclaim task 0: tail moves up to match top (logically empty).
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, top_after);

    // space_at_end = HEAP_SIZE - top_after (e.g. 64). < 128 -> must wrap.
    // After the wrap, the 64 trailing bytes are unreachable.
    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf)) << "Allocation wrapped to beginning";

    // Available now reflects the new (post-wrap) top and the stale tail.
    uint64_t avail = allocator.heap_available();
    EXPECT_LT(avail, HEAP_SIZE) << "Wasted space at end reduces available capacity";
}

// ---------------------------------------------------------------------------
// Allocation of exactly heap_size: consumes entire heap in one shot.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, AllocExactlyHeapSize) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    // No more space (and no reclamation) -> next alloc spins to deadlock.
    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "No space after full allocation";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

// ---------------------------------------------------------------------------
// Allocation larger than heap_size: must fail (heap deadlock).
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, AllocLargerThanHeap) {
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed()) << "Cannot allocate more than heap size";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

// ---------------------------------------------------------------------------
// Task window saturates: allocator.alloc blocks when
// (local_task_id - last_alive + 1) >= window_size.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, TaskWindowSaturates) {
    // Allocate until the window is full: window allows window_size - 1 active.
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i);
    }
    EXPECT_EQ(allocator.active_count(), WINDOW_SIZE - 1);

    // The next alloc would push active_count to window_size and is refused
    // (spins until deadlock since last_alive is not advancing).
    auto overflow = allocator.alloc(0);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
}

// ---------------------------------------------------------------------------
// Slot mapping uses `task_id & window_mask` -- with a power-of-two window
// this is equivalent to modulo. Every consecutive window_size task IDs
// visit every slot exactly once.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, SlotMappingPowerOfTwoWindow) {
    std::set<int32_t> slots;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        // Advance last_alive so we can keep allocating past the window.
        consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, i, 0);
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed());
        EXPECT_EQ(r.slot, r.task_id & (WINDOW_SIZE - 1));
        slots.insert(r.slot);
    }
    EXPECT_EQ(slots.size(), static_cast<size_t>(WINDOW_SIZE))
        << "Every slot should be visited exactly once over one window cycle";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-11 (adapted): Task IDs grow monotonically as int32_t.
// Near INT32_MAX, fetch-like behavior would overflow in the old design;
// the new allocator increments a local counter and publishes it -- the
// same signed-overflow concern applies but is cosmetic here since we
// use `task_id & window_mask` for indexing.
// ---------------------------------------------------------------------------
TEST_F(TaskAllocatorEdgeTest, TaskIdNearInt32Max) {
    // Seed the shared counter near INT32_MAX and re-init so the allocator
    // picks up the seed as its local counter.
    current_index.store(INT32_MAX - 2);
    last_alive.store(INT32_MAX - 2);
    allocator.init(descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code);

    auto r1 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.task_id, INT32_MAX - 2);
    EXPECT_EQ(r1.slot, (INT32_MAX - 2) & (WINDOW_SIZE - 1));

    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, INT32_MAX - 1);

    auto r3 = allocator.alloc(0);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(r3.task_id, INT32_MAX);
    // Slot mask still yields a valid slot regardless of sign.
    EXPECT_GE(r3.slot, 0);
    EXPECT_LT(r3.slot, WINDOW_SIZE);
}

// =============================================================================
// DepListPool edge-case fixture
// =============================================================================
class DepPoolEdgeTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 8;
    PTO2DepListEntry entries[POOL_CAP]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool{};

    void SetUp() override {
        std::memset(entries, 0, sizeof(entries));
        error_code.store(PTO2_ERROR_NONE);
        pool.init(entries, POOL_CAP, &error_code);
    }
};

// ---------------------------------------------------------------------------
// DC-7 (contract): advance_tail does not validate new_tail <= top.
// Caller (orchestrator) is contracted to pass monotonic top-bounded values;
// these two tests document what happens if that contract is violated, to
// anchor the API shape -- they are not bug reports.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, AdvanceTailBeyondTop_ContractViolationProducesNegativeUsed) {
    pool.alloc();  // top=2
    pool.alloc();  // top=3

    pool.advance_tail(100);  // caller contract violation

    int32_t u = pool.used();       // top(3) - tail(100) = -97
    int32_t a = pool.available();  // capacity(8) - (-97) = 105

    EXPECT_LT(u, 0) << "used() goes negative when tail > top";
    EXPECT_GT(a, pool.capacity) << "available() exceeds capacity when tail > top";
}

TEST_F(DepPoolEdgeTest, AdvanceTailBeyondTop_ContractViolationLetsAllocProceed) {
    pool.alloc();            // top=2
    pool.advance_tail(100);  // caller contract violation

    // used() is negative -> overflow check (used >= capacity) is false -> alloc proceeds.
    PTO2DepListEntry *e = pool.alloc();
    EXPECT_NE(e, nullptr) << "Alloc succeeds with corrupted tail (negative used)";
    EXPECT_LT(pool.used(), 0) << "Pool state remains corrupted: negative used count";
}

// ---------------------------------------------------------------------------
// Prepend chain integrity under pool exhaustion: chain must be walkable.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, PrependUnderExhaustion) {
    PTO2TaskSlotState slots[POOL_CAP]{};
    PTO2DepListEntry *head = nullptr;

    int count = 0;
    while (count < POOL_CAP + 5) {  // Try beyond capacity
        PTO2DepListEntry *new_head = pool.prepend(head, &slots[count % POOL_CAP]);
        if (!new_head) break;
        head = new_head;
        count++;
    }

    // Walk the chain -- should be intact (no cycles, no overruns).
    int walk = 0;
    PTO2DepListEntry *cur = head;
    while (cur) {
        walk++;
        cur = cur->next;
        if (walk > count + 1) {
            FAIL() << "Chain has cycle -- walked more entries than allocated";
            break;
        }
    }
    EXPECT_EQ(walk, count);
}

// ---------------------------------------------------------------------------
// Prepend builds linked list correctly: verify each slot_state pointer.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, PrependChainCorrectness) {
    PTO2TaskSlotState slots[5]{};
    PTO2DepListEntry *head = nullptr;

    for (int i = 0; i < 5; i++) {
        head = pool.prepend(head, &slots[i]);
        ASSERT_NE(head, nullptr);
    }

    // LIFO order: head -> slots[4] -> slots[3] -> ... -> slots[0] -> nullptr.
    PTO2DepListEntry *cur = head;
    for (int i = 4; i >= 0; i--) {
        ASSERT_NE(cur, nullptr);
        EXPECT_EQ(cur->slot_state, &slots[i]) << "Entry " << (4 - i) << " should point to slots[" << i << "]";
        cur = cur->next;
    }
    EXPECT_EQ(cur, nullptr) << "Chain should terminate with nullptr";
}

// ---------------------------------------------------------------------------
// High-water mark accuracy after reclaim cycles (ABI contract: diagnostic field).
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, HighWaterAccuracy) {
    for (int i = 0; i < 5; i++)
        pool.alloc();
    EXPECT_EQ(pool.high_water, 5);

    pool.advance_tail(4);
    EXPECT_EQ(pool.high_water, 5) << "High water never decreases";

    for (int i = 0; i < 3; i++)
        pool.alloc();
    EXPECT_GE(pool.high_water, 5);
}

// ---------------------------------------------------------------------------
// Advance tail backwards is a no-op.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, AdvanceTailBackwards) {
    pool.alloc();
    pool.alloc();
    pool.advance_tail(3);

    pool.advance_tail(1);  // Should be no-op.
    EXPECT_EQ(pool.used(), 0) << "advance_tail backwards is a no-op";
}

// ---------------------------------------------------------------------------
// Pool init state verification.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, InitState) {
    EXPECT_EQ(pool.used(), 0) << "initially empty";
    EXPECT_EQ(pool.available(), POOL_CAP) << "full capacity available";
    EXPECT_EQ(entries[0].slot_state, nullptr) << "sentinel slot_state is null";
    EXPECT_EQ(entries[0].next, nullptr) << "sentinel next is null";
}

// ---------------------------------------------------------------------------
// Alloc all then overflow: verify error code is set.
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, OverflowSetsErrorCode) {
    for (int i = 0; i < POOL_CAP; i++) {
        pool.alloc();
    }

    PTO2DepListEntry *overflow_result = pool.alloc();
    EXPECT_EQ(overflow_result, nullptr) << "Overflow returns nullptr";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW) << "Error code set on overflow";
}
