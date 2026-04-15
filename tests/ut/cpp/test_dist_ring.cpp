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

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "dist_ring.h"

namespace {

// Most tests only need a small heap to exercise wrap/back-pressure quickly.
constexpr uint64_t kSmallHeap = 8ULL * DIST_HEAP_ALIGN;  // 8 KiB
constexpr uint32_t kQuickTimeoutMs = 500;

}  // namespace

TEST(DistRing, SlotOnlyIdsAreMonotonic) {
    DistRing a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    for (int i = 0; i < 8; ++i) {
        auto r = a.alloc();
        ASSERT_EQ(r.slot, i);
        EXPECT_EQ(r.heap_ptr, nullptr);
    }
}

TEST(DistRing, SlotAllocGrowsPastLegacyWindow) {
    // The old fixed window was 128 slots; confirm we can allocate well past
    // that without hitting a "task window full" error.
    DistRing a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    for (int i = 0; i < 2048; ++i) {
        auto r = a.alloc();
        ASSERT_EQ(r.slot, i);
    }
    EXPECT_EQ(a.active_count(), 2048);
}

TEST(DistRing, HeapSlabsAreAlignedAndDistinct) {
    DistRing a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    auto r1 = a.alloc(100);  // rounds to 1024
    auto r2 = a.alloc(100);
    ASSERT_NE(r1.heap_ptr, nullptr);
    ASSERT_NE(r2.heap_ptr, nullptr);
    uintptr_t p1 = reinterpret_cast<uintptr_t>(r1.heap_ptr);
    uintptr_t p2 = reinterpret_cast<uintptr_t>(r2.heap_ptr);
    EXPECT_EQ(p1 % DIST_HEAP_ALIGN, 0u);
    EXPECT_EQ(p2 % DIST_HEAP_ALIGN, 0u);
    EXPECT_EQ(p2 - p1, DIST_HEAP_ALIGN);
}

TEST(DistRing, AllocBytesGreaterThanHeapThrows) {
    DistRing a;
    a.init(kSmallHeap, kQuickTimeoutMs);
    EXPECT_THROW(a.alloc(kSmallHeap + 1), std::runtime_error);
}

TEST(DistRing, AllocBytesWithHeapDisabledThrows) {
    DistRing a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    EXPECT_THROW(a.alloc(32), std::runtime_error);
}

TEST(DistRing, HeapReclamationIsFifo) {
    DistRing a;
    a.init(4 * DIST_HEAP_ALIGN, kQuickTimeoutMs);

    auto r0 = a.alloc(100);
    auto r1 = a.alloc(100);
    auto r2 = a.alloc(100);
    auto r3 = a.alloc(100);  // heap exactly full

    // Releasing r2 first does not free heap space (r0/r1 still FIFO-alive).
    a.release(r2.slot);
    EXPECT_THROW(a.alloc(100), std::runtime_error);

    // Releasing r0, r1 advances last_alive; r2 then releases for free and a
    // fresh alloc succeeds. (r3 remains live, capping the forward tail.)
    a.release(r0.slot);
    a.release(r1.slot);
    auto r4 = a.alloc(100);
    ASSERT_NE(r4.heap_ptr, nullptr);
    a.release(r3.slot);
    a.release(r4.slot);
}

TEST(DistRing, HeapWrapsAroundWhenTailLeadsTop) {
    DistRing a;
    a.init(4 * DIST_HEAP_ALIGN, kQuickTimeoutMs);

    auto r0 = a.alloc(DIST_HEAP_ALIGN);
    auto r1 = a.alloc(DIST_HEAP_ALIGN);
    auto r2 = a.alloc(DIST_HEAP_ALIGN);
    auto r3 = a.alloc(DIST_HEAP_ALIGN);  // heap full

    // Free the front of the heap so the next alloc must wrap to offset 0.
    a.release(r0.slot);
    a.release(r1.slot);

    auto wrapped = a.alloc(DIST_HEAP_ALIGN);
    ASSERT_NE(wrapped.heap_ptr, nullptr);
    // Wrapped allocation lives at the base of the region.
    EXPECT_EQ(wrapped.heap_ptr, a.heap_base());

    a.release(r2.slot);
    a.release(r3.slot);
    a.release(wrapped.slot);
}

TEST(DistRing, SlotStateIsPointerStable) {
    DistRing a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);

    auto r0 = a.alloc();
    DistTaskSlotState *p0 = a.slot_state(r0.slot);
    ASSERT_NE(p0, nullptr);

    // Push many more slots through — the deque may grow/chain, but the
    // pointer we grabbed for slot 0 has to stay valid.
    for (int i = 0; i < 1000; ++i) {
        (void)a.alloc();
    }
    EXPECT_EQ(a.slot_state(r0.slot), p0);
}

TEST(DistRing, ResetToEmptyRequiresAllReleased) {
    DistRing a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    (void)a.alloc();
    EXPECT_THROW(a.reset_to_empty(), std::logic_error);
}

TEST(DistRing, ResetToEmptyResetsCounters) {
    DistRing a;
    a.init(kSmallHeap, kQuickTimeoutMs);
    auto r0 = a.alloc(100);
    auto r1 = a.alloc(100);
    a.release(r0.slot);
    a.release(r1.slot);
    EXPECT_EQ(a.active_count(), 0);
    EXPECT_EQ(a.next_task_id(), 2);
    a.reset_to_empty();
    EXPECT_EQ(a.next_task_id(), 0);

    auto r2 = a.alloc(100);
    EXPECT_EQ(r2.slot, 0);
    a.release(r2.slot);
}

TEST(DistRing, BackPressureThenReleaseUnblocks) {
    DistRing a;
    a.init(2 * DIST_HEAP_ALIGN, /*timeout_ms=*/5000);

    auto r0 = a.alloc(DIST_HEAP_ALIGN);
    auto r1 = a.alloc(DIST_HEAP_ALIGN);  // heap full
    EXPECT_EQ(a.active_count(), 2);

    // Release both so the wrap has room. try_bump_heap_locked keeps a
    // one-slab guard (`tail > aligned`, strictly) between full and empty,
    // so releasing only one slab can still leave the waiter stuck.
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        a.release(r0.slot);
        a.release(r1.slot);
    });

    auto r = a.alloc(DIST_HEAP_ALIGN);  // blocks until releaser runs
    EXPECT_NE(r.slot, DIST_INVALID_SLOT);
    releaser.join();
    a.release(r.slot);

    a.shutdown();
}

TEST(DistRing, TimeoutThrowsRuntimeError) {
    DistRing a;
    a.init(2 * DIST_HEAP_ALIGN, /*timeout_ms=*/50);
    (void)a.alloc(DIST_HEAP_ALIGN);
    (void)a.alloc(DIST_HEAP_ALIGN);  // heap full, nobody releases
    EXPECT_THROW(a.alloc(DIST_HEAP_ALIGN), std::runtime_error);
}

TEST(DistRing, ShutdownUnblocksAlloc) {
    DistRing a;
    a.init(2 * DIST_HEAP_ALIGN, /*timeout_ms=*/5000);
    (void)a.alloc(DIST_HEAP_ALIGN);
    (void)a.alloc(DIST_HEAP_ALIGN);  // heap full

    std::thread t([&] {
        auto r = a.alloc(DIST_HEAP_ALIGN);  // should unblock on shutdown, not timeout
        EXPECT_EQ(r.slot, DIST_INVALID_SLOT);
        EXPECT_EQ(r.heap_ptr, nullptr);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    a.shutdown();
    t.join();
}
