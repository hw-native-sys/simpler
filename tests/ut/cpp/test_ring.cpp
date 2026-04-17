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

#include "ring.h"

namespace {

// Most tests only need a small heap to exercise wrap/back-pressure quickly.
// This is the PER-RING size now that Ring owns MAX_RING_DEPTH rings.
constexpr uint64_t kSmallHeap = 8ULL * HEAP_ALIGN;  // 8 KiB per ring
constexpr uint32_t kQuickTimeoutMs = 500;

}  // namespace

TEST(Ring, SlotOnlyIdsAreMonotonic) {
    Ring a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    for (int i = 0; i < 8; ++i) {
        auto r = a.alloc();
        ASSERT_EQ(r.slot, i);
        EXPECT_EQ(r.heap_ptr, nullptr);
    }
}

TEST(Ring, SlotAllocGrowsPastLegacyWindow) {
    // The old fixed window was 128 slots; confirm we can allocate well past
    // that without hitting a "task window full" error.
    Ring a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    for (int i = 0; i < 2048; ++i) {
        auto r = a.alloc();
        ASSERT_EQ(r.slot, i);
    }
    EXPECT_EQ(a.active_count(), 2048);
}

TEST(Ring, HeapSlabsAreAlignedAndDistinct) {
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    auto r1 = a.alloc(100);  // rounds to 1024
    auto r2 = a.alloc(100);
    ASSERT_NE(r1.heap_ptr, nullptr);
    ASSERT_NE(r2.heap_ptr, nullptr);
    uintptr_t p1 = reinterpret_cast<uintptr_t>(r1.heap_ptr);
    uintptr_t p2 = reinterpret_cast<uintptr_t>(r2.heap_ptr);
    EXPECT_EQ(p1 % HEAP_ALIGN, 0u);
    EXPECT_EQ(p2 % HEAP_ALIGN, 0u);
    EXPECT_EQ(p2 - p1, HEAP_ALIGN);
}

TEST(Ring, AllocBytesGreaterThanHeapThrows) {
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);
    EXPECT_THROW(a.alloc(kSmallHeap + 1), std::runtime_error);
}

TEST(Ring, AllocBytesWithHeapDisabledThrows) {
    Ring a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    EXPECT_THROW(a.alloc(32), std::runtime_error);
}

TEST(Ring, HeapReclamationIsFifo) {
    Ring a;
    a.init(4 * HEAP_ALIGN, kQuickTimeoutMs);

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

TEST(Ring, HeapWrapsAroundWhenTailLeadsTop) {
    Ring a;
    a.init(4 * HEAP_ALIGN, kQuickTimeoutMs);

    auto r0 = a.alloc(HEAP_ALIGN);
    auto r1 = a.alloc(HEAP_ALIGN);
    auto r2 = a.alloc(HEAP_ALIGN);
    auto r3 = a.alloc(HEAP_ALIGN);  // heap full

    // Free the front of the heap so the next alloc must wrap to offset 0.
    a.release(r0.slot);
    a.release(r1.slot);

    auto wrapped = a.alloc(HEAP_ALIGN);
    ASSERT_NE(wrapped.heap_ptr, nullptr);
    // Wrapped allocation lives at the base of the region.
    EXPECT_EQ(wrapped.heap_ptr, a.heap_base(0));

    a.release(r2.slot);
    a.release(r3.slot);
    a.release(wrapped.slot);
}

TEST(Ring, SlotStateIsPointerStable) {
    Ring a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);

    auto r0 = a.alloc();
    TaskSlotState *p0 = a.slot_state(r0.slot);
    ASSERT_NE(p0, nullptr);

    // Push many more slots through — the deque may grow/chain, but the
    // pointer we grabbed for slot 0 has to stay valid.
    for (int i = 0; i < 1000; ++i) {
        (void)a.alloc();
    }
    EXPECT_EQ(a.slot_state(r0.slot), p0);
}

TEST(Ring, ResetToEmptyRequiresAllReleased) {
    Ring a;
    a.init(/*heap_bytes=*/0, kQuickTimeoutMs);
    (void)a.alloc();
    EXPECT_THROW(a.reset_to_empty(), std::logic_error);
}

TEST(Ring, ResetToEmptyResetsCounters) {
    Ring a;
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

TEST(Ring, BackPressureThenReleaseUnblocks) {
    Ring a;
    a.init(2 * HEAP_ALIGN, /*timeout_ms=*/5000);

    auto r0 = a.alloc(HEAP_ALIGN);
    auto r1 = a.alloc(HEAP_ALIGN);  // heap full
    EXPECT_EQ(a.active_count(), 2);

    // Release both so the wrap has room. try_bump_heap_locked keeps a
    // one-slab guard (`tail > aligned`, strictly) between full and empty,
    // so releasing only one slab can still leave the waiter stuck.
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        a.release(r0.slot);
        a.release(r1.slot);
    });

    auto r = a.alloc(HEAP_ALIGN);  // blocks until releaser runs
    EXPECT_NE(r.slot, INVALID_SLOT);
    releaser.join();
    a.release(r.slot);

    a.shutdown();
}

TEST(Ring, TimeoutThrowsRuntimeError) {
    Ring a;
    a.init(2 * HEAP_ALIGN, /*timeout_ms=*/50);
    (void)a.alloc(HEAP_ALIGN);
    (void)a.alloc(HEAP_ALIGN);  // heap full, nobody releases
    EXPECT_THROW(a.alloc(HEAP_ALIGN), std::runtime_error);
}

TEST(Ring, ShutdownUnblocksAlloc) {
    Ring a;
    a.init(2 * HEAP_ALIGN, /*timeout_ms=*/5000);
    (void)a.alloc(HEAP_ALIGN);
    (void)a.alloc(HEAP_ALIGN);  // heap full

    std::thread t([&] {
        auto r = a.alloc(HEAP_ALIGN);  // should unblock on shutdown, not timeout
        EXPECT_EQ(r.slot, INVALID_SLOT);
        EXPECT_EQ(r.heap_ptr, nullptr);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    a.shutdown();
    t.join();
}

// ---------------------------------------------------------------------------
// Per-scope ring tests (Strict-1)
// ---------------------------------------------------------------------------

TEST(Ring, ScopeDepthMapsToRingIdx) {
    EXPECT_EQ(ring_idx_for_scope(0), 0);
    EXPECT_EQ(ring_idx_for_scope(1), 1);
    EXPECT_EQ(ring_idx_for_scope(2), 2);
    EXPECT_EQ(ring_idx_for_scope(3), 3);
    // Scopes deeper than MAX_RING_DEPTH share the innermost ring.
    EXPECT_EQ(ring_idx_for_scope(4), MAX_RING_DEPTH - 1);
    EXPECT_EQ(ring_idx_for_scope(64), MAX_RING_DEPTH - 1);
    // Negative scope depths clamp to 0 (defensive).
    EXPECT_EQ(ring_idx_for_scope(-1), 0);
}

TEST(Ring, PerRingHeapsAreDistinctMmaps) {
    // Total VA = 4 × 4 KiB; verify each ring has its own mapping.
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    std::vector<uintptr_t> bases;
    for (int r = 0; r < MAX_RING_DEPTH; ++r) {
        void *b = a.heap_base(r);
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(a.heap_size(r), kSmallHeap);
        bases.push_back(reinterpret_cast<uintptr_t>(b));
    }
    for (int i = 0; i < MAX_RING_DEPTH; ++i) {
        for (int j = i + 1; j < MAX_RING_DEPTH; ++j) {
            EXPECT_NE(bases[i], bases[j])
                << "rings " << i << " and " << j << " share a mapping — expected 4 separate mmaps";
        }
    }
}

TEST(Ring, AllocRoutesToRingChosenByScopeDepth) {
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    auto r0 = a.alloc(100, /*scope_depth=*/0);
    auto r1 = a.alloc(100, /*scope_depth=*/1);
    auto r2 = a.alloc(100, /*scope_depth=*/2);
    auto r3 = a.alloc(100, /*scope_depth=*/3);
    auto rd = a.alloc(100, /*scope_depth=*/7);  // clamps to ring 3

    EXPECT_EQ(r0.ring_idx, 0);
    EXPECT_EQ(r1.ring_idx, 1);
    EXPECT_EQ(r2.ring_idx, 2);
    EXPECT_EQ(r3.ring_idx, 3);
    EXPECT_EQ(rd.ring_idx, 3);

    // The allocation lives in the ring's own mmap region.
    for (const auto *res : {&r0, &r1, &r2, &r3}) {
        uintptr_t base = reinterpret_cast<uintptr_t>(a.heap_base(res->ring_idx));
        uintptr_t ptr = reinterpret_cast<uintptr_t>(res->heap_ptr);
        EXPECT_GE(ptr, base);
        EXPECT_LT(ptr, base + a.heap_size(res->ring_idx));
    }

    // Slot state remembers the ring assignment.
    EXPECT_EQ(a.slot_state(r0.slot)->ring_idx, 0);
    EXPECT_EQ(a.slot_state(r3.slot)->ring_idx, 3);

    a.release(r0.slot);
    a.release(r1.slot);
    a.release(r2.slot);
    a.release(r3.slot);
    a.release(rd.slot);
}

TEST(Ring, RingsReclaimIndependently) {
    // Fill ring 1 but keep ring 0 empty. A subsequent alloc on ring 0
    // must not be blocked by ring 1's fullness.
    Ring a;
    a.init(2 * HEAP_ALIGN, /*timeout_ms=*/100);

    auto r1a = a.alloc(HEAP_ALIGN, /*scope_depth=*/1);
    auto r1b = a.alloc(HEAP_ALIGN, /*scope_depth=*/1);  // ring 1 full
    EXPECT_EQ(r1a.ring_idx, 1);
    EXPECT_EQ(r1b.ring_idx, 1);

    // Ring 0 is untouched — this must succeed instantly, not time out.
    auto r0 = a.alloc(HEAP_ALIGN, /*scope_depth=*/0);
    EXPECT_EQ(r0.ring_idx, 0);
    ASSERT_NE(r0.heap_ptr, nullptr);

    // Ring 3 is also untouched.
    auto r3 = a.alloc(HEAP_ALIGN, /*scope_depth=*/3);
    EXPECT_EQ(r3.ring_idx, 3);

    // Now verify ring 1 still times out (no release on ring 1 yet).
    EXPECT_THROW(a.alloc(HEAP_ALIGN, /*scope_depth=*/1), std::runtime_error);

    a.release(r1a.slot);
    a.release(r1b.slot);
    a.release(r0.slot);
    a.release(r3.slot);
}

TEST(Ring, InnerRingReclaimsWhileOuterHolds) {
    // Simulate nested scope: outer task on ring 0 holds an allocation while
    // inner tasks on ring 1 churn through their heap. Ring 1's heap_top /
    // heap_tail advance as tasks cycle; ring 0's cursors stay put because
    // the outer task is still alive.
    Ring a;
    a.init(4 * HEAP_ALIGN, /*timeout_ms=*/500);

    auto outer = a.alloc(HEAP_ALIGN, /*scope_depth=*/0);
    EXPECT_EQ(a.heap_top(0), HEAP_ALIGN);
    EXPECT_EQ(a.heap_tail(0), 0u);

    // Churn on the inner ring — allocate, release, allocate, release, ...
    for (int i = 0; i < 8; ++i) {
        auto inner = a.alloc(HEAP_ALIGN, /*scope_depth=*/1);
        a.release(inner.slot);
    }

    // Outer ring unchanged (one live slab at offset 0).
    EXPECT_EQ(a.heap_top(0), HEAP_ALIGN);
    EXPECT_EQ(a.heap_tail(0), 0u);
    // Inner ring reclaimed each slab — tail caught up to top.
    EXPECT_EQ(a.heap_tail(1), a.heap_top(1));

    a.release(outer.slot);
}

TEST(Ring, ResetRewindsAllRings) {
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    auto r0 = a.alloc(100, /*scope_depth=*/0);
    auto r1 = a.alloc(100, /*scope_depth=*/1);
    auto r2 = a.alloc(100, /*scope_depth=*/3);

    a.release(r0.slot);
    a.release(r1.slot);
    a.release(r2.slot);

    a.reset_to_empty();

    EXPECT_EQ(a.next_task_id(), 0);
    for (int r = 0; r < MAX_RING_DEPTH; ++r) {
        EXPECT_EQ(a.heap_top(r), 0u) << "ring " << r << " heap_top not rewound";
        EXPECT_EQ(a.heap_tail(r), 0u) << "ring " << r << " heap_tail not rewound";
    }
}

TEST(Ring, NewSlotStatesCarryRingMetadata) {
    Ring a;
    a.init(kSmallHeap, kQuickTimeoutMs);

    auto r = a.alloc(100, /*scope_depth=*/2);
    auto *s = a.slot_state(r.slot);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->ring_idx, 2);
    EXPECT_EQ(s->ring_slot_idx, 0);  // first allocation on ring 2

    auto r2 = a.alloc(100, /*scope_depth=*/2);
    auto *s2 = a.slot_state(r2.slot);
    EXPECT_EQ(s2->ring_idx, 2);
    EXPECT_EQ(s2->ring_slot_idx, 1);  // second allocation on the same ring

    auto r3 = a.alloc(100, /*scope_depth=*/0);
    auto *s3 = a.slot_state(r3.slot);
    EXPECT_EQ(s3->ring_idx, 0);
    EXPECT_EQ(s3->ring_slot_idx, 0);  // ring 0's own in-ring index resets to 0

    a.release(r.slot);
    a.release(r2.slot);
    a.release(r3.slot);
}
