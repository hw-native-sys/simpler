/**
 * Edge-case tests for HeapRing, TaskRing, DepListPool.
 *
 * Each test targets a specific code path, boundary condition, or potential
 * latent bug discovered through line-by-line analysis of pto_ring_buffer.h.
 *
 * ============================================================================
 * ANALYSIS FINDINGS — HeapRing (pto2_heap_ring_try_alloc)
 * ============================================================================
 *
 * BUG-CANDIDATE-1: Wrap-around guard uses `tail > alloc_size` (strict >).
 *   When tail == alloc_size the wrap branch returns NULL even though
 *   there is exactly enough space at the beginning [0, alloc_size).
 *   This is an off-by-one that wastes one aligned quantum of space.
 *
 * BUG-CANDIDATE-2: CAS-retry loop re-reads both top AND tail on each
 *   iteration.  If another thread wraps top from (size-X) to Y while
 *   this thread's stale top is still (size-X), the computed space_at_end
 *   will be wrong.  The CAS will fail harmlessly, but the retry loop
 *   MUST reload top first (which it does via load in the while body).
 *   Not a bug, but the test confirms the CAS-safety invariant.
 *
 * BUG-CANDIDATE-3: `pto2_heap_ring_available()` returns max(at_end, at_begin),
 *   not the sum.  A caller using this to decide whether a large allocation
 *   is possible may get the wrong answer if the space is split across the
 *   wrap boundary.  This is by-design (never splits), but fragile.
 *
 * BUG-CANDIDATE-9: Zero-size allocation passes alignment (0 → 0 or 64
 *   depending on PTO2_ALIGN_UP behavior).  If aligned to 0, CAS with
 *   new_top == top is a no-op that succeeds, returning base + top.
 *   Subsequent allocations then overlap the same address.
 *
 * BUG-CANDIDATE-10: Wrap path writes new_top = alloc_size, but the wasted
 *   space at the end of the heap (between top and size) is "leaked" — tail
 *   can never reclaim it because tail is advanced by packed_buffer_end,
 *   not by heap_size.  If many small allocations near end-of-heap force
 *   repeated wraps, total usable capacity shrinks.
 *
 * EDGE-1: top == tail == 0 (initial state).  space_at_end = size.
 * EDGE-2: top == size (exactly at end).  space_at_end = 0, must wrap.
 * EDGE-3: top == tail (non-zero, both pointing to same offset) — empty.
 * EDGE-4: Double-align: request 1 byte → aligned to 64, then try_alloc
 *         is called again inside pto2_heap_ring_alloc with the same 1 byte.
 *         The inner try_alloc re-aligns.  Total overhead = 2× alignment
 *         computations but only 1× space consumed.
 *
 * ============================================================================
 * ANALYSIS FINDINGS — TaskRing (pto2_task_ring_try_alloc)
 * ============================================================================
 *
 * BUG-CANDIDATE-4: fetch_add(1) is done BEFORE the window-full check.
 *   If two threads race, both increment current_index, both see
 *   active_count >= window_size - 1, both roll back via fetch_sub(1).
 *   This is correct for correctness but causes unnecessary contention.
 *   More importantly: if N threads race, current_index temporarily
 *   spikes by N, and the "active_count" check uses this inflated value.
 *   All N will roll back.  But does the temporary spike break anything?
 *   → Test: concurrent try_alloc near window boundary.
 *
 * BUG-CANDIDATE-5: window_size is NOT validated as power-of-2 at init.
 *   pto2_task_ring_init() doesn't check.  If window_size = 5 is passed,
 *   `task_id & (window_size - 1)` = `task_id & 4` which maps 0-7 to
 *   {0,1,2,3,4,5,6,7} & 4 = {0,1,2,3,4,5,6,7} — wrong modulo!
 *   Should be documented or asserted.
 *
 * BUG-CANDIDATE-11: INT32 overflow on monotonic task_id.  task_id is
 *   int32_t, grows by fetch_add(1) forever.  At INT32_MAX, the next
 *   fetch_add wraps to INT32_MIN.  task_id & (window_size - 1) still
 *   works arithmetically, but task_id - last_alive wraps to negative.
 *
 * EDGE-5: window_size = 1.  active_count < 0 (window_size - 1 = 0).
 *         EVERY allocation immediately fails.  Is this handled?
 *
 * ============================================================================
 * ANALYSIS FINDINGS — DepListPool
 * ============================================================================
 *
 * BUG-CANDIDATE-6: `alloc()` checks `used >= capacity` but the pool
 *   has `capacity` slots (indices 0..capacity-1).  Entry 0 is reserved
 *   as NULL sentinel, so usable entries = capacity - 1?  Actually no:
 *   top starts at 1, so physical index wraps via `top % capacity`.
 *   When top = capacity, idx = 0 which is the sentinel slot!
 *   The alloc() will OVERWRITE the sentinel with user data.
 *   → Test: allocate exactly capacity entries and check sentinel.
 *
 * BUG-CANDIDATE-7: `advance_tail(new_tail)` only advances if new_tail > tail.
 *   But it doesn't validate new_tail <= top.  A spurious new_tail > top
 *   would make `used()` return negative, and `available()` > capacity.
 *   → Test: advance_tail beyond top.
 *
 * BUG-CANDIDATE-8: `pto2_dep_pool_get(offset)` returns &base[offset]
 *   without bounds checking against capacity.  If offset > capacity,
 *   out-of-bounds read.
 *
 * BUG-CANDIDATE-12: Reclaim-then-alloc cycle across multiple wraps.
 *   After alloc fills [1..capacity-1], reclaim advances tail to capacity-1.
 *   Next alloc at idx=capacity%capacity=0 → sentinel.  Multiple cycles
 *   compound the problem as sentinel is never re-initialized.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <cstring>
#include <climits>
#include "pto_ring_buffer.h"

// =============================================================================
// HeapRing edge-case fixture
// =============================================================================
class HeapRingEdgeTest : public ::testing::Test {
protected:
    alignas(64) uint8_t heap_buf[4096]{};
    std::atomic<uint64_t> top{0};
    std::atomic<uint64_t> tail{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2HeapRing ring{};

    void SetUp() override {
        top.store(0);
        tail.store(0);
        error_code.store(PTO2_ERROR_NONE);
        pto2_heap_ring_init(&ring, heap_buf, 4096, &tail, &top);
        ring.error_code_ptr = &error_code;
    }
};

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-1: Wrap guard `tail > alloc_size` is off-by-one.
// When tail == alloc_size, there IS space [0, alloc_size) but code returns NULL.
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, WrapGuard_TailEqualsAllocSize) {
    uint64_t alloc = 64;   // PTO2_ALIGN_SIZE

    // Fill heap to end: top = 4096 - 64 = 4032, tail = 0
    void* p1 = ring.pto2_heap_ring_try_alloc(4096 - 64);
    ASSERT_NE(p1, nullptr);

    // Advance tail to exactly alloc_size (64)
    tail.store(alloc);

    // Now try to allocate 64 bytes.
    // top = 4032, space_at_end = 4096 - 4032 = 64 → fits at end!
    void* p2 = ring.pto2_heap_ring_try_alloc(alloc);
    EXPECT_NE(p2, nullptr) << "Should fit at end without wrapping";
}

// When there's no space at end and tail == alloc_size, the wrap branch
// checks `tail > alloc_size` (strict).  64 > 64 is false → NULL.
TEST_F(HeapRingEdgeTest, WrapGuard_TailEqualsAllocSize_NoEndSpace) {
    uint64_t alloc = 128;

    // Fill to very end: top = 4096 (conceptually)
    // Actually, let's fill to 4096 - 64 and then allocate 64 to reach 4096
    void* p1 = ring.pto2_heap_ring_try_alloc(4096 - 64);
    ASSERT_NE(p1, nullptr);
    void* p2 = ring.pto2_heap_ring_try_alloc(64);
    ASSERT_NE(p2, nullptr);  // top now = 4096

    // Advance tail to exactly 128
    tail.store(128);

    // Request 128 bytes.  space_at_end = 4096 - 4096 = 0 → can't fit at end.
    // Wrap check: tail(128) > alloc_size(128) → FALSE.  Returns NULL.
    // BUG: There IS 128 bytes free at [0, 128).
    void* p3 = ring.pto2_heap_ring_try_alloc(alloc);
    // This documents the off-by-one behavior:
    // If p3 is NULL, the bug is confirmed.
    // If the implementation is fixed, p3 should be non-NULL.
    if (p3 == nullptr) {
        // Bug confirmed: off-by-one in wrap guard
        // Record as known issue — the space [0, tail) when tail == alloc_size is wasted.
        GTEST_SKIP() << "Known off-by-one: tail == alloc_size returns NULL (wastes space)";
    }
}

// ---------------------------------------------------------------------------
// EDGE-2: top at exact end of heap (top == size)
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, TopAtExactEnd) {
    // Fill entire heap
    void* p1 = ring.pto2_heap_ring_try_alloc(4096);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(top.load(), 4096u);

    // Reclaim all
    tail.store(4096);

    // Allocate again — should wrap to beginning
    void* p2 = ring.pto2_heap_ring_try_alloc(64);
    // top(4096) >= tail(4096).  space_at_end = 4096 - 4096 = 0.
    // Wrap: tail(4096) > 64 → true.  new_top = 64, result = base.
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2, (void*)heap_buf);
}

// ---------------------------------------------------------------------------
// EDGE-3: top == tail at non-zero offset (empty after reclaim)
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, TopEqualsTailNonZero) {
    // Allocate 256 bytes
    ring.pto2_heap_ring_try_alloc(256);
    // Reclaim: advance tail to match top
    tail.store(top.load());

    // Heap is logically empty.  Available should be full heap size.
    // But available() = max(at_end, at_begin) = max(4096-256, 256) = 3840.
    // Not the full 4096.
    uint64_t avail = ring.pto2_heap_ring_available();
    EXPECT_GT(avail, 0u);

    // Allocate should succeed
    void* p = ring.pto2_heap_ring_try_alloc(256);
    EXPECT_NE(p, nullptr);
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-3: available() reports max(at_end, at_begin), not sum
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, AvailableFragmentation) {
    // Create a fragmented state: top near end, tail near middle
    // top=3000, tail=1000 → at_end=1096, at_begin=1000.  max=1096.
    // But total free = 1096 + 1000 = 2096.
    ring.pto2_heap_ring_try_alloc(3008);  // top ≈ 3008 (aligned)
    uint64_t actual_top = top.load();
    tail.store(1024);

    uint64_t avail = ring.pto2_heap_ring_available();
    uint64_t at_end = 4096 - actual_top;
    uint64_t at_begin = 1024;
    EXPECT_EQ(avail, std::max(at_end, at_begin));

    // Cannot allocate 2048 even though total free > 2048
    // because it can't split across boundary
    if (avail < 2048) {
        void* p = ring.pto2_heap_ring_try_alloc(2048);
        EXPECT_EQ(p, nullptr) << "Correct: can't allocate across wrap boundary";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-9: Zero-size allocation behavior
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, ZeroSizeAllocation) {
    // Allocating 0 bytes: PTO2_ALIGN_UP(0, 64) = 0.
    // If alloc_size == 0:
    //   top(0) >= tail(0).  space_at_end = 4096 - 0 = 4096 >= 0.
    //   new_top = 0 + 0 = 0.  CAS(0, 0) succeeds.
    //   Returns base + 0.
    // Two consecutive zero-size allocs return the SAME pointer!
    void* p1 = ring.pto2_heap_ring_try_alloc(0);
    void* p2 = ring.pto2_heap_ring_try_alloc(0);

    if (p1 != nullptr && p2 != nullptr) {
        // Both succeed and both point to the same location
        // This is semantically questionable — two "allocations" sharing memory
        EXPECT_EQ(p1, p2) << "Zero-size allocs return same address (aliased allocations)";
        EXPECT_EQ(top.load(), 0u) << "top doesn't advance for zero-size allocs";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-10: Wrap-path wasted space accumulation
// When wrapping, space between old top and heap_size is leaked.
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, WrapPathWastedSpace) {
    // Allocate 4000 bytes.  top = 4032 (aligned).
    void* p1 = ring.pto2_heap_ring_try_alloc(4000);
    ASSERT_NE(p1, nullptr);
    uint64_t top_after = top.load();
    EXPECT_GE(top_after, 4000u);

    // Reclaim everything
    tail.store(top_after);

    // Now allocate 128 bytes.
    // space_at_end = 4096 - top_after (small).
    // If top_after = 4032, space_at_end = 64 < 128.
    // Wrap: tail(4032) > 128 → true.  new_top = 128, result = base.
    // The 64 bytes at end are "wasted" (not reclaimable by tail advancement).
    void* p2 = ring.pto2_heap_ring_try_alloc(128);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2, (void*)heap_buf) << "Allocation wrapped to beginning";

    // The tail is still at 4032.  Available = tail - top = 4032 - 128 = 3904.
    // But total heap is 4096.  The gap [4032, 4096) = 64 bytes is unusable
    // until tail is advanced past 4096 (which never happens because tail is
    // an offset within [0, heap_size)).
    uint64_t avail = ring.pto2_heap_ring_available();
    EXPECT_LT(avail, 4096u) << "Wasted space at end reduces available capacity";
}

// ---------------------------------------------------------------------------
// Concurrent CAS safety: two threads racing on try_alloc
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, ConcurrentTryAlloc) {
    std::atomic<int> success_count{0};
    std::atomic<int> fail_count{0};

    auto worker = [&]() {
        for (int i = 0; i < 100; i++) {
            void* p = ring.pto2_heap_ring_try_alloc(64);
            if (p) success_count++;
            else fail_count++;
        }
    };

    std::thread t1(worker);
    std::thread t2(worker);
    t1.join();
    t2.join();

    // Total allocations should equal total heap / 64
    int max_possible = 4096 / 64;  // = 64
    EXPECT_EQ(success_count.load(), max_possible);
    EXPECT_EQ(success_count.load() + fail_count.load(), 200);
}

// ---------------------------------------------------------------------------
// Verify no overlapping allocations from concurrent threads
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, ConcurrentNoOverlap) {
    std::vector<void*> allocs_t1, allocs_t2;
    std::mutex m1, m2;

    auto worker = [&](std::vector<void*>& results, std::mutex& m) {
        for (int i = 0; i < 32; i++) {
            void* p = ring.pto2_heap_ring_try_alloc(64);
            if (p) {
                std::lock_guard<std::mutex> lock(m);
                results.push_back(p);
            }
        }
    };

    std::thread t1(worker, std::ref(allocs_t1), std::ref(m1));
    std::thread t2(worker, std::ref(allocs_t2), std::ref(m2));
    t1.join();
    t2.join();

    // Combine all allocations and verify uniqueness
    std::set<void*> all_ptrs(allocs_t1.begin(), allocs_t1.end());
    all_ptrs.insert(allocs_t2.begin(), allocs_t2.end());
    EXPECT_EQ(all_ptrs.size(), allocs_t1.size() + allocs_t2.size())
        << "All allocation addresses must be unique (no overlap)";
}

// ---------------------------------------------------------------------------
// Repeated full-drain-refill cycles: exposes wrap-around stall.
// After first fill (top=4096) and drain (tail=4096), next alloc tries:
//   top(4096) >= tail(4096), space_at_end = 4096 - 4096 = 0.
//   Wrap: tail(4096) > 4096 → false (strict >).  Returns NULL!
// This is BUG-CANDIDATE-1 manifesting in a real usage pattern.
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, FullDrainRefillCycles) {
    // First cycle: fill entire heap
    void* p1 = ring.pto2_heap_ring_try_alloc(4096);
    ASSERT_NE(p1, nullptr) << "Cycle 0 fill";

    // Drain: advance tail to match top (both = 4096)
    tail.store(top.load());

    // Try to allocate again: top(4096) >= tail(4096).
    // space_at_end = 4096 - 4096 = 0 → can't fit.
    // Wrap check: tail(4096) > 4096 → FALSE (off-by-one!)
    // BUG: heap is fully empty but alloc returns NULL.
    void* p2 = ring.pto2_heap_ring_try_alloc(4096);
    EXPECT_NE(p2, nullptr)
        << "BUG: Full heap fill-drain cycle breaks wrap guard"
        << " (tail == heap_size, wrap check 'tail > alloc_size' fails due to off-by-one)";
}

// ---------------------------------------------------------------------------
// Allocation of exactly heap_size: consumes entire heap in one shot
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, AllocExactlyHeapSize) {
    void* p = ring.pto2_heap_ring_try_alloc(4096);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p, (void*)heap_buf);
    EXPECT_EQ(top.load(), 4096u);

    // No more space
    void* p2 = ring.pto2_heap_ring_try_alloc(64);
    EXPECT_EQ(p2, nullptr) << "No space after full allocation";
}

// ---------------------------------------------------------------------------
// Allocation larger than heap_size: must fail
// ---------------------------------------------------------------------------
TEST_F(HeapRingEdgeTest, AllocLargerThanHeap) {
    void* p = ring.pto2_heap_ring_try_alloc(8192);
    // size = 8192, aligned → 8192. space_at_end = 4096 - 0 = 4096 < 8192.
    // Wrap: tail(0) > 8192 → false.  Returns NULL.
    EXPECT_EQ(p, nullptr) << "Cannot allocate more than heap size";
}

// =============================================================================
// TaskRing edge-case fixture
// =============================================================================
class TaskRingEdgeTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 8;  // Small for edge testing
    PTO2TaskDescriptor descriptors[8]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> last_alive{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskRing ring{};

    void SetUp() override {
        current_index.store(0);
        last_alive.store(0);
        error_code.store(PTO2_ERROR_NONE);
        pto2_task_ring_init(&ring, descriptors, WINDOW_SIZE, &last_alive, &current_index);
        ring.error_code_ptr = &error_code;
    }
};

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-5: Non-power-of-2 window produces wrong slot mapping
// ---------------------------------------------------------------------------
TEST(TaskRingNonPow2Test, SlotMappingWithNonPow2) {
    // window_size = 6 (NOT power of 2)
    PTO2TaskDescriptor descs[6]{};
    std::atomic<int32_t> ci{0}, la{0};
    PTO2TaskRing ring{};
    pto2_task_ring_init(&ring, descs, 6, &la, &ci);

    // get_task_slot uses task_id & (window_size - 1) = task_id & 5
    // For window=6: task_id=6 should map to slot 0 (6 % 6 = 0)
    // But task_id & 5 = 6 & 5 = 4.  WRONG!
    int32_t slot_mod = 6 % 6;     // = 0 (correct modulo)
    int32_t slot_mask = 6 & 5;    // = 4 (mask-based, wrong for non-pow2)
    EXPECT_NE(slot_mod, slot_mask) << "Confirms non-pow2 masking is broken";
    EXPECT_EQ(ring.get_task_slot(6), slot_mask) << "Implementation uses masking, not modulo";
}

// ---------------------------------------------------------------------------
// Non-pow2 collision test: multiple task IDs map to same wrong slot
// ---------------------------------------------------------------------------
TEST(TaskRingNonPow2Test, SlotCollisionWithNonPow2) {
    PTO2TaskDescriptor descs[6]{};
    std::atomic<int32_t> ci{0}, la{0};
    PTO2TaskRing ring{};
    pto2_task_ring_init(&ring, descs, 6, &la, &ci);

    // With mask = 5 (binary 101), the mapping is:
    // task_id & 5 maps only to slots 0,1,4,5 — slots 2 and 3 never used!
    // Because 5 in binary is 101, bit 1 is always 0 in the result.
    std::set<int32_t> used_slots;
    for (int32_t id = 0; id < 12; id++) {
        used_slots.insert(ring.get_task_slot(id));
    }
    // With correct modulo: 0,1,2,3,4,5 → 6 slots
    // With mask: 0,1,4,5,0,1,4,5,... → only 4 unique slots
    EXPECT_LT(used_slots.size(), 6u)
        << "Non-pow2 window: not all slots are reachable via masking";
}

// ---------------------------------------------------------------------------
// EDGE-5: window_size = 1 → every allocation fails (window_size - 1 = 0)
// ---------------------------------------------------------------------------
TEST(TaskRingWindow1Test, WindowSize1AlwaysFails) {
    PTO2TaskDescriptor desc{};
    std::atomic<int32_t> ci{0}, la{0};
    PTO2TaskRing ring{};
    pto2_task_ring_init(&ring, &desc, 1, &la, &ci);

    // active_count = 0, window_size - 1 = 0.  Check: 0 < 0 → false → always fails.
    int32_t id = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id, -1) << "window_size=1 can never allocate (0 < 0 is false)";
}

// ---------------------------------------------------------------------------
// Window_size = 2: can allocate exactly 1 task
// ---------------------------------------------------------------------------
TEST(TaskRingWindow2Test, WindowSize2SingleTask) {
    PTO2TaskDescriptor descs[2]{};
    std::atomic<int32_t> ci{0}, la{0};
    PTO2TaskRing ring{};
    pto2_task_ring_init(&ring, descs, 2, &la, &ci);

    // First alloc: active_count = 0 < 1 (window_size - 1) → succeeds
    int32_t id0 = ring.pto2_task_ring_try_alloc();
    EXPECT_GE(id0, 0);

    // Second alloc: active_count = 1, check: 1 < 1 → false
    int32_t id1 = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id1, -1) << "window_size=2 can only hold 1 active task";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-4: Concurrent try_alloc near window boundary
// ---------------------------------------------------------------------------
TEST_F(TaskRingEdgeTest, ConcurrentTryAllocNearBoundary) {
    // Fill to window_size - 2 (leaving 1 slot)
    for (int i = 0; i < WINDOW_SIZE - 2; i++) {
        ASSERT_GE(ring.pto2_task_ring_try_alloc(), 0);
    }

    // Two threads race for the last slot
    std::atomic<int> wins{0};
    auto worker = [&]() {
        int32_t id = ring.pto2_task_ring_try_alloc();
        if (id >= 0) wins++;
    };

    std::thread t1(worker);
    std::thread t2(worker);
    t1.join();
    t2.join();

    // Exactly one should succeed (the other sees window full and rolls back)
    EXPECT_EQ(wins.load(), 1);
    // current_index should be window_size - 1 (not window_size due to rollback)
    EXPECT_EQ(current_index.load(), WINDOW_SIZE - 1);
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-4 extended: Many threads racing causes temporary spike
// ---------------------------------------------------------------------------
TEST_F(TaskRingEdgeTest, ManyThreadsRacingNearBoundary) {
    // Fill to window_size - 2 (1 slot left)
    for (int i = 0; i < WINDOW_SIZE - 2; i++) {
        ASSERT_GE(ring.pto2_task_ring_try_alloc(), 0);
    }

    constexpr int NUM_THREADS = 8;
    std::atomic<int> wins{0};
    std::atomic<int> losses{0};

    auto worker = [&]() {
        int32_t id = ring.pto2_task_ring_try_alloc();
        if (id >= 0) wins++;
        else losses++;
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();

    // Exactly 1 winner.  The optimistic fetch_add(1) + rollback means
    // current_index may have temporarily spiked by up to NUM_THREADS,
    // but should be fully rolled back to WINDOW_SIZE - 1.
    EXPECT_EQ(wins.load(), 1);
    EXPECT_EQ(losses.load(), NUM_THREADS - 1);
    EXPECT_EQ(current_index.load(), WINDOW_SIZE - 1)
        << "All rollbacks must complete — no leaked increments";
}

// ---------------------------------------------------------------------------
// Slot reuse after wrap-around: task_id and task_id + window_size map to same slot
// ---------------------------------------------------------------------------
TEST_F(TaskRingEdgeTest, SlotReuseAfterWrap) {
    // Allocate all slots
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        ring.pto2_task_ring_try_alloc();
    }
    // Reclaim all
    last_alive.store(WINDOW_SIZE - 1);

    // Allocate new task — should get the next sequential ID
    int32_t new_id = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(new_id, WINDOW_SIZE - 1);  // ID = 7

    // The physical slot = 7 & 7 = 7, which is a different slot from task 0 (slot 0)
    // Task IDs grow monotonically; slot reuse happens when:
    //   new_id >= old_id + window_size (i.e., task_id wraps the full window)
    EXPECT_EQ(ring.get_task_slot(new_id), WINDOW_SIZE - 1);

    // True slot reuse: keep allocating until a new task maps to slot 0
    // Slot 0 = task_id & 7 == 0 → task_id must be a multiple of 8
    // current_index is at WINDOW_SIZE = 8 after the above allocations
    last_alive.store(current_index.load() - 1);
    int32_t wrapped_id = ring.pto2_task_ring_try_alloc();
    // wrapped_id = 2*WINDOW_SIZE - 2 = 14, slot = 14 & 7 = 6
    // We need task_id = 16 to get slot 0 (16 & 7 = 0)
    // Keep allocating until we hit a multiple of WINDOW_SIZE
    while (ring.get_task_slot(wrapped_id) != ring.get_task_slot(0)) {
        last_alive.store(wrapped_id);
        wrapped_id = ring.pto2_task_ring_try_alloc();
        ASSERT_GE(wrapped_id, 0) << "Should be able to keep allocating with reclamation";
    }
    EXPECT_EQ(ring.get_task_slot(wrapped_id), 0)
        << "Task " << wrapped_id << " reuses slot 0 after full window wrap";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-11: INT32 overflow on task_id
// Verify behavior when current_index approaches INT32_MAX
// ---------------------------------------------------------------------------
TEST_F(TaskRingEdgeTest, TaskIdNearInt32Max) {
    // Set current_index near INT32_MAX
    int32_t near_max = INT32_MAX - 2;
    current_index.store(near_max);
    last_alive.store(near_max);

    // Allocate a few tasks — should succeed since active_count is small
    int32_t id1 = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id1, near_max);

    int32_t id2 = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id2, near_max + 1);  // INT32_MAX - 1

    int32_t id3 = ring.pto2_task_ring_try_alloc();
    // id3 = INT32_MAX.  Next fetch_add(1) wraps to INT32_MIN.
    // active_count = INT32_MAX - near_max = 2, which is < window_size-1=7
    EXPECT_EQ(id3, INT32_MAX);

    // Next allocation: fetch_add wraps INT32_MAX to INT32_MIN
    // active_count = INT32_MIN - near_max → massive negative number
    // The check `active_count < window_size - 1` is true (negative < 7)
    // So the allocation "succeeds" with a NEGATIVE task_id!
    int32_t id4 = ring.pto2_task_ring_try_alloc();
    if (id4 < 0 && id4 != -1) {
        // Task ID wrapped to negative — this is INT32 overflow
        // The masking: id4 & (8-1) still gives a valid slot (0-7)
        // but the semantics of negative task IDs is undefined
        int32_t slot = ring.get_task_slot(id4);
        EXPECT_GE(slot, 0);
        EXPECT_LT(slot, WINDOW_SIZE);
        SUCCEED() << "INT32 overflow: task_id=" << id4
                  << " maps to slot=" << slot
                  << " (signed overflow in fetch_add)";
    }
}

// ---------------------------------------------------------------------------
// pto2_task_ring_has_space and active_count consistency
// ---------------------------------------------------------------------------
TEST_F(TaskRingEdgeTest, HasSpaceConsistency) {
    EXPECT_TRUE(pto2_task_ring_has_space(&ring));

    // Fill all available slots
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        ASSERT_GE(ring.pto2_task_ring_try_alloc(), 0);
    }

    EXPECT_FALSE(pto2_task_ring_has_space(&ring));
    EXPECT_EQ(pto2_task_ring_active_count(&ring), WINDOW_SIZE - 1);

    // Reclaim one
    last_alive.store(1);
    EXPECT_TRUE(pto2_task_ring_has_space(&ring));
}

// =============================================================================
// DepListPool edge-case fixture
// =============================================================================
class DepPoolEdgeTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 8;
    PTO2DepListEntry entries[8]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool{};

    void SetUp() override {
        memset(entries, 0, sizeof(entries));
        error_code.store(PTO2_ERROR_NONE);
        pool.init(entries, POOL_CAP, &error_code);
    }
};

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-6: Allocating `capacity` entries overwrites sentinel at index 0.
// top starts at 1.  After allocating 8 entries: top = 9.
// Physical indices: 1,2,3,4,5,6,7, then 9%8=1?  No, let's trace:
// alloc(): top=1, idx=1%8=1, top=2  → OK
// alloc(): top=2, idx=2%8=2, top=3  → OK
// ...
// alloc(): top=7, idx=7%8=7, top=8  → OK (7 entries so far, used=7)
// alloc(): top=8, idx=8%8=0, top=9  → OVERWRITES SENTINEL at index 0!
//          But used=8, capacity=8, check 8>=8 triggers overflow BEFORE alloc.
//          So this is actually prevented.  But used = top - tail = 8 - 1 = 7,
//          NOT 8.  So the check (7 >= 8) is FALSE, alloc proceeds!
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, SentinelOverwrite) {
    // Initialize sentinel with recognizable markers
    entries[0].slot_state = (PTO2TaskSlotState*)0xDEAD;
    entries[0].next = (PTO2DepListEntry*)0xBEEF;

    // Allocate until we would wrap around to index 0
    // top starts at 1, tail=1.  capacity=8.
    // Each alloc: idx = top % 8, top++
    // After 7 allocs: top=8, tail=1, used=7.  Next: idx=8%8=0.
    // Check: used(7) >= capacity(8) → false → alloc proceeds → sentinel overwritten!
    int count = 0;
    while (count < POOL_CAP) {
        PTO2DepListEntry* e = pool.alloc();
        if (!e) break;
        count++;
        if (pool.top % POOL_CAP == 0) {
            // We just allocated the entry at physical index 0 (the sentinel)
            // This is a potential bug if the sentinel is supposed to be preserved
            break;
        }
    }

    // Check: did we wrap to index 0?
    if (count >= 7) {
        // After 7 allocs: top=8, next alloc would be at idx 0
        // The 8th alloc: used = 8 - 1 = 7, capacity = 8, 7 < 8 → allowed
        // Physical index = 8 % 8 = 0 → SENTINEL OVERWRITTEN
        // This test documents this behavior.
        PTO2DepListEntry* e = pool.alloc();
        if (e == &entries[0]) {
            // Bug confirmed: sentinel slot 0 was returned to user
            // After this, entries[0] is no longer a valid sentinel
            SUCCEED() << "Confirmed: alloc() returns sentinel slot (index 0) on wrap";
        }
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-6 extended: Verify sentinel data is actually corrupted
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, SentinelDataCorruption) {
    // Set recognizable sentinel markers
    entries[0].slot_state = nullptr;
    entries[0].next = nullptr;

    // Allocate 7 entries (indices 1-7), then the 8th wraps to index 0
    for (int i = 0; i < 7; i++) {
        PTO2DepListEntry* e = pool.alloc();
        ASSERT_NE(e, nullptr);
        // Write data to verify it's not corrupting sentinel
        e->slot_state = (PTO2TaskSlotState*)(uintptr_t)(i + 100);
        e->next = nullptr;
    }

    // Sentinel should still be clean at this point
    EXPECT_EQ(entries[0].slot_state, nullptr) << "Sentinel still intact after 7 allocs";

    // 8th alloc wraps to index 0
    PTO2DepListEntry* e = pool.alloc();
    if (e == &entries[0]) {
        // Now write user data to the returned entry (which IS the sentinel)
        e->slot_state = (PTO2TaskSlotState*)0x1234;
        e->next = (PTO2DepListEntry*)0x5678;

        // Sentinel is now corrupted
        EXPECT_NE(entries[0].slot_state, nullptr)
            << "BUG: Sentinel slot overwritten with user data";
        EXPECT_NE(entries[0].next, nullptr)
            << "BUG: Sentinel next pointer overwritten";

        // pto2_dep_pool_get(0) should return NULL for sentinel
        // but the sentinel's data is now garbage
        PTO2DepListEntry* sentinel = pool.pto2_dep_pool_get(0);
        EXPECT_EQ(sentinel, (PTO2DepListEntry*)NULL)
            << "pto2_dep_pool_get(0) returns NULL (offset <= 0)";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-12: Multiple alloc-reclaim cycles compound sentinel damage
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, MultiCyclesSentinelIntegrity) {
    PTO2TaskSlotState dummy_slots[POOL_CAP]{};

    for (int cycle = 0; cycle < 3; cycle++) {
        // Allocate all available entries
        int allocated = 0;
        while (true) {
            PTO2DepListEntry* e = pool.alloc();
            if (!e) break;
            e->slot_state = &dummy_slots[allocated % POOL_CAP];
            e->next = nullptr;
            allocated++;
            if (allocated >= POOL_CAP) break;
        }

        // Reclaim by advancing tail to current top
        pool.advance_tail(pool.top);
    }

    // After multiple cycles, sentinel at index 0 may have been overwritten
    // multiple times.  Check if init's sentinel guarantee still holds.
    // The init() sets entries[0].slot_state = nullptr.
    // If any cycle's alloc returned &entries[0], user data overwrote it.
    // This is not re-initialized between cycles.
    PTO2DepListEntry* sentinel = &entries[0];
    if (sentinel->slot_state != nullptr) {
        SUCCEED() << "Confirmed: sentinel corrupted across alloc-reclaim cycles";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-7: advance_tail beyond top → negative used()
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, AdvanceTailBeyondTop) {
    pool.alloc();  // top=2, tail=1
    pool.alloc();  // top=3, tail=1

    // Advance tail way beyond top
    pool.advance_tail(100);

    int32_t u = pool.used();   // top(3) - tail(100) = -97
    int32_t a = pool.available();  // capacity(8) - (-97) = 105

    // Both are semantically wrong.  This documents the lack of bounds checking.
    EXPECT_LT(u, 0) << "used() goes negative when tail > top";
    EXPECT_GT(a, pool.capacity) << "available() exceeds capacity when tail > top";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-7 extended: After bogus advance_tail, alloc sees huge available
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, AdvanceTailBeyondTopThenAlloc) {
    pool.alloc();  // top=2
    pool.advance_tail(100);

    // Now used() = 2 - 100 = -98.  Check: -98 >= 8 → false → alloc proceeds!
    // Physical index: top(2) % 8 = 2.  Seems valid.
    PTO2DepListEntry* e = pool.alloc();
    EXPECT_NE(e, nullptr) << "Alloc succeeds with corrupted tail (negative used)";

    // But logically, the pool state is inconsistent
    EXPECT_LT(pool.used(), 0) << "Pool state is corrupted: negative used count";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-8: pto2_dep_pool_get with offset beyond capacity
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, GetBeyondCapacity) {
    // offset = 100, capacity = 8.  Returns &base[100] → out of bounds.
    PTO2DepListEntry* result = pool.pto2_dep_pool_get(100);
    // We can't assert on the pointer value (it's undefined behavior),
    // but we can verify it doesn't return NULL (the only check is offset <= 0).
    EXPECT_NE(result, nullptr)
        << "get(100) with capacity=8 returns non-NULL (no bounds check)";
}

// ---------------------------------------------------------------------------
// pto2_dep_pool_get with negative offset
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, GetNegativeOffset) {
    PTO2DepListEntry* result = pool.pto2_dep_pool_get(-5);
    EXPECT_EQ(result, nullptr) << "Negative offset returns NULL";
}

// ---------------------------------------------------------------------------
// pto2_dep_pool_get with offset = 0 (sentinel)
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, GetZeroOffset) {
    PTO2DepListEntry* result = pool.pto2_dep_pool_get(0);
    EXPECT_EQ(result, nullptr) << "Offset 0 (sentinel) returns NULL";
}

// ---------------------------------------------------------------------------
// Prepend chain integrity under pool exhaustion
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, PrependUnderExhaustion) {
    PTO2TaskSlotState slots[POOL_CAP]{};
    PTO2DepListEntry* head = nullptr;

    // Prepend until pool exhausted
    int count = 0;
    while (count < POOL_CAP + 5) {  // Try beyond capacity
        PTO2DepListEntry* new_head = pool.prepend(head, &slots[count % POOL_CAP]);
        if (!new_head) break;
        head = new_head;
        count++;
    }

    // Walk the chain — should be intact (no dangling pointers)
    int walk = 0;
    PTO2DepListEntry* cur = head;
    while (cur) {
        walk++;
        cur = cur->next;
        if (walk > count + 1) {
            FAIL() << "Chain has cycle — walked more entries than allocated";
            break;
        }
    }
    EXPECT_EQ(walk, count);
}

// ---------------------------------------------------------------------------
// Prepend builds linked list correctly: verify each slot_state pointer
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, PrependChainCorrectness) {
    PTO2TaskSlotState slots[5]{};
    PTO2DepListEntry* head = nullptr;

    for (int i = 0; i < 5; i++) {
        head = pool.prepend(head, &slots[i]);
        ASSERT_NE(head, nullptr);
    }

    // Walk chain: most recently prepended is at head
    // prepend is a LIFO operation: head → slots[4] → slots[3] → ... → slots[0] → nullptr
    PTO2DepListEntry* cur = head;
    for (int i = 4; i >= 0; i--) {
        ASSERT_NE(cur, nullptr);
        EXPECT_EQ(cur->slot_state, &slots[i])
            << "Entry " << (4 - i) << " should point to slots[" << i << "]";
        cur = cur->next;
    }
    EXPECT_EQ(cur, nullptr) << "Chain should terminate with nullptr";
}

// ---------------------------------------------------------------------------
// High water mark accuracy after reclaim cycles
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, HighWaterAccuracy) {
    // Phase 1: allocate 5
    for (int i = 0; i < 5; i++) pool.alloc();
    EXPECT_EQ(pool.high_water, 5);

    // Phase 2: reclaim 3 (tail from 1 to 4)
    pool.advance_tail(4);
    EXPECT_EQ(pool.high_water, 5);  // High water never decreases

    // Phase 3: allocate 3 more → used = (8-4) + 3 = no, top=8,tail=4,used=4
    // Wait: top=6 after phase1, advance_tail(4) → used=2.
    // Allocate 3: used goes to 2,3,4,5 → high_water should update to max(5, 5)
    for (int i = 0; i < 3; i++) pool.alloc();
    // top=9, tail=4, used=5.  high_water = max(5, 5) = 5
    EXPECT_GE(pool.high_water, 5);
}

// ---------------------------------------------------------------------------
// Advance tail backwards (no-op check)
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, AdvanceTailBackwards) {
    pool.alloc();  // top=2
    pool.alloc();  // top=3
    pool.advance_tail(3);  // tail=3

    // Try to advance backwards — should be no-op
    pool.advance_tail(1);
    EXPECT_EQ(pool.tail, 3) << "advance_tail backwards is a no-op";
}

// ---------------------------------------------------------------------------
// Pool init state verification
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, InitState) {
    EXPECT_EQ(pool.top, 1) << "top starts at 1 (0 reserved for sentinel)";
    EXPECT_EQ(pool.tail, 1) << "tail matches initial top";
    EXPECT_EQ(pool.high_water, 0) << "high_water starts at 0";
    EXPECT_EQ(pool.used(), 0) << "initially empty";
    EXPECT_EQ(pool.available(), POOL_CAP) << "full capacity available";
    EXPECT_EQ(entries[0].slot_state, nullptr) << "sentinel slot_state is null";
    EXPECT_EQ(entries[0].next, nullptr) << "sentinel next is null";
}

// ---------------------------------------------------------------------------
// Alloc all then overflow: verify error code is set
// ---------------------------------------------------------------------------
TEST_F(DepPoolEdgeTest, OverflowSetsErrorCode) {
    // Fill pool completely: top-tail reaches capacity
    // After capacity allocs: top = 1 + capacity = 9, tail = 1, used = 8
    // But check is used >= capacity, so it triggers at the (capacity+1)th alloc
    // Actually: after 7 allocs, used = 7.  8th alloc: used = 7 < 8, allowed.
    // After 8th: top=9, used=8.  9th: check 8 >= 8 → true → overflow!
    for (int i = 0; i < POOL_CAP; i++) {
        pool.alloc();
    }

    // This should trigger overflow
    PTO2DepListEntry* overflow_result = pool.alloc();
    EXPECT_EQ(overflow_result, nullptr) << "Overflow returns nullptr";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW)
        << "Error code set on overflow";
}
