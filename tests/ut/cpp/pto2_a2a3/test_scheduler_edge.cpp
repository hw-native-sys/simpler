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
 * Edge-case tests for ReadyQueue, SharedMemory, and TaskState.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- PTO2ReadyQueue (Vyukov MPMC)
 * ============================================================================
 *
 * DC-1 (sequence wrap): The sequence counter is int64_t.  After 2^63
 *   push/pop operations it wraps; comparisons still work because both
 *   positions wrap identically (two's complement).  Practically
 *   unreachable; `-ftrapv` would convert it into a crash.
 *
 * DC-2 (pop fast-path): pop() checks `enqueue_pos == dequeue_pos` as an
 *   early-empty hint.  A push between the hint and the CAS can race; this
 *   is the standard TOCTOU of Vyukov MPMC queues and acceptable.
 *
 * DC-3 (push returns false near full): All producers that see a full slot
 *   return false simultaneously even if a pop happens right after.
 *   Acceptable back-pressure, not a defect.
 *
 * DC-9 (size() relaxed ordering): size() reads both positions with
 *   memory_order_relaxed and is a hint, not a point-in-time snapshot.
 *   If a stale read produces d > e the guard returns 0.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- Scheduler
 * ============================================================================
 *
 * DC-10 (release_fanin_and_check_ready CAS): the non-profiling overload
 *   does NOT CAS task_state before pushing.  The profiling overload CASes
 *   purely so the operation can be counted; dispatch correctness in both
 *   builds derives from `fanin_refcount.fetch_add` -- only the thread that
 *   observes `new_refcount == fanin_count` pushes.  NOT a bug.
 *
 * DC-11 (LocalReadyBuffer LIFO dispatch): try_push appends at count++, pop
 *   returns slot_states[--count].  LIFO reversal is intentional for
 *   cache-locality when a producer immediately dispatches its fanout.
 *
 * DC-12 (on_subtask_complete double-completion): fetch_add is idempotent
 *   on a pure counter; a repeat call returns false because prev+1 !=
 *   total_required_subtasks.  No detection of double-call as a logic error
 *   -- caller contract.
 *
 * DC-13 (advance_ring_pointers, FORMERLY a null-deref candidate):
 *   HISTORICAL -- advance_ring_pointers no longer touches slot.task at all.
 *   It reads task_state == PTO2_TASK_CONSUMED only (see pto_scheduler.h).
 *   Heap reclamation was moved to PTO2TaskAllocator::update_heap_tail.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- SharedMemory
 * ============================================================================
 *
 * DC-4 (pto2_sm_validate): checks `top > heap_size`.  top == heap_size is
 *   a legitimate "filled exactly to end" state, so strict > is correct.
 *
 * BUG-CANDIDATE-5 (size calculation with task_window_size=0): if the
 *   runtime ever called `pto2_sm_calculate_size()` with 0, all ring
 *   descriptors/payloads would alias the same address.  The current entry
 *   path is pto2_sm_create, which is called only with valid sizes, but
 *   there is no explicit guard.  Real defect -- pto2_sm_create should
 *   reject task_window_size==0.  Tests below pin this behavior.
 *
 * BUG-CANDIDATE-6 (flow control heap_top validation): `validate()` does
 *   not verify `heap_top <= heap_size`.  After a corruption or an
 *   unbounded caller, heap_top could exceed heap_size without detection.
 *   Real defect -- validate should check both bounds.
 *
 * ============================================================================
 * DESIGN CONTRACTS -- TaskState
 * ============================================================================
 *
 * EDGE-1: CAS on task_state with memory_order_relaxed could reorder with
 *   subsequent reads of fanin_refcount.  The actual scheduler code uses
 *   acquire/release on task_state.
 *
 * EDGE-2: completed_subtasks uses fetch_add(1) with acq_rel ordering; the
 *   thread that observes (prev+1) == total is the sole completer.
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <set>
#include <cstring>
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "../test_helpers.h"

// =============================================================================
// ReadyQueue edge cases
// =============================================================================
class ReadyQueueEdgeTest : public ::testing::Test {
protected:
    static constexpr uint64_t QUEUE_CAP = 8;  // Small for edge testing
    PTO2ReadyQueueSlot slots[8]{};
    PTO2ReadyQueue queue{};
    PTO2TaskSlotState dummy[8]{};

    void SetUp() override { test_ready_queue_init(&queue, slots, QUEUE_CAP); }
};

// ---------------------------------------------------------------------------
// Push and pop interleaving: push(A), pop() -> A, push(B), pop() -> B
// Ensures sequence numbers are correctly advanced after each operation.
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, InterleavedPushPop) {
    for (int i = 0; i < 20; i++) {
        EXPECT_TRUE(queue.push(&dummy[0]));
        PTO2TaskSlotState *s = queue.pop();
        EXPECT_EQ(s, &dummy[0]);
    }
    // After 20 interleaved push/pop, queue should be empty
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.pop(), nullptr);
}

// ---------------------------------------------------------------------------
// Exactly fill queue, then pop all -- boundary at capacity
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, ExactCapacityFillDrain) {
    // Push exactly capacity items
    int pushed = 0;
    for (uint64_t i = 0; i < QUEUE_CAP; i++) {
        if (queue.push(&dummy[i % 8])) pushed++;
        else break;
    }
    // Vyukov MPMC with capacity N can hold N-1 items (one slot is always empty)
    // OR exactly N depending on implementation.
    // The actual implementation checks `sequence == pos` which allows N items.
    EXPECT_GE(pushed, (int)(QUEUE_CAP - 1));

    // Pop all
    for (int i = 0; i < pushed; i++) {
        EXPECT_NE(queue.pop(), nullptr);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// ---------------------------------------------------------------------------
// Push to full queue: must return false
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, PushToFullQueue) {
    // Fill the queue
    int pushed = 0;
    while (queue.push(&dummy[0]))
        pushed++;

    // Queue is now full
    EXPECT_FALSE(queue.push(&dummy[1])) << "Push to full queue returns false";

    // Pop one, then push should succeed again
    EXPECT_NE(queue.pop(), nullptr);
    EXPECT_TRUE(queue.push(&dummy[1])) << "Push succeeds after pop from full queue";
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-9: size() with relaxed ordering can be stale
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, SizeRelaxedOrdering) {
    // Push 3 items
    queue.push(&dummy[0]);
    queue.push(&dummy[1]);
    queue.push(&dummy[2]);

    // In single-threaded context, size should be exact
    EXPECT_EQ(queue.size(), 3u);

    // Pop 1
    queue.pop();
    EXPECT_EQ(queue.size(), 2u);

    // Pop remaining
    queue.pop();
    queue.pop();
    EXPECT_EQ(queue.size(), 0u);
}

// ---------------------------------------------------------------------------
// size() guard: after many push/pop cycles, size never goes negative
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, SizeNeverNegative) {
    // After many push/pop cycles, verify size() always returns a sane value
    for (int i = 0; i < 100; i++) {
        ASSERT_TRUE(queue.push(&dummy[0]));
        queue.pop();
    }
    // Queue is empty -- size must be 0, not negative or wrapped
    EXPECT_EQ(queue.size(), 0u) << "size() returns 0 after balanced push/pop cycles";
}

// ---------------------------------------------------------------------------
// FIFO ordering: items come out in the order they were pushed
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, FIFOOrdering) {
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(queue.push(&dummy[i]));
    }

    for (int i = 0; i < 5; i++) {
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s, &dummy[i]) << "FIFO: item " << i << " should come out in order";
    }
}

// ---------------------------------------------------------------------------
// Concurrent stress: many producers, many consumers, large volume
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, HighContentionStress) {
    // Use a larger queue for stress testing
    static constexpr uint64_t BIG_CAP = 256;
    PTO2ReadyQueueSlot big_slots[BIG_CAP];
    PTO2ReadyQueue big_queue{};
    test_ready_queue_init(&big_queue, big_slots, BIG_CAP);

    constexpr int N = 5000;
    constexpr int P = 4, C = 4;
    std::vector<PTO2TaskSlotState> items(N);
    std::atomic<int> produced{0}, consumed{0};

    auto producer = [&](int id) {
        for (int i = id; i < N; i += P) {
            while (!big_queue.push(&items[i])) {}
            produced++;
        }
    };
    auto consumer = [&]() {
        while (consumed.load() < N) {
            PTO2TaskSlotState *s = big_queue.pop();
            if (s) consumed++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < P; i++)
        threads.emplace_back(producer, i);
    for (int i = 0; i < C; i++)
        threads.emplace_back(consumer);
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(produced.load(), N);
    EXPECT_EQ(consumed.load(), N);
}

// ---------------------------------------------------------------------------
// Concurrent stress: verify no duplicates consumed
// Uses pointer identity (address comparison) instead of repurposing
// production struct fields as test tags.
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, NoDuplicateConsumption) {
    static constexpr uint64_t BIG_CAP = 128;
    PTO2ReadyQueueSlot big_slots[BIG_CAP];
    PTO2ReadyQueue big_queue{};
    test_ready_queue_init(&big_queue, big_slots, BIG_CAP);

    constexpr int N = 1000;
    std::vector<PTO2TaskSlotState> items(N);

    // Track consumed items by pointer address in a separate array
    std::vector<int> consumed_count(N, 0);

    auto item_index = [&](PTO2TaskSlotState *s) -> int {
        return static_cast<int>(s - items.data());
    };

    // Push all items
    for (int i = 0; i < N; i++) {
        while (!big_queue.push(&items[i])) {
            // Drain some if full
            PTO2TaskSlotState *s = big_queue.pop();
            if (s) consumed_count[item_index(s)]++;
        }
    }

    // Pop remaining
    while (true) {
        PTO2TaskSlotState *s = big_queue.pop();
        if (!s) break;
        consumed_count[item_index(s)]++;
    }

    // Verify each item consumed exactly once
    int total_consumed = 0;
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(consumed_count[i], 1) << "Item " << i << " consumed " << consumed_count[i] << " times";
        total_consumed += consumed_count[i];
    }
    EXPECT_EQ(total_consumed, N) << "Each item should be consumed exactly once";
}

// ---------------------------------------------------------------------------
// Pop from empty queue multiple times -- must always return nullptr
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, RepeatedEmptyPop) {
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(queue.pop(), nullptr);
    }
    // After 100 empty pops, size should still be 0
    EXPECT_EQ(queue.size(), 0u);
}

// ---------------------------------------------------------------------------
// Push-pop cycles beyond sequence counter wrap (small queue, many cycles)
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, ManyPushPopCycles) {
    // With capacity 8, sequence numbers grow by 1 per push/pop.
    // After many cycles, sequences grow large but should remain correct.
    for (int i = 0; i < 10000; i++) {
        ASSERT_TRUE(queue.push(&dummy[0]));
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s, &dummy[0]);
    }

    // Queue should be empty and still functional
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_TRUE(queue.push(&dummy[1]));
    EXPECT_EQ(queue.pop(), &dummy[1]);
}

// =============================================================================
// LocalReadyBuffer edge cases
// =============================================================================

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-11: LocalReadyBuffer LIFO dispatch order
// push adds at [count++], pop returns [--count].
// Last pushed = first popped = LIFO, not FIFO.
// ---------------------------------------------------------------------------
TEST(LocalReadyBufferTest, LIFODispatchOrder) {
    PTO2TaskSlotState *storage[8]{};
    PTO2LocalReadyBuffer buf;
    buf.reset(storage, 8);

    PTO2TaskSlotState items[4]{};
    // Push in order: 0, 1, 2, 3
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(buf.try_push(&items[i]));
    }

    // Pop order should be LIFO: 3, 2, 1, 0 (reverse of push)
    // Use pointer identity to verify ordering
    for (int i = 3; i >= 0; i--) {
        PTO2TaskSlotState *s = buf.pop();
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s, &items[i]) << "LocalReadyBuffer pops in LIFO order (priority reversed)";
    }

    // This means if tasks A, B, C, D become ready (in dependency order),
    // they are dispatched as D, C, B, A -- reverse of optimal order.
    EXPECT_EQ(buf.pop(), nullptr) << "Empty after draining";
}

// ---------------------------------------------------------------------------
// LocalReadyBuffer overflow: try_push returns false at capacity
// ---------------------------------------------------------------------------
TEST(LocalReadyBufferTest, OverflowBehavior) {
    PTO2TaskSlotState *storage[4]{};
    PTO2LocalReadyBuffer buf;
    buf.reset(storage, 4);

    PTO2TaskSlotState items[6]{};
    int pushed = 0;
    for (int i = 0; i < 6; i++) {
        if (buf.try_push(&items[i])) pushed++;
    }

    EXPECT_EQ(pushed, 4) << "Only 4 items fit in capacity-4 buffer";
    EXPECT_FALSE(buf.try_push(&items[5])) << "5th push fails";
}

// ---------------------------------------------------------------------------
// LocalReadyBuffer with nullptr backing: all pushes fail
// ---------------------------------------------------------------------------
TEST(LocalReadyBufferTest, NullBackingBuffer) {
    PTO2LocalReadyBuffer buf;
    buf.reset(nullptr, 0);

    PTO2TaskSlotState item{};
    EXPECT_FALSE(buf.try_push(&item)) << "Push fails with null backing";
    EXPECT_EQ(buf.pop(), nullptr) << "Pop returns null with null backing";
}

// ---------------------------------------------------------------------------
// LocalReadyBuffer reset clears state
// ---------------------------------------------------------------------------
TEST(LocalReadyBufferTest, ResetClearsState) {
    PTO2TaskSlotState *storage[8]{};
    PTO2LocalReadyBuffer buf;
    buf.reset(storage, 8);

    PTO2TaskSlotState item{};
    buf.try_push(&item);
    buf.try_push(&item);

    // After reset, buffer should behave as empty
    buf.reset(storage, 8);
    EXPECT_EQ(buf.pop(), nullptr) << "Buffer is empty after reset";

    // Should accept pushes again up to capacity
    for (int i = 0; i < 8; i++) {
        EXPECT_TRUE(buf.try_push(&item));
    }
    EXPECT_FALSE(buf.try_push(&item)) << "Full after pushing capacity items";
}

// =============================================================================
// SharedMemory edge cases
// =============================================================================

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-5: Zero window size
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, ZeroWindowSize) {
    uint64_t size = pto2_sm_calculate_size(0);
    // With window=0, only header is counted
    uint64_t header_size = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    EXPECT_EQ(size, header_size);

    PTO2SharedMemoryHandle *h = pto2_sm_create(0, 4096);
    if (h) {
        // All ring descriptors should point to the same location (after header)
        for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
            EXPECT_EQ(h->header->rings[r].task_descriptors, h->header->rings[r + 1].task_descriptors)
                << "Zero window: all rings' descriptor pointers collapse to same address";
        }
        pto2_sm_destroy(h);
    }
}

// ---------------------------------------------------------------------------
// Validate detects corrupted flow control
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, ValidateDetectsCorruption) {
    PTO2SharedMemoryHandle *h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(pto2_sm_validate(h));

    // Corrupt: set current_task_index to negative value
    h->header->rings[0].fc.current_task_index.store(-1);
    EXPECT_FALSE(pto2_sm_validate(h));

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Validate with null handle
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, ValidateNullHandle) { EXPECT_FALSE(pto2_sm_validate(nullptr)); }

// ---------------------------------------------------------------------------
// Create from undersized buffer
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, CreateFromUndersizedBuffer) {
    char buf[64]{};
    PTO2SharedMemoryHandle *h = pto2_sm_create_from_buffer(buf, 64, 256, 4096);
    EXPECT_EQ(h, nullptr) << "Undersized buffer should fail";
}

// ---------------------------------------------------------------------------
// Per-ring different window sizes via pto2_sm_calculate_size_per_ring
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, PerRingDifferentSizes) {
    uint64_t ws[PTO2_MAX_RING_DEPTH] = {128, 256, 512, 1024};
    uint64_t size = pto2_sm_calculate_size_per_ring(ws);

    // Size should be larger than uniform 128
    uint64_t uniform_size = pto2_sm_calculate_size(128);
    EXPECT_GT(size, uniform_size);
}

// ---------------------------------------------------------------------------
// Shared memory layout: descriptor and payload regions don't overlap
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, RegionsNonOverlapping) {
    PTO2SharedMemoryHandle *h = pto2_sm_create(64, 4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        uintptr_t desc_start = (uintptr_t)h->header->rings[r].task_descriptors;
        uintptr_t desc_end = desc_start + 64 * sizeof(PTO2TaskDescriptor);
        uintptr_t payload_start = (uintptr_t)h->header->rings[r].task_payloads;

        // Payloads should start at or after descriptors end
        EXPECT_GE(payload_start, desc_end) << "Ring " << r << ": payload region should not overlap descriptors";
    }

    // Adjacent rings should not overlap
    for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
        uintptr_t this_payload_end = (uintptr_t)h->header->rings[r].task_payloads + 64 * sizeof(PTO2TaskPayload);
        uintptr_t next_desc_start = (uintptr_t)h->header->rings[r + 1].task_descriptors;
        EXPECT_GE(next_desc_start, this_payload_end) << "Ring " << r << " and " << (r + 1) << " should not overlap";
    }

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Shared memory header alignment
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, HeaderAlignment) {
    PTO2SharedMemoryHandle *h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);

    uintptr_t header_addr = (uintptr_t)h->header;
    EXPECT_EQ(header_addr % PTO2_ALIGN_SIZE, 0u) << "Header must be cache-line aligned";

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Flow control init state
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, FlowControlInitState) {
    PTO2SharedMemoryHandle *h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto &fc = h->header->rings[r].fc;
        EXPECT_EQ(fc.current_task_index.load(), 0) << "Ring " << r << " current_task_index should init to 0";
        EXPECT_EQ(fc.last_task_alive.load(), 0) << "Ring " << r << " last_task_alive should init to 0";
    }

    pto2_sm_destroy(h);
}

// =============================================================================
// TaskState edge cases
// =============================================================================

// ---------------------------------------------------------------------------
// DC-14 (design contract): Non-profiling release_fanin skips task_state CAS.
//
// The non-profiling release_fanin_and_check_ready() intentionally does NOT
// CAS(PENDING -> READY).  Readiness is determined solely by fanin_refcount
// reaching fanin_count -- the atomic fetch_add guarantees exactly one thread
// sees the final count and pushes to the ready queue.  The profiling overload
// adds the CAS only to count atomic operations.  No consumer inspects
// task_state == READY for dispatch; it is metadata for profiling only.
//
// This test anchors the design: task_state stays PENDING after the
// non-profiling ready path, confirming the CAS is profiling-only.
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, NonProfilingReadyPath_TaskStateStaysPending) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIC;

    // Simulate non-profiling release_fanin_and_check_ready:
    int32_t new_refcount = slot.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
    bool ready = (new_refcount == slot.fanin_count);
    ASSERT_TRUE(ready) << "Task should be detected as ready via refcount";

    // task_state remains PENDING -- this is correct by design.
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING)
        << "Non-profiling path intentionally does not transition task_state to READY";
}

// ---------------------------------------------------------------------------
// EDGE-2: Simultaneous subtask completion -- verify exactly one completer
// Uses the current completed_subtasks counter model (not deprecated done_mask).
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, SimultaneousSubtaskCompletion) {
    constexpr int ROUNDS = 1000;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
        slot.total_required_subtasks = 2;  // 1 block * 2 active subtasks
        slot.completed_subtasks.store(0);
        std::atomic<int> completers{0};

        auto complete_subtask = [&]() {
            int16_t prev = slot.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
            if ((prev + 1) == slot.total_required_subtasks) {
                completers++;
            }
        };

        std::thread t1(complete_subtask);
        std::thread t2(complete_subtask);
        t1.join();
        t2.join();

        // Exactly ONE thread should see full completion
        EXPECT_EQ(completers.load(), 1) << "Round " << round << ": exactly 1 thread should trigger completion";
    }
}

// ---------------------------------------------------------------------------
// Double subtask completion (same core completes twice)
// With the counter model, a double-completion increments the counter twice,
// potentially reaching total_required_subtasks prematurely -- a real bug risk
// that the bitmask model was immune to (fetch_or is idempotent for same bit).
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, DoubleSubtaskCompletion) {
    PTO2TaskSlotState slot{};
    slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
    slot.total_required_subtasks = 2;
    slot.completed_subtasks.store(0);

    // Complete AIC subtask
    int16_t prev1 = slot.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
    bool first_complete = ((prev1 + 1) == slot.total_required_subtasks);
    EXPECT_FALSE(first_complete) << "Single completion doesn't complete the task";
    EXPECT_EQ(prev1, 0);

    // Complete AIC AGAIN (double-completion -- logic error)
    // With counter model, this incorrectly reaches total_required_subtasks
    int16_t prev2 = slot.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
    bool second_complete = ((prev2 + 1) == slot.total_required_subtasks);
    EXPECT_TRUE(second_complete) << "Counter model: double-completion of same core falsely triggers completion. "
                                    "Unlike the old bitmask model, the counter cannot detect duplicate completions.";
}

// ---------------------------------------------------------------------------
// Three subtasks: AIC + AIV0 + AIV1 (counter model)
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, ThreeSubtaskCompletion) {
    constexpr int ROUNDS = 500;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
        slot.total_required_subtasks = 3;  // 1 block * 3 active subtasks
        slot.completed_subtasks.store(0);
        std::atomic<int> completers{0};

        auto complete = [&]() {
            int16_t prev = slot.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
            if ((prev + 1) == slot.total_required_subtasks) {
                completers++;
            }
        };

        std::thread t1(complete);
        std::thread t2(complete);
        std::thread t3(complete);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(completers.load(), 1) << "Round " << round << ": exactly 1 of 3 threads triggers completion";
    }
}

// ---------------------------------------------------------------------------
// Fanout lock contention: two threads trying to lock the same task
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, FanoutLockContention) {
    PTO2TaskSlotState slot{};
    slot.fanout_lock.store(0);

    constexpr int N = 10000;
    std::atomic<int> acquired{0};

    auto lock_unlock = [&]() {
        for (int i = 0; i < N; i++) {
            // Spin-lock: CAS(0 -> 1)
            int32_t expected = 0;
            while (!slot.fanout_lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
                expected = 0;
            }
            acquired++;
            slot.fanout_lock.store(0, std::memory_order_release);
        }
    };

    std::thread t1(lock_unlock);
    std::thread t2(lock_unlock);
    t1.join();
    t2.join();

    EXPECT_EQ(acquired.load(), 2 * N);
}

// ---------------------------------------------------------------------------
// Fanin refcount: verify exactly-once ready detection
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, FaninExactlyOnceReady) {
    constexpr int ROUNDS = 1000;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.fanin_count = 3;
        slot.fanin_refcount.store(0);
        std::atomic<int> ready_detectors{0};

        auto release_fanin = [&]() {
            int32_t prev = slot.fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
            if (prev + 1 == slot.fanin_count) {
                ready_detectors++;
            }
        };

        std::thread t1(release_fanin);
        std::thread t2(release_fanin);
        std::thread t3(release_fanin);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(ready_detectors.load(), 1) << "Round " << round << ": exactly 1 thread detects task ready";
    }
}

// ---------------------------------------------------------------------------
// Fanout refcount: verify exactly-once CONSUMED detection
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, FanoutExactlyOnceConsumed) {
    constexpr int ROUNDS = 1000;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.fanout_count = 4;  // 1 scope + 3 consumers
        slot.fanout_refcount.store(0);
        slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
        std::atomic<int> consumed_detectors{0};

        auto release_fanout = [&]() {
            int32_t prev = slot.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
            if (prev + 1 == slot.fanout_count) {
                // Only one thread should see this
                PTO2TaskState expected = PTO2_TASK_COMPLETED;
                if (slot.task_state.compare_exchange_strong(
                        expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
                    )) {
                    consumed_detectors++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < 4; i++) {
            threads.emplace_back(release_fanout);
        }
        for (auto &t : threads)
            t.join();

        EXPECT_EQ(consumed_detectors.load(), 1) << "Round " << round << ": exactly 1 thread detects CONSUMED";
        EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
    }
}

// ---------------------------------------------------------------------------
// Task state machine: full lifecycle PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, FullLifecycle) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // PENDING -> READY (when all fanin satisfied)
    PTO2TaskState expected = PTO2_TASK_PENDING;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_READY);

    // READY -> RUNNING (when dispatched to core)
    expected = PTO2_TASK_READY;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_RUNNING);

    // RUNNING -> COMPLETED (when subtasks done)
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_COMPLETED);

    // COMPLETED -> CONSUMED (when all fanout released)
    expected = PTO2_TASK_COMPLETED;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// ---------------------------------------------------------------------------
// Task state: invalid transition PENDING -> COMPLETED (skip READY/RUNNING)
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, InvalidTransition) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // Try to CAS COMPLETED when state is actually PENDING -- should fail
    PTO2TaskState expected = PTO2_TASK_COMPLETED;
    EXPECT_FALSE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED))
        << "Cannot transition from non-COMPLETED to CONSUMED";
    EXPECT_EQ(expected, PTO2_TASK_PENDING) << "CAS returns actual state";
}

// ---------------------------------------------------------------------------
// check_and_handle_consumed race: two threads calling simultaneously
// Only one should succeed in the CAS(COMPLETED -> CONSUMED)
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, ConsumedRace) {
    constexpr int ROUNDS = 1000;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_relaxed);
        slot.fanout_count = 2;
        slot.fanout_refcount.store(2, std::memory_order_relaxed);  // All released
        std::atomic<int> consumed{0};

        auto try_consume = [&]() {
            if (slot.fanout_refcount.load() != slot.fanout_count) return;
            PTO2TaskState exp = PTO2_TASK_COMPLETED;
            if (slot.task_state.compare_exchange_strong(
                    exp, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                consumed++;
            }
        };

        std::thread t1(try_consume);
        std::thread t2(try_consume);
        t1.join();
        t2.join();

        EXPECT_EQ(consumed.load(), 1) << "Round " << round << ": exactly 1 thread succeeds in CONSUMED CAS";
    }
}
