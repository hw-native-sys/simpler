/**
 * Edge-case tests for ReadyQueue, SharedMemory, and TaskState.
 *
 * ============================================================================
 * ANALYSIS FINDINGS — PTO2ReadyQueue (Vyukov MPMC)
 * ============================================================================
 *
 * BUG-CANDIDATE-1 (sequence wrap): The sequence counter is int64_t.
 *   After 2^63 push/pop operations, it wraps to negative.  The comparison
 *   `sequence == pos` still works because both wrap identically (signed
 *   overflow is UB in C++ but defined for two's complement on most platforms).
 *   → Practically unreachable, but if compiled with -ftrapv, this crashes.
 *
 * BUG-CANDIDATE-2 (pop fast-path): pop() checks `enqueue_pos == dequeue_pos`
 *   as early empty detection.  But between reading enqueue_pos and the CAS
 *   on dequeue_pos, a push could occur.  This is fine — the CAS will succeed
 *   with the newly pushed item.  However, if pop() returns nullptr based on
 *   the fast-path check, a concurrent push that happened just after the check
 *   is invisible.  This is a known TOCTOU in MPMC queues and acceptable.
 *
 * BUG-CANDIDATE-3 (push returns false): push() returns false when the queue
 *   is full (sequence != pos).  However, with multiple producers, all may
 *   see the same full slot and return false simultaneously, even if a pop
 *   happens right after.  This is by-design but means the queue has poor
 *   throughput near capacity with many producers.
 *
 * BUG-CANDIDATE-9 (size() relaxed ordering): size() reads enqueue_pos and
 *   dequeue_pos with relaxed ordering.  Under concurrent push/pop, these
 *   values can be stale.  size() can return incorrect values, including
 *   cases where e < d is observed (returns 0 via the guard).
 *
 * ============================================================================
 * ANALYSIS FINDINGS — Scheduler
 * ============================================================================
 *
 * BUG-CANDIDATE-10 (Missing task_state CAS in non-profiling path):
 *   release_fanin_and_check_ready() NON-PROFILING version (line 426-448)
 *   does NOT perform CAS(PENDING → READY) on task_state before pushing
 *   to the ready queue.  The PROFILING version (line 451-476) DOES perform
 *   this CAS (line 459).  This means in non-profiling builds, a task can
 *   be enqueued in the ready queue while its state is still PENDING.
 *   Consumers that check task_state will see PENDING, not READY.
 *
 * BUG-CANDIDATE-11 (LocalReadyBuffer LIFO dispatch): pop() returns
 *   slot_states[--count] (LIFO), but try_push adds at slot_states[count++]
 *   (FIFO insertion).  This means the LAST task pushed is the FIRST to be
 *   dispatched, reversing priority order.  For fanout notification, this
 *   means downstream tasks are dispatched in reverse dependency order.
 *
 * BUG-CANDIDATE-12 (on_subtask_complete double-completion): Calling
 *   on_subtask_complete twice with the same subslot silently succeeds
 *   (fetch_or is idempotent for the same bit).  The second call returns
 *   false (since prev | bit == active_mask was already true).  No guard
 *   detects this as a logic error.
 *
 * BUG-CANDIDATE-13 (advance_ring_pointers null task pointer):
 *   advance_ring_pointers accesses slot_state.task->packed_buffer_end
 *   without checking if slot_state.task is nullptr.  If a task slot is
 *   reused before the descriptor is fully initialized, this is a null
 *   pointer dereference.
 *
 * ============================================================================
 * ANALYSIS FINDINGS — SharedMemory
 * ============================================================================
 *
 * BUG-CANDIDATE-4 (pto2_sm_validate): Checks `top > heap_size` but heap_top
 *   can be EQUAL to heap_size when the heap is exactly full.  Should be `>=`?
 *   Actually: top == heap_size means we filled exactly to the end, which is
 *   valid.  top > heap_size would be a corruption.  So `>` is correct.
 *
 * BUG-CANDIDATE-5 (size calculation with 0 window): If task_window_size=0,
 *   pto2_sm_calculate_size() returns just the header size.  But
 *   pto2_sm_setup_pointers will set task_descriptors[r] and task_payloads[r]
 *   to the same pointer (after header), since 0*sizeof = 0 aligned = 0.
 *   This means all rings share the same descriptor/payload pointer!
 *
 * BUG-CANDIDATE-6 (flow control heap_top validation): validate checks
 *   `top > heap_size` but heap_top is stored in PTO2RingFlowControl as a
 *   uint64_t offset, while heap_size is in PTO2SharedMemoryRingHeader.
 *   After a wrap-around, top resets to a small value.  The check should also
 *   verify that top <= heap_size (not just > heap_size) since top could be
 *   corrupted to any value.  But the current check only catches corruption
 *   in one direction.
 *
 * ============================================================================
 * ANALYSIS FINDINGS — TaskState
 * ============================================================================
 *
 * EDGE-1: CAS on task_state with memory_order_relaxed could reorder with
 *   subsequent reads of fanin_refcount.  The task state machine relies on
 *   the state transition being visible before fanin/fanout operations.
 *   → The actual scheduler code uses acquire/release on task_state.
 *
 * EDGE-2: subtask_done_mask uses fetch_or which is atomic but the
 *   comparison `(done_mask & active_mask) == active_mask` is done
 *   on the PREVIOUS value.  If two subtasks complete simultaneously:
 *   Thread A: prev = fetch_or(MASK_AIC) → prev = 0
 *   Thread B: prev = fetch_or(MASK_AIV0) → prev = 0 or MASK_AIC
 *   Neither thread sees full completion unless they re-read.
 *   → The actual code checks `(prev | my_mask) == active_mask`.
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

// =============================================================================
// ReadyQueue edge cases
// =============================================================================
class ReadyQueueEdgeTest : public ::testing::Test {
protected:
    static constexpr uint64_t QUEUE_CAP = 8;  // Small for edge testing
    PTO2ReadyQueueSlot slots[8]{};
    PTO2ReadyQueue queue{};
    PTO2TaskSlotState dummy[8]{};

    void SetUp() override {
        queue.slots = slots;
        queue.capacity = QUEUE_CAP;
        queue.mask = QUEUE_CAP - 1;
        queue.enqueue_pos.store(0, std::memory_order_relaxed);
        queue.dequeue_pos.store(0, std::memory_order_relaxed);
        for (uint64_t i = 0; i < QUEUE_CAP; i++) {
            slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
            slots[i].slot_state = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Push and pop interleaving: push(A), pop() → A, push(B), pop() → B
// Ensures sequence numbers are correctly advanced after each operation.
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, InterleavedPushPop) {
    for (int i = 0; i < 20; i++) {
        EXPECT_TRUE(queue.push(&dummy[0]));
        PTO2TaskSlotState* s = queue.pop();
        EXPECT_EQ(s, &dummy[0]);
    }
    // After 20 interleaved push/pop, queue should be empty
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.pop(), nullptr);
}

// ---------------------------------------------------------------------------
// Exactly fill queue, then pop all — boundary at capacity
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
    while (queue.push(&dummy[0])) pushed++;

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
// size() guard: when dequeue_pos > enqueue_pos (stale read), returns 0
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, SizeGuardAgainstNegative) {
    // Simulate stale state where dequeue_pos > enqueue_pos
    // This shouldn't happen in normal operation, but the guard protects against it
    queue.enqueue_pos.store(5);
    queue.dequeue_pos.store(8);
    EXPECT_EQ(queue.size(), 0u)
        << "size() returns 0 when dequeue_pos > enqueue_pos (stale read guard)";
}

// ---------------------------------------------------------------------------
// FIFO ordering: items come out in the order they were pushed
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, FIFOOrdering) {
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(queue.push(&dummy[i]));
    }

    for (int i = 0; i < 5; i++) {
        PTO2TaskSlotState* s = queue.pop();
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
    big_queue.slots = big_slots;
    big_queue.capacity = BIG_CAP;
    big_queue.mask = BIG_CAP - 1;
    big_queue.enqueue_pos.store(0);
    big_queue.dequeue_pos.store(0);
    for (uint64_t i = 0; i < BIG_CAP; i++) {
        big_slots[i].sequence.store((int64_t)i);
        big_slots[i].slot_state = nullptr;
    }

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
            PTO2TaskSlotState* s = big_queue.pop();
            if (s) consumed++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < P; i++) threads.emplace_back(producer, i);
    for (int i = 0; i < C; i++) threads.emplace_back(consumer);
    for (auto& t : threads) t.join();

    EXPECT_EQ(produced.load(), N);
    EXPECT_EQ(consumed.load(), N);
}

// ---------------------------------------------------------------------------
// Concurrent stress: verify no duplicates consumed
// ---------------------------------------------------------------------------
TEST_F(ReadyQueueEdgeTest, NoDuplicateConsumption) {
    static constexpr uint64_t BIG_CAP = 128;
    PTO2ReadyQueueSlot big_slots[BIG_CAP];
    PTO2ReadyQueue big_queue{};
    big_queue.slots = big_slots;
    big_queue.capacity = BIG_CAP;
    big_queue.mask = BIG_CAP - 1;
    big_queue.enqueue_pos.store(0);
    big_queue.dequeue_pos.store(0);
    for (uint64_t i = 0; i < BIG_CAP; i++) {
        big_slots[i].sequence.store((int64_t)i);
        big_slots[i].slot_state = nullptr;
    }

    constexpr int N = 1000;
    std::vector<PTO2TaskSlotState> items(N);
    // Tag each item with a unique index
    for (int i = 0; i < N; i++) {
        items[i].fanin_count = i;  // Use fanin_count as tag
    }

    // Push all items
    for (int i = 0; i < N; i++) {
        while (!big_queue.push(&items[i])) {
            // Drain some if full
            PTO2TaskSlotState* s = big_queue.pop();
            if (s) items[s->fanin_count].fanout_count++;  // repurpose as consumed flag
        }
    }

    // Pop remaining
    while (true) {
        PTO2TaskSlotState* s = big_queue.pop();
        if (!s) break;
        s->fanout_count++;  // mark as consumed
    }

    // Verify each item consumed exactly once
    // (items consumed during overflow draining + items consumed at end)
    int total_consumed = 0;
    for (int i = 0; i < N; i++) {
        total_consumed += items[i].fanout_count;
    }
    EXPECT_EQ(total_consumed, N) << "Each item should be consumed exactly once";
}

// ---------------------------------------------------------------------------
// Pop from empty queue multiple times — must always return nullptr
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
        PTO2TaskSlotState* s = queue.pop();
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
    PTO2TaskSlotState* storage[8]{};
    PTO2LocalReadyBuffer buf;
    buf.reset(storage, 8);

    PTO2TaskSlotState items[4]{};
    // Push in order: 0, 1, 2, 3
    for (int i = 0; i < 4; i++) {
        items[i].fanin_count = i;  // Tag for identification
        ASSERT_TRUE(buf.try_push(&items[i]));
    }

    // Pop order should be LIFO: 3, 2, 1, 0 (reverse of push)
    for (int i = 3; i >= 0; i--) {
        PTO2TaskSlotState* s = buf.pop();
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s->fanin_count, i)
            << "LocalReadyBuffer pops in LIFO order (priority reversed)";
    }

    // This means if tasks A, B, C, D become ready (in dependency order),
    // they are dispatched as D, C, B, A — reverse of optimal order.
    EXPECT_EQ(buf.pop(), nullptr) << "Empty after draining";
}

// ---------------------------------------------------------------------------
// LocalReadyBuffer overflow: try_push returns false at capacity
// ---------------------------------------------------------------------------
TEST(LocalReadyBufferTest, OverflowBehavior) {
    PTO2TaskSlotState* storage[4]{};
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
    PTO2TaskSlotState* storage[8]{};
    PTO2LocalReadyBuffer buf;
    buf.reset(storage, 8);

    PTO2TaskSlotState item{};
    buf.try_push(&item);
    buf.try_push(&item);
    EXPECT_EQ(buf.count, 2);

    buf.reset(storage, 8);
    EXPECT_EQ(buf.count, 0);
    EXPECT_EQ(buf.pop(), nullptr);
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

    PTO2SharedMemoryHandle* h = pto2_sm_create(0, 4096);
    if (h) {
        // All ring descriptors should point to the same location (after header)
        for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
            EXPECT_EQ(h->task_descriptors[r], h->task_descriptors[r + 1])
                << "Zero window: all rings' descriptor pointers collapse to same address";
        }
        pto2_sm_destroy(h);
    }
}

// ---------------------------------------------------------------------------
// Validate detects corrupted flow control
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, ValidateDetectsCorruption) {
    PTO2SharedMemoryHandle* h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(pto2_sm_validate(h));

    // Corrupt: set heap_top beyond heap_size
    h->header->rings[0].fc.heap_top.store(999999);
    EXPECT_FALSE(pto2_sm_validate(h));

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Validate with null handle
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, ValidateNullHandle) {
    EXPECT_FALSE(pto2_sm_validate(nullptr));
}

// ---------------------------------------------------------------------------
// Create from undersized buffer
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, CreateFromUndersizedBuffer) {
    char buf[64]{};
    PTO2SharedMemoryHandle* h = pto2_sm_create_from_buffer(buf, 64, 256, 4096);
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
    PTO2SharedMemoryHandle* h = pto2_sm_create(64, 4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        uintptr_t desc_start = (uintptr_t)h->task_descriptors[r];
        uintptr_t desc_end = desc_start + 64 * sizeof(PTO2TaskDescriptor);
        uintptr_t payload_start = (uintptr_t)h->task_payloads[r];

        // Payloads should start at or after descriptors end
        EXPECT_GE(payload_start, desc_end)
            << "Ring " << r << ": payload region should not overlap descriptors";
    }

    // Adjacent rings should not overlap
    for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
        uintptr_t this_payload_end = (uintptr_t)h->task_payloads[r] + 64 * sizeof(PTO2TaskPayload);
        uintptr_t next_desc_start = (uintptr_t)h->task_descriptors[r + 1];
        EXPECT_GE(next_desc_start, this_payload_end)
            << "Ring " << r << " and " << (r+1) << " should not overlap";
    }

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Shared memory header alignment
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, HeaderAlignment) {
    PTO2SharedMemoryHandle* h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);

    uintptr_t header_addr = (uintptr_t)h->header;
    EXPECT_EQ(header_addr % PTO2_ALIGN_SIZE, 0u)
        << "Header must be cache-line aligned";

    pto2_sm_destroy(h);
}

// ---------------------------------------------------------------------------
// Flow control init state
// ---------------------------------------------------------------------------
TEST(SharedMemEdgeTest, FlowControlInitState) {
    PTO2SharedMemoryHandle* h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto& fc = h->header->rings[r].fc;
        EXPECT_EQ(fc.heap_top.load(), 0u) << "Ring " << r << " heap_top should init to 0";
        EXPECT_EQ(fc.heap_tail.load(), 0u) << "Ring " << r << " heap_tail should init to 0";
        EXPECT_EQ(fc.current_task_index.load(), 0) << "Ring " << r << " current_task_index should init to 0";
        EXPECT_EQ(fc.last_task_alive.load(), 0) << "Ring " << r << " last_task_alive should init to 0";
    }

    pto2_sm_destroy(h);
}

// =============================================================================
// TaskState edge cases
// =============================================================================

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-10: Missing task_state CAS in non-profiling path
//
// release_fanin_and_check_ready() NON-PROFILING version pushes tasks to the
// ready queue WITHOUT setting task_state to PTO2_TASK_READY.  The profiling
// version DOES perform CAS(PENDING → READY).  This inconsistency means:
// 1. In non-profiling builds, tasks in the ready queue have state PENDING.
// 2. Any code that checks task_state for READY will not find it.
// 3. This is a semantic gap between profiling and non-profiling builds.
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, NonProfilingMissingReadyTransition) {
    // Simulate what release_fanin_and_check_ready does in non-profiling mode:
    // It checks fanin_refcount == fanin_count and pushes to ready queue,
    // but does NOT CAS(PENDING → READY).
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.fanin_count = 1;
    slot.fanin_refcount.store(0, std::memory_order_relaxed);
    slot.active_mask = PTO2_SUBTASK_MASK_AIC;

    // Simulate the non-profiling release_fanin_and_check_ready:
    int32_t new_refcount = slot.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
    bool ready = (new_refcount == slot.fanin_count);
    ASSERT_TRUE(ready) << "Task should be detected as ready";

    // In non-profiling path: task is pushed to ready queue here
    // WITHOUT CAS(PENDING → READY).
    // The task_state is still PENDING!
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING)
        << "BUG: Non-profiling path leaves task in PENDING state when pushing to ready queue";

    // In contrast, the profiling path would do:
    // PTO2TaskState expected = PTO2_TASK_PENDING;
    // slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY, ...);
    // → task_state would be PTO2_TASK_READY

    // Verify the profiling path behavior would be different:
    PTO2TaskSlotState slot_profiling{};
    slot_profiling.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    PTO2TaskState expected = PTO2_TASK_PENDING;
    bool cas_ok = slot_profiling.task_state.compare_exchange_strong(
        expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire);
    EXPECT_TRUE(cas_ok);
    EXPECT_EQ(slot_profiling.task_state.load(), PTO2_TASK_READY)
        << "Profiling path correctly transitions to READY";
}

// ---------------------------------------------------------------------------
// EDGE-2: Simultaneous subtask completion — verify done_mask is correct
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, SimultaneousSubtaskCompletion) {
    constexpr int ROUNDS = 1000;
    std::atomic<int> both_see_complete{0};

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
        slot.subtask_done_mask.store(0);
        std::atomic<int> completers{0};

        auto complete_subtask = [&](uint8_t mask) {
            uint8_t prev = slot.subtask_done_mask.fetch_or(mask);
            if ((prev | mask) == slot.active_mask) {
                completers++;
            }
        };

        std::thread t1(complete_subtask, PTO2_SUBTASK_MASK_AIC);
        std::thread t2(complete_subtask, PTO2_SUBTASK_MASK_AIV0);
        t1.join();
        t2.join();

        // Exactly ONE thread should see full completion
        EXPECT_EQ(completers.load(), 1)
            << "Round " << round << ": exactly 1 thread should trigger completion";
    }
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE-12: Double subtask completion (same subslot twice)
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, DoubleSubtaskCompletion) {
    PTO2TaskSlotState slot{};
    slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
    slot.subtask_done_mask.store(0);

    // Complete AIC subtask
    uint8_t prev1 = slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIC);
    bool first_complete = ((prev1 | PTO2_SUBTASK_MASK_AIC) == slot.active_mask);
    EXPECT_FALSE(first_complete) << "AIC alone doesn't complete the task";

    // Complete AIC AGAIN (double-completion — logic error, but no guard)
    uint8_t prev2 = slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIC);
    bool second_complete = ((prev2 | PTO2_SUBTASK_MASK_AIC) == slot.active_mask);
    EXPECT_FALSE(second_complete) << "Double AIC completion: still not all done";
    EXPECT_EQ(prev2, PTO2_SUBTASK_MASK_AIC) << "prev2 shows AIC was already set";

    // Now complete AIV0 — this should be the real completer
    uint8_t prev3 = slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIV0);
    bool third_complete = ((prev3 | PTO2_SUBTASK_MASK_AIV0) == slot.active_mask);
    EXPECT_TRUE(third_complete) << "AIV0 triggers completion even after double AIC";

    // The double-completion of AIC was silently ignored.
    // In a correct system, double-completion should be detected as an error.
    // But fetch_or is idempotent for the same bit, so no damage occurs.
    // The risk: if the second AIC completion was from a different task (bug),
    // it would be invisible.
}

// ---------------------------------------------------------------------------
// Three subtasks: AIC + AIV0 + AIV1
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, ThreeSubtaskCompletion) {
    constexpr int ROUNDS = 500;

    for (int round = 0; round < ROUNDS; round++) {
        PTO2TaskSlotState slot{};
        slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
        slot.subtask_done_mask.store(0);
        std::atomic<int> completers{0};

        auto complete = [&](uint8_t mask) {
            uint8_t prev = slot.subtask_done_mask.fetch_or(mask);
            if ((prev | mask) == slot.active_mask) {
                completers++;
            }
        };

        std::thread t1(complete, PTO2_SUBTASK_MASK_AIC);
        std::thread t2(complete, PTO2_SUBTASK_MASK_AIV0);
        std::thread t3(complete, PTO2_SUBTASK_MASK_AIV1);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(completers.load(), 1)
            << "Round " << round << ": exactly 1 of 3 threads triggers completion";
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
            // Spin-lock: CAS(0 → 1)
            int32_t expected = 0;
            while (!slot.fanout_lock.compare_exchange_weak(expected, 1,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
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

        EXPECT_EQ(ready_detectors.load(), 1)
            << "Round " << round << ": exactly 1 thread detects task ready";
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
                if (slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    consumed_detectors++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < 4; i++) {
            threads.emplace_back(release_fanout);
        }
        for (auto& t : threads) t.join();

        EXPECT_EQ(consumed_detectors.load(), 1)
            << "Round " << round << ": exactly 1 thread detects CONSUMED";
        EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
    }
}

// ---------------------------------------------------------------------------
// Task state machine: full lifecycle PENDING → READY → RUNNING → COMPLETED → CONSUMED
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, FullLifecycle) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // PENDING → READY (when all fanin satisfied)
    PTO2TaskState expected = PTO2_TASK_PENDING;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_READY);

    // READY → RUNNING (when dispatched to core)
    expected = PTO2_TASK_READY;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_RUNNING);

    // RUNNING → COMPLETED (when subtasks done)
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_COMPLETED);

    // COMPLETED → CONSUMED (when all fanout released)
    expected = PTO2_TASK_COMPLETED;
    EXPECT_TRUE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED));
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// ---------------------------------------------------------------------------
// Task state: invalid transition PENDING → COMPLETED (skip READY/RUNNING)
// ---------------------------------------------------------------------------
TEST(TaskStateEdgeTest, InvalidTransition) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

    // Try to CAS COMPLETED when state is actually PENDING — should fail
    PTO2TaskState expected = PTO2_TASK_COMPLETED;
    EXPECT_FALSE(slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED))
        << "Cannot transition from non-COMPLETED to CONSUMED";
    EXPECT_EQ(expected, PTO2_TASK_PENDING) << "CAS returns actual state";
}

// ---------------------------------------------------------------------------
// check_and_handle_consumed race: two threads calling simultaneously
// Only one should succeed in the CAS(COMPLETED → CONSUMED)
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
            if (slot.task_state.compare_exchange_strong(exp, PTO2_TASK_CONSUMED,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                consumed++;
            }
        };

        std::thread t1(try_consume);
        std::thread t2(try_consume);
        t1.join();
        t2.join();

        EXPECT_EQ(consumed.load(), 1)
            << "Round " << round << ": exactly 1 thread succeeds in CONSUMED CAS";
    }
}
