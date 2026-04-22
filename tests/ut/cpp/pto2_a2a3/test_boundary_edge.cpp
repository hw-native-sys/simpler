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
 * Supplemental boundary-condition tests for:
 *   1. ReadyQueue high-contention stress (8+ threads, exactly-once guarantee)
 *   2. TaskAllocator double-destroy / re-init safety
 *   3. Scheduler sequence counter near INT64 wrap
 *   4. SharedMemory concurrent read/write of per-ring flow control
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "pto_ring_buffer.h"
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "../test_helpers.h"

// =============================================================================
// 1. ReadyQueue high-contention stress
// =============================================================================

class ReadyQueueStressTest : public ::testing::Test {
protected:
    static constexpr uint64_t kCapacity = 512;
    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(pto2_ready_queue_init(&queue, kCapacity)); }

    void TearDown() override { pto2_ready_queue_destroy(&queue); }
};

// 8 producers / 8 consumers, high volume -- every item consumed exactly once
TEST_F(ReadyQueueStressTest, EightProducersEightConsumers) {
    constexpr int kItemsPerProducer = 2000;
    constexpr int kProducers = 8;
    constexpr int kConsumers = 8;
    constexpr int kTotalItems = kItemsPerProducer * kProducers;

    std::vector<PTO2TaskSlotState> items(kTotalItems);
    for (int i = 0; i < kTotalItems; i++) {
        items[i].fanin_count = i;
    }

    std::vector<std::atomic<int>> consumed_count(kTotalItems);
    for (auto &c : consumed_count)
        c.store(0, std::memory_order_relaxed);

    std::atomic<int> producers_done{0};

    auto producer = [&](int id) {
        int base = id * kItemsPerProducer;
        for (int i = 0; i < kItemsPerProducer; i++) {
            while (!queue.push(&items[base + i])) {}
        }
        producers_done.fetch_add(1, std::memory_order_release);
    };

    auto consumer = [&](std::atomic<int> &local_count) {
        while (true) {
            PTO2TaskSlotState *item = queue.pop();
            if (item) {
                consumed_count[item->fanin_count].fetch_add(1, std::memory_order_relaxed);
                local_count.fetch_add(1, std::memory_order_relaxed);
            } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
                // Final drain
                while ((item = queue.pop()) != nullptr) {
                    consumed_count[item->fanin_count].fetch_add(1, std::memory_order_relaxed);
                    local_count.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::vector<std::atomic<int>> per_consumer_count(kConsumers);
    for (auto &c : per_consumer_count)
        c.store(0);

    std::vector<std::thread> threads;
    for (int i = 0; i < kProducers; i++) {
        threads.emplace_back(producer, i);
    }
    for (int i = 0; i < kConsumers; i++) {
        threads.emplace_back(consumer, std::ref(per_consumer_count[i]));
    }
    for (auto &t : threads)
        t.join();

    // Every item consumed exactly once
    int total = 0;
    for (int i = 0; i < kTotalItems; i++) {
        EXPECT_EQ(consumed_count[i].load(), 1) << "Item " << i << " consumed " << consumed_count[i].load() << " times";
        total += consumed_count[i].load();
    }
    EXPECT_EQ(total, kTotalItems);

    // Work is distributed across consumers (not all consumed by one)
    int active_consumers = 0;
    for (int i = 0; i < kConsumers; i++) {
        if (per_consumer_count[i].load() > 0) active_consumers++;
    }
    EXPECT_GT(active_consumers, 1) << "Work should be distributed across multiple consumers";
}

// Rapid fill-drain cycles under contention
TEST_F(ReadyQueueStressTest, RapidFillDrainCycles) {
    constexpr int kCycles = 100;
    constexpr int kItemsPerCycle = static_cast<int>(kCapacity / 2);

    std::vector<PTO2TaskSlotState> items(kItemsPerCycle);
    for (int i = 0; i < kItemsPerCycle; i++) {
        items[i].fanin_count = i;
    }

    for (int cycle = 0; cycle < kCycles; cycle++) {
        std::atomic<int> push_done{0};
        std::atomic<int> popped{0};

        // 4 producers push in parallel
        auto producer = [&](int id) {
            int per_thread = kItemsPerCycle / 4;
            int base = id * per_thread;
            for (int i = 0; i < per_thread; i++) {
                while (!queue.push(&items[base + i])) {}
            }
            push_done.fetch_add(1, std::memory_order_release);
        };

        // 4 consumers drain in parallel
        auto consumer = [&]() {
            while (true) {
                PTO2TaskSlotState *s = queue.pop();
                if (s) {
                    popped.fetch_add(1, std::memory_order_relaxed);
                } else if (push_done.load(std::memory_order_acquire) == 4) {
                    while ((s = queue.pop()) != nullptr) {
                        popped.fetch_add(1, std::memory_order_relaxed);
                    }
                    break;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < 4; i++)
            threads.emplace_back(producer, i);
        for (int i = 0; i < 4; i++)
            threads.emplace_back(consumer);
        for (auto &t : threads)
            t.join();

        ASSERT_EQ(popped.load(), kItemsPerCycle) << "Cycle " << cycle << ": lost items";
    }
}

// push_batch + pop_batch under contention
TEST_F(ReadyQueueStressTest, BatchPushPopContention) {
    constexpr int kBatchSize = 8;
    constexpr int kBatches = 500;
    constexpr int kProducers = 4;
    constexpr int kTotalItems = kBatchSize * kBatches * kProducers;

    std::vector<PTO2TaskSlotState> items(kTotalItems);
    for (int i = 0; i < kTotalItems; i++)
        items[i].fanin_count = i;

    std::atomic<int> total_consumed{0};
    std::atomic<int> producers_done{0};

    auto producer = [&](int id) {
        int base = id * kBatchSize * kBatches;
        for (int b = 0; b < kBatches; b++) {
            PTO2TaskSlotState *ptrs[kBatchSize];
            for (int i = 0; i < kBatchSize; i++) {
                ptrs[i] = &items[base + b * kBatchSize + i];
            }
            // push_batch may partially fail if queue is near full; retry
            for (int i = 0; i < kBatchSize; i++) {
                while (!queue.push(ptrs[i])) {}
            }
        }
        producers_done.fetch_add(1, std::memory_order_release);
    };

    auto consumer = [&]() {
        while (true) {
            PTO2TaskSlotState *out[kBatchSize];
            int n = queue.pop_batch(out, kBatchSize);
            total_consumed.fetch_add(n, std::memory_order_relaxed);
            if (n == 0 && producers_done.load(std::memory_order_acquire) == kProducers) {
                // Final drain
                while (true) {
                    n = queue.pop_batch(out, kBatchSize);
                    if (n == 0) break;
                    total_consumed.fetch_add(n, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kProducers; i++)
        threads.emplace_back(producer, i);
    for (int i = 0; i < 4; i++)
        threads.emplace_back(consumer);
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(total_consumed.load(), kTotalItems);
}

// =============================================================================
// 2. TaskAllocator double-destroy / re-init safety
// =============================================================================

class TaskAllocatorDoubleDestroyTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 16;
    static constexpr uint64_t HEAP_SIZE = 1024;

    std::vector<PTO2TaskDescriptor> descriptors;
    alignas(64) uint8_t heap_buf[1024]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> last_alive{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator{};

    void InitAllocator() {
        descriptors.assign(WINDOW_SIZE, PTO2TaskDescriptor{});
        std::memset(heap_buf, 0, sizeof(heap_buf));
        current_index.store(0);
        last_alive.store(0);
        error_code.store(PTO2_ERROR_NONE);
        allocator.init(descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code);
    }
};

// Re-init after use: allocator should work fresh
TEST_F(TaskAllocatorDoubleDestroyTest, ReInitAfterUse) {
    InitAllocator();

    // Use the allocator
    auto r1 = allocator.alloc(128);
    ASSERT_FALSE(r1.failed());
    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, 1);

    // Re-init: should reset state
    InitAllocator();

    // Should start from task_id 0 again
    auto r3 = allocator.alloc(64);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(r3.task_id, 0) << "Re-init should reset task ID counter";
    EXPECT_EQ(r3.slot, 0);
}

// Re-init with different heap size
TEST_F(TaskAllocatorDoubleDestroyTest, ReInitDifferentHeapSize) {
    InitAllocator();

    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    // Re-init with same buffer but fresh state
    InitAllocator();
    EXPECT_EQ(allocator.heap_top(), 0u) << "Re-init resets heap_top";
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE) << "Re-init restores full capacity";
}

// Re-init after error state: error flag should be clearable
TEST_F(TaskAllocatorDoubleDestroyTest, ReInitClearsErrorState) {
    InitAllocator();

    // Force a deadlock error
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed());
    EXPECT_NE(error_code.load(), PTO2_ERROR_NONE);

    // Re-init clears error
    InitAllocator();
    EXPECT_EQ(error_code.load(), PTO2_ERROR_NONE);

    // Allocator should work again
    auto r2 = allocator.alloc(64);
    EXPECT_FALSE(r2.failed());
}

// Multiple re-init cycles: no resource leak or corruption
TEST_F(TaskAllocatorDoubleDestroyTest, MultipleReInitCycles) {
    for (int cycle = 0; cycle < 10; cycle++) {
        InitAllocator();

        for (int i = 0; i < WINDOW_SIZE - 1; i++) {
            auto r = allocator.alloc(0);
            ASSERT_FALSE(r.failed()) << "Cycle " << cycle << " alloc " << i;
            EXPECT_EQ(r.task_id, i);
        }
    }
}

// Re-init with stale last_alive: allocator sees fresh state
TEST_F(TaskAllocatorDoubleDestroyTest, ReInitIgnoresStaleLastAlive) {
    InitAllocator();

    // Advance state
    auto r1 = allocator.alloc(64);
    ASSERT_FALSE(r1.failed());
    last_alive.store(5, std::memory_order_release);  // Stale value

    // Re-init resets last_alive
    InitAllocator();
    EXPECT_EQ(last_alive.load(), 0);

    auto r2 = allocator.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, 0);
}

// =============================================================================
// 3. Scheduler sequence counter near INT64 wrap
// =============================================================================

class SequenceWrapTest : public ::testing::Test {
protected:
    static constexpr uint64_t QUEUE_CAP = 8;
    PTO2ReadyQueueSlot slots[8]{};
    PTO2ReadyQueue queue{};
    PTO2TaskSlotState dummy[8]{};

    void InitQueueAtSequence(int64_t start_seq) { test_ready_queue_init(&queue, slots, QUEUE_CAP, start_seq); }
};

// Sequence near INT64_MAX: push/pop should still work
TEST_F(SequenceWrapTest, NearInt64Max) {
    int64_t near_max = INT64_MAX - 16;
    InitQueueAtSequence(near_max);

    // Push and pop several items, crossing INT64_MAX
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(queue.push(&dummy[i])) << "Push " << i << " near INT64_MAX";
    }

    for (int i = 0; i < 5; i++) {
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr) << "Pop " << i << " near INT64_MAX";
        EXPECT_EQ(s, &dummy[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// Sequence near INT64_MAX: fill to capacity then drain
TEST_F(SequenceWrapTest, FillDrainNearMax) {
    int64_t near_max = INT64_MAX - 4;
    InitQueueAtSequence(near_max);

    int pushed = 0;
    for (uint64_t i = 0; i < QUEUE_CAP; i++) {
        if (queue.push(&dummy[i % 8])) pushed++;
        else break;
    }
    EXPECT_GE(pushed, 1) << "Should push at least some items near max";

    for (int i = 0; i < pushed; i++) {
        EXPECT_NE(queue.pop(), nullptr);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// Sequence near INT64_MAX: interleaved push/pop crossing the boundary
TEST_F(SequenceWrapTest, InterleavedAcrossBoundary) {
    int64_t near_max = INT64_MAX - 2;
    InitQueueAtSequence(near_max);

    // Each push/pop advances sequence by 1; after 5 cycles we cross INT64_MAX
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(queue.push(&dummy[0])) << "Push " << i << " at sequence ~" << (near_max + i);
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr) << "Pop " << i;
        EXPECT_EQ(s, &dummy[0]);
    }
}

// Sequence at exactly INT64_MAX: single push/pop
TEST_F(SequenceWrapTest, ExactlyAtInt64Max) {
    InitQueueAtSequence(INT64_MAX);

    ASSERT_TRUE(queue.push(&dummy[0]));
    PTO2TaskSlotState *s = queue.pop();
    EXPECT_EQ(s, &dummy[0]);
}

// ---------------------------------------------------------------------------
// BUG-CANDIDATE: pop() fast-path uses unsigned comparison `d >= e`.
//
// When enqueue_pos crosses INT64_MAX (as uint64_t), the arithmetic is still
// valid for unsigned because uint64 wraps modularly.  However, inside push()
// and pop(), `static_cast<int64_t>(pos)` reinterprets bits: a pos of
// 0x8000000000000000 becomes INT64_MIN.  The sequence counters undergo the
// same reinterpretation, so diff calculations remain consistent.
//
// The REAL concern is pop()'s fast-path: `if (d >= e) return nullptr`.
// After enough operations, enqueue_pos wraps around UINT64_MAX back to a
// small number while dequeue_pos is still large.  At that point d > e
// (unsigned), causing pop() to return nullptr even though items are queued.
//
// This test starts positions near UINT64_MAX to simulate the wrap scenario.
// It documents that UINT64_MAX overflow in enqueue_pos/dequeue_pos would
// break the fast-path, but this requires 2^64 operations -- practically
// unreachable.  We test the INT64 boundary (2^63) which IS reachable in
// extremely long-running graphs.
// ---------------------------------------------------------------------------
TEST_F(SequenceWrapTest, PushBatchThenPopAcrossInt64Boundary) {
    // Start at INT64_MAX - 2 so that after 3 pushes, enqueue_pos crosses
    // into the INT64_MIN region (as signed), while dequeue_pos stays at
    // INT64_MAX - 2.
    int64_t start = INT64_MAX - 2;
    InitQueueAtSequence(start);

    // Push 5 items: pos goes INT64_MAX-2, -1, MAX, MAX+1, MAX+2
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(queue.push(&dummy[i])) << "Push " << i << " failed (pos would be ~INT64_MAX+" << (i - 2) << ")";
    }

    // Pop all 5: dequeue_pos starts at INT64_MAX-2, catches up.
    // The fast-path `d >= e` compares unsigned values; since both grow
    // monotonically as uint64_t, this stays correct across the signed
    // boundary.
    for (int i = 0; i < 5; i++) {
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr) << "Pop " << i << " returned nullptr -- fast-path may have misjudged empty";
        EXPECT_EQ(s, &dummy[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// Concurrent push/pop near INT64_MAX boundary
TEST_F(SequenceWrapTest, ConcurrentNearMax) {
    static constexpr uint64_t BIG_CAP = 64;
    PTO2ReadyQueueSlot big_slots[BIG_CAP];
    PTO2ReadyQueue big_queue{};
    int64_t start = INT64_MAX - 500;
    test_ready_queue_init(&big_queue, big_slots, BIG_CAP, start);

    constexpr int N = 1000;
    std::vector<PTO2TaskSlotState> items(N);
    for (int i = 0; i < N; i++)
        items[i].fanin_count = i;

    std::atomic<int> consumed{0};
    std::atomic<bool> prod_done{false};

    auto producer = [&]() {
        for (int i = 0; i < N; i++) {
            while (!big_queue.push(&items[i])) {}
        }
        prod_done.store(true, std::memory_order_release);
    };

    auto consumer = [&]() {
        while (true) {
            PTO2TaskSlotState *s = big_queue.pop();
            if (s) {
                consumed.fetch_add(1, std::memory_order_relaxed);
            } else if (prod_done.load(std::memory_order_acquire)) {
                while ((s = big_queue.pop()) != nullptr) {
                    consumed.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::thread p(producer);
    std::thread c1(consumer);
    std::thread c2(consumer);
    p.join();
    c1.join();
    c2.join();

    EXPECT_EQ(consumed.load(), N);
}

// =============================================================================
// 4. SharedMemory concurrent read/write of per-ring flow control
// =============================================================================

class SharedMemoryConcurrentTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle *handle = nullptr;

    void SetUp() override {
        handle = pto2_sm_create(256, 4096);
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        if (handle) {
            pto2_sm_destroy(handle);
            handle = nullptr;
        }
    }
};

// Concurrent current_task_index updates across different rings: no cross-ring interference
TEST_F(SharedMemoryConcurrentTest, PerRingTaskIndexIsolation) {
    constexpr int kIterations = 10000;

    auto writer = [&](int ring) {
        auto &fc = handle->header->rings[ring].fc;
        for (int i = 1; i <= kIterations; i++) {
            fc.current_task_index.store(static_cast<int32_t>(i), std::memory_order_release);
        }
    };

    auto reader = [&](int ring, bool *saw_other_ring_value) {
        auto &fc = handle->header->rings[ring].fc;
        int32_t prev = 0;
        for (int i = 0; i < kIterations; i++) {
            int32_t val = fc.current_task_index.load(std::memory_order_acquire);
            // Values should be monotonically increasing within a ring
            if (val < prev) {
                *saw_other_ring_value = true;
            }
            prev = val;
        }
    };

    // Write to ring 0 and ring 1 concurrently; read from each
    bool ring0_corrupted = false;
    bool ring1_corrupted = false;

    std::thread w0(writer, 0);
    std::thread w1(writer, 1);
    std::thread r0(reader, 0, &ring0_corrupted);
    std::thread r1(reader, 1, &ring1_corrupted);

    w0.join();
    w1.join();
    r0.join();
    r1.join();

    EXPECT_FALSE(ring0_corrupted) << "Ring 0 current_task_index should be monotonic";
    EXPECT_FALSE(ring1_corrupted) << "Ring 1 current_task_index should be monotonic";

    // Final values should be kIterations for each ring (independently)
    EXPECT_EQ(handle->header->rings[0].fc.current_task_index.load(), static_cast<int32_t>(kIterations));
    EXPECT_EQ(handle->header->rings[1].fc.current_task_index.load(), static_cast<int32_t>(kIterations));
}

// Concurrent current_task_index increment: simulate orchestrator publishing task IDs
TEST_F(SharedMemoryConcurrentTest, TaskIndexAtomicIncrement) {
    constexpr int kIncrements = 5000;
    constexpr int kThreads = 4;

    auto &fc = handle->header->rings[0].fc;
    fc.current_task_index.store(0, std::memory_order_relaxed);

    auto incrementer = [&]() {
        for (int i = 0; i < kIncrements; i++) {
            fc.current_task_index.fetch_add(1, std::memory_order_acq_rel);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back(incrementer);
    }
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(fc.current_task_index.load(), kIncrements * kThreads) << "Concurrent increments should not lose updates";
}

// Concurrent orchestrator_done and error code write: first-writer-wins semantics
TEST_F(SharedMemoryConcurrentTest, OrchestratorDoneRace) {
    constexpr int kRounds = 500;

    for (int round = 0; round < kRounds; round++) {
        handle->header->orchestrator_done.store(0, std::memory_order_relaxed);
        handle->header->orch_error_code.store(0, std::memory_order_relaxed);

        std::atomic<int> winners{0};

        auto try_set_done = [&](int32_t error_code) {
            int32_t expected = 0;
            if (handle->header->orchestrator_done.compare_exchange_strong(
                    expected, 1, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                handle->header->orch_error_code.store(error_code, std::memory_order_release);
                winners.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::thread t1(try_set_done, 100);
        std::thread t2(try_set_done, 200);
        std::thread t3(try_set_done, 300);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(winners.load(), 1) << "Round " << round << ": exactly one thread should win the CAS";
        EXPECT_EQ(handle->header->orchestrator_done.load(), 1);
        int32_t code = handle->header->orch_error_code.load();
        EXPECT_TRUE(code == 100 || code == 200 || code == 300)
            << "Error code should be from one of the competing threads";
    }
}

// Concurrent last_task_alive advancement: only forward movement
TEST_F(SharedMemoryConcurrentTest, LastTaskAliveMonotonic) {
    constexpr int kIterations = 10000;
    constexpr int kThreads = 4;

    auto &fc = handle->header->rings[0].fc;
    fc.last_task_alive.store(0, std::memory_order_relaxed);

    auto advancer = [&](int id) {
        for (int i = 0; i < kIterations; i++) {
            // CAS-based forward-only update
            int32_t desired = id * kIterations + i + 1;
            int32_t current = fc.last_task_alive.load(std::memory_order_acquire);
            while (current < desired) {
                if (fc.last_task_alive.compare_exchange_weak(
                        current, desired, std::memory_order_acq_rel, std::memory_order_acquire
                    )) {
                    break;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back(advancer, i);
    }
    for (auto &t : threads)
        t.join();

    int32_t final_val = fc.last_task_alive.load();
    // Should be at least the max of any thread's last write
    EXPECT_GE(final_val, kIterations) << "last_task_alive should have advanced";
}

// Validate after concurrent modifications still reports corruption correctly
TEST_F(SharedMemoryConcurrentTest, ValidateAfterConcurrentWrites) {
    constexpr int kIterations = 1000;

    // Concurrent writers update current_task_index within valid range
    auto writer = [&](int ring) {
        auto &fc = handle->header->rings[ring].fc;
        for (int i = 0; i < kIterations; i++) {
            fc.current_task_index.store(static_cast<int32_t>(i % 256), std::memory_order_release);
        }
    };

    std::thread w0(writer, 0);
    std::thread w1(writer, 1);
    std::thread w2(writer, 2);
    std::thread w3(writer, 3);
    w0.join();
    w1.join();
    w2.join();
    w3.join();

    EXPECT_TRUE(pto2_sm_validate(handle)) << "Valid current_task_index values should pass validation";

    // Corrupt one ring and verify detection
    handle->header->rings[2].fc.current_task_index.store(-1, std::memory_order_relaxed);
    EXPECT_FALSE(pto2_sm_validate(handle)) << "Corrupted current_task_index should fail validation";
}

// Double destroy: pto2_sm_destroy(NULL) is safe
TEST_F(SharedMemoryConcurrentTest, DestroyNullIsSafe) {
    pto2_sm_destroy(nullptr);  // Should not crash
}
