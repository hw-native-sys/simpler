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
 * Unit tests for PTO2ReadyQueue and PTO2LocalReadyBuffer from pto_scheduler.h
 *
 * Tests the lock-free bounded MPMC queue (Vyukov design) and the thread-local
 * ready buffer used for local-first dispatch optimization.
 *
 * Design contracts:
 *
 * - Sequence wrap: The sequence counter is int64_t.  Practically unreachable
 *   wrap at 2^63; two's-complement comparisons still work.
 *
 * - Pop fast-path: pop() checks enqueue_pos == dequeue_pos as an early-empty
 *   hint.  A push between the hint and the CAS can race; standard TOCTOU of
 *   Vyukov MPMC, acceptable.
 *
 * - Push near full: All producers that see a full slot return false
 *   simultaneously even if a pop happens right after.  Acceptable
 *   back-pressure.
 *
 * - size() relaxed ordering: size() reads both positions with
 *   memory_order_relaxed and is a hint, not a snapshot.  If a stale read
 *   produces d > e the guard returns 0.
 *
 * - LocalReadyBuffer LIFO dispatch: try_push appends at count++, pop returns
 *   slot_states[--count].  LIFO reversal is intentional for cache-locality
 *   when a producer immediately dispatches its fanout.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#include "scheduler/pto_scheduler.h"

// =============================================================================
// ReadyQueue: Single-threaded fixture (malloc-backed)
// =============================================================================

class ReadyQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t CAPACITY = 16;  // Power of 2

    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(ready_queue_init(&queue, CAPACITY)); }

    void TearDown() override { ready_queue_destroy(&queue); }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(ReadyQueueTest, EmptyPopReturnsNullptr) { EXPECT_EQ(queue.pop(), nullptr); }

TEST_F(ReadyQueueTest, SinglePushPop) {
    PTO2TaskSlotState item;
    ASSERT_TRUE(queue.push(&item));

    PTO2TaskSlotState *result = queue.pop();
    EXPECT_EQ(result, &item);
}

TEST_F(ReadyQueueTest, FIFOOrdering) {
    PTO2TaskSlotState a, b, c;

    ASSERT_TRUE(queue.push(&a));
    ASSERT_TRUE(queue.push(&b));
    ASSERT_TRUE(queue.push(&c));

    EXPECT_EQ(queue.pop(), &a);
    EXPECT_EQ(queue.pop(), &b);
    EXPECT_EQ(queue.pop(), &c);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, QueueFullReturnsFalse) {
    std::vector<PTO2TaskSlotState> items(CAPACITY);

    for (uint64_t i = 0; i < CAPACITY; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState extra;
    EXPECT_FALSE(queue.push(&extra));
}

TEST_F(ReadyQueueTest, SlotReuseAfterFullDrain) {
    std::vector<PTO2TaskSlotState> items(CAPACITY);

    for (uint64_t i = 0; i < CAPACITY; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }
    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);

    for (uint64_t i = 0; i < CAPACITY; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }
    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, PushBatchThenIndividualPop) {
    constexpr int BATCH_SIZE = 5;
    PTO2TaskSlotState items[BATCH_SIZE];
    PTO2TaskSlotState *ptrs[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        ptrs[i] = &items[i];
    }

    queue.push_batch(ptrs, BATCH_SIZE);

    for (int i = 0; i < BATCH_SIZE; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, PushBatchZeroIsNoop) {
    queue.push_batch(nullptr, 0);

    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, PopBatchReturnsFive) {
    constexpr int PUSH_COUNT = 10;
    PTO2TaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 5);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchPartial) {
    constexpr int PUSH_COUNT = 3;
    PTO2TaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, PUSH_COUNT);

    for (int i = 0; i < PUSH_COUNT; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchEmpty) {
    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 0);
}

TEST_F(ReadyQueueTest, SizeAccuracy) {
    EXPECT_EQ(queue.size(), 0u);

    PTO2TaskSlotState items[8];

    queue.push(&items[0]);
    EXPECT_EQ(queue.size(), 1u);

    queue.push(&items[1]);
    queue.push(&items[2]);
    EXPECT_EQ(queue.size(), 3u);

    queue.pop();
    EXPECT_EQ(queue.size(), 2u);

    queue.pop();
    queue.pop();
    EXPECT_EQ(queue.size(), 0u);

    for (int i = 0; i < 5; i++) {
        queue.push(&items[i]);
    }
    EXPECT_EQ(queue.size(), 5u);
}

// =============================================================================
// Boundary conditions (small capacity for precise boundary testing)
// =============================================================================

class ReadyQueueBoundaryTest : public ::testing::Test {
protected:
    static constexpr uint64_t QUEUE_CAP = 8;  // Small for boundary testing
    PTO2ReadyQueue queue{};
    PTO2TaskSlotState dummy[8]{};

    void SetUp() override { ASSERT_TRUE(ready_queue_init(&queue, QUEUE_CAP)); }
    void TearDown() override { ready_queue_destroy(&queue); }
};

TEST_F(ReadyQueueBoundaryTest, ExactCapacityFillDrain) {
    int pushed = 0;
    for (uint64_t i = 0; i < QUEUE_CAP; i++) {
        if (queue.push(&dummy[i % 8])) pushed++;
        else break;
    }
    EXPECT_GE(pushed, (int)(QUEUE_CAP - 1));

    for (int i = 0; i < pushed; i++) {
        EXPECT_NE(queue.pop(), nullptr);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueBoundaryTest, PushToFullThenRecover) {
    int pushed = 0;
    while (queue.push(&dummy[0]))
        pushed++;

    EXPECT_FALSE(queue.push(&dummy[1])) << "Push to full queue returns false";

    EXPECT_NE(queue.pop(), nullptr);
    EXPECT_TRUE(queue.push(&dummy[1])) << "Push succeeds after pop from full queue";
}

// size() with relaxed ordering: exact in single-threaded context.
TEST_F(ReadyQueueBoundaryTest, SizeRelaxedOrdering) {
    queue.push(&dummy[0]);
    queue.push(&dummy[1]);
    queue.push(&dummy[2]);
    EXPECT_EQ(queue.size(), 3u);

    queue.pop();
    EXPECT_EQ(queue.size(), 2u);

    queue.pop();
    queue.pop();
    EXPECT_EQ(queue.size(), 0u);
}

// size() guard: after many push/pop cycles, never goes negative.
TEST_F(ReadyQueueBoundaryTest, SizeNeverNegative) {
    for (int i = 0; i < 100; i++) {
        ASSERT_TRUE(queue.push(&dummy[0]));
        queue.pop();
    }
    EXPECT_EQ(queue.size(), 0u) << "size() returns 0 after balanced push/pop cycles";
}

TEST_F(ReadyQueueBoundaryTest, RepeatedEmptyPop) {
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(queue.pop(), nullptr);
    }
    EXPECT_EQ(queue.size(), 0u);
}

// Sequence numbers grow large after many cycles but remain correct.
TEST_F(ReadyQueueBoundaryTest, ManyPushPopCycles) {
    for (int i = 0; i < 10000; i++) {
        ASSERT_TRUE(queue.push(&dummy[0]));
        PTO2TaskSlotState *s = queue.pop();
        ASSERT_NE(s, nullptr);
        EXPECT_EQ(s, &dummy[0]);
    }

    EXPECT_EQ(queue.size(), 0u);
    EXPECT_TRUE(queue.push(&dummy[1]));
    EXPECT_EQ(queue.pop(), &dummy[1]);
}

// =============================================================================
// Concurrency
// =============================================================================

// Parameterized MPMC stress test: {producers, consumers, items_per_producer}
struct MPMCConfig {
    int producers;
    int consumers;
    int items_per_producer;
};

class ReadyQueueMPMCTest : public ::testing::TestWithParam<MPMCConfig> {
protected:
    static constexpr uint64_t CAPACITY = 1024;
    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(ready_queue_init(&queue, CAPACITY)); }
    void TearDown() override { ready_queue_destroy(&queue); }
};

TEST_P(ReadyQueueMPMCTest, NoDuplicateNoLoss) {
    auto cfg = GetParam();
    int total = cfg.producers * cfg.items_per_producer;

    std::vector<PTO2TaskSlotState> items(total);
    std::vector<std::atomic<int>> consumed_count(total);
    for (int i = 0; i < total; i++) {
        consumed_count[i].store(0, std::memory_order_relaxed);
    }

    auto item_index = [&](PTO2TaskSlotState *s) -> int {
        return static_cast<int>(s - items.data());
    };

    std::atomic<int> producers_done{0};

    auto producer = [&](int id) {
        for (int i = id; i < total; i += cfg.producers) {
            while (!queue.push(&items[i])) {}
        }
        producers_done.fetch_add(1, std::memory_order_release);
    };

    std::atomic<int> total_consumed{0};

    auto consumer = [&]() {
        while (true) {
            PTO2TaskSlotState *item = queue.pop();
            if (item != nullptr) {
                consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
                total_consumed.fetch_add(1, std::memory_order_relaxed);
            } else if (producers_done.load(std::memory_order_acquire) == cfg.producers) {
                // Drain remaining
                while ((item = queue.pop()) != nullptr) {
                    consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
                    total_consumed.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < cfg.producers; i++)
        threads.emplace_back(producer, i);
    for (int i = 0; i < cfg.consumers; i++)
        threads.emplace_back(consumer);
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(total_consumed.load(), total);
    for (int i = 0; i < total; i++) {
        EXPECT_EQ(consumed_count[i].load(), 1)
            << "Item " << i << " consumed " << consumed_count[i].load() << " times (expected 1)";
    }
}

INSTANTIATE_TEST_SUITE_P(
    MPMCVariants, ReadyQueueMPMCTest,
    ::testing::Values(
        MPMCConfig{2, 2, 200},   // TwoProducersTwoConsumers
        MPMCConfig{1, 4, 500},   // OneProducerNConsumers
        MPMCConfig{4, 4, 1250},  // HighContentionStress
        MPMCConfig{8, 8, 2000}   // EightProducersEightConsumers
    )
);

// =============================================================================
// LocalReadyBuffer
// =============================================================================

class LocalReadyBufferTest : public ::testing::Test {
protected:
    static constexpr int CAPACITY = 8;

    PTO2LocalReadyBuffer buffer;
    PTO2TaskSlotState *backing[CAPACITY];

    void SetUp() override { buffer.reset(backing, CAPACITY); }
};

// --- Normal path ---

TEST_F(LocalReadyBufferTest, PopEmptyReturnsNullptr) { EXPECT_EQ(buffer.pop(), nullptr); }

// LIFO dispatch: try_push appends at count++, pop returns slot_states[--count].
TEST_F(LocalReadyBufferTest, LIFOOrdering) {
    PTO2TaskSlotState a, b;

    ASSERT_TRUE(buffer.try_push(&a));
    ASSERT_TRUE(buffer.try_push(&b));

    EXPECT_EQ(buffer.pop(), &b);
    EXPECT_EQ(buffer.pop(), &a);
    EXPECT_EQ(buffer.pop(), nullptr);
}

TEST_F(LocalReadyBufferTest, TryPushFullReturnsFalse) {
    PTO2TaskSlotState items[CAPACITY + 1];

    for (int i = 0; i < CAPACITY; i++) {
        ASSERT_TRUE(buffer.try_push(&items[i]));
    }

    EXPECT_FALSE(buffer.try_push(&items[CAPACITY]));
}

TEST_F(LocalReadyBufferTest, ResetSetsCleanState) {
    EXPECT_EQ(buffer.pop(), nullptr) << "Fresh buffer is empty";

    PTO2TaskSlotState a, b;
    ASSERT_TRUE(buffer.try_push(&a));
    ASSERT_TRUE(buffer.try_push(&b));

    buffer.reset(backing, CAPACITY);
    EXPECT_EQ(buffer.pop(), nullptr) << "Buffer is empty after reset";

    PTO2TaskSlotState items[CAPACITY];
    for (int i = 0; i < CAPACITY; i++) {
        EXPECT_TRUE(buffer.try_push(&items[i]));
    }
    EXPECT_FALSE(buffer.try_push(&a)) << "Full after pushing capacity items post-reset";
}

// --- Boundary conditions ---

TEST_F(LocalReadyBufferTest, NullBackingBuffer) {
    PTO2LocalReadyBuffer buf;
    buf.reset(nullptr, 0);

    PTO2TaskSlotState item{};
    EXPECT_FALSE(buf.try_push(&item)) << "Push fails with null backing";
    EXPECT_EQ(buf.pop(), nullptr) << "Pop returns null with null backing";
}

// =============================================================================
// High-contention stress tests
// =============================================================================

class ReadyQueueStressTest : public ::testing::Test {
protected:
    static constexpr uint64_t kCapacity = 512;
    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(ready_queue_init(&queue, kCapacity)); }
    void TearDown() override { ready_queue_destroy(&queue); }
};

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

        auto producer = [&](int id) {
            int per_thread = kItemsPerCycle / 4;
            int base = id * per_thread;
            for (int i = 0; i < per_thread; i++) {
                while (!queue.push(&items[base + i])) {}
            }
            push_done.fetch_add(1, std::memory_order_release);
        };

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

TEST_F(ReadyQueueStressTest, PopBatchUnderContention) {
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
