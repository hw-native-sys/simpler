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
 * Unit tests for PTO2ReadyQueue from pto_scheduler.h
 *
 * Tests the lock-free bounded MPMC queue (Vyukov design).
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
 */

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/pto_scheduler.h"

// =============================================================================
// ReadyQueue: Single-threaded fixture (malloc-backed)
// =============================================================================

class ReadyQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t CAPACITY = 16;  // Power of 2

    PTO2ReadyQueue queue;
    DeviceArena arena;

    void SetUp() override {
        const size_t off = ready_queue_reserve_layout(arena, CAPACITY);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(ready_queue_init_data_from_layout(&queue, arena, off, CAPACITY));
        ready_queue_wire_arena_pointers(&queue, arena, off);
    }

    void TearDown() override {
        ready_queue_destroy(&queue);
        arena.release();
    }
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

TEST_F(ReadyQueueTest, TaggedBatchPopPreservesTaskGeneration) {
    PTO2TaskSlotState items[2];
    PTO2TaskSlotState *input[2]{&items[0], &items[1]};
    constexpr uint64_t queued_task_ids[2]{0x123456789abcdef0ULL, 0x0fedcba987654321ULL};
    queue.push_batch_tagged(input, queued_task_ids, 2);

    PTO2TaskSlotState *out[2];
    uint64_t task_id_snapshots[2]{};
    ASSERT_EQ(queue.pop_batch_tagged(out, task_id_snapshots, 2), 2);
    EXPECT_EQ(out[0], &items[0]);
    EXPECT_EQ(out[1], &items[1]);
    EXPECT_EQ(task_id_snapshots[0], queued_task_ids[0]);
    EXPECT_EQ(task_id_snapshots[1], queued_task_ids[1]);
}

TEST_F(ReadyQueueTest, TaggedPopPreservesTaskGeneration) {
    PTO2TaskSlotState item;
    constexpr uint64_t queued_task_id = 0xfedcba9876543210ULL;
    ASSERT_TRUE(queue.push_tagged(&item, queued_task_id));

    uint64_t task_id_snapshot = 0;
    EXPECT_EQ(queue.pop_tagged(&task_id_snapshot), &item);
    EXPECT_EQ(task_id_snapshot, queued_task_id);
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

    DeviceArena arena;

    void SetUp() override {
        const size_t off = ready_queue_reserve_layout(arena, QUEUE_CAP);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(ready_queue_init_data_from_layout(&queue, arena, off, QUEUE_CAP));
        ready_queue_wire_arena_pointers(&queue, arena, off);
    }
    void TearDown() override {
        ready_queue_destroy(&queue);
        arena.release();
    }
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

    DeviceArena arena;

    void SetUp() override {
        const size_t off = ready_queue_reserve_layout(arena, CAPACITY);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(ready_queue_init_data_from_layout(&queue, arena, off, CAPACITY));
        ready_queue_wire_arena_pointers(&queue, arena, off);
    }
    void TearDown() override {
        ready_queue_destroy(&queue);
        arena.release();
    }
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
        MPMCConfig{2, 2, 200},  // TwoProducersTwoConsumers
        MPMCConfig{1, 4, 500},  // OneProducerNConsumers
        MPMCConfig{4, 4, 1250}  // HighContentionStress
    )
);
