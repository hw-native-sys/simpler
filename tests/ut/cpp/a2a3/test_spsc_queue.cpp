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
 * Unit tests for PTO2SpscQueue from pto_scheduler.h
 *
 * Tests the Rigtorp cached-index SPSC queue used as the orchestrator →
 * scheduler wiring channel:
 * - Basic push / pop_batch correctness
 * - Full / empty detection (including cached-index lazy refresh)
 * - Wrap-around via modulo indexing
 * - Capacity is capacity-1 (one sentinel slot)
 * - pop_batch partial reads
 * - size() accuracy
 */

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

#include "scheduler/pto_scheduler.h"

// =============================================================================
// Fixture
// =============================================================================

class SpscQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t CAPACITY = 16;  // must be power of 2

    PTO2SpscQueue queue{};
    // Dummy slot states used as push values
    alignas(64) PTO2TaskSlotState slots[64]{};

    void SetUp() override {
        memset(&queue, 0, sizeof(queue));
        ASSERT_TRUE(queue.init(CAPACITY));
    }

    void TearDown() override { queue.destroy(); }
};

// =============================================================================
// Initialization
// =============================================================================

TEST_F(SpscQueueTest, InitValidState) {
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.mask_, CAPACITY - 1);
    EXPECT_NE(queue.buffer_, nullptr);
}

TEST_F(SpscQueueTest, InitRejectsNonPowerOfTwo) {
    PTO2SpscQueue bad{};
    EXPECT_FALSE(bad.init(3));
    EXPECT_FALSE(bad.init(7));
    EXPECT_FALSE(bad.init(0));
}

TEST_F(SpscQueueTest, InitAcceptsPowerOfTwo) {
    PTO2SpscQueue q{};
    EXPECT_TRUE(q.init(4));
    q.destroy();
    EXPECT_TRUE(q.init(1024));
    q.destroy();
}

// =============================================================================
// Basic push / pop
// =============================================================================

TEST_F(SpscQueueTest, PushPopSingle) {
    EXPECT_TRUE(queue.push(&slots[0]));
    EXPECT_EQ(queue.size(), 1u);

    PTO2TaskSlotState *out[1];
    int count = queue.pop_batch(out, 1);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(out[0], &slots[0]);
    EXPECT_EQ(queue.size(), 0u);
}

TEST_F(SpscQueueTest, FIFOOrdering) {
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(queue.push(&slots[i]));
    }

    PTO2TaskSlotState *out[5];
    int count = queue.pop_batch(out, 5);
    ASSERT_EQ(count, 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(out[i], &slots[i]) << "FIFO order violated at i=" << i;
    }
}

TEST_F(SpscQueueTest, PopBatchPartial) {
    for (int i = 0; i < 3; i++) {
        queue.push(&slots[i]);
    }

    // Request 5 but only 3 available
    PTO2TaskSlotState *out[5];
    int count = queue.pop_batch(out, 5);
    EXPECT_EQ(count, 3);
}

TEST_F(SpscQueueTest, PopBatchEmpty) {
    PTO2TaskSlotState *out[5];
    int count = queue.pop_batch(out, 5);
    EXPECT_EQ(count, 0);
}

// =============================================================================
// Full detection
// =============================================================================

TEST_F(SpscQueueTest, FullReturnsFalse) {
    // Usable capacity = CAPACITY - 1 = 15
    for (uint64_t i = 0; i < CAPACITY - 1; i++) {
        ASSERT_TRUE(queue.push(&slots[i])) << "push failed at i=" << i;
    }
    EXPECT_EQ(queue.size(), CAPACITY - 1);

    // Queue full
    EXPECT_FALSE(queue.push(&slots[CAPACITY - 1])) << "Push to full queue must return false";
}

TEST_F(SpscQueueTest, UsableCapacityIsCapacityMinusOne) {
    int pushed = 0;
    while (queue.push(&slots[pushed % 64])) {
        pushed++;
        if (pushed > 100) break;  // safety
    }
    EXPECT_EQ(pushed, static_cast<int>(CAPACITY - 1));
}

// =============================================================================
// Full then recover
// =============================================================================

TEST_F(SpscQueueTest, FullThenPopThenPush) {
    for (uint64_t i = 0; i < CAPACITY - 1; i++) {
        queue.push(&slots[i]);
    }
    EXPECT_FALSE(queue.push(&slots[0]));

    // Pop one
    PTO2TaskSlotState *out[1];
    int count = queue.pop_batch(out, 1);
    ASSERT_EQ(count, 1);

    // Now push should succeed
    EXPECT_TRUE(queue.push(&slots[0]));
}

// =============================================================================
// Wrap-around
// =============================================================================

TEST_F(SpscQueueTest, WrapAroundCorrectness) {
    // Push-pop cycles to advance head/tail past capacity boundary
    for (int cycle = 0; cycle < 100; cycle++) {
        ASSERT_TRUE(queue.push(&slots[cycle % 64])) << "push failed at cycle=" << cycle;
        PTO2TaskSlotState *out[1];
        int count = queue.pop_batch(out, 1);
        ASSERT_EQ(count, 1) << "pop_batch failed at cycle=" << cycle;
        EXPECT_EQ(out[0], &slots[cycle % 64]);
    }
    EXPECT_EQ(queue.size(), 0u);
}

TEST_F(SpscQueueTest, WrapAroundBatchCorrectness) {
    // Multiple cycles of batch push/pop across wrap boundary
    for (int cycle = 0; cycle < 20; cycle++) {
        int batch = 5;
        for (int i = 0; i < batch; i++) {
            ASSERT_TRUE(queue.push(&slots[(cycle * batch + i) % 64]));
        }
        PTO2TaskSlotState *out[5];
        int count = queue.pop_batch(out, batch);
        ASSERT_EQ(count, batch);
        for (int i = 0; i < batch; i++) {
            EXPECT_EQ(out[i], &slots[(cycle * batch + i) % 64]);
        }
    }
}

// =============================================================================
// size() accuracy
// =============================================================================

TEST_F(SpscQueueTest, SizeTracksOperations) {
    EXPECT_EQ(queue.size(), 0u);

    queue.push(&slots[0]);
    EXPECT_EQ(queue.size(), 1u);

    queue.push(&slots[1]);
    queue.push(&slots[2]);
    EXPECT_EQ(queue.size(), 3u);

    PTO2TaskSlotState *out[2];
    queue.pop_batch(out, 2);
    EXPECT_EQ(queue.size(), 1u);

    queue.pop_batch(out, 1);
    EXPECT_EQ(queue.size(), 0u);
}

// =============================================================================
// Producer-consumer (two threads)
// =============================================================================

TEST_F(SpscQueueTest, TwoThreadProducerConsumer) {
    constexpr int TOTAL = 10000;
    std::vector<PTO2TaskSlotState *> consumed;
    consumed.reserve(TOTAL);

    // Use a large pool of slot states for unique pointers
    std::vector<PTO2TaskSlotState> big_pool(TOTAL);

    std::thread producer([&]() {
        for (int i = 0; i < TOTAL; i++) {
            while (!queue.push(&big_pool[i])) {
                // spin
            }
        }
    });

    std::thread consumer([&]() {
        int total = 0;
        PTO2TaskSlotState *out[32];
        while (total < TOTAL) {
            int count = queue.pop_batch(out, 32);
            for (int i = 0; i < count; i++) {
                consumed.push_back(out[i]);
            }
            total += count;
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(consumed.size(), static_cast<size_t>(TOTAL));
    // Verify FIFO order
    for (int i = 0; i < TOTAL; i++) {
        EXPECT_EQ(consumed[i], &big_pool[i]) << "FIFO violated at i=" << i;
    }
}

// =============================================================================
// Cached index behavior
// =============================================================================

TEST_F(SpscQueueTest, CachedIndexLazyRefresh) {
    // Fill queue
    for (uint64_t i = 0; i < CAPACITY - 1; i++) {
        queue.push(&slots[i]);
    }

    // Consumer pops all
    PTO2TaskSlotState *out[16];
    int count = queue.pop_batch(out, CAPACITY);
    EXPECT_EQ(count, static_cast<int>(CAPACITY - 1));

    // Producer's tail_cached_ is stale (still thinks queue is full)
    // Next push should refresh tail_cached_ and succeed
    EXPECT_TRUE(queue.push(&slots[0]));
}

TEST_F(SpscQueueTest, CachedIndexConsumerRefresh) {
    // Consumer calls pop_batch on empty queue (head_cached_ is 0)
    PTO2TaskSlotState *out[1];
    EXPECT_EQ(queue.pop_batch(out, 1), 0);

    // Producer pushes
    queue.push(&slots[0]);

    // Consumer's head_cached_ is stale, pop_batch must refresh
    int count = queue.pop_batch(out, 1);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(out[0], &slots[0]);
}
