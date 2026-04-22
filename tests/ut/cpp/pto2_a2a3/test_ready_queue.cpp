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
 */

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#include "pto_scheduler.h"

// =============================================================================
// ReadyQueue: Single-threaded tests
// =============================================================================

class ReadyQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t kCapacity = 16;  // Power of 2

    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(pto2_ready_queue_init(&queue, kCapacity)); }

    void TearDown() override { pto2_ready_queue_destroy(&queue); }
};

// 1. Empty pop returns nullptr
TEST_F(ReadyQueueTest, EmptyPopReturnsNullptr) { EXPECT_EQ(queue.pop(), nullptr); }

// 2. Single push/pop returns correct item
TEST_F(ReadyQueueTest, SinglePushPop) {
    PTO2TaskSlotState item;
    ASSERT_TRUE(queue.push(&item));

    PTO2TaskSlotState *result = queue.pop();
    EXPECT_EQ(result, &item);
}

// 3. FIFO ordering: push A,B,C then pop A,B,C
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

// 4. Queue full: push returns false at capacity
TEST_F(ReadyQueueTest, QueueFullReturnsFalse) {
    std::vector<PTO2TaskSlotState> items(kCapacity);

    for (uint64_t i = 0; i < kCapacity; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState extra;
    EXPECT_FALSE(queue.push(&extra));
}

// 5. Slot reuse after full drain (push/pop cycle)
TEST_F(ReadyQueueTest, SlotReuseAfterFullDrain) {
    std::vector<PTO2TaskSlotState> items(kCapacity);

    // Fill the queue
    for (uint64_t i = 0; i < kCapacity; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    // Drain the queue
    for (uint64_t i = 0; i < kCapacity; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);

    // Refill and re-drain to verify slot reuse
    for (uint64_t i = 0; i < kCapacity; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }
    for (uint64_t i = 0; i < kCapacity; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// 6. push_batch: batch enqueue then individual dequeue
TEST_F(ReadyQueueTest, PushBatchThenIndividualPop) {
    constexpr int kBatchSize = 5;
    PTO2TaskSlotState items[kBatchSize];
    PTO2TaskSlotState *ptrs[kBatchSize];
    for (int i = 0; i < kBatchSize; i++) {
        ptrs[i] = &items[i];
    }

    queue.push_batch(ptrs, kBatchSize);

    for (int i = 0; i < kBatchSize; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// 7. push_batch count=0: no-op
TEST_F(ReadyQueueTest, PushBatchZeroIsNoop) {
    queue.push_batch(nullptr, 0);

    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.pop(), nullptr);
}

// 8. pop_batch: push 10, pop_batch(5) returns 5
TEST_F(ReadyQueueTest, PopBatchReturnsFive) {
    constexpr int kPushCount = 10;
    PTO2TaskSlotState items[kPushCount];

    for (int i = 0; i < kPushCount; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 5);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

// 9. pop_batch partial: push 3, pop_batch(5) returns 3
TEST_F(ReadyQueueTest, PopBatchPartial) {
    constexpr int kPushCount = 3;
    PTO2TaskSlotState items[kPushCount];

    for (int i = 0; i < kPushCount; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, kPushCount);

    for (int i = 0; i < kPushCount; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

// 10. pop_batch empty: returns 0
TEST_F(ReadyQueueTest, PopBatchEmpty) {
    PTO2TaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 0);
}

// 11. size() accuracy after various push/pop
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

    // Push 5 more
    for (int i = 0; i < 5; i++) {
        queue.push(&items[i]);
    }
    EXPECT_EQ(queue.size(), 5u);
}

// =============================================================================
// ReadyQueue: Multi-threaded tests
// =============================================================================

class ReadyQueueMTTest : public ::testing::Test {
protected:
    static constexpr uint64_t kCapacity = 1024;  // Power of 2

    PTO2ReadyQueue queue;

    void SetUp() override { ASSERT_TRUE(pto2_ready_queue_init(&queue, kCapacity)); }

    void TearDown() override { pto2_ready_queue_destroy(&queue); }
};

// 12. 2 producers / 2 consumers: all items consumed exactly once
TEST_F(ReadyQueueMTTest, TwoProducersTwoConsumers) {
    constexpr int kItemsPerProducer = 200;
    constexpr int kTotalItems = kItemsPerProducer * 2;

    std::vector<PTO2TaskSlotState> items(kTotalItems);

    std::atomic<int> produced{0};
    std::atomic<bool> producers_done{false};

    // Tracking: atomic counter per item to verify exactly-once consumption
    // Use pointer identity (index into items array) instead of struct field tagging
    std::vector<std::atomic<int>> consumed_count(kTotalItems);
    for (int i = 0; i < kTotalItems; i++) {
        consumed_count[i].store(0, std::memory_order_relaxed);
    }

    auto item_index = [&](PTO2TaskSlotState *s) -> int {
        return static_cast<int>(s - items.data());
    };

    auto producer = [&](int offset) {
        for (int i = 0; i < kItemsPerProducer; i++) {
            while (!queue.push(&items[offset + i])) {
                // Queue full, retry
            }
        }
        produced.fetch_add(kItemsPerProducer, std::memory_order_release);
    };

    auto consumer = [&](std::vector<PTO2TaskSlotState *> &results) {
        while (true) {
            PTO2TaskSlotState *item = queue.pop();
            if (item != nullptr) {
                results.push_back(item);
                consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
            } else if (producers_done.load(std::memory_order_acquire)) {
                // Drain remaining
                while ((item = queue.pop()) != nullptr) {
                    results.push_back(item);
                    consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::vector<PTO2TaskSlotState *> results_c1, results_c2;
    std::thread p1(producer, 0);
    std::thread p2(producer, kItemsPerProducer);
    std::thread c1(consumer, std::ref(results_c1));
    std::thread c2(consumer, std::ref(results_c2));

    p1.join();
    p2.join();
    producers_done.store(true, std::memory_order_release);
    c1.join();
    c2.join();

    // Verify all items consumed exactly once
    int total_consumed = static_cast<int>(results_c1.size() + results_c2.size());
    EXPECT_EQ(total_consumed, kTotalItems);

    for (int i = 0; i < kTotalItems; i++) {
        EXPECT_EQ(consumed_count[i].load(), 1)
            << "Item " << i << " consumed " << consumed_count[i].load() << " times (expected 1)";
    }
}

// 13. 1 producer / N consumers: all items consumed exactly once
TEST_F(ReadyQueueMTTest, OneProducerNConsumers) {
    constexpr int kTotalItems = 500;
    constexpr int kNumConsumers = 4;

    std::vector<PTO2TaskSlotState> items(kTotalItems);

    std::atomic<bool> producer_done{false};
    std::vector<std::atomic<int>> consumed_count(kTotalItems);
    for (int i = 0; i < kTotalItems; i++) {
        consumed_count[i].store(0, std::memory_order_relaxed);
    }

    auto item_index = [&](PTO2TaskSlotState *s) -> int {
        return static_cast<int>(s - items.data());
    };

    auto producer = [&]() {
        for (int i = 0; i < kTotalItems; i++) {
            while (!queue.push(&items[i])) {
                // Queue full, retry
            }
        }
        producer_done.store(true, std::memory_order_release);
    };

    std::atomic<int> total_consumed{0};

    auto consumer = [&]() {
        while (true) {
            PTO2TaskSlotState *item = queue.pop();
            if (item != nullptr) {
                consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
                total_consumed.fetch_add(1, std::memory_order_relaxed);
            } else if (producer_done.load(std::memory_order_acquire)) {
                // Drain remaining
                while ((item = queue.pop()) != nullptr) {
                    consumed_count[item_index(item)].fetch_add(1, std::memory_order_relaxed);
                    total_consumed.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
        }
    };

    std::thread prod(producer);
    std::vector<std::thread> consumers;
    for (int i = 0; i < kNumConsumers; i++) {
        consumers.emplace_back(consumer);
    }

    prod.join();
    for (auto &c : consumers) {
        c.join();
    }

    EXPECT_EQ(total_consumed.load(), kTotalItems);

    for (int i = 0; i < kTotalItems; i++) {
        EXPECT_EQ(consumed_count[i].load(), 1)
            << "Item " << i << " consumed " << consumed_count[i].load() << " times (expected 1)";
    }
}

// =============================================================================
// LocalReadyBuffer tests
// =============================================================================

class LocalReadyBufferTest : public ::testing::Test {
protected:
    static constexpr int kCapacity = 8;

    PTO2LocalReadyBuffer buffer;
    PTO2TaskSlotState *backing[kCapacity];

    void SetUp() override { buffer.reset(backing, kCapacity); }
};

// 14. reset produces empty buffer that accepts pushes
TEST_F(LocalReadyBufferTest, ResetSetsCleanState) {
    // After reset, buffer should behave as empty
    EXPECT_EQ(buffer.pop(), nullptr) << "Fresh buffer is empty";

    // Push and verify it works
    PTO2TaskSlotState a, b;
    ASSERT_TRUE(buffer.try_push(&a));
    ASSERT_TRUE(buffer.try_push(&b));

    // Reset and verify empty behavior is restored
    buffer.reset(backing, kCapacity);
    EXPECT_EQ(buffer.pop(), nullptr) << "Buffer is empty after reset";

    // Should accept full capacity of pushes again
    PTO2TaskSlotState items[kCapacity];
    for (int i = 0; i < kCapacity; i++) {
        EXPECT_TRUE(buffer.try_push(&items[i]));
    }
    EXPECT_FALSE(buffer.try_push(&a)) << "Full after pushing capacity items post-reset";
}

// 15. try_push/pop LIFO: push A,B -> pop returns B,A
TEST_F(LocalReadyBufferTest, LIFOOrdering) {
    PTO2TaskSlotState a, b;

    ASSERT_TRUE(buffer.try_push(&a));
    ASSERT_TRUE(buffer.try_push(&b));

    EXPECT_EQ(buffer.pop(), &b);
    EXPECT_EQ(buffer.pop(), &a);
    EXPECT_EQ(buffer.pop(), nullptr);
}

// 16. try_push full: returns false at capacity
TEST_F(LocalReadyBufferTest, TryPushFullReturnsFalse) {
    PTO2TaskSlotState items[kCapacity + 1];

    for (int i = 0; i < kCapacity; i++) {
        ASSERT_TRUE(buffer.try_push(&items[i]));
    }

    EXPECT_FALSE(buffer.try_push(&items[kCapacity]));
}

// 17. pop empty: returns nullptr
TEST_F(LocalReadyBufferTest, PopEmptyReturnsNullptr) { EXPECT_EQ(buffer.pop(), nullptr); }
