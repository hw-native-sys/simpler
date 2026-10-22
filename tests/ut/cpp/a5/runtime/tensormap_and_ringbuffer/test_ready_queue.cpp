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
 * Unit tests for ChipReadyQueue from scheduler.h
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
#include <chrono>
#include <algorithm>
#include <set>
#include <thread>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"

// =============================================================================
// ReadyQueue: Single-threaded fixture (malloc-backed)
// =============================================================================

class ReadyQueueTest : public ::testing::Test {
protected:
    static constexpr uint64_t CAPACITY = 16;  // Power of 2

    ChipReadyQueue queue;
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
    ChipTaskSlotState item;
    ASSERT_TRUE(queue.push(&item));

    ChipTaskSlotState *result = queue.pop();
    EXPECT_EQ(result, &item);
}

TEST_F(ReadyQueueTest, FIFOOrdering) {
    ChipTaskSlotState a, b, c;

    ASSERT_TRUE(queue.push(&a));
    ASSERT_TRUE(queue.push(&b));
    ASSERT_TRUE(queue.push(&c));

    EXPECT_EQ(queue.pop(), &a);
    EXPECT_EQ(queue.pop(), &b);
    EXPECT_EQ(queue.pop(), &c);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, QueueFullReturnsFalse) {
    std::vector<ChipTaskSlotState> items(CAPACITY);

    for (uint64_t i = 0; i < CAPACITY; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    ChipTaskSlotState extra;
    EXPECT_FALSE(queue.push(&extra));
}

TEST_F(ReadyQueueTest, SlotReuseAfterFullDrain) {
    std::vector<ChipTaskSlotState> items(CAPACITY);

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
    ChipTaskSlotState items[BATCH_SIZE];
    ChipTaskSlotState *ptrs[BATCH_SIZE];
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
    ChipTaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    ChipTaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 5);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchPartial) {
    constexpr int PUSH_COUNT = 3;
    ChipTaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    ChipTaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, PUSH_COUNT);

    for (int i = 0; i < PUSH_COUNT; i++) {
        EXPECT_EQ(out[i], &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchEmpty) {
    ChipTaskSlotState *out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 0);
}

TEST_F(ReadyQueueTest, SizeAccuracy) {
    EXPECT_EQ(queue.size(), 0u);

    ChipTaskSlotState items[8];

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
    ChipReadyQueue queue{};
    ChipTaskSlotState dummy[8]{};

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
        ChipTaskSlotState *s = queue.pop();
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
    ChipReadyQueue queue;

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

    std::vector<ChipTaskSlotState> items(total);
    std::vector<std::atomic<int>> consumed_count(total);
    for (int i = 0; i < total; i++) {
        consumed_count[i].store(0, std::memory_order_relaxed);
    }

    auto item_index = [&](ChipTaskSlotState *s) -> int {
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
            ChipTaskSlotState *item = queue.pop();
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

class MixReadyQueueTest : public ReadyQueueTest {
protected:
    void SetUp() override {
        ReadyQueueTest::SetUp();
        queue.track_mix_resources = true;
    }
};

TEST_F(MixReadyQueueTest, MaskUnionTracksCellsAndTheirOriginalGeneration) {
    ChipTaskSlotState left, right, both, full;
    left.active_mask = ActiveMask(3);
    right.active_mask = ActiveMask(5);
    both.active_mask = ActiveMask(6);
    full.active_mask = ActiveMask(7);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
    ASSERT_TRUE(queue.push_tagged(&left, TaskId::make(0, 1), 3));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 3);
    ASSERT_TRUE(queue.push(&right, 5));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 7);
    // A stale early entry must release its cell's mask, not a reused task's mask.
    left.active_mask = ActiveMask(2);
    TaskId tag;
    EXPECT_EQ(queue.pop_tagged(&tag), &left);
    EXPECT_EQ(tag, TaskId::make(0, 1));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 5);
    ASSERT_TRUE(queue.push(&both, 6));
    ASSERT_TRUE(queue.push(&full, 7));
    EXPECT_EQ(queue.pop(), &right);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 7);
    EXPECT_EQ(queue.pop(), &both);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 7);
    EXPECT_EQ(queue.pop(), &full);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}

TEST_F(MixReadyQueueTest, FailedPushAndBatchRollBackReservations) {
    ChipTaskSlotState left, right;
    left.active_mask = ActiveMask(3);
    right.active_mask = ActiveMask(5);
    for (uint64_t i = 0; i < CAPACITY; ++i)
        ASSERT_TRUE(queue.push(&left, 3));
    EXPECT_FALSE(queue.push(&right, 5));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 3);
    ChipTaskSlotState *batch[] = {&right, &right};
    uint8_t core_masks[] = {5, 5};
    EXPECT_FALSE(queue.push_batch(batch, 2, core_masks));
    EXPECT_FALSE(queue.push_batch(batch, CAPACITY + 1, core_masks));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 3);
    for (uint64_t i = 0; i < CAPACITY; ++i)
        EXPECT_EQ(queue.pop(), &left);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}

TEST_F(MixReadyQueueTest, InFlightReservationAndResetKeepMasksVisible) {
    ChipTaskSlotState left;
    left.active_mask = ActiveMask(3);
    {
        ChipReadyQueue::PushReservation in_flight(queue, &left, 3);
        EXPECT_EQ(queue.size(), 0);
        EXPECT_EQ(queue.reserved_mix_core_mask(), 3);
    }
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
    ASSERT_TRUE(queue.push(&left, 3));
    queue.reset_for_reuse();
    EXPECT_EQ(queue.reserved_mix_core_mask(), 3);
    EXPECT_EQ(queue.pop(), &left);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}

TEST_F(MixReadyQueueTest, BatchWrapKeepsPointerTagsAndMaskCounts) {
    ChipTaskSlotState left, right;
    left.active_mask = ActiveMask(3);
    right.active_mask = ActiveMask(5);
    ChipTaskSlotState *batch[] = {&left, &right};
    TaskId tags[] = {TaskId::make(0, 3), TaskId::make(0, 4)};
    uint8_t core_masks[] = {3, 5};
    for (int round = 0; round < 50; ++round) {
        ASSERT_TRUE(queue.push_batch_tagged(batch, tags, 2, core_masks));
        EXPECT_EQ(queue.reserved_mix_core_mask(), 7);
        ChipTaskSlotState *out[2];
        TaskId out_tags[2];
        uint8_t out_masks[2];
        ASSERT_EQ(queue.pop_batch_tagged(out, out_tags, 2, out_masks), 2);
        EXPECT_EQ(out[0], &left);
        EXPECT_EQ(out[1], &right);
        EXPECT_EQ(out_tags[0], tags[0]);
        EXPECT_EQ(out_tags[1], tags[1]);
        EXPECT_EQ(out_masks[0], 3);
        EXPECT_EQ(out_masks[1], 5);
        EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
    }
}

TEST_F(MixReadyQueueTest, BatchReservationLeavesUnusedLaneAvailableBeforeTicket) {
    ChipTaskSlotState left;
    left.active_mask = ActiveMask(3);
    ChipTaskSlotState *items[] = {&left, &left};
    uint8_t core_masks[] = {3, 3};
    // Park the producer in preflight so the reservation is observable before
    // any cell publication. Restore the empty sequence before it takes a ticket.
    queue.slots[0].sequence.store(1, std::memory_order_release);
    bool pushed = false;
    std::thread producer([&] {
        pushed = queue.push_batch(items, 2, core_masks);
    });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    uint8_t reserved = 0;
    while ((reserved = queue.reserved_mix_core_mask()) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_EQ(reserved, 3);
    EXPECT_EQ(queue.mix_mask_counts[0].load(), 2);
    // The queue transports the captured mask even after descriptor reuse.
    left.active_mask = ActiveMask(2);
    queue.slots[0].sequence.store(0, std::memory_order_release);
    producer.join();
    EXPECT_TRUE(pushed);
    EXPECT_EQ(queue.pop(), &left);
    EXPECT_EQ(queue.pop(), &left);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}

TEST_F(MixReadyQueueTest, ConcurrentProducersAndConsumersReleaseAllMasksExactlyOnce) {
    constexpr int total = 4096;
    std::vector<ChipTaskSlotState> items(total);
    std::vector<std::atomic<int>> seen(total);
    constexpr uint8_t masks[] = {3, 5, 6, 7};
    for (int i = 0; i < total; ++i) {
        items[i].active_mask = ActiveMask(masks[i % 4]);
        seen[i].store(0);
    }
    std::atomic<int> done{0};
    std::atomic<int> consumed{0};
    std::vector<std::thread> threads;
    for (int producer = 0; producer < 4; ++producer)
        threads.emplace_back([&, producer] {
            int end = (producer + 1) * (total / 4);
            for (int i = producer * (total / 4); i < end;) {
                int count = producer & 1 ? std::min(3, end - i) : 1;
                ChipTaskSlotState *batch[3];
                uint8_t batch_masks[3];
                for (int j = 0; j < count; ++j) {
                    batch[j] = &items[i + j];
                    batch_masks[j] = masks[(i + j) % 4];
                }
                while (!queue.push_batch(batch, count, batch_masks))
                    std::this_thread::yield();
                i += count;
            }
            done.fetch_add(1, std::memory_order_release);
        });
    for (int consumer = 0; consumer < 4; ++consumer)
        threads.emplace_back([&] {
            while (done.load(std::memory_order_acquire) != 4 || queue.size() != 0) {
                ChipTaskSlotState *out[3];
                int count = queue.pop_batch(out, 3);
                for (int i = 0; i < count; ++i) {
                    seen[out[i] - items.data()].fetch_add(1);
                    consumed.fetch_add(1);
                }
                if (count == 0) std::this_thread::yield();
            }
        });
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(consumed.load(), total);
    for (auto &count : seen)
        EXPECT_EQ(count.load(), 1);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
    for (auto &count : queue.mix_mask_counts)
        EXPECT_EQ(count.load(), 0);
}

TEST_F(MixReadyQueueTest, MaskSnapshotsRoundTripWithoutDereferencingOpaqueTokens) {
    // Only transport is exercised: these aligned tokens are never admitted as
    // tasks. Queue operations must not read the memory they point at.
    auto *token = reinterpret_cast<ChipTaskSlotState *>(uintptr_t{0x10000});
    TaskId original = TaskId::make(0, 8);
    ASSERT_TRUE(queue.push_tagged(token, original, 3));
    TaskId tag;
    uint8_t mask = 0;
    EXPECT_EQ(queue.pop_tagged(&tag, &mask), token);
    EXPECT_EQ(tag, original);
    EXPECT_EQ(mask, 3);
    ChipTaskSlotState *tokens[] = {token, token};
    TaskId tags[] = {tag, TaskId::make(0, 9)};
    uint8_t masks[] = {mask, 5};
    ASSERT_TRUE(queue.push_batch_tagged(tokens, tags, 2, masks));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 7);
    ChipTaskSlotState *out[2];
    TaskId out_tags[2];
    uint8_t out_masks[2];
    ASSERT_EQ(queue.pop_batch_tagged(out, out_tags, 2, out_masks), 2);
    EXPECT_EQ(out[0], token);
    EXPECT_EQ(out[1], token);
    EXPECT_EQ(out_masks[0], 3);
    EXPECT_EQ(out_masks[1], 5);
    EXPECT_EQ(out_tags[0], tags[0]);
    EXPECT_EQ(out_tags[1], tags[1]);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}

#if SIMPLER_SCHED_PROFILING
TEST_F(MixReadyQueueTest, ProfiledPushAndPopReleaseMaskReservations) {
    ChipTaskSlotState item;
    item.active_mask = ActiveMask(5);
    uint64_t atomics = 0, wait = 0;
    ASSERT_TRUE(queue.push(&item, atomics, wait, 5));
    EXPECT_EQ(queue.reserved_mix_core_mask(), 5);
    EXPECT_EQ(queue.pop(atomics, wait), &item);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
    ASSERT_TRUE(queue.push(&item, atomics, wait, 5));
    ChipTaskSlotState *out[1];
    EXPECT_EQ(queue.pop_batch(out, 1, atomics, wait), 1);
    EXPECT_EQ(out[0], &item);
    EXPECT_EQ(queue.reserved_mix_core_mask(), 0);
}
#endif
