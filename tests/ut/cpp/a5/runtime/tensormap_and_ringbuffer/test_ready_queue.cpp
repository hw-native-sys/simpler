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

namespace {
constexpr LaneMask kAicAiv0{true, true, false};
constexpr LaneMask kAicAiv1{true, false, true};
}  // namespace

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

TEST_F(ReadyQueueTest, RequeueBatchThenIndividualPop) {
    constexpr int BATCH_SIZE = 5;
    ChipTaskSlotState items[BATCH_SIZE];
    LaneTaggedSlot handles[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) {
        handles[i] = queue.capture(&items[i]);
    }

    ASSERT_TRUE(queue.requeue_batch(handles, BATCH_SIZE));

    for (int i = 0; i < BATCH_SIZE; i++) {
        EXPECT_EQ(queue.pop(), &items[i]);
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, RequeueBatchZeroIsNoop) {
    EXPECT_TRUE(queue.requeue_batch(nullptr, 0));

    EXPECT_EQ(queue.size(), 0u);
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST_F(ReadyQueueTest, PopBatchReturnsFive) {
    constexpr int PUSH_COUNT = 10;
    ChipTaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    LaneTaggedSlot out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, 5);

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(out[i].ptr(), &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchPartial) {
    constexpr int PUSH_COUNT = 3;
    ChipTaskSlotState items[PUSH_COUNT];

    for (int i = 0; i < PUSH_COUNT; i++) {
        ASSERT_TRUE(queue.push(&items[i]));
    }

    LaneTaggedSlot out[5];
    int popped = queue.pop_batch(out, 5);
    EXPECT_EQ(popped, PUSH_COUNT);

    for (int i = 0; i < PUSH_COUNT; i++) {
        EXPECT_EQ(out[i].ptr(), &items[i]);
    }
}

TEST_F(ReadyQueueTest, PopBatchEmpty) {
    LaneTaggedSlot out[5];
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
    void expect_counts(uint32_t aic, uint32_t v0, uint32_t v1) {
        EXPECT_EQ(queue.lane_demand_.count(LaneMask::aic()), aic);
        EXPECT_EQ(queue.lane_demand_.count(LaneMask::aiv0()), v0);
        EXPECT_EQ(queue.lane_demand_.count(LaneMask::aiv1()), v1);
    }
    void expect_empty() {
        EXPECT_TRUE(queue.reserved_mix_lanes().is_empty());
        expect_counts(0, 0, 0);
    }
};

TEST_F(MixReadyQueueTest, UnionTracksPublishedCells) {
    ChipTaskSlotState left{}, right{};
    left.active_mask = kAicAiv0;
    right.active_mask = kAicAiv1;
    ASSERT_TRUE(queue.push(&left));
    EXPECT_EQ(queue.reserved_mix_lanes(), left.active_mask);
    ASSERT_TRUE(queue.push(&right));
    EXPECT_TRUE(queue.reserved_mix_lanes().is_full());
    expect_counts(2, 1, 1);
    EXPECT_EQ(queue.pop(), &left);
    EXPECT_EQ(queue.reserved_mix_lanes(), right.active_mask);
    EXPECT_EQ(queue.pop(), &right);
    expect_empty();
}

TEST_F(MixReadyQueueTest, ReusedSlotConsumesOriginalCellTagAndTaskId) {
    ChipTaskSlotState slot{};
    TaskDescriptor first{}, replacement{};
    first.task_id = TaskId::make(0, 8);
    replacement.task_id = TaskId::make(0, 9);
    slot.task = &first;
    const LaneMask original = kAicAiv0;
    slot.active_mask = original;
    ASSERT_TRUE(queue.push_tagged(&slot, first.task_id));
    slot.active_mask = LaneMask::both_aiv();
    slot.task = &replacement;
    LaneTaggedSlot out[1];
    TaskId snapshots[1];
    ASSERT_EQ(queue.pop_batch_tagged(out, snapshots, 1), 1);
    EXPECT_EQ(out[0].ptr(), &slot);
    EXPECT_EQ(out[0].tag(), original);
    EXPECT_EQ(snapshots[0], first.task_id);
    EXPECT_NE(out[0].ptr()->task->task_id, snapshots[0]);
    expect_empty();
}

TEST_F(MixReadyQueueTest, RequeuePreservesOldMixTagWhenSlotBecomesPureAiv) {
    ChipTaskSlotState slot{};
    TaskDescriptor first{}, replacement{};
    first.task_id = TaskId::make(0, 12);
    replacement.task_id = TaskId::make(0, 13);
    slot.task = &first;
    const LaneMask original = kAicAiv0;
    slot.active_mask = original;
    ASSERT_TRUE(queue.push_tagged(&slot, first.task_id));
    LaneTaggedSlot handles[1];
    TaskId snapshots[1];
    ASSERT_EQ(queue.pop_batch_tagged(handles, snapshots, 1), 1);
    slot.active_mask = LaneMask::aiv1();
    slot.task = &replacement;
    ASSERT_TRUE(queue.requeue_batch_tagged(handles, snapshots, 1));
    EXPECT_EQ(queue.reserved_mix_lanes(), original);
    expect_counts(1, 1, 0);
    LaneTaggedSlot again[1];
    TaskId saved[1];
    ASSERT_EQ(queue.pop_batch_tagged(again, saved, 1), 1);
    EXPECT_EQ(again[0].ptr(), &slot);
    EXPECT_EQ(again[0].tag(), original);
    EXPECT_EQ(saved[0], first.task_id);
    EXPECT_NE(again[0].ptr()->task->task_id, saved[0]);
    expect_empty();
}

TEST_F(MixReadyQueueTest, SingleRequeueAndBatchWrapPreserveHandles) {
    ChipTaskSlotState slots[2]{};
    slots[0].active_mask = kAicAiv0;
    slots[1].active_mask = kAicAiv1;
    LaneTaggedSlot items[] = {queue.capture(&slots[0]), queue.capture(&slots[1])};
    TaskId snapshots[] = {TaskId::make(0, 1), TaskId::make(0, 2)};
    for (int round = 0; round < 64; ++round) {
        ASSERT_TRUE(queue.requeue_batch_tagged(items, snapshots, 2));
        LaneTaggedSlot out[2];
        TaskId ids[2];
        ASSERT_EQ(queue.pop_batch_tagged(out, ids, 2), 2);
        EXPECT_EQ(out[0].tag(), slots[0].active_mask);
        EXPECT_EQ(out[1].tag(), slots[1].active_mask);
        EXPECT_EQ(ids[0], snapshots[0]);
        EXPECT_EQ(ids[1], snapshots[1]);
        ASSERT_TRUE(queue.requeue_tagged(out[0], ids[0]));
        TaskId saved;
        EXPECT_EQ(queue.pop_tagged(&saved), &slots[0]);
        EXPECT_EQ(saved, snapshots[0]);
        expect_empty();
    }
}

TEST_F(MixReadyQueueTest, FailedPushAndBatchRollBackAllFields) {
    std::vector<ChipTaskSlotState> slots(CAPACITY);
    const LaneMask original = kAicAiv0;
    for (auto &slot : slots) {
        slot.active_mask = original;
        ASSERT_TRUE(queue.push(&slot));
    }
    ChipTaskSlotState extra{};
    extra.active_mask = kAicAiv1;
    EXPECT_FALSE(queue.push(&extra));
    LaneTaggedSlot batch[] = {queue.capture(&extra), queue.capture(&extra)};
    EXPECT_FALSE(queue.requeue_batch(batch, 2));
    EXPECT_FALSE(queue.requeue_batch(batch, CAPACITY + 1));
    EXPECT_EQ(queue.reserved_mix_lanes(), original);
    expect_counts(CAPACITY, CAPACITY, 0);
    for (auto &slot : slots)
        EXPECT_EQ(queue.pop(), &slot);
    expect_empty();
}

TEST_F(MixReadyQueueTest, InFlightReservationRollsBackWithoutPublication) {
    ChipTaskSlotState slot{};
    const LaneMask demand = kAicAiv1;
    {
        ChipReadyQueue::PushReservation reservation(queue, LaneTaggedSlot(&slot, demand));
        EXPECT_EQ(queue.reserved_mix_lanes(), demand);
        expect_counts(1, 0, 1);
        EXPECT_EQ(queue.size(), 0u);
    }
    expect_empty();
}

TEST_F(MixReadyQueueTest, PartialBatchPublicationReleasesOnlyConsumedCells) {
    ChipTaskSlotState slots[2]{};
    LaneTaggedSlot handles[] = {
        LaneTaggedSlot(&slots[0], kAicAiv0),
        LaneTaggedSlot(&slots[1], kAicAiv1),
    };
    ChipReadyQueue::BatchReservation reservation(queue, handles, 2);
    // A producer owns both tickets but has published only the first cell.
    queue.enqueue_pos.store(2, std::memory_order_relaxed);
    reservation.transfer_to_cells();
    queue.slots[0].slot = handles[0];
    queue.slots[0].task_id_snapshot = TaskId::make(0, 20);
    queue.slots[0].sequence.store(1, std::memory_order_release);
    LaneTaggedSlot out[2];
    ASSERT_EQ(queue.pop_batch(out, 2), 1);
    EXPECT_EQ(out[0].ptr(), &slots[0]);
    EXPECT_EQ(queue.reserved_mix_lanes(), handles[1].tag());
    expect_counts(1, 0, 1);
    queue.slots[1].slot = handles[1];
    queue.slots[1].task_id_snapshot = TaskId::make(0, 21);
    queue.slots[1].sequence.store(2, std::memory_order_release);
    ASSERT_EQ(queue.pop_batch(out, 2), 1);
    EXPECT_EQ(out[0].ptr(), &slots[1]);
    expect_empty();
}

TEST_F(MixReadyQueueTest, BatchReservesBeforeTakingTicket) {
    ChipTaskSlotState slots[2]{};
    const LaneMask demand = kAicAiv0;
    LaneTaggedSlot handles[] = {LaneTaggedSlot(&slots[0], demand), LaneTaggedSlot(&slots[1], demand)};
    {
        ChipReadyQueue::BatchReservation reservation(queue, handles, 2);
        EXPECT_EQ(queue.enqueue_pos.load(), 0u);
        EXPECT_EQ(queue.size(), 0u);
        EXPECT_EQ(queue.reserved_mix_lanes(), demand);
        expect_counts(2, 2, 0);
    }
    expect_empty();
}

TEST_F(MixReadyQueueTest, ConcurrentProducersAndConsumersReleaseAllTagsExactlyOnce) {
    constexpr int total = 4096;
    const LaneMask masks[] = {kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all()};
    std::vector<ChipTaskSlotState> items(total);
    std::vector<std::atomic<int>> seen(total);
    for (int i = 0; i < total; ++i) {
        items[i].active_mask = masks[i % 4];
        seen[i].store(0);
    }
    std::atomic<int> done{0}, consumed{0};
    std::vector<std::thread> threads;
    for (int producer = 0; producer < 4; ++producer)
        threads.emplace_back([&, producer] {
            const int end = (producer + 1) * (total / 4);
            for (int i = producer * (total / 4); i < end;) {
                const int count = producer & 1 ? std::min(3, end - i) : 1;
                LaneTaggedSlot batch[3];
                for (int j = 0; j < count; ++j)
                    batch[j] = queue.capture(&items[i + j]);
                while (!queue.requeue_batch(batch, count))
                    std::this_thread::yield();
                i += count;
            }
            done.fetch_add(1, std::memory_order_release);
        });
    for (int consumer = 0; consumer < 4; ++consumer)
        threads.emplace_back([&] {
            while (done.load(std::memory_order_acquire) != 4 || queue.size() != 0) {
                LaneTaggedSlot out[3];
                const int count = queue.pop_batch(out, 3);
                for (int i = 0; i < count; ++i) {
                    seen[out[i].ptr() - items.data()].fetch_add(1);
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
    expect_empty();
}

TEST_F(MixReadyQueueTest, RejectsSingleLaneMixAndOversizedCapacity) {
    ChipTaskSlotState slot{};
    slot.active_mask = LaneMask::aiv0();
    EXPECT_ANY_THROW(queue.push(&slot));
    expect_empty();
    EXPECT_ANY_THROW(ready_queue_init_data_from_layout(&queue, arena, 0, ChipReadyQueue::kMaxCapacity + 1, true));
}

#if SIMPLER_SCHED_PROFILING
TEST_F(MixReadyQueueTest, ProfiledPushAndPopReleaseReservations) {
    ChipTaskSlotState item{};
    item.active_mask = kAicAiv1;
    uint64_t atomics = 0, wait = 0;
    ASSERT_TRUE(queue.push(&item, atomics, wait));
    EXPECT_EQ(queue.reserved_mix_lanes(), item.active_mask);
    EXPECT_EQ(queue.pop(atomics, wait), &item);
    expect_empty();
    ASSERT_TRUE(queue.push(&item, atomics, wait));
    LaneTaggedSlot out[1];
    EXPECT_EQ(queue.pop_batch(out, 1, atomics, wait), 1);
    EXPECT_EQ(out[0].ptr(), &item);
    expect_empty();
}
#endif
