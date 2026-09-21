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
 * The slot array of a host_build_graph ready queue is reserved past the uploaded
 * range, so it reaches the device holding whatever the pooled allocation last
 * had. seed_slots() is what makes it an empty queue.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "host_build_graph/ready_queue_sizing.h"
#include "scheduler/scheduler.h"

namespace {

constexpr uint64_t CAPACITY = 8;

// GraphExecutionBatch16Seq3500 in qwen3_14b_decode reaches 9400 AIC
// tasks in one bind, so the supported ceiling must continue to cover it.
constexpr uint64_t QWEN_AIC_REACHABLE_TASKS = 9400;
static_assert(READY_QUEUE_CAPACITY_LIMIT >= QWEN_AIC_REACHABLE_TASKS);

// A slots region under the test's control, filled with a pattern standing in for
// unseeded device memory. 0 is deliberate for one case: zeroed memory is the
// tempting assumption, and it is the one that silently behaves like a full queue.
class SlotsRegion {
public:
    explicit SlotsRegion(uint8_t fill) :
        storage_(CAPACITY * sizeof(ChipReadyQueueSlot)) {
        std::memset(storage_.data(), fill, storage_.size());
    }

    ChipReadyQueueSlot *slots() { return reinterpret_cast<ChipReadyQueueSlot *>(storage_.data()); }

private:
    std::vector<std::byte> storage_;
};

// ChipReadyQueue holds atomics and so is neither copyable nor movable: the caller
// owns the storage and this only fills it, exactly as the arena path does.
void make_queue(ChipReadyQueue *queue, SlotsRegion &region) {
    ready_queue_init_data_from_layout(queue, CAPACITY);
    queue->slots = region.slots();
}

// Distinct non-null slot_state values; the queue only moves the pointer around.
ChipTaskSlotState *fake_slot_state(size_t i) {
    static ChipTaskSlotState states[CAPACITY * 2];
    return &states[i];
}

}  // namespace

// The header alone does not make a usable queue: push claims slots[pos] only when
// its sequence already equals pos, so on zeroed memory position 0 happens to
// match and every later position reads a lower sequence, which is the full-queue
// signal. One push then succeeds and the rest report full.
TEST(HbgReadyQueueSeed, ZeroedSlotsReportFullAfterOnePush) {
    SlotsRegion region(0x00);
    ChipReadyQueue queue{};
    make_queue(&queue, region);

    EXPECT_TRUE(queue.push(fake_slot_state(0)));
    EXPECT_FALSE(queue.push(fake_slot_state(1)));
    EXPECT_FALSE(queue.push(fake_slot_state(2)));
}

TEST(HbgReadyQueueSeed, SeededSlotsAcceptCapacityPushes) {
    SlotsRegion region(0xAA);
    ChipReadyQueue queue{};
    make_queue(&queue, region);

    queue.seed_slots();

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_TRUE(queue.push(fake_slot_state(i))) << "push " << i;
    }
    // Capacity is a hard bound: a full queue rejects rather than overwrites.
    EXPECT_FALSE(queue.push(fake_slot_state(CAPACITY)));

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_EQ(queue.pop(), fake_slot_state(i)) << "pop " << i;
    }
    EXPECT_EQ(queue.pop(), nullptr);
}

// The ramp is a once-per-region cost, not a per-run one: pop releases a slot with
// the sequence the next lap's push expects, so a drained queue keeps accepting
// work across laps with no re-seed.
TEST(HbgReadyQueueSeed, DrainedQueueWrapsWithoutReseeding) {
    SlotsRegion region(0xAA);
    ChipReadyQueue queue{};
    make_queue(&queue, region);
    queue.seed_slots();

    for (uint64_t lap = 0; lap < 4; lap++) {
        for (uint64_t i = 0; i < CAPACITY; i++) {
            ASSERT_TRUE(queue.push(fake_slot_state(i))) << "lap " << lap << " push " << i;
        }
        for (uint64_t i = 0; i < CAPACITY; i++) {
            ASSERT_EQ(queue.pop(), fake_slot_state(i)) << "lap " << lap << " pop " << i;
        }
    }

    EXPECT_EQ(queue.max_occupancy.load(std::memory_order_relaxed), CAPACITY);
}

// Seeding is unconditional on attach, so it must also recover a region left
// mid-lap by an earlier run rather than assuming a drained predecessor.
TEST(HbgReadyQueueSeed, ReseedRecoversAPartiallyUsedRegion) {
    SlotsRegion region(0xAA);
    ChipReadyQueue queue{};
    make_queue(&queue, region);
    queue.seed_slots();
    ASSERT_TRUE(queue.push(fake_slot_state(0)));
    ASSERT_TRUE(queue.push(fake_slot_state(1)));

    // What a fresh bind does: the uploaded header returns to zeroed positions and
    // the device re-establishes the lap-0 ramp underneath it.
    ready_queue_init_data_from_layout(&queue, CAPACITY);
    queue.seed_slots();

    for (uint64_t i = 0; i < CAPACITY; i++) {
        EXPECT_TRUE(queue.push(fake_slot_state(i))) << "push " << i;
    }
    EXPECT_EQ(queue.pop(), fake_slot_state(0));
}

TEST(HbgReadyQueueSizing, DerivesCapacityForEachReachablePopulation) {
    ReadyQueuePopulations populations{};
    TaskAttrs sync_start;
    sync_start.set_sync_start();
    TaskAttrs predicate;
    predicate.set_predicate();
    TaskAttrs predicated_sync_start;
    predicated_sync_start.set_predicate();
    predicated_sync_start.set_sync_start();

    populations.add_task(ActiveMask(SUBTASK_MASK_AIC), TaskAttrs{}, TaskKind::KERNEL, 9400);
    populations.add_task(ActiveMask(SUBTASK_MASK_AIV0), sync_start, TaskKind::KERNEL, 3);
    populations.add_task(ActiveMask(SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0), predicate, TaskKind::KERNEL, 5);
    populations.add_task(ActiveMask{}, TaskAttrs{}, TaskKind::DUMMY);
    populations.add_task(ActiveMask(SUBTASK_MASK_AIV1), predicated_sync_start, TaskKind::KERNEL, 3);
    populations.add_task(ActiveMask{}, TaskAttrs{}, TaskKind::GRAPH, 7);

    ReadyQueueCapacities capacities{};
    ASSERT_TRUE(populations.derive_capacities(&capacities));
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::AIC)], 16384);
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::AIV)], 2);
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::MIX)], 8);
    EXPECT_EQ(capacities.ready_sync[static_cast<int32_t>(ResourceShape::AIV)], 8);
    EXPECT_EQ(capacities.dummy, 16);
    EXPECT_EQ(capacities.graph_ready, 8);
    EXPECT_EQ(capacities.graph_prepare, 8);
}

TEST(HbgReadyQueueSizing, RejectsPopulationPastReservationLimit) {
    ReadyQueuePopulations populations{};
    populations.add_task(ActiveMask(SUBTASK_MASK_AIC), TaskAttrs{}, TaskKind::KERNEL, READY_QUEUE_CAPACITY_LIMIT + 1);

    ReadyQueueCapacities capacities{};
    EXPECT_FALSE(populations.derive_capacities(&capacities));
}

TEST(HbgReadyQueueSizing, RejectsMergedPopulationPastReservationLimit) {
    ReadyQueuePopulations first{};
    ReadyQueuePopulations second{};
    first.add_task(ActiveMask(SUBTASK_MASK_AIC), TaskAttrs{}, TaskKind::KERNEL, 20000);
    second.add_task(ActiveMask(SUBTASK_MASK_AIC), TaskAttrs{}, TaskKind::KERNEL, 20000);

    first.add(second);

    ReadyQueueCapacities capacities{};
    EXPECT_FALSE(first.derive_capacities(&capacities));
}

TEST(HbgReadyQueueSizing, BindRejectionReturnsReadyQueueOverflow) {
    ReadyQueuePopulations populations{};
    populations.add_task(ActiveMask(SUBTASK_MASK_AIC), TaskAttrs{}, TaskKind::KERNEL, READY_QUEUE_CAPACITY_LIMIT + 1);
    ReadyQueueCapacities capacities{};

    const int32_t status = derive_ready_queue_capacities(populations, &capacities);

    EXPECT_EQ(status, -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(HbgReadyQueueSizing, InitializesEveryLogicalQueueCapacityFromLayout) {
    DeviceArena scheduler_arena;
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm_handle, nullptr);
    SchedulerLayout layout = SchedulerState::reserve_layout(scheduler_arena);
    layout.capacities.ready[0] = 2;
    layout.capacities.ready[1] = 4;
    layout.capacities.ready[2] = 8;
    layout.capacities.ready_sync[0] = 16;
    layout.capacities.ready_sync[1] = 32;
    layout.capacities.ready_sync[2] = 64;
    layout.capacities.dummy = 128;
    layout.capacities.graph_ready = 256;
    layout.capacities.graph_prepare = 512;
    SchedulerState scheduler{};

    ASSERT_TRUE(scheduler.init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        EXPECT_EQ(scheduler.ready_queues[i].capacity, layout.capacities.ready[i]);
        EXPECT_EQ(scheduler.ready_sync_queues[i].capacity, layout.capacities.ready_sync[i]);
    }
    EXPECT_EQ(scheduler.dummy_ready_queue.capacity, layout.capacities.dummy);
    EXPECT_EQ(scheduler.graph_ready_queue.capacity, layout.capacities.graph_ready);
    EXPECT_EQ(scheduler.graph_prepare_queue.capacity, layout.capacities.graph_prepare);
}
