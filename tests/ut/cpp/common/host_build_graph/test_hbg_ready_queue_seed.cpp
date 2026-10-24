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

#include "host_build_graph/host/ready_queue_sizing.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/runtime_core.h"

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

    populations.add_task(LaneMask::aic(), TaskAttrs{}, TaskKind::KERNEL, 9400);
    populations.add_task(LaneMask::aiv0(), sync_start, TaskKind::KERNEL, 3);
    populations.add_task(LaneMask::aic() | LaneMask::aiv0(), predicate, TaskKind::KERNEL, 5);
    populations.add_task(LaneMask{}, TaskAttrs{}, TaskKind::DUMMY);
    populations.add_task(LaneMask::aiv1(), predicated_sync_start, TaskKind::KERNEL, 3);
    populations.add_task(LaneMask{}, TaskAttrs{}, TaskKind::GRAPH, 7);

    ReadyQueueCapacities capacities{};
    ASSERT_TRUE(populations.derive_capacities(&capacities));
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::AIC)], 16384);
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::AIV)], 2);
    EXPECT_EQ(capacities.ready[static_cast<int32_t>(ResourceShape::MIX)], 8);
    EXPECT_EQ(capacities.ready_sync[static_cast<int32_t>(ResourceShape::AIV)], 8);
    EXPECT_EQ(capacities.dummy, 16);
    EXPECT_EQ(capacities.graph_prepare, 16);
}

TEST(HbgReadyQueueSizing, RejectsPopulationPastReservationLimit) {
    ReadyQueuePopulations populations{};
    populations.add_task(LaneMask::aic(), TaskAttrs{}, TaskKind::KERNEL, READY_QUEUE_CAPACITY_LIMIT + 1);

    ReadyQueueCapacities capacities{};
    EXPECT_FALSE(populations.derive_capacities(&capacities));
}

TEST(HbgReadyQueueSizing, RejectsMergedPopulationPastReservationLimit) {
    ReadyQueuePopulations first{};
    ReadyQueuePopulations second{};
    first.add_task(LaneMask::aic(), TaskAttrs{}, TaskKind::KERNEL, 20000);
    second.add_task(LaneMask::aic(), TaskAttrs{}, TaskKind::KERNEL, 20000);

    first.add(second);

    ReadyQueueCapacities capacities{};
    EXPECT_FALSE(first.derive_capacities(&capacities));
}

TEST(HbgReadyQueueSizing, BindRejectionReturnsReadyQueueOverflow) {
    ReadyQueuePopulations populations{};
    populations.add_task(LaneMask::aic(), TaskAttrs{}, TaskKind::KERNEL, READY_QUEUE_CAPACITY_LIMIT + 1);
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
    layout.capacities.graph_prepare = 512;
    SchedulerState scheduler{};

    ASSERT_TRUE(scheduler.init_data_from_layout(layout, scheduler_arena, sm_handle->header));

    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        EXPECT_EQ(scheduler.ready_queues[i].capacity, layout.capacities.ready[i]);
        EXPECT_EQ(scheduler.ready_sync_queues[i].capacity, layout.capacities.ready_sync[i]);
    }
    EXPECT_EQ(scheduler.dummy_ready_queue.capacity, layout.capacities.dummy);
    EXPECT_EQ(scheduler.graph_prepare_queue.capacity, layout.capacities.graph_prepare);
}

TEST(HbgReadyQueueSizing, DoublesRoundedGraphCapacityIncludingTheMinimum) {
    for (uint64_t graphs : {0u, 1u, 2u, 3u, 31u, 32u, 33u, 40u, 16384u, 16385u, 32768u}) {
        ReadyQueuePopulations populations{};
        populations.graph_prepare = graphs;
        populations.ready[0] = 3;
        populations.ready_sync[1] = 9;
        populations.dummy = 5;
        ReadyQueueCapacities capacities{};
        ASSERT_TRUE(populations.derive_capacities(&capacities));
        uint64_t rounded = 2;
        while (rounded < graphs)
            rounded *= 2;
        EXPECT_EQ(capacities.graph_prepare, 2 * rounded) << graphs;
        EXPECT_EQ(capacities.ready[0], 4);
        EXPECT_EQ(capacities.ready_sync[1], 16);
        EXPECT_EQ(capacities.dummy, 8);
        EXPECT_EQ(capacities.ready[1], 2);
        EXPECT_EQ(capacities.ready_sync[0], 2);
    }
}

TEST(HbgReadyQueueSizing, GraphHeadroomDoesNotRaisePopulationAcceptance) {
    ReadyQueuePopulations populations{};
    populations.graph_prepare = READY_QUEUE_CAPACITY_LIMIT;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i)
        populations.ready[i] = populations.ready_sync[i] = READY_QUEUE_CAPACITY_LIMIT;
    populations.dummy = READY_QUEUE_CAPACITY_LIMIT;
    ReadyQueueCapacities capacities{};
    ASSERT_TRUE(populations.derive_capacities(&capacities));
    EXPECT_EQ(capacities.graph_prepare, 2 * READY_QUEUE_CAPACITY_LIMIT);
    populations.graph_prepare++;
    EXPECT_EQ(derive_ready_queue_capacities(populations, &capacities), -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    populations.graph_prepare--;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        populations.ready[i]++;
        EXPECT_FALSE(populations.derive_capacities(&capacities));
        populations.ready[i]--;
        populations.ready_sync[i]++;
        EXPECT_FALSE(populations.derive_capacities(&capacities));
        populations.ready_sync[i]--;
    }
    populations.dummy++;
    EXPECT_FALSE(populations.derive_capacities(&capacities));
}

TEST(HbgReadyQueueSizing, DoubledQueueToleratesAReleaseWindowButCanStillWrap) {
    ReadyQueuePopulations populations{};
    populations.graph_prepare = 40;
    ReadyQueueCapacities capacities{};
    ASSERT_TRUE(populations.derive_capacities(&capacities));
    EXPECT_EQ(capacities.graph_prepare, 128);
    for (uint64_t attempts : {100u, 140u}) {
        const uint64_t capacity = capacities.graph_prepare;
        std::vector<ChipReadyQueueSlot> slots(capacity);
        ChipReadyQueue queue{};
        ready_queue_init_data_from_layout(&queue, capacity);
        queue.slots = slots.data();
        queue.seed_slots();
        for (int i = 0; i < 40; ++i)
            ASSERT_TRUE(queue.push(fake_slot_state(0)));
        // A consumer owns ticket 0 but has not released its sequence yet.
        queue.dequeue_pos.store(1, std::memory_order_relaxed);
        uint64_t rejected_at = 0;
        ChipTaskSlotState *pending = nullptr;
        for (uint64_t ticket = 40; ticket < attempts; ++ticket) {
            pending = queue.pop();
            ASSERT_NE(pending, nullptr);
            if (!queue.push(pending)) {
                rejected_at = ticket;
                break;
            }
            pending = nullptr;
        }
        EXPECT_EQ(rejected_at, attempts <= 128 ? 0 : 128);
        slots[0].sequence.store(static_cast<int64_t>(capacity), std::memory_order_release);
        if (pending != nullptr) ASSERT_TRUE(queue.push(pending));
        int drained = 0;
        while (queue.pop() != nullptr)
            ++drained;
        EXPECT_EQ(drained, 39);
    }
}
TEST(HbgReadyQueueSizing, EnlargesOnlyGraphReservationAndSeedsOnlyLogicalSlots) {
    DeviceArena arena;
    const auto layout = SchedulerState::reserve_layout(arena);
    const size_t slot_bytes = sizeof(ChipReadyQueueSlot);
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
        EXPECT_EQ(arena.region_size(layout.off_ready_queue_slots[i]), READY_QUEUE_CAPACITY_LIMIT * slot_bytes);
        EXPECT_EQ(arena.region_size(layout.off_ready_sync_queue_slots[i]), READY_QUEUE_CAPACITY_LIMIT * slot_bytes);
    }
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i)
        EXPECT_EQ(
            arena.region_size(layout.off_early_dispatch_queue_slots[i]), CHIP_EARLY_DISPATCH_QUEUE_SIZE * slot_bytes
        );
    EXPECT_EQ(arena.region_size(layout.off_early_sync_start_queue_slots), CHIP_EARLY_DISPATCH_QUEUE_SIZE * slot_bytes);
    EXPECT_EQ(arena.region_size(layout.off_ed_publish_drain_queue_slots), CHIP_EARLY_DISPATCH_QUEUE_SIZE * slot_bytes);
    EXPECT_EQ(arena.region_size(layout.off_dummy_ready_queue_slots), READY_QUEUE_CAPACITY_LIMIT * slot_bytes);
    ASSERT_EQ(arena.region_size(layout.off_graph_prepare_queue_slots), (2 * READY_QUEUE_CAPACITY_LIMIT) * slot_bytes);
    EXPECT_EQ(layout.off_graph_prepare_queue_slots % CHIP_ALIGN_SIZE, 0);
    const size_t end = layout.off_graph_prepare_queue_slots + (2 * READY_QUEUE_CAPACITY_LIMIT) * slot_bytes;
    const size_t following = arena.reserve(CHIP_ALIGN_SIZE, CHIP_ALIGN_SIZE);
    EXPECT_GE(following, end);
    ASSERT_NE(arena.commit(), nullptr);
    auto *slots = static_cast<ChipReadyQueueSlot *>(arena.region_ptr(layout.off_graph_prepare_queue_slots));
    std::memset(slots, 0xAA, (2 * READY_QUEUE_CAPACITY_LIMIT) * slot_bytes);
    ChipReadyQueue queue{};
    queue.slots = slots;
    bool first_seed = true;
    for (uint64_t capacity : {uint64_t{4}, uint64_t{32768}, (2 * READY_QUEUE_CAPACITY_LIMIT), uint64_t{4}}) {
        ready_queue_init_data_from_layout(&queue, capacity);
        queue.seed_slots();
        EXPECT_EQ(queue.enqueue_pos.load(), 0);
        EXPECT_EQ(queue.dequeue_pos.load(), 0);
        EXPECT_EQ(slots[capacity - 1].sequence.load(), static_cast<int64_t>(capacity - 1));
        EXPECT_TRUE(queue.push(fake_slot_state(0)));
        EXPECT_EQ(queue.pop(), fake_slot_state(0));
        if (first_seed) {
            const auto *bytes = reinterpret_cast<const uint8_t *>(&slots[4]);
            for (size_t i = 0; i < slot_bytes; ++i)
                EXPECT_EQ(bytes[i], 0xAA);
        }
        first_seed = false;
    }
}

TEST(HbgReadyQueueSizing, FullGraphSeedPreservesTheCopiedRuntimeAndSharedMemoryTail) {
    DeviceArena arena;
    const auto layout = runtime_reserve_layout(arena, 2048);
    const size_t graph_end =
        layout.sched.off_graph_prepare_queue_slots + (2 * READY_QUEUE_CAPACITY_LIMIT) * sizeof(ChipReadyQueueSlot);
    ASSERT_LE(graph_end, layout.off_copied_begin);
    EXPECT_EQ(layout.off_copied_begin, layout.off_runtime);
    EXPECT_EQ(layout.off_copied_end - layout.off_copied_begin, CHIP_ALIGN_UP(sizeof(RuntimeContext), CHIP_ALIGN_SIZE));
    EXPECT_EQ(layout.arena_size, layout.off_copied_end);
    constexpr size_t sm_tail_bytes = 2048;
    const size_t tail_offset = arena.reserve(sm_tail_bytes, CHIP_ALIGN_SIZE);
    ASSERT_EQ(tail_offset, layout.off_copied_end);
    ASSERT_NE(arena.commit(), nullptr);
    std::memset(arena.base(), 0xA5, arena.total_size());
    auto *runtime = runtime_init_data_from_layout(arena, layout, MODE_EXECUTE, nullptr, 0);
    ASSERT_NE(runtime, nullptr);
    runtime_wire_arena_pointers(arena, layout, runtime);
    EXPECT_EQ(runtime->scheduler, arena.region_ptr(layout.off_scheduler));
    EXPECT_EQ(
        runtime->scheduler->graph_prepare_queue.slots, arena.region_ptr(layout.sched.off_graph_prepare_queue_slots)
    );
    auto *copied = static_cast<uint8_t *>(arena.region_ptr(layout.off_runtime));
    const size_t copied_bytes = layout.off_copied_end - layout.off_copied_begin;
    std::vector<uint8_t> snapshot(copied, copied + copied_bytes);
    auto &queue = runtime->scheduler->graph_prepare_queue;
    ready_queue_init_data_from_layout(&queue, (2 * READY_QUEUE_CAPACITY_LIMIT));
    queue.seed_slots();
    EXPECT_EQ(
        queue.slots[(2 * READY_QUEUE_CAPACITY_LIMIT) - 1].sequence.load(),
        static_cast<int64_t>((2 * READY_QUEUE_CAPACITY_LIMIT) - 1)
    );
    EXPECT_EQ(std::memcmp(copied, snapshot.data(), copied_bytes), 0);
    const auto *tail = static_cast<const uint8_t *>(arena.region_ptr(tail_offset));
    for (size_t i = 0; i < sm_tail_bytes; ++i)
        EXPECT_EQ(tail[i], 0xA5);
}
