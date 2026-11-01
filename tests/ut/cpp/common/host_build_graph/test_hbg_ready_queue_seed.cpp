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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "host_build_graph/host/orchestrator.h"
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
    populations.add_task(LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false), predicate, TaskKind::KERNEL, 5);
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

TEST(HbgReadyQueueSeed, DirtyDeviceRebindPreservesThePopulatedGraphAndExternalInout) {
    constexpr uint64_t task_capacity = 8;
    DeviceArena arena;
    auto layout = runtime_reserve_layout(arena, task_capacity);
    for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
        layout.sched.capacities.ready[shape] = CAPACITY;
        layout.sched.capacities.ready_sync[shape] = CAPACITY;
    }
    layout.sched.capacities.dummy = CAPACITY;
    layout.sched.capacities.graph_prepare = CAPACITY;
    const size_t sm_bytes = SharedMemoryHandle::calculate_size(task_capacity);
    const size_t sm_offset = arena.reserve(sm_bytes, CHIP_ALIGN_SIZE);
    ASSERT_NE(arena.commit(), nullptr);
    auto *sm = static_cast<uint8_t *>(arena.region_ptr(sm_offset));
    std::memset(sm, 0, sm_bytes);
    SharedMemoryHandle host_handle{};
    ASSERT_TRUE(host_handle.init(sm, sm_bytes, task_capacity));

    std::array<float, 16> inout{};
    alignas(CHIP_ALIGN_SIZE) std::array<char, 4096> output_heap{};
    uint32_t shape[] = {static_cast<uint32_t>(inout.size())};
    auto tensor = simpler::hbg::make_tensor_external(inout.data(), shape, 1, DataType::FLOAT32);
    OrchestratorState orch{};
    ASSERT_TRUE(orch.init(sm, output_heap.data(), output_heap.size(), task_capacity));
    orch.begin_scope();
    CoreTaskArgs args;
    args.add_inout(tensor);
    args.add_scalar(uint64_t{73});
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    const auto task = orch.submit_task(kernels, args);
    ASSERT_TRUE(task.task_id().is_valid());
    TensorCreateInfo created(shape, 1, DataType::FLOAT32);
    CoreTaskArgs allocation;
    allocation.add_output(created);
    const auto hidden = orch.alloc_tensors(allocation);
    ASSERT_TRUE(hidden.task_id().is_valid());
    orch.end_scope();
    auto &tasks = host_handle.header->tasks;
    tasks.total_tasks = orch.task_allocator.active_count();
    ASSERT_TRUE(tasks.is_completed(hidden.task_id().local_id()));
    ASSERT_FALSE(tasks.is_completed(task.task_id().local_id()));

    auto *runtime = runtime_init_data_from_layout(arena, layout, MODE_EXECUTE, sm, sm_bytes);
    ASSERT_NE(runtime, nullptr);
    runtime->prebuilt_layout = layout;
    runtime_wire_arena_pointers(arena, layout, runtime);
    auto *copied = static_cast<uint8_t *>(arena.region_ptr(layout.off_copied_begin));
    const size_t copied_bytes = layout.off_copied_end - layout.off_copied_begin;
    const std::vector<uint8_t> copied_image(copied, copied + copied_bytes);
    const std::vector<uint8_t> graph_image(sm, sm + sm_bytes);
    const auto heap_image = output_heap;
    auto *slot = &tasks.get_slot_state_by_task_id(task.task_id().local_id());

    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        inout.fill(static_cast<float>(round + 11));
        const auto inout_before = inout;
        // This prefix has no uploaded image. The middle round inherits actual
        // queued work and a held async lock; the others model allocator bytes.
        if (round != 1) std::memset(arena.base(), 0xA5 + round, layout.off_copied_begin);
        runtime_wire_arena_pointers(arena, layout, runtime);
        std::memset(runtime->sm_handle, 0, sizeof(*runtime->sm_handle));
        ASSERT_TRUE(runtime->sm_handle->attach_populated(sm, sm_bytes, task_capacity, task_capacity, sm_bytes));
        auto &scheduler = *runtime->scheduler;
        ASSERT_TRUE(scheduler.init_data_from_layout(layout.sched, arena, sm));
        scheduler.seed_queue_slots();
        runtime->aicore_mailbox->init_empty();

        ASSERT_EQ(std::memcmp(copied, copied_image.data(), copied_bytes), 0);
        ASSERT_EQ(std::memcmp(sm, graph_image.data(), sm_bytes), 0);
        EXPECT_TRUE(tasks.is_completed(hidden.task_id().local_id()));
        EXPECT_FALSE(tasks.is_completed(task.task_id().local_id()));
        EXPECT_EQ(
            tasks.storage_at(task.task_id().local_id()).payload.tensor_data()[0].buffer.addr,
            reinterpret_cast<uint64_t>(inout.data())
        );
        EXPECT_EQ(inout, inout_before);
        EXPECT_EQ(output_heap, heap_image);
        EXPECT_EQ(scheduler.async_wait_list.count, 0);
        EXPECT_EQ(scheduler.async_wait_list.mpsc_skipped_count.load(), 0u);
        ASSERT_TRUE(scheduler.async_wait_list.try_lock());
        scheduler.async_wait_list.unlock();
        AICoreCompletionMsgView message{};
        EXPECT_FALSE(runtime->aicore_mailbox->try_pop(message));
        ASSERT_TRUE(runtime->aicore_mailbox->try_push_normal_done(task.task_id(), reinterpret_cast<uint64_t>(slot)));
        ASSERT_TRUE(runtime->aicore_mailbox->try_pop(message));
        EXPECT_EQ(message.task_token, task.task_id());

        std::vector<ChipReadyQueue *> queues;
        for (int resource = 0; resource < NUM_RESOURCE_SHAPES; ++resource) {
            queues.push_back(&scheduler.ready_queues[resource]);
            queues.push_back(&scheduler.ready_sync_queues[resource]);
            queues.push_back(&scheduler.early_dispatch_queues[resource]);
        }
        queues.push_back(&scheduler.dummy_ready_queue);
        queues.push_back(&scheduler.graph_prepare_queue);
        queues.push_back(&scheduler.early_sync_start_queue);
        queues.push_back(&scheduler.ed_publish_drain_queue);
        for (size_t index = 0; index < queues.size(); ++index) {
            const auto begin = reinterpret_cast<uintptr_t>(queues[index]->slots);
            const auto end = begin + queues[index]->capacity * sizeof(ChipReadyQueueSlot);
            for (size_t other = 0; other < index; ++other) {
                const auto other_begin = reinterpret_cast<uintptr_t>(queues[other]->slots);
                const auto other_end = other_begin + queues[other]->capacity * sizeof(ChipReadyQueueSlot);
                ASSERT_TRUE(end <= other_begin || other_end <= begin) << index << " overlaps " << other;
            }
        }
        for (size_t index = 0; index < queues.size(); ++index) {
            SCOPED_TRACE(index);
            auto &queue = *queues[index];
            for (uint64_t position = 0; position < CAPACITY; ++position)
                ASSERT_TRUE(queue.push(slot)) << position;
        }
        // Every queue is occupied simultaneously: aliased slot arrays cannot
        // pass by having their contents drained before the next queue runs.
        for (size_t index = 0; index < queues.size(); ++index) {
            SCOPED_TRACE(index);
            auto &queue = *queues[index];
            for (uint64_t position = 0; position < CAPACITY; ++position)
                ASSERT_EQ(queue.pop(), slot) << position;
            ASSERT_EQ(queue.pop(), nullptr);
            // A predecessor need not have left a drained queue or async list.
            ASSERT_TRUE(queue.push(slot));
        }
        scheduler.async_wait_list.count = 2;
        ASSERT_TRUE(scheduler.async_wait_list.try_lock());
        ASSERT_TRUE(runtime->aicore_mailbox->try_push_normal_done(task.task_id(), reinterpret_cast<uint64_t>(slot)));
        EXPECT_EQ(std::memcmp(copied, copied_image.data(), copied_bytes), 0);
        EXPECT_EQ(std::memcmp(sm, graph_image.data(), sm_bytes), 0);
        EXPECT_EQ(inout, inout_before);
        EXPECT_EQ(output_heap, heap_image);
    }
}
