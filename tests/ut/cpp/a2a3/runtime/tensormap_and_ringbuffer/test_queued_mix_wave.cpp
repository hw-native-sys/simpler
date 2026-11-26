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

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <thread>
#include <vector>

#include "callable.h"
#include "scheduler/scheduler_context.h"

namespace {
constexpr int owners = 3;
constexpr int clusters = 5;
constexpr int lanes = PLATFORM_CORES_PER_BLOCKDIM;
}  // namespace

class SchedulerContextTestPeer {
public:
    static void
    bind(SchedulerContext &ctx, SchedulerState &scheduler, uint64_t *callables, uint64_t *regs, size_t stride) {
        ctx.sched_ = &scheduler;
        ctx.active_sched_threads_ = owners;
        ctx.cores_total_num_ = clusters * lanes;
        ctx.func_id_to_addr_ = callables;
        ctx.func_id_to_addr_count_ = 1;
        for (int owner = 0; owner < owners; ++owner) {
            auto &tracker = ctx.core_trackers_[owner];
            tracker.init((clusters + owners - 1 - owner) / owners);
            for (int c = owner; c < clusters; c += owners)
                tracker.set_cluster(c / owners, c, clusters + 2 * c, clusters + 2 * c + 1);
        }
        for (int c = 0; c < ctx.cores_total_num_; ++c) {
            ctx.retirement_state_[c] = SchedulerContext::RETIREMENT_READY;
            auto &core = ctx.core_exec_states_[c];
            core = {};
            core.reg_addr = reinterpret_cast<uint64_t>(regs + c * stride);
            core.running_reg_task_id = core.pending_reg_task_id = AICPU_TASK_INVALID;
        }
    }

    static bool first(SchedulerContext &ctx) {
        bool local = false;
        return ctx.allow_first_queue(0, local);
    }
    static bool request(SchedulerContext &ctx, ChipTaskSlotState &slot) {
        return ctx.enter_ready_drain(&slot, slot.logical_block_num);
    }
    static int service(SchedulerContext &ctx, int owner) { return ctx.service_queued_wave(owner); }
    static void dispatch(SchedulerContext &ctx, int owner) {
        bool progress = false;
        bool pushed = false;
        ctx.dispatch_ready_tasks(owner, ctx.core_trackers_[owner], false, progress, pushed);
    }
    static int pending(SchedulerContext &ctx) { return ctx.drain_state_.sync_start_pending.load(); }
    static void mark_running(SchedulerContext &ctx, ChipTaskSlotState *slot, bool pending = false) {
        ctx.core_trackers_[0].change_core_state(0);
        ctx.core_exec_states_[0].running_slot_state = slot;
        if (pending) ctx.core_exec_states_[0].pending_slot_state = slot;
    }
    static int placed_block(SchedulerContext &ctx, int core, ChipTaskSlotState &slot) {
        const auto &state = ctx.core_exec_states_[core];
        if (state.running_slot_state != &slot) return -1;
        return ctx.payload_per_core_[core][state.running_reg_task_id & 1].local_context.block_idx;
    }
};

namespace {
using Peer = SchedulerContextTestPeer;

class QueuedMixWave : public testing::Test {
protected:
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> ctx = std::make_unique<SchedulerContext>();
    DeviceArena shared_arena;
    DeviceArena scheduler_arena;
    TaskDescriptor task{};
    TaskPayload payload{};
    ChipTaskSlotState slot{};
    std::vector<uint8_t> callable = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t functions[1]{};
    std::vector<uint64_t> registers;

    void SetUp() override {
        auto *shared = SharedMemoryHandle::create_and_init_default(shared_arena);
        ASSERT_NE(shared, nullptr);
        auto layout = SchedulerState::reserve_layout(scheduler_arena);
        ASSERT_NE(scheduler_arena.commit(), nullptr);
        ASSERT_TRUE(scheduler->init_data_from_layout(layout, scheduler_arena, shared->header));
        scheduler->wire_arena_pointers(layout, scheduler_arena);
        reinterpret_cast<CoreCallable *>(callable.data())->set_resolved_addr(1);
        functions[0] = reinterpret_cast<uint64_t>(callable.data());
        const size_t words = reg_offset(RegId::COND) / sizeof(uint64_t) + 1;
        registers.resize(clusters * lanes * words);
        Peer::bind(*ctx, *scheduler, functions, registers.data(), words);
        slot.task = &task;
        slot.payload = &payload;
        slot.active_mask = LaneMask::all();
        slot.logical_block_num = clusters;
        slot.task_attrs.set_early_resolve(true);
        task.task_id = TaskId::make(0, 0);
        for (auto &id : task.kernel_id)
            id = 0;
    }
    void TearDown() override { scheduler->destroy(); }

    std::array<int, owners> publish() {
        std::array<int, owners> published{};
        std::array<std::thread, owners> threads;
        for (int owner = 0; owner < owners; ++owner) {
            threads[owner] = std::thread([this, owner, &published] {
                while (Peer::pending(*ctx) != 0)
                    published[owner] += Peer::service(*ctx, owner);
            });
        }
        for (auto &thread : threads)
            thread.join();
        return published;
    }
};

TEST_F(QueuedMixWave, FirstWideMixIssuesDisjointRangesWithPartialLanes) {
    slot.logical_block_num = 4;
    slot.active_mask = LaneMask(true, false, true);
    slot.total_required_subtasks = 8;
    task.kernel_id[1] = INVALID_KERNEL_ID;
    auto &q = scheduler->ready_queues[static_cast<int>(ResourceShape::MIX)];
    ASSERT_TRUE(q.push_mix(&slot));
    EXPECT_FALSE(Peer::first(*ctx));
    ASSERT_EQ(q.size(), 0u);
    ASSERT_EQ(slot.next_block_idx.load(), 4);
    ASSERT_FALSE(slot.has_dispatch_propagated());
    EXPECT_EQ(publish(), (std::array<int, owners>{2, 1, 1}));
    for (int owner = 0; owner < owners; ++owner)
        EXPECT_EQ(Peer::service(*ctx, owner), 0);
    constexpr std::array<int, clusters> expected{0, 2, 3, 1, -1};
    for (int c = 0; c < clusters; ++c) {
        EXPECT_EQ(Peer::placed_block(*ctx, c, slot), expected[c]);
        EXPECT_EQ(Peer::placed_block(*ctx, clusters + c * 2, slot), -1);
        EXPECT_EQ(Peer::placed_block(*ctx, clusters + c * 2 + 1, slot), expected[c]);
    }
    EXPECT_EQ(payload.published_block_count.load(), 4);
    EXPECT_TRUE(slot.has_dispatch_propagated());
    EXPECT_TRUE(Peer::first(*ctx));
}

TEST_F(QueuedMixWave, FullSyncGroupClaimsOnlyAfterOccupiedOwnerReleasesItsLane) {
    slot.task_attrs.set_sync_start();
    slot.total_required_subtasks = clusters * lanes;
    TaskPayload predecessor_payload{};
    ChipTaskSlotState predecessor{};
    predecessor.payload = &predecessor_payload;
    predecessor.logical_block_num = 1;
    Peer::mark_running(*ctx, &predecessor);
    ASSERT_TRUE(Peer::request(*ctx, slot));
    for (int owner = 0; owner < owners; ++owner)
        EXPECT_EQ(Peer::service(*ctx, owner), 0);
    ASSERT_EQ(slot.next_block_idx.load(), 0);
    EXPECT_FALSE(slot.has_dispatch_propagated());
    Peer::mark_running(*ctx, nullptr);
    EXPECT_EQ(publish(), (std::array<int, owners>{2, 2, 1}));
    EXPECT_EQ(slot.next_block_idx.load(), clusters);
    EXPECT_EQ(payload.published_block_count.load(), clusters);
    EXPECT_TRUE(slot.has_dispatch_propagated());
}

TEST_F(QueuedMixWave, PendingLaneHandsUnclaimedTaskToCapacityDrain) {
    slot.task_attrs.set_sync_start();
    TaskPayload predecessor_payload{};
    ChipTaskSlotState predecessor{};
    predecessor.payload = &predecessor_payload;
    predecessor.logical_block_num = 1;
    Peer::mark_running(*ctx, &predecessor, true);
    ASSERT_TRUE(Peer::request(*ctx, slot));
    EXPECT_EQ(Peer::service(*ctx, 0), 0);
    EXPECT_EQ(Peer::pending(*ctx), clusters);
    EXPECT_EQ(slot.next_block_idx.load(), 0);
    EXPECT_EQ(payload.published_block_count.load(), 0);
}

TEST_F(QueuedMixWave, UnpublishedSyncQueueHeadAllowsOrdinaryMixToDispatch) {
    const int mix = static_cast<int>(ResourceShape::MIX);
    auto &sync = scheduler->ready_sync_queues[mix];
    auto &ordinary = scheduler->ready_queues[mix];
    auto pos = sync.enqueue_pos.fetch_add(1);
    auto &cell = sync.slots[pos & sync.mask];
    slot.logical_block_num = 1;
    slot.total_required_subtasks = lanes;
    ASSERT_TRUE(ordinary.push_mix(&slot));
    ASSERT_TRUE(Peer::first(*ctx));
    Peer::dispatch(*ctx, 0);
    EXPECT_EQ(payload.published_block_count.load(), 1);
    EXPECT_EQ(ordinary.size(), 0u);
    EXPECT_EQ(sync.dequeue_pos.load(), pos);

    TaskDescriptor next_task{};
    next_task.task_id = TaskId::make(0, 1);
    for (auto &id : next_task.kernel_id)
        id = 0;
    TaskPayload next_payload{};
    ChipTaskSlotState next{};
    next.task = &next_task;
    next.payload = &next_payload;
    next.active_mask = LaneMask::all();
    next.logical_block_num = 1;
    next.total_required_subtasks = lanes;
    next.task_attrs.set_sync_start();
    next.task_attrs.set_early_resolve(true);
    cell.slot_value = CapturedReadySlot(&next, next.active_mask);
    cell.task_id_snapshot = next_task.task_id;
    sync.mixed_lanes.add(next.active_mask);
    cell.sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
    Peer::dispatch(*ctx, 1);
    EXPECT_EQ(sync.size(), 0u);
    EXPECT_EQ(next_payload.published_block_count.load(), 1);
    EXPECT_EQ(payload.published_block_count.load(), 1);
}
}  // namespace
