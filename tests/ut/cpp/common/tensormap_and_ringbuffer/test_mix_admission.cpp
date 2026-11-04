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

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "callable.h"
#include "scheduler/scheduler_context.h"

static_assert(std::is_default_constructible_v<CoreTracker::BitStates>);
static_assert(!std::is_constructible_v<CoreTracker::BitStates, CoreTracker::BitStates::Storage>);
static_assert(!std::is_constructible_v<CoreTracker::BitStates, uint64_t>);

class MixAdmissionTestPeer {
public:
    static void bind(
        SchedulerContext &context, SchedulerState &scheduler, uint64_t *callables, int clusters, bool pending,
        ChipTaskSlotState &blocker, uint64_t *registers, size_t register_words
    ) {
        context.sched_ = &scheduler;
        context.active_sched_threads_ = 1;
        context.func_id_to_addr_ = callables;
        context.func_id_to_addr_count_ = 1;
        auto &tracker = context.core_trackers_[0];
        tracker.init(clusters);
        for (int i = 0; i < clusters; ++i)
            tracker.set_cluster(i, 3 * i, 3 * i + 1, 3 * i + 2);
        for (int i = 0; i < 3 * clusters; ++i) {
            auto &core = context.core_exec_states_[i];
            core = {};
            core.reg_addr = reinterpret_cast<uint64_t>(registers + i * register_words);
            core.running_reg_task_id = pending ? 0 : AICPU_TASK_INVALID;
            core.pending_reg_task_id = AICPU_TASK_INVALID;
            if (pending) {
                tracker.change_core_state(i);
                core.running_slot_state = &blocker;
            }
        }
    }

    static void normal(SchedulerContext &context, CoreTracker::DispatchPhase phase) {
        bool drain = false;
        bool progress = false;
        bool pushed = false;
        dispatch(context, phase, drain, progress, pushed, 0);
        EXPECT_FALSE(drain);
        EXPECT_TRUE(progress);
        EXPECT_TRUE(pushed);
    }

    template <typename Context, typename = typename Context::MixPriority>
    static void
    dispatch(Context &context, CoreTracker::DispatchPhase phase, bool &drain, bool &progress, bool &pushed, int) {
        LaneMask observed = LaneMask::none();
        context.dispatch_shape(
            0, context.sched_->ready_queues, ResourceShape::MIX, phase, context.core_trackers_[0], drain, progress,
            pushed, Context::MixPriority::Ready, observed
        );
    }

    template <typename Context>
    static void
    dispatch(Context &context, CoreTracker::DispatchPhase phase, bool &drain, bool &progress, bool &pushed, long) {
        context.dispatch_shape(
            0, context.sched_->ready_queues, ResourceShape::MIX, phase, context.core_trackers_[0], drain, progress,
            pushed
        );
    }

    template <typename Context, typename = typename Context::MixPriority>
    static void dispatch_early(Context &context, CoreTracker::DispatchPhase phase, int) {
        LaneMask observed = LaneMask::none();
        context.early_dispatch_shape(0, ResourceShape::MIX, phase, observed);
    }

    template <typename Context>
    static void dispatch_early(Context &context, CoreTracker::DispatchPhase phase, long) {
        context.early_dispatch_shape(0, ResourceShape::MIX, phase);
    }

    static void early(SchedulerContext &context, CoreTracker::DispatchPhase phase) {
        dispatch_early(context, phase, 0);
    }

    template <typename Queue>
    static auto push_normal(Queue &queue, ChipTaskSlotState *slot, int) -> decltype(queue.push_mix(slot)) {
        return queue.push_mix(slot);
    }
    template <typename Queue>
    static bool push_normal(Queue &queue, ChipTaskSlotState *slot, long) {
        return queue.push(slot);
    }
    template <typename Queue>
    static auto push_early(Queue &queue, ChipTaskSlotState *slot, TaskId id, int)
        -> decltype(queue.push_mix(slot, id)) {
        return queue.push_mix(slot, id);
    }
    template <typename Queue>
    static bool push_early(Queue &queue, ChipTaskSlotState *slot, TaskId id, long) {
        return queue.push_tagged(slot, id);
    }

    static ChipTaskSlotState *placed(SchedulerContext &context, int core, bool pending) {
        const auto &state = context.core_exec_states_[core];
        return pending ? state.pending_slot_state : state.running_slot_state;
    }
};

using MixAdmissionConfig = std::tuple<int, uint8_t, bool>;

class MixAdmission : public testing::TestWithParam<MixAdmissionConfig> {
protected:
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    DeviceArena shared_arena;
    DeviceArena scheduler_arena;
    std::vector<uint8_t> callable_storage = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t callables[1]{};
    TaskDescriptor task{};
    TaskPayload payload{};
    ChipTaskSlotState slot{};
    ChipTaskSlotState blocker{};
    std::vector<uint64_t> registers;

    void SetUp() override {
        auto *shared = SharedMemoryHandle::create_and_init_default(shared_arena);
        ASSERT_NE(shared, nullptr);
        auto layout = SchedulerState::reserve_layout(scheduler_arena);
        ASSERT_NE(scheduler_arena.commit(), nullptr);
        ASSERT_TRUE(scheduler->init_data_from_layout(layout, scheduler_arena, shared->header));
        scheduler->wire_arena_pointers(layout, scheduler_arena);
        auto *callable = reinterpret_cast<CoreCallable *>(callable_storage.data());
        callable->set_resolved_addr(1);
        callables[0] = reinterpret_cast<uint64_t>(callable);
        const auto [clusters, mask, pending] = GetParam();
        slot.task = &task;
        slot.payload = &payload;
        slot.active_mask = LaneMask((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        slot.logical_block_num = clusters;
        slot.total_required_subtasks = clusters * __builtin_popcount(mask);
        task.task_id = TaskId::make(0, 0);
        for (int i = 0; i < SUBTASK_SLOT_COUNT; ++i)
            task.kernel_id[i] = mask & (1u << i) ? 0 : INVALID_KERNEL_ID;
        const size_t register_words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
        registers.resize(3 * clusters * register_words);
        MixAdmissionTestPeer::bind(
            *context, *scheduler, callables, clusters, pending, blocker, registers.data(), register_words
        );
    }

    void TearDown() override {
        scheduler->destroy();
        scheduler_arena.release();
        shared_arena.release();
    }

    CoreTracker::DispatchPhase phase() const {
        return std::get<2>(GetParam()) ? CoreTracker::DispatchPhase::PENDING : CoreTracker::DispatchPhase::IDLE;
    }

    void expect_all_blocks() {
        const auto [clusters, mask, pending] = GetParam();
        ASSERT_EQ(slot.next_block_idx.load(), clusters);
        for (int cluster = 0; cluster < clusters; ++cluster) {
            for (int lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
                SCOPED_TRACE(testing::Message() << "cluster=" << cluster << " lane=" << lane);
                EXPECT_EQ(
                    MixAdmissionTestPeer::placed(*context, 3 * cluster + lane, pending),
                    mask & (1u << lane) ? &slot : nullptr
                );
            }
        }
    }
};

TEST_P(MixAdmission, NormalSelectsEveryOwnedCluster) {
    ASSERT_TRUE(
        MixAdmissionTestPeer::push_normal(scheduler->ready_queues[static_cast<int>(ResourceShape::MIX)], &slot, 0)
    );
    MixAdmissionTestPeer::normal(*context, phase());
    expect_all_blocks();
}

TEST_P(MixAdmission, EarlySelectsEveryOwnedCluster) {
    payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    ASSERT_TRUE(
        MixAdmissionTestPeer::push_early(
            scheduler->early_dispatch_queues[static_cast<int>(ResourceShape::MIX)], &slot, task.task_id, 0
        )
    );
    MixAdmissionTestPeer::early(*context, phase());
    expect_all_blocks();
}

INSTANTIATE_TEST_SUITE_P(
    TrackerWidth, MixAdmission,
    testing::Combine(
        testing::Values(22, 23, CoreTracker::MAX_CLUSTERS),
        testing::Values(uint8_t{3}, uint8_t{5}, uint8_t{6}, uint8_t{7}), testing::Bool()
    )
);
