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

void write_reg(uint64_t, RegId, uint64_t) {}

class SchedulerDispatchTestPeer {
public:
    static void bind(SchedulerContext &context, SchedulerState &scheduler, uint64_t *callables) {
        context.sched_ = &scheduler;
        context.active_sched_threads_ = 1;
        context.func_id_to_addr_ = callables;
        context.func_id_to_addr_count_ = 1;
        context.core_trackers_[0].init(1);
        context.core_trackers_[0].set_cluster(0, 0, 1, 2);
        for (int i = 0; i < 3; ++i) {
            context.core_exec_states_[i] = {};
            context.core_exec_states_[i].reg_addr = i + 1;
            context.core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            context.core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }
    }
    static void occupy(SchedulerContext &context, int offset, bool pending = true) {
        context.core_trackers_[0].change_core_state(offset);
        if (pending) context.core_trackers_[0].set_pending_occupied(offset);
    }
    static void dispatch(SchedulerContext &context, bool pmu = false) {
        bool progress = false;
        bool pushed = false;
        context.dispatch_ready_tasks(0, context.core_trackers_[0], pmu, progress, pushed);
    }
    static ChipTaskSlotState *running(SchedulerContext &context, int core) {
        return context.core_exec_states_[core].running_slot_state;
    }
    static ChipTaskSlotState *pending(SchedulerContext &context, int core) {
        return context.core_exec_states_[core].pending_slot_state;
    }
    static bool drain(SchedulerContext &context, ChipTaskSlotState &slot) {
        return context.enter_drain_mode(&slot, slot.logical_block_num);
    }
    static uint32_t drain_mask(SchedulerContext &context) { return context.drain_state_.reserved_core_mask.load(); }
    static void spare_before_ack(SchedulerContext &context, bool pmu = false) {
        bool progress = false;
        bool pushed = false;
        context.dispatch_spare_aiv_before_drain(0, context.core_trackers_[0], pmu, progress, pushed);
    }
    static void ack(SchedulerContext &context) {
        context.drain_ack_tokens_[0].store(
            sync_start_drain_ack_subtree_token(context.drain_state_.drain_attempt.load())
        );
    }
    static void retry(SchedulerContext &context) {
        context.drain_state_.drain_attempt.store(
            sync_start_drain_next_attempt(context.drain_state_.drain_attempt.load())
        );
    }
    static void early(SchedulerContext &context) {
        bool progress = false;
        bool pushed = false;
        context.try_early_dispatch(0, context.core_trackers_[0], false, progress, pushed);
    }
    static void second_owner(SchedulerContext &context) {
        context.active_sched_threads_ = 2;
        context.core_trackers_[1].init(1);
        context.core_trackers_[1].set_cluster(0, 3, 4, 5);
        for (int i = 3; i < 6; ++i) {
            context.core_exec_states_[i] = {};
            context.core_exec_states_[i].reg_addr = i + 1;
            context.core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            context.core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }
        for (int i : {0, 1}) {
            context.core_trackers_[1].change_core_state(i);
            context.core_trackers_[1].set_pending_occupied(i);
        }
    }
    static void clusters(SchedulerContext &context, int count) {
        context.core_trackers_[0].init(count);
        for (int i = 0; i < count; ++i)
            context.core_trackers_[0].set_cluster(i, 3 * i, 3 * i + 1, 3 * i + 2);
        for (int i = 0; i < 3 * count; ++i) {
            context.core_exec_states_[i] = {};
            context.core_exec_states_[i].reg_addr = i + 1;
            context.core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            context.core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }
    }
    static void drain_attempt(SchedulerContext &context, int owner) {
        bool progress = false;
        bool pushed = false;
        context.dispatch_spare_aiv_before_drain(owner, context.core_trackers_[owner], false, progress, pushed);
        context.handle_drain_mode(owner);
    }
    static uint64_t attempt(SchedulerContext &context) { return context.drain_state_.drain_attempt.load(); }
};

class MixSpareLane : public testing::Test {
protected:
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    DeviceArena shared_arena;
    DeviceArena scheduler_arena;
    std::vector<uint8_t> callable_storage = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t callables[1]{};
    std::array<TaskDescriptor, 3> tasks{};
    std::array<TaskPayload, 3> payloads{};
    std::array<ChipTaskSlotState, 3> slots{};

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
        SchedulerDispatchTestPeer::bind(*context, *scheduler, callables);
    }
    void TearDown() override {
        scheduler->destroy();
        scheduler_arena.release();
        shared_arena.release();
    }
    ChipTaskSlotState &task(int index, uint8_t mask) {
        auto &slot = slots[index];
        slot.task = &tasks[index];
        slot.payload = &payloads[index];
        slot.active_mask = ActiveMask(mask);
        slot.logical_block_num = 1;
        slot.total_required_subtasks = __builtin_popcount(mask);
        slot.task->task_id = TaskId::make(0, index);
        for (int i = 0; i < 3; ++i)
            slot.task->kernel_id[i] = mask & (1u << i) ? 0 : INVALID_KERNEL_ID;
        return slot;
    }
    void queue(ChipTaskSlotState &slot) {
        ASSERT_TRUE(scheduler->ready_queues[static_cast<int>(slot.active_mask.to_shape())].push(
            &slot, slot.active_mask.core_mask()
        ));
    }
    void occupy_pair(uint8_t mask) {
        for (int i = 0; i < 3; ++i)
            if (mask & (1u << i)) SchedulerDispatchTestPeer::occupy(*context, i);
    }
};

TEST_F(MixSpareLane, ResidualCAndV0AllowsIndependentV1) {
    auto &mixed = task(0, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    auto &vector = task(1, SUBTASK_MASK_AIV0);
    occupy_pair(mixed.active_mask.core_mask());
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &vector);
    EXPECT_EQ(vector.next_block_idx.load(), 1);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, ResidualCAndV1AllowsIndependentV0) {
    auto &mixed = task(0, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV1);
    auto &vector = task(1, SUBTASK_MASK_AIV0);
    occupy_pair(mixed.active_mask.core_mask());
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 1), &vector);
    EXPECT_EQ(vector.next_block_idx.load(), 1);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, QueuedFullMixKeepsBothVectorLanesReserved) {
    auto &mixed = task(0, 7);
    auto &vector = task(1, SUBTASK_MASK_AIV0);
    occupy_pair(SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), nullptr);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, DifferentQueuedMixMasksReserveTheirUnion) {
    auto &left = task(0, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    auto &right = task(1, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV1);
    auto &vector = task(2, SUBTASK_MASK_AIV0);
    occupy_pair(SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    queue(left);
    queue(right);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), nullptr);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, RunningUnusedVectorDoesNotPreventMixAdmission) {
    auto &mixed = task(0, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    SchedulerDispatchTestPeer::occupy(*context, 2);
    queue(mixed);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 0), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 1), &mixed);
    EXPECT_EQ(mixed.next_block_idx.load(), 1);
}

TEST_F(MixSpareLane, ResidualMixAllowsSparePendingVector) {
    auto &mixed = task(0, 3);
    auto &vector = task(1, 2);
    occupy_pair(3);
    SchedulerDispatchTestPeer::occupy(*context, 2, false);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 2), &vector);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, PmuAllowsSpareIdleButWithholdsPending) {
    auto &mixed = task(0, 5);
    auto &vector = task(1, 2);
    occupy_pair(5);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context, true);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 1), &vector);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 1), nullptr);
}

TEST_F(MixSpareLane, PmuWithholdsSparePendingVector) {
    auto &mixed = task(0, 3);
    auto &vector = task(1, 2);
    occupy_pair(3);
    SchedulerDispatchTestPeer::occupy(*context, 2, false);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context, true);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 2), nullptr);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, DualVectorResidualAllowsIndependentCube) {
    auto &mixed = task(0, 6);
    auto &cube = task(1, 1);
    occupy_pair(6);
    queue(mixed);
    queue(cube);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 0), &cube);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, PartialMixPendingDrainsReservationAndAdmitsVectorSamePass) {
    auto &mixed = task(0, 3);
    auto &vector = task(1, 2);
    SchedulerDispatchTestPeer::clusters(*context, 2);
    SchedulerDispatchTestPeer::occupy(*context, 0, false);
    SchedulerDispatchTestPeer::occupy(*context, 1, false);
    SchedulerDispatchTestPeer::occupy(*context, 2);
    SchedulerDispatchTestPeer::occupy(*context, 3);
    SchedulerDispatchTestPeer::occupy(*context, 4, false);
    SchedulerDispatchTestPeer::occupy(*context, 5);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 0), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 1), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), nullptr);
    // Newly published work holds its pickup until AICore ACK. The second
    // cluster has V capacity but cannot take MIX because its C pickup is full.
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 2), nullptr);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 4), &vector);
    EXPECT_EQ(scheduler->ready_queues[2].reserved_mix_core_mask(), 0);
}

TEST_F(MixSpareLane, FullMixPendingPreservesWholePassPriority) {
    auto &mixed = task(0, 7);
    auto &vector = task(1, 2);
    SchedulerDispatchTestPeer::clusters(*context, 2);
    SchedulerDispatchTestPeer::occupy(*context, 0, false);
    SchedulerDispatchTestPeer::occupy(*context, 1, false);
    SchedulerDispatchTestPeer::occupy(*context, 3);
    SchedulerDispatchTestPeer::occupy(*context, 4, false);
    SchedulerDispatchTestPeer::occupy(*context, 5, false);
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 0), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 1), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 4), nullptr);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
    EXPECT_EQ(scheduler->ready_queues[2].reserved_mix_core_mask(), 0);
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::pending(*context, 4), &vector);
}

TEST_F(MixSpareLane, TwoOwnersFillSpareLanesBeforeActualRetryAckTree) {
    auto &cohort = task(0, 3);
    auto &vector = task(1, 2);
    cohort.logical_block_num = 2;
    vector.logical_block_num = 2;
    vector.total_required_subtasks = 2;
    occupy_pair(3);
    SchedulerDispatchTestPeer::second_owner(*context);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    uint64_t attempt = SchedulerDispatchTestPeer::attempt(*context);
    queue(vector);
    std::thread peer([&] {
        SchedulerDispatchTestPeer::drain_attempt(*context, 1);
    });
    SchedulerDispatchTestPeer::drain_attempt(*context, 0);
    peer.join();
    EXPECT_EQ(SchedulerDispatchTestPeer::attempt(*context), sync_start_drain_next_attempt(attempt));
    EXPECT_EQ(vector.next_block_idx.load(), 2);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &vector);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 5), &vector);
    EXPECT_EQ(cohort.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, BusyUnusedVectorDoesNotPreventReadySyncMixAdmission) {
    auto &cohort = task(0, 3);
    cohort.task_attrs.set_sync_start();
    SchedulerDispatchTestPeer::occupy(*context, 2);
    scheduler->sync_task_seen.store(1);
    ASSERT_TRUE(scheduler->ready_sync_queues[static_cast<int>(ResourceShape::MIX)].push(&cohort, 3));
    SchedulerDispatchTestPeer::dispatch(*context);
    EXPECT_EQ(cohort.next_block_idx.load(), 1);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 0), &cohort);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 1), &cohort);
}

TEST_F(MixSpareLane, WaitingCAndV0CohortAllowsSpareVectorBeforeAck) {
    auto &cohort = task(0, 3);
    auto &vector = task(1, 2);
    cohort.task_attrs.set_sync_start();
    occupy_pair(3);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    queue(vector);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::drain_mask(*context), 3);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &vector);
    EXPECT_EQ(cohort.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, WaitingCAndV1CohortAllowsSpareVectorBeforeAck) {
    auto &cohort = task(0, 5);
    auto &vector = task(1, 2);
    occupy_pair(5);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    queue(vector);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 1), &vector);
}

TEST_F(MixSpareLane, DrainDoesNotWriteTrackerAfterCurrentAttemptAck) {
    auto &cohort = task(0, 3);
    auto &vector = task(1, 2);
    occupy_pair(3);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    queue(vector);
    SchedulerDispatchTestPeer::ack(*context);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), nullptr);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
    SchedulerDispatchTestPeer::retry(*context);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &vector);
}

TEST_F(MixSpareLane, AivCohortReservesBothPhysicalVectorLanes) {
    auto &cohort = task(0, 2);
    auto &vector = task(1, 2);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    queue(vector);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::drain_mask(*context), 6);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, DrainHonorsOtherQueuedMixMasksAndPmu) {
    auto &cohort = task(0, 3);
    auto &mixed = task(1, 5);
    auto &vector = task(2, 2);
    occupy_pair(3);
    ASSERT_TRUE(SchedulerDispatchTestPeer::drain(*context, cohort));
    queue(vector);
    SchedulerDispatchTestPeer::spare_before_ack(*context, true);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
    queue(mixed);
    SchedulerDispatchTestPeer::spare_before_ack(*context);
    EXPECT_EQ(vector.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, EarlyResidualMixAllowsOnlyUnreservedVectorLane) {
    auto &mixed = task(0, 3);
    auto &vector = task(1, 2);
    occupy_pair(3);
    for (auto *slot : {&mixed, &vector}) {
        slot->payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING);
        ASSERT_TRUE(scheduler->early_dispatch_queues[static_cast<int>(slot->active_mask.to_shape())].push_tagged(
            slot, slot->task->task_id, slot->active_mask.core_mask()
        ));
    }
    SchedulerDispatchTestPeer::early(*context);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 2), &vector);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, MixAdmissionUsesTrackerBitsAbove64) {
    SchedulerDispatchTestPeer::clusters(*context, 25);
    auto &mixed = task(0, 3);
    mixed.logical_block_num = 25;
    mixed.total_required_subtasks = 50;
    queue(mixed);
    SchedulerDispatchTestPeer::dispatch(*context, true);
    EXPECT_EQ(mixed.next_block_idx.load(), 25);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 72), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 73), &mixed);
}

TEST_F(MixSpareLane, ResidualReservationFiltersTrackerBitsAbove64) {
    SchedulerDispatchTestPeer::clusters(*context, 25);
    auto &mixed = task(0, 3);
    auto &vector = task(1, 2);
    for (int i = 0; i < 25; ++i) {
        SchedulerDispatchTestPeer::occupy(*context, 3 * i);
        SchedulerDispatchTestPeer::occupy(*context, 3 * i + 1);
        if (i < 24) SchedulerDispatchTestPeer::occupy(*context, 3 * i + 2);
    }
    queue(mixed);
    queue(vector);
    SchedulerDispatchTestPeer::dispatch(*context, true);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 74), &vector);
    EXPECT_EQ(mixed.next_block_idx.load(), 0);
}

TEST_F(MixSpareLane, EarlyMixAdmissionUsesTrackerBitsAbove64) {
    SchedulerDispatchTestPeer::clusters(*context, 25);
    auto &mixed = task(0, 3);
    mixed.logical_block_num = 25;
    mixed.total_required_subtasks = 50;
    mixed.payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    ASSERT_TRUE(scheduler->early_dispatch_queues[static_cast<int>(ResourceShape::MIX)].push_tagged(
        &mixed, mixed.task->task_id, 3
    ));
    SchedulerDispatchTestPeer::early(*context);
    EXPECT_EQ(mixed.next_block_idx.load(), 25);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 72), &mixed);
    EXPECT_EQ(SchedulerDispatchTestPeer::running(*context, 73), &mixed);
}
