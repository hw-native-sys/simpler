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
 * Unit tests for PTO2SchedulerState from pto_scheduler.h
 *
 * Tests task state transitions, fanin/fanout logic, subtask completion.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>

#include "utils/device_arena.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/pto_scheduler.h"

class SchedulerStateTest : public ::testing::Test {
protected:
    PTO2SchedulerState sched;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = PTO2SchedulerState::reserve_layout(sched_arena);
        ASSERT_NE(sched_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, sched_arena, sm_handle->header));
        sched.wire_arena_pointers(layout, sched_arena);
    }

    void TearDown() override {
        sched.destroy();
        sched_arena.release();
        sm_arena.release();
    }

    void init_slot(
        PTO2TaskSlotState &slot, PTO2TaskState state, int32_t fanin_count, int32_t fanout_count, uint8_t ring_id = 0
    ) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(0);
        slot.fanout_lock.store(0);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
    }
};

// =============================================================================
// check_and_handle_consumed
// =============================================================================

TEST_F(SchedulerStateTest, ConsumedNotReady) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // 1 != 2

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_COMPLETED);
}

TEST_F(SchedulerStateTest, ConsumedTransition) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(2);  // matches fanout_count

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

TEST_F(SchedulerStateTest, ConsumedNotCompletedState) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    // CAS fails because state is PENDING, not COMPLETED
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING);
}

TEST_F(SchedulerStateTest, ConsumedIdempotent) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_CONSUMED, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// =============================================================================
// release_producer
// =============================================================================

TEST_F(SchedulerStateTest, ReleaseProducerIncrements) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_COMPLETED, 1, 3);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 1);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 2);
}

TEST_F(SchedulerStateTest, ReleaseProducerTriggersConsumed) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // One away

    sched.release_producer(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// =============================================================================
// on_subtask_complete
// =============================================================================

TEST_F(SchedulerStateTest, SubtaskCompleteSingle) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 1;
    slot.completed_subtasks.store(0);

    EXPECT_TRUE(sched.on_subtask_complete(slot));
}

TEST_F(SchedulerStateTest, SubtaskCompleteMultiBlock) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 6;  // 3 cores * 2 blocks
    slot.completed_subtasks.store(0);

    for (int i = 0; i < 5; i++) {
        EXPECT_FALSE(sched.on_subtask_complete(slot));
    }
    EXPECT_TRUE(sched.on_subtask_complete(slot));
}

// =============================================================================
// on_scope_end
// =============================================================================

TEST_F(SchedulerStateTest, ScopeEndBatchRelease) {
    constexpr int N = 4;
    alignas(64) PTO2TaskSlotState slots[N];
    PTO2TaskSlotState *ptrs[N];

    for (int i = 0; i < N; i++) {
        init_slot(slots[i], PTO2_TASK_COMPLETED, 1, 2);
        ptrs[i] = &slots[i];
    }

    sched.on_scope_end(ptrs, N);

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(slots[i].fanout_refcount.load(), 1);
    }
}

// =============================================================================
// get_ready_tasks_batch: local buffer first
// =============================================================================

TEST_F(SchedulerStateTest, GetReadyTasksBatchLocalFirst) {
    alignas(64) PTO2TaskSlotState slot_a, slot_b;
    init_slot(slot_a, PTO2_TASK_PENDING, 0, 1);
    init_slot(slot_b, PTO2_TASK_PENDING, 1, 1);

    PTO2TaskSlotState *local_buf_storage[4];
    PTO2LocalReadyBuffer local_buf;
    local_buf.reset(local_buf_storage, 4);
    local_buf.try_push(&slot_a);

    // Use src API to route slot_b into the global ready queue
    sched.release_fanin_and_check_ready(slot_b);

    PTO2TaskSlotState *out[4];
    int count = sched.get_ready_tasks_batch(PTO2ResourceShape::AIC, local_buf, out, 4);

    EXPECT_EQ(count, 2);
    // Local buffer drains first (LIFO), so slot_a comes first
    EXPECT_EQ(out[0], &slot_a);
    EXPECT_EQ(out[1], &slot_b);
}

TEST(CoreTrackerTest, MixPendingRejectsPartiallyRunningCluster) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running, AIC/AIV1 idle
    tracker.clear_pending_occupied(cluster_offset + 1);

    EXPECT_TRUE(tracker.is_aic_core_idle(cluster_offset));
    EXPECT_FALSE(tracker.is_aiv0_core_idle(cluster_offset));
    EXPECT_TRUE(tracker.is_aiv1_core_idle(cluster_offset));

    const bool would_place_aic_pending = !tracker.is_aic_core_idle(cluster_offset);
    const bool would_place_aiv0_pending = !tracker.is_aiv0_core_idle(cluster_offset);
    const bool would_place_aiv1_pending = !tracker.is_aiv1_core_idle(cluster_offset);
    EXPECT_FALSE(would_place_aic_pending);
    EXPECT_TRUE(would_place_aiv0_pending);
    EXPECT_FALSE(would_place_aiv1_pending);

    auto idle = tracker.get_idle_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_FALSE(idle.has_value());

    auto pending = tracker.get_pending_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value()) << "Pending admission here would split one MIX block across running AIC/AIV1 "
                                      << "placement and pending AIV0 placement.";
}

TEST(CoreTrackerTest, MixPendingAcceptsFullyRunningClusterWithFreePendingSlots) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.change_core_state(cluster_offset + 2);
    tracker.clear_pending_occupied(cluster_offset);
    tracker.clear_pending_occupied(cluster_offset + 1);
    tracker.clear_pending_occupied(cluster_offset + 2);

    auto pending = tracker.get_pending_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_TRUE(pending.has_value());
    EXPECT_EQ(pending.count(), 1);
}

TEST(CoreTrackerTest, MixPendingRejectsFullyRunningClusterWithOccupiedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.change_core_state(cluster_offset + 2);
    tracker.set_pending_occupied(cluster_offset + 1);

    auto pending = tracker.get_pending_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixIdleAndPendingDoNotDoubleAdmitFullyIdleCluster) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    auto idle = tracker.get_idle_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_TRUE(idle.has_value());
    EXPECT_EQ(idle.count(), 1);

    auto pending = tracker.get_pending_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixClassifyIgnoresUnusedBusyCoreForRunningPlacement) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 2);  // AIV1 running, unused by this 1c1v task

    auto placement = tracker.classify_mix_cluster(cluster_offset, PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::RUNNING);
}

TEST(CoreTrackerTest, MixClassifyAllowsPendingForUsedRunningCoresOnly) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.set_pending_occupied(cluster_offset + 2);  // Unused AIV1 must not block this 1c1v task

    auto placement = tracker.classify_mix_cluster(cluster_offset, PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::PENDING);
}

TEST(CoreTrackerTest, MixClassifyRejectsMixedUsedCores) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running while AIC is idle

    auto placement = tracker.classify_mix_cluster(cluster_offset, PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::REJECT);
}

TEST(CoreTrackerTest, MixClassifyRejectsOccupiedPendingSlotInUsedMask) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset);
    tracker.change_core_state(cluster_offset + 1);
    tracker.set_pending_occupied(cluster_offset + 1);

    auto placement = tracker.classify_mix_cluster(cluster_offset, PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::REJECT);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersUseActiveMask) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(2);
    tracker.change_core_state(5);

    constexpr uint8_t used_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;

    EXPECT_EQ(tracker.get_idle_core_offset_states(PTO2ResourceShape::MIX).count(), 0);
    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 2);
    EXPECT_EQ(tracker.get_mix_running_cluster_offset_states(used_mask).count(), 2);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersRejectOccupiedUsedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0;
    tracker.set_pending_occupied(1);

    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 0);
}
