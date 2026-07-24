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

TEST(SyncStartDrainAttemptTest, LateAckCannotSatisfyNextAttemptBarrier) {
    std::atomic<uint64_t> ack_attempts[3]{};

    for (auto &ack : ack_attempts)
        ack.store(41, std::memory_order_relaxed);
    EXPECT_EQ(sync_start_drain_ack_mask_for_attempt(ack_attempts, 3, 41), 0b111u);

    // Attempt 42 starts, then thread 0 from attempt 41 publishes late.
    ack_attempts[1].store(42, std::memory_order_relaxed);
    ack_attempts[2].store(42, std::memory_order_relaxed);
    ack_attempts[0].store(41, std::memory_order_relaxed);
    EXPECT_EQ(sync_start_drain_ack_mask_for_attempt(ack_attempts, 3, 42), 0b110u);

    ack_attempts[0].store(42, std::memory_order_relaxed);
    EXPECT_EQ(sync_start_drain_ack_mask_for_attempt(ack_attempts, 3, 42), 0b111u);
}

TEST(SyncStartDrainAttemptTest, StaleParticipantCannotOwnNextAttempt) {
    SyncStartDrainState state;
    state.drain_attempt.store(42, std::memory_order_relaxed);

    EXPECT_FALSE(sync_start_drain_try_elect(state, 0, 41));
    EXPECT_EQ(state.drain_worker_elected.load(std::memory_order_relaxed), 0);

    EXPECT_TRUE(sync_start_drain_try_elect(state, 1, 42));
    EXPECT_EQ(state.drain_worker_elected.load(std::memory_order_relaxed), 2);
}

class SchedulerStateTest : public ::testing::Test {
protected:
    PTO2SchedulerState sched;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    PTO2TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

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
        PTO2TaskPayload &slot_pl = slot_payload_pool_[slot_payload_pool_idx_++ % kSlotPayloadPoolSize];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot.payload = &slot_pl;
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
        // on_scope_end releases the owning-scope ref via release_producer_scope,
        // which adds PTO2_FANOUT_SCOPE_BIT (bit31) to fanout_refcount.
        EXPECT_EQ(slots[i].fanout_refcount.load(), PTO2_FANOUT_SCOPE_BIT);
    }
}

// =============================================================================
// get_ready_tasks_batch: drains the shared ready queue
// =============================================================================

TEST_F(SchedulerStateTest, GetReadyTasksBatchDrainsSharedQueue) {
    alignas(64) PTO2TaskSlotState slot_a, slot_b;
    // fanin_count = 1 so a single release_fanin_and_check_ready call drives each
    // slot to ready (new_refcount 0->1 == fanin_count) and enqueues it.
    init_slot(slot_a, PTO2_TASK_PENDING, 1, 1);
    init_slot(slot_b, PTO2_TASK_PENDING, 1, 1);

    // Route both slots into the global ready queue via the src API.
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_a));
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_b));

    PTO2TaskSlotState *out[4];
    int count = sched.get_ready_tasks_batch(PTO2ResourceShape::AIC, out, 4);

    EXPECT_EQ(count, 2);
    // Shared queue is FIFO, so slot_a (pushed first) comes first.
    EXPECT_EQ(out[0], &slot_a);
    EXPECT_EQ(out[1], &slot_b);
}

TEST(CoreTrackerTest, MixPartiallyRunningClusterAdmittedAsPerCorePlacement) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running (unrelated task), AIC/AIV1 idle
    tracker.clear_pending_occupied(cluster_offset + 1);

    EXPECT_TRUE(tracker.is_aic_core_idle(cluster_offset));
    EXPECT_FALSE(tracker.is_aiv0_core_idle(cluster_offset));
    EXPECT_TRUE(tracker.is_aiv1_core_idle(cluster_offset));

    // A 1C2V MIX task is admitted on this partial cluster as a PENDING placement.
    // Per-core dispatch then puts the idle AIC/AIV1 on their running slots (marked
    // running so the completion poller tracks them) and the busy AIV0 on its pending
    // slot, executing after the in-flight AIV-only task.
    constexpr uint8_t used_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    EXPECT_EQ(tracker.classify_mix_cluster(cluster_offset, used_mask), CoreTracker::MixPlacement::PENDING);

    // Not all used cores are idle, so the IDLE phase skips this cluster; it is
    // consumed by the PENDING phase.
    auto idle = tracker.get_idle_core_offset_states(PTO2ResourceShape::MIX);
    EXPECT_FALSE(idle.has_value());
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

TEST(CoreTrackerTest, MixClassifyAdmitsMixedUsedCoresAsPending) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 1);  // AIV0 running while AIC is idle

    // Mixed used-core state (AIC idle, AIV0 running) is admitted as PENDING; the
    // idle AIC takes its running slot and the busy AIV0 takes its pending slot.
    auto placement = tracker.classify_mix_cluster(cluster_offset, PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::PENDING);
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
