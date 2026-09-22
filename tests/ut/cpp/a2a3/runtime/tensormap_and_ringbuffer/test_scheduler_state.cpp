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
 * Unit tests for SchedulerState from scheduler.h
 *
 * Tests task state transitions, fanin/fanout logic, subtask completion.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>

#include "utils/device_arena.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler.h"

TEST(SyncStartDrainAttemptTest, LateAckCannotSatisfyNextAttemptBarrier) {
    std::atomic<uint64_t> ack_tokens[3]{};
    uint64_t old_attempt = 41;
    uint64_t next_attempt = sync_start_drain_next_attempt(old_attempt);
    uint64_t old_subtree = sync_start_drain_ack_subtree_token(old_attempt);
    uint64_t next_subtree = sync_start_drain_ack_subtree_token(next_attempt);

    ack_tokens[1].store(old_subtree, std::memory_order_relaxed);
    ack_tokens[2].store(old_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[1].load(std::memory_order_relaxed), old_subtree);
    EXPECT_EQ(ack_tokens[2].load(std::memory_order_relaxed), old_subtree);

    // A late child token from attempt 41 cannot satisfy attempt 42's tree barrier.
    ack_tokens[1].store(next_subtree, std::memory_order_relaxed);
    ack_tokens[2].store(old_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[1].load(std::memory_order_relaxed), next_subtree);
    EXPECT_NE(ack_tokens[2].load(std::memory_order_relaxed), next_subtree);

    ack_tokens[2].store(next_subtree, std::memory_order_relaxed);
    ack_tokens[0].store(old_subtree, std::memory_order_relaxed);
    EXPECT_NE(ack_tokens[0].load(std::memory_order_relaxed), next_subtree);
    ack_tokens[0].store(next_subtree, std::memory_order_relaxed);
    EXPECT_EQ(ack_tokens[0].load(std::memory_order_relaxed), next_subtree);

    EXPECT_EQ(sync_start_drain_next_attempt(SYNC_START_DRAIN_ATTEMPT_MASK), 1u);
}

class SchedulerStateTest : public ::testing::Test {
protected:
    SchedulerState sched;
    SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = SchedulerState::reserve_layout(sched_arena);
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
        ChipTaskSlotState &slot, ChipTaskState state, int32_t fanin_count, int32_t fanout_count, uint8_t ring_id = 0
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
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        TaskPayload &slot_pl = slot_payload_pool_[slot_payload_pool_idx_++ % kSlotPayloadPoolSize];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot.payload = &slot_pl;
    }

    void init_ring_slot(
        SharedMemoryRingHeader &ring, int32_t task_id, ChipTaskState state, uint8_t ring_id, uint32_t fanout_count = 1,
        uint32_t fanout_refcount = 1
    ) {
        ChipTaskSlotState &slot = ring.get_slot_state_by_task_id(task_id);
        TaskPayload &payload = ring.get_payload_by_task_id(task_id);
        TaskDescriptor &task = ring.get_task_by_task_id(task_id);
        memset(&slot, 0, sizeof(slot));
        memset(&payload, 0, sizeof(payload));
        memset(&task, 0, sizeof(task));
        slot.task_state.store(state, std::memory_order_relaxed);
        slot.fanin_count = 0;
        slot.fanin_refcount.store(0, std::memory_order_relaxed);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(fanout_refcount, std::memory_order_relaxed);
        slot.fanout_lock.store(0, std::memory_order_relaxed);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0, std::memory_order_relaxed);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        slot.lifecycle_flags.store(COMPLETION_DONE, std::memory_order_relaxed);
        slot.payload = &payload;
        slot.task = &task;
    }

    void setup_ring_for_reclaim_race(int32_t ring_id, int32_t current_task_index, int32_t blocked_task_id) {
        SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
        SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];

        ring.fc.current_task_index.store(current_task_index, std::memory_order_release);
        ring.fc.last_task_alive.store(0, std::memory_order_release);
        ring_sched.last_task_alive = 0;
        ring_sched.advance_lock.store(0, std::memory_order_release);
        sched.advance_pending_mask.store(0, std::memory_order_release);

        for (int32_t task_id = 0; task_id < current_task_index; task_id++) {
            ChipTaskState state = task_id < blocked_task_id ? CHIP_TASK_CONSUMED : CHIP_TASK_COMPLETED;
            init_ring_slot(ring, task_id, state, static_cast<uint8_t>(ring_id));
        }
    }

    void setup_contended_head_case(int32_t ring_id, int32_t head_task_id) {
        SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
        SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
        ring.fc.current_task_index.store(head_task_id + 2, std::memory_order_release);
        ring.fc.last_task_alive.store(head_task_id, std::memory_order_release);
        ring_sched.last_task_alive = head_task_id;
        ring_sched.advance_lock.store(0, std::memory_order_release);
        sched.advance_pending_mask.store(0, std::memory_order_release);
        init_ring_slot(ring, head_task_id, CHIP_TASK_COMPLETED, static_cast<uint8_t>(ring_id));
        init_ring_slot(ring, head_task_id + 1, CHIP_TASK_COMPLETED, static_cast<uint8_t>(ring_id), 1, 0);
    }
};

// =============================================================================
// check_and_handle_consumed
// =============================================================================

TEST_F(SchedulerStateTest, ConsumedNotReady) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // 1 != 2

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_COMPLETED);
}

TEST_F(SchedulerStateTest, ConsumedTransition) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(2);  // matches fanout_count

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

TEST_F(SchedulerStateTest, ConsumedHeadAdvancesAfterContendedAdvanceLock) {
    constexpr int32_t ring_id = CHIP_MAX_RING_DEPTH - 1;
    constexpr int32_t head_task_id = 0;
    setup_ring_for_reclaim_race(ring_id, /*current_task_index=*/1, head_task_id);

    SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
    SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
    ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);
    uint32_t pending_bit = SchedulerState::ring_advance_pending_bit(ring_id);

    ring_sched.advance_lock.store(1, std::memory_order_release);
    std::thread unlocker([&]() {
        while ((sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit) == 0) {
            std::this_thread::yield();
        }
        ring_sched.advance_lock.store(0, std::memory_order_release);
    });

    sched.check_and_handle_consumed(head);
    unlocker.join();

    EXPECT_TRUE(sched.drain_pending_ring_advances());
    EXPECT_EQ(head.task_state.load(std::memory_order_acquire), CHIP_TASK_CONSUMED);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), 1)
        << "a CONSUMED ring head must not remain pinned after advance_lock contention clears";
}

TEST_F(SchedulerStateTest, ConsumedNotCompletedState) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    // CAS fails because state is PENDING, not COMPLETED
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_PENDING);
}

TEST_F(SchedulerStateTest, ConsumedIdempotent) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_CONSUMED, 1, 1);
    slot.fanout_refcount.store(1);

    sched.check_and_handle_consumed(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

TEST_F(SchedulerStateTest, ContendedConsumedHeadSetsPendingAndIdleDrainAdvances) {
    constexpr int32_t ring_id = CHIP_MAX_RING_DEPTH - 1;
    constexpr int32_t head_task_id = 17;
    setup_contended_head_case(ring_id, head_task_id);

    SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
    SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
    ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);
    uint32_t pending_bit = SchedulerState::ring_advance_pending_bit(ring_id);

    ring_sched.advance_lock.store(1, std::memory_order_release);
    sched.check_and_handle_consumed(head);

    EXPECT_EQ(head.task_state.load(std::memory_order_acquire), CHIP_TASK_CONSUMED);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id);
    EXPECT_NE(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);

    EXPECT_FALSE(sched.drain_pending_ring_advances());
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id);
    EXPECT_NE(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);

    ring_sched.advance_lock.store(0, std::memory_order_release);
    EXPECT_TRUE(sched.drain_pending_ring_advances());
    EXPECT_EQ(ring_sched.last_task_alive, head_task_id + 1);
    EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id + 1);
    EXPECT_EQ(sched.advance_pending_mask.load(std::memory_order_acquire) & pending_bit, 0u);
}

TEST_F(SchedulerStateTest, ContendedConsumedHeadIdleDrainStress) {
    const int32_t head_task_ids[] = {0, 1, 127, CHIP_TASK_WINDOW_SIZE - 2, CHIP_TASK_WINDOW_SIZE + 3};

    for (int32_t ring_id = 0; ring_id < CHIP_MAX_RING_DEPTH; ring_id++) {
        for (int32_t head_task_id : head_task_ids) {
            SCOPED_TRACE(::testing::Message() << "ring_id=" << ring_id << " head_task_id=" << head_task_id);
            setup_contended_head_case(ring_id, head_task_id);

            SharedMemoryRingHeader &ring = sm_handle->header->rings[ring_id];
            SchedulerState::RingSchedState &ring_sched = sched.ring_sched_states[ring_id];
            ChipTaskSlotState &head = ring.get_slot_state_by_task_id(head_task_id);

            ring_sched.advance_lock.store(1, std::memory_order_release);
            sched.check_and_handle_consumed(head);
            ring_sched.advance_lock.store(0, std::memory_order_release);

            EXPECT_TRUE(sched.drain_pending_ring_advances());
            EXPECT_EQ(ring_sched.last_task_alive, head_task_id + 1);
            EXPECT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), head_task_id + 1);
        }
    }
}

// =============================================================================
// release_producer
// =============================================================================

TEST_F(SchedulerStateTest, ReleaseProducerIncrements) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 3);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 1);

    sched.release_producer(slot);
    EXPECT_EQ(slot.fanout_refcount.load(), 2);
}

TEST_F(SchedulerStateTest, ReleaseProducerTriggersConsumed) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_COMPLETED, 1, 2);
    slot.fanout_refcount.store(1);  // One away

    sched.release_producer(slot);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
}

// =============================================================================
// on_subtask_complete
// =============================================================================

TEST_F(SchedulerStateTest, SubtaskCompleteSingle) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 1;
    slot.completed_subtasks.store(0);

    EXPECT_TRUE(sched.on_subtask_complete(slot));
}

TEST_F(SchedulerStateTest, SubtaskCompleteMultiBlock) {
    alignas(64) ChipTaskSlotState slot;
    init_slot(slot, CHIP_TASK_PENDING, 1, 1);
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
    alignas(64) ChipTaskSlotState slots[N];
    ChipTaskSlotState *ptrs[N];

    for (int i = 0; i < N; i++) {
        init_slot(slots[i], CHIP_TASK_COMPLETED, 1, 2);
        ptrs[i] = &slots[i];
    }

    sched.on_scope_end(ptrs, N);

    for (int i = 0; i < N; i++) {
        // on_scope_end releases the owning-scope ref via release_producer_scope,
        // which adds FANOUT_SCOPE_BIT (bit31) to fanout_refcount.
        EXPECT_EQ(slots[i].fanout_refcount.load(), FANOUT_SCOPE_BIT);
    }
}

// =============================================================================
// get_ready_tasks_batch: drains the shared ready queue
// =============================================================================

TEST_F(SchedulerStateTest, GetReadyTasksBatchDrainsSharedQueue) {
    alignas(64) ChipTaskSlotState slot_a, slot_b;
    // fanin_count = 1 so a single release_fanin_and_check_ready call drives each
    // slot to ready (new_refcount 0->1 == fanin_count) and enqueues it.
    init_slot(slot_a, CHIP_TASK_PENDING, 1, 1);
    init_slot(slot_b, CHIP_TASK_PENDING, 1, 1);

    // Route both slots into the global ready queue via the src API.
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_a));
    ASSERT_TRUE(sched.release_fanin_and_check_ready(slot_b));

    ChipTaskSlotState *out[4];
    int count = sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 4);

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
    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0 | SUBTASK_MASK_AIV1;
    EXPECT_EQ(tracker.classify_mix_cluster(cluster_offset, used_mask), CoreTracker::MixPlacement::PENDING);

    // Not all used cores are idle, so the IDLE phase skips this cluster; it is
    // consumed by the PENDING phase.
    auto idle = tracker.get_idle_core_offset_states(ResourceShape::MIX);
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

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
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

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixIdleAndPendingDoNotDoubleAdmitFullyIdleCluster) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    auto idle = tracker.get_idle_core_offset_states(ResourceShape::MIX);
    EXPECT_TRUE(idle.has_value());
    EXPECT_EQ(idle.count(), 1);

    auto pending = tracker.get_pending_core_offset_states(ResourceShape::MIX);
    EXPECT_FALSE(pending.has_value());
}

TEST(CoreTrackerTest, MixClassifyIgnoresUnusedBusyCoreForRunningPlacement) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr int32_t cluster_offset = 0;
    tracker.change_core_state(cluster_offset + 2);  // AIV1 running, unused by this 1c1v task

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
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

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
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
    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
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

    auto placement = tracker.classify_mix_cluster(cluster_offset, SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0);
    EXPECT_EQ(placement, CoreTracker::MixPlacement::REJECT);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersUseActiveMask) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(2);
    tracker.change_core_state(5);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;

    EXPECT_EQ(tracker.get_idle_core_offset_states(ResourceShape::MIX).count(), 0);
    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 2);
    EXPECT_EQ(tracker.get_mix_running_cluster_offset_states(used_mask).count(), 2);
}

TEST(CoreTrackerTest, MixRunningClusterHelpersRejectOccupiedUsedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.set_pending_occupied(1);

    EXPECT_EQ(tracker.count_mix_running_clusters(used_mask), 0);
}

TEST(CoreTrackerTest, CountAvailableBlocksAivIncludesFreePendingSlots) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(1);
    tracker.change_core_state(4);

    constexpr uint8_t aiv_mask = SUBTASK_MASK_AIV0;
    constexpr int32_t exact_fit = 4;
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, false), 2);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, true), exact_fit);

    tracker.set_pending_occupied(4);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::AIV, aiv_mask, true), exact_fit - 1);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixCountsLogicalClusters) {
    CoreTracker tracker;
    tracker.init(2);
    tracker.set_cluster(0, 0, 1, 2);
    tracker.set_cluster(1, 3, 4, 5);

    tracker.change_core_state(4);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0 | SUBTASK_MASK_AIV1;
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 1);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 2);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixRejectsOccupiedUsedPendingSlot) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.change_core_state(1);
    tracker.set_pending_occupied(1);

    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 0);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 0);
}

TEST(CoreTrackerTest, CountAvailableBlocksMixIgnoresUnavailableUnusedCore) {
    CoreTracker tracker;
    tracker.init(1);
    tracker.set_cluster(0, 0, 1, 2);

    constexpr uint8_t used_mask = SUBTASK_MASK_AIC | SUBTASK_MASK_AIV0;
    tracker.change_core_state(2);
    tracker.set_pending_occupied(2);

    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, false), 1);
    EXPECT_EQ(tracker.count_available_blocks(ResourceShape::MIX, used_mask, true), 1);
}
