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

#include "scheduler/pto_scheduler.h"

class SchedulerStateTest : public ::testing::Test {
protected:
    PTO2SchedulerState sched;
    PTO2SharedMemoryHandle *sm_handle = nullptr;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_default();
        ASSERT_NE(sm_handle, nullptr);
        bool ok = sched.init(sm_handle->header);
        ASSERT_TRUE(ok);
    }

    void TearDown() override {
        sched.destroy();
        if (sm_handle) {
            sm_handle->destroy();
        }
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
