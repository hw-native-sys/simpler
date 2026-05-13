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
 * Unit tests for PTO2TaskSlotState lifecycle through PTO2SchedulerState API.
 *
 * These tests drive state transitions via src methods (release_fanin,
 * on_subtask_complete, check_and_handle_consumed) rather than manually
 * operating atomic fields.  For concurrent exactly-once semantics of
 * fanin/subtask/fanout, see test_scheduler_state.cpp which already
 * covers those paths via the same API.
 *
 * This file focuses on:
 * - Full lifecycle through src API
 * - Ready-path behavior (task_state stays PENDING through dispatch)
 * - Double subtask completion (counter-model weakness)
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include "scheduler/pto_scheduler.h"

class TaskStateTest : public ::testing::Test {
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

    void init_slot(PTO2TaskSlotState &slot, PTO2TaskState state, int32_t fanin_count, int32_t fanout_count) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(0);
        slot.fanout_lock.store(0);
        slot.fanout_head = nullptr;
        slot.ring_id = 0;
        slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
    }
};

// =============================================================================
// Full lifecycle through src API: PENDING -> (fanin) -> (queued + dispatched)
// -> (subtask) -> COMPLETED -> (fanout) -> CONSUMED
// =============================================================================
TEST_F(TaskStateTest, FullLifecycleThroughAPI) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 1;
    slot.completed_subtasks.store(0);

    // Fanin satisfied -> task becomes ready
    bool ready = sched.release_fanin_and_check_ready(slot);
    EXPECT_TRUE(ready);

    // Subtask completes -> task done
    bool done = sched.on_subtask_complete(slot);
    EXPECT_TRUE(done);

    // Manually transition to COMPLETED (normally done by scheduler dispatch loop)
    slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    // Fanout released -> CONSUMED
    sched.release_producer(slot);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// =============================================================================
// release_fanin does not write task_state.
//
// Readiness is determined solely by fanin_refcount reaching fanin_count.
// task_state stays PENDING from submit through "queued in ready_queue" and
// "dispatched to a worker" until the worker stores COMPLETED.
// =============================================================================
TEST_F(TaskStateTest, ReadyPathStaysPending) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);

    bool ready = sched.release_fanin_and_check_ready(slot);
    ASSERT_TRUE(ready) << "Task should be detected as ready via refcount";

    // task_state remains PENDING -- there is no intermediate ready/running state.
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING) << "release_fanin_and_check_ready must not write task_state";
}

// =============================================================================
// Multi-fanin: partial release does not trigger ready
// =============================================================================
TEST_F(TaskStateTest, MultiFaninPartialNotReady) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 3, 1);

    EXPECT_FALSE(sched.release_fanin_and_check_ready(slot));
    EXPECT_FALSE(sched.release_fanin_and_check_ready(slot));
    EXPECT_TRUE(sched.release_fanin_and_check_ready(slot));
}

// =============================================================================
// Concurrent fanin: exactly one thread detects ready (via src API)
// =============================================================================
TEST_F(TaskStateTest, ConcurrentFaninExactlyOneReady) {
    constexpr int ROUNDS = 500;

    for (int round = 0; round < ROUNDS; round++) {
        alignas(64) PTO2TaskSlotState slot;
        init_slot(slot, PTO2_TASK_PENDING, 3, 1);
        std::atomic<int> ready_count{0};

        auto release = [&]() {
            if (sched.release_fanin_and_check_ready(slot)) {
                ready_count.fetch_add(1);
            }
        };

        std::thread t1(release), t2(release), t3(release);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(ready_count.load(), 1) << "Round " << round;
    }
}

// =============================================================================
// Concurrent subtask completion: exactly one thread sees done (via src API)
// =============================================================================
TEST_F(TaskStateTest, ConcurrentSubtaskCompletion) {
    constexpr int ROUNDS = 500;

    for (int round = 0; round < ROUNDS; round++) {
        alignas(64) PTO2TaskSlotState slot;
        init_slot(slot, PTO2_TASK_PENDING, 1, 1);
        slot.total_required_subtasks = 3;
        slot.completed_subtasks.store(0);
        std::atomic<int> done_count{0};

        auto complete = [&]() {
            if (sched.on_subtask_complete(slot)) {
                done_count.fetch_add(1);
            }
        };

        std::thread t1(complete), t2(complete), t3(complete);
        t1.join();
        t2.join();
        t3.join();

        EXPECT_EQ(done_count.load(), 1) << "Round " << round;
        EXPECT_EQ(slot.completed_subtasks.load(), 3);
    }
}

// =============================================================================
// Double subtask completion (counter-model weakness).
// With the counter model, double-completing the same subtask increments
// completed_subtasks twice, potentially reaching total prematurely.
// Unlike the old bitmask model, the counter cannot detect duplicates.
// =============================================================================
TEST_F(TaskStateTest, DoubleSubtaskCompletionCounterWeakness) {
    alignas(64) PTO2TaskSlotState slot;
    init_slot(slot, PTO2_TASK_PENDING, 1, 1);
    slot.total_required_subtasks = 2;
    slot.completed_subtasks.store(0);

    // First subtask completion
    bool done1 = sched.on_subtask_complete(slot);
    EXPECT_FALSE(done1) << "Single completion doesn't complete the task";

    // Same subtask completes AGAIN (logic error at caller level)
    bool done2 = sched.on_subtask_complete(slot);
    EXPECT_TRUE(done2) << "Counter model: double-completion falsely triggers done";
}
