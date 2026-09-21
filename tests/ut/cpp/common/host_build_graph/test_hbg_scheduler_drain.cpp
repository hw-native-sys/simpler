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

#include <cstdint>

#include "scheduler/scheduler_context.h"

// Weak, because the a2a3 target also links sim/aicpu/inner_platform_regs.cpp, which
// defines these strongly and overrides the definitions here. The a5 target links no
// equivalent, so it supplies its own strong stubs in
// test_hbg_scheduler_drain_a5_stubs.cpp. Both targets compile this file.
uint64_t __attribute__((weak)) read_reg(uint64_t, RegId) { return 0; }

void __attribute__((weak)) reg_store_release(volatile uint32_t *, uint32_t) {}

extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }

extern "C" uint64_t get_platform_regs() { return 0; }

int SchedulerContext::prepare_block_for_dispatch(
    int32_t, int32_t, ChipTaskSlotState &, ResourceShape, bool, int32_t, PublishHandle *, bool
) {
    return 0;
}

class SchedulerContextTestPeer {
public:
    // The state a thread finds when it enters handle_drain_mode after the drain it
    // entered for has already reopened: the ack token for this round is published, so
    // the reduction and the follower barrier both pass without spinning, while
    // pending_task is already cleared. sync_start_pending stays non-zero because the
    // reopen clears it after pending_task -- that ordering is what lets a thread get
    // this far -- and drain_attempt still names this round, so neither barrier
    // early-exit fires.
    static void leave_ack_barrier_after_reopen(SchedulerContext &context) {
        constexpr int32_t kActiveThreads = 2;
        constexpr int32_t kFollowerThread = 1;
        constexpr uint64_t kAttempt = 41;
        const uint64_t subtree_token = sync_start_drain_ack_subtree_token(kAttempt);

        context.active_sched_threads_ = kActiveThreads;
        context.completed_.store(false, std::memory_order_relaxed);
        context.drain_state_.sync_start_pending.store(1, std::memory_order_relaxed);
        context.drain_state_.drain_attempt.store(kAttempt, std::memory_order_relaxed);
        context.drain_state_.drain_stage_go.store(0, std::memory_order_relaxed);
        context.drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);

        context.drain_ack_tokens_[0].store(subtree_token, std::memory_order_release);
        context.drain_state_.pending_task.store(nullptr, std::memory_order_release);

        context.handle_drain_mode(kFollowerThread);
    }

    static uint32_t stage_done_mask(const SchedulerContext &context) {
        return context.drain_state_.drain_stage_done_mask.load(std::memory_order_relaxed);
    }
    static int32_t sync_start_pending(const SchedulerContext &context) {
        return context.drain_state_.sync_start_pending.load(std::memory_order_relaxed);
    }
    static uint32_t drain_stage_go(const SchedulerContext &context) {
        return context.drain_state_.drain_stage_go.load(std::memory_order_relaxed);
    }
};

// Surviving the call only proves the load is no longer dereferenced. What the guard
// owes is that the late arrival does nothing: it must not stage, must not ack for a
// round whose acks were already collected, and must not reopen a gate it does not own.
// Asserting that is what separates an early return from a fallback that keeps going.
TEST(SchedulerDrainTest, FollowerReturnsWhenCoordinatorReopenedAfterAckPublication) {
    SchedulerContext context;

    SchedulerContextTestPeer::leave_ack_barrier_after_reopen(context);

    EXPECT_EQ(SchedulerContextTestPeer::stage_done_mask(context), 0u)
        << "a thread that arrived after reopen must not report staging";
    EXPECT_EQ(SchedulerContextTestPeer::drain_stage_go(context), 0u)
        << "only the coordinator of a live round releases staging";
    EXPECT_EQ(SchedulerContextTestPeer::sync_start_pending(context), 1) << "the late arrival must not reopen the gate";
}
