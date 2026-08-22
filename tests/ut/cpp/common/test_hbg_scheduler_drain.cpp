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

uint64_t __attribute__((weak)) read_reg(uint64_t, RegId) { return 0; }

void __attribute__((weak)) reg_store_release(volatile uint32_t *, uint32_t) {}

extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }

extern "C" uint64_t get_platform_regs() { return 0; }

int SchedulerContext::prepare_block_for_dispatch(
    int32_t, int32_t, PTO2TaskSlotState &, PTO2ResourceShape, bool, int32_t, PublishHandle *, bool
) {
    return 0;
}

class SchedulerContextTestPeer {
public:
    static void leave_ack_barrier_after_reopen(SchedulerContext &context) {
        constexpr int32_t kActiveThreads = 2;
        constexpr int32_t kFollowerThread = 1;
        constexpr uint64_t kAttempt = 41;
        const uint64_t subtree_token = sync_start_drain_ack_subtree_token(kAttempt);

        context.active_sched_threads_ = kActiveThreads;
        context.completed_.store(false, std::memory_order_relaxed);
        context.drain_state_.sync_start_pending.store(1, std::memory_order_relaxed);
        context.drain_state_.drain_attempt.store(kAttempt, std::memory_order_relaxed);

        context.drain_ack_tokens_[0].store(subtree_token, std::memory_order_release);
        context.drain_state_.pending_task.store(nullptr, std::memory_order_release);

        context.handle_drain_mode(kFollowerThread);
    }
};

TEST(SchedulerDrainTest, FollowerReturnsWhenCoordinatorReopenedAfterAckPublication) {
    SchedulerContext context;

    SchedulerContextTestPeer::leave_ack_barrier_after_reopen(context);
}
