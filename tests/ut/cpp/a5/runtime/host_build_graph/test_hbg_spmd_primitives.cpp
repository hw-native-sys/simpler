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

#include "hbg_scheduler_test_support.h"

namespace {

using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;

void append_ready(FixtureStorage &storage, int64_t first, int64_t end) {
    SchedulerReadyBatch batch{};
    auto *local = storage.local_context(&storage.contexts[0]);
    for (int64_t task = first; task < end; ++task) {
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, task, &batch));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch));
}

TEST(HbgSpmdPrimitiveBoundary, PositiveHintDoesNotOwnOrPinTheNextTask) {
    FixtureStorage storage(3, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(3);
    append_ready(storage, 0, 3);
    auto *first = storage.local_context(&storage.contexts[0]);
    auto *second = storage.local_context(&storage.contexts[3]);
    EXPECT_TRUE(scheduler_ready_pool_may_have_work(storage.scheduler_state->base(), first, 0));
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), second, storage.run_control, 6, 0, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), first, storage.run_control, 6, 0, &claim
    ));
    EXPECT_EQ(claim.task_ref, 1);
    const auto *pool =
        scheduler_state_at<SchedulerReadyPoolHeader>(storage.scheduler_state->base(), first->config.ready_pool_offset);
    const auto queue = scheduler_ready_pool_queue(storage.scheduler_state->base(), *pool, 0);
    EXPECT_EQ(queue.control->head, 2u);
}

TEST(HbgSpmdPrimitiveBoundary, TailAppendDoesNotChangeTheHead) {
    FixtureStorage storage(2, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    append_ready(storage, 0, 1);
    auto *local = storage.local_context(&storage.contexts[0]);
    const auto *pool =
        scheduler_state_at<SchedulerReadyPoolHeader>(storage.scheduler_state->base(), local->config.ready_pool_offset);
    const auto queue = scheduler_ready_pool_queue(storage.scheduler_state->base(), *pool, 0);
    const uint64_t head = scheduler_gm_query(queue.control->head);
    EXPECT_EQ(queue.control->reserved_tail, 1u);
    append_ready(storage, 1, 2);
    EXPECT_EQ(queue.control->head, head);
    EXPECT_EQ(queue.control->reserved_tail, 2u);
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 3, 0, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(queue.control->head, head + 1);
}

TEST(HbgSpmdPrimitiveBoundary, AnEmptyHintDoesNotProveFutureQueueQuiescence) {
    FixtureStorage storage(1, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    auto *local = storage.local_context(&storage.contexts[0]);
    const bool empty_hint = !scheduler_ready_pool_may_have_work(storage.scheduler_state->base(), local, 0);
    ASSERT_TRUE(empty_hint);
    append_ready(storage, 0, 1);
    EXPECT_TRUE(scheduler_ready_pool_may_have_work(storage.scheduler_state->base(), local, 0));
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 3, 0, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
}

TEST(HbgSpmdPrimitiveBoundary, PeekPreservesFifoAndParkedSlotRejoinsAtTheTail) {
    SchedulerLocalState local{};
    for (auto &slot : local.slots)
        slot.state = SchedulerDispatchSlotState::READY;
    scheduler_local_ready_publish(&local, 1);
    scheduler_local_ready_publish(&local, 0);
    uint32_t head = UINT32_MAX;
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 1u);
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 1u);
    EXPECT_FALSE(scheduler_local_ready_execute(&local, 0));
    EXPECT_EQ(local.local_ready_mask, 3u);
    scheduler_local_ready_remove(&local, 1);
    scheduler_local_ready_publish(&local, 1);
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 0u);
    ASSERT_TRUE(scheduler_local_ready_execute(&local, 0));
    EXPECT_EQ(local.slots[0].state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_FALSE(scheduler_local_ready_execute(&local, 0));
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 1u);
    ASSERT_TRUE(scheduler_local_ready_execute(&local, 1));
    EXPECT_EQ(local.local_ready_mask, 0u);
    EXPECT_FALSE(scheduler_local_ready_peek(&local, &head));
}

}  // namespace
