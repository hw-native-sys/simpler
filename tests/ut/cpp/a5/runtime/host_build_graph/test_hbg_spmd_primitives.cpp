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
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, task, &batch, nullptr));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch, nullptr));
}

TEST(HbgSpmdPrimitiveBoundary, GenericPopSnapshotDoesNotPinTheExpectedTask) {
    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    append_ready(storage, 0, 3);
    auto *local = storage.local_context(&storage.contexts[0]);
    auto *inbox = scheduler_ready_inbox_at(storage.scheduler_state->base(), local, 0, 0);
    const uint64_t observed_a = scheduler_gm_query(inbox->endpoints);
    SchedulerReadyStats stats{};
    int64_t popped = SCHEDULER_TASK_REF_INVALID;

    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &popped, &stats, nullptr,
        &observed_a
    ));
    EXPECT_EQ(popped, 0);

    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &popped, &stats, nullptr,
        &observed_a
    ));
    EXPECT_EQ(popped, 1);
    EXPECT_EQ(stats.cas_retry_count, 1u);
    EXPECT_EQ(scheduler_ready_endpoints_head(scheduler_gm_query(inbox->endpoints)), 2);
}

TEST(HbgSpmdPrimitiveBoundary, TailAppendChangesEndpointsWithoutChangingTheHead) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    append_ready(storage, 0, 1);
    auto *local = storage.local_context(&storage.contexts[0]);
    auto *inbox = scheduler_ready_inbox_at(storage.scheduler_state->base(), local, 0, 0);
    const uint64_t before_append = scheduler_gm_query(inbox->endpoints);
    append_ready(storage, 1, 2);
    SchedulerReadyStats stats{};
    int64_t popped = SCHEDULER_TASK_REF_INVALID;

    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &popped, &stats, nullptr,
        &before_append
    ));
    EXPECT_EQ(popped, 0);
    EXPECT_EQ(stats.cas_retry_count, 1u);
    EXPECT_EQ(scheduler_ready_endpoints_head(scheduler_gm_query(inbox->endpoints)), 1);
}

TEST(HbgSpmdPrimitiveBoundary, AnEmptySnapshotDoesNotProveCurrentQueueQuiescence) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    auto *local = storage.local_context(&storage.contexts[0]);
    const uint64_t empty = SCHEDULER_READY_ENDPOINTS_EMPTY;
    append_ready(storage, 0, 1);
    int64_t popped = SCHEDULER_TASK_REF_INVALID;

    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &popped, nullptr, nullptr,
        &empty
    ));
    EXPECT_EQ(popped, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &popped, nullptr
    ));
    EXPECT_EQ(popped, 0);
}

TEST(HbgSpmdPrimitiveBoundary, OwnAndRemoteMixGenerationsAreIndependentDomains) {
    SchedulerLocalState local{};
    local.dispatch_sequence = 100;
    const SchedulerFreeSlotClaim own{0, 0, 0, 0, false};
    const SchedulerFreeSlotClaim remote_mix{1, 0, 0, 1, true};
    EXPECT_EQ(local.next_slot_generation(own), 101u);
    EXPECT_EQ(local.next_slot_generation(remote_mix), 1u);
    EXPECT_EQ(local.dispatch_sequence, 101u);
    EXPECT_EQ(local.mix_dispatch_sequences[1], 1u);
}

}  // namespace
