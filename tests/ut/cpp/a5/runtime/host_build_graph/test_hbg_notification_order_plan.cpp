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

#include "notification_order_plan.h"

namespace {
SchedulerNotificationIdentity counter(uint64_t offset = 0, uint64_t epoch = 0) {
    return {19, 3, 7, 11, 23, offset, 0, epoch};
}
SchedulerNotificationEvent event(
    SchedulerNotificationKind kind, int64_t ref, uint64_t value, uint32_t rank = 0,
    SchedulerNotificationIdentity identity = counter(), uint64_t index = 0
) {
    return {identity, rank, ref, index, kind, value};
}

TEST(HbgNotificationOrderPlan, ExactCounterAndProvenEpochSelectTheLaterSubmittedProducer) {
    std::vector<SchedulerNotificationEvent> events{
        event(SchedulerNotificationKind::WAIT, 0, 3, 0, counter(4, 1)),
        event(SchedulerNotificationKind::SET, 6, 3, 0, counter(4, 1)),
        event(SchedulerNotificationKind::SET, 7, 3, 0, counter(8, 1)),
        event(SchedulerNotificationKind::SET, 5, 4, 0, counter(4, 2))
    };
    SchedulerNotificationOrderPlan plan;
    std::string error;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    ASSERT_EQ(plan.local_edges.size(), 1u);
    EXPECT_EQ(plan.local_edges[0].predecessor, 6);
    EXPECT_EQ(plan.local_edges[0].successor, 0);
    EXPECT_TRUE(plan.unresolved_waits.empty());
    const SchedulerNotificationDataBefore wait_before_set = [](uint32_t rank, int64_t predecessor, int64_t successor) {
        return rank == 0 && predecessor == 0 && successor == 6;
    };
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error, wait_before_set)) << error;
    EXPECT_TRUE(plan.local_edges.empty());
    EXPECT_EQ(plan.unresolved_waits.size(), 1u);
    events[0].counter.notification_epoch = 0;
    events[1].counter.notification_epoch = 0;
    events[3].counter.notification_epoch = 0;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    EXPECT_TRUE(plan.local_edges.empty());
    ASSERT_EQ(plan.unresolved_waits.size(), 1u);
}

TEST(HbgNotificationOrderPlan, RemoteOnlyWaitKeepsExistingBehaviorAndOutgoingNotifyIsNotItsSource) {
    auto outgoing = counter();
    outgoing.target_rank = 1;
    std::vector<SchedulerNotificationEvent> events{
        event(SchedulerNotificationKind::WAIT, 1, 9), event(SchedulerNotificationKind::SET, 6, 9, 0, outgoing)
    };
    SchedulerNotificationOrderPlan plan;
    std::string error;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    EXPECT_TRUE(plan.local_edges.empty());
    ASSERT_EQ(plan.unresolved_waits.size(), 1u);
    EXPECT_EQ(plan.unresolved_waits[0].node_ref, 1);
    events.push_back(event(SchedulerNotificationKind::SET, 7, 9));
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    ASSERT_EQ(plan.local_edges.size(), 1u);
    EXPECT_EQ(plan.local_edges[0].predecessor, 7);
    EXPECT_EQ(plan.local_edges[0].successor, 1);
    EXPECT_TRUE(plan.unresolved_waits.empty());
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, {}, &plan, &error)) << error;
    EXPECT_TRUE(plan.local_edges.empty());
    EXPECT_TRUE(plan.unresolved_waits.empty());
}

TEST(HbgNotificationOrderPlan, AddKeepsUnknownInitialValueUnresolvedAndCountsDistinctContributions) {
    std::vector<SchedulerNotificationEvent> events{
        event(SchedulerNotificationKind::WAIT, 0, 4), event(SchedulerNotificationKind::ADD, 5, 1, 0, counter(), 0),
        event(SchedulerNotificationKind::ADD, 5, 1, 0, counter(), 1), event(SchedulerNotificationKind::ADD, 6, 1)
    };
    SchedulerNotificationOrderPlan plan;
    std::string error;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    ASSERT_EQ(plan.unresolved_waits.size(), 1u);
    events.push_back(event(SchedulerNotificationKind::INITIAL_VALUE, -1, 1));
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    EXPECT_TRUE(plan.unresolved_waits.empty());
    ASSERT_EQ(plan.local_edges.size(), 2u);
    EXPECT_EQ(plan.local_edges[0].predecessor, 5);
    EXPECT_EQ(plan.local_edges[1].predecessor, 6);
    events[2].event_index = 0;
    events[2].block_index = 1;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    ASSERT_EQ(plan.matches.size(), 1u);
    EXPECT_EQ(plan.matches[0].sources.size(), 4u);
    events[2].block_index = 0;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error)) << error;
    ASSERT_EQ(plan.unresolved_waits.size(), 1u);
    EXPECT_TRUE(plan.local_edges.empty());
}

TEST(HbgNotificationOrderPlan, RepeatedSetUsesMonotonicDataChainAndLeavesUnprovedOrderOrResetUnresolved) {
    std::vector<SchedulerNotificationEvent> events{
        event(SchedulerNotificationKind::WAIT, 6, 2), event(SchedulerNotificationKind::WAIT, 7, 3),
        event(SchedulerNotificationKind::SET, 5, 3),  event(SchedulerNotificationKind::SET, 1, 1),
        event(SchedulerNotificationKind::SET, 3, 2),  event(SchedulerNotificationKind::SET, 3, 2, 0, counter(), 1)
    };
    std::vector<std::vector<int64_t>> successors(8);
    successors[1] = {2};
    successors[2] = {3};
    successors[3] = {4};
    successors[4] = {5};
    const SchedulerNotificationDataBefore data_before = [&](uint32_t rank, int64_t producer, int64_t consumer) {
        if (rank != 0) return false;
        std::vector<int64_t> pending{producer};
        std::set<int64_t> visited;
        while (!pending.empty()) {
            const auto ref = pending.back();
            pending.pop_back();
            if (!visited.insert(ref).second) continue;
            for (const auto next : successors[ref]) {
                if (next == consumer) return true;
                pending.push_back(next);
            }
        }
        return false;
    };
    SchedulerNotificationOrderPlan plan;
    std::string error;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error, data_before)) << error;
    ASSERT_EQ(plan.local_edges.size(), 2u);
    EXPECT_EQ(plan.local_edges[0].predecessor, 3);
    EXPECT_EQ(plan.local_edges[0].successor, 6);
    EXPECT_EQ(plan.local_edges[1].predecessor, 5);
    EXPECT_EQ(plan.local_edges[1].successor, 7);
    ASSERT_EQ(plan.matches.size(), 2u);
    EXPECT_EQ(plan.matches[0].sources.size(), 4u);
    EXPECT_EQ(plan.matches[0].wait.counter.notification_epoch, 0u);
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error));
    EXPECT_EQ(plan.unresolved_waits.size(), 2u);
    successors[2].clear();
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error, data_before));
    EXPECT_EQ(plan.unresolved_waits.size(), 2u);
    successors[2] = {3};
    events[2].value = 1;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error, data_before));
    EXPECT_EQ(plan.unresolved_waits.size(), 2u);
    events[2].value = 3;
    events[5].value = 3;
    ASSERT_TRUE(scheduler_plan_notification_order(0, 8, events, &plan, &error, data_before));
    EXPECT_EQ(plan.unresolved_waits.size(), 2u);
}
}  // namespace
