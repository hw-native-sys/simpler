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

#include "hbg_gm_mix_test_support.h"

namespace {
using namespace scheduler_test::gm_mix;

enum class HandoffScenario { EMPTY_QUEUE, QUEUED_MIX, TWO_SUCCESSORS, BUSY_LANE };

class HbgDirectMixHandoff : public ::testing::TestWithParam<std::tuple<uint32_t, LaneMask>> {
protected:
    void run_handoff(bool protocol, HandoffScenario scenario = HandoffScenario::EMPTY_QUEUE) {
        const auto [producer_lane, mask] = GetParam();
        SCOPED_TRACE(producer_lane);
        SCOPED_TRACE(lane_names(mask));
        const bool second_mix = scenario == HandoffScenario::QUEUED_MIX || scenario == HandoffScenario::TWO_SUCCESSORS;
        const bool busy = scenario == HandoffScenario::BUSY_LANE;
        const LaneMask ordinary = producer_lane == 0 ? LaneMask::aic() : LaneMask::aiv0();
        std::vector<GmShape> shapes{{1, ordinary}, {1, mask}};
        if (second_mix) shapes.push_back({1, mask});
        const uint32_t held_worker = producer_lane != 0 ? 0 : (mask.has_aiv0() ? 1 : 2);
        if (busy) shapes.push_back({1, held_worker == 0 ? LaneMask::aic() : LaneMask::aiv0()});
        GmMixFixture fixture(shapes, protocol, 1);
        const uint8_t first = mask.has_aic() ? 0 : 1;
        fixture.graph.executable(1, first, {0});
        fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_EQ(
            scheduler_route_task(
                fixture.graph.graph(), fixture.base(), fixture.locals[producer_lane], fixture.run(), 1
            ),
            SchedulerRouteResult::WAITING
        );
        if (scenario == HandoffScenario::TWO_SUCCESSORS) {
            fixture.graph.executable(2, first, {0});
            fixture.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
            ASSERT_EQ(
                scheduler_route_task(
                    fixture.graph.graph(), fixture.base(), fixture.locals[producer_lane], fixture.run(), 2
                ),
                SchedulerRouteResult::WAITING
            );
        }
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.enqueue(0, producer_lane));
        ASSERT_TRUE(fixture.fill(producer_lane));
        GmExecution producer{};
        ASSERT_TRUE(fixture.begin(producer_lane, &producer));
        ASSERT_EQ(producer.task, 0);
        GmExecution held{};
        if (busy) {
            ASSERT_TRUE(fixture.enqueue(2, held_worker));
            ASSERT_TRUE(fixture.fill(held_worker));
            ASSERT_TRUE(fixture.begin(held_worker, &held));
            ASSERT_EQ(held.task, 2);
        }
        if (scenario == HandoffScenario::QUEUED_MIX) ASSERT_TRUE(fixture.enqueue(2));
        ASSERT_TRUE(fixture.complete(producer));
        EXPECT_EQ(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        const auto *pool =
            scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.storage.ready_pool_offset);
        const auto queue = scheduler_ready_pool_queue(fixture.base(), *pool, 0);
        // One successor belongs to ordered admission, never an ordinary slot.
        // Existing queued tickets retain their head and FIFO order.
        const int64_t direct_task = scenario == HandoffScenario::TWO_SUCCESSORS ? 2 : 1;
        std::set<int64_t> reachable;
        for (const auto &candidate : fixture.admissions[producer_lane].candidates) {
            if (scheduler_task_ref_is_null(candidate.ready.task_ref)) continue;
            EXPECT_EQ(candidate.ready.task_ref, direct_task);
            EXPECT_EQ(candidate.ready.source, SchedulerReadySource::DIRECT_RESOLVE);
            EXPECT_TRUE(reachable.insert(candidate.ready.task_ref).second);
        }
        EXPECT_EQ(reachable.size(), 1u);
        EXPECT_EQ(fixture.locals[producer_lane]->slots[producer.slot].state, SchedulerDispatchSlotState::FREE);
        ASSERT_EQ(queue.control->reserved_tail - queue.control->head, second_mix ? 1u : 0u);
        if (second_mix) {
            const int64_t queued = scenario == HandoffScenario::QUEUED_MIX ? 2 : 1;
            EXPECT_EQ(queue.entries[queue.control->head].task_ref, queued);
            EXPECT_TRUE(reachable.insert(queued).second);
        }
        EXPECT_EQ(reachable, second_mix ? (std::set<int64_t>{1, 2}) : (std::set<int64_t>{1}));
        for (uint32_t worker : {1u, 2u}) {
            SchedulerReadyClaim aiv{};
            ASSERT_TRUE(scheduler_claim_ready_for_slot(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 3, 1, &aiv
            ));
            EXPECT_EQ(aiv.task_ref, SCHEDULER_TASK_REF_INVALID);
        }
        if (busy) {
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.fill(worker)) << "worker=" << worker << " site=" << fixture.run()->error_site
                                                  << " task=" << fixture.run()->error_task_ref;
            for (uint32_t worker = 0; worker < 3; ++worker) {
                GmExecution premature{};
                EXPECT_FALSE(fixture.begin(worker, &premature));
            }
            EXPECT_EQ(fixture.locals[held_worker]->slots[held.slot].state, SchedulerDispatchSlotState::EXECUTING);
            EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 0u);
            ASSERT_TRUE(fixture.complete(held));
        }
        std::set<std::tuple<int64_t, uint32_t, uint8_t>> executions;
        std::vector<int64_t> first_entries;
        bool finished = false;
        for (uint32_t step = 0; step < 128 && !finished; ++step) {
            for (uint32_t worker = 0; worker < 3; ++worker) {
                ASSERT_TRUE(fixture.fill(worker)) << "worker=" << worker << " site=" << fixture.run()->error_site
                                                  << " task=" << fixture.run()->error_task_ref;
                GmExecution execution{};
                if (!fixture.begin(worker, &execution)) continue;
                EXPECT_GE(execution.task, 1);
                EXPECT_TRUE(executions.emplace(execution.task, execution.block, execution.subtask).second);
                if (std::find(first_entries.begin(), first_entries.end(), execution.task) == first_entries.end())
                    first_entries.push_back(execution.task);
                ASSERT_TRUE(fixture.complete(execution));
            }
            uint64_t completed = 0;
            for (const auto *local : fixture.locals)
                completed += local->pending_completed;
            finished = completed == shapes.size();
        }
        ASSERT_TRUE(finished);
        EXPECT_EQ(executions.size(), static_cast<size_t>(mask.count()) * (second_mix ? 2 : 1));
        EXPECT_EQ(first_entries.size(), second_mix ? 2u : 1u);
        EXPECT_EQ(
            (std::set<int64_t>(first_entries.begin(), first_entries.end())),
            second_mix ? (std::set<int64_t>{1, 2}) : (std::set<int64_t>{1})
        );
        for (uint32_t task = 0; task < shapes.size(); ++task)
            EXPECT_EQ(fixture.control(task)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(queue.control->head, queue.control->reserved_tail);
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
};

TEST_P(HbgDirectMixHandoff, RetirementKeepsUnlockedMixReachable) { run_handoff(true); }
TEST_P(HbgDirectMixHandoff, UnprofiledRetirementKeepsUnlockedMixReachable) { run_handoff(false); }
TEST_P(HbgDirectMixHandoff, UnlockedMixDoesNotDisplaceAnExistingQueueHead) {
    run_handoff(true, HandoffScenario::QUEUED_MIX);
}
TEST_P(HbgDirectMixHandoff, TwoMixSuccessorsUnlockedTogetherExecuteExactlyOnce) {
    run_handoff(true, HandoffScenario::TWO_SUCCESSORS);
}
TEST_P(HbgDirectMixHandoff, BusyOrdinaryLaneRetainsItsSlotUntilRetirement) {
    run_handoff(true, HandoffScenario::BUSY_LANE);
}
INSTANTIATE_TEST_SUITE_P(
    ProducerLanesAndMixMasks, HbgDirectMixHandoff,
    ::testing::Combine(::testing::Values(0u, 1u, 2u), ::testing::ValuesIn(kMixMasks))
);
}  // namespace
