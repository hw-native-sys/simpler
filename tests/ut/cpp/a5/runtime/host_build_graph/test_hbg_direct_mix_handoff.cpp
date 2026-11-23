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
#include "dependency_plan.h"

namespace {
using namespace scheduler_test::gm_mix;

enum class HandoffScenario { SINGLE_SUCCESSOR, INDEPENDENT_MIX, TWO_SUCCESSORS, BUSY_LANE };

class HbgDirectMixHandoff : public ::testing::TestWithParam<std::tuple<uint32_t, LaneMask>> {
protected:
    void run_handoff(bool protocol, HandoffScenario scenario = HandoffScenario::SINGLE_SUCCESSOR) {
        const auto [producer_lane, mask] = GetParam();
        SCOPED_TRACE(producer_lane);
        SCOPED_TRACE(lane_names(mask));
        const bool second_mix =
            scenario == HandoffScenario::INDEPENDENT_MIX || scenario == HandoffScenario::TWO_SUCCESSORS;
        const bool busy = scenario == HandoffScenario::BUSY_LANE;
        const bool independent = scenario == HandoffScenario::INDEPENDENT_MIX;
        const uint32_t successor = busy || independent ? 2 : 1;
        const LaneMask ordinary = producer_lane == 0 ? LaneMask::aic() : LaneMask::aiv0();
        const uint32_t held_worker = producer_lane != 0 ? 0 : (mask.has_aiv0() ? 1 : 2);
        std::vector<GmShape> shapes{{1, ordinary}};
        if (busy) shapes.push_back({1, held_worker == 0 ? LaneMask::aic() : LaneMask::aiv0()});
        if (independent) shapes.push_back({1, mask});
        shapes.push_back({1, mask});
        if (scenario == HandoffScenario::TWO_SUCCESSORS) shapes.push_back({1, mask});
        GmMixFixture fixture(shapes, protocol, 1);
        const uint8_t first = mask.has_aic() ? 0 : 1;
        fixture.graph.executable(successor, first, {0});
        if (scenario == HandoffScenario::TWO_SUCCESSORS) {
            fixture.graph.executable(2, first, {0});
        }
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.claim_ordinary(0, producer_lane));
        ASSERT_TRUE(fixture.progress_ordinary(producer_lane));
        GmExecution producer{};
        ASSERT_TRUE(fixture.begin(producer_lane, &producer));
        ASSERT_EQ(producer.task, 0);
        GmExecution held{};
        if (busy) {
            ASSERT_TRUE(fixture.claim_ordinary(1, held_worker));
            ASSERT_TRUE(fixture.progress_ordinary(held_worker));
            ASSERT_TRUE(fixture.begin(held_worker, &held));
            ASSERT_EQ(held.task, 1);
        }
        // Service can distribute an independent MIX or a successor descriptor.
        // Successor kernels must still wait for the producer's real return.
        for (uint32_t pass = 0; pass < 4; ++pass) {
            for (uint32_t worker = 0; worker < 3; ++worker) {
                if (worker == producer_lane || (busy && worker == held_worker)) continue;
                ASSERT_TRUE(fixture.fill(worker));
                const auto &mix = fixture.locals[worker]->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION);
                if (independent && mix.task_ref == 1) continue;
                EXPECT_NE(mix.state, SchedulerDispatchSlotState::READY);
                GmExecution premature{};
                ASSERT_FALSE(fixture.begin(worker, &premature));
            }
        }
        EXPECT_FALSE(fixture.task_done(0));
        EXPECT_EQ(scheduler_completion_count(fixture.control(successor)->completion_state), 0u);
        std::set<std::tuple<int64_t, uint32_t, uint8_t>> executions;
        std::vector<int64_t> first_entries;
        if (independent && producer_lane == 0 && mask == LaneMask::both_aiv()) {
            GmExecution independent_entry{};
            ASSERT_TRUE(fixture.begin(1, &independent_entry));
            ASSERT_EQ(independent_entry.task, 1);
            ASSERT_FALSE(fixture.task_done(0));
            executions.emplace(independent_entry.task, independent_entry.block, independent_entry.subtask);
            first_entries.push_back(independent_entry.task);
            ASSERT_TRUE(fixture.complete(independent_entry));
        }
        ASSERT_TRUE(fixture.complete(producer));
        EXPECT_TRUE(fixture.task_done(0));
        if (busy) {
            for (uint32_t worker = 0; worker < 3; ++worker) {
                if (worker == held_worker) continue;
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution member{};
                if (!fixture.begin(worker, &member)) continue;
                ASSERT_EQ(member.task, successor);
                EXPECT_TRUE(executions.emplace(member.task, member.block, member.subtask).second);
                if (first_entries.empty()) first_entries.push_back(member.task);
                ASSERT_TRUE(fixture.complete(member));
            }
            if (producer_lane == 0 && mask == kAicAiv1) {
                EXPECT_EQ(executions.count({successor, 0, 0}), 1u);
            }
            const auto &owner = fixture.locals[held_worker]->execution_at(held.slot);
            EXPECT_EQ(owner.task_ref, held.task);
            EXPECT_EQ(owner.state, SchedulerDispatchSlotState::EXECUTING);
            if (mask.has(static_cast<SubtaskSlot>(held_worker))) {
                EXPECT_FALSE(fixture.task_done(successor));
                EXPECT_EQ(scheduler_completion_count(fixture.control(successor)->completion_state), 0u);
            }
            ASSERT_TRUE(fixture.complete(held));
        }
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
            finished = fixture.storage.completed_gm_records() == shapes.size();
        }
        ASSERT_TRUE(finished);
        EXPECT_EQ(executions.size(), static_cast<size_t>(mask.count()) * (second_mix ? 2 : 1));
        const std::vector<int64_t> expected_tasks =
            second_mix ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{successor};
        std::sort(first_entries.begin(), first_entries.end());
        EXPECT_EQ(first_entries, expected_tasks);
        for (uint32_t task = 0; task < shapes.size(); ++task)
            EXPECT_EQ(
                scheduler_completion_task_state(fixture.control(task)->completion_state),
                static_cast<int64_t>(SchedulerTaskState::DONE)
            );
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
};

TEST_P(HbgDirectMixHandoff, DiscoveryFindsMixAfterProducerRetirement) { run_handoff(true); }
TEST_P(HbgDirectMixHandoff, UnprofiledDiscoveryFindsMixAfterProducerRetirement) { run_handoff(false); }
TEST_P(HbgDirectMixHandoff, DiscoveryPreservesAnIndependentAvailableMix) {
    run_handoff(true, HandoffScenario::INDEPENDENT_MIX);
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

TEST(HbgDirectMixHandoffOrder, AllEarlierSyncBlocksCanRunBeforeAnIndependentLaterCohort) {
    for (const auto successor : {LaneMask::aiv0(), LaneMask::both_aiv(), LaneMask::all()}) {
        GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, LaneMask::aic(), true}, {2, successor, true}});
        fixture.graph.executable(2, successor.has_aic() ? 0 : 1, {0});
        ASSERT_TRUE(fixture.prepare_calls());
        std::set<std::tuple<int64_t, uint32_t, uint8_t>> completed;
        for (uint32_t pass = 0; pass < 64 && fixture.storage.completed_gm_records() != 3; ++pass) {
            for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution unit{};
                if (!fixture.begin(worker, &unit)) continue;
                if (unit.task == 2) EXPECT_TRUE(fixture.task_done(0));
                EXPECT_TRUE(completed.emplace(unit.task, unit.block, unit.subtask).second);
                ASSERT_TRUE(fixture.complete(unit));
            }
        }
        EXPECT_EQ(completed.size(), 4u + 2u * successor.count());
        EXPECT_EQ(fixture.storage.completed_gm_records(), 3u);
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
}
}  // namespace
