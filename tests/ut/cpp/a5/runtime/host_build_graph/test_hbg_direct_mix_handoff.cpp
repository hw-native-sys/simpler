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
        ASSERT_TRUE(fixture.bootstrap_admission());
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
        // Completion owns one unlocked successor through admission. An existing
        // queue entry, or excess fanout, remains queued without a second owner.
        const int64_t direct_task = scenario == HandoffScenario::TWO_SUCCESSORS ? 2 : 1;
        const int64_t queued_task = scenario == HandoffScenario::TWO_SUCCESSORS ? 1 : 2;
        const auto &admission = fixture.admissions[producer_lane];
        const auto *shared = scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), admission.control_offset);
        const auto *log = scheduler_state_at<int64_t>(fixture.base(), admission.log_offset);
        ASSERT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(log[0], direct_task);
        EXPECT_EQ(
            fixture.control(direct_task)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
        );
        EXPECT_EQ(admission.pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_EQ(queue.control->reserved_tail - queue.control->head, second_mix ? 1u : 0u);
        if (second_mix) {
            EXPECT_EQ(queue.entries[queue.control->head].task_ref, queued_task);
            EXPECT_EQ(
                fixture.control(queued_task)->admission_state,
                static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
            );
        }
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
        const std::vector<int64_t> expected_entries =
            second_mix ? std::vector<int64_t>{direct_task, queued_task} : std::vector<int64_t>{direct_task};
        EXPECT_EQ(first_entries, expected_entries);
        ASSERT_EQ(shared->commit_tail, expected_entries.size());
        for (size_t index = 0; index < expected_entries.size(); ++index)
            EXPECT_EQ(log[index], expected_entries[index]);
        for (uint32_t task = 0; task < shapes.size(); ++task)
            EXPECT_EQ(fixture.control(task)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(queue.control->head, queue.control->reserved_tail);
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
};

TEST_P(HbgDirectMixHandoff, RetirementKeepsUnlockedMixReachable) { run_handoff(true); }
TEST_P(HbgDirectMixHandoff, UnprofiledRetirementKeepsUnlockedMixReachable) { run_handoff(false); }
TEST_P(HbgDirectMixHandoff, DirectAdmissionPreservesAnExistingQueuedMix) {
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

TEST(HbgDirectAdmissionRetry, CsrAndWakeListRetainCrossTypeSuccessorsBehindAnExistingPendingOwner) {
    for (bool csr : {false, true}) {
        for (const auto child_mask : {LaneMask::aiv0(), LaneMask::both_aiv(), LaneMask::all()}) {
            SCOPED_TRACE(csr);
            SCOPED_TRACE(lane_names(child_mask));
            SchedulerDependencyPlan dependencies{};
            GmMixFixture fixture(
                {{2, LaneMask::aic()}, {2, child_mask}, {2, LaneMask::aic()}}, false, 2,
                [&](AicoreSchedulerLayout &layout) {
                    if (!csr) return;
                    EXPECT_TRUE(scheduler_plan_dependencies(3, 1, layout.total_size, &dependencies));
                    layout.total_size = dependencies.total_size;
                }
            );
            fixture.graph.executable(1, 1, {0});
            fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
            ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
            if (csr) {
                auto *pool =
                    scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.storage.ready_pool_offset);
                pool->dependencies_offset = dependencies.header_offset;
                pool->storage_end_offset = dependencies.total_size;
                ASSERT_TRUE(scheduler_init_dependencies(fixture.base(), dependencies, fixture.graph.graph(), {}));
            }
            ASSERT_TRUE(fixture.bootstrap_admission());
            ASSERT_EQ(
                scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1),
                SchedulerRouteResult::WAITING
            );
            ASSERT_TRUE(fixture.admit(0));
            ASSERT_TRUE(fixture.claim_pending(2));
            fixture.control(0)->next_block = 2;
            fixture.control(0)->claim_seal = 2;
            fixture.control(0)->completion_state |= 2;
            ASSERT_TRUE(scheduler_complete_task(
                fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, false, 0, nullptr
            ));
            EXPECT_EQ(fixture.admissions[0].pending_task_ref, 2);
            EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
            unsigned retained = 0;
            for (const auto &candidate : fixture.admissions[0].candidates) {
                if (candidate.ready.task_ref != 1) continue;
                ++retained;
                EXPECT_EQ(candidate.slot_index, UINT8_MAX);
            }
            ASSERT_EQ(retained, 1u);
            ASSERT_EQ(
                scheduler_ordered_publish_pending(
                    fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()
                ),
                SchedulerAdmissionResult::PROGRESS
            );
            ASSERT_TRUE(fixture.fill(0));
            EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
            auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
                fixture.base(), fixture.storage.layout.admission_control_offset
            );
            ASSERT_EQ(shared->commit_tail, 3u);
            auto *log = scheduler_state_at<int64_t>(fixture.base(), fixture.storage.layout.admission_log_offset);
            EXPECT_EQ(log[0], 0);
            EXPECT_EQ(log[1], 2);
            EXPECT_EQ(log[2], 1);
            for (const auto &slot : fixture.locals[0]->slots)
                EXPECT_NE(slot.task_ref, 1);
            EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        }
    }
}
}  // namespace
