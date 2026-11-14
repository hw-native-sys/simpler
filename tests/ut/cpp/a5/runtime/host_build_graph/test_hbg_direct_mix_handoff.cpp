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
        fixture.storage.metadata[successor].flags |= SCHEDULER_TASK_HAS_FANIN;
        if (scenario == HandoffScenario::TWO_SUCCESSORS) {
            fixture.graph.executable(2, first, {0});
            fixture.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
        }
        ASSERT_TRUE(fixture.bootstrap_admission());
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
            EXPECT_EQ(
                fixture.control(successor)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
            );
            EXPECT_EQ(fixture.locals[producer_lane]->slots[producer.slot].state, SchedulerDispatchSlotState::EXECUTING);
            ASSERT_EQ(held.task, 1);
        }
        if (independent) {
            bool progress = false;
            // Task 1 is the shared issue head. Discovery cannot skip its
            // unowned block to reserve task 2, even though task 2 is visible.
            ASSERT_TRUE(scheduler_cluster_mix_discover_task(
                fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 2, &progress
            ));
            EXPECT_TRUE(scheduler_task_ref_is_null(fixture.locals[0]->cluster_pending.task_ref));
            EXPECT_EQ(fixture.control(2)->next_block, 0u);
            ASSERT_TRUE(scheduler_cluster_mix_discover_task(
                fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(fixture.locals[0]->cluster_pending.task_ref, 1);
        }
        ASSERT_TRUE(fixture.complete(producer));
        EXPECT_EQ(
            scheduler_completion_task_state(fixture.control(0)->completion_state),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
        EXPECT_EQ(
            fixture.control(successor)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
        );
        if (independent) {
            EXPECT_EQ(fixture.locals[0]->cluster_pending.task_ref, 1);
            EXPECT_EQ(fixture.control(1)->next_block, 1u);
            EXPECT_EQ(fixture.control(2)->next_block, 0u);
        }
        if (scenario == HandoffScenario::TWO_SUCCESSORS)
            EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
        EXPECT_EQ(fixture.storage.layout.admission_control_offset, 0u);
        for (const auto *local : fixture.locals)
            EXPECT_EQ(local->admission, nullptr);
        if (busy) {
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.fill(worker)) << "worker=" << worker << " site=" << fixture.run()->error_site
                                                  << " task=" << fixture.run()->error_task_ref;
            for (uint32_t worker = 0; worker < 3; ++worker) {
                GmExecution premature{};
                EXPECT_FALSE(fixture.begin(worker, &premature));
            }
            EXPECT_EQ(fixture.locals[held_worker]->slots[held.slot].state, SchedulerDispatchSlotState::EXECUTING);
            EXPECT_EQ(scheduler_completion_count(fixture.control(successor)->completion_state), 0u);
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
        const std::vector<int64_t> expected_tasks =
            second_mix ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{successor};
        std::sort(first_entries.begin(), first_entries.end());
        EXPECT_EQ(first_entries, expected_tasks);
        for (const int64_t task : expected_tasks)
            EXPECT_EQ(
                fixture.control(task)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
            );
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

TEST(HbgDirectAdmissionRetry, SyncSuccessorsRemainDiscoverableBehindAnExistingPendingOwner) {
    for (const auto child_mask : {LaneMask::aiv0(), LaneMask::both_aiv(), LaneMask::all()}) {
        SCOPED_TRACE(lane_names(child_mask));
        GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, LaneMask::aic(), true}, {2, child_mask, true}}, false, 2);
        fixture.graph.executable(2, child_mask.has_aic() ? 0 : 1, {0});
        fixture.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
        const std::vector<SchedulerTaskMetadata> metadata(
            fixture.storage.metadata, fixture.storage.metadata + fixture.storage.layout.task_count
        );
        ASSERT_TRUE(scheduler_validate_dependencies(fixture.graph.graph(), metadata, {}));
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        std::array<GmExecution, 2> producers{};
        std::array<bool, 2> entered{};
        for (uint32_t step = 0; step < 16 && (!entered[0] || !entered[1]); ++step) {
            for (uint32_t cluster = 0; cluster < 2; ++cluster) {
                if (entered[cluster]) continue;
                const uint32_t worker = cluster * 3;
                bool failed = false;
                scheduler_fill_ordered_slots<false>(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &failed
                );
                ASSERT_FALSE(failed);
                entered[cluster] = fixture.begin(worker, &producers[cluster]);
            }
        }
        ASSERT_TRUE(entered[0] && entered[1]);
        ASSERT_EQ(producers[0].task, 0);
        ASSERT_EQ(producers[1].task, 0);
        ASSERT_NE(producers[0].block, producers[1].block);
        ASSERT_TRUE(fixture.claim_pending(1));
        ASSERT_TRUE(fixture.complete(producers[0]));
        ASSERT_TRUE(fixture.complete(producers[1]));
        EXPECT_TRUE(fixture.task_done(0));
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 1);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        if (child_mask.is_mix()) {
            EXPECT_FALSE(fixture.claim_pending(2));
            EXPECT_EQ(fixture.admissions[0].pending_task_ref, 1);
            EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        }
        // A real producer return retires its physical slots before this owner
        // closes the ordered obligation and publishes its already-held candidate.
        std::array<bool, 6> closed{};
        for (uint32_t step = 0; step < 16 && std::find(closed.begin(), closed.end(), false) != closed.end(); ++step)
            for (uint32_t worker = 0; worker < 6; ++worker) {
                if (closed[worker]) continue;
                if (scheduler_task_ref_is_null(fixture.admissions[worker].current_task_ref)) {
                    ASSERT_GE(fixture.admissions[worker].scan_cursor, 1u);
                    closed[worker] = true;
                    continue;
                }
                ASSERT_TRUE(
                    scheduler_close_ordered_obligation<false>(
                        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &closed[worker]
                    )
                );
            }
        ASSERT_EQ(std::find(closed.begin(), closed.end(), false), closed.end());
        ASSERT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        ASSERT_TRUE(fixture.refresh_all());
        // Publication alone does not close supply. Give both blocks of task 1
        // physical owners before admitting the later same-engine successor.
        entered.fill(false);
        std::array<GmExecution, 2> independent{};
        for (uint32_t step = 0; step < 16 && (!entered[0] || !entered[1]); ++step) {
            for (uint32_t cluster = 0; cluster < 2; ++cluster) {
                if (entered[cluster]) continue;
                const uint32_t worker = cluster * 3;
                bool failed = false;
                scheduler_fill_ordered_slots<false>(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &failed
                );
                ASSERT_FALSE(failed);
                entered[cluster] = fixture.begin(worker, &independent[cluster]);
            }
        }
        ASSERT_TRUE(entered[0] && entered[1]);
        ASSERT_EQ(independent[0].task, 1);
        ASSERT_EQ(independent[1].task, 1);
        ASSERT_NE(independent[0].block, independent[1].block);
        const uint32_t discovery_owner = child_mask.is_mix() ? 0 : 1;
        ASSERT_TRUE(fixture.admit(2, discovery_owner));
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        ASSERT_EQ(shared->commit_tail, 3u);
        auto *log = scheduler_state_at<int64_t>(fixture.base(), fixture.storage.layout.admission_log_offset);
        EXPECT_EQ(log[0], 0);
        EXPECT_EQ(log[1], 1);
        EXPECT_EQ(log[2], 2);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

}  // namespace
