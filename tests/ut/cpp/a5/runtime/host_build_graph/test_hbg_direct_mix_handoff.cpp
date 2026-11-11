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

#include <tuple>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_dispatch.h"

namespace {

enum class HandoffScenario { EMPTY_QUEUE, QUEUED_MIX, TWO_SUCCESSORS, BUSY_MIX_LANES };

uint8_t first_subtask(LaneMask lanes) { return lanes.has_aic() ? 0 : (lanes.has_aiv0() ? 1 : 2); }

class HbgMixQueueHandoff : public ::testing::TestWithParam<std::tuple<uint32_t, LaneMask>> {
protected:
    void expect_successor_reachable(HandoffScenario scenario) {
        const auto [producer_lane, mix_mask] = GetParam();
        SCOPED_TRACE(producer_lane);
        SCOPED_TRACE(mix_mask.count());
        const bool second = scenario == HandoffScenario::QUEUED_MIX || scenario == HandoffScenario::TWO_SUCCESSORS;
        const uint32_t count = second ? 3 : 2;
        scheduler_test::FixtureStorage storage(count, 3, 3);
        scheduler_test::GraphBuffer graph(count);
        const uint8_t producer_subtask = producer_lane == 0 ? 0 : 1;
        graph.executable(0, producer_subtask);
        for (uint32_t task = 1; task < count; ++task) {
            graph.mixed(task, mix_mask);
            if (task == 1 || scenario == HandoffScenario::TWO_SUCCESSORS)
                graph.executable(task, first_subtask(mix_mask), {0});
        }
        for (uint32_t task = 0; task < count; ++task) {
            auto &metadata = storage.metadata[task];
            metadata.active_mask = task == 0 ? LaneMask::of(static_cast<SubtaskSlot>(producer_subtask)) : mix_mask;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (task != 0) metadata.flags |= SCHEDULER_TASK_MIX;
            if (task == 1 || (task == 2 && scenario == HandoffScenario::TWO_SUCCESSORS))
                metadata.flags |= SCHEDULER_TASK_HAS_FANIN;
            metadata.trace_index_base = task * 3;
            metadata.timing_slot = -1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = metadata.active_mask.has(static_cast<SubtaskSlot>(lane)) ? 1 : UINT16_MAX;
        }
        storage.prepare_ready_pool();
        for (uint32_t lane = 0; lane < 3; ++lane) {
            auto &context = storage.contexts[lane];
            context.is_scheduler = 1;
            context.scheduler_count = 3;
            for (uint32_t peer = 0; peer < 3; ++peer)
                context.cluster_worker_ids[peer] = peer;
        }
        auto *producer = storage.local_context(&storage.contexts[producer_lane]);
        auto *coordinator = storage.local_context(&storage.contexts[0]);
        void *base = storage.scheduler_state->base();
        ASSERT_EQ(
            scheduler_route_task(graph.graph(), base, producer, storage.run_control, 1), SchedulerRouteResult::WAITING
        );
        if (scenario == HandoffScenario::TWO_SUCCESSORS) {
            ASSERT_EQ(
                scheduler_route_task(graph.graph(), base, producer, storage.run_control, 2),
                SchedulerRouteResult::WAITING
            );
        } else if (scenario == HandoffScenario::QUEUED_MIX) {
            ASSERT_EQ(
                scheduler_route_task(graph.graph(), base, coordinator, storage.run_control, 2),
                SchedulerRouteResult::READY_TO_ENQUEUE
            );
            SchedulerReadyBatch batch{};
            ASSERT_TRUE(scheduler_ready_batch_append(base, coordinator, 2, &batch));
            ASSERT_TRUE(scheduler_ready_batch_push(base, coordinator, 0, &batch));
        }
        SchedulerReadyClaim initial{};
        initial.task_ref = 0;
        SchedulerFreeSlotClaim slot{producer_lane, 0, producer->slots[0].generation, producer_lane};
        ASSERT_TRUE(scheduler_fill_dispatch_slot(graph.graph(), base, producer, storage.run_control, slot, initial, 0));
        uint32_t executed_slot = UINT32_MAX;
        uint64_t publication = 0;
        ASSERT_TRUE(scheduler_local_ready_pop(producer, &executed_slot, &publication));
        producer->local_completed_generations[executed_slot] = scheduler_dispatch_generation(publication);
        uint64_t refilled = 0;
        ASSERT_TRUE(
            scheduler_service_cluster_completions(graph.graph(), base, producer, storage.run_control, 0, &refilled)
        );
        EXPECT_EQ(refilled, 0u);
        EXPECT_EQ(producer->slots[executed_slot].state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(scheduler_task_control_at(base, producer, 0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(storage.run_control->scheduler_error, 0u);

        // AIV workers do not claim MIX, including a task containing only two AIV lanes.
        for (uint32_t lane : {1u, 2u}) {
            SchedulerReadyClaim aiv{};
            ASSERT_TRUE(scheduler_claim_ready_for_slot(
                graph.graph(), base, storage.local_context(&storage.contexts[lane]), storage.run_control, 3, 1, &aiv
            ));
            EXPECT_EQ(aiv.task_ref, SCHEDULER_TASK_REF_INVALID);
        }
        if (scenario == HandoffScenario::BUSY_MIX_LANES) {
            for (uint32_t lane : {1u, 2u})
                for (auto &pending : storage.local_context(&storage.contexts[lane])->slots)
                    pending.state = SchedulerDispatchSlotState::FILLING;
        }
        // Ready ownership precedes bounded admission. Busy physical lanes must
        // not pin discovery to a shared queue head or lose an unlocked successor.
        for (int64_t expected : second ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{1}) {
            SchedulerReadyClaim ready{};
            ASSERT_TRUE(
                scheduler_claim_ready_for_slot(graph.graph(), base, coordinator, storage.run_control, 3, 0, &ready)
            );
            EXPECT_EQ(ready.task_ref, expected);
            EXPECT_EQ(ready.source, SchedulerReadySource::POOL);
        }
        SchedulerReadyClaim empty{};
        ASSERT_TRUE(
            scheduler_claim_ready_for_slot(graph.graph(), base, coordinator, storage.run_control, 3, 0, &empty)
        );
        EXPECT_EQ(empty.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(storage.run_control->scheduler_error, 0u);
    }
};

TEST_P(HbgMixQueueHandoff, CompletionQueuesMixInsteadOfUsingOrdinaryDirectRefill) {
    expect_successor_reachable(HandoffScenario::EMPTY_QUEUE);
}
TEST_P(HbgMixQueueHandoff, ExistingQueueHeadPrecedesUnlockedMix) {
    expect_successor_reachable(HandoffScenario::QUEUED_MIX);
}
TEST_P(HbgMixQueueHandoff, TwoUnlockedSuccessorsAreClaimedExactlyOnce) {
    expect_successor_reachable(HandoffScenario::TWO_SUCCESSORS);
}
TEST_P(HbgMixQueueHandoff, BusyPhysicalLanesDoNotHideReadyOwnership) {
    expect_successor_reachable(HandoffScenario::BUSY_MIX_LANES);
}

INSTANTIATE_TEST_SUITE_P(
    ProducerLanesAndMixMasks, HbgMixQueueHandoff,
    ::testing::Combine(
        ::testing::Values(0u, 1u, 2u),
        ::testing::Values(
            LaneMask(true, true, false), LaneMask(true, false, true), LaneMask::both_aiv(), LaneMask::all()
        )
    )
);

}  // namespace
