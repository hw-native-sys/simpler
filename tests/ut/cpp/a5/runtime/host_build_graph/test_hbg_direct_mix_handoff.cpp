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
#include "scheduler/scheduler_mix.h"

namespace {

enum class HandoffScenario { EMPTY_QUEUE, QUEUED_MIX, TWO_SUCCESSORS, BUSY_MIX_LANES };

class HbgDirectMixHandoff : public ::testing::TestWithParam<std::tuple<uint32_t, uint8_t>> {
protected:
    void expect_successor_dispatched(
        bool provide_direct_candidate, HandoffScenario scenario = HandoffScenario::EMPTY_QUEUE
    ) {
        const auto [producer_lane, mix_mask] = GetParam();
        SCOPED_TRACE(producer_lane);
        SCOPED_TRACE(static_cast<uint32_t>(mix_mask));
        const bool has_second_mix =
            scenario == HandoffScenario::QUEUED_MIX || scenario == HandoffScenario::TWO_SUCCESSORS;
        const uint32_t task_count = has_second_mix ? 3 : 2;
        scheduler_test::FixtureStorage storage{task_count, 3, 3};
        scheduler_test::GraphBuffer graph{task_count};
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            auto &context = storage.contexts[lane];
            context.is_scheduler = 1;
            context.scheduler_count = PLATFORM_CORES_PER_BLOCKDIM;
            for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
                context.cluster_worker_ids[peer] = peer;
            auto *local = storage.local_context(&context);
            local->has_mix = true;
            for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
                for (auto &slot : local->mix_slots[peer])
                    scheduler_initialize_free_slot(&slot);
        }
        auto *producer = storage.local_context(&storage.contexts[producer_lane]);
        auto *coordinator = storage.local_context(&storage.contexts[0]);
        ASSERT_EQ(producer->config.self_lane, producer_lane);
        ASSERT_EQ(producer->is_mix_coordinator(), producer_lane == 0);

        const uint8_t producer_subtask = producer_lane == 0 ? 0 : 1;
        graph.executable(0, producer_subtask);
        graph.mixed(1, mix_mask);
        graph.executable(1, static_cast<uint8_t>(__builtin_ctz(mix_mask)), {0});
        if (has_second_mix) {
            graph.mixed(2, mix_mask);
            if (scenario == HandoffScenario::TWO_SUCCESSORS)
                graph.executable(2, static_cast<uint8_t>(__builtin_ctz(mix_mask)), {0});
        }
        for (uint32_t task = 0; task < task_count; ++task) {
            auto &metadata = storage.metadata[task];
            metadata.active_mask = task == 0 ? static_cast<uint8_t>(1U << producer_subtask) : mix_mask;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (task != 0) metadata.flags |= SCHEDULER_TASK_MIX;
            if (task == 1 || (task == 2 && scenario == HandoffScenario::TWO_SUCCESSORS))
                metadata.flags |= SCHEDULER_TASK_HAS_FANIN;
            metadata.trace_index_base = task * 3;
            metadata.timing_slot = -1;
            for (uint8_t subtask = 0; subtask < 3; ++subtask)
                metadata.kernel_ids[subtask] = (metadata.active_mask & (1U << subtask)) != 0 ? 1 : UINT16_MAX;
        }
        void *base = storage.scheduler_state->base();
        ASSERT_EQ(
            scheduler_route_task(graph.graph(), base, producer, storage.run_control, 1, nullptr),
            SchedulerRouteResult::WAITING
        );
        auto *producer_control = scheduler_task_control_at(base, producer, 0);
        ASSERT_EQ(producer_control->wake_list_head, 1);
        auto *mix_inbox = scheduler_ready_inbox_at(base, coordinator, SCHEDULER_MIX_QUEUE, 0);
        ASSERT_TRUE(scheduler_ready_endpoints_empty(mix_inbox->endpoints));
        if (scenario == HandoffScenario::TWO_SUCCESSORS) {
            ASSERT_EQ(
                scheduler_route_task(graph.graph(), base, producer, storage.run_control, 2, nullptr),
                SchedulerRouteResult::WAITING
            );
            ASSERT_EQ(producer_control->wake_list_head, 2);
        } else if (scenario == HandoffScenario::QUEUED_MIX) {
            ASSERT_EQ(
                scheduler_route_task(graph.graph(), base, coordinator, storage.run_control, 2, nullptr),
                SchedulerRouteResult::READY_TO_ENQUEUE
            );
            SchedulerReadyBatch batch{};
            ASSERT_TRUE(scheduler_ready_batch_append(base, coordinator, 2, &batch, nullptr));
            ASSERT_TRUE(scheduler_ready_batch_push(base, coordinator, SCHEDULER_MIX_QUEUE, &batch, nullptr));
            ASSERT_EQ(scheduler_ready_endpoints_head(mix_inbox->endpoints), 2);
        }

        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        SchedulerFreeSlotClaim slot{producer_lane, 0, producer->slots[0].generation, producer_lane};
        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), base, producer, storage.run_control, slot, ready, 0, storage.ssbuf_region
        ));
        uint32_t executed_slot = UINT32_MAX;
        uint64_t publication = 0;
        ASSERT_TRUE(scheduler_local_ready_pop(producer, &executed_slot, &publication));
        ASSERT_EQ(executed_slot, 0u);
        ASSERT_EQ(scheduler_dispatch_state(publication), SchedulerDispatchSlotState::READY);
        producer->local_completed_generations[executed_slot] = scheduler_dispatch_generation(publication);

        int64_t unconsumed_candidate = SCHEDULER_TASK_REF_INVALID;
        {
            // The executor owns this hand-off only for one scheduling iteration.
            SchedulerRefillCandidates candidates{};
            uint64_t refilled_slots = 0;
            ASSERT_TRUE(scheduler_service_cluster_completions(
                graph.graph(), base, producer, storage.run_control, nullptr, nullptr, nullptr, 0, &refilled_slots,
                storage.ssbuf_region, provide_direct_candidate ? &candidates : nullptr
            ));
            ASSERT_EQ(storage.run_control->scheduler_error, 0u);
            ASSERT_EQ(producer_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
            ASSERT_EQ(producer_control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
            if (has_second_mix) {
                EXPECT_EQ(candidates.mix_ready.task_ref, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(scheduler_ready_endpoints_head(mix_inbox->endpoints), 2);
                EXPECT_EQ(scheduler_ready_endpoints_tail(mix_inbox->endpoints), 1);
            } else if (provide_direct_candidate && producer_lane == 0) {
                EXPECT_EQ(candidates.mix_ready.task_ref, 1);
                EXPECT_TRUE(scheduler_ready_endpoints_empty(mix_inbox->endpoints));
            } else {
                EXPECT_EQ(candidates.mix_ready.task_ref, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(scheduler_ready_endpoints_head(mix_inbox->endpoints), 1);
            }
            if (scenario == HandoffScenario::BUSY_MIX_LANES) {
                for (uint32_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
                    for (auto &mix_slot : coordinator->mix_slots[lane])
                        mix_slot.state = SchedulerDispatchSlotState::FILLING;
            }
            uint64_t cursor = 0;
            bool progress = false;
            ASSERT_TRUE(scheduler_fill_mix_after_completions(
                graph.graph(), base, producer, storage.run_control, &candidates.mix_ready, &cursor, nullptr, 0, true,
                storage.ssbuf_region, &progress
            ));
            unconsumed_candidate = candidates.mix_ready.task_ref;
        }

        // AIV can publish to the Mix inbox, but only AIC can place its successor.
        SchedulerRefillCandidates coordinator_candidates{};
        uint64_t coordinator_cursor = 0;
        bool coordinator_progress = false;
        ASSERT_TRUE(scheduler_fill_mix_after_completions(
            graph.graph(), base, coordinator, storage.run_control, &coordinator_candidates.mix_ready,
            &coordinator_cursor, nullptr, 0, true, storage.ssbuf_region, &coordinator_progress
        ));
        ASSERT_EQ(storage.run_control->scheduler_error, 0u);
        if (scenario == HandoffScenario::BUSY_MIX_LANES) {
            EXPECT_FALSE(coordinator_progress);
            EXPECT_EQ(scheduler_ready_endpoints_head(mix_inbox->endpoints), 1);
            for (const auto &tracker : coordinator->mix_trackers)
                EXPECT_EQ(tracker.task_ref, SCHEDULER_TASK_REF_INVALID);
            for (uint32_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
                for (auto &mix_slot : coordinator->mix_slots[lane])
                    scheduler_initialize_free_slot(&mix_slot);
            ASSERT_TRUE(scheduler_fill_mix_after_completions(
                graph.graph(), base, coordinator, storage.run_control, &coordinator_candidates.mix_ready,
                &coordinator_cursor, nullptr, 0, true, storage.ssbuf_region, &coordinator_progress
            ));
            EXPECT_TRUE(coordinator_progress);
            ASSERT_EQ(storage.run_control->scheduler_error, 0u);
        }
        uint32_t dispatched_successors = 0;
        for (const auto &tracker : coordinator->mix_trackers)
            if (tracker.task_ref == 1) ++dispatched_successors;
        EXPECT_EQ(dispatched_successors, 1u)
            << "producer_lane=" << producer_lane << ", mix_mask=" << static_cast<uint32_t>(mix_mask)
            << ", candidate left at iteration end=" << unconsumed_candidate
            << ", Mix inbox head=" << scheduler_ready_endpoints_head(mix_inbox->endpoints);
        EXPECT_EQ(unconsumed_candidate, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(coordinator_candidates.mix_ready.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_TRUE(scheduler_ready_endpoints_empty(mix_inbox->endpoints));
        if (has_second_mix) {
            uint32_t dispatched_second = 0;
            for (const auto &tracker : coordinator->mix_trackers)
                if (tracker.task_ref == 2) ++dispatched_second;
            EXPECT_EQ(dispatched_second, 1u);
            EXPECT_EQ(coordinator->mix_trackers[0].task_ref, 2);
            EXPECT_EQ(coordinator->mix_trackers[1].task_ref, 1);
        }
    }
};

TEST_P(HbgDirectMixHandoff, CompletionIterationKeepsUnlockedMixReachable) { expect_successor_dispatched(true); }

TEST_P(HbgDirectMixHandoff, QueueHandoffWithoutDirectSinkKeepsUnlockedMixReachable) {
    expect_successor_dispatched(false);
}

TEST_P(HbgDirectMixHandoff, UnlockedMixDoesNotDisplaceAnExistingQueueHead) {
    expect_successor_dispatched(true, HandoffScenario::QUEUED_MIX);
}

TEST_P(HbgDirectMixHandoff, TwoMixSuccessorsUnlockedTogetherAreDispatchedExactlyOnce) {
    expect_successor_dispatched(true, HandoffScenario::TWO_SUCCESSORS);
}

TEST_P(HbgDirectMixHandoff, BusyPlacementQueuesCandidateUntilMixLanesBecomeAvailable) {
    expect_successor_dispatched(true, HandoffScenario::BUSY_MIX_LANES);
}

INSTANTIATE_TEST_SUITE_P(
    ProducerLanesAndMixMasks, HbgDirectMixHandoff,
    ::testing::Combine(::testing::Values(0u, 1u, 2u), ::testing::Values<uint8_t>(3, 5, 6, 7))
);

}  // namespace
