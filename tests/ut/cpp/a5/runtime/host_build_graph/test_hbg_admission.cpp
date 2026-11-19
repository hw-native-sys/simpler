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
#include "scheduler/scheduler_admission.h"

namespace {
struct OrderTopology {
    explicit OrderTopology(uint32_t clusters, bool compact = false) :
        store(1, 3 * clusters, 1, {3 * clusters, clusters, 0, 0, 0}),
        graph(1) {
        auto &task = store.metadata[0];
        task.flags = SCHEDULER_TASK_EXECUTABLE;
        task.logical_block_num = 1;
        task.active_mask = LaneMask::aic();
        task.kernel_ids[0] = 0;
        graph.executable(0, 0);
        store.prepare_catalog();
        for (uint32_t cluster = 0; cluster < clusters; ++cluster) {
            const uint32_t ids[] = {
                compact ? cluster : cluster * 3, compact ? clusters + 2 * cluster : cluster * 3 + 2,
                compact ? clusters + 2 * cluster + 1 : cluster * 3 + 1
            };
            for (uint32_t lane = 0; lane < 3; ++lane) {
                auto &context = store.contexts[ids[lane]];
                context.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
                context.physical_core_id = 17 + 3 * ids[lane];
                context.cluster_index = cluster;
                context.type_rank = lane == 0 ? cluster : 2 * cluster + lane - 1;
                for (uint32_t peer = 0; peer < 3; ++peer)
                    context.cluster_worker_ids[peer] = ids[peer];
                store.test_graphs[ids[lane]] = graph.graph();
            }
        }
        summary()->topology_layout =
            compact ? SchedulerTopologyLayout::COMPACT_LAUNCH : SchedulerTopologyLayout::GENERAL;
    }

    void *base() { return store.scheduler_state->base(); }
    SchedulerGraphSummary *summary() {
        return scheduler_state_at<SchedulerGraphSummary>(base(), store.layout.graph_summary_offset);
    }
    SchedulerLocalState *worker(uint32_t id) { return store.local_context(&store.contexts[id]); }

    scheduler_test::FixtureStorage store;
    scheduler_test::BasicGraphBuffer<4> graph;
};

TEST(HbgOrderTopology, GeneralAndCompactLayoutsRetainPhysicalLaneAndWorkerAbove63) {
    for (const bool compact : {false, true}) {
        OrderTopology f(32, compact);
        constexpr struct {
            uint32_t worker, general_lane, compact_lane;
        } cases[] = {{0, 0, 0}, {32, 1, 1}, {64, 2, 1}, {95, 1, 2}};
        for (const auto &[worker, general_lane, compact_lane] : cases) {
            const auto expected_lane = compact ? compact_lane : general_lane;
            auto *local = f.worker(worker);
            ASSERT_TRUE(scheduler_initialize_order(f.base(), local));
            EXPECT_EQ(local->worker_id(), worker);
            EXPECT_EQ(local->config.self_lane, expected_lane);
            EXPECT_EQ(local->is_mix_coordinator(), expected_lane == 1);
            EXPECT_EQ(local->next_mix_ref, UINT32_MAX);
        }
    }
}

TEST(HbgOrderTopology, InvalidPeerAndUnprovedCompactTopologyAreRejected) {
    for (const uint32_t peer : {2u, 4u}) {
        OrderTopology general(2);
        auto *local = general.worker(0);
        general.store.contexts[peer].cluster_worker_ids[0] = peer == 2 ? 3 : 0;
        EXPECT_FALSE(scheduler_initialize_order(general.base(), local));
    }
    OrderTopology compact(2, true);
    auto *local = compact.worker(3);
    local->config.worker_ids[0] = 1;
    EXPECT_FALSE(scheduler_initialize_order(compact.base(), local));
}

TEST(HbgOrderTopology, SuccessfulBindingRetainsZeroAndMaximumEpochForPureWork) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_MAX}) {
        OrderTopology f(1);
        f.summary()->run_epoch = epoch;
        auto *local = f.worker(0);
        EXPECT_EQ(scheduler_error_identity(local, 0).valid_fields & SCHEDULER_ERROR_HAS_EPOCH, 0u);
        ASSERT_TRUE(scheduler_initialize_order(f.base(), local));
        const auto identity = scheduler_error_identity(local, 0);
        EXPECT_NE(identity.valid_fields & SCHEDULER_ERROR_HAS_EPOCH, 0u);
        EXPECT_EQ(identity.run_epoch, epoch);
    }
}
}  // namespace
