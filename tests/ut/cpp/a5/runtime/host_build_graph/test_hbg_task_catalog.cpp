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

#include <cstdint>
#include <vector>

#include "dependency_plan.h"
#include "scheduler/scheduler_issue.h"
#include "hbg_scheduler_test_support.h"
#include "task_catalog_plan.h"

namespace {

SchedulerTaskMetadata physical(LaneMask lanes, uint16_t blocks = 1) {
    SchedulerTaskMetadata result{};
    result.flags = SCHEDULER_TASK_EXECUTABLE | (lanes.is_mix() ? SCHEDULER_TASK_MIX : 0);
    result.active_mask = lanes;
    result.logical_block_num = blocks;
    return result;
}

struct CatalogImage {
    SchedulerTaskCatalogPlan plan;
    std::vector<uint8_t> storage;
    void *base{nullptr};

    explicit CatalogImage(
        const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerTaskOrderEdge> &edges = {}
    ) {
        if (!scheduler_plan_task_catalog(metadata, 2, 64, &plan, edges))
            throw std::logic_error("cannot plan test catalog");
        storage.resize(plan.total_size + SCHEDULER_STATE_BASE_ALIGNMENT);
        base = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_BASE_ALIGNMENT - 1) &
            ~(uintptr_t{SCHEDULER_STATE_BASE_ALIGNMENT} - 1)
        );
        if (edges.empty()) scheduler_init_task_catalog(base, plan);
    }

    bool valid() {
        return scheduler_task_catalog_validate(base, plan.total_size, plan.header_offset, plan.header.task_count, 2);
    }
};

TEST(HbgTaskCatalog, PhysicalTasksUseTwoEngineListsAndControlsRemainSeparate) {
    std::vector<SchedulerTaskMetadata> metadata{
        physical(LaneMask::aic(), 4),
        physical(LaneMask::aiv0(), 2),
        physical(LaneMask(true, true, false)),
        physical(LaneMask::both_aiv()),
        physical(LaneMask::aiv0()),
        {},
        {}
    };
    metadata[4].flags |= SCHEDULER_TASK_INLINE;
    for (auto &kernel : metadata[4].kernel_ids)
        kernel = UINT16_MAX;
    metadata[5].logical_block_num = 1;
    CatalogImage image(metadata);
    ASSERT_TRUE(image.valid());
    ASSERT_EQ(image.plan.descriptors.size(), 2u);
    EXPECT_EQ(image.plan.header.task_count, 7u);
    EXPECT_EQ(image.plan.header.entry_count, 4u);
    EXPECT_EQ(image.plan.header.control_count, 2u);
    const auto cubes = scheduler_task_catalog(image.base, image.plan.header, 0);
    const auto vectors = scheduler_task_catalog(image.base, image.plan.header, 1);
    EXPECT_EQ(std::vector<int64_t>(cubes.refs, cubes.refs + cubes.count), (std::vector<int64_t>{0, 2, 3}));
    EXPECT_EQ(std::vector<int64_t>(vectors.refs, vectors.refs + vectors.count), (std::vector<int64_t>{1}));
    EXPECT_EQ(metadata[2].active_mask.raw(), 3u);
    EXPECT_EQ(metadata[3].active_mask.raw(), 6u);
}

TEST(HbgTaskCatalog, EmptyAndSingleEngineRunsKeepBothDirectories) {
    for (uint32_t count : {0u, 1u}) {
        CatalogImage image(std::vector<SchedulerTaskMetadata>(count, physical(LaneMask::aic())));
        ASSERT_TRUE(image.valid());
        ASSERT_EQ(image.plan.descriptors.size(), 2u);
        EXPECT_EQ(scheduler_task_catalog(image.base, image.plan.header, 0).count, count);
        EXPECT_EQ(scheduler_task_catalog(image.base, image.plan.header, 1).count, 0u);
        EXPECT_EQ(image.plan.header.control_count, 0u);
    }
}

TEST(HbgTaskCatalog, TopologicalGenerationsExposeIndependentBranchesBeforeDeeperTasks) {
    scheduler_test::BasicGraphBuffer<16> graph(8);
    graph.executable(0, 0);
    graph.executable(1, 0);
    graph.executable(2, 0, {0});
    graph.executable(3, 0, {2});
    graph.executable(4, 0, {3});
    graph.executable(5, 0, {0});
    graph.executable(6, 0, {5});
    graph.executable(7, 0);
    const std::vector<SchedulerTaskMetadata> metadata{
        physical(LaneMask::aiv0()),
        physical(LaneMask::aiv0()),
        physical(LaneMask::aic()),
        physical(LaneMask::aiv0()),
        physical(LaneMask(true, true, false)),
        physical(LaneMask::aic()),
        physical(LaneMask::aiv0()),
        physical(LaneMask::aic())
    };
    CatalogImage image(metadata);
    std::string error;
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error)) << error;
    scheduler_init_task_catalog(image.base, image.plan);
    ASSERT_TRUE(image.valid());
    const auto cubes = scheduler_task_catalog(image.base, image.plan.header, 0);
    const auto vectors = scheduler_task_catalog(image.base, image.plan.header, 1);
    EXPECT_EQ(std::vector<int64_t>(cubes.refs, cubes.refs + cubes.count), (std::vector<int64_t>{7, 2, 5, 4}));
    EXPECT_EQ(std::vector<int64_t>(vectors.refs, vectors.refs + vectors.count), (std::vector<int64_t>{0, 1, 3, 6}));
    EXPECT_EQ(image.plan.task_order, (std::vector<int64_t>{0, 1, 7, 2, 5, 3, 6, 4}));
    EXPECT_EQ(image.plan.task_levels, (std::vector<uint32_t>{0, 0, 1, 2, 3, 1, 2, 0}));
}

TEST(HbgTaskCatalog, NotificationPrecedesEarlierWaiterWithoutChangingDataFanin) {
    scheduler_test::BasicGraphBuffer<8> graph(4);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0, {0});
    graph.executable(3, 0);
    const std::vector<SchedulerTaskMetadata> metadata(4, physical(LaneMask::aiv0()));
    CatalogImage image(metadata, {{3, 0}, {3, 0}});
    std::string error;
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error)) << error;
    scheduler_init_task_catalog(image.base, image.plan);
    EXPECT_EQ(image.plan.header.ordering_edge_count, 0u);
    EXPECT_EQ(image.plan.header.ordering_rows_offset, 0u);
    EXPECT_TRUE(image.valid());
    EXPECT_EQ(image.plan.catalogs[1], (std::vector<int64_t>{3, 0, 1, 2}));
    SchedulerFaninView waiter{};
    ASSERT_EQ(scheduler_node_fanin_view(graph.graph(), 0, &waiter), SchedulerGraphResult::OK);
    EXPECT_EQ(waiter.count, 0u);
    SchedulerFaninView consumer{};
    ASSERT_EQ(scheduler_node_fanin_view(graph.graph(), 1, &consumer), SchedulerGraphResult::OK);
    int64_t producer = -1;
    ASSERT_EQ(scheduler_fanin_ref(consumer, 1, 0, &producer), SchedulerGraphResult::OK);
    EXPECT_EQ(producer, 0);

    SchedulerTaskCatalogPlan cyclic;
    ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 2, 64, &cyclic, {{1, 0}}));
    const auto catalogs = cyclic.catalogs;
    EXPECT_FALSE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &cyclic, &error));
    EXPECT_NE(error.find("cycle"), std::string::npos);
    EXPECT_EQ(cyclic.catalogs, catalogs);
    EXPECT_THROW(scheduler_init_task_catalog(image.base, cyclic), std::logic_error);
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, 2, 64, &cyclic, {{4, 0}}));
    EXPECT_EQ(cyclic.catalogs, catalogs);
}

TEST(HbgTaskCatalog, CompletedNodesDoNotReadStaleIncomingDependenciesDuringOrdering) {
    scheduler_test::BasicGraphBuffer<8> graph(3);
    graph.executable(0, 0, {INT32_MAX});
    graph.executable(1, 0, {0, 0});
    graph.executable(2, 0);
    const std::vector<SchedulerTaskMetadata> metadata(3, physical(LaneMask::aic()));
    CatalogImage image(metadata);
    std::string error;
    EXPECT_FALSE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error));
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {0}, &image.plan, &error)) << error;
    EXPECT_EQ(image.plan.catalogs[0], (std::vector<int64_t>{0, 2, 1}));
}

TEST(HbgTaskCatalog, CrossEngineNotificationRequiresIssuedSupplyRatherThanLogicalCompletion) {
    for (const bool sync_start : {false, true}) {
        SCOPED_TRACE(sync_start);
        scheduler_test::BasicGraphBuffer<4> graph(2);
        graph.executable(0, 0);
        graph.executable(1, 0);
        std::vector<SchedulerTaskMetadata> metadata{physical(LaneMask::aiv0()), physical(LaneMask::aic(), 2)};
        if (sync_start) metadata[1].flags |= SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_SPMD;
        SchedulerTaskCatalogPlan plan;
        scheduler_test::FixtureStorage storage(2, 3, 1, {}, false, [&](AicoreSchedulerLayout &layout) {
            ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 1, layout.total_size, &plan, {{1, 0}}));
            layout.total_size = plan.total_size;
        });
        std::string error;
        ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &plan, &error)) << error;
        void *base = storage.scheduler_state->base();
        std::copy(metadata.begin(), metadata.end(), storage.metadata);
        scheduler_init_task_catalog(base, plan);
        storage.task_catalog_offset = plan.header_offset;
        auto *worker = storage.local_context(&storage.contexts[1]);
        auto *producer = scheduler_task_control_at(base, worker, 1);
        producer->completion_state = scheduler_completion_initial_state(2);
        const auto query = [&]() {
            int64_t ref = -1;
            bool progress = false;
            EXPECT_TRUE(scheduler_query_issue_head(graph.graph(), base, worker, storage.run_control, &ref, &progress));
            return ref;
        };
        EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
        producer->next_block = 2;
        if (sync_start) {
            EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
            producer->claim_seal = 2;
        }
        EXPECT_EQ(query(), 0);
        EXPECT_FALSE(scheduler_completion_done(producer->completion_state));
        EXPECT_EQ(scheduler_read_error(storage.run_control), 0u);
    }
}

TEST(HbgTaskCatalog, OrderingCsrRejectsMalformedExtentsAndRows) {
    const std::vector<SchedulerTaskMetadata> metadata{physical(LaneMask::aiv0()), physical(LaneMask::aic())};
    scheduler_test::BasicGraphBuffer<4> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    CatalogImage image(metadata, {{1, 0}});
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan));
    scheduler_init_task_catalog(image.base, image.plan);
    ASSERT_TRUE(image.valid());
    auto *header = scheduler_state_at<SchedulerTaskCatalogHeader>(image.base, image.plan.header_offset);
    SchedulerOrderingFanin view;
    ASSERT_TRUE(scheduler_task_ordering_fanin(image.base, *header, 0, &view));
    ASSERT_EQ(view.count, 1u);
    EXPECT_EQ(view.refs[0], 1);
    auto *rows = scheduler_state_at<uint32_t>(image.base, header->ordering_rows_offset);
    rows[1] = 2;
    EXPECT_FALSE(scheduler_task_ordering_fanin(image.base, *header, 0, &view));
    rows[1] = 1;
    rows[2] = 0;
    EXPECT_FALSE(image.valid());
    rows[2] = 1;
    header->ordering_rows_offset = header->storage_end_offset;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->ordering_edge_count = UINT32_MAX;
    EXPECT_FALSE(image.valid());
}

TEST(HbgTaskCatalog, MalformedImageCannotAuthorizeDescriptorAccess) {
    CatalogImage image({physical(LaneMask::aic()), physical(LaneMask::aiv0())});
    auto *header = scheduler_state_at<SchedulerTaskCatalogHeader>(image.base, image.plan.header_offset);
    auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(image.base, header->descriptors_offset);
    ASSERT_TRUE(image.valid());
    header->version = 0;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->control_count = header->task_count;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->entry_count = header->task_count + 1;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    descriptors[0].entries_offset = image.plan.header_offset;
    EXPECT_FALSE(image.valid());
    descriptors[0] = image.plan.descriptors[0];
    descriptors[0].capacity = UINT64_MAX;
    EXPECT_FALSE(image.valid());
    descriptors[0] = image.plan.descriptors[0];
    header->storage_end_offset = image.plan.total_size + 64;
    EXPECT_FALSE(image.valid());
}

TEST(HbgTaskCatalog, InvalidTopologyOrOverflowLeavesThePreviousPlanIntact) {
    const std::vector<SchedulerTaskMetadata> metadata{physical(LaneMask::aic())};
    SchedulerTaskCatalogPlan plan;
    ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 1, 128, &plan));
    const auto size = plan.total_size;
    const auto catalogs = plan.catalogs;
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, 0, 128, &plan));
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, SCHEDULER_CLUSTER_CAPACITY + 1, 128, &plan));
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, 1, UINT64_MAX - 64, &plan));
    EXPECT_EQ(plan.total_size, size);
    EXPECT_EQ(plan.catalogs, catalogs);
}

TEST(HbgHostDependencies, ValidatesEveryLiveFaninWithoutAllocatingProducerCredits) {
    scheduler_test::BasicGraphBuffer<4> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0, 0});
    const std::vector<SchedulerTaskMetadata> metadata(2, physical(LaneMask::aic()));
    EXPECT_TRUE(scheduler_validate_dependencies(graph.graph(), metadata, {}));
    EXPECT_TRUE(scheduler_validate_dependencies(graph.graph(), metadata, {0}));
    EXPECT_FALSE(scheduler_validate_dependencies(graph.graph(), metadata, {2}));
    EXPECT_FALSE(scheduler_validate_dependencies(graph.graph(), {metadata[0]}, {}));
    for (int32_t producer : {-1, 1, 2, INT32_MAX}) {
        SCOPED_TRACE(producer);
        graph.executable(1, 0, {producer});
        EXPECT_FALSE(scheduler_validate_dependencies(graph.graph(), metadata, {}));
    }
}

}  // namespace
