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

    explicit CatalogImage(const std::vector<SchedulerTaskMetadata> &metadata) {
        if (!scheduler_plan_task_catalog(metadata, 2, 64, &plan)) throw std::logic_error("cannot plan test catalog");
        storage.resize(plan.total_size + SCHEDULER_STATE_BASE_ALIGNMENT);
        base = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_BASE_ALIGNMENT - 1) &
            ~(uintptr_t{SCHEDULER_STATE_BASE_ALIGNMENT} - 1)
        );
        scheduler_init_task_catalog(base, plan);
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
