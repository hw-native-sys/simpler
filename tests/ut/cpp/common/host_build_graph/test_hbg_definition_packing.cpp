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

#include <limits>

#include "host_build_graph/host/definition_packing.h"

namespace {
using hbg::DefinitionPackingPlan;
using hbg::measure_definition_packing;
constexpr size_t align = GRAPH_DEFINITION_OBJECT_ALIGN;
constexpr size_t image_bytes = sizeof(GraphDefinition);
constexpr size_t object_bytes = (sizeof(GraphDefinitionHeader) + image_bytes + align - 1) & ~(align - 1);
const std::byte spill{};

TEST(HbgDefinitionPacking, EmptyGraphHasNoSection) {
    DefinitionPackingPlan plan;
    ASSERT_TRUE(measure_definition_packing({}, 0, plan));
    EXPECT_EQ(plan.bytes, 0U);
    EXPECT_TRUE(plan.objects.empty());
}

TEST(HbgDefinitionPacking, RetainedAndSpilledObjectsShareOneMeasuredSection) {
    GraphHostDefinitionList definitions{{
        {11, 0, nullptr, image_bytes},
        {12, GRAPH_NO_OBJECT_OFFSET, &spill, image_bytes},
    }};
    DefinitionPackingPlan plan;
    ASSERT_TRUE(measure_definition_packing(definitions, object_bytes, plan));
    EXPECT_EQ(plan.bytes, 2 * object_bytes);
    EXPECT_EQ(plan.spilled, 1U);
    ASSERT_EQ(plan.objects.size(), 2U);
    EXPECT_EQ(plan.objects[0].object_offset, 0U);
    EXPECT_EQ(plan.objects[1].object_offset, object_bytes);
    EXPECT_EQ(plan.objects[1].copy, &spill);
}

TEST(HbgDefinitionPacking, RejectsDuplicateDefinitionKeysWithoutPublishingPartialPlan) {
    GraphHostDefinitionList definitions{{
        {11, GRAPH_NO_OBJECT_OFFSET, &spill, image_bytes},
        {11, GRAPH_NO_OBJECT_OFFSET, &spill, image_bytes},
    }};
    DefinitionPackingPlan plan;
    plan.bytes = 77;
    EXPECT_FALSE(measure_definition_packing(definitions, 0, plan));
    EXPECT_EQ(plan.bytes, 77U);
    EXPECT_TRUE(plan.objects.empty());
}

TEST(HbgDefinitionPacking, RejectsOverlappingRetainedObjects) {
    GraphHostDefinitionList definitions{{{11, 0, nullptr, image_bytes}, {12, 0, nullptr, image_bytes}}};
    DefinitionPackingPlan plan;
    EXPECT_FALSE(measure_definition_packing(definitions, object_bytes, plan));
}

TEST(HbgDefinitionPacking, RejectsMisalignedOrOutOfRangeRetainedObjects) {
    DefinitionPackingPlan plan;
    EXPECT_FALSE(measure_definition_packing({{{11, 1, nullptr, image_bytes}}}, object_bytes, plan));
    EXPECT_FALSE(measure_definition_packing({{{11, 0, nullptr, image_bytes}}}, object_bytes - align, plan));
    EXPECT_FALSE(measure_definition_packing({}, 1, plan));
}

TEST(HbgDefinitionPacking, RejectsInvalidSpillAndTruncatedDefinition) {
    DefinitionPackingPlan plan;
    EXPECT_FALSE(measure_definition_packing({{{11, 0, &spill, image_bytes}}}, 0, plan));
    EXPECT_FALSE(measure_definition_packing({{{11, GRAPH_NO_OBJECT_OFFSET, nullptr, image_bytes}}}, 0, plan));
    EXPECT_FALSE(measure_definition_packing({{{11, GRAPH_NO_OBJECT_OFFSET, &spill, image_bytes - 1}}}, 0, plan));
}

TEST(HbgDefinitionPacking, RejectsOverflowAndUnrepresentableLaunchSectionBeforeAllocation) {
    DefinitionPackingPlan plan;
    EXPECT_FALSE(measure_definition_packing({{{11, GRAPH_NO_OBJECT_OFFSET, &spill, SIZE_MAX}}}, 0, plan));
    EXPECT_FALSE(measure_definition_packing({{{11, GRAPH_NO_OBJECT_OFFSET, &spill, UINT32_MAX}}}, 0, plan));
    const size_t maximum_aligned = size_t{UINT32_MAX} & ~(align - 1);
    EXPECT_FALSE(
        measure_definition_packing({{{11, GRAPH_NO_OBJECT_OFFSET, &spill, image_bytes}}}, maximum_aligned, plan)
    );
}

}  // namespace
