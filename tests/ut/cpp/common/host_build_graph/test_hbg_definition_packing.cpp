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
#include "host_build_graph/kernel_resource_requirements.h"
#include "host_build_graph/kernel_argument_snapshot.h"

namespace {
using hbg::DefinitionPackingPlan;
using hbg::measure_definition_packing;
constexpr size_t align = GRAPH_DEFINITION_OBJECT_ALIGN;
constexpr size_t image_bytes = sizeof(GraphDefinition);
constexpr size_t object_bytes = (sizeof(GraphDefinitionHeader) + image_bytes + align - 1) & ~(align - 1);
const std::byte spill{};

TEST(HbgArgumentSnapshot, TracksAddressesScalarsAndGeometryWithoutReadingTensorData) {
    ChipStorageTaskArgs args;
    ChipTensor tensor{};
    tensor.buffer = {0x12340000, 16};
    tensor.ndims = 1;
    tensor.shapes[0] = 4;
    tensor.strides[0] = 1;
    tensor.dtype = DataType::FLOAT32;
    tensor.address_space = AddressSpace::DEVICE;
    args.add_tensor(tensor);
    args.add_scalar(uint64_t{0x3f800000});
    const auto initial = hbg::kernel_argument_snapshot_hash(args);
    ASSERT_NE(initial, 0U);
    auto changed = args;
    changed.tensor(0).shapes[4] = 999;
    changed.tensor(0).strides[4] = 999;
    EXPECT_EQ(hbg::kernel_argument_snapshot_hash(changed), initial);
    changed.tensor(0).buffer.addr += 64;
    EXPECT_NE(hbg::kernel_argument_snapshot_hash(changed), initial);
    changed = args;
    changed.scalars_[0] = 0x40000000;
    EXPECT_NE(hbg::kernel_argument_snapshot_hash(changed), initial);
    changed = args;
    changed.tensor(0).shapes[0] = 3;
    EXPECT_NE(hbg::kernel_argument_snapshot_hash(changed), initial);
}

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

hbg::GraphResourceRequirements
resource_graph(uint64_t heap, uint64_t runtime, uint64_t definitions, uint64_t scheduler = 0) {
    return {{hbg::RuntimeArchitecture::A2A3, 16, 1024, 512, 1024}, heap, runtime, definitions, scheduler};
}

TEST(HbgGraphResources, LogicalBytesIncludeCompactImageOnce) {
    auto graph = resource_graph(1024, 2048, 128, 64);
    uint64_t bytes = 0;
    ASSERT_TRUE(graph.required_bytes(bytes));
    EXPECT_EQ(bytes, 3264U);
}

TEST(HbgGraphResources, IndependentRegionMaximaAdmitEveryPreparedGraph) {
    const hbg::GraphResourceRequirements graphs[] = {
        resource_graph(4096, 2048, 128), resource_graph(1024, 8192, 256, 512)
    };
    hbg::KernelResourceCapacity capacity;
    ASSERT_EQ(hbg::KernelResourceCapacity::create(graphs, 2, capacity), 0);
    EXPECT_EQ(capacity.capacity().gm_heap_bytes, 4096U);
    EXPECT_EQ(capacity.capacity().runtime_arena_bytes, 8192U);
    EXPECT_EQ(capacity.capacity().graph_definition_bytes, 256U);
    EXPECT_EQ(capacity.capacity().scheduler_state_bytes, 512U);
    EXPECT_TRUE(capacity.admits(graphs[0]));
    EXPECT_TRUE(capacity.admits(graphs[1]));
    auto oversized = graphs[1];
    ++oversized.graph_definition_bytes;
    EXPECT_FALSE(capacity.admits(oversized));
}

TEST(HbgGraphResources, SameByteTotalsCannotMixArchitecturesOrTaskLayouts) {
    hbg::GraphResourceRequirements graphs[] = {resource_graph(1024, 2048, 0), resource_graph(1024, 2048, 0)};
    hbg::KernelResourceCapacity capacity;
    graphs[1].layout.architecture = hbg::RuntimeArchitecture::A5;
    EXPECT_EQ(hbg::KernelResourceCapacity::create(graphs, 2, capacity), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
    graphs[1] = graphs[0];
    ++graphs[1].layout.task_capacity;
    EXPECT_EQ(hbg::KernelResourceCapacity::create(graphs, 2, capacity), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
}

TEST(HbgGraphResources, RejectsOverflowWithoutPublishingCapacity) {
    auto graph = resource_graph(1024, 2048, UINT64_MAX);
    uint64_t bytes = 77;
    EXPECT_FALSE(graph.required_bytes(bytes));
    EXPECT_EQ(bytes, 77U);
    hbg::KernelResourceCapacity capacity;
    EXPECT_LT(hbg::KernelResourceCapacity::create(&graph, 1, capacity), 0);
    EXPECT_EQ(capacity.capacity().gm_heap_bytes, 0U);
    EXPECT_EQ(hbg::KernelResourceCapacity::create(nullptr, 1, capacity), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(HbgGraphResources, AggregateRegionMaximaMustAlsoFit) {
    const hbg::GraphResourceRequirements graphs[] = {
        resource_graph(UINT64_MAX - 4096, 2048, 0), resource_graph(1024, 8192, 0)
    };
    hbg::KernelResourceCapacity capacity;
    EXPECT_LT(hbg::KernelResourceCapacity::create(graphs, 2, capacity), 0);
}

TEST(HbgGraphResources, RejectsInvalidLayoutsAndMissingExecutionRegions) {
    auto graph = resource_graph(0, 2048, 0);
    uint64_t bytes = 0;
    EXPECT_FALSE(graph.required_bytes(bytes));
    graph = resource_graph(1024, 512, 0);
    EXPECT_FALSE(graph.required_bytes(bytes));
    graph = resource_graph(1024, 2048, 0);
    graph.layout.copied_end = graph.layout.copied_begin;
    EXPECT_FALSE(graph.required_bytes(bytes));
}

}  // namespace
