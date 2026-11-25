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

#include <cstring>
#include <numeric>
#include <vector>

#include "host_build_graph/graph.h"
#include "graph_validation.h"

namespace {
struct DefinitionFixture {
    Graph header{};
    std::vector<std::byte> bytes{sizeof(Graph)};

    template <typename T>
    uint32_t append(const std::vector<T> &values) {
        if (values.empty()) return 0;
        const size_t start = (bytes.size() + alignof(T) - 1) & ~(alignof(T) - 1);
        bytes.resize(start + values.size() * sizeof(T));
        std::memcpy(bytes.data() + start, values.data(), values.size() * sizeof(T));
        return static_cast<uint32_t>(start);
    }
    template <typename T>
    void put(uint32_t offset, uint32_t index, const T &value) {
        std::memcpy(bytes.data() + offset + sizeof(T) * index, &value, sizeof(T));
    }
    explicit DefinitionFixture(
        const std::vector<std::vector<uint16_t>> &fanin, bool dummy = false, int32_t scalars_per_child = 0,
        int32_t tensors_per_child = 0
    ) {
        header.graph_key = 77;
        header.task_count = static_cast<int32_t>(fanin.size());
        std::vector<int32_t> in_rows{0}, out_rows{0};
        std::vector<uint16_t> in_edges, out_edges, roots;
        std::vector<std::vector<uint16_t>> fanout(fanin.size());
        for (uint16_t child = 0; child < fanin.size(); ++child) {
            if (fanin[child].empty()) roots.push_back(child);
            for (auto producer : fanin[child]) {
                in_edges.push_back(producer);
                fanout[producer].push_back(child);
            }
            in_rows.push_back(static_cast<int32_t>(in_edges.size()));
        }
        for (const auto &row : fanout) {
            out_edges.insert(out_edges.end(), row.begin(), row.end());
            out_rows.push_back(static_cast<int32_t>(out_edges.size()));
        }
        header.edge_count = static_cast<int32_t>(in_edges.size());
        header.root_count = static_cast<int32_t>(roots.size());
        header.off_fanout_offsets = append(out_rows);
        header.off_fanout_indices = append(out_edges);
        header.off_fanin_offsets = append(in_rows);
        header.off_fanin_indices = append(in_edges);
        header.off_root_indices = append(roots);
        header.off_sub_task_offsets = append(std::vector<uint64_t>(fanin.size(), 0));
        std::vector<SubTaskDefinition> tasks(fanin.size());
        for (size_t i = 0; i < tasks.size(); ++i) {
            auto &task = tasks[i];
            task.kernel_id[0] = dummy ? -1 : 0;
            task.kernel_id[1] = task.kernel_id[2] = -1;
            task.active_mask = dummy ? LaneMask::none() : LaneMask::aic();
            task.logical_block_num = 1;
            task.total_required_subtasks = dummy ? 0 : 1;
            task.scalar_count = scalars_per_child;
            task.scalar_offset = static_cast<int32_t>(i) * scalars_per_child;
            task.tensor_count = tensors_per_child;
            task.tensor_offset = static_cast<int32_t>(i) * tensors_per_child;
        }
        header.off_sub_tasks = append(tasks);
        if (tensors_per_child != 0) {
            header.boundary_tensor_count = 1;
            header.tensor_arg_count = header.task_count * tensors_per_child;
            simpler::hbg::TensorData tensor{};
            tensor.owner_task_id = TaskId::make_param(0);
            tensor.ndims = 1;
            tensor.dtype = DataType::INT32;
            tensor.address_space = AddressSpace::DEVICE;
            tensor.shapes[0] = tensor.strides[0] = 1;
            tensor.extent_elem_cache = 1;
            header.off_tensors = append(std::vector<simpler::hbg::TensorData>(header.tensor_arg_count, tensor));
        }
        if (scalars_per_child != 0) {
            header.scalar_arg_count = header.task_count * scalars_per_child;
            header.off_scalars = append(std::vector<uint64_t>(header.scalar_arg_count, 42));
            header.off_scalar_inheritance = append(
                std::vector<GraphScalarInheritance>(header.scalar_arg_count, GraphScalarInheritance::self_value())
            );
        }
        header.total_bytes = static_cast<uint32_t>(bytes.size());
        sync();
    }
    void sync() { std::memcpy(bytes.data(), &header, sizeof(header)); }
    bool valid(std::string &error) const {
        return validate_graph_image(bytes.data(), bytes.size(), header.graph_key, &error);
    }
};
}  // namespace

TEST(HbgGraphValidation, AcceptsWideFanInAndMaximumArgumentPools) {
    std::string error;
    for (int32_t count : {1, 128, 129, SUB_TASK_MAX_NUM - 1}) {
        std::vector<std::vector<uint16_t>> edges(static_cast<size_t>(count + 1));
        edges.back().resize(count);
        std::iota(edges.back().begin(), edges.back().end(), 0);
        DefinitionFixture fixture(edges);
        ASSERT_TRUE(fixture.valid(error)) << error;
    }
    DefinitionFixture fixture(
        std::vector<std::vector<uint16_t>>(SUB_TASK_MAX_NUM), false, MAX_SCALAR_ARGS, MAX_TENSOR_ARGS
    );
    EXPECT_TRUE(fixture.valid(error)) << error;
}

TEST(HbgGraphValidation, RejectsOverlappingAndGappedChildArgumentSpans) {
    std::string error;
    for (bool tensors : {false, true}) {
        DefinitionFixture fixture({{}, {}}, false, tensors ? 0 : 1, tensors ? 1 : 0);
        ASSERT_TRUE(fixture.valid(error)) << error;
        SubTaskDefinition child{};
        std::memcpy(&child, fixture.bytes.data() + fixture.header.off_sub_tasks + sizeof(child), sizeof(child));
        if (tensors) child.tensor_offset = 0;
        else child.scalar_offset = 0;
        fixture.put(fixture.header.off_sub_tasks, 1, child);
        EXPECT_FALSE(fixture.valid(error));
        EXPECT_EQ(error, "noncanonical child argument span");

        child.tensor_offset = tensors ? 1 : 0;
        child.scalar_offset = tensors ? 0 : 1;
        fixture.put(fixture.header.off_sub_tasks, 1, child);
        std::memcpy(&child, fixture.bytes.data() + fixture.header.off_sub_tasks, sizeof(child));
        if (tensors) child.tensor_count = 0;
        else child.scalar_count = 0;
        fixture.put(fixture.header.off_sub_tasks, 0, child);
        EXPECT_FALSE(fixture.valid(error));
        EXPECT_EQ(error, "noncanonical child argument span");
    }
}

TEST(HbgGraphValidation, RejectsUnusedArgumentPoolTailAndOverflowingChildSpan) {
    DefinitionFixture fixture({{}, {}}, false, 1, 1);
    std::string error;
    ASSERT_TRUE(fixture.valid(error)) << error;
    SubTaskDefinition child{};
    std::memcpy(&child, fixture.bytes.data() + fixture.header.off_sub_tasks + sizeof(child), sizeof(child));
    child.tensor_count = 0;
    fixture.put(fixture.header.off_sub_tasks, 1, child);
    EXPECT_FALSE(fixture.valid(error));
    EXPECT_EQ(error, "incomplete child argument pool coverage");
    child.tensor_count = INT32_MAX;
    fixture.put(fixture.header.off_sub_tasks, 1, child);
    EXPECT_FALSE(fixture.valid(error));
    EXPECT_EQ(error, "invalid child shape or argument span");
}

TEST(HbgGraphValidation, RejectsDuplicateEdgesRootsAndAsymmetricCsr) {
    std::string error;
    DefinitionFixture duplicate({{}, {0}, {0, 1}});
    duplicate.put<uint16_t>(duplicate.header.off_fanin_indices, 2, 0);
    EXPECT_FALSE(duplicate.valid(error));
    DefinitionFixture wrong_fanout(std::vector<std::vector<uint16_t>>{{}, {}, {0}, {1}});
    wrong_fanout.put<uint16_t>(wrong_fanout.header.off_fanout_indices, 0, 3);
    EXPECT_FALSE(wrong_fanout.valid(error));
    DefinitionFixture duplicate_root({{}, {}, {0, 1}});
    duplicate_root.put<uint16_t>(duplicate_root.header.off_root_indices, 1, 0);
    EXPECT_FALSE(duplicate_root.valid(error));
    DefinitionFixture self_edge(std::vector<std::vector<uint16_t>>{{}, {0}});
    self_edge.put<uint16_t>(self_edge.header.off_fanin_indices, 0, 1);
    EXPECT_FALSE(self_edge.valid(error));
}

TEST(HbgGraphValidation, RejectsBadExtentsBeforeReading) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}});
    std::string error;
    ASSERT_TRUE(fixture.valid(error)) << error;
    EXPECT_FALSE(validate_graph_image(fixture.bytes.data(), sizeof(Graph) - 1, fixture.header.graph_key, &error));
    fixture.header.off_sub_tasks = UINT32_MAX;
    fixture.sync();
    EXPECT_FALSE(fixture.valid(error));
    fixture.header.task_count = INT32_MAX;
    fixture.sync();
    EXPECT_FALSE(fixture.valid(error));
}

TEST(HbgGraphValidation, CooperativeShapeStillRejectsMalformedTopology) {
    DefinitionFixture fixture(std::vector<std::vector<uint16_t>>{{}, {0}});
    SubTaskDefinition child{};
    std::memcpy(&child, fixture.bytes.data() + fixture.header.off_sub_tasks, sizeof(child));
    child.logical_block_num = 2;
    child.total_required_subtasks = 2;
    fixture.put(fixture.header.off_sub_tasks, 0, child);
    std::string error;
    ASSERT_TRUE(fixture.valid(error)) << error;
    fixture.put<uint16_t>(fixture.header.off_fanin_indices, 0, 1);
    EXPECT_FALSE(fixture.valid(error));
}
