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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

#include "graph_cache.h"
#include "graph_execution.h"

namespace {

template <typename T>
uint32_t append_section(std::vector<std::byte> &image, const std::vector<T> &values) {
    if (values.empty()) return 0;
    const size_t offset = PTO2_ALIGN_UP(image.size(), alignof(T));
    image.resize(offset + values.size() * sizeof(T));
    std::memcpy(image.data() + offset, values.data(), values.size() * sizeof(T));
    return static_cast<uint32_t>(offset);
}

GraphTensor make_test_tensor(uint64_t address) {
    GraphTensor tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size = 64;
    tensor.extent_elem = 1;
    tensor.shapes[0] = 1;
    tensor.strides[0] = 1;
    tensor.ndims = 1;
    tensor.dtype = static_cast<uint8_t>(DataType::FLOAT32);
    tensor.is_contiguous = 1;
    return tensor;
}

std::vector<std::byte> make_test_definition(uint64_t graph_key, uint64_t boundary_address) {
    std::vector<std::byte> image(sizeof(GraphDefinition));

    std::vector<uint32_t> fanin_offsets{0, 0, 1};
    std::vector<uint16_t> fanin_indices{0};
    std::vector<uint32_t> fanout_offsets{0, 1, 1};
    std::vector<uint16_t> fanout_indices{1};
    std::vector<uint16_t> roots{0};
    std::vector<uint64_t> node_offsets{0, 64};
    std::vector<GraphNodeDefinition> nodes(2);
    for (GraphNodeDefinition &node : nodes) {
        std::fill(std::begin(node.kernel_id), std::end(node.kernel_id), INVALID_KERNEL_ID);
        node.kernel_id[0] = 42;
        node.active_mask = 1;
        node.logical_block_num = 1;
        node.total_required_subtasks = 1;
        node.tensor_count = 1;
        node.scalar_count = 1;
        node.total_output_size = 64;
    }
    nodes[1].tensor_offset = 1;
    nodes[1].scalar_offset = 1;
    std::vector<GraphTensor> tensors{make_test_tensor(boundary_address), make_test_tensor(boundary_address)};
    tensors[1].buffer_size = 32;
    std::vector<GraphTensorSourceRef> tensor_sources(2);
    tensor_sources[0].source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT);
    tensor_sources[1].source = static_cast<uint8_t>(GraphTensorSource::INTERNAL);
    tensor_sources[1].packed_offset = 16;
    std::vector<uint64_t> scalars{17, 18};

    GraphDefinition definition{};
    definition.full_key = graph_key;
    definition.required_heap = 128;
    definition.task_count = 2;
    definition.edge_count = 1;
    definition.root_count = 1;
    definition.boundary_count = 1;
    definition.tensor_arg_count = 2;
    definition.scalar_arg_count = 2;
    definition.off_fanin_offsets = append_section(image, fanin_offsets);
    definition.off_fanin_indices = append_section(image, fanin_indices);
    definition.off_fanout_offsets = append_section(image, fanout_offsets);
    definition.off_fanout_indices = append_section(image, fanout_indices);
    definition.off_root_indices = append_section(image, roots);
    definition.off_node_offsets = append_section(image, node_offsets);
    definition.off_nodes = append_section(image, nodes);
    definition.off_tensors = append_section(image, tensors);
    definition.off_tensor_sources = append_section(image, tensor_sources);
    definition.off_scalars = append_section(image, scalars);
    definition.total_bytes = static_cast<uint32_t>(image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));

    definition.content_hash = graph_hash_bytes(1469598103934665603ULL, image.data(), image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));
    return image;
}

std::vector<std::byte> make_test_submission(
    uint64_t graph_key, uint64_t boundary_address, uint64_t execution_storage, size_t execution_storage_bytes
) {
    const std::vector<std::byte> definition = make_test_definition(graph_key, boundary_address);
    const size_t definition_offset = PTO2_ALIGN_UP(sizeof(GraphSubmission), alignof(GraphDefinition));
    const size_t tensors_offset = PTO2_ALIGN_UP(definition_offset + definition.size(), alignof(GraphTensor));
    std::vector<std::byte> image(tensors_offset + sizeof(GraphTensor));
    std::memcpy(image.data() + definition_offset, definition.data(), definition.size());
    const GraphTensor boundary = make_test_tensor(boundary_address);
    std::memcpy(image.data() + tensors_offset, &boundary, sizeof(boundary));

    GraphSubmission submission{};
    submission.graph_key = graph_key;
    submission.execution_storage = execution_storage;
    submission.execution_storage_bytes = execution_storage_bytes;
    submission.total_bytes = static_cast<uint32_t>(image.size());
    submission.definition_offset = static_cast<uint32_t>(definition_offset);
    submission.tensors_offset = static_cast<uint32_t>(tensors_offset);
    submission.tensor_count = 1;
    std::memcpy(image.data(), &submission, sizeof(submission));
    return image;
}

class AlignedStorage {
public:
    explicit AlignedStorage(size_t bytes) :
        bytes_(bytes) {
        data_ = ::operator new(bytes, std::align_val_t(alignof(GraphNodeStorage)));
        std::memset(data_, 0, bytes);
    }

    ~AlignedStorage() { ::operator delete(data_, std::align_val_t(alignof(GraphNodeStorage))); }

    void *data() const { return data_; }
    size_t size() const { return bytes_; }

private:
    void *data_{nullptr};
    size_t bytes_{0};
};

}  // namespace

TEST(GraphCache, RejectsEmptyBoundary) {
    CoreTaskArgs args;

    EXPECT_FALSE(rt_graph_args_cacheable(args));
}

TEST(GraphExecutionStorage, ComputesAlignedExactSize) {
    constexpr int32_t NODE_COUNT = 7;
    constexpr size_t DEFINITION_BYTES = 321;
    constexpr uint32_t TENSOR_PATCH_COUNT = 11;
    size_t nodes_offset = 0;
    size_t tensor_patches_offset = 0;
    size_t definition_offset = 0;
    size_t storage_bytes = 0;

    ASSERT_TRUE(graph_execution_storage_layout(
        NODE_COUNT, TENSOR_PATCH_COUNT, DEFINITION_BYTES, &nodes_offset, &tensor_patches_offset, &definition_offset,
        &storage_bytes
    ));
    EXPECT_EQ(nodes_offset % alignof(GraphNodeStorage), 0U);
    EXPECT_EQ(tensor_patches_offset % alignof(GraphTensorAddressPatch), 0U);
    EXPECT_EQ(definition_offset % alignof(GraphDefinition), 0U);
    EXPECT_GE(tensor_patches_offset, nodes_offset + NODE_COUNT * sizeof(GraphNodeStorage));
    EXPECT_GE(definition_offset, tensor_patches_offset + TENSOR_PATCH_COUNT * sizeof(GraphTensorAddressPatch));
    EXPECT_GE(storage_bytes, definition_offset + DEFINITION_BYTES);
    EXPECT_EQ(storage_bytes % alignof(GraphNodeStorage), 0U);
}

TEST(GraphExecutionStorage, RejectsInvalidCapacity) {
    size_t storage_bytes = 0;

    EXPECT_FALSE(graph_execution_storage_bytes(0, 0, sizeof(GraphDefinition), &storage_bytes));
    EXPECT_FALSE(graph_execution_storage_bytes(1, 0, SIZE_MAX, &storage_bytes));
}

TEST(GraphExecutionReplay, AffineHitRefreshesOnlyDynamicFields) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    std::array<uint8_t, 128> first_heap{};
    std::array<uint8_t, 128> second_heap{};
    std::array<uint8_t, 64> first_boundary{};
    std::array<uint8_t, 64> second_boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(first_boundary.data()));
    size_t execution_bytes = 0;
    ASSERT_TRUE(graph_execution_storage_bytes(2, 2, definition.size(), &execution_bytes));
    AlignedStorage execution_storage(execution_bytes);
    std::vector<std::byte> submission_image = make_test_submission(
        GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(first_boundary.data()),
        reinterpret_cast<uint64_t>(execution_storage.data()), execution_storage.size()
    );
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 7);
    outer_task.packed_buffer_base = first_heap.data();
    outer_task.packed_buffer_end = first_heap.data() + first_heap.size();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task = &outer_task;
    outer_slot.graph_context = &submission;

    GraphExecution *execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    GraphNodeStorage &node = execution->node_storage[0];
    ASSERT_EQ(node.payload.scalar_count, 1);
    ASSERT_EQ(node.payload.tensor_count, 1);

    graph_execution_mark_completed(*execution);
    execution->retired_nodes.store(2, std::memory_order_release);
    submission.local_execution = 0;
    outer_task.task_id = PTO2TaskId::make(1, 8);
    outer_task.packed_buffer_base = second_heap.data();
    outer_task.packed_buffer_end = second_heap.data() + second_heap.size();
    auto *boundary = reinterpret_cast<GraphTensor *>(submission_image.data() + submission.tensors_offset);
    boundary->buffer_addr = reinterpret_cast<uint64_t>(second_boundary.data());

    execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    ASSERT_TRUE(execution->definition_affine_reuse);

    // Write probes make an otherwise same-valued store observable. An affine
    // replay must not touch these static fields; only the tensor address and
    // per-run descriptor/scheduling state below are dynamic.
    node.task.kernel_id[0] = 314;
    node.slot.active_mask = ActiveMask(3);
    node.payload.scalars[0] = 2718;
    node.payload.tensors[0].version = 1618;
    node.slot.completed_subtasks.store(1, std::memory_order_relaxed);
    node.payload.dispatch_fanin.store(1, std::memory_order_relaxed);

    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    EXPECT_EQ(node.task.kernel_id[0], 314);
    EXPECT_EQ(node.slot.active_mask.raw(), 3);
    EXPECT_EQ(node.payload.scalars[0], 2718U);
    EXPECT_EQ(node.payload.tensors[0].version, 1618);
    EXPECT_EQ(node.task.task_id, PTO2TaskId::make(1, (8U << 10U)));
    EXPECT_EQ(node.task.packed_buffer_base, second_heap.data());
    EXPECT_EQ(node.payload.tensors[0].buffer.addr, reinterpret_cast<uint64_t>(second_boundary.data()));
    EXPECT_EQ(
        execution->node_storage[1].payload.tensors[0].buffer.addr, reinterpret_cast<uint64_t>(second_heap.data() + 16)
    );
    EXPECT_EQ(node.slot.completed_subtasks.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(node.payload.dispatch_fanin.load(std::memory_order_relaxed), 0);
}
