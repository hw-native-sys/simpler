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
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "graph_cache.h"
#include "graph_execution.h"
#include "runtime_status/error_names.h"
#include "scheduler/pto_scheduler.h"

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

std::vector<std::byte>
make_test_definition(uint64_t graph_key, uint64_t boundary_address, uint32_t boundary_scalar_count = 1) {
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
    nodes[0].dump_metadata.dump_arg_mask = uint64_t{1} << 0;
    nodes[0].dump_metadata.scalar_dtypes[0] = static_cast<uint8_t>(DataType::FLOAT32);
    nodes[1].dump_metadata.dump_arg_mask = uint64_t{1} << 1;
    nodes[1].dump_metadata.scalar_dtypes[0] = static_cast<uint8_t>(DataType::INT32);
    nodes[1].tensor_offset = 1;
    nodes[1].scalar_offset = 1;
    std::vector<GraphTensor> tensors{make_test_tensor(boundary_address), make_test_tensor(boundary_address)};
    tensors[1].buffer_size = 32;
    std::vector<GraphTensorSourceRef> tensor_sources(2);
    tensor_sources[0].source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT);
    tensor_sources[1].source = static_cast<uint8_t>(GraphTensorSource::INTERNAL);
    tensor_sources[1].packed_offset = 16;
    std::vector<uint64_t> scalars{0, 18};
    std::vector<GraphScalarSourceRef> scalar_sources(2);
    scalar_sources[0].source = static_cast<uint8_t>(GraphScalarSource::BOUNDARY);
    scalar_sources[0].source_index = boundary_scalar_count - 1;
    scalar_sources[1].source = static_cast<uint8_t>(GraphScalarSource::STATIC_VALUE);

    GraphDefinition definition{};
    definition.full_key = graph_key;
    definition.required_heap = 128;
    definition.task_count = 2;
    definition.edge_count = 1;
    definition.root_count = 1;
    definition.boundary_count = 1;
    definition.boundary_scalar_count = boundary_scalar_count;
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
    definition.off_scalar_sources = append_section(image, scalar_sources);
    size_t execution_storage_bytes = 0;
    graph_execution_storage_bytes(
        static_cast<int32_t>(definition.task_count), definition.tensor_arg_count, definition.scalar_arg_count,
        &execution_storage_bytes
    );
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);
    definition.total_bytes = static_cast<uint32_t>(image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));

    definition.content_hash = graph_definition_content_hash(image.data(), image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));
    return image;
}

// The pool holds boundary_scalar_count entries; boundary_scalar lands in the
// last one, where make_test_definition's BOUNDARY ref reads it.
std::vector<std::byte> make_test_submission(
    uint64_t graph_key, uint64_t boundary_address, uint64_t boundary_scalar, uint32_t boundary_scalar_count = 1
) {
    const size_t tensors_offset = PTO2_ALIGN_UP(sizeof(GraphSubmission), alignof(GraphTensor));
    const size_t scalars_offset = PTO2_ALIGN_UP(tensors_offset + sizeof(GraphTensor), alignof(uint64_t));
    std::vector<std::byte> image(scalars_offset + boundary_scalar_count * sizeof(uint64_t));
    const GraphTensor boundary = make_test_tensor(boundary_address);
    std::memcpy(image.data() + tensors_offset, &boundary, sizeof(boundary));
    std::memcpy(
        image.data() + scalars_offset + (boundary_scalar_count - 1) * sizeof(uint64_t), &boundary_scalar,
        sizeof(boundary_scalar)
    );

    GraphSubmission submission{};
    submission.graph_key = graph_key;
    submission.total_bytes = static_cast<uint32_t>(image.size());
    submission.tensors_offset = static_cast<uint32_t>(tensors_offset);
    submission.tensor_count = 1;
    submission.scalars_offset = static_cast<uint32_t>(scalars_offset);
    submission.scalar_count = boundary_scalar_count;
    std::memcpy(image.data(), &submission, sizeof(submission));
    return image;
}

// A Definition device object exactly as upload_graph_submissions builds it:
// [GraphDefinitionHeader][Definition image].
class TestDefinitionObject {
public:
    explicit TestDefinitionObject(const std::vector<std::byte> &definition, uint32_t retained_definition_bytes = 0) {
        const size_t object_bytes = sizeof(GraphDefinitionHeader) + definition.size();
        data_ = ::operator new(object_bytes, std::align_val_t(alignof(GraphDefinitionHeader)));
        std::memset(data_, 0, object_bytes);
        auto *header = static_cast<GraphDefinitionHeader *>(data_);
        const auto *def = reinterpret_cast<const GraphDefinition *>(definition.data());
        header->magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        header->verify_state.store(
            static_cast<uint32_t>(GraphDefinitionVerifyState::UPLOADED), std::memory_order_relaxed
        );
        header->definition_bytes =
            retained_definition_bytes == 0 ? static_cast<uint32_t>(definition.size()) : retained_definition_bytes;
        header->content_hash = def->content_hash;
        header->full_key = def->full_key;
        std::memcpy(
            static_cast<uint8_t *>(data_) + sizeof(GraphDefinitionHeader), definition.data(), definition.size()
        );
    }

    ~TestDefinitionObject() { ::operator delete(data_, std::align_val_t(alignof(GraphDefinitionHeader))); }

    uint64_t address() const { return reinterpret_cast<uint64_t>(data_); }
    uint64_t hash() const { return static_cast<GraphDefinitionHeader *>(data_)->content_hash; }
    GraphDefinitionVerifyState verify_state() const {
        return static_cast<GraphDefinitionVerifyState>(
            static_cast<GraphDefinitionHeader *>(data_)->verify_state.load(std::memory_order_acquire)
        );
    }

private:
    void *data_{nullptr};
};

class AlignedStorage {
public:
    explicit AlignedStorage(size_t bytes, uint8_t fill = 0) :
        bytes_(bytes) {
        data_ = ::operator new(bytes, std::align_val_t(alignof(GraphNodeStorage)));
        std::memset(data_, fill, bytes);
    }

    ~AlignedStorage() { ::operator delete(data_, std::align_val_t(alignof(GraphNodeStorage))); }

    void *data() const { return data_; }
    uint8_t *bytes() const { return static_cast<uint8_t *>(data_); }
    size_t size() const { return bytes_; }

private:
    void *data_{nullptr};
    size_t bytes_{0};
};

// One outer GRAPH task's heap allocation, laid out as graph_submit_definition
// sizes it: required_heap bytes of node outputs followed by the execution
// storage. `fill` stands in for whatever the reclaimed heap last held.
class OuterHeap {
public:
    OuterHeap(const std::vector<std::byte> &definition_image, uint8_t fill = 0) {
        const auto *definition = reinterpret_cast<const GraphDefinition *>(definition_image.data());
        heap_bytes_ = static_cast<size_t>(definition->required_heap);
        storage_ = std::make_unique<AlignedStorage>(heap_bytes_ + definition->execution_storage_bytes, fill);
    }

    uint8_t *base() const { return storage_->bytes(); }
    uint8_t *end() const { return storage_->bytes() + storage_->size(); }
    // Where localize places the GraphExecution.
    void *execution() const { return storage_->bytes() + heap_bytes_; }

private:
    size_t heap_bytes_{0};
    std::unique_ptr<AlignedStorage> storage_;
};

}  // namespace

TEST(GraphDefinitionHash, IgnoresOnlyEmbeddedContentHash) {
    std::vector<std::byte> image = make_test_definition(23, 0x1000);
    const uint64_t expected = graph_definition_content_hash(image.data(), image.size());

    const uint64_t replacement = 0x0123456789abcdefULL;
    std::memcpy(image.data() + offsetof(GraphDefinition, content_hash), &replacement, sizeof(replacement));
    EXPECT_EQ(graph_definition_content_hash(image.data(), image.size()), expected);

    image.back() ^= std::byte{1};
    EXPECT_NE(graph_definition_content_hash(image.data(), image.size()), expected);
}

TEST(GraphCache, RejectsEmptyBoundary) {
    GraphTaskArgs args;

    EXPECT_FALSE(rt_graph_args_cacheable(args));
}

TEST(GraphCache, AcceptsBoundaryScalars) {
    std::array<uint8_t, 64> boundary{};
    const GraphTensor packed = make_test_tensor(reinterpret_cast<uint64_t>(boundary.data()));
    ChipTensor tensor{};
    graph_tensor_unpack(packed, &tensor);

    GraphTaskArgs args;
    args.add_input(tensor);
    args.add_scalar(uint32_t{17});

    EXPECT_TRUE(rt_graph_args_cacheable(args));
}

TEST(GraphCache, ConfigValuesSelectDifferentDefinitions) {
    constexpr uint64_t GRAPH_ID = 0x1234;

    EXPECT_NE(rt_graph_make_key(GRAPH_ID, 0), rt_graph_make_key(GRAPH_ID, 1));
    EXPECT_EQ(rt_graph_make_key(GRAPH_ID, 0), rt_graph_make_key(GRAPH_ID, 0));
}

TEST(GraphScalarProvenance, ForwardedScalarRetainsBoundarySource) {
    uint32_t value = 17;
    CoreTaskArgs boundary_args;
    boundary_args.add_scalar(value, value);
    boundary_args.anchor_scalar_sources();
    CoreTaskArgs forwarded_args;
    forwarded_args.copy_scalars_from(boundary_args, 1, 1);
    CoreTaskArgs node_args;

    node_args.copy_scalars_from(forwarded_args, 0, 1);

    EXPECT_EQ(node_args.scalar_source(0), static_cast<const void *>(&std::as_const(boundary_args).scalar(1)));
}

TEST(GraphScalarProvenance, MutableAccessInvalidatesForwardedSource) {
    CoreTaskArgs boundary_args;
    boundary_args.add_scalar(uint32_t{17});
    boundary_args.anchor_scalar_sources();
    CoreTaskArgs node_args;
    node_args.copy_scalars_from(boundary_args, 0, 1);
    ASSERT_NE(node_args.scalar_source(0), nullptr);

    node_args.scalar(0) = 18;

    EXPECT_EQ(node_args.scalar_source(0), nullptr);
    EXPECT_EQ(
        node_args.invalidated_scalar_source(0), static_cast<const void *>(&std::as_const(boundary_args).scalar(0))
    );
}

TEST(GraphExecutionStorage, ComputesAlignedExactSize) {
    constexpr int32_t NODE_COUNT = 7;
    constexpr uint32_t TENSOR_ARGS = 11;
    constexpr uint32_t SCALAR_ARGS = 5;
    GraphExecutionStorageLayout layout{};

    ASSERT_TRUE(graph_execution_storage_layout(NODE_COUNT, TENSOR_ARGS, SCALAR_ARGS, &layout));
    EXPECT_EQ(layout.nodes_offset % alignof(GraphNodeStorage), 0U);
    EXPECT_GE(layout.nodes_offset, sizeof(GraphExecution));
    // The node array is type-strided now that the payload carries no array of its own,
    // and the two argument pools follow it.
    EXPECT_EQ(layout.tensors_offset, layout.nodes_offset + NODE_COUNT * sizeof(GraphNodeStorage));
    EXPECT_EQ(layout.tensors_offset % alignof(ChipTensor), 0U);
    EXPECT_EQ(layout.scalars_offset, layout.tensors_offset + TENSOR_ARGS * sizeof(ChipTensor));
    EXPECT_EQ(layout.total_bytes, layout.scalars_offset + SCALAR_ARGS * sizeof(uint64_t));
}

// The pools are sized by the Definition's arg tables, so a Definition whose nodes
// declare fewer arguments reserves less — the same property the node stride used to
// carry, moved to the pools.
TEST(GraphExecutionStorage, NarrowerArgTablesReserveLess) {
    constexpr int32_t NODE_COUNT = 4;
    size_t wide = 0;
    size_t narrow = 0;
    ASSERT_TRUE(graph_execution_storage_bytes(NODE_COUNT, 32, 16, &wide));
    ASSERT_TRUE(graph_execution_storage_bytes(NODE_COUNT, 4, 2, &narrow));
    EXPECT_LT(narrow, wide);
}

TEST(GraphExecutionStorage, RejectsInvalidNodeCount) {
    size_t storage_bytes = 0;

    EXPECT_FALSE(graph_execution_storage_bytes(0, 1, 1, &storage_bytes));
    EXPECT_FALSE(graph_execution_storage_bytes(-1, 1, 1, &storage_bytes));
    EXPECT_FALSE(graph_execution_storage_bytes(static_cast<int32_t>(GRAPH_MAX_NODES) + 1, 1, 1, &storage_bytes));
    // A Definition with no arguments at all still needs its node array.
    EXPECT_TRUE(graph_execution_storage_bytes(1, 0, 0, &storage_bytes));
    EXPECT_GE(storage_bytes, sizeof(GraphExecution) + sizeof(GraphNodeStorage));
}

// A resubmission gets the same heap block back, so the bytes it starts from are
// the previous execution's. Every field must come from the Definition or this
// submission's boundary, never from what the block last held.
TEST(GraphExecutionReplay, ResubmissionRebuildsFromDefinition) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    std::array<uint8_t, 64> first_boundary{};
    std::array<uint8_t, 64> second_boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(first_boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    std::vector<std::byte> submission_image =
        make_test_submission(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(first_boundary.data()), 17);
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());
    submission.definition_addr = definition_object.address();
    submission.definition_hash = definition_object.hash();

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task.set(&outer_task);
    outer_slot.graph_context = &submission;

    GraphExecution *execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    // Execution storage is the heap allocation's tail: anything lower would
    // overlap the node outputs occupying the leading required_heap bytes.
    EXPECT_EQ(static_cast<void *>(execution), heap.execution());
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    GraphNodeStorage &node = execution->node_at(0);
    ASSERT_EQ(node.payload.scalar_count, 1);
    ASSERT_EQ(node.payload.tensor_count, 1);
    EXPECT_EQ(node.payload.scalar_data()[0], 17U);
    EXPECT_EQ(execution->node_at(1).payload.scalar_data()[0], 18U);
    EXPECT_EQ(node.payload.dump_metadata.dump_arg_mask, uint64_t{1} << 0);
    EXPECT_EQ(node.payload.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_EQ(execution->node_at(1).payload.dump_metadata.dump_arg_mask, uint64_t{1} << 1);
    EXPECT_EQ(execution->node_at(1).payload.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::INT32));

    graph_execution_mark_completed(*execution);
    execution->retired_nodes.store(2, std::memory_order_release);
    submission.local_execution = 0;
    outer_task.task_id = PTO2TaskId::make(1, 8);
    auto *boundary = reinterpret_cast<GraphTensor *>(submission_image.data() + submission.tensors_offset);
    boundary->buffer_addr = reinterpret_cast<uint64_t>(second_boundary.data());
    auto *boundary_scalar = reinterpret_cast<uint64_t *>(submission_image.data() + submission.scalars_offset);
    *boundary_scalar = 99;

    // Poison every field the rebuild is responsible for restoring. A replay that
    // preserved any of them would leave the poison observable.
    node.task.kernel_id[0] = 314;
    node.slot.active_mask = ActiveMask(3);
    node.payload.scalar_data()[0] = 2718;
    execution->node_at(1).payload.scalar_data()[0] = 31415;
    node.payload.tensor_data()[0].version = 1618;
    node.slot.completed_subtasks.store(1, std::memory_order_relaxed);
    node.payload.dispatch_fanin.store(1, std::memory_order_relaxed);

    execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    // Same block: it is this allocation's own tail, so the rebuild lands on the
    // bytes the previous execution left behind.
    EXPECT_EQ(static_cast<void *>(execution), heap.execution());

    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    EXPECT_EQ(node.task.kernel_id[0], 42);
    EXPECT_EQ(node.slot.active_mask.raw(), 1);
    EXPECT_EQ(node.payload.scalar_data()[0], 99U);
    EXPECT_EQ(execution->node_at(1).payload.scalar_data()[0], 18U);
    EXPECT_EQ(node.payload.tensor_data()[0].version, 0);
    EXPECT_EQ(node.task.task_id, PTO2TaskId::make(1, (8U << 10U)));
    EXPECT_EQ(node.task.packed_buffer_base, heap.base());
    EXPECT_EQ(node.payload.tensor_data()[0].buffer.addr, reinterpret_cast<uint64_t>(second_boundary.data()));
    EXPECT_EQ(execution->node_at(1).payload.tensor_data()[0].buffer.addr, reinterpret_cast<uint64_t>(heap.base() + 16));
    EXPECT_EQ(node.slot.completed_subtasks.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(node.payload.dispatch_fanin.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(node.payload.dump_metadata.dump_arg_mask, uint64_t{1} << 0);
}

// The boundary scalar pool is bounded by the Graph boundary contract
// (GRAPH_MAX_SCALAR_ARGS), not by a single task payload's MAX_SCALAR_ARGS —
// a node stages at most MAX_SCALAR_ARGS entries from it, but the pool itself
// may be wider.
TEST(GraphExecutionReplay, LocalizesBoundaryScalarPoolWiderThanTaskPayload) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    constexpr uint32_t POOL = MAX_SCALAR_ARGS + 3;
    static_assert(POOL <= GRAPH_MAX_SCALAR_ARGS);
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), POOL);
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    std::vector<std::byte> submission_image =
        make_test_submission(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 21, POOL);
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());
    submission.definition_addr = definition_object.address();
    submission.definition_hash = definition_object.hash();

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task.set(&outer_task);
    outer_slot.graph_context = &submission;

    GraphExecution *execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    EXPECT_EQ(execution->node_storage[0].payload.scalar_data()[0], 21U);
}

TEST(GraphExecutionReplay, RejectsBoundaryScalarPoolBeyondContract) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    constexpr uint32_t POOL = GRAPH_MAX_SCALAR_ARGS + 1;
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), POOL);
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    std::vector<std::byte> submission_image =
        make_test_submission(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 21, POOL);
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());
    submission.definition_addr = definition_object.address();
    submission.definition_hash = definition_object.hash();

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task.set(&outer_task);
    outer_slot.graph_context = &submission;

    EXPECT_EQ(graph_execution_localize(outer_slot), nullptr);
}

TEST(GraphDefinitionObject, RejectsDefinitionBeyondRetainedBytes) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::array<uint8_t, 64> boundary{};
    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    ASSERT_GT(definition.size(), sizeof(GraphDefinition));
    const TestDefinitionObject definition_object(definition, sizeof(GraphDefinition));
    OuterHeap heap(definition);
    std::vector<std::byte> submission_image =
        make_test_submission(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 17);
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());
    submission.definition_addr = definition_object.address();
    submission.definition_hash = definition_object.hash();

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task.set(&outer_task);
    outer_slot.graph_context = &submission;

    EXPECT_EQ(graph_execution_localize(outer_slot), nullptr);
    EXPECT_EQ(definition_object.verify_state(), GraphDefinitionVerifyState::INVALID);
}

TEST(GraphSubmissionWire, RequiresExactAvailableSize) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::vector<std::byte> image = make_test_submission(GRAPH_KEY_VALUE, 0x1000, 17);
    const auto &submission = *reinterpret_cast<const GraphSubmission *>(image.data());

    EXPECT_TRUE(graph_submission_wire_size_valid(submission, image.size()));
    EXPECT_FALSE(graph_submission_wire_size_valid(submission, image.size() - 1));
    EXPECT_FALSE(graph_submission_wire_size_valid(submission, image.size() + 1));
}

TEST(GraphSubmissionActivationGate, ActivatesExactlyOnceUnderContention) {
    constexpr int ITERATIONS = 1000;
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        GraphSubmission submission{};
        std::atomic<int32_t> activations{0};
        std::thread prepared([&] {
            if (graph_submission_signal(submission, 0x1)) activations.fetch_add(1, std::memory_order_relaxed);
        });
        std::thread ready([&] {
            if (graph_submission_signal(submission, 0x2)) activations.fetch_add(1, std::memory_order_relaxed);
        });
        prepared.join();
        ready.join();
        EXPECT_EQ(submission.activation_gate, 0x3U);
        EXPECT_EQ(activations.load(std::memory_order_relaxed), 1);
    }
}

TEST(GraphSubmissionActivationGate, RetriesDoNotReactivate) {
    GraphSubmission submission{};

    EXPECT_FALSE(graph_submission_signal(submission, 0x1));
    EXPECT_TRUE(graph_submission_signal(submission, 0x2));
    EXPECT_FALSE(graph_submission_signal(submission, 0x1));
    EXPECT_FALSE(graph_submission_signal(submission, 0x2));
}

TEST(GraphExecutionErrors, ReadyQueueOverflowHasTriageText) {
    EXPECT_STREQ(error_name(PTO2_ERROR_READY_QUEUE_OVERFLOW), "READY_QUEUE_OVERFLOW");
    EXPECT_STRNE(error_desc(PTO2_ERROR_READY_QUEUE_OVERFLOW), "");
    EXPECT_STRNE(error_hint(PTO2_ERROR_READY_QUEUE_OVERFLOW), "");
}

TEST(GraphExecutionErrors, GraphReadyQueueOverflowIsReported) {
    PTO2SharedMemoryHeader header{};
    PTO2SchedulerState scheduler{};
    scheduler.sm_header = &header;
    PTO2ReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.graph_ready_queue.slots = queue_slots;
    scheduler.graph_ready_queue.capacity = 2;
    scheduler.graph_ready_queue.mask = 1;
    scheduler.graph_ready_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.graph_ready_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    PTO2TaskSlotState graph_slots[3]{};
    for (PTO2TaskSlotState &slot : graph_slots) {
        slot.task_kind = TaskKind::GRAPH;
    }

    scheduler.push_ready_routed(&graph_slots[0]);
    scheduler.push_ready_routed(&graph_slots[1]);
    scheduler.push_ready_routed(&graph_slots[2]);

    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), PTO2_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(GraphExecutionErrors, GraphPrepareQueueOverflowIsReported) {
    PTO2SharedMemoryHeader header{};
    PTO2SchedulerState scheduler{};
    scheduler.sm_header = &header;
    PTO2ReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.slots = queue_slots;
    scheduler.graph_prepare_queue.capacity = 2;
    scheduler.graph_prepare_queue.mask = 1;
    scheduler.graph_prepare_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    PTO2TaskSlotState graph_slots[3]{};

    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[0], 10, 3));
    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[1], 11, 3));
    EXPECT_FALSE(scheduler.push_graph_prepare(&graph_slots[2], 12, 3));

    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), PTO2_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(std::memory_order_acquire), 3);
    EXPECT_EQ(header.sched_error_bitmap.load(std::memory_order_acquire), 1U << 3);
}

TEST(GraphExecutionErrors, InvalidNodeCompletionIsReported) {
    PTO2SchedulerState scheduler{};
    PTO2TaskSlotState slot{};
    slot.task_kind = TaskKind::GRAPH_NODE;

    const PTO2SchedulerState::TaskCompletionOutcome outcome = scheduler.complete_task(slot);

    EXPECT_EQ(outcome.error_code, PTO2_ERROR_INVALID_ARGS);
    EXPECT_EQ(outcome.stream_tasks_completed, 0);
}

TEST(GraphExecutionProgress, InternalNodeResolutionIsNotAHostCompletion) {
    PTO2SchedulerState scheduler{};
    GraphDefinition definition{};
    GraphNodeStorage node{};
    GraphExecution execution{};
    execution.definition = &definition;
    execution.nodes = &node;
    execution.node_storage = &node;
    execution.node_count = 1;
    execution.remaining_nodes.store(1, std::memory_order_relaxed);
    execution.state.store(GraphExecutionState::ACTIVE, std::memory_order_relaxed);
    node.slot.task_kind = TaskKind::GRAPH_NODE;
    node.slot.graph_context = &execution;
    node.slot.graph_node_index = 0;

    AsyncWaitList wait_list{};
    wait_list.entries[0].slot_state = &node.slot;
    wait_list.entries[0].task_token = PTO2TaskId::make(0, 1);
    wait_list.entries[0].normal_done = true;
    wait_list.count = 1;

    const AsyncPollResult result = wait_list.poll_and_complete<false>(nullptr, &scheduler);

    EXPECT_EQ(result.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(result.resolved, 1);
    EXPECT_EQ(result.completed, 0);
}

// Device-side execution storage is not guaranteed to be zero-initialized.
// localize + materialize must produce a fully valid execution from arbitrary
// initial bytes.
TEST(GraphExecutionMaterialize, DirtyStorageYieldsValidExecution) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x9753;
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    std::vector<std::byte> submission_image =
        make_test_submission(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 17);
    auto &submission = *reinterpret_cast<GraphSubmission *>(submission_image.data());
    submission.definition_addr = definition_object.address();
    submission.definition_hash = definition_object.hash();

    PTO2TaskDescriptor outer_task{};
    outer_task.task_id = PTO2TaskId::make(1, 5);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    PTO2TaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.task.set(&outer_task);
    outer_slot.graph_context = &submission;

    GraphExecution *execution = graph_execution_localize(outer_slot);
    ASSERT_NE(execution, nullptr);
    EXPECT_EQ(static_cast<void *>(execution), heap.execution());
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);

    // Every observable scheduling field must be a materialize-written value,
    // not the 0xAA fill: state machine, counters and atomics all start from
    // values only the device side wrote.
    for (int32_t i = 0; i < execution->node_count; ++i) {
        const GraphNodeStorage &node = execution->node_at(i);
        ASSERT_EQ(node.slot.task_state.load(std::memory_order_relaxed), PTO2_TASK_PENDING);
        ASSERT_EQ(node.slot.task_kind, TaskKind::GRAPH_NODE);
        ASSERT_EQ(node.slot.completed_subtasks.load(std::memory_order_relaxed), 0);
        ASSERT_EQ(node.payload.dispatch_fanin.load(std::memory_order_relaxed), 0);
        ASSERT_EQ(node.payload.tensor_count, 1);
        ASSERT_EQ(node.payload.scalar_count, 1);
        // A tensor address of 0xAAAAAAAAAAAAAAAA would mean the fill leaked
        // through into a field the scheduler later dereferences.
        ASSERT_NE(node.payload.tensor_data()[0].buffer.addr, 0xAAAAAAAAAAAAAAAAULL);
        // make_test_definition assigns node i the heap offset 64*i, so the
        // packed window starts at outer_base + 64*i, not at outer_base.
        ASSERT_EQ(node.task.packed_buffer_base, static_cast<void *>(heap.base() + static_cast<size_t>(i) * 64));
    }
    EXPECT_EQ(execution->materialized_nodes, execution->node_count);
    EXPECT_EQ(execution->consumed_tensor_args, 2U);
}
