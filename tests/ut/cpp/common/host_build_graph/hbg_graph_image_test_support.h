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

#pragma once

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

#include "host_build_graph/runtime_status.h"

#include "graph_cache.h"
#include "graph_config.h"
#include "graph_execution.h"
#include "graph_image_view.h"
#include "runtime_status/error_names.h"
#include "host_build_graph/runtime_types.h"
#include "task_interface/assert_compat.h"
#include "host_build_graph/task_id.h"

namespace {

template <typename T>
uint32_t append_section(std::vector<std::byte> &image, const std::vector<T> &values) {
    if (values.empty()) return 0;
    const size_t offset = CHIP_ALIGN_UP(image.size(), alignof(T));
    image.resize(offset + values.size() * sizeof(T));
    std::memcpy(image.data() + offset, values.data(), values.size() * sizeof(T));
    return static_cast<uint32_t>(offset);
}

simpler::hbg::TensorData make_test_tensor(uint64_t address, TaskId owner) {
    simpler::hbg::TensorData tensor{};
    tensor.buffer.addr = address;
    tensor.buffer.size = 64;
    tensor.owner_task_id = owner;
    tensor.extent_elem_cache = 1;
    tensor.shapes[0] = 1;
    tensor.strides[0] = 1;
    tensor.ndims = 1;
    tensor.dtype = DataType::FLOAT32;
    tensor.is_contiguous = true;
    return tensor;
}

// The same tensor in the form a boundary holds it. A boundary is this invocation's actual
// arguments rather than anything the image carries, so it is a simpler::hbg::Tensor, and
// it carries the caller's own provenance.
simpler::hbg::Tensor make_test_boundary(uint64_t address) {
    simpler::hbg::Tensor tensor{};
    tensor.init_from(make_test_tensor(address, TaskId::invalid()));
    return tensor;
}

// What task 0 — this body's only root — is shaped like. Materialization decides
// a root's staging verdict from the root itself, so each variant is one term of
// that decision.
enum class TestRoot { ORDINARY, PREDICATED, DUMMY };

// Ways an image can carry ED_FLAG_CANDIDATE without the conjunction the recorder
// decides it by. graph_fill_definition produces none of these; bind_graph_topology
// is what has to say so.
enum class TestEdDefect { NONE, CANDIDATE_ON_ROOT, CANDIDATE_WITH_PREDICATE, CANDIDATE_ON_DUMMY };

// Whether this body reaches its buffers through the boundary or entirely through its own
// heap. NONE is the shape a Config-driven Graph has: nothing is passed as a tensor, so no
// body tensor can name a parameter and the boundary carries scalars only.
enum class TestBoundary { WITH_TENSOR, NONE };

std::vector<std::byte> make_test_definition(
    uint64_t graph_key, uint64_t boundary_address, uint32_t boundary_scalar_count = 1,
    TestRoot root_variant = TestRoot::ORDINARY, TestEdDefect ed_defect = TestEdDefect::NONE,
    TestBoundary boundary_variant = TestBoundary::WITH_TENSOR
) {
    std::vector<std::byte> image(sizeof(GraphDefinition));

    std::vector<int32_t> fanin_offsets{0, 0, 1};
    std::vector<uint16_t> fanin_indices{0};
    std::vector<int32_t> fanout_offsets{0, 1, 1};
    std::vector<uint16_t> fanout_indices{1};
    std::vector<uint16_t> roots{0};
    std::vector<uint64_t> sub_task_offsets{0, 64};
    std::vector<SubTaskDefinition> tasks(2);
    for (SubTaskDefinition &task : tasks) {
        std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
        task.kernel_id[0] = 42;
        task.active_mask = LaneMask::aic();
        task.logical_block_num = 1;
        task.total_required_subtasks = 1;
        task.tensor_count = 1;
        task.scalar_count = 1;
        task.total_output_size = 64;
    }
    tasks[0].dump_metadata.dump_arg_mask = uint64_t{1} << 0;
    tasks[0].dump_metadata.scalar_dtypes[0] = static_cast<uint8_t>(DataType::FLOAT32);
    tasks[1].dump_metadata.dump_arg_mask = uint64_t{1} << 1;
    tasks[1].dump_metadata.scalar_dtypes[0] = static_cast<uint8_t>(DataType::INT32);
    tasks[1].tensor_offset = 1;
    tasks[1].scalar_offset = 1;
    std::vector<GraphPredicate> predicates;
    if (root_variant == TestRoot::PREDICATED) {
        // The attribute bit and the one-based slot are written together; a
        // Definition where they disagree is rejected before the verdict is read.
        TaskAttrs root_attrs{};
        root_attrs.set_predicate();
        tasks[0].task_attrs = root_attrs.raw();
        tasks[0].predicate_slot = 1;
        GraphPredicate predicate{};
        // The operand came through the boundary, which its owner is what says: replay
        // takes the buffer from parameter 0, so the recorded address is dropped.
        predicate.operand = make_test_tensor(0, TaskId::make_param(0));
        predicate.elem_size = static_cast<uint8_t>(get_element_size(DataType::FLOAT32));
        predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
        predicates.push_back(predicate);
    } else if (root_variant == TestRoot::DUMMY) {
        // A dependency-only task: no core mask, and therefore no subtask to
        // require and no kernel on any slot — bind_graph_topology pairs each
        // mask bit with its slot's kernel id. active_mask is what
        // ChipTaskSlotState::task_kind is read from.
        tasks[0].active_mask = LaneMask::none();
        tasks[0].total_required_subtasks = 0;
        std::fill(std::begin(tasks[0].kernel_id), std::end(tasks[0].kernel_id), INVALID_KERNEL_ID);
    }
    // Task 0 is this body's root (empty CSR row); task 1 has one producer.
    if (ed_defect == TestEdDefect::CANDIDATE_ON_ROOT) {
        tasks[0].ed_flags |= ED_FLAG_CANDIDATE;
    } else if (ed_defect == TestEdDefect::CANDIDATE_WITH_PREDICATE) {
        tasks[1].ed_flags |= ED_FLAG_CANDIDATE;
        TaskAttrs attrs{};
        attrs.set_predicate();
        tasks[1].task_attrs = attrs.raw();
        tasks[1].predicate_slot = static_cast<uint16_t>(predicates.size() + 1);
        GraphPredicate predicate{};
        predicate.operand = make_test_tensor(0, TaskId::make_param(0));
        predicate.elem_size = static_cast<uint8_t>(get_element_size(DataType::FLOAT32));
        predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
        predicates.push_back(predicate);
    } else if (ed_defect == TestEdDefect::CANDIDATE_ON_DUMMY) {
        // Cleared kernel ids keep the mask/kernel pairing check from rejecting
        // this image first, so only the flag conjunction can.
        tasks[1].ed_flags |= ED_FLAG_CANDIDATE;
        tasks[1].active_mask = LaneMask::none();
        tasks[1].total_required_subtasks = 0;
        std::fill(std::begin(tasks[1].kernel_id), std::end(tasks[1].kernel_id), INVALID_KERNEL_ID);
    }
    // Tensor 0 came through the boundary: its owner names parameter 0, and its recorded
    // address is dropped because replay takes the buffer from this invocation's argument.
    // Tensor 1 is task 0's output, so its address is an offset into the graph heap.
    //
    // Under TestBoundary::NONE tensor 0 is heap-owned too, at an offset clear of tensor 1:
    // with no boundary tensor there is no parameter for it to name, and graph_rebind_tensor
    // refuses a PARAM owner whose index is past boundary_tensor_count.
    std::vector<simpler::hbg::TensorData> tensors{
        boundary_variant == TestBoundary::WITH_TENSOR ? make_test_tensor(0, TaskId::make_param(0)) :
                                                        make_test_tensor(64, TaskId::make_sub_task(0, 0)),
        make_test_tensor(16, TaskId::make_sub_task(0, 0)),
    };
    tensors[1].buffer.size = 32;
    std::vector<uint64_t> scalars{0, 18};
    std::vector<GraphScalarInheritance> scalar_inheritance{
        GraphScalarInheritance::from_boundary(boundary_scalar_count - 1),
        GraphScalarInheritance::self_value(),
    };

    GraphDefinition definition{};
    definition.graph_key = graph_key;
    definition.required_heap = 128;
    definition.task_count = 2;
    definition.edge_count = 1;
    definition.root_count = 1;
    definition.boundary_tensor_count = boundary_variant == TestBoundary::WITH_TENSOR ? 1 : 0;
    definition.boundary_scalar_count = boundary_scalar_count;
    definition.tensor_arg_count = 2;
    definition.scalar_arg_count = 2;
    definition.off_fanin_offsets = append_section(image, fanin_offsets);
    definition.off_fanin_indices = append_section(image, fanin_indices);
    definition.off_fanout_offsets = append_section(image, fanout_offsets);
    definition.off_fanout_indices = append_section(image, fanout_indices);
    definition.off_root_indices = append_section(image, roots);
    definition.off_sub_task_offsets = append_section(image, sub_task_offsets);
    definition.off_sub_tasks = append_section(image, tasks);
    definition.off_tensors = append_section(image, tensors);
    definition.off_scalars = append_section(image, scalars);
    definition.off_scalar_inheritance = append_section(image, scalar_inheritance);
    if (!predicates.empty()) {
        definition.predicate_count = static_cast<uint32_t>(predicates.size());
        definition.off_predicates = append_section(image, predicates);
    }
    size_t execution_storage_bytes = 0;
    graph_execution_storage_bytes(
        definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
        &execution_storage_bytes
    );
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);
    definition.total_bytes = static_cast<uint32_t>(image.size());
    std::memcpy(image.data(), &definition, sizeof(definition));
    return image;
}

// A Definition device object exactly as bind_graph_definitions builds it:
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
        header->definition_bytes =
            retained_definition_bytes == 0 ? static_cast<uint32_t>(definition.size()) : retained_definition_bytes;
        header->graph_key = def->graph_key;
        std::memcpy(
            static_cast<uint8_t *>(data_) + sizeof(GraphDefinitionHeader), definition.data(), definition.size()
        );
        object_bytes_ = object_bytes;
    }

    ~TestDefinitionObject() { ::operator delete(data_, std::align_val_t(alignof(GraphDefinitionHeader))); }

    uint64_t address() const { return reinterpret_cast<uint64_t>(data_); }
    // The object as a one-object Definition section, which is what a reader
    // decodes: the image starts one header in, and that offset is what a task
    // carries.
    GraphImageView view() const {
        GraphImageView v{};
        v.section = static_cast<const std::byte *>(data_);
        v.bytes = static_cast<uint32_t>(object_bytes_);
        return v;
    }
    static uint32_t image_offset() { return static_cast<uint32_t>(sizeof(GraphDefinitionHeader)); }
    // The object's bytes, writable, so a case can poison the source while it is
    // still alive rather than inferring anything from a freed allocation.
    std::byte *mutable_section() { return static_cast<std::byte *>(data_); }
    size_t section_bytes() const { return object_bytes_; }
    const GraphDefinition *definition() const {
        return reinterpret_cast<const GraphDefinition *>(
            static_cast<const uint8_t *>(data_) + sizeof(GraphDefinitionHeader)
        );
    }
    // Frames the object around a Graph other than the one its image holds, which is
    // the shape a mis-packed shared block produces.
    void reframe_graph_key(uint64_t graph_key) { static_cast<GraphDefinitionHeader *>(data_)->graph_key = graph_key; }

private:
    void *data_{nullptr};
    size_t object_bytes_{0};
};

class AlignedStorage {
public:
    explicit AlignedStorage(size_t bytes, uint8_t fill = 0) :
        bytes_(bytes) {
        data_ = ::operator new(bytes, std::align_val_t(alignof(ChipTaskStorage)));
        std::memset(data_, fill, bytes);
    }

    ~AlignedStorage() { ::operator delete(data_, std::align_val_t(alignof(ChipTaskStorage))); }

    void *data() const { return data_; }
    uint8_t *bytes() const { return static_cast<uint8_t *>(data_); }
    size_t size() const { return bytes_; }

private:
    void *data_{nullptr};
    size_t bytes_{0};
};

// One outer GRAPH task's heap allocation and payload, laid out as
// graph_submit_definition sizes them. Device localization constructs the
// execution in the heap tail and reads invocation boundaries from the payload.
//
// The boundary regions are sized by what graph_submit_outer reserves — one
// simpler::hbg::Tensor slot per boundary tensor, and the ARG_POOL_ALIGN-rounded scalar
// span — rather than by the GraphTaskArgs element caps, so the fixture reserves what
// production reserves for the widest legal boundary. They are members, not separate
// allocations: a payload names its regions through an int32 SelfRelativePtr delta, which
// silently binds as unbound past ±2 GiB.
class OuterHeap {
public:
    static constexpr size_t TENSOR_SLOTS = GRAPH_MAX_TENSOR_ARGS;
    static constexpr size_t SCALAR_SPAN =
        CHIP_ALIGN_UP(static_cast<size_t>(GRAPH_MAX_SCALAR_ARGS), ARG_POOL_ALIGN / sizeof(uint64_t));

    OuterHeap(const std::vector<std::byte> &definition_image, uint8_t fill = 0) {
        const auto *definition = reinterpret_cast<const GraphDefinition *>(definition_image.data());
        heap_bytes_ = static_cast<size_t>(definition->required_heap);
        storage_ = std::make_unique<AlignedStorage>(heap_bytes_ + definition->execution_storage_bytes, fill);
        storage_entry_.task.packed_buffer_base = base();
        storage_entry_.task.packed_buffer_end = end();
        storage_entry_.slot.task_kind = TaskKind::GRAPH;
        storage_entry_.payload.bind_regions(boundary_tensors_.data(), boundary_scalars_.data(), nullptr);
    }

    uint8_t *base() const { return storage_->bytes(); }
    uint8_t *end() const { return storage_->bytes() + storage_->size(); }
    void *execution() const { return base() + heap_bytes_; }

    // The tensor region past the boundary. Production reserves one slot per boundary
    // tensor, so a write anywhere beyond them lands in another task's arguments on a real
    // ring.
    const std::byte *boundary_tail(uint32_t boundary_count) const {
        return reinterpret_cast<const std::byte *>(boundary_tensors_.data() + boundary_count);
    }
    size_t boundary_tail_bytes(uint32_t boundary_count) const {
        return (TENSOR_SLOTS - boundary_count) * sizeof(simpler::hbg::Tensor);
    }

    GraphExecution *initialize_execution(
        const TestDefinitionObject &definition_object, uint64_t boundary_address, uint64_t boundary_scalar
    ) {
        const GraphDefinition *definition = definition_object.definition();
        if (static_cast<size_t>(definition->boundary_tensor_count) > TENSOR_SLOTS ||
            definition->boundary_scalar_count > SCALAR_SPAN) {
            return nullptr;
        }
        std::memset(boundary_tensors_.data(), 0, TENSOR_SLOTS * sizeof(simpler::hbg::Tensor));
        // Only as many slots as the Definition counts. A boundary of none reserves no slot,
        // so slot 0 is already the tail the cases below expect untouched.
        if (definition->boundary_tensor_count > 0) {
            boundary_tensors_[0] = make_test_boundary(boundary_address);
        }
        storage_entry_.payload.tensor_count = definition->boundary_tensor_count;
        storage_entry_.payload.scalar_count = definition->boundary_scalar_count;
        std::fill_n(boundary_scalars_.data(), definition->boundary_scalar_count, uint64_t{0});
        if (definition->boundary_scalar_count != 0) {
            boundary_scalars_[definition->boundary_scalar_count - 1] = boundary_scalar;
        }
        storage_entry_.slot.graph_context = nullptr;
        storage_entry_.slot.to_descriptor().graph_definition_offset = TestDefinitionObject::image_offset();
        image_ = definition_object.view();
        return graph_execution_localize(storage_entry_.slot, image_);
    }

    // The view the last initialize_execution built, so a case materializes
    // through the same section its localize read.
    const GraphImageView &image() const { return image_; }

private:
    GraphImageView image_{};
    size_t heap_bytes_{0};
    std::unique_ptr<AlignedStorage> storage_;
    // One storage entry, as production has it: a slot state reaches its descriptor
    // and payload by ChipTaskStorage's layout, so three separate members would
    // resolve outside themselves.
    ChipTaskStorage storage_entry_{};
    std::array<simpler::hbg::Tensor, TENSOR_SLOTS> boundary_tensors_{};
    std::array<uint64_t, SCALAR_SPAN> boundary_scalars_{};
};

}  // namespace
