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

#include "host_build_graph/runtime_status.h"

#include "graph_cache.h"
#include "graph_execution.h"
#include "runtime_status/error_names.h"
#include "scheduler/scheduler.h"
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

// What task 0 — this body's only root — is shaped like. Materialization decides
// a root's staging verdict from the root itself, so each variant is one term of
// that decision.
enum class TestRoot { ORDINARY, PREDICATED, DUMMY };

// Ways an image can carry ED_FLAG_CANDIDATE without the conjunction the recorder
// decides it by. graph_fill_definition produces none of these; bind_graph_topology
// is what has to say so.
enum class TestEdDefect { NONE, CANDIDATE_ON_ROOT, CANDIDATE_WITH_PREDICATE, CANDIDATE_ON_DUMMY };

std::vector<std::byte> make_test_definition(
    uint64_t graph_key, uint64_t boundary_address, uint32_t boundary_scalar_count = 1,
    TestRoot root_variant = TestRoot::ORDINARY, TestEdDefect ed_defect = TestEdDefect::NONE
) {
    std::vector<std::byte> image(sizeof(GraphDefinition));

    std::vector<int32_t> fanin_offsets{0, 0, 1};
    std::vector<uint16_t> fanin_indices{0};
    std::vector<int32_t> fanout_offsets{0, 1, 1};
    std::vector<uint16_t> fanout_indices{1};
    std::vector<uint16_t> roots{0};
    std::vector<uint64_t> in_graph_task_offsets{0, 64};
    std::vector<InGraphTaskDefinition> tasks(2);
    for (InGraphTaskDefinition &task : tasks) {
        std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
        task.kernel_id[0] = 42;
        task.active_mask = 1;
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
        predicate.operand = make_test_tensor(boundary_address);
        predicate.operand_source.source_kind = static_cast<uint8_t>(GraphTensorSourceKind::BOUNDARY_EXACT);
        predicate.elem_size = static_cast<uint8_t>(get_element_size(DataType::FLOAT32));
        predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
        predicates.push_back(predicate);
    } else if (root_variant == TestRoot::DUMMY) {
        // A dependency-only task: no core mask, and therefore no subtask to
        // require and no kernel on any slot — bind_graph_topology pairs each
        // mask bit with its slot's kernel id. active_mask is what
        // ChipTaskSlotState::task_kind is read from.
        tasks[0].active_mask = 0;
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
        predicate.operand = make_test_tensor(boundary_address);
        predicate.operand_source.source_kind = static_cast<uint8_t>(GraphTensorSourceKind::BOUNDARY_EXACT);
        predicate.elem_size = static_cast<uint8_t>(get_element_size(DataType::FLOAT32));
        predicate.op = static_cast<uint8_t>(PredicateOp::EQ);
        predicates.push_back(predicate);
    } else if (ed_defect == TestEdDefect::CANDIDATE_ON_DUMMY) {
        // Cleared kernel ids keep the mask/kernel pairing check from rejecting
        // this image first, so only the flag conjunction can.
        tasks[1].ed_flags |= ED_FLAG_CANDIDATE;
        tasks[1].active_mask = 0;
        tasks[1].total_required_subtasks = 0;
        std::fill(std::begin(tasks[1].kernel_id), std::end(tasks[1].kernel_id), INVALID_KERNEL_ID);
    }
    std::vector<GraphTensor> tensors{make_test_tensor(boundary_address), make_test_tensor(boundary_address)};
    tensors[1].buffer_size = 32;
    std::vector<GraphTensorSourceRef> tensor_sources(2);
    tensor_sources[0].source_kind = static_cast<uint8_t>(GraphTensorSourceKind::BOUNDARY_EXACT);
    tensor_sources[1].source_kind = static_cast<uint8_t>(GraphTensorSourceKind::INTERNAL);
    tensor_sources[1].packed_offset = 16;
    std::vector<uint64_t> scalars{0, 18};
    std::vector<GraphScalarInheritance> scalar_inheritance{
        GraphScalarInheritance::from_boundary(boundary_scalar_count - 1),
        GraphScalarInheritance::self_value(),
    };

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
    definition.off_in_graph_task_offsets = append_section(image, in_graph_task_offsets);
    definition.off_in_graph_tasks = append_section(image, tasks);
    definition.off_tensors = append_section(image, tensors);
    definition.off_tensor_sources = append_section(image, tensor_sources);
    definition.off_scalars = append_section(image, scalars);
    definition.off_scalar_inheritance = append_section(image, scalar_inheritance);
    if (!predicates.empty()) {
        definition.predicate_count = static_cast<uint32_t>(predicates.size());
        definition.off_predicates = append_section(image, predicates);
    }
    size_t execution_storage_bytes = 0;
    graph_execution_storage_bytes(
        definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, &execution_storage_bytes
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
        header->full_key = def->full_key;
        std::memcpy(
            static_cast<uint8_t *>(data_) + sizeof(GraphDefinitionHeader), definition.data(), definition.size()
        );
    }

    ~TestDefinitionObject() { ::operator delete(data_, std::align_val_t(alignof(GraphDefinitionHeader))); }

    uint64_t address() const { return reinterpret_cast<uint64_t>(data_); }
    const GraphDefinition *definition() const {
        return reinterpret_cast<const GraphDefinition *>(
            static_cast<const uint8_t *>(data_) + sizeof(GraphDefinitionHeader)
        );
    }
    // Frames the object around a Graph other than the one its image holds, which is
    // the shape a mis-packed shared block produces.
    void reframe_full_key(uint64_t full_key) { static_cast<GraphDefinitionHeader *>(data_)->full_key = full_key; }

private:
    void *data_{nullptr};
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
// The boundary regions are sized by the same helpers graph_submit_outer reserves
// with — the simpler::hbg::Tensor slot span that holds GRAPH_MAX_TENSOR_ARGS packed
// GraphTensors, and the ARG_POOL_ALIGN-rounded scalar span — rather than by the
// GraphTaskArgs element caps, so the fixture reserves what production reserves for
// the widest legal boundary. They are members, not separate allocations: a payload
// names its regions through an int32 SelfRelativePtr delta, which silently binds as
// unbound past ±2 GiB.
class OuterHeap {
public:
    static constexpr size_t TENSOR_SLOTS = graph_boundary_tensor_pool_slots(GRAPH_MAX_TENSOR_ARGS);
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

    // The tensor region past the packed boundary. Production reserves only the packed
    // span rounded up to a whole slot, so a write anywhere beyond the packed bytes
    // lands in another task's arguments on a real ring.
    const std::byte *boundary_tail(uint32_t boundary_count) const {
        return reinterpret_cast<const std::byte *>(boundary_tensors_.data()) + boundary_count * sizeof(GraphTensor);
    }
    size_t boundary_tail_bytes(uint32_t boundary_count) const {
        return TENSOR_SLOTS * sizeof(simpler::hbg::Tensor) - boundary_count * sizeof(GraphTensor);
    }

    GraphExecution *initialize_execution(
        const TestDefinitionObject &definition_object, uint64_t boundary_address, uint64_t boundary_scalar
    ) {
        const GraphDefinition *definition = definition_object.definition();
        if (graph_boundary_tensor_pool_slots(definition->boundary_count) > TENSOR_SLOTS ||
            definition->boundary_scalar_count > SCALAR_SPAN) {
            return nullptr;
        }
        std::memset(boundary_tensors_.data(), 0, TENSOR_SLOTS * sizeof(simpler::hbg::Tensor));
        const GraphTensor boundary = make_test_tensor(boundary_address);
        new (boundary_tensors_.data()) GraphTensor{boundary};
        storage_entry_.payload.tensor_count = definition->boundary_count;
        storage_entry_.payload.scalar_count = definition->boundary_scalar_count;
        std::fill_n(boundary_scalars_.data(), definition->boundary_scalar_count, uint64_t{0});
        if (definition->boundary_scalar_count != 0) {
            boundary_scalars_[definition->boundary_scalar_count - 1] = boundary_scalar;
        }
        storage_entry_.slot.graph_context = const_cast<GraphDefinition *>(definition);
        return graph_execution_localize(storage_entry_.slot);
    }

private:
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

TEST(GraphCache, RejectsEmptyBoundary) {
    GraphTaskArgs args;

    EXPECT_FALSE(rt_graph_args_cacheable(args));
}

TEST(GraphCache, AcceptsBoundaryScalars) {
    std::array<uint8_t, 64> boundary{};
    const GraphTensor packed = make_test_tensor(reinterpret_cast<uint64_t>(boundary.data()));
    simpler::hbg::Tensor tensor{};
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

// Arg's storage stays unreachable only while the base is private. Under a public base an
// implicit derived-to-base conversion reaches the same subobject, whose members are
// public there however Arg hides their names -- so tags_, tensors_ and scalars_ would be
// readable raw, each without the array that qualifies it.
static_assert(
    !std::is_convertible_v<
        const CoreTaskArgs *,
        const TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType> *>,
    "Arg must not be convertible to its storage base"
);

// The slot array is handed out as void*, so index it the way recording does.
const void *slot_addr(const void *base, int32_t i) { return static_cast<const uint64_t *>(base) + i; }

// A boundary parameter must name no origin of its own. That is what makes the boundary's
// slot array the basis recording resolves against: scalar(i) then folds to
// &scalars_[i], and subtracting the base yields i. Were a boundary parameter to name an
// origin -- a caller local, or a scratch buffer used while deep-copying -- scalar(i)
// would hand out that address instead, it would fall outside the slot array, and
// graph_classify_scalars would record the parameter as static. Every dynamic parameter
// would silently stop being refreshed on replay.
TEST(GraphScalarProvenance, AStaticParameterIsItsOwnOrigin) {
    GraphTaskArgs boundary_args;
    uint64_t scratch = 18;
    boundary_args.add_static_scalar(scratch);
    CoreTaskArgs task_args;

    task_args.add_scalar(boundary_args.scalar(0));

    EXPECT_EQ(task_args.scalar_origin(0), boundary_args.scalar_slot_base())
        << "a task slot must name the boundary's own slot, not whatever the value came from";
    EXPECT_NE(task_args.scalar_origin(0), &scratch);
}

TEST(GraphScalarProvenance, AnLvalueDeclaresADynamicParameter) {
    GraphTaskArgs args;
    uint32_t token_pos = 17;

    // An lvalue is a parameter the caller holds and may change between invocations; a
    // literal cannot be changed by anyone. Width does not enter into it -- the value is
    // converted here, where its type is still known.
    args.add_scalar(token_pos);
    args.add_scalar(uint32_t{18});
    args.add_static_scalar(token_pos);

    EXPECT_TRUE(args.scalar_dynamic(0));
    EXPECT_EQ(args.scalar_origin(0), &token_pos);
    EXPECT_FALSE(args.scalar_dynamic(1));
    EXPECT_EQ(args.scalar_origin(1), nullptr);
    EXPECT_FALSE(args.scalar_dynamic(2)) << "add_static_scalar overrides value category";
    EXPECT_EQ(args.scalar<uint32_t>(2), 17u);
}

TEST(GraphScalarProvenance, ForwardedScalarRetainsBoundarySource) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint32_t{17}, uint32_t{18});
    CoreTaskArgs task_args;

    task_args.add_scalar(boundary_args.scalar(1));

    EXPECT_TRUE(task_args.scalar_dynamic(0));
    EXPECT_EQ(task_args.scalar_origin(0), slot_addr(boundary_args.scalar_slot_base(), 1));
    EXPECT_EQ(task_args.scalar<uint64_t>(0), uint64_t{18});
}

TEST(GraphScalarProvenance, ForwardedScalarNamesOriginThroughAnIntermediary) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint32_t{17}, uint32_t{18});
    CoreTaskArgs forwarded_args;
    forwarded_args.add_scalar(boundary_args.scalar(1));
    CoreTaskArgs task_args;

    task_args.add_scalar(forwarded_args.scalar(0));

    // A -> B -> C still records A: the handle names the origin, not the Arg it came through.
    EXPECT_TRUE(task_args.scalar_dynamic(0));
    EXPECT_EQ(task_args.scalar_origin(0), slot_addr(boundary_args.scalar_slot_base(), 1));
}

TEST(GraphScalarProvenance, FreezingAParameterDropsItsOrigin) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint32_t{17}, uint32_t{18});
    CoreTaskArgs task_args;

    // add_static_scalar resolves the handle to its value: this is how an enclosing Graph's
    // parameter is deliberately frozen rather than followed.
    task_args.add_static_scalar(boundary_args.scalar(1));

    EXPECT_FALSE(task_args.scalar_dynamic(0));
    EXPECT_EQ(task_args.scalar_origin(0), nullptr);
    EXPECT_EQ(task_args.scalar<uint64_t>(0), uint64_t{18});
}

TEST(GraphScalarProvenance, ValueScalarHoldsItsOwnValue) {
    CoreTaskArgs task_args;
    task_args.add_scalar(uint32_t{17}, 2.5F);

    EXPECT_FALSE(task_args.scalar_dynamic(0));
    EXPECT_FALSE(task_args.scalar_dynamic(1));
    EXPECT_EQ(task_args.scalar<uint64_t>(0), uint64_t{17});
    EXPECT_EQ(task_args.scalar<float>(1), 2.5F);
}

TEST(GraphScalarProvenance, ZeroInitialisedSlotsReadAsStaticZero) {
    CoreTaskArgs task_args;
    uint64_t packed[4] = {1, 2, 3, 4};

    task_args.add_scalar(uint64_t{0});
    task_args.pack_scalars(packed);

    EXPECT_FALSE(task_args.scalar_dynamic(0));
    EXPECT_EQ(packed[0], uint64_t{0});
}

TEST(GraphScalarProvenance, PackCopiesEveryValue) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint64_t{100}, uint64_t{200});
    CoreTaskArgs task_args;
    // A slot is always a value, whether or not it names an origin, so a mixed run copies
    // out whole.
    task_args.add_scalar(uint64_t{7});
    task_args.add_scalar(boundary_args.scalar(1));
    task_args.add_scalar(uint64_t{9});
    uint64_t packed[3] = {0, 0, 0};

    task_args.pack_scalars(packed);

    EXPECT_EQ(packed[0], uint64_t{7});
    EXPECT_EQ(packed[1], uint64_t{200});
    EXPECT_EQ(packed[2], uint64_t{9});
}

TEST(GraphScalarProvenance, AValueOutlivesItsOrigin) {
    CoreTaskArgs task_args;
    {
        uint64_t transient = 42;
        task_args.add_scalar(transient);
    }

    // The origin now dangles, and that is by design: the value was copied at add_scalar
    // time, and the address is only ever compared, never read through.
    EXPECT_TRUE(task_args.scalar_dynamic(0));
    EXPECT_EQ(task_args.scalar<uint64_t>(0), uint64_t{42});
}

TEST(GraphScalarProvenance, TaskSlotsInheritFromEachOther) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint32_t{17}, uint32_t{18});
    CoreTaskArgs a;
    a.add_scalar(boundary_args.scalar(1));
    CoreTaskArgs b;

    // Any slot is an inheritance source, not just a boundary parameter.
    b.add_scalar(a.scalar(0));

    // scalar(i) folds: a[0] already names an origin, so b records that origin rather than
    // a[0] itself. A chain is therefore one hop and recording resolves it without a walk.
    EXPECT_TRUE(b.scalar_dynamic(0));
    EXPECT_EQ(b.scalar_origin(0), slot_addr(boundary_args.scalar_slot_base(), 1));
    EXPECT_EQ(b.scalar<uint64_t>(0), uint64_t{18});
}

TEST(GraphScalarProvenance, InheritingAValueSlotNamesThatSlot) {
    CoreTaskArgs a;
    a.add_scalar(uint64_t{7});
    CoreTaskArgs b;

    b.add_scalar(a.scalar(0));

    // a[0] names no origin, so it is itself the origin.
    EXPECT_TRUE(b.scalar_dynamic(0));
    EXPECT_EQ(b.scalar_origin(0), a.scalar_slot_base());
    EXPECT_EQ(b.scalar<uint64_t>(0), uint64_t{7});
}

TEST(GraphExecutionStorage, ComputesAlignedExactSize) {
    constexpr int32_t TASK_COUNT = 7;
    constexpr uint32_t TENSOR_ARGS = 11;
    constexpr uint32_t SCALAR_ARGS = 5;
    GraphExecutionStorageLayout layout{};

    ASSERT_TRUE(graph_execution_storage_layout(TASK_COUNT, TENSOR_ARGS, SCALAR_ARGS, &layout));
    EXPECT_EQ(layout.tasks_offset % alignof(ChipTaskStorage), 0U);
    EXPECT_GE(layout.tasks_offset, sizeof(GraphExecution));
    EXPECT_EQ(layout.tensors_offset, layout.tasks_offset + TASK_COUNT * sizeof(ChipTaskStorage));
    EXPECT_EQ(layout.tensors_offset % alignof(simpler::hbg::Tensor), 0U);
    EXPECT_EQ(layout.scalars_offset, layout.tensors_offset + TENSOR_ARGS * sizeof(simpler::hbg::Tensor));
    // The state array is last and byte-aligned, so it starts exactly where the
    // scalar pool ends and closes the image.
    EXPECT_EQ(layout.states_offset, layout.scalars_offset + SCALAR_ARGS * sizeof(uint64_t));
    EXPECT_EQ(layout.total_bytes, layout.states_offset + TASK_COUNT * sizeof(std::atomic<ChipTaskState>));
}

// The outer Graph payload's tensor region is counted in simpler::hbg::Tensor pool slots but
// holds densely packed GraphTensor values, so the slot count must cover the packed
// bytes and be the smallest count that does — anything larger silently overdraws the
// shared pool, anything smaller lets localize read past the region.
TEST(GraphBoundaryPool, TensorSlotsCoverPackedBytesMinimally) {
    EXPECT_EQ(graph_boundary_tensor_pool_slots(0), 0U);
    for (uint32_t count = 1; count <= GRAPH_MAX_TENSOR_ARGS; ++count) {
        const size_t slots = graph_boundary_tensor_pool_slots(count);
        const size_t packed = static_cast<size_t>(count) * sizeof(GraphTensor);
        EXPECT_GE(slots * sizeof(simpler::hbg::Tensor), packed) << "count " << count;
        EXPECT_LT((slots - 1) * sizeof(simpler::hbg::Tensor), packed) << "count " << count;
    }
}

// A Graph boundary is GraphTaskArgs-wide while the pools budget MAX_TENSOR_ARGS /
// MAX_SCALAR_ARGS per window slot, so the widest legal boundary draws several slots'
// worth. graph_submit_outer's preflight exists because of that gap; pin the gap itself
// so a cap or type-size change cannot quietly close or widen it unnoticed.
TEST(GraphBoundaryPool, WidestBoundaryExceedsOneSlotBudget) {
    EXPECT_GT(graph_boundary_tensor_pool_slots(GRAPH_MAX_TENSOR_ARGS), static_cast<size_t>(MAX_TENSOR_ARGS));
    EXPECT_GT(
        static_cast<size_t>(
            CHIP_ALIGN_UP(static_cast<int32_t>(GRAPH_MAX_SCALAR_ARGS), ARG_POOL_ALIGN / (int32_t)sizeof(uint64_t))
        ),
        static_cast<size_t>(MAX_SCALAR_ARGS)
    );
    // The widest boundary that still fits one slot's tensor budget.
    EXPECT_LE(
        graph_boundary_tensor_pool_slots(MAX_TENSOR_ARGS * sizeof(simpler::hbg::Tensor) / sizeof(GraphTensor)),
        static_cast<size_t>(MAX_TENSOR_ARGS)
    );
}

// The pools are sized by the Definition's arg tables, so a Definition whose
// tasks declare fewer arguments reserves less.
TEST(GraphExecutionStorage, NarrowerArgTablesReserveLess) {
    constexpr int32_t TASK_COUNT = 4;
    size_t wide = 0;
    size_t narrow = 0;
    ASSERT_TRUE(graph_execution_storage_bytes(TASK_COUNT, 32, 16, &wide));
    ASSERT_TRUE(graph_execution_storage_bytes(TASK_COUNT, 4, 2, &narrow));
    EXPECT_LT(narrow, wide);
}

TEST(GraphExecutionStorage, RejectsInvalidInGraphTaskCount) {
    size_t storage_bytes = 0;

    EXPECT_FALSE(graph_execution_storage_bytes(0, 1, 1, &storage_bytes));
    EXPECT_FALSE(graph_execution_storage_bytes(-1, 1, 1, &storage_bytes));
    EXPECT_FALSE(graph_execution_storage_bytes(MAX_IN_GRAPH_TASKS + 1, 1, 1, &storage_bytes));
    // A Definition with no arguments at all still needs its in-graph task array.
    EXPECT_TRUE(graph_execution_storage_bytes(1, 0, 0, &storage_bytes));
    EXPECT_GE(storage_bytes, sizeof(GraphExecution) + sizeof(ChipTaskStorage));
}

// A resubmission gets the same heap tail back, so the bytes it starts from are
// the previous execution's. Every field must come from the Definition or this
// invocation's payload, never from what the heap last held.
TEST(GraphExecutionReplay, ResubmissionRebuildsFromDefinition) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    std::array<uint8_t, 64> first_boundary{};
    std::array<uint8_t, 64> second_boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(first_boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    GraphExecution *execution =
        heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(first_boundary.data()), 17);
    ASSERT_NE(execution, nullptr);

    // A whole storage entry: the slot reaches its descriptor by ChipTaskStorage's
    // layout, so a bare slot state would resolve outside itself.
    ChipTaskStorage outer{};
    TaskDescriptor &outer_task = outer.task;
    outer_task.task_id = TaskId::make_global(7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    ChipTaskSlotState &outer_slot = outer.slot;
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.graph_context = execution;
    // Only a shell the host qualified can stage its body's roots, so this is
    // what makes the root verdict below reachable at all.
    outer_slot.ed_flags = ED_FLAG_CANDIDATE;

    // The execution and in-graph task storage both occupy the outer heap tail after
    // required_heap.
    EXPECT_EQ(static_cast<void *>(execution), heap.execution());
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    // A shell released early stages its body's roots, and materialization is what
    // decides which roots it may stage. Task 0 is this body's root and is an
    // ordinary dispatchable task, so it qualifies; task 1 has a producer, so its
    // verdict is the recorded one instead, which this Definition leaves clear.
    EXPECT_NE(execution->task_at(0).slot.ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_EQ(execution->task_at(1).slot.ed_flags & ED_FLAG_CANDIDATE, 0);
    ChipTaskStorage &storage = execution->task_at(0);
    ASSERT_EQ(storage.payload.scalar_count, 1);
    ASSERT_EQ(storage.payload.tensor_count, 1);
    EXPECT_EQ(storage.payload.scalar_data()[0], 17U);
    EXPECT_EQ(execution->task_at(1).payload.scalar_data()[0], 18U);
    EXPECT_EQ(storage.payload.dump_metadata.dump_arg_mask, uint64_t{1} << 0);
    EXPECT_EQ(storage.payload.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::FLOAT32));
    EXPECT_EQ(execution->task_at(1).payload.dump_metadata.dump_arg_mask, uint64_t{1} << 1);
    EXPECT_EQ(execution->task_at(1).payload.dump_metadata.scalar_dtypes[0], static_cast<uint8_t>(DataType::INT32));

    graph_execution_mark_completed(*execution);
    execution->retired_tasks.store(2, std::memory_order_release);
    outer_task.task_id = TaskId::make_global(8);

    // Poison every field the rebuild is responsible for restoring. A replay that
    // preserved any of them would leave the poison observable.
    storage.task.kernel_id[0] = 314;
    storage.slot.active_mask = ActiveMask(3);
    storage.payload.scalar_data()[0] = 2718;
    execution->task_at(1).payload.scalar_data()[0] = 31415;
    storage.payload.tensor_data()[0].version = 1618;
    storage.slot.completed_subtasks.store(1, std::memory_order_relaxed);
    storage.payload.published_block_count.store(1, std::memory_order_relaxed);

    execution = heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(second_boundary.data()), 99);
    ASSERT_NE(execution, nullptr);
    outer_slot.graph_context = execution;
    // Same heap block: the rebuild lands on the bytes the previous execution left behind.
    EXPECT_EQ(static_cast<void *>(execution), heap.execution());

    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    EXPECT_EQ(storage.task.kernel_id[0], 42);
    EXPECT_EQ(storage.slot.active_mask.raw(), 1);
    EXPECT_EQ(storage.payload.scalar_data()[0], 99U);
    EXPECT_EQ(execution->task_at(1).payload.scalar_data()[0], 18U);
    EXPECT_EQ(storage.payload.tensor_data()[0].version, 0);
    EXPECT_EQ(storage.task.task_id, TaskId::make_in_graph(/*graph_task_id=*/8, /*in_graph_local_id=*/0));
    EXPECT_EQ(storage.task.packed_buffer_base, heap.base());
    EXPECT_EQ(storage.payload.tensor_data()[0].buffer.addr, reinterpret_cast<uint64_t>(second_boundary.data()));
    EXPECT_EQ(execution->task_at(1).payload.tensor_data()[0].buffer.addr, reinterpret_cast<uint64_t>(heap.base() + 16));
    EXPECT_EQ(storage.slot.completed_subtasks.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(storage.payload.published_block_count.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(storage.payload.dump_metadata.dump_arg_mask, uint64_t{1} << 0);
}

// Each term that withholds ED_FLAG_CANDIDATE from a body root, one per case.
// ResubmissionRebuildsFromDefinition covers the conjunction's true side; these
// are the three ways it comes out false, and each is load-bearing: a shell the
// host did not qualify never stages anything, a predicated task must reach the
// predicate test an early release returns before, and a DUMMY task has no
// dispatchable shape to index a per-shape early-dispatch queue with.
TEST(GraphExecutionReplay, RootStagingVerdictWithholdsCandidate) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;

    struct Case {
        const char *name;
        TestRoot root_variant;
        uint8_t shell_ed_flags;
    };
    const Case cases[] = {
        {"shell the host did not qualify", TestRoot::ORDINARY, 0},
        {"predicated root", TestRoot::PREDICATED, ED_FLAG_CANDIDATE},
        {"dummy root", TestRoot::DUMMY, ED_FLAG_CANDIDATE},
    };

    for (const Case &test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::array<uint8_t, 64> boundary{};
        const std::vector<std::byte> definition = make_test_definition(
            GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 1, test_case.root_variant
        );
        const TestDefinitionObject definition_object(definition);
        OuterHeap heap(definition, 0xAA);
        GraphExecution *execution =
            heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17);
        ASSERT_NE(execution, nullptr);

        ChipTaskStorage outer{};
        outer.task.task_id = TaskId::make_global(7);
        outer.task.packed_buffer_base = heap.base();
        outer.task.packed_buffer_end = heap.end();
        outer.slot.task_kind = TaskKind::GRAPH;
        outer.slot.graph_context = execution;
        outer.slot.ed_flags = test_case.shell_ed_flags;

        ASSERT_EQ(graph_execution_materialize_slice(outer.slot, *execution, 2), GraphMaterializeResult::PREPARED);
        EXPECT_EQ(execution->task_at(0).slot.ed_flags & ED_FLAG_CANDIDATE, 0);
    }
}

// ED_FLAG_CANDIDATE stopped being inert once materialization began replaying it
// onto a slot, so the image reader has to hold the conjunction that decides it
// rather than only rejecting unknown bits. None of these images is one
// graph_fill_definition can produce; each would otherwise reach dispatch, and
// the DUMMY one would index early_dispatch_queues[] one past its last shape.
TEST(GraphExecutionReplay, RejectsCandidateFlagWithoutItsConjunction) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;

    struct Case {
        const char *name;
        TestEdDefect defect;
    };
    const Case cases[] = {
        {"candidate with no producer", TestEdDefect::CANDIDATE_ON_ROOT},
        {"candidate carrying a predicate", TestEdDefect::CANDIDATE_WITH_PREDICATE},
        {"candidate with no dispatchable shape", TestEdDefect::CANDIDATE_ON_DUMMY},
    };

    for (const Case &test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::array<uint8_t, 64> boundary{};
        const std::vector<std::byte> definition = make_test_definition(
            GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), 1, TestRoot::ORDINARY, test_case.defect
        );
        const TestDefinitionObject definition_object(definition);
        OuterHeap heap(definition, 0xAA);
        EXPECT_EQ(
            heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr
        );
    }
}

// The same builder without a defect must still localize, so the test above
// rejects on the flag conjunction and not on something it broke in passing.
TEST(GraphExecutionReplay, AcceptsTheSameImageWithoutTheDefect) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    std::array<uint8_t, 64> boundary{};
    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    EXPECT_NE(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr);
}

// The boundary scalar pool is bounded by the Graph boundary contract
// (GRAPH_MAX_SCALAR_ARGS), not by a single task payload's MAX_SCALAR_ARGS —
// an in-graph task stages at most MAX_SCALAR_ARGS entries from it, but the pool itself
// may be wider.
TEST(GraphExecutionReplay, MaterializesBoundaryScalarPoolWiderThanTaskPayload) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    constexpr uint32_t POOL = MAX_SCALAR_ARGS + 3;
    static_assert(POOL <= GRAPH_MAX_SCALAR_ARGS);
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), POOL);
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    GraphExecution *execution =
        heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 21);
    ASSERT_NE(execution, nullptr);

    ChipTaskStorage outer{};
    TaskDescriptor &outer_task = outer.task;
    outer_task.task_id = TaskId::make_global(7);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    ChipTaskSlotState &outer_slot = outer.slot;
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.graph_context = execution;

    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);
    EXPECT_EQ(execution->task_storage[0].payload.scalar_data()[0], 21U);
}

TEST(GraphExecutionReplay, RejectsBoundaryScalarPoolBeyondContract) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x1234;
    constexpr uint32_t POOL = GRAPH_MAX_SCALAR_ARGS + 1;
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()), POOL);
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    EXPECT_EQ(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 21), nullptr);
}

TEST(GraphDefinitionObject, RejectsDefinitionBeyondRetainedBytes) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::array<uint8_t, 64> boundary{};
    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    ASSERT_GT(definition.size(), sizeof(GraphDefinition));
    const TestDefinitionObject definition_object(definition, sizeof(GraphDefinition));
    OuterHeap heap(definition);
    EXPECT_EQ(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr);
}

// The objects of a bind share one block, so a header and the image behind it are two
// separately written things. The header's Graph key is what says they belong
// together, and an object that fails that agreement must be refused rather than
// replayed — that mismatch is the whole failure mode of packing objects by offset.
TEST(GraphDefinitionObject, RejectsHeaderFramingAnotherGraph) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::array<uint8_t, 64> boundary{};
    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    ASSERT_NE(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr)
        << "the object localizes while its header and image agree";

    definition_object.reframe_full_key(GRAPH_KEY_VALUE + 1);
    EXPECT_EQ(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr);
}

// task_count is read off the wire and signed, so a corrupt Definition can present a
// non-positive one. Two layers refuse it -- graph_execution_localize's own gate and
// graph_execution_storage_layout behind it -- and this pins the contract rather than
// either implementation: whichever layer moves, a Definition with no tasks must not
// localize. It matters because every CSR section is fetched with `task_count + 1`, so a
// -1 that reached bind_graph_topology would ask for zero elements, get a live pointer,
// and read fanin_offsets one slot before the array.
TEST(GraphDefinitionObject, RejectsNonPositiveInGraphTaskCount) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::array<uint8_t, 64> boundary{};
    for (const int32_t task_count : {-1, 0}) {
        std::vector<std::byte> definition =
            make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
        reinterpret_cast<GraphDefinition *>(definition.data())->task_count = task_count;
        const TestDefinitionObject definition_object(definition);
        OuterHeap heap(definition);
        EXPECT_EQ(
            heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr
        ) << "task_count="
          << task_count;
    }
}

// An in-graph task's pool offsets are signed, which makes a negative one representable
// where the unsigned span tests used to reject it as a huge value. Signed, those tests go
// blind: a negative offset is below tensor_arg_count, and subtracting it *widens* the
// remaining span, so both `offset > count` and `count > total - offset` pass. Only the
// explicit negative test in bind_graph_topology rejects it, and it belongs there rather
// than in the per-slice materialize path, which runs after the pointer is bound.
TEST(GraphDefinitionObject, RejectsNegativeInGraphTaskArgOffset) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x4567;
    std::array<uint8_t, 64> boundary{};
    std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    {
        const TestDefinitionObject definition_object(definition);
        OuterHeap heap(definition);
        ASSERT_NE(
            heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr
        ) << "the unmodified Definition localizes";
    }

    auto *header = reinterpret_cast<GraphDefinition *>(definition.data());
    auto *tasks = reinterpret_cast<InGraphTaskDefinition *>(definition.data() + header->off_in_graph_tasks);
    tasks[1].tensor_offset = -1;
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition);
    EXPECT_EQ(heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17), nullptr);
}

TEST(GraphExecutionActivationState, ExternalReadySurvivesConcurrentLifecycleTransition) {
    constexpr int ITERATIONS = 1000;
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        GraphExecution execution{};
        std::thread materialize([&] {
            EXPECT_TRUE(graph_execution_transition(
                execution, GraphExecutionState::SUBMITTED, GraphExecutionState::MATERIALIZING
            ));
        });
        std::thread ready([&] {
            EXPECT_TRUE(graph_execution_signal_external_ready(execution));
        });
        materialize.join();
        ready.join();
        EXPECT_EQ(graph_execution_state(execution), GraphExecutionState::MATERIALIZING);
        EXPECT_TRUE(graph_execution_external_ready(execution));
    }
}

TEST(GraphExecutionActivationState, RetriesDoNotClearReadiness) {
    GraphExecution execution{};

    EXPECT_TRUE(graph_execution_signal_external_ready(execution));
    EXPECT_FALSE(graph_execution_signal_external_ready(execution));
    graph_execution_set_state(execution, GraphExecutionState::PREPARED);
    EXPECT_TRUE(graph_execution_external_ready(execution));
    EXPECT_TRUE(graph_execution_transition(execution, GraphExecutionState::PREPARED, GraphExecutionState::ACTIVE));
    EXPECT_FALSE(graph_execution_transition(execution, GraphExecutionState::PREPARED, GraphExecutionState::ACTIVE));
}

TEST(GraphExecutionActivationState, ExternalReadyBeforePrepareActivatesAtMeet) {
    SchedulerState scheduler{};
    GraphExecution execution{};
    ChipTaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.graph_context = &execution;
    execution.outer_slot = &outer_slot;

    EXPECT_EQ(scheduler.activate_graph_task(outer_slot), 0);
    EXPECT_EQ(graph_execution_state(execution), GraphExecutionState::SUBMITTED);
    EXPECT_TRUE(graph_execution_external_ready(execution));

    graph_execution_set_state(execution, GraphExecutionState::PREPARED);
    EXPECT_EQ(scheduler.activate_prepared_graph(execution), 0);
    EXPECT_EQ(graph_execution_state(execution), GraphExecutionState::ACTIVE);
    EXPECT_EQ(scheduler.activate_prepared_graph(execution), 0);
}

TEST(GraphExecutionActivationState, PrepareBeforeExternalReadyActivatesAtMeet) {
    SchedulerState scheduler{};
    GraphExecution execution{};
    ChipTaskSlotState outer_slot{};
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.graph_context = &execution;
    execution.outer_slot = &outer_slot;
    graph_execution_set_state(execution, GraphExecutionState::PREPARED);

    EXPECT_EQ(scheduler.activate_graph_task(outer_slot), 0);
    EXPECT_EQ(graph_execution_state(execution), GraphExecutionState::ACTIVE);
    EXPECT_TRUE(graph_execution_external_ready(execution));
    EXPECT_EQ(scheduler.activate_graph_task(outer_slot), 0);
}

TEST(GraphExecutionErrors, ReadyQueueOverflowHasTriageText) {
    EXPECT_STREQ(error_name(SIMPLER_ERROR_READY_QUEUE_OVERFLOW), "READY_QUEUE_OVERFLOW");
    EXPECT_STRNE(error_desc(SIMPLER_ERROR_READY_QUEUE_OVERFLOW), "");
    EXPECT_STRNE(error_hint(SIMPLER_ERROR_READY_QUEUE_OVERFLOW), "");
}

TEST(GraphExecutionErrors, GraphReadyQueueOverflowIsReported) {
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    ChipReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.graph_ready_queue.slots = queue_slots;
    scheduler.graph_ready_queue.capacity = 2;
    scheduler.graph_ready_queue.mask = 1;
    scheduler.graph_ready_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.graph_ready_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    ChipTaskSlotState graph_slots[3]{};
    for (ChipTaskSlotState &slot : graph_slots) {
        slot.task_kind = TaskKind::GRAPH;
    }

    scheduler.push_ready_routed(&graph_slots[0]);
    scheduler.push_ready_routed(&graph_slots[1]);
    scheduler.push_ready_routed(&graph_slots[2]);

    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(GraphExecutionErrors, GraphPrepareQueueOverflowIsReported) {
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    ChipReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.slots = queue_slots;
    scheduler.graph_prepare_queue.capacity = 2;
    scheduler.graph_prepare_queue.mask = 1;
    scheduler.graph_prepare_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    ChipTaskSlotState graph_slots[3]{};

    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[0], 10, 3));
    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[1], 11, 3));
    EXPECT_FALSE(scheduler.push_graph_prepare(&graph_slots[2], 12, 3));

    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(std::memory_order_acquire), 3);
    EXPECT_EQ(header.sched_error_bitmap.load(std::memory_order_acquire), 1U << 3);
}

TEST(GraphExecutionErrors, InvalidInGraphTaskCompletionIsReported) {
    SchedulerState scheduler{};
    ChipTaskSlotState slot{};
    // Membership without a usable execution: graph_context names one, but neither its
    // definition nor its in-graph task array was ever bound.
    GraphExecution execution{};
    slot.graph_context = &execution;

    const SchedulerState::TaskCompletionOutcome outcome = scheduler.complete_task(slot);

    EXPECT_EQ(outcome.error_code, SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(outcome.stream_tasks_completed, 0);
}

TEST(GraphExecutionProgress, InGraphTaskResolutionIsNotAHostCompletion) {
    SchedulerState scheduler{};
    GraphDefinition definition{};
    ChipTaskStorage task{};
    std::atomic<ChipTaskState> states[1]{};
    GraphExecution execution{};
    execution.definition = &definition;
    execution.tasks = &task;
    execution.task_storage = &task;
    execution.task_states = states;
    execution.task_count = 1;
    execution.remaining_tasks.store(1, std::memory_order_relaxed);
    graph_execution_set_state(execution, GraphExecutionState::ACTIVE, std::memory_order_relaxed);
    task.slot.graph_context = &execution;
    task.slot.in_graph_local_id = 0;

    AsyncWaitList wait_list{};
    wait_list.entries[0].slot_state = &task.slot;
    wait_list.entries[0].task_token = TaskId::make_global(1);
    wait_list.entries[0].normal_done = true;
    wait_list.count = 1;

    const AsyncPollResult result = wait_list.poll_and_complete<false>(nullptr, &scheduler);

    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(result.resolved, 1);
    EXPECT_EQ(result.completed, 0);
}

// Device-side execution storage is not guaranteed to be zero-initialized.
// Localization plus materialize must produce a valid execution from arbitrary
// initial bytes.
TEST(GraphExecutionMaterialize, DirtyStorageYieldsValidExecution) {
    constexpr uint64_t GRAPH_KEY_VALUE = 0x9753;
    std::array<uint8_t, 64> boundary{};

    const std::vector<std::byte> definition =
        make_test_definition(GRAPH_KEY_VALUE, reinterpret_cast<uint64_t>(boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    GraphExecution *execution =
        heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17);
    ASSERT_NE(execution, nullptr);

    ChipTaskStorage outer{};
    TaskDescriptor &outer_task = outer.task;
    outer_task.task_id = TaskId::make_global(5);
    outer_task.packed_buffer_base = heap.base();
    outer_task.packed_buffer_end = heap.end();
    ChipTaskSlotState &outer_slot = outer.slot;
    outer_slot.task_kind = TaskKind::GRAPH;
    outer_slot.graph_context = execution;

    EXPECT_EQ(static_cast<void *>(execution), heap.execution());
    EXPECT_EQ(graph_execution_materialize_slice(outer_slot, *execution, 2), GraphMaterializeResult::PREPARED);

    // Every observable scheduling field must be a materialize-written value,
    // not the 0xAA fill: state machine, counters and atomics all start from
    // values only the device side wrote.
    for (int32_t i = 0; i < execution->task_count; ++i) {
        const ChipTaskStorage &storage = execution->task_at(i);
        ASSERT_EQ(execution->task_states[i].load(std::memory_order_relaxed), CHIP_TASK_PENDING);
        ASSERT_EQ(storage.slot.task_kind, TaskKind::KERNEL);
        ASSERT_EQ(storage.slot.completed_subtasks.load(std::memory_order_relaxed), 0);
        ASSERT_EQ(storage.payload.published_block_count.load(std::memory_order_relaxed), 0);
        ASSERT_EQ(storage.payload.tensor_count, 1);
        ASSERT_EQ(storage.payload.scalar_count, 1);
        // A tensor address of 0xAAAAAAAAAAAAAAAA would mean the fill leaked
        // through into a field the scheduler later dereferences.
        ASSERT_NE(storage.payload.tensor_data()[0].buffer.addr, 0xAAAAAAAAAAAAAAAAULL);
        // make_test_definition assigns in-graph task i the heap offset 64*i, so the
        // packed window starts at outer_base + 64*i, not at outer_base.
        ASSERT_EQ(storage.task.packed_buffer_base, static_cast<void *>(heap.base() + static_cast<size_t>(i) * 64));
    }
    EXPECT_EQ(execution->materialized_tasks, execution->task_count);
    EXPECT_EQ(execution->consumed_tensor_args, 2U);

    // Localize and materialize read the boundary and write in-graph task arguments; neither may
    // touch the tensor region past the packed boundary values.
    const std::byte *tail = heap.boundary_tail(1);
    for (size_t i = 0; i < heap.boundary_tail_bytes(1); ++i) {
        ASSERT_EQ(tail[i], std::byte{0}) << "byte " << i << " past the packed boundary";
    }
}
