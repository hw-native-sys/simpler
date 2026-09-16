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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "host_build_graph/task_id.h"
#include "host_build_graph/runtime_types.h"
#include "tensor.h"

inline constexpr int32_t MAX_IN_GRAPH_TASKS = 1024;
static_assert(
    MAX_IN_GRAPH_TASKS <= (1 << TaskId::IN_GRAPH_LOCAL_ID_BITS),
    "an in-graph local id must fit the low field of an IN_GRAPH task id"
);
// A body's producers precede their consumers, so a fanin CSR row holds at most
// task_count - 1 entries, and both of the slot's row cursors index one of them.
// An in-graph row has no cap of its own — unlike a GLOBAL task's inline row,
// which append_fanin_or_fail holds to CHIP_MAX_FANIN — so this is what bounds
// them, and a cursor too narrow for it would report a row scanned that was not.
static_assert(MAX_IN_GRAPH_TASKS - 1 < 0xFFFF, "a fanin CSR row index must fit ChipTaskSlotState::wake_scan_cursor");
static_assert(
    MAX_IN_GRAPH_TASKS - 1 < 0xFFFF && CHIP_MAX_FANIN - 1 < 0xFFFF,
    "a fanin row index from either cohort must fit ChipTaskSlotState::ed_publish_scan_cursor"
);
inline constexpr int32_t GRAPH_MATERIALIZE_SLICE_TASKS = 4;

enum class GraphTensorSourceKind : uint8_t {
    BOUNDARY_EXACT = 0,
    BOUNDARY_VIEW = 1,
    INTERNAL = 2,
    OWN_OUTPUT = 3,
};

// Wire representation of simpler::hbg::Tensor. simpler::hbg::Tensor itself is a host/runtime C++ type with
// 64-byte alignment and helper methods; placing it inside vector<std::byte>
// would not guarantee that alignment. Keep the boundary image C-compatible and
// copy only semantic fields into this naturally 8-byte-aligned POD.
struct GraphTensor {
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint64_t owner_task_id;
    uint64_t start_offset;
    uint64_t extent_elem;
    int32_t version;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    uint8_t ndims;
    uint8_t dtype;
    uint8_t manual_dep;
    uint8_t is_contiguous;
    uint8_t address_space;
    uint8_t reserved[3];
};

// Definition records are copied across the host-device boundary. Keep them
// pointer-free, fixed-width and position-independent: every reference is an
// offset from its owning header.
//
// How one recorded tensor argument is rebuilt for an execution: which object of
// that execution supplies its storage, and where inside that object it starts.
// `source_kind` decides what the other two fields mean, because each kind adds its
// offset to a different field of the rebound tensor (graph_rebind_tensor):
//
//   source_kind     source_index indexes       packed_offset adds to     unit
//   BOUNDARY_EXACT  the boundary tensors       nothing (must be zero)    --
//   BOUNDARY_VIEW   the boundary tensors       rebound.start_offset      elements
//   INTERNAL        the in-graph tasks         rebound.buffer_addr       bytes
//   OWN_OUTPUT      the consuming task itself  rebound.buffer_addr       bytes
//
// The unit follows the field the offset lands in rather than being a choice. A
// BOUNDARY kind keeps the boundary tensor's whole buffer, since that buffer is what
// graph_tensor_wire_valid bounds the view against, so its offset has nowhere to go
// but start_offset — which simpler::hbg::Tensor counts in elements. An INTERNAL
// tensor instead owns the slice it names, so its offset moves the buffer itself.
// An element offset is meaningful only while the consumer and the boundary tensor
// share a dtype.
//
// OWN_OUTPUT carries a source_index equal to the consuming task's own index, which
// replay checks rather than reads.
struct GraphTensorSourceRef {
    uint8_t source_kind;
    uint8_t reserved;
    uint16_t source_index;
    uint32_t reserved2;
    uint64_t packed_offset;
};

// Where one in-graph task scalar slot takes its value from: the Definition's own
// scalars[] entry, or the boundary parameter named by boundary_index(). It is the wire
// form of the two things recording knows about a slot -- whether it inherits, and which
// parameter it inherits -- so inherited() is the same predicate as Arg::scalar_inherited.
//
// Only a parameter of the replaying Graph's boundary can be refreshed; the index reaches
// that boundary and nothing else, and means nothing while inherited() is false. The
// fields are private so the pair can only be set together, through a factory that decides
// both: an entry carrying an index while claiming not to inherit, or the reverse, cannot
// be spelled. What the index means is still a claim about a boundary this entry cannot
// see, so the readers bound it against the boundary they do have.
class GraphScalarInheritance {
public:
    // The image's scalar section is allocated as an array, so a default-constructible
    // slot is required; the factories below are what a caller fills one with.
    GraphScalarInheritance() = default;

    static GraphScalarInheritance self_value() { return {0, false}; }
    static GraphScalarInheritance from_boundary(uint16_t index) { return {index, true}; }

    bool inherited() const { return inherited_ != 0; }
    uint16_t boundary_index() const { return boundary_index_; }

private:
    GraphScalarInheritance(uint16_t index, bool inherits) :
        boundary_index_(index),
        inherited_(inherits ? 1 : 0) {}

    // One access level for every field, which is what keeps this standard-layout and so
    // safe to memcpy to the device. inherited_ is a uint16_t rather than a bool so the
    // two fields fill the size alignof(uint16_t) rounds this type up to: there is no
    // padding byte, and so no indeterminate byte in the image's scalar section, which the
    // static_assert below pins the size of because that is what the section indexes by.
    uint16_t boundary_index_;
    uint16_t inherited_;
};

// Wire representation of an in-graph task's dispatch predicate. The operand's absolute GM
// address is not replay-invariant, so the Definition names the tensor the
// operand element sits in plus its element offset within that tensor;
// materialize rebinds the tensor for the execution and resolves the pair into
// the address the scheduler reads at the dispatch point.
struct GraphPredicate {
    GraphTensor operand;
    GraphTensorSourceRef operand_source;
    // Element index into the rebound operand tensor, added to its start_offset.
    // Fixed at record time: a Graph with a variable simpler::hbg::Tensor shape is rejected
    // before recording, so the operand's strides cannot change across replays.
    uint64_t elem_offset;
    int64_t target;
    uint8_t elem_size;
    uint8_t op;
    uint8_t reserved[6];
};

struct InGraphTaskDefinition {
    int32_t kernel_id[SUBTASK_SLOT_COUNT];
    uint8_t active_mask;
    uint8_t task_attrs;
    // Early-dispatch verdicts (ED_FLAG_CANDIDATE / ED_FLAG_TRACKED), decided once
    // per recorded shape rather than per execution: a body's structure is fixed by
    // its Definition, so every execution of it qualifies the same tasks. Carries
    // the same meaning as ChipTaskSlotState::ed_flags, against the Definition's
    // fanin CSR instead of a payload fanin region.
    uint8_t ed_flags;
    uint8_t reserved;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    // One-based index into the Definition's predicate array; 0 means the task
    // carries no dispatch predicate. Biased so that a zeroed InGraphTaskDefinition
    // is a valid predicate-free task. Predicated tasks are rare, so the
    // predicates live in their own array rather than inline.
    uint16_t predicate_slot;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    // Element indices into the Definition's argument pools, not byte offsets: each
    // pairs with the count above it to form [offset, offset + count), so it carries
    // that count's type.
    int32_t tensor_offset;
    int32_t scalar_offset;
    ArgsDumpTaskMetadata dump_metadata;
};

struct GraphBoundarySignature {
    uint64_t buffer_size;
    uint32_t shapes[MAX_TENSOR_DIMS];
    uint32_t strides[MAX_TENSOR_DIMS];
    uint16_t alias_rep;
    uint8_t ndims;
    uint8_t dtype;
    uint8_t tag;
    uint8_t manual_dep;
    uint8_t is_contiguous;
    uint8_t reserved;
};

inline constexpr uint64_t GRAPH_DEFINITION_OBJECT_MAGIC = 0x4752415048455844ULL;

// Every Definition object is [GraphDefinitionHeader][Definition image], aligned to
// this and padded to a multiple of it. A Definition section is reached as an offset
// from the image base, and the widest one may ask for max_align_t, so this is what
// an image base has to carry; the header in front preserves it because its own size
// is a multiple of it. Both the recorder that claims room for an object and the
// upload that packs them share the value.
inline constexpr size_t GRAPH_DEFINITION_OBJECT_ALIGN = alignof(std::max_align_t);

// Header prefixing each device-resident Definition object. The definition buffer
// uploaded by the host is [GraphDefinitionHeader][GraphDefinition image], and these
// three fields are the framing the device checks before it reads a section offset
// out of the image: the object is one of ours, it is the size the upload recorded,
// and it holds the Graph the outer task was bound to.
//
// alignas, not a pad field: the image starts immediately behind the header and its
// sections are offsets from that base, so the header's own size has to be a multiple
// of the object alignment.
struct alignas(GRAPH_DEFINITION_OBJECT_ALIGN) GraphDefinitionHeader {
    uint64_t magic;
    uint64_t full_key;
    uint32_t definition_bytes;
};

static_assert(std::is_trivially_copyable_v<GraphDefinitionHeader>);
static_assert(std::is_standard_layout_v<GraphDefinitionHeader>);
static_assert(sizeof(GraphDefinitionHeader) % GRAPH_DEFINITION_OBJECT_ALIGN == 0);

struct GraphDefinition {
    uint64_t full_key;
    uint64_t required_heap;
    // The header splits by range, not by name. Everything that counts *things* is
    // signed: each is capped by MAX_IN_GRAPH_TASKS times a per-task constant, so the
    // largest of them (edge_count, at 1024 x 1023) still clears INT32_MAX by three
    // orders of magnitude, and a signed count makes a corrupt wire value testable
    // with `< 0` instead of turning it into a huge index. Everything that counts
    // *bytes* stays unsigned because graph_layout_section dimensions the image
    // against UINT32_MAX: total_bytes and the off_ fields below reach 4 GB by design,
    // and halving that range would be a functional change.
    uint32_t total_bytes;
    int32_t task_count;
    int32_t edge_count;
    int32_t root_count;
    int32_t boundary_count;
    int32_t boundary_scalar_count;
    int32_t tensor_arg_count;
    int32_t scalar_arg_count;
    int32_t predicate_count;
    // Bytes the GraphExecution header, in-graph task array and in-graph task
    // argument pools need in the outer Graph task's heap tail. Invocation
    // boundaries live in the outer task payload's compact argument-pool regions
    // instead.
    uint32_t execution_storage_bytes;
    uint32_t off_fanout_offsets;
    uint32_t off_fanout_indices;
    uint32_t off_fanin_offsets;
    uint32_t off_fanin_indices;
    uint32_t off_root_indices;
    uint32_t off_in_graph_task_offsets;
    uint32_t off_in_graph_tasks;
    uint32_t off_tensors;
    uint32_t off_tensor_sources;
    uint32_t off_scalars;
    uint32_t off_scalar_inheritance;
    uint32_t off_boundary_signatures;
    uint32_t off_predicates;
};

static_assert(std::is_trivially_copyable_v<GraphTensorSourceRef>);
static_assert(std::is_standard_layout_v<GraphTensorSourceRef>);
static_assert(std::is_trivially_copyable_v<GraphTensor>);
static_assert(std::is_standard_layout_v<GraphTensor>);
static_assert(std::is_trivially_copyable_v<GraphScalarInheritance>);
static_assert(std::is_standard_layout_v<GraphScalarInheritance>);
static_assert(sizeof(GraphScalarInheritance) == 4, "the image's scalar section assumes this layout");
static_assert(std::is_trivially_copyable_v<InGraphTaskDefinition>);
static_assert(std::is_standard_layout_v<InGraphTaskDefinition>);
// graph_fill_definition assigns this struct field by field, so its interior padding
// is the one part of the in-graph task section no writer reaches. Nothing reads it
// either, but pinning the size makes a new field's padding cost visible in the diff
// that adds it rather than silently.
static_assert(sizeof(InGraphTaskDefinition) == 80, "an InGraphTaskDefinition field changed the wire layout");
static_assert(std::is_trivially_copyable_v<GraphPredicate>);
static_assert(std::is_standard_layout_v<GraphPredicate>);
static_assert(std::is_trivially_copyable_v<GraphBoundarySignature>);
static_assert(std::is_standard_layout_v<GraphBoundarySignature>);
static_assert(std::is_trivially_copyable_v<GraphDefinition>);
static_assert(std::is_standard_layout_v<GraphDefinition>);

// Section starts are offsets from the image base, so a section is correctly
// aligned only if the buffer holding the image is. The builder writes each
// section through a typed pointer into a std::vector<std::byte>, whose data() is
// aligned for any type with fundamental alignment and no further — so a section
// type that asked for more would make every one of those stores undefined, with
// no diagnostic.
static_assert(
    alignof(InGraphTaskDefinition) <= alignof(std::max_align_t) && alignof(GraphTensor) <= alignof(std::max_align_t) &&
        alignof(GraphTensorSourceRef) <= alignof(std::max_align_t) &&
        alignof(GraphScalarInheritance) <= alignof(std::max_align_t) &&
        alignof(GraphBoundarySignature) <= alignof(std::max_align_t) &&
        alignof(GraphPredicate) <= alignof(std::max_align_t),
    "a Definition section type must not be over-aligned: its storage is a byte vector"
);

inline GraphTensor graph_tensor_pack(const simpler::hbg::Tensor &tensor) {
    GraphTensor packed{};
    packed.buffer_addr = tensor.buffer.addr;
    packed.buffer_size = tensor.buffer.size;
    packed.owner_task_id = tensor.owner_task_id.raw;
    packed.start_offset = tensor.start_offset;
    packed.extent_elem = tensor.extent_elem_cache;
    packed.version = tensor.version;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        packed.shapes[i] = tensor.shapes[i];
        packed.strides[i] = tensor.strides[i];
    }
    packed.ndims = static_cast<uint8_t>(tensor.ndims);
    packed.dtype = static_cast<uint8_t>(tensor.dtype);
    packed.manual_dep = tensor.manual_dep ? 1 : 0;
    packed.is_contiguous = tensor.is_contiguous ? 1 : 0;
    packed.address_space = static_cast<uint8_t>(tensor.address_space);
    return packed;
}

inline void graph_tensor_unpack(const GraphTensor &packed, simpler::hbg::Tensor *tensor) {
    tensor->buffer = PTOBufferHandle{packed.buffer_addr, packed.buffer_size};
    tensor->owner_task_id = TaskId{packed.owner_task_id};
    tensor->start_offset = packed.start_offset;
    tensor->extent_elem_cache = packed.extent_elem;
    tensor->version = packed.version;
    tensor->ndims = packed.ndims;
    tensor->dtype = static_cast<DataType>(packed.dtype);
    tensor->manual_dep = packed.manual_dep != 0;
    tensor->is_contiguous = packed.is_contiguous != 0;
    tensor->address_space = static_cast<AddressSpace>(packed.address_space);
    for (uint32_t i = 0; i < MAX_TENSOR_DIMS; ++i) {
        tensor->shapes[i] = packed.shapes[i];
        tensor->strides[i] = packed.strides[i];
    }
    for (uint8_t &byte : tensor->_pad_cl2)
        byte = 0;
}

inline bool graph_tensor_wire_valid(const GraphTensor &tensor) {
    if (tensor.buffer_addr == 0 || tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS ||
        tensor.dtype >= static_cast<uint8_t>(DataType::DATA_TYPE_NUM) || tensor.manual_dep > 1 ||
        tensor.is_contiguous > 1 || tensor.address_space > 1) {
        return false;
    }

    uint64_t extent = 1;
    uint64_t expected_stride = 1;
    bool contiguous = true;
    for (int32_t i = static_cast<int32_t>(tensor.ndims) - 1; i >= 0; --i) {
        const uint64_t shape = tensor.shapes[i];
        const uint64_t stride = tensor.strides[i];
        if (shape == 0 || stride == 0) return false;
        contiguous &= stride == expected_stride;
        if (shape - 1 > (UINT64_MAX - extent) / stride || expected_stride > UINT64_MAX / shape) return false;
        extent += (shape - 1) * stride;
        expected_stride *= shape;
    }
    if (extent != tensor.extent_elem || contiguous != (tensor.is_contiguous != 0)) return false;

    const uint64_t element_size = get_element_size(static_cast<DataType>(tensor.dtype));
    const uint64_t buffer_elements = tensor.buffer_size / element_size;
    return tensor.start_offset <= buffer_elements && tensor.extent_elem <= buffer_elements - tensor.start_offset;
}

// A section's length is the same int32 every counting field of the header carries.
// A negative one is rejected outright rather than left to wrap through the size_t
// comparison, so a corrupt header cannot widen a section by either route.
template <typename T>
inline const T *graph_definition_array(const GraphDefinition &definition, uint32_t offset, int32_t count) {
    if (count < 0 || offset == 0 || offset > definition.total_bytes || offset % alignof(T) != 0) return nullptr;
    const size_t remaining = static_cast<size_t>(definition.total_bytes - offset);
    if (static_cast<size_t>(count) > remaining / sizeof(T)) return nullptr;
    return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(&definition) + offset);
}

template <typename T>
inline const T *graph_definition_ptr(const GraphDefinition &definition, uint32_t offset) {
    return graph_definition_array<T>(definition, offset, 1);
}

enum class GraphExecutionState : uint8_t {
    SUBMITTED = 0,
    MATERIALIZING = 1,
    PREPARED = 2,
    ACTIVE = 3,
    COMPLETED = 4,
};

enum class GraphMaterializeResult : uint8_t {
    INVALID = 0,
    BUSY = 1,
    PENDING = 2,
    PREPARED = 3,
};

inline constexpr uint8_t GRAPH_EXECUTION_STATE_MASK = 0x7;
inline constexpr uint8_t GRAPH_EXECUTION_EXTERNAL_READY = 0x8;

struct GraphExecution {
    // The low bits hold GraphExecutionState. EXTERNAL_READY shares this byte so
    // dependency readiness can arrive before materialization without a separate
    // per-submission gate object.
    std::atomic<uint8_t> state{static_cast<uint8_t>(GraphExecutionState::SUBMITTED)};
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_tasks{0};
    std::atomic<int32_t> retired_tasks{0};
    // Incremental activation: tasks in [0, published_tasks) are fully
    // materialized and registered, so a route pass may consider them. route_cursor
    // is the next such task index a route pass will claim; roots below it have
    // been pushed to the ready queue exactly once. Both advance monotonically and
    // reset per (re)submission.
    std::atomic<int32_t> published_tasks{0};
    std::atomic<int32_t> route_cursor{0};
    int32_t task_count{0};
    int32_t materialized_tasks{0};
    int32_t constructed_tasks{0};
    int32_t consumed_tensor_args{0};
    ChipTaskSlotState *outer_slot{nullptr};
    ChipTaskStorage *tasks{nullptr};
    ChipTaskStorage *task_storage{nullptr};
    // Polling-progress state, one ChipTaskState byte per in-graph task, in the
    // storage tail. Carries the same PENDING -> PUBLISHED -> COMPLETED meaning
    // as the shared-memory task_states array a GLOBAL task uses, against
    // in-graph local ids instead of task-table slots, so both cohorts answer
    // readiness the same way.
    //
    // A byte array of its own rather than a field of ChipTaskStorage, for the
    // reason the top-level array gives: a fanin scan reads many producers'
    // states at once, which a couple of cache lines answer here and would take
    // one line per producer inside the ChipTaskStorage stride. It is also
    // read-mostly — each byte is written once per execution at completion —
    // so concurrent scanners share those lines rather than contend for them.
    std::atomic<ChipTaskState> *task_states{nullptr};
    // This execution's task argument pools, in the storage tail past task_storage.
    // Every task payload's tensor and scalar deltas point here; its pool position is
    // the Definition's tensor_offset / scalar_offset.
    simpler::hbg::Tensor *task_tensor_pool{nullptr};
    uint64_t *task_scalar_pool{nullptr};
    const GraphDefinition *definition{nullptr};
    const int32_t *fanin_offsets{nullptr};
    const uint16_t *fanin_indices{nullptr};
    const GraphTensor *boundary_tensors{nullptr};
    int32_t boundary_tensor_count{0};
    const uint64_t *boundary_scalars{nullptr};
    int32_t boundary_scalar_count{0};

    ChipTaskStorage &task_at(int32_t index) const { return task_storage[index]; }

    // Readiness accessors, named and ordered as SharedMemoryTaskHeader's so a
    // reader of one cohort reads the other the same way. The byte only ever
    // advances, so an index a scan has already cleared stays cleared.
    bool is_completed(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_COMPLETED;
    }

    void store_completed(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_COMPLETED, order);
    }

    bool is_published(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_PUBLISHED;
    }

    void store_published(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_PUBLISHED, order);
    }

    void reset_task_state(int32_t index) const {
        task_states[index].store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    }
};

static_assert(std::is_trivially_destructible_v<ChipTaskStorage>);
// The tensor pool starts right after the in-graph task array, and the scalar pool starts
// after a whole number of ChipTensors.
static_assert(
    alignof(ChipTaskStorage) % alignof(simpler::hbg::Tensor) == 0,
    "an in-graph task entry must be at least simpler::hbg::Tensor-aligned: the tensor pool follows the "
    "in-graph task array"
);
static_assert(
    sizeof(simpler::hbg::Tensor) % alignof(uint64_t) == 0, "the tensor stride must keep the scalar pool aligned"
);
static_assert(std::is_trivially_destructible_v<GraphExecution>);
// The whole storage is aligned for its widest member, so one base check covers the
// header as well as the in-graph task array that follows it.
static_assert(
    alignof(ChipTaskStorage) % alignof(GraphExecution) == 0,
    "the in-graph task array's alignment must subsume the execution header's"
);
static_assert(sizeof(GraphTensor) <= sizeof(simpler::hbg::Tensor));

inline constexpr size_t graph_boundary_tensor_pool_slots(uint32_t tensor_count) {
    const size_t bytes = static_cast<size_t>(tensor_count) * sizeof(GraphTensor);
    return (bytes + sizeof(simpler::hbg::Tensor) - 1) / sizeof(simpler::hbg::Tensor);
}

// The outer GRAPH task's heap tail occupies
// [GraphExecution][ChipTaskStorage x task_count][simpler::hbg::Tensor x tensor_arg_count]
// [uint64_t x scalar_arg_count].
//
// The last two regions are the in-graph task payloads' argument pools, indexed by the
// Definition's own tensor_offset / scalar_offset — which is why the Definition's
// arg-table counts size them rather than a per-task sum: in-graph task i's arguments
// occupy [offset, offset + count) in both the table and the pool. There is no fanin
// region: an in-graph task's dependencies live in the Definition's fanin CSR, so its
// fanin_count stays 0 and its fanin delta unbound.
struct GraphExecutionStorageLayout {
    size_t tasks_offset;
    size_t tensors_offset;
    size_t scalars_offset;
    size_t states_offset;
    size_t total_bytes;
};

inline bool graph_execution_storage_layout(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, GraphExecutionStorageLayout *out
) {
    if (out == nullptr || task_count <= 0 || task_count > MAX_IN_GRAPH_TASKS || tensor_arg_count < 0 ||
        scalar_arg_count < 0) {
        return false;
    }
    constexpr size_t ALIGNMENT = alignof(ChipTaskStorage);
    out->tasks_offset = (sizeof(GraphExecution) + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    out->tensors_offset = out->tasks_offset + static_cast<size_t>(task_count) * sizeof(ChipTaskStorage);
    out->scalars_offset = out->tensors_offset + static_cast<size_t>(tensor_arg_count) * sizeof(simpler::hbg::Tensor);
    // The state array is last because it is the one region with no alignment of
    // its own: a byte needs none, so appending it disturbs no other section's
    // offset. Every other region is entered through a typed pointer whose
    // alignment the base already guarantees.
    out->states_offset = out->scalars_offset + static_cast<size_t>(scalar_arg_count) * sizeof(uint64_t);
    out->total_bytes = out->states_offset + static_cast<size_t>(task_count) * sizeof(std::atomic<ChipTaskState>);
    return true;
}

inline bool graph_execution_storage_bytes(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, size_t *storage_bytes
) {
    GraphExecutionStorageLayout layout{};
    if (storage_bytes == nullptr ||
        !graph_execution_storage_layout(task_count, tensor_arg_count, scalar_arg_count, &layout)) {
        return false;
    }
    *storage_bytes = layout.total_bytes;
    return true;
}

GraphExecution *graph_execution_localize(ChipTaskSlotState &outer_slot);
GraphMaterializeResult graph_execution_materialize_slice(
    ChipTaskSlotState &outer_slot, GraphExecution &execution, int32_t max_tasks, int32_t *tasks_materialized = nullptr
);

// An outer GRAPH slot's graph_context holds the shared Definition's device address
// until graph_execution_localize replaces it with the execution, so this cast is only
// valid after that call. What makes it safe is the boot sequence, not this slot: every
// AICPU thread localizes a disjoint slice of the task window in classify_partition and
// all of them barrier before runtime_init_ready_ is published, so no dispatch — and
// therefore no caller of this function — observes an unlocalized GRAPH slot. A slot
// whose localization failed carries nullptr.
inline GraphExecution *graph_execution_from_outer_slot(ChipTaskSlotState &slot) {
    return slot.task_kind == TaskKind::GRAPH ? static_cast<GraphExecution *>(slot.graph_context) : nullptr;
}

inline GraphExecutionState
graph_execution_state(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return static_cast<GraphExecutionState>(execution.state.load(order) & GRAPH_EXECUTION_STATE_MASK);
}

inline bool
graph_execution_external_ready(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return (execution.state.load(order) & GRAPH_EXECUTION_EXTERNAL_READY) != 0;
}

inline void graph_execution_set_state(
    GraphExecution &execution, GraphExecutionState next, std::memory_order order = std::memory_order_release
) {
    uint8_t observed = execution.state.load(std::memory_order_relaxed);
    // Masked, so a state added past GRAPH_EXECUTION_STATE_MASK cannot reach the
    // readiness bit sharing this byte.
    const uint8_t next_state = static_cast<uint8_t>(static_cast<uint8_t>(next) & GRAPH_EXECUTION_STATE_MASK);
    while (!execution.state.compare_exchange_weak(
        observed, static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | next_state), order,
        std::memory_order_relaxed
    )) {}
}

inline bool graph_execution_transition(
    GraphExecution &execution, GraphExecutionState expected_state, GraphExecutionState next_state
) {
    uint8_t observed = execution.state.load(std::memory_order_acquire);
    const uint8_t desired_state = static_cast<uint8_t>(static_cast<uint8_t>(next_state) & GRAPH_EXECUTION_STATE_MASK);
    while ((observed & GRAPH_EXECUTION_STATE_MASK) == static_cast<uint8_t>(expected_state)) {
        const uint8_t desired = static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | desired_state);
        if (execution.state.compare_exchange_weak(
                observed, desired, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return true;
        }
    }
    return false;
}

inline bool graph_execution_signal_external_ready(GraphExecution &execution) {
    return (execution.state.fetch_or(GRAPH_EXECUTION_EXTERNAL_READY, std::memory_order_acq_rel) &
            GRAPH_EXECUTION_EXTERNAL_READY) == 0;
}

inline bool graph_execution_complete_in_graph_task(GraphExecution &execution) {
    return execution.remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void graph_execution_mark_completed(GraphExecution &execution) {
    graph_execution_set_state(execution, GraphExecutionState::COMPLETED);
}

inline void graph_execution_retire_in_graph_task(GraphExecution &execution) {
    execution.retired_tasks.fetch_add(1, std::memory_order_release);
}
