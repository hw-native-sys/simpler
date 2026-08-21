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
#include <type_traits>

#include "pto_runtime2_types.h"
#include "tensor.h"

inline constexpr uint32_t GRAPH_MAX_NODES = 1024;
inline constexpr int32_t GRAPH_MATERIALIZE_SLICE_NODES = 4;

enum class GraphTensorSource : uint8_t {
    BOUNDARY_EXACT = 0,
    BOUNDARY_VIEW = 1,
    INTERNAL = 2,
    OWN_OUTPUT = 3,
};

// Wire representation of ChipTensor. ChipTensor itself is a host/runtime C++ type with
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
struct GraphTensorSourceRef {
    uint8_t source;
    uint8_t reserved;
    uint16_t source_index;
    uint32_t reserved2;
    uint64_t packed_offset;
};

enum class GraphScalarSource : uint8_t {
    STATIC_VALUE = 0,
    BOUNDARY = 1,
};

struct GraphScalarSourceRef {
    uint16_t source_index;
    uint8_t source;
    uint8_t reserved;
};

// Wire representation of a node's dispatch predicate. The operand's absolute GM
// address is not replay-invariant, so the Definition names the tensor the
// operand element sits in plus its element offset within that tensor;
// materialize rebinds the tensor for the execution and resolves the pair into
// the address the scheduler reads at the dispatch point.
struct GraphPredicate {
    GraphTensor operand;
    GraphTensorSourceRef operand_source;
    // Element index into the rebound operand tensor, added to its start_offset.
    // Fixed at record time: a Graph with a variable ChipTensor shape is rejected
    // before recording, so the operand's strides cannot change across replays.
    uint64_t elem_offset;
    int64_t target;
    uint8_t elem_size;
    uint8_t op;
    uint8_t reserved[6];
};

struct GraphNodeDefinition {
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];
    uint8_t active_mask;
    uint8_t task_attrs;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    // One-based index into the Definition's predicate array; 0 means the node
    // carries no dispatch predicate. Biased so that a zeroed GraphNodeDefinition
    // is a valid predicate-free node. Predicated nodes are rare, so the
    // predicates live in their own array rather than inline.
    uint16_t predicate_slot;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    uint32_t tensor_offset;
    uint32_t scalar_offset;
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

// Header prefixing each device-resident Definition object. The definition
// buffer uploaded by the host is [GraphDefinitionHeader][GraphDefinition image];
// verify_state gates the one-time integrity hash so submissions sharing one
// Definition each pay only a state load.
inline constexpr uint64_t GRAPH_DEFINITION_OBJECT_MAGIC = 0x4752415048455844ULL;

enum class GraphDefinitionVerifyState : uint32_t {
    UPLOADED = 0,
    VERIFYING = 1,
    VERIFIED = 2,
    INVALID = 3,
};

struct GraphDefinitionHeader {
    uint64_t magic;
    std::atomic<uint32_t> verify_state;
    uint32_t definition_bytes;
    uint64_t content_hash;
    uint64_t full_key;
};

static_assert(std::is_standard_layout_v<GraphDefinitionHeader>);

struct GraphDefinition {
    uint64_t full_key;
    uint64_t content_hash;
    uint64_t required_heap;
    uint32_t total_bytes;
    uint32_t task_count;
    uint32_t edge_count;
    uint32_t root_count;
    uint32_t boundary_count;
    uint32_t boundary_scalar_count;
    uint32_t tensor_arg_count;
    uint32_t scalar_arg_count;
    uint32_t predicate_count;
    // Bytes the GraphExecution header, node array and node argument pools need in
    // the outer GRAPH task's heap tail. Invocation boundaries live in the outer
    // task payload's compact argument-pool regions instead.
    uint32_t execution_storage_bytes;
    uint32_t off_fanout_offsets;
    uint32_t off_fanout_indices;
    uint32_t off_fanin_offsets;
    uint32_t off_fanin_indices;
    uint32_t off_root_indices;
    uint32_t off_node_offsets;
    uint32_t off_nodes;
    uint32_t off_tensors;
    uint32_t off_tensor_sources;
    uint32_t off_scalars;
    uint32_t off_scalar_sources;
    uint32_t off_boundary_signatures;
    uint32_t off_predicates;
};

static_assert(std::is_trivially_copyable_v<GraphTensorSourceRef>);
static_assert(std::is_standard_layout_v<GraphTensorSourceRef>);
static_assert(std::is_trivially_copyable_v<GraphTensor>);
static_assert(std::is_standard_layout_v<GraphTensor>);
static_assert(std::is_trivially_copyable_v<GraphScalarSourceRef>);
static_assert(std::is_standard_layout_v<GraphScalarSourceRef>);
static_assert(std::is_trivially_copyable_v<GraphNodeDefinition>);
static_assert(std::is_standard_layout_v<GraphNodeDefinition>);
static_assert(std::is_trivially_copyable_v<GraphPredicate>);
static_assert(std::is_standard_layout_v<GraphPredicate>);
static_assert(std::is_trivially_copyable_v<GraphBoundarySignature>);
static_assert(std::is_standard_layout_v<GraphBoundarySignature>);
static_assert(std::is_trivially_copyable_v<GraphDefinition>);
static_assert(std::is_standard_layout_v<GraphDefinition>);

inline GraphTensor graph_tensor_pack(const ChipTensor &tensor) {
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

inline void graph_tensor_unpack(const GraphTensor &packed, ChipTensor *tensor) {
    tensor->buffer = PTOBufferHandle{packed.buffer_addr, packed.buffer_size};
    tensor->owner_task_id = PTO2TaskId{packed.owner_task_id};
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

template <typename T>
inline const T *graph_definition_array(const GraphDefinition &definition, uint32_t offset, uint32_t count) {
    if (offset == 0 || offset > definition.total_bytes || offset % alignof(T) != 0) return nullptr;
    const size_t remaining = static_cast<size_t>(definition.total_bytes - offset);
    if (count > remaining / sizeof(T)) return nullptr;
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

struct alignas(64) GraphNodeStorage {
    PTO2TaskDescriptor task;
    PTO2TaskSlotState slot;
    PTO2TaskPayload payload;
};

inline constexpr uint8_t GRAPH_EXECUTION_STATE_MASK = 0x7;
inline constexpr uint8_t GRAPH_EXECUTION_EXTERNAL_READY = 0x8;

struct GraphExecution {
    // The low bits hold GraphExecutionState. EXTERNAL_READY shares this byte so
    // dependency readiness can arrive before materialization without a separate
    // per-submission gate object.
    std::atomic<uint8_t> state{static_cast<uint8_t>(GraphExecutionState::SUBMITTED)};
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_nodes{0};
    std::atomic<int32_t> retired_nodes{0};
    // Incremental activation: nodes in [0, published_nodes) are fully
    // materialized and registered, so a route pass may consider them. route_cursor
    // is the next such node index a route pass will claim; roots below it have
    // been pushed to the ready queue exactly once. Both advance monotonically and
    // reset per (re)submission.
    std::atomic<int32_t> published_nodes{0};
    std::atomic<int32_t> route_cursor{0};
    int32_t node_count{0};
    int32_t materialized_nodes{0};
    int32_t constructed_nodes{0};
    uint32_t consumed_tensor_args{0};
    PTO2TaskSlotState *outer_slot{nullptr};
    GraphNodeStorage *nodes{nullptr};
    GraphNodeStorage *node_storage{nullptr};
    // This execution's node argument pools, in the storage tail past node_storage.
    // Every node payload's tensor and scalar deltas point here; its pool position is
    // the Definition's tensor_offset / scalar_offset.
    ChipTensor *node_tensor_pool{nullptr};
    uint64_t *node_scalar_pool{nullptr};
    const GraphDefinition *definition{nullptr};
    const uint32_t *fanin_offsets{nullptr};
    const uint16_t *fanin_indices{nullptr};
    const GraphTensor *boundary_tensors{nullptr};
    uint32_t boundary_tensor_count{0};
    const uint64_t *boundary_scalars{nullptr};
    uint32_t boundary_scalar_count{0};

    GraphNodeStorage &node_at(int32_t index) const { return node_storage[index]; }
};

static_assert(std::is_trivially_destructible_v<GraphNodeStorage>);
// The tensor pool starts right after the node array, and the scalar pool starts
// after a whole number of ChipTensors.
static_assert(
    alignof(GraphNodeStorage) % alignof(ChipTensor) == 0,
    "a node entry must be at least ChipTensor-aligned: the tensor pool follows the node array"
);
static_assert(sizeof(ChipTensor) % alignof(uint64_t) == 0, "the tensor stride must keep the scalar pool aligned");
static_assert(std::is_trivially_destructible_v<GraphExecution>);
static_assert(std::is_trivially_copyable_v<GraphExecution>);
static_assert(sizeof(GraphTensor) <= sizeof(ChipTensor));

inline size_t graph_boundary_tensor_pool_slots(uint32_t tensor_count) {
    const size_t bytes = static_cast<size_t>(tensor_count) * sizeof(GraphTensor);
    return (bytes + sizeof(ChipTensor) - 1) / sizeof(ChipTensor);
}

// The outer GRAPH task's heap tail occupies
// [GraphExecution][GraphNodeStorage x node_count][ChipTensor x tensor_arg_count]
// [uint64_t x scalar_arg_count].
//
// The last two regions are the node payloads' argument pools, indexed by the
// Definition's own tensor_offset / scalar_offset — which is why the Definition's
// arg-table counts size them rather than a per-node sum: node i's arguments occupy
// [offset, offset + count) in both the table and the pool. There is no fanin region:
// node dependencies live in the Definition's fanin CSR, so a node's fanin_count stays
// 0 and its fanin delta unbound.
struct GraphExecutionStorageLayout {
    size_t nodes_offset;
    size_t tensors_offset;
    size_t scalars_offset;
    size_t total_bytes;
};

inline bool graph_execution_storage_layout(
    int32_t node_count, uint32_t tensor_arg_count, uint32_t scalar_arg_count, GraphExecutionStorageLayout *out
) {
    if (out == nullptr || node_count <= 0 || node_count > static_cast<int32_t>(GRAPH_MAX_NODES)) {
        return false;
    }
    constexpr size_t ALIGNMENT = alignof(GraphNodeStorage);
    out->nodes_offset = (sizeof(GraphExecution) + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    out->tensors_offset = out->nodes_offset + static_cast<size_t>(node_count) * sizeof(GraphNodeStorage);
    out->scalars_offset = out->tensors_offset + static_cast<size_t>(tensor_arg_count) * sizeof(ChipTensor);
    out->total_bytes = out->scalars_offset + static_cast<size_t>(scalar_arg_count) * sizeof(uint64_t);
    return true;
}

inline bool graph_execution_storage_bytes(
    int32_t node_count, uint32_t tensor_arg_count, uint32_t scalar_arg_count, size_t *storage_bytes
) {
    GraphExecutionStorageLayout layout{};
    if (storage_bytes == nullptr ||
        !graph_execution_storage_layout(node_count, tensor_arg_count, scalar_arg_count, &layout)) {
        return false;
    }
    *storage_bytes = layout.total_bytes;
    return true;
}

GraphExecution *graph_execution_localize(PTO2TaskSlotState &outer_slot);
GraphMaterializeResult graph_execution_materialize_slice(
    PTO2TaskSlotState &outer_slot, GraphExecution &execution, int32_t max_nodes, int32_t *nodes_materialized = nullptr
);

inline GraphExecution *graph_execution_from_slot(PTO2TaskSlotState &slot) {
    return slot.task_kind == TaskKind::GRAPH_NODE ? static_cast<GraphExecution *>(slot.graph_context) : nullptr;
}

inline GraphExecution *graph_execution_from_outer_slot(PTO2TaskSlotState &slot) {
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
    const uint8_t next_state = static_cast<uint8_t>(next);
    while (!execution.state.compare_exchange_weak(
        observed, static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | next_state), order,
        std::memory_order_relaxed
    )) {}
}

inline bool graph_execution_transition(
    GraphExecution &execution, GraphExecutionState expected_state, GraphExecutionState next_state
) {
    uint8_t observed = execution.state.load(std::memory_order_acquire);
    while ((observed & GRAPH_EXECUTION_STATE_MASK) == static_cast<uint8_t>(expected_state)) {
        const uint8_t desired =
            static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_STATE_MASK) | static_cast<uint8_t>(next_state));
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

inline bool graph_execution_complete_node(GraphExecution &execution) {
    return execution.remaining_nodes.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void graph_execution_mark_completed(GraphExecution &execution) {
    graph_execution_set_state(execution, GraphExecutionState::COMPLETED);
}

inline void graph_execution_retire_node(GraphExecution &execution) {
    execution.retired_nodes.fetch_add(1, std::memory_order_release);
}
