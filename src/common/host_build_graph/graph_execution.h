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

// Everything from GraphTensorSourceRef through GraphSubmission is copied
// across the host-device boundary. Keep it pointer-free, fixed-width and
// position-independent: every reference is an offset from its owning header.
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
    // Bytes an execution of this Definition needs for its GraphExecution header,
    // node storage and patch arrays. Derived from task_count / tensor_arg_count /
    // scalar_arg_count, so host and device read one value instead of each
    // computing it. The outer GRAPH task's heap allocation covers
    // required_heap + this, and the execution lives at
    // packed_buffer_base + required_heap.
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

inline uint64_t graph_definition_hash_rotl(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

inline uint64_t graph_definition_hash_round(uint64_t accumulator, uint64_t input) {
    constexpr uint64_t PRIME2 = 14029467366897019727ULL;
    constexpr uint64_t PRIME1 = 11400714785074694791ULL;
    accumulator += input * PRIME2;
    accumulator = graph_definition_hash_rotl(accumulator, 31);
    return accumulator * PRIME1;
}

inline uint64_t graph_definition_hash_word(const uint8_t *bytes, size_t offset) {
    constexpr size_t CONTENT_HASH_OFFSET = offsetof(GraphDefinition, content_hash);
    static_assert(CONTENT_HASH_OFFSET % sizeof(uint64_t) == 0);
    if (offset == CONTENT_HASH_OFFSET) return 0;
    uint64_t word = 0;
    __builtin_memcpy(&word, bytes + offset, sizeof(word));
    return word;
}

// The Definition hash follows XXH64's four-lane structure so large images do
// not serialize one multiply per word. content_hash is treated as zero on both
// host and device, allowing verification without mutating the uploaded image.
// Definitions are rebuilt and rehashed by the same runtime version, so this is
// an integrity checksum rather than a persisted wire-format identifier.
inline uint64_t graph_definition_content_hash(const void *data, size_t size) {
    constexpr uint64_t SEED = 1469598103934665603ULL;
    constexpr uint64_t PRIME1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME2 = 14029467366897019727ULL;
    constexpr uint64_t PRIME3 = 1609587929392839161ULL;
    constexpr uint64_t PRIME4 = 9650029242287828579ULL;
    constexpr uint64_t PRIME5 = 2870177450012600261ULL;
    const auto *bytes = static_cast<const uint8_t *>(data);
    size_t offset = 0;
    uint64_t hash = 0;

    if (size >= 32) {
        uint64_t lane1 = SEED + PRIME1 + PRIME2;
        uint64_t lane2 = SEED + PRIME2;
        uint64_t lane3 = SEED;
        uint64_t lane4 = SEED - PRIME1;
        const size_t stripes_end = size - 32;
        do {
            lane1 = graph_definition_hash_round(lane1, graph_definition_hash_word(bytes, offset));
            lane2 = graph_definition_hash_round(lane2, graph_definition_hash_word(bytes, offset + 8));
            lane3 = graph_definition_hash_round(lane3, graph_definition_hash_word(bytes, offset + 16));
            lane4 = graph_definition_hash_round(lane4, graph_definition_hash_word(bytes, offset + 24));
            offset += 32;
        } while (offset <= stripes_end);
        hash = graph_definition_hash_rotl(lane1, 1) + graph_definition_hash_rotl(lane2, 7) +
               graph_definition_hash_rotl(lane3, 12) + graph_definition_hash_rotl(lane4, 18);
        hash ^= graph_definition_hash_round(0, lane1);
        hash = hash * PRIME1 + PRIME4;
        hash ^= graph_definition_hash_round(0, lane2);
        hash = hash * PRIME1 + PRIME4;
        hash ^= graph_definition_hash_round(0, lane3);
        hash = hash * PRIME1 + PRIME4;
        hash ^= graph_definition_hash_round(0, lane4);
        hash = hash * PRIME1 + PRIME4;
    } else {
        hash = SEED + PRIME5;
    }

    hash += size;
    while (offset + sizeof(uint64_t) <= size) {
        const uint64_t mixed = graph_definition_hash_round(0, graph_definition_hash_word(bytes, offset));
        hash ^= mixed;
        hash = graph_definition_hash_rotl(hash, 27) * PRIME1 + PRIME4;
        offset += sizeof(uint64_t);
    }
    if (offset + sizeof(uint32_t) <= size) {
        uint32_t word = 0;
        __builtin_memcpy(&word, bytes + offset, sizeof(word));
        hash ^= static_cast<uint64_t>(word) * PRIME1;
        hash = graph_definition_hash_rotl(hash, 23) * PRIME2 + PRIME3;
        offset += sizeof(uint32_t);
    }
    while (offset < size) {
        hash ^= static_cast<uint64_t>(bytes[offset]) * PRIME5;
        hash = graph_definition_hash_rotl(hash, 11) * PRIME1;
        ++offset;
    }
    hash ^= hash >> 33;
    hash *= PRIME2;
    hash ^= hash >> 29;
    hash *= PRIME3;
    hash ^= hash >> 32;
    return hash;
}

// One submission of a Graph. The static Definition is a shared device object
// this references by address; the execution storage is not referenced at all —
// it sits at outer_slot.task->packed_buffer_base + definition->required_heap,
// so both sides derive it from the Definition rather than carrying it on the
// wire.
struct GraphSubmission {
    uint64_t graph_key;
    uint64_t local_execution;
    // Device GM address of the shared Definition object this replay references
    // (an integer-typed absolute address per the wire rules) plus the content
    // hash the host computed for it.
    uint64_t definition_addr;
    uint64_t definition_hash;
    uint32_t activation_gate;
    uint32_t total_bytes;
    uint32_t tensors_offset;
    uint32_t tensor_count;
    uint32_t scalars_offset;
    uint32_t scalar_count;
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
static_assert(std::is_trivially_copyable_v<GraphSubmission>);
static_assert(std::is_standard_layout_v<GraphSubmission>);

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

inline GraphSubmission *graph_submission_from_slot(PTO2TaskSlotState &slot) {
    return slot.task_kind == TaskKind::GRAPH ? static_cast<GraphSubmission *>(slot.graph_context) : nullptr;
}

inline bool graph_submission_wire_size_valid(const GraphSubmission &submission, size_t available_bytes) {
    return available_bytes >= sizeof(GraphSubmission) && submission.total_bytes == available_bytes;
}

inline const GraphTensor *graph_submission_tensors(const GraphSubmission &submission) {
    if (submission.tensors_offset == 0 || submission.tensors_offset % alignof(GraphTensor) != 0 ||
        submission.tensors_offset > submission.total_bytes ||
        submission.tensor_count > (submission.total_bytes - submission.tensors_offset) / sizeof(GraphTensor)) {
        return nullptr;
    }
    return reinterpret_cast<const GraphTensor *>(
        reinterpret_cast<const uint8_t *>(&submission) + submission.tensors_offset
    );
}

inline const uint64_t *graph_submission_scalars(const GraphSubmission &submission) {
    if (submission.scalar_count == 0) return nullptr;
    if (submission.scalars_offset == 0 || submission.scalars_offset % alignof(uint64_t) != 0 ||
        submission.scalars_offset > submission.total_bytes ||
        submission.scalar_count > (submission.total_bytes - submission.scalars_offset) / sizeof(uint64_t)) {
        return nullptr;
    }
    return reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const uint8_t *>(&submission) + submission.scalars_offset
    );
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
    // The payload carries its argument regions as deltas into pools past the node
    // array, so its size is the same for every node and the slot names it by a delta
    // from the slot's own address. Field order here therefore constrains nothing, and
    // node_at strides the storage by this type.
    PTO2TaskPayload payload;
};

inline constexpr uint64_t GRAPH_EXECUTION_INITIALIZING = 1;

struct GraphExecution {
    std::atomic<GraphExecutionState> state{GraphExecutionState::SUBMITTED};
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
// The tensor pool starts right after the node array, so the node stride has to carry
// ChipTensor's alignment; the scalar pool then starts after a whole number of
// ChipTensors. Neither holds by construction — both are properties of the two types.
static_assert(
    alignof(GraphNodeStorage) % alignof(ChipTensor) == 0,
    "a node entry must be at least ChipTensor-aligned: the tensor pool follows the node array"
);
static_assert(sizeof(ChipTensor) % alignof(uint64_t) == 0, "the tensor stride must keep the scalar pool aligned");
static_assert(std::is_trivially_destructible_v<GraphExecution>);

// The execution occupies
// [GraphExecution][GraphNodeStorage x node_count][ChipTensor x tensor_arg_count]
// [uint64_t x scalar_arg_count] at the tail of the outer GRAPH task's heap allocation.
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

inline bool graph_execution_complete_node(GraphExecution &execution) {
    return execution.remaining_nodes.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void graph_execution_mark_completed(GraphExecution &execution) {
    execution.state.store(GraphExecutionState::COMPLETED, std::memory_order_release);
}

inline void graph_execution_retire_node(GraphExecution &execution) {
    execution.retired_nodes.fetch_add(1, std::memory_order_release);
}

inline bool graph_submission_signal(GraphSubmission &submission, uint32_t bit) {
    constexpr uint32_t BOTH = 0x3;
    uint32_t observed = __atomic_fetch_or(&submission.activation_gate, bit, __ATOMIC_ACQ_REL);
    return observed != BOTH && (observed | bit) == BOTH;
}

inline GraphExecution *graph_submission_local_execution(GraphSubmission &submission) {
    uint64_t raw = __atomic_load_n(&submission.local_execution, __ATOMIC_ACQUIRE);
    if (raw <= GRAPH_EXECUTION_INITIALIZING) return nullptr;
    return reinterpret_cast<GraphExecution *>(static_cast<uintptr_t>(raw));
}

inline bool graph_submission_execution_initializing(const GraphSubmission &submission) {
    return __atomic_load_n(&submission.local_execution, __ATOMIC_ACQUIRE) == GRAPH_EXECUTION_INITIALIZING;
}
