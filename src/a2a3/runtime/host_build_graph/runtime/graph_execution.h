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
    uint8_t child_memory;
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

struct GraphNodeDefinition {
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];
    uint8_t active_mask;
    uint8_t task_attrs;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    uint16_t reserved;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    uint32_t tensor_offset;
    uint32_t scalar_offset;
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
};

struct GraphSubmission {
    uint64_t graph_key;
    uint64_t execution_storage;
    uint64_t execution_storage_bytes;
    uint64_t local_execution;
    uint32_t activation_gate;
    uint32_t total_bytes;
    uint32_t definition_offset;
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
    packed.child_memory = tensor.child_memory;
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
    tensor->child_memory = packed.child_memory;
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
        tensor.is_contiguous > 1 || tensor.child_memory > 1) {
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

inline const GraphDefinition *graph_submission_definition(const GraphSubmission &submission) {
    if (submission.definition_offset == 0 || submission.definition_offset % alignof(GraphDefinition) != 0 ||
        submission.definition_offset > submission.total_bytes ||
        sizeof(GraphDefinition) > submission.total_bytes - submission.definition_offset) {
        return nullptr;
    }
    return reinterpret_cast<const GraphDefinition *>(
        reinterpret_cast<const uint8_t *>(&submission) + submission.definition_offset
    );
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

enum class GraphTensorAddressSource : uint8_t {
    BOUNDARY = 0,
    INTERNAL = 1,
};

// Precomputed on the first materialization and retained next to the node
// storage. Affine replay walks this compact POD instead of re-reading and
// classifying GraphTensorSourceRef entries from the Definition.
struct GraphTensorAddressPatch {
    uint64_t address_offset;
    uint16_t source_index;
    uint8_t source;
    uint8_t reserved[5];
};

static_assert(std::is_trivially_copyable_v<GraphTensorAddressPatch>);
static_assert(std::is_standard_layout_v<GraphTensorAddressPatch>);
static_assert(sizeof(GraphTensorAddressPatch) == 16);

struct GraphScalarPatch {
    uint16_t node_index;
    uint8_t node_scalar_index;
    uint8_t boundary_scalar_index;
};

static_assert(std::is_trivially_copyable_v<GraphScalarPatch>);
static_assert(std::is_standard_layout_v<GraphScalarPatch>);
static_assert(sizeof(GraphScalarPatch) == 4);
static_assert(GRAPH_MAX_NODES <= UINT16_MAX);
static_assert(MAX_SCALAR_ARGS <= UINT8_MAX);

struct alignas(64) GraphNodeStorage {
    PTO2TaskDescriptor task;
    PTO2TaskPayload payload;
    PTO2TaskSlotState slot;
};

inline constexpr uint64_t GRAPH_EXECUTION_STORAGE_MAGIC = 0x4752415048455845ULL;
inline constexpr uint64_t GRAPH_EXECUTION_INITIALIZING = 1;

struct GraphExecution {
    uint64_t storage_magic{0};
    std::atomic<GraphExecutionState> state{GraphExecutionState::SUBMITTED};
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_nodes{0};
    std::atomic<int32_t> retired_nodes{0};
    int32_t node_count{0};
    int32_t node_capacity{0};
    int32_t materialized_nodes{0};
    int32_t materialized_node_count{0};
    int32_t constructed_nodes{0};
    uint32_t tensor_patch_capacity{0};
    uint32_t materialized_tensor_patches{0};
    uint32_t materialized_tensor_patch_count{0};
    uint32_t scalar_patch_capacity{0};
    uint32_t materialized_scalar_patches{0};
    uint32_t materialized_scalar_patch_count{0};
    size_t allocation_bytes{0};
    size_t definition_capacity{0};
    uint64_t graph_key{0};
    uint64_t definition_hash{0};
    uint64_t materialized_graph_key{0};
    uint64_t materialized_definition_hash{0};
    uintptr_t materialized_outer_base{0};
    bool definition_affine_reuse{false};
    PTO2TaskSlotState *outer_slot{nullptr};
    GraphNodeStorage *nodes{nullptr};
    GraphNodeStorage *node_storage{nullptr};
    GraphTensorAddressPatch *tensor_patches{nullptr};
    GraphScalarPatch *scalar_patches{nullptr};
    void *definition_storage{nullptr};
    const GraphDefinition *definition{nullptr};
    const uint32_t *fanin_offsets{nullptr};
    const uint16_t *fanin_indices{nullptr};
    const GraphTensor *boundary_tensors{nullptr};
    uint32_t boundary_tensor_count{0};
    const uint64_t *boundary_scalars{nullptr};
    uint32_t boundary_scalar_count{0};
};

static_assert(std::is_trivially_destructible_v<GraphNodeStorage>);
static_assert(std::is_trivially_destructible_v<GraphExecution>);

inline bool graph_execution_storage_layout(
    int32_t node_capacity, uint32_t tensor_patch_capacity, uint32_t scalar_patch_capacity, size_t definition_capacity,
    size_t *nodes_offset, size_t *tensor_patches_offset, size_t *scalar_patches_offset, size_t *definition_offset,
    size_t *storage_bytes
) {
    if (nodes_offset == nullptr || tensor_patches_offset == nullptr || scalar_patches_offset == nullptr ||
        definition_offset == nullptr || storage_bytes == nullptr || node_capacity <= 0 ||
        static_cast<size_t>(node_capacity) > SIZE_MAX / sizeof(GraphNodeStorage) ||
        tensor_patch_capacity > GRAPH_MAX_NODES * MAX_TENSOR_ARGS ||
        scalar_patch_capacity > GRAPH_MAX_NODES * MAX_SCALAR_ARGS) {
        return false;
    }
    auto checked_align_up = [](size_t value, size_t alignment, size_t *result) {
        if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return false;
        *result = (value + alignment - 1) & ~(alignment - 1);
        return true;
    };
    const size_t nodes_bytes = static_cast<size_t>(node_capacity) * sizeof(GraphNodeStorage);
    const size_t tensor_patches_bytes = static_cast<size_t>(tensor_patch_capacity) * sizeof(GraphTensorAddressPatch);
    const size_t scalar_patches_bytes = static_cast<size_t>(scalar_patch_capacity) * sizeof(GraphScalarPatch);
    if (!checked_align_up(sizeof(GraphExecution), alignof(GraphNodeStorage), nodes_offset) ||
        *nodes_offset > SIZE_MAX - nodes_bytes ||
        !checked_align_up(*nodes_offset + nodes_bytes, alignof(GraphTensorAddressPatch), tensor_patches_offset) ||
        *tensor_patches_offset > SIZE_MAX - tensor_patches_bytes ||
        !checked_align_up(
            *tensor_patches_offset + tensor_patches_bytes, alignof(GraphScalarPatch), scalar_patches_offset
        ) ||
        *scalar_patches_offset > SIZE_MAX - scalar_patches_bytes ||
        !checked_align_up(*scalar_patches_offset + scalar_patches_bytes, alignof(GraphDefinition), definition_offset) ||
        *definition_offset > SIZE_MAX - definition_capacity) {
        return false;
    }
    return checked_align_up(*definition_offset + definition_capacity, alignof(GraphNodeStorage), storage_bytes);
}

inline bool graph_execution_storage_bytes(
    int32_t node_capacity, uint32_t tensor_patch_capacity, uint32_t scalar_patch_capacity, size_t definition_capacity,
    size_t *storage_bytes
) {
    size_t nodes_offset = 0;
    size_t tensor_patches_offset = 0;
    size_t scalar_patches_offset = 0;
    size_t definition_offset = 0;
    return graph_execution_storage_layout(
        node_capacity, tensor_patch_capacity, scalar_patch_capacity, definition_capacity, &nodes_offset,
        &tensor_patches_offset, &scalar_patches_offset, &definition_offset, storage_bytes
    );
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
    return (observed | bit) == BOTH;
}

inline GraphExecution *graph_submission_local_execution(GraphSubmission &submission) {
    uint64_t raw = __atomic_load_n(&submission.local_execution, __ATOMIC_ACQUIRE);
    if (raw <= GRAPH_EXECUTION_INITIALIZING) return nullptr;
    return reinterpret_cast<GraphExecution *>(static_cast<uintptr_t>(raw));
}

inline bool graph_submission_execution_initializing(const GraphSubmission &submission) {
    return __atomic_load_n(&submission.local_execution, __ATOMIC_ACQUIRE) == GRAPH_EXECUTION_INITIALIZING;
}
