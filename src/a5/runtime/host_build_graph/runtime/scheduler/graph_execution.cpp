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

#include "graph_execution.h"

#include <algorithm>
#include <cstring>
#include <new>

#include "graph_cache.h"

namespace {

void destroy_execution_nodes(GraphExecution *execution) {
    for (int32_t i = 0; i < execution->constructed_nodes; ++i) {
        execution->node_storage[i].~GraphNodeStorage();
    }
    execution->constructed_nodes = 0;
}

void reset_execution(
    GraphExecution *execution, int32_t node_count, uint32_t tensor_patch_count, uint64_t graph_key,
    uint64_t definition_hash
) {
    size_t nodes_offset = 0;
    size_t tensor_patches_offset = 0;
    size_t definition_offset = 0;
    size_t bytes = 0;
    if (!graph_execution_storage_layout(
            execution->node_capacity, execution->tensor_patch_capacity, execution->definition_capacity, &nodes_offset,
            &tensor_patches_offset, &definition_offset, &bytes
        )) {
        return;
    }
    execution->definition_affine_reuse = graph_key != 0 && execution->materialized_graph_key == graph_key &&
                                         execution->materialized_definition_hash == definition_hash &&
                                         execution->materialized_node_count == node_count &&
                                         execution->materialized_tensor_patch_count == tensor_patch_count &&
                                         execution->constructed_nodes >= node_count;
    execution->state.store(GraphExecutionState::SUBMITTED, std::memory_order_relaxed);
    execution->materialize_busy.store(0, std::memory_order_relaxed);
    execution->remaining_nodes.store(node_count, std::memory_order_relaxed);
    execution->retired_nodes.store(0, std::memory_order_relaxed);
    execution->node_count = node_count;
    execution->materialized_nodes = 0;
    execution->materialized_tensor_patches = 0;
    execution->allocation_bytes = bytes;
    execution->graph_key = graph_key;
    execution->definition_hash = definition_hash;
    execution->outer_slot = nullptr;
    execution->nodes = nullptr;
    execution->node_storage =
        reinterpret_cast<GraphNodeStorage *>(reinterpret_cast<uint8_t *>(execution) + nodes_offset);
    execution->tensor_patches =
        reinterpret_cast<GraphTensorAddressPatch *>(reinterpret_cast<uint8_t *>(execution) + tensor_patches_offset);
    execution->definition_storage = reinterpret_cast<uint8_t *>(execution) + definition_offset;
    if (!execution->definition_affine_reuse) {
        execution->definition = nullptr;
        execution->fanin_offsets = nullptr;
        execution->fanin_indices = nullptr;
    }
    execution->boundary_tensors = nullptr;
    execution->boundary_tensor_count = 0;
}

bool reusable_execution_header_valid(const GraphExecution &execution, size_t storage_bytes) {
    if (execution.storage_magic != GRAPH_EXECUTION_STORAGE_MAGIC || execution.node_capacity <= 0 ||
        execution.node_capacity > static_cast<int32_t>(GRAPH_MAX_NODES) || execution.definition_capacity == 0 ||
        execution.tensor_patch_capacity > GRAPH_MAX_NODES * MAX_TENSOR_ARGS ||
        execution.materialized_tensor_patch_count > execution.tensor_patch_capacity ||
        execution.constructed_nodes < 0 || execution.constructed_nodes > execution.node_capacity ||
        execution.node_count <= 0 || execution.node_count > execution.node_capacity ||
        execution.state.load(std::memory_order_acquire) != GraphExecutionState::COMPLETED ||
        execution.retired_nodes.load(std::memory_order_acquire) < execution.node_count) {
        return false;
    }
    size_t expected_bytes = 0;
    return graph_execution_storage_bytes(
               execution.node_capacity, execution.tensor_patch_capacity, execution.definition_capacity, &expected_bytes
           ) &&
           execution.allocation_bytes == expected_bytes && expected_bytes <= storage_bytes;
}

GraphExecution *acquire_host_execution_storage(
    GraphSubmission &submission, int32_t node_count, uint64_t graph_key, uint64_t definition_hash,
    uint32_t tensor_patch_count, size_t definition_bytes
) {
    if (node_count <= 0 || node_count > static_cast<int32_t>(GRAPH_MAX_NODES) || definition_bytes == 0 ||
        submission.execution_storage == 0 || submission.execution_storage_bytes > SIZE_MAX ||
        submission.execution_storage % alignof(GraphNodeStorage) != 0) {
        return nullptr;
    }
    const size_t storage_bytes = static_cast<size_t>(submission.execution_storage_bytes);
    size_t required_bytes = 0;
    if (!graph_execution_storage_bytes(node_count, tensor_patch_count, definition_bytes, &required_bytes) ||
        required_bytes > storage_bytes) {
        return nullptr;
    }

    auto *execution = reinterpret_cast<GraphExecution *>(static_cast<uintptr_t>(submission.execution_storage));
    uint64_t observed_magic = 0;
    std::memcpy(&observed_magic, execution, sizeof(observed_magic));
    const bool has_existing_execution = observed_magic == GRAPH_EXECUTION_STORAGE_MAGIC;
    const bool valid_header = has_existing_execution && reusable_execution_header_valid(*execution, storage_bytes);
    if (has_existing_execution && !valid_header) return nullptr;
    const bool capacities_fit = valid_header && execution->node_capacity >= node_count &&
                                execution->tensor_patch_capacity >= tensor_patch_count &&
                                execution->definition_capacity >= definition_bytes;
    if (!capacities_fit) {
        if (valid_header) destroy_execution_nodes(execution);
        execution = new (execution) GraphExecution{};
        execution->node_capacity = node_count;
        execution->tensor_patch_capacity = tensor_patch_count;
        execution->definition_capacity = definition_bytes;
        execution->allocation_bytes = required_bytes;
    }
    reset_execution(execution, node_count, tensor_patch_count, graph_key, definition_hash);
    execution->storage_magic = GRAPH_EXECUTION_STORAGE_MAGIC;
    return execution;
}

void reset_graph_payload(PTO2TaskPayload &payload) {
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; ++w) {
        payload.staged_core_mask[w].store(0, std::memory_order_relaxed);
    }
    payload.dispatch_fanin.store(0, std::memory_order_relaxed);
    payload.dispatch_propagated.store(0, std::memory_order_relaxed);
    payload.published_block_count.store(0, std::memory_order_relaxed);
    payload.early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload.running_slot_count.store(0, std::memory_order_relaxed);
    payload.early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

bool bind_graph_topology(GraphExecution &execution) {
    if (execution.definition == nullptr) return false;
    const GraphDefinition &definition = *execution.definition;
    const uint32_t *fanin_offsets =
        graph_definition_array<uint32_t>(definition, definition.off_fanin_offsets, definition.task_count + 1);
    const uint16_t *fanin_indices =
        definition.edge_count == 0 ?
            nullptr :
            graph_definition_array<uint16_t>(definition, definition.off_fanin_indices, definition.edge_count);
    const uint32_t *fanout_offsets =
        graph_definition_array<uint32_t>(definition, definition.off_fanout_offsets, definition.task_count + 1);
    const uint16_t *fanout_indices =
        definition.edge_count == 0 ?
            nullptr :
            graph_definition_array<uint16_t>(definition, definition.off_fanout_indices, definition.edge_count);
    const uint16_t *roots =
        graph_definition_array<uint16_t>(definition, definition.off_root_indices, definition.root_count);
    const GraphNodeDefinition *nodes =
        graph_definition_array<GraphNodeDefinition>(definition, definition.off_nodes, definition.task_count);
    const uint64_t *node_offsets =
        graph_definition_array<uint64_t>(definition, definition.off_node_offsets, definition.task_count);
    if (fanin_offsets == nullptr || fanout_offsets == nullptr || roots == nullptr || nodes == nullptr ||
        node_offsets == nullptr ||
        (definition.edge_count != 0 && (fanin_indices == nullptr || fanout_indices == nullptr)) ||
        fanin_offsets[0] != 0 || fanout_offsets[0] != 0 ||
        fanin_offsets[definition.task_count] != definition.edge_count ||
        fanout_offsets[definition.task_count] != definition.edge_count) {
        return false;
    }

    uint64_t required_heap = 0;
    constexpr uint8_t VALID_ACTIVE_MASK = (1U << PTO2_SUBTASK_SLOT_COUNT) - 1U;
    for (uint32_t i = 0; i < definition.task_count; ++i) {
        const GraphNodeDefinition &node = nodes[i];
        if (node_offsets[i] != required_heap || node.total_output_size < 0 || node.tensor_count < 0 ||
            node.tensor_count > MAX_TENSOR_ARGS || node.scalar_count < 0 || node.scalar_count > MAX_SCALAR_ARGS ||
            node.tensor_offset > definition.tensor_arg_count ||
            static_cast<uint32_t>(node.tensor_count) > definition.tensor_arg_count - node.tensor_offset ||
            node.scalar_offset > definition.scalar_arg_count ||
            static_cast<uint32_t>(node.scalar_count) > definition.scalar_arg_count - node.scalar_offset ||
            (node.active_mask & ~VALID_ACTIVE_MASK) != 0 || node.logical_block_num <= 0 ||
            node.total_required_subtasks < 0) {
            return false;
        }
        for (int32_t slot = 0; slot < PTO2_SUBTASK_SLOT_COUNT; ++slot) {
            const bool active = (node.active_mask & (1U << slot)) != 0;
            if (active != (node.kernel_id[slot] != INVALID_KERNEL_ID)) return false;
        }
        const uint64_t output_bytes = PTO2_ALIGN_UP(static_cast<uint64_t>(node.total_output_size), PTO2_ALIGN_SIZE);
        if (output_bytes > definition.required_heap - required_heap) return false;
        required_heap += output_bytes;
    }
    if (required_heap != definition.required_heap) return false;

    uint32_t observed_roots = 0;
    for (uint32_t consumer = 0; consumer < definition.task_count; ++consumer) {
        const uint32_t begin = fanin_offsets[consumer];
        const uint32_t end = fanin_offsets[consumer + 1];
        if (begin > end || end > definition.edge_count) return false;
        if (begin == end) observed_roots++;
        for (uint32_t edge = begin; edge < end; ++edge) {
            if (fanin_indices[edge] >= consumer) return false;
        }
    }
    if (observed_roots != definition.root_count) return false;
    for (uint32_t i = 0; i < definition.root_count; ++i) {
        const uint16_t root = roots[i];
        if (root >= definition.task_count || fanin_offsets[root] != fanin_offsets[root + 1]) return false;
    }
    for (uint32_t producer = 0; producer < definition.task_count; ++producer) {
        const uint32_t begin = fanout_offsets[producer];
        const uint32_t end = fanout_offsets[producer + 1];
        if (begin > end || end > definition.edge_count) return false;
        for (uint32_t edge = begin; edge < end; ++edge) {
            if (fanout_indices[edge] <= producer || fanout_indices[edge] >= definition.task_count) return false;
        }
    }

    execution.fanin_offsets = fanin_offsets;
    execution.fanin_indices = fanin_indices;
    return true;
}

bool graph_definition_hash_matches(const GraphDefinition &definition) {
    if (definition.content_hash == 0 || definition.total_bytes < sizeof(GraphDefinition)) return false;
    constexpr size_t HASH_OFFSET = offsetof(GraphDefinition, content_hash);
    constexpr size_t HASH_END = HASH_OFFSET + sizeof(GraphDefinition::content_hash);
    const auto *bytes = reinterpret_cast<const uint8_t *>(&definition);
    uint64_t hash = graph_hash_bytes(1469598103934665603ULL, bytes, HASH_OFFSET);
    const uint64_t zero_hash = 0;
    hash = graph_hash_bytes(hash, &zero_hash, sizeof(zero_hash));
    hash = graph_hash_bytes(hash, bytes + HASH_END, definition.total_bytes - HASH_END);
    return hash == definition.content_hash;
}

bool register_initial_graph_waiter(GraphExecution &execution, int32_t consumer_index) {
    const uint32_t begin = execution.fanin_offsets[consumer_index];
    const uint32_t end = execution.fanin_offsets[consumer_index + 1];
    if (begin == end) return true;

    const uint16_t producer_index = execution.fanin_indices[begin];
    PTO2TaskSlotState &producer = execution.node_storage[producer_index].slot;
    PTO2TaskSlotState &consumer = execution.node_storage[consumer_index].slot;

    // Orch has already wired the dependency as a producer index in fanin CSR.
    // This only creates the execution-local polling subscription. Activation
    // is gated on PREPARED, so no producer can close its list concurrently.
    PTO2TaskSlotState *head = producer.wake_list_head.load(std::memory_order_relaxed);
    if (head == WAKE_LIST_SENTINEL) return false;
    consumer.next_in_wake_list = head;
    producer.wake_list_head.store(&consumer, std::memory_order_relaxed);
    return true;
}

}  // namespace

GraphExecution *graph_execution_localize(PTO2TaskSlotState &outer_slot) {
    GraphSubmission *submission = graph_submission_from_slot(outer_slot);
    if (submission == nullptr) return nullptr;
    if (GraphExecution *existing = graph_submission_local_execution(*submission)) return existing;
    if (graph_submission_execution_initializing(*submission)) return nullptr;

    const GraphDefinition *definition = graph_submission_definition(*submission);
    const GraphTensor *boundary_tensors = graph_submission_tensors(*submission);
    if (definition == nullptr || definition->total_bytes == 0 || definition->task_count == 0 ||
        definition->task_count > GRAPH_MAX_NODES ||
        definition->total_bytes > submission->total_bytes - submission->definition_offset ||
        submission->graph_key != definition->full_key || submission->tensor_count != definition->boundary_count ||
        boundary_tensors == nullptr ||
        submission->tensors_offset < submission->definition_offset + definition->total_bytes ||
        !graph_definition_hash_matches(*definition) || outer_slot.task == nullptr ||
        outer_slot.task->packed_buffer_base == nullptr || outer_slot.task->packed_buffer_end == nullptr) {
        return nullptr;
    }
    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_base);
    const uintptr_t outer_end = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_end);
    if (outer_end < outer_base || definition->required_heap > outer_end - outer_base) return nullptr;
    for (uint32_t i = 0; i < submission->tensor_count; ++i) {
        if (!graph_tensor_wire_valid(boundary_tensors[i])) return nullptr;
    }

    uint64_t expected = 0;
    if (!__atomic_compare_exchange_n(
            &submission->local_execution, &expected, GRAPH_EXECUTION_INITIALIZING, false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        )) {
        return expected > GRAPH_EXECUTION_INITIALIZING ?
                   reinterpret_cast<GraphExecution *>(static_cast<uintptr_t>(expected)) :
                   nullptr;
    }

    GraphExecution *execution = acquire_host_execution_storage(
        *submission, static_cast<int32_t>(definition->task_count), submission->graph_key, definition->content_hash,
        definition->tensor_arg_count, definition->total_bytes
    );
    if (execution == nullptr) {
        __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
        return nullptr;
    }

    if (!execution->definition_affine_reuse) {
        std::memcpy(execution->definition_storage, definition, definition->total_bytes);
        execution->definition = static_cast<const GraphDefinition *>(execution->definition_storage);
        if (!bind_graph_topology(*execution)) {
            execution->retired_nodes.store(execution->node_count, std::memory_order_relaxed);
            execution->state.store(GraphExecutionState::COMPLETED, std::memory_order_release);
            __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
            return nullptr;
        }
    }
    execution->boundary_tensors = boundary_tensors;
    execution->boundary_tensor_count = submission->tensor_count;
    execution->outer_slot = &outer_slot;

    const uint64_t desired = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(execution));
    __atomic_store_n(&submission->local_execution, desired, __ATOMIC_RELEASE);
    return execution;
}

GraphMaterializeResult graph_execution_materialize_slice(
    PTO2TaskSlotState &outer_slot, GraphExecution &execution, int32_t max_nodes, int32_t *nodes_materialized
) {
    if (nodes_materialized != nullptr) *nodes_materialized = 0;
    if (outer_slot.task_kind != TaskKind::GRAPH || outer_slot.task == nullptr ||
        outer_slot.task->packed_buffer_base == nullptr || max_nodes <= 0 || execution.definition == nullptr ||
        execution.node_storage == nullptr || execution.tensor_patches == nullptr) {
        return GraphMaterializeResult::INVALID;
    }

    GraphExecutionState state = execution.state.load(std::memory_order_acquire);
    if (state >= GraphExecutionState::PREPARED) return GraphMaterializeResult::PREPARED;

    uint8_t expected_busy = 0;
    if (!execution.materialize_busy.compare_exchange_strong(
            expected_busy, 1, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return GraphMaterializeResult::BUSY;
    }

    state = execution.state.load(std::memory_order_acquire);
    if (state == GraphExecutionState::SUBMITTED) {
        GraphExecutionState expected = GraphExecutionState::SUBMITTED;
        if (!execution.state.compare_exchange_strong(
                expected, GraphExecutionState::MATERIALIZING, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::BUSY;
        }
    } else if (state != GraphExecutionState::MATERIALIZING) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    const bool affine_reuse = execution.definition_affine_reuse;
    const GraphDefinition &definition = *execution.definition;
    const GraphNodeDefinition *nodes = nullptr;
    const uint64_t *node_offsets = nullptr;
    const GraphTensor *definition_tensors = nullptr;
    const GraphTensorSourceRef *tensor_sources = nullptr;
    const uint64_t *definition_scalars = nullptr;
    if (!affine_reuse) {
        nodes = graph_definition_array<GraphNodeDefinition>(definition, definition.off_nodes, definition.task_count);
        node_offsets = graph_definition_array<uint64_t>(definition, definition.off_node_offsets, definition.task_count);
        definition_tensors =
            definition.tensor_arg_count == 0 ?
                nullptr :
                graph_definition_array<GraphTensor>(definition, definition.off_tensors, definition.tensor_arg_count);
        tensor_sources = definition.tensor_arg_count == 0 ?
                             nullptr :
                             graph_definition_array<GraphTensorSourceRef>(
                                 definition, definition.off_tensor_sources, definition.tensor_arg_count
                             );
        definition_scalars =
            definition.scalar_arg_count == 0 ?
                nullptr :
                graph_definition_array<uint64_t>(definition, definition.off_scalars, definition.scalar_arg_count);
        if (nodes == nullptr || node_offsets == nullptr ||
            (definition.tensor_arg_count != 0 && (definition_tensors == nullptr || tensor_sources == nullptr)) ||
            (definition.scalar_arg_count != 0 && definition_scalars == nullptr)) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
    }

    const int32_t first = execution.materialized_nodes;
    const int32_t last = std::min(execution.node_count, first + max_nodes);
    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_base);
    for (int32_t i = first; i < last; ++i) {
        GraphNodeStorage *storage = &execution.node_storage[i];
        if (i >= execution.constructed_nodes) {
            storage = new (storage) GraphNodeStorage;
            execution.constructed_nodes++;
        }
        PTO2TaskDescriptor &task = storage->task;
        PTO2TaskPayload &payload = storage->payload;
        PTO2TaskSlotState &slot = storage->slot;

        const uint32_t synthetic_local =
            (static_cast<uint32_t>(outer_slot.task->task_id.local()) << 10) | static_cast<uint32_t>(i);
        task.task_id = PTO2TaskId::make(1, synthetic_local);
        uint64_t node_offset = 0;
        uint64_t output_bytes = 0;
        if (affine_reuse) {
            const uintptr_t previous_base = reinterpret_cast<uintptr_t>(task.packed_buffer_base);
            const uintptr_t previous_end = reinterpret_cast<uintptr_t>(task.packed_buffer_end);
            node_offset = previous_base - execution.materialized_outer_base;
            output_bytes = previous_end - previous_base;
        } else {
            const GraphNodeDefinition &source = nodes[i];
            node_offset = node_offsets[i];
            output_bytes = PTO2_ALIGN_UP(static_cast<uint64_t>(source.total_output_size), PTO2_ALIGN_SIZE);
            for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
                task.kernel_id[k] = source.kernel_id[k];
        }
        task.packed_buffer_base = reinterpret_cast<void *>(outer_base + node_offset);
        task.packed_buffer_end = reinterpret_cast<void *>(outer_base + node_offset + output_bytes);

        slot.reset_for_reuse(affine_reuse);
        slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        if (!affine_reuse) {
            const GraphNodeDefinition &source = nodes[i];
            slot.bind_buffers(&payload, &task);
            slot.active_mask = ActiveMask(source.active_mask);
            slot.task_attrs = TaskAttrs(source.task_attrs);
            slot.total_required_subtasks = source.total_required_subtasks;
            slot.logical_block_num = source.logical_block_num;
            slot.graph_node_index = i;
            slot.task_kind = TaskKind::GRAPH_NODE;
            slot.graph_context = &execution;
            payload.tensor_count = source.tensor_count;
            payload.scalar_count = source.scalar_count;
            if (source.tensor_count < 0 || source.tensor_count > MAX_TENSOR_ARGS || source.scalar_count < 0 ||
                source.scalar_count > MAX_SCALAR_ARGS ||
                static_cast<uint32_t>(source.tensor_count) > definition.tensor_arg_count ||
                static_cast<uint32_t>(source.scalar_count) > definition.scalar_arg_count ||
                source.tensor_offset > definition.tensor_arg_count - static_cast<uint32_t>(source.tensor_count) ||
                source.scalar_offset > definition.scalar_arg_count - static_cast<uint32_t>(source.scalar_count)) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
            for (int32_t j = 0; j < source.tensor_count; ++j) {
                const uint32_t tensor_index = source.tensor_offset + static_cast<uint32_t>(j);
                ChipTensor &tensor = payload.tensors[j];
                GraphTensor rebound = definition_tensors[tensor_index];
                GraphTensorAddressPatch patch{};
                if (!graph_tensor_wire_valid(rebound)) {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
                const GraphTensorSourceRef &ref = tensor_sources[tensor_index];
                if (ref.source == static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT)) {
                    if (ref.source_index >= execution.boundary_tensor_count || ref.packed_offset != 0) {
                        execution.materialize_busy.store(0, std::memory_order_release);
                        return GraphMaterializeResult::INVALID;
                    }
                    rebound = execution.boundary_tensors[ref.source_index];
                    patch.source = static_cast<uint8_t>(GraphTensorAddressSource::BOUNDARY);
                    patch.source_index = ref.source_index;
                } else if (ref.source == static_cast<uint8_t>(GraphTensorSource::BOUNDARY_VIEW)) {
                    if (ref.source_index >= execution.boundary_tensor_count) {
                        execution.materialize_busy.store(0, std::memory_order_release);
                        return GraphMaterializeResult::INVALID;
                    }
                    const GraphTensor &boundary = execution.boundary_tensors[ref.source_index];
                    if (ref.packed_offset > UINT64_MAX - boundary.start_offset) {
                        execution.materialize_busy.store(0, std::memory_order_release);
                        return GraphMaterializeResult::INVALID;
                    }
                    rebound.buffer_addr = boundary.buffer_addr;
                    rebound.buffer_size = boundary.buffer_size;
                    rebound.owner_task_id = boundary.owner_task_id;
                    rebound.start_offset = boundary.start_offset + ref.packed_offset;
                    rebound.version = boundary.version;
                    rebound.child_memory = boundary.child_memory;
                    patch.source = static_cast<uint8_t>(GraphTensorAddressSource::BOUNDARY);
                    patch.source_index = ref.source_index;
                } else if (ref.source == static_cast<uint8_t>(GraphTensorSource::INTERNAL) ||
                           ref.source == static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT)) {
                    const bool own_output = ref.source == static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT);
                    const int32_t producer_index = own_output ? i : static_cast<int32_t>(ref.source_index);
                    if (producer_index < 0 || producer_index > i || (own_output && ref.source_index != i) ||
                        (!own_output && producer_index == i)) {
                        execution.materialize_busy.store(0, std::memory_order_release);
                        return GraphMaterializeResult::INVALID;
                    }
                    PTO2TaskDescriptor &producer = execution.node_storage[producer_index].task;
                    const uint64_t producer_bytes = static_cast<uint64_t>(nodes[producer_index].total_output_size);
                    const uintptr_t producer_base = reinterpret_cast<uintptr_t>(producer.packed_buffer_base);
                    if (ref.packed_offset > producer_bytes ||
                        rebound.buffer_size > producer_bytes - ref.packed_offset ||
                        ref.packed_offset > UINTPTR_MAX - producer_base ||
                        ref.packed_offset > UINT64_MAX - node_offsets[producer_index]) {
                        execution.materialize_busy.store(0, std::memory_order_release);
                        return GraphMaterializeResult::INVALID;
                    }
                    rebound.buffer_addr = producer_base + ref.packed_offset;
                    rebound.owner_task_id = producer.task_id.raw;
                    patch.source = static_cast<uint8_t>(GraphTensorAddressSource::INTERNAL);
                    patch.address_offset = node_offsets[producer_index] + ref.packed_offset;
                } else {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
                if (!graph_tensor_wire_valid(rebound) ||
                    execution.materialized_tensor_patches >= execution.tensor_patch_capacity) {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
                execution.tensor_patches[execution.materialized_tensor_patches++] = patch;
                graph_tensor_unpack(rebound, &tensor);
            }
            if (source.scalar_count > 0) {
                std::memcpy(
                    payload.scalars, definition_scalars + source.scalar_offset,
                    static_cast<size_t>(source.scalar_count) * sizeof(uint64_t)
                );
            }
        } else {
            for (int32_t j = 0; j < payload.tensor_count; ++j) {
                const GraphTensorAddressPatch &patch =
                    execution.tensor_patches[execution.materialized_tensor_patches++];
                if (patch.source == static_cast<uint8_t>(GraphTensorAddressSource::BOUNDARY)) {
                    payload.tensors[j].buffer.addr = execution.boundary_tensors[patch.source_index].buffer_addr;
                } else {
                    payload.tensors[j].buffer.addr = outer_base + patch.address_offset;
                }
            }
        }
        reset_graph_payload(payload);
        if (!register_initial_graph_waiter(execution, i)) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
    }
    execution.materialized_nodes = last;
    if (nodes_materialized != nullptr) *nodes_materialized = last - first;

    if (last < execution.node_count) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::PENDING;
    }

    const uint32_t expected_tensor_patches =
        affine_reuse ? execution.materialized_tensor_patch_count : definition.tensor_arg_count;
    if (execution.materialized_tensor_patches != expected_tensor_patches) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    // nodes is published before PREPARED. An activator that acquires the state
    // may therefore route saved roots without observing partially built nodes.
    execution.nodes = execution.node_storage;
    execution.materialized_graph_key = execution.graph_key;
    execution.materialized_definition_hash = execution.definition_hash;
    execution.materialized_node_count = execution.node_count;
    execution.materialized_tensor_patch_count = execution.materialized_tensor_patches;
    execution.materialized_outer_base = outer_base;
    execution.state.store(GraphExecutionState::PREPARED, std::memory_order_release);
    execution.materialize_busy.store(0, std::memory_order_release);
    return GraphMaterializeResult::PREPARED;
}
