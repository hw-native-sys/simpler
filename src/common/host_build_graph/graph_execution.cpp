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

// The storage is the tail of the outer GRAPH task's heap allocation, so its
// bytes are whatever that region last held; every field an execution needs is
// written here or by the materialize pass, never inherited from those bytes.
GraphExecution *acquire_host_execution_storage(
    uintptr_t storage_addr, size_t storage_bytes, int32_t node_count, uint32_t tensor_arg_count,
    uint32_t scalar_arg_count
) {
    GraphExecutionStorageLayout layout{};
    if (storage_addr == 0 || storage_addr % alignof(GraphNodeStorage) != 0 ||
        !graph_execution_storage_layout(node_count, tensor_arg_count, scalar_arg_count, &layout) ||
        layout.total_bytes > storage_bytes) {
        return nullptr;
    }
    auto *execution = new (reinterpret_cast<void *>(storage_addr)) GraphExecution{};
    execution->node_count = node_count;
    execution->remaining_nodes.store(node_count, std::memory_order_relaxed);
    auto *base = reinterpret_cast<uint8_t *>(execution);
    execution->node_storage = reinterpret_cast<GraphNodeStorage *>(base + layout.nodes_offset);
    execution->node_tensor_pool = reinterpret_cast<ChipTensor *>(base + layout.tensors_offset);
    execution->node_scalar_pool = reinterpret_cast<uint64_t *>(base + layout.scalars_offset);
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
    // GRAPH_MAX_SCALAR_ARGS, not MAX_SCALAR_ARGS: this counts the scalars the
    // Graph BOUNDARY carries, which the recorder sizes with
    // GraphTaskArgs = Arg<GRAPH_MAX_TENSOR_ARGS, GRAPH_MAX_SCALAR_ARGS> and the
    // image hands over as a pointer into the submission, never through a node
    // payload. MAX_SCALAR_ARGS is the per-AICore-task cap (16) and applies to
    // GraphNodeDefinition::scalar_count below, which is checked separately; using
    // it here rejected every boundary wider than one kernel call could take.
    if (definition.boundary_scalar_count > GRAPH_MAX_SCALAR_ARGS) return false;
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

bool graph_definition_hash_matches(const GraphDefinition &definition, uint32_t definition_bytes) {
    if (definition_bytes < sizeof(GraphDefinition) || definition.total_bytes != definition_bytes ||
        definition.content_hash == 0) {
        return false;
    }
    return graph_definition_content_hash(&definition, definition_bytes) == definition.content_hash;
}

// One-time integrity gate for a shared Definition object. The first localizer
// wins the UPLOADED->VERIFYING CAS, hashes the image once, and publishes
// VERIFIED/INVALID. Localizers of the other submissions sharing this object
// spin on the state word — never re-hashing, never failing while a peer is
// mid-verify (the verify is bounded by the image size, so the spin is short).
GraphDefinition *graph_definition_object_verified(GraphDefinitionHeader &header) {
    if (header.magic != GRAPH_DEFINITION_OBJECT_MAGIC) return nullptr;
    if (header.definition_bytes < sizeof(GraphDefinition)) {
        header.verify_state.store(
            static_cast<uint32_t>(GraphDefinitionVerifyState::INVALID), std::memory_order_release
        );
        return nullptr;
    }
    auto *definition = reinterpret_cast<GraphDefinition *>(&header + 1);
    uint32_t observed = header.verify_state.load(std::memory_order_acquire);
    if (observed == static_cast<uint32_t>(GraphDefinitionVerifyState::VERIFIED)) {
        return definition;
    }
    if (observed == static_cast<uint32_t>(GraphDefinitionVerifyState::INVALID)) return nullptr;
    uint32_t expected = static_cast<uint32_t>(GraphDefinitionVerifyState::UPLOADED);
    if (!header.verify_state.compare_exchange_strong(
            expected, static_cast<uint32_t>(GraphDefinitionVerifyState::VERIFYING), std::memory_order_acq_rel,
            std::memory_order_acquire
        )) {
        // A peer is verifying; wait for its verdict rather than racing it.
        while (true) {
            observed = header.verify_state.load(std::memory_order_acquire);
            if (observed == static_cast<uint32_t>(GraphDefinitionVerifyState::VERIFIED)) {
                return definition;
            }
            if (observed == static_cast<uint32_t>(GraphDefinitionVerifyState::INVALID)) return nullptr;
#if defined(__aarch64__)
            __asm__ volatile("yield");
#elif defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
    const bool matched = header.content_hash == definition->content_hash && header.full_key == definition->full_key &&
                         graph_definition_hash_matches(*definition, header.definition_bytes);
    header.verify_state.store(
        static_cast<uint32_t>(matched ? GraphDefinitionVerifyState::VERIFIED : GraphDefinitionVerifyState::INVALID),
        std::memory_order_release
    );
    return matched ? definition : nullptr;
}

// Rebind one Definition tensor template onto this execution. A BOUNDARY_* ref
// takes the invocation's boundary tensor; an INTERNAL / OWN_OUTPUT ref takes the
// producer node's materialized output base. `node_index` is the consuming node,
// which bounds a producer reference to a node that is already constructed.
// Returns false when the ref addresses no valid source — the Definition is then
// invalid, since every ref is written by the recorder from a classified source.
bool graph_rebind_tensor(
    const GraphExecution &execution, const GraphNodeDefinition *nodes, const uint64_t *node_offsets,
    const GraphTensor &tensor_template, const GraphTensorSourceRef &ref, int32_t node_index, GraphTensor *rebound_out
) {
    GraphTensor rebound = tensor_template;
    if (!graph_tensor_wire_valid(rebound)) return false;
    if (ref.source == static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT)) {
        if (ref.source_index >= execution.boundary_tensor_count || ref.packed_offset != 0) return false;
        rebound = execution.boundary_tensors[ref.source_index];
    } else if (ref.source == static_cast<uint8_t>(GraphTensorSource::BOUNDARY_VIEW)) {
        if (ref.source_index >= execution.boundary_tensor_count) return false;
        const GraphTensor &boundary = execution.boundary_tensors[ref.source_index];
        if (ref.packed_offset > UINT64_MAX - boundary.start_offset) return false;
        rebound.buffer_addr = boundary.buffer_addr;
        rebound.buffer_size = boundary.buffer_size;
        rebound.owner_task_id = boundary.owner_task_id;
        rebound.start_offset = boundary.start_offset + ref.packed_offset;
        rebound.version = boundary.version;
        rebound.address_space = boundary.address_space;
    } else if (ref.source == static_cast<uint8_t>(GraphTensorSource::INTERNAL) ||
               ref.source == static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT)) {
        const bool own_output = ref.source == static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT);
        const int32_t producer_index = own_output ? node_index : static_cast<int32_t>(ref.source_index);
        if (producer_index < 0 || producer_index > node_index || (own_output && ref.source_index != node_index) ||
            (!own_output && producer_index == node_index)) {
            return false;
        }
        PTO2TaskDescriptor &producer = execution.node_at(producer_index).task;
        const uint64_t producer_bytes = static_cast<uint64_t>(nodes[producer_index].total_output_size);
        const uintptr_t producer_base = reinterpret_cast<uintptr_t>(producer.packed_buffer_base);
        if (ref.packed_offset > producer_bytes || rebound.buffer_size > producer_bytes - ref.packed_offset ||
            ref.packed_offset > UINTPTR_MAX - producer_base ||
            ref.packed_offset > UINT64_MAX - node_offsets[producer_index]) {
            return false;
        }
        rebound.buffer_addr = producer_base + ref.packed_offset;
        rebound.owner_task_id = producer.task_id.raw;
    } else {
        return false;
    }
    if (!graph_tensor_wire_valid(rebound)) return false;
    *rebound_out = rebound;
    return true;
}

// Turn a Definition predicate plus its rebound operand tensor into the address
// the scheduler reads at the dispatch point. start_offset and elem_offset are
// element counts, so the byte offset is their sum scaled by the element width —
// the same arithmetic the ordinary submit path runs on ChipTensor.
// The Definition crossed the host boundary, so every field it contributes is
// range-checked here: pass() memcpys elem_size bytes into an int64_t, and the
// address must land inside the operand's own buffer.
bool graph_predicate_resolve(
    const GraphTensor &operand, const GraphPredicate &predicate, DispatchPredicate *resolved_out
) {
    // pass() treats an operator it does not recognize as "always dispatch", so an
    // unknown code from the image must not reach it. Enumerating the operators
    // without a default makes a newly added one a build warning here rather than
    // a silent pass.
    bool operator_known = false;
    switch (static_cast<PredicateOp>(predicate.op)) {
    case PredicateOp::EQ:
    case PredicateOp::NE:
    case PredicateOp::GT:
    case PredicateOp::LT:
    case PredicateOp::GE:
    case PredicateOp::LE:
        operator_known = true;
        break;
    case PredicateOp::NONE:
        break;
    }
    if (!operator_known) return false;
    const uint64_t element_size = get_element_size(static_cast<DataType>(operand.dtype));
    if (element_size != 1 && element_size != 2 && element_size != 4 && element_size != 8) return false;
    if (predicate.elem_size != element_size || predicate.elem_offset >= operand.extent_elem) return false;
    // graph_tensor_wire_valid bounds start_offset + extent_elem by the buffer's
    // element count, so the scaled sum cannot leave the buffer.
    const uint64_t byte_offset = (operand.start_offset + predicate.elem_offset) * element_size;

    resolved_out->addr = operand.buffer_addr + byte_offset;
    resolved_out->target = predicate.target;
    resolved_out->elem_size = predicate.elem_size;
    resolved_out->op = static_cast<PredicateOp>(predicate.op);
    return true;
}

}  // namespace

GraphExecution *graph_execution_localize(PTO2TaskSlotState &outer_slot) {
    GraphSubmission *submission = graph_submission_from_slot(outer_slot);
    if (submission == nullptr) return nullptr;
    if (GraphExecution *existing = graph_submission_local_execution(*submission)) return existing;
    if (graph_submission_execution_initializing(*submission)) return nullptr;

    // The Definition lives in its own shared GM object; only its header is
    // inspected here. The content hash gate runs inside the submission CAS
    // below, so exactly one localizer verifies a shared object — concurrent
    // localizers of other submissions see INITIALIZING and retry as BUSY
    // rather than racing the verify state.
    if (submission->definition_addr == 0 || submission->definition_addr % alignof(GraphDefinitionHeader) != 0) {
        return nullptr;
    }
    auto *definition_header =
        reinterpret_cast<GraphDefinitionHeader *>(static_cast<uintptr_t>(submission->definition_addr));
    if (definition_header->magic != GRAPH_DEFINITION_OBJECT_MAGIC) return nullptr;
    const GraphTensor *boundary_tensors = graph_submission_tensors(*submission);
    const uint64_t *boundary_scalars = graph_submission_scalars(*submission);
    const size_t boundary_tensor_end = static_cast<size_t>(submission->tensors_offset) +
                                       static_cast<size_t>(submission->tensor_count) * sizeof(GraphTensor);
    if (boundary_tensors == nullptr || outer_slot.task == nullptr || outer_slot.task->packed_buffer_base == nullptr ||
        outer_slot.task->packed_buffer_end == nullptr ||
        (submission->scalar_count != 0 && boundary_scalars == nullptr) ||
        (submission->scalar_count != 0 && submission->scalars_offset < boundary_tensor_end)) {
        return nullptr;
    }
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

    const GraphDefinition *definition = graph_definition_object_verified(*definition_header);
    if (definition == nullptr || definition->total_bytes == 0 || definition->task_count == 0 ||
        definition->task_count > GRAPH_MAX_NODES || submission->definition_hash != definition->content_hash ||
        submission->graph_key != definition->full_key || submission->tensor_count != definition->boundary_count ||
        submission->scalar_count != definition->boundary_scalar_count) {
        __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
        return nullptr;
    }
    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_base);
    const uintptr_t outer_end = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_end);
    // The outer task's allocation covers the nodes' packed outputs followed by
    // this execution's storage, so the span must hold both and the execution
    // starts past required_heap.
    const uint64_t owned_bytes = definition->required_heap + static_cast<uint64_t>(definition->execution_storage_bytes);
    if (outer_end < outer_base || definition->execution_storage_bytes == 0 ||
        definition->required_heap > UINT64_MAX - definition->execution_storage_bytes ||
        owned_bytes > outer_end - outer_base) {
        __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
        return nullptr;
    }

    GraphExecution *execution = acquire_host_execution_storage(
        outer_base + definition->required_heap, definition->execution_storage_bytes,
        static_cast<int32_t>(definition->task_count), definition->tensor_arg_count, definition->scalar_arg_count
    );
    if (execution == nullptr) {
        __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
        return nullptr;
    }

    // The Definition is read in place from the shared GM object; no
    // per-occurrence copy is taken.
    execution->definition = definition;
    if (!bind_graph_topology(*execution)) {
        execution->retired_nodes.store(execution->node_count, std::memory_order_relaxed);
        execution->state.store(GraphExecutionState::COMPLETED, std::memory_order_release);
        __atomic_store_n(&submission->local_execution, 0, __ATOMIC_RELEASE);
        return nullptr;
    }
    execution->boundary_tensors = boundary_tensors;
    execution->boundary_tensor_count = submission->tensor_count;
    execution->boundary_scalars = boundary_scalars;
    execution->boundary_scalar_count = submission->scalar_count;
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
        execution.node_storage == nullptr) {
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
        // Incremental activation reads producer slots through execution.nodes
        // while the graph is still materializing, so publish the storage base
        // once, before the first range. Topological node order guarantees every
        // producer index a materialized node references is already constructed,
        // and materialize_busy serializes this with any concurrent slice.
        execution.nodes = execution.node_storage;
    } else if (state != GraphExecutionState::MATERIALIZING) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    const GraphDefinition &definition = *execution.definition;
    const GraphNodeDefinition *nodes =
        graph_definition_array<GraphNodeDefinition>(definition, definition.off_nodes, definition.task_count);
    const uint64_t *node_offsets =
        graph_definition_array<uint64_t>(definition, definition.off_node_offsets, definition.task_count);
    const GraphTensor *definition_tensors =
        definition.tensor_arg_count == 0 ?
            nullptr :
            graph_definition_array<GraphTensor>(definition, definition.off_tensors, definition.tensor_arg_count);
    const GraphTensorSourceRef *tensor_sources =
        definition.tensor_arg_count == 0 ? nullptr :
                                           graph_definition_array<GraphTensorSourceRef>(
                                               definition, definition.off_tensor_sources, definition.tensor_arg_count
                                           );
    const uint64_t *definition_scalars =
        definition.scalar_arg_count == 0 ?
            nullptr :
            graph_definition_array<uint64_t>(definition, definition.off_scalars, definition.scalar_arg_count);
    const GraphScalarSourceRef *scalar_sources =
        definition.scalar_arg_count == 0 ? nullptr :
                                           graph_definition_array<GraphScalarSourceRef>(
                                               definition, definition.off_scalar_sources, definition.scalar_arg_count
                                           );
    const GraphPredicate *predicates =
        definition.predicate_count == 0 ?
            nullptr :
            graph_definition_array<GraphPredicate>(definition, definition.off_predicates, definition.predicate_count);
    if (nodes == nullptr || node_offsets == nullptr ||
        (definition.tensor_arg_count != 0 && (definition_tensors == nullptr || tensor_sources == nullptr)) ||
        (definition.scalar_arg_count != 0 && (definition_scalars == nullptr || scalar_sources == nullptr)) ||
        (definition.predicate_count != 0 && predicates == nullptr)) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    const int32_t first = execution.materialized_nodes;
    const int32_t last = std::min(execution.node_count, first + max_nodes);
    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.task->packed_buffer_base);
    for (int32_t i = first; i < last; ++i) {
        GraphNodeStorage *storage = &execution.node_at(i);
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
        const GraphNodeDefinition &source = nodes[i];
        const uint64_t node_offset = node_offsets[i];
        const uint64_t output_bytes = PTO2_ALIGN_UP(static_cast<uint64_t>(source.total_output_size), PTO2_ALIGN_SIZE);
        for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
            task.kernel_id[k] = source.kernel_id[k];
        task.packed_buffer_base = reinterpret_cast<void *>(outer_base + node_offset);
        task.packed_buffer_end = reinterpret_cast<void *>(outer_base + node_offset + output_bytes);

        slot.reset_for_reuse();
        slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
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
        payload.dump_metadata = source.dump_metadata;
        if (source.tensor_count < 0 || source.tensor_count > MAX_TENSOR_ARGS || source.scalar_count < 0 ||
            source.scalar_count > MAX_SCALAR_ARGS ||
            static_cast<uint32_t>(source.tensor_count) > definition.tensor_arg_count ||
            static_cast<uint32_t>(source.scalar_count) > definition.scalar_arg_count ||
            source.tensor_offset > definition.tensor_arg_count - static_cast<uint32_t>(source.tensor_count) ||
            source.scalar_offset > definition.scalar_arg_count - static_cast<uint32_t>(source.scalar_count)) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
        // A node's arguments occupy the same span in this execution's pools as in the
        // Definition's arg tables, so the region starts at the node's own offset. No
        // fanin region: its dependencies come from the Definition's CSR, and
        // reset_graph_payload below keeps fanin_count at 0.
        payload.bind_regions(
            execution.node_tensor_pool + source.tensor_offset, execution.node_scalar_pool + source.scalar_offset,
            nullptr
        );
        ChipTensor *node_tensors = payload.tensor_data();
        for (int32_t j = 0; j < source.tensor_count; ++j) {
            const uint32_t tensor_index = source.tensor_offset + static_cast<uint32_t>(j);
            GraphTensor rebound;
            if (!graph_rebind_tensor(
                    execution, nodes, node_offsets, definition_tensors[tensor_index], tensor_sources[tensor_index], i,
                    &rebound
                )) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
            execution.consumed_tensor_args++;
            graph_tensor_unpack(rebound, &node_tensors[j]);
        }
        uint64_t *node_scalars = payload.scalar_data();
        for (int32_t j = 0; j < source.scalar_count; ++j) {
            const uint32_t scalar_index = source.scalar_offset + static_cast<uint32_t>(j);
            const GraphScalarSourceRef &ref = scalar_sources[scalar_index];
            if (ref.source == static_cast<uint8_t>(GraphScalarSource::STATIC_VALUE)) {
                node_scalars[j] = definition_scalars[scalar_index];
            } else if (ref.source == static_cast<uint8_t>(GraphScalarSource::BOUNDARY)) {
                if (ref.source_index >= execution.boundary_scalar_count || execution.boundary_scalars == nullptr) {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
                node_scalars[j] = execution.boundary_scalars[ref.source_index];
            } else {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
        }
        reset_graph_payload(payload);
        // The attribute bit and the predicate slot are written together by the
        // recorder. A Definition where they disagree would either route the node
        // through a predicate the scheduler never reads, or leave a resolved
        // predicate that no dispatch consults.
        if (slot.task_attrs.has_predicate() != (source.predicate_slot != 0)) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
        // Resolved after the reset, which clears the predicate every node starts from.
        if (source.predicate_slot != 0) {
            const uint32_t predicate_index = static_cast<uint32_t>(source.predicate_slot) - 1;
            GraphTensor operand;
            // OWN_OUTPUT is a valid source for a tensor arg but never for an
            // operand: it would bind the predicate to the buffer this node has
            // yet to write, so the dispatch decision would read whatever the heap
            // last held. The recorder refuses it; so does the image reader.
            if (predicate_index >= definition.predicate_count ||
                predicates[predicate_index].operand_source.source ==
                    static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT) ||
                !graph_rebind_tensor(
                    execution, nodes, node_offsets, predicates[predicate_index].operand,
                    predicates[predicate_index].operand_source, i, &operand
                ) ||
                !graph_predicate_resolve(operand, predicates[predicate_index], &payload.predicate)) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
        }
    }
    execution.materialized_nodes = last;
    if (nodes_materialized != nullptr) *nodes_materialized = last - first;

    if (last < execution.node_count) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::PENDING;
    }

    // Every node's [tensor_offset, tensor_offset + tensor_count) range is bounds-
    // checked on its own. This total additionally requires the ranges to account
    // for the whole tensor array, rejecting a Definition that under- or
    // over-consumes it.
    if (execution.consumed_tensor_args != definition.tensor_arg_count) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    execution.state.store(GraphExecutionState::PREPARED, std::memory_order_release);
    execution.materialize_busy.store(0, std::memory_order_release);
    return GraphMaterializeResult::PREPARED;
}
