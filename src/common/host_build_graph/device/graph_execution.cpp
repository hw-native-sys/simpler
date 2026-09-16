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

#include "host_build_graph/task_id.h"

namespace {

GraphExecution *acquire_execution_storage(
    uintptr_t storage_addr, size_t storage_bytes, int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count
) {
    GraphExecutionStorageLayout layout{};
    // ChipTaskStorage, not GraphExecution: the in-graph task array's alignment is the widest
    // the storage carries, and tasks_offset only rounds up relative to this base, so
    // an under-aligned base would leave every alignas(64) in-graph task entry misaligned.
    if (storage_addr == 0 || storage_addr % alignof(ChipTaskStorage) != 0 ||
        !graph_execution_storage_layout(task_count, tensor_arg_count, scalar_arg_count, &layout) ||
        layout.total_bytes > storage_bytes) {
        return nullptr;
    }
    auto *execution = new (reinterpret_cast<void *>(storage_addr)) GraphExecution{};
    execution->task_count = task_count;
    execution->remaining_tasks.store(task_count, std::memory_order_relaxed);
    auto *base = reinterpret_cast<uint8_t *>(execution);
    execution->task_storage = reinterpret_cast<ChipTaskStorage *>(base + layout.tasks_offset);
    execution->task_tensor_pool = reinterpret_cast<simpler::hbg::Tensor *>(base + layout.tensors_offset);
    execution->task_scalar_pool = reinterpret_cast<uint64_t *>(base + layout.scalars_offset);
    execution->task_states = reinterpret_cast<std::atomic<ChipTaskState> *>(base + layout.states_offset);
    return execution;
}

void reset_graph_payload(TaskPayload &payload) {
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; ++w) {
        payload.staged_core_mask[w].store(0, std::memory_order_relaxed);
    }
    payload.published_block_count.store(0, std::memory_order_relaxed);
    payload.early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload.running_slot_count.store(0, std::memory_order_relaxed);
    payload.early_sync_drain_state.store(EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

bool bind_graph_topology(GraphExecution &execution) {
    if (execution.definition == nullptr) return false;
    const GraphDefinition &definition = *execution.definition;
    // Re-checked rather than inherited from graph_execution_localize: every section
    // below is fetched with task_count or task_count + 1, so a wire value outside this
    // range overflows the increment before any bound check can see it.
    if (definition.task_count <= 0 || definition.task_count > MAX_IN_GRAPH_TASKS) return false;
    // GRAPH_MAX_SCALAR_ARGS, not MAX_SCALAR_ARGS: this counts the scalars the
    // Graph BOUNDARY carries, which the recorder sizes with GraphTaskArgs and the
    // outer Graph payload hands it to GraphExecution, never through an in-graph task
    // payload. MAX_SCALAR_ARGS is the per-AICore-task cap (16) and applies to
    // InGraphTaskDefinition::scalar_count below, which is checked separately; using
    // it here rejected every boundary wider than one kernel call could take.
    if (definition.boundary_scalar_count > GRAPH_MAX_SCALAR_ARGS) return false;
    const int32_t *fanin_offsets =
        graph_definition_array<int32_t>(definition, definition.off_fanin_offsets, definition.task_count + 1);
    const uint16_t *fanin_indices =
        definition.edge_count == 0 ?
            nullptr :
            graph_definition_array<uint16_t>(definition, definition.off_fanin_indices, definition.edge_count);
    const int32_t *fanout_offsets =
        graph_definition_array<int32_t>(definition, definition.off_fanout_offsets, definition.task_count + 1);
    const uint16_t *fanout_indices =
        definition.edge_count == 0 ?
            nullptr :
            graph_definition_array<uint16_t>(definition, definition.off_fanout_indices, definition.edge_count);
    const uint16_t *roots =
        graph_definition_array<uint16_t>(definition, definition.off_root_indices, definition.root_count);
    const InGraphTaskDefinition *tasks =
        graph_definition_array<InGraphTaskDefinition>(definition, definition.off_in_graph_tasks, definition.task_count);
    const uint64_t *in_graph_task_offsets =
        graph_definition_array<uint64_t>(definition, definition.off_in_graph_task_offsets, definition.task_count);
    if (fanin_offsets == nullptr || fanout_offsets == nullptr || roots == nullptr || tasks == nullptr ||
        in_graph_task_offsets == nullptr ||
        (definition.edge_count != 0 && (fanin_indices == nullptr || fanout_indices == nullptr)) ||
        fanin_offsets[0] != 0 || fanout_offsets[0] != 0 ||
        fanin_offsets[definition.task_count] != definition.edge_count ||
        fanout_offsets[definition.task_count] != definition.edge_count) {
        return false;
    }

    uint64_t required_heap = 0;
    constexpr uint8_t VALID_ACTIVE_MASK = (1U << SUBTASK_SLOT_COUNT) - 1U;
    for (int32_t i = 0; i < definition.task_count; ++i) {
        const InGraphTaskDefinition &task = tasks[i];
        if (in_graph_task_offsets[i] != required_heap || task.total_output_size < 0 || task.tensor_count < 0 ||
            task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 || task.scalar_count > MAX_SCALAR_ARGS ||
            // Negative before the span tests, which cannot see it on their own: a
            // negative offset is below tensor_arg_count and *widens* the remaining
            // span it is subtracted from, so both comparisons below pass.
            task.tensor_offset < 0 || task.scalar_offset < 0 || task.tensor_offset > definition.tensor_arg_count ||
            task.tensor_count > definition.tensor_arg_count - task.tensor_offset ||
            task.scalar_offset > definition.scalar_arg_count ||
            task.scalar_count > definition.scalar_arg_count - task.scalar_offset ||
            (task.active_mask & ~VALID_ACTIVE_MASK) != 0 ||
            (task.ed_flags & ~(ED_FLAG_CANDIDATE | ED_FLAG_TRACKED)) != 0 || task.reserved != 0 ||
            task.logical_block_num <= 0 || task.total_required_subtasks < 0) {
            return false;
        }
        for (int32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            const bool active = (task.active_mask & (1U << slot)) != 0;
            if (active != (task.kernel_id[slot] != INVALID_KERNEL_ID)) return false;
        }
        const uint64_t output_bytes = CHIP_ALIGN_UP(static_cast<uint64_t>(task.total_output_size), CHIP_ALIGN_SIZE);
        if (output_bytes > definition.required_heap - required_heap) return false;
        required_heap += output_bytes;
    }
    if (required_heap != definition.required_heap) return false;

    int32_t observed_roots = 0;
    for (int32_t consumer = 0; consumer < definition.task_count; ++consumer) {
        const int32_t begin = fanin_offsets[consumer];
        const int32_t end = fanin_offsets[consumer + 1];
        if (begin > end || end > definition.edge_count) return false;
        if (begin == end) observed_roots++;
        // ED_FLAG_CANDIDATE steers dispatch from materialization onward, so the
        // image must carry the whole conjunction the recorder decided it by, not
        // merely a known bit: a candidate with no producer has nothing to bet on,
        // a DUMMY one would index early_dispatch_queues[] one past its last
        // shape, and a predicated one would be released before its predicate is
        // ever tested. graph_fill_definition is the only writer of this field and
        // holds all three, so a violation means the image is not one it produced.
        if ((tasks[consumer].ed_flags & ED_FLAG_CANDIDATE) != 0 &&
            (begin == end || tasks[consumer].predicate_slot != 0 ||
             ActiveMask(tasks[consumer].active_mask).to_shape() == ResourceShape::DUMMY)) {
            return false;
        }
        for (int32_t edge = begin; edge < end; ++edge) {
            if (fanin_indices[edge] >= consumer) return false;
        }
    }
    if (observed_roots != definition.root_count) return false;
    for (int32_t i = 0; i < definition.root_count; ++i) {
        const uint16_t root = roots[i];
        if (root >= definition.task_count || fanin_offsets[root] != fanin_offsets[root + 1]) return false;
    }
    for (int32_t producer = 0; producer < definition.task_count; ++producer) {
        const int32_t begin = fanout_offsets[producer];
        const int32_t end = fanout_offsets[producer + 1];
        if (begin > end || end > definition.edge_count) return false;
        for (int32_t edge = begin; edge < end; ++edge) {
            if (fanout_indices[edge] <= producer || fanout_indices[edge] >= definition.task_count) return false;
        }
    }

    execution.fanin_offsets = fanin_offsets;
    execution.fanin_indices = fanin_indices;
    return true;
}

// Framing gate for a shared Definition object: the header must agree with the image
// behind it before any of the image's section offsets is read. Framing, not
// integrity — every section is fetched through graph_definition_array, which bounds
// its offset, alignment and extent against total_bytes, and bind_graph_topology
// below walks the whole edge list; neither trusts a field this returns.
//
// definition_bytes is what the upload recorded for this object, so comparing it
// against the image's own total_bytes is what rejects a header framing a region
// that does not hold the Definition it claims.
GraphDefinition *graph_definition_object_framed(GraphDefinitionHeader &header) {
    if (header.magic != GRAPH_DEFINITION_OBJECT_MAGIC) return nullptr;
    if (header.definition_bytes < sizeof(GraphDefinition)) return nullptr;
    auto *definition = reinterpret_cast<GraphDefinition *>(&header + 1);
    if (definition->total_bytes != header.definition_bytes) return nullptr;
    if (definition->full_key != header.full_key) return nullptr;
    return definition;
}

// Rebind one Definition tensor template onto this execution. A BOUNDARY_* ref
// takes the invocation's boundary tensor; an INTERNAL / OWN_OUTPUT ref takes the
// producer in-graph task's materialized output base. `task_index` is the consuming
// task, which bounds a producer reference to a task that is already constructed.
// Returns false when the ref addresses no valid source — the Definition is then
// invalid, since every ref is written by the recorder from a classified source.
bool graph_rebind_tensor(
    const GraphExecution &execution, const InGraphTaskDefinition *tasks, const uint64_t *in_graph_task_offsets,
    const GraphTensor &tensor_template, const GraphTensorSourceRef &ref, int32_t task_index, GraphTensor *rebound_out
) {
    GraphTensor rebound = tensor_template;
    if (!graph_tensor_wire_valid(rebound)) return false;
    if (ref.source_kind == static_cast<uint8_t>(GraphTensorSourceKind::BOUNDARY_EXACT)) {
        if (ref.source_index >= execution.boundary_tensor_count || ref.packed_offset != 0) return false;
        rebound = execution.boundary_tensors[ref.source_index];
    } else if (ref.source_kind == static_cast<uint8_t>(GraphTensorSourceKind::BOUNDARY_VIEW)) {
        if (ref.source_index >= execution.boundary_tensor_count) return false;
        const GraphTensor &boundary = execution.boundary_tensors[ref.source_index];
        if (ref.packed_offset > UINT64_MAX - boundary.start_offset) return false;
        rebound.buffer_addr = boundary.buffer_addr;
        rebound.buffer_size = boundary.buffer_size;
        rebound.owner_task_id = boundary.owner_task_id;
        rebound.start_offset = boundary.start_offset + ref.packed_offset;
        rebound.version = boundary.version;
        rebound.address_space = boundary.address_space;
    } else if (ref.source_kind == static_cast<uint8_t>(GraphTensorSourceKind::INTERNAL) ||
               ref.source_kind == static_cast<uint8_t>(GraphTensorSourceKind::OWN_OUTPUT)) {
        const bool own_output = ref.source_kind == static_cast<uint8_t>(GraphTensorSourceKind::OWN_OUTPUT);
        const int32_t producer_index = own_output ? task_index : static_cast<int32_t>(ref.source_index);
        if (producer_index < 0 || producer_index > task_index || (own_output && ref.source_index != task_index) ||
            (!own_output && producer_index == task_index)) {
            return false;
        }
        TaskDescriptor &producer = execution.task_at(producer_index).task;
        const uint64_t producer_bytes = static_cast<uint64_t>(tasks[producer_index].total_output_size);
        const uintptr_t producer_base = reinterpret_cast<uintptr_t>(producer.packed_buffer_base);
        if (ref.packed_offset > producer_bytes || rebound.buffer_size > producer_bytes - ref.packed_offset ||
            ref.packed_offset > UINTPTR_MAX - producer_base ||
            ref.packed_offset > UINT64_MAX - in_graph_task_offsets[producer_index]) {
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
// the same arithmetic the ordinary submit path runs on simpler::hbg::Tensor.
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

GraphExecution *graph_execution_localize(ChipTaskSlotState &outer_slot) {
    if (outer_slot.task_kind != TaskKind::GRAPH || outer_slot.to_descriptor().packed_buffer_base == nullptr ||
        outer_slot.to_descriptor().packed_buffer_end == nullptr || outer_slot.graph_context == nullptr) {
        return nullptr;
    }

    const uintptr_t definition_addr = reinterpret_cast<uintptr_t>(outer_slot.graph_context);
    if (definition_addr < sizeof(GraphDefinitionHeader) ||
        (definition_addr - sizeof(GraphDefinitionHeader)) % alignof(GraphDefinitionHeader) != 0) {
        return nullptr;
    }
    auto *definition_header =
        reinterpret_cast<GraphDefinitionHeader *>(definition_addr - sizeof(GraphDefinitionHeader));
    const GraphDefinition *definition = graph_definition_object_framed(*definition_header);
    TaskPayload &payload = outer_slot.to_payload();
    if (definition == nullptr || definition->total_bytes == 0 || definition->task_count <= 0 ||
        definition->task_count > MAX_IN_GRAPH_TASKS || payload.tensor_count != definition->boundary_count ||
        payload.scalar_count != definition->boundary_scalar_count ||
        (payload.tensor_count != 0 && payload.tensor_data() == nullptr) ||
        (payload.scalar_count != 0 && payload.scalar_data() == nullptr)) {
        return nullptr;
    }

    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.to_descriptor().packed_buffer_base);
    const uintptr_t outer_end = reinterpret_cast<uintptr_t>(outer_slot.to_descriptor().packed_buffer_end);
    if (outer_end < outer_base || definition->required_heap > UINTPTR_MAX - outer_base ||
        definition->execution_storage_bytes > outer_end - outer_base ||
        definition->required_heap > outer_end - outer_base - definition->execution_storage_bytes) {
        return nullptr;
    }
    GraphExecution *execution = acquire_execution_storage(
        outer_base + definition->required_heap, definition->execution_storage_bytes, definition->task_count,
        definition->tensor_arg_count, definition->scalar_arg_count
    );
    if (execution == nullptr) return nullptr;

    execution->definition = definition;
    execution->outer_slot = &outer_slot;
    execution->boundary_tensors = reinterpret_cast<const GraphTensor *>(payload.tensor_data());
    execution->boundary_tensor_count = payload.tensor_count;
    execution->boundary_scalars = payload.scalar_data();
    execution->boundary_scalar_count = payload.scalar_count;
    if (!bind_graph_topology(*execution)) {
        execution->retired_tasks.store(execution->task_count, std::memory_order_relaxed);
        graph_execution_mark_completed(*execution);
        return nullptr;
    }
    outer_slot.graph_context = execution;
    return execution;
}

GraphMaterializeResult graph_execution_materialize_slice(
    ChipTaskSlotState &outer_slot, GraphExecution &execution, int32_t max_tasks, int32_t *tasks_materialized
) {
    if (tasks_materialized != nullptr) *tasks_materialized = 0;
    if (outer_slot.task_kind != TaskKind::GRAPH || outer_slot.to_descriptor().packed_buffer_base == nullptr ||
        max_tasks <= 0 || execution.definition == nullptr || execution.task_storage == nullptr) {
        return GraphMaterializeResult::INVALID;
    }

    GraphExecutionState state = graph_execution_state(execution);
    if (state >= GraphExecutionState::PREPARED) return GraphMaterializeResult::PREPARED;

    uint8_t expected_busy = 0;
    if (!execution.materialize_busy.compare_exchange_strong(
            expected_busy, 1, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return GraphMaterializeResult::BUSY;
    }

    state = graph_execution_state(execution);
    if (state == GraphExecutionState::SUBMITTED) {
        if (!graph_execution_transition(
                execution, GraphExecutionState::SUBMITTED, GraphExecutionState::MATERIALIZING
            )) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::BUSY;
        }
        // Incremental activation reads producer slots through execution.tasks
        // while the graph is still materializing, so publish the storage base
        // once, before the first range. Topological task order guarantees every
        // producer index a materialized task references is already constructed,
        // and materialize_busy serializes this with any concurrent slice.
        execution.tasks = execution.task_storage;
    } else if (state != GraphExecutionState::MATERIALIZING) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    const GraphDefinition &definition = *execution.definition;
    const InGraphTaskDefinition *tasks =
        graph_definition_array<InGraphTaskDefinition>(definition, definition.off_in_graph_tasks, definition.task_count);
    const uint64_t *in_graph_task_offsets =
        graph_definition_array<uint64_t>(definition, definition.off_in_graph_task_offsets, definition.task_count);
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
    const GraphScalarInheritance *scalar_inheritance =
        definition.scalar_arg_count == 0 ?
            nullptr :
            graph_definition_array<GraphScalarInheritance>(
                definition, definition.off_scalar_inheritance, definition.scalar_arg_count
            );
    const GraphPredicate *predicates =
        definition.predicate_count == 0 ?
            nullptr :
            graph_definition_array<GraphPredicate>(definition, definition.off_predicates, definition.predicate_count);
    if (tasks == nullptr || in_graph_task_offsets == nullptr ||
        (definition.tensor_arg_count != 0 && (definition_tensors == nullptr || tensor_sources == nullptr)) ||
        (definition.scalar_arg_count != 0 && (definition_scalars == nullptr || scalar_inheritance == nullptr)) ||
        (definition.predicate_count != 0 && predicates == nullptr)) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    const int32_t first = execution.materialized_tasks;
    const int32_t last = std::min(execution.task_count, first + max_tasks);
    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.to_descriptor().packed_buffer_base);
    // stage_graph_roots_early is the only path that gives a body root a staging
    // claim, and it runs only when the shell itself is released early, so under
    // a shell the host did not qualify no root can ever be staged. The shell's
    // verdict is written by the host before upload and never changes, so this is
    // loop-invariant for the whole execution.
    const bool shell_stages_roots = (outer_slot.ed_flags & ED_FLAG_CANDIDATE) != 0;
    for (int32_t i = first; i < last; ++i) {
        ChipTaskStorage *storage = &execution.task_at(i);
        if (i >= execution.constructed_tasks) {
            storage = new (storage) ChipTaskStorage;
            execution.constructed_tasks++;
        }
        TaskDescriptor &task = storage->task;
        TaskPayload &payload = storage->payload;
        ChipTaskSlotState &slot = storage->slot;

        task.task_id = TaskId::make_in_graph(outer_slot.to_descriptor().task_id.local_id(), i);
        const InGraphTaskDefinition &source = tasks[i];
        const uint64_t task_offset = in_graph_task_offsets[i];
        const uint64_t output_bytes = CHIP_ALIGN_UP(static_cast<uint64_t>(source.total_output_size), CHIP_ALIGN_SIZE);
        for (int k = 0; k < SUBTASK_SLOT_COUNT; ++k)
            task.kernel_id[k] = source.kernel_id[k];
        task.packed_buffer_base = reinterpret_cast<void *>(outer_base + task_offset);
        task.packed_buffer_end = reinterpret_cast<void *>(outer_base + task_offset + output_bytes);

        slot.reset_for_reuse();
        // The readiness a consumer polls, cleared before this task becomes
        // visible to one: materialization publishes tasks incrementally, so a
        // peer may scan this index as soon as published_tasks passes it.
        execution.reset_task_state(i);
        slot.active_mask = ActiveMask(source.active_mask);
        slot.task_attrs = TaskAttrs(source.task_attrs);
        // Recording decided these once for the whole body; every execution of the
        // same Definition qualifies the same tasks, so materialization only
        // replays the verdict.
        slot.ed_flags = source.ed_flags;
        slot.total_required_subtasks = source.total_required_subtasks;
        slot.logical_block_num = source.logical_block_num;
        slot.in_graph_local_id = i;
        // A task in a Graph body is an ordinary leaf, classified by the same rule as
        // one submitted outside a Graph. Its membership is carried by graph_context.
        slot.task_kind = slot.active_mask.is_dummy() ? TaskKind::DUMMY : TaskKind::KERNEL;
        // A root carries no recorded verdict — qualification needs a producer to
        // bet on and a root has none inside the body — but an early-released
        // shell stages roots on the body's behalf, and push_ready_routed reads
        // this flag to decide whether a task may hold a staging claim. The two
        // per-task terms are the ones the recorded conjunction applies to the
        // task itself, and both are load-bearing rather than defensive: a DUMMY
        // task has no dispatchable shape to index a per-shape queue with, and a
        // predicated task must reach the predicate test in push_ready_routed,
        // which an early release returns before. The shell term keeps the flag
        // off a root nothing can stage, which would otherwise pay a seq_cst CAS
        // on every route for a claim it can never hold.
        //
        // Deciding it here rather than at staging time is what makes it safe:
        // materialization owns this slot exclusively and runs strictly before
        // any path can route the root, so the flag is never written beside a
        // reader.
        const bool root_stageable = shell_stages_roots &&
                                    execution.fanin_offsets[i] == execution.fanin_offsets[i + 1] &&
                                    !slot.task_attrs.has_predicate() && slot.task_kind != TaskKind::DUMMY;
        if (root_stageable) slot.ed_flags |= ED_FLAG_CANDIDATE;
        slot.graph_context = &execution;
        payload.tensor_count = source.tensor_count;
        payload.scalar_count = source.scalar_count;
        payload.dump_metadata = source.dump_metadata;
        if (source.tensor_count < 0 || source.tensor_count > MAX_TENSOR_ARGS || source.scalar_count < 0 ||
            source.scalar_count > MAX_SCALAR_ARGS || source.tensor_offset < 0 || source.scalar_offset < 0 ||
            source.tensor_count > definition.tensor_arg_count || source.scalar_count > definition.scalar_arg_count ||
            source.tensor_offset > definition.tensor_arg_count - source.tensor_count ||
            source.scalar_offset > definition.scalar_arg_count - source.scalar_count) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
        // A task's arguments occupy the same span in this execution's pools as in the
        // Definition's arg tables, so the region starts at the task's own offset. No
        // fanin region: its dependencies come from the Definition's CSR, and
        // reset_graph_payload below keeps fanin_count at 0.
        payload.bind_regions(
            execution.task_tensor_pool + source.tensor_offset, execution.task_scalar_pool + source.scalar_offset,
            nullptr
        );
        simpler::hbg::Tensor *task_tensors = payload.tensor_data();
        for (int32_t j = 0; j < source.tensor_count; ++j) {
            const int32_t tensor_index = source.tensor_offset + j;
            GraphTensor rebound;
            if (!graph_rebind_tensor(
                    execution, tasks, in_graph_task_offsets, definition_tensors[tensor_index],
                    tensor_sources[tensor_index], i, &rebound
                )) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
            execution.consumed_tensor_args++;
            graph_tensor_unpack(rebound, &task_tensors[j]);
        }
        uint64_t *task_scalars = payload.scalar_data();
        for (int32_t j = 0; j < source.scalar_count; ++j) {
            const int32_t scalar_index = source.scalar_offset + j;
            const GraphScalarInheritance &ref = scalar_inheritance[scalar_index];
            if (!ref.inherited()) {
                task_scalars[j] = definition_scalars[scalar_index];
            } else {
                if (ref.boundary_index() >= execution.boundary_scalar_count || execution.boundary_scalars == nullptr) {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
                task_scalars[j] = execution.boundary_scalars[ref.boundary_index()];
            }
        }
        reset_graph_payload(payload);
        // The attribute bit and the predicate slot are written together by the
        // recorder. A Definition where they disagree would either route the task
        // through a predicate the scheduler never reads, or leave a resolved
        // predicate that no dispatch consults.
        if (slot.task_attrs.has_predicate() != (source.predicate_slot != 0)) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
        // Resolved after the reset, which clears the predicate every task starts from.
        if (source.predicate_slot != 0) {
            const int32_t predicate_index = static_cast<int32_t>(source.predicate_slot) - 1;
            GraphTensor operand;
            // OWN_OUTPUT is a valid source for a tensor arg but never for an
            // operand: it would bind the predicate to the buffer this task has
            // yet to write, so the dispatch decision would read whatever the heap
            // last held. The recorder refuses it; so does the image reader.
            if (predicate_index >= definition.predicate_count ||
                predicates[predicate_index].operand_source.source_kind ==
                    static_cast<uint8_t>(GraphTensorSourceKind::OWN_OUTPUT) ||
                !graph_rebind_tensor(
                    execution, tasks, in_graph_task_offsets, predicates[predicate_index].operand,
                    predicates[predicate_index].operand_source, i, &operand
                ) ||
                !graph_predicate_resolve(operand, predicates[predicate_index], &payload.predicate)) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
        }
    }
    execution.materialized_tasks = last;
    if (tasks_materialized != nullptr) *tasks_materialized = last - first;

    if (last < execution.task_count) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::PENDING;
    }

    // Every task's [tensor_offset, tensor_offset + tensor_count) range is bounds-
    // checked on its own. This total additionally requires the ranges to account
    // for the whole tensor array, rejecting a Definition that under- or
    // over-consumes it.
    if (execution.consumed_tensor_args != definition.tensor_arg_count) {
        execution.materialize_busy.store(0, std::memory_order_release);
        return GraphMaterializeResult::INVALID;
    }

    graph_execution_set_state(execution, GraphExecutionState::PREPARED);
    execution.materialize_busy.store(0, std::memory_order_release);
    return GraphMaterializeResult::PREPARED;
}
