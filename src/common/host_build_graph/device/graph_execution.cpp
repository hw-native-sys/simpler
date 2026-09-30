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

#include "graph_image_view.h"
#include "host_build_graph/task_id.h"

namespace {

GraphExecution *acquire_execution_storage(
    uintptr_t storage_addr, size_t storage_bytes, int32_t task_count, int32_t tensor_arg_count,
    int32_t scalar_arg_count, int32_t edge_count
) {
    GraphExecutionStorageLayout layout{};
    // ChipTaskStorage, not GraphExecution: the sub-task array's alignment is the widest
    // the storage carries, and tasks_offset only rounds up relative to this base, so
    // an under-aligned base would leave every alignas(64) sub-task entry misaligned.
    if (storage_addr == 0 || storage_addr % alignof(ChipTaskStorage) != 0 ||
        !graph_execution_storage_layout(task_count, tensor_arg_count, scalar_arg_count, edge_count, &layout) ||
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
    // The execution's own fanin CSR. bind_graph_topology fills it out of the
    // reading thread's section and validates what it wrote; until then the rows
    // are the storage's zeroes and no scan runs, because the slot's
    // graph_context is not published before localize returns.
    execution->fanin_offsets = reinterpret_cast<int32_t *>(base + layout.fanin_offsets_offset);
    execution->fanin_indices =
        edge_count == 0 ? nullptr : reinterpret_cast<uint16_t *>(base + layout.fanin_indices_offset);
    execution->edge_count = edge_count;
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

// Validate this Graph's topology and take the fanin CSR into the execution's own
// storage.
//
// Every read is a bounded copy out of `image` into a local: the section is the
// calling thread's launch arguments, which no other thread may address and which
// carry no alignment guarantee. The fanin rows and indices are validated from the
// execution's copy rather than from the section, because that copy is what every
// later scan reads — a check against the source would leave the copy unproven.
bool bind_graph_topology(GraphExecution &execution, const GraphImageView &image, const GraphDefinitionValue &object) {
    const GraphDefinition &definition = object.definition;
    // Re-checked rather than inherited from graph_execution_localize: every section
    // below is fetched with task_count or task_count + 1, so a wire value outside this
    // range overflows the increment before any bound check can see it.
    if (definition.task_count <= 0 || definition.task_count > SUB_TASK_MAX_NUM) return false;
    if (definition.edge_count < 0 || definition.root_count < 0) return false;
    // GRAPH_MAX_SCALAR_ARGS, not MAX_SCALAR_ARGS: this counts the scalars the
    // Graph BOUNDARY carries, which the recorder sizes with GraphTaskArgs and the
    // outer Graph payload hands it to GraphExecution, never through a sub-task
    // payload. MAX_SCALAR_ARGS is the per-AICore-task cap (16) and applies to
    // SubTaskDefinition::scalar_count below, which is checked separately; using
    // it here rejected every boundary wider than one kernel call could take.
    if (definition.boundary_scalar_count > GRAPH_MAX_SCALAR_ARGS) return false;
    if (execution.fanin_offsets == nullptr || execution.edge_count != definition.edge_count) return false;
    if (definition.edge_count != 0 && execution.fanin_indices == nullptr) return false;

    // The rows and indices this execution will read for the rest of the run.
    int32_t *rows = execution.fanin_offsets;
    uint16_t *indices = execution.fanin_indices;
    if (!graph_definition_copy_array<int32_t>(
            image, object, definition.off_fanin_offsets, definition.task_count + 1, rows
        )) {
        return false;
    }
    if (definition.edge_count != 0 && !graph_definition_copy_array<uint16_t>(
                                          image, object, definition.off_fanin_indices, definition.edge_count, indices
                                      )) {
        return false;
    }

    int32_t fanout_first = 0;
    int32_t fanout_last = 0;
    if (!graph_definition_load_element<int32_t>(
            image, object, definition.off_fanout_offsets, definition.task_count + 1, 0, &fanout_first
        ) ||
        !graph_definition_load_element<int32_t>(
            image, object, definition.off_fanout_offsets, definition.task_count + 1, definition.task_count, &fanout_last
        )) {
        return false;
    }
    if (rows[0] != 0 || fanout_first != 0 || rows[definition.task_count] != definition.edge_count ||
        fanout_last != definition.edge_count) {
        return false;
    }

    uint64_t required_heap = 0;
    constexpr uint8_t VALID_ACTIVE_MASK = (1U << SUBTASK_SLOT_COUNT) - 1U;
    for (int32_t i = 0; i < definition.task_count; ++i) {
        SubTaskDefinition task{};
        uint64_t sub_task_offset = 0;
        if (!graph_definition_load_element<SubTaskDefinition>(
                image, object, definition.off_sub_tasks, definition.task_count, i, &task
            ) ||
            !graph_definition_load_element<uint64_t>(
                image, object, definition.off_sub_task_offsets, definition.task_count, i, &sub_task_offset
            )) {
            return false;
        }
        if (sub_task_offset != required_heap || task.total_output_size < 0 || task.tensor_count < 0 ||
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
        const int32_t begin = rows[consumer];
        const int32_t end = rows[consumer + 1];
        if (begin < 0 || begin > end || end > definition.edge_count) return false;
        if (begin == end) observed_roots++;
        // ED_FLAG_CANDIDATE steers dispatch from materialization onward, so the
        // image must carry the whole conjunction the recorder decided it by, not
        // merely a known bit: a candidate with no producer has nothing to bet on,
        // a DUMMY one would index early_dispatch_queues[] one past its last
        // shape, and a predicated one would be released before its predicate is
        // ever tested. graph_fill_definition is the only writer of this field and
        // holds all three, so a violation means the image is not one it produced.
        SubTaskDefinition consumer_task{};
        if (!graph_definition_load_element<SubTaskDefinition>(
                image, object, definition.off_sub_tasks, definition.task_count, consumer, &consumer_task
            )) {
            return false;
        }
        if ((consumer_task.ed_flags & ED_FLAG_CANDIDATE) != 0 &&
            (begin == end || consumer_task.predicate_slot != 0 ||
             ActiveMask(consumer_task.active_mask).to_shape() == ResourceShape::DUMMY)) {
            return false;
        }
        for (int32_t edge = begin; edge < end; ++edge) {
            if (indices[edge] >= consumer) return false;
        }
    }
    if (observed_roots != definition.root_count) return false;
    for (int32_t i = 0; i < definition.root_count; ++i) {
        uint16_t root = 0;
        if (!graph_definition_load_element<uint16_t>(
                image, object, definition.off_root_indices, definition.root_count, i, &root
            )) {
            return false;
        }
        if (root >= definition.task_count || rows[root] != rows[root + 1]) return false;
    }
    for (int32_t producer = 0; producer < definition.task_count; ++producer) {
        int32_t begin = 0;
        int32_t end = 0;
        if (!graph_definition_load_element<int32_t>(
                image, object, definition.off_fanout_offsets, definition.task_count + 1, producer, &begin
            ) ||
            !graph_definition_load_element<int32_t>(
                image, object, definition.off_fanout_offsets, definition.task_count + 1, producer + 1, &end
            )) {
            return false;
        }
        if (begin < 0 || begin > end || end > definition.edge_count) return false;
        for (int32_t edge = begin; edge < end; ++edge) {
            uint16_t consumer = 0;
            if (!graph_definition_load_element<uint16_t>(
                    image, object, definition.off_fanout_indices, definition.edge_count, edge, &consumer
                )) {
                return false;
            }
            if (consumer <= producer || consumer >= definition.task_count) return false;
        }
    }
    return true;
}

// Rebuild one recorded tensor for this execution. A recorded tensor is a relocation
// record, not a tensor: the two fields replay has a base for are stored relative, and
// this is where the base is added. Which base follows the owner:
//
//   a parameter    buffer comes whole from this call's argument, and `start_offset` is
//                  the view's own offset inside that parameter, so the argument's origin
//                  is added to it
//   a body tensor  buffer address is an offset into the graph heap, whose base this
//                  execution was given; `start_offset` is already its own view origin
//
// Everything else travels absolute, because graph_boundary_matches pins it equal before a
// Definition may be reused. owner_task_id and version are not touched either -- nothing on
// the device reads them off a task's arguments, the scheduler hands the kernel a Tensor
// pointer and the body's dependencies come from the Definition's CSR.
//
// Adding the parameter's *own* origin, rather than a partition-wide shift, is what makes
// this hold even when a boundary slipped through the arrangement check: each tensor is
// rebuilt against the parameter it actually came from.
//
// The values themselves were checked where they were produced: the recorder resolved each
// tensor against the parameter or the producing block it actually came from, and refused
// the body otherwise. What is checked here is only what indexes an array of this
// execution.
bool graph_rebind_tensor(
    const GraphExecution &execution, const simpler::hbg::TensorData &tensor_template, simpler::hbg::Tensor *rebound_out
) {
    rebound_out->init_from(tensor_template);
    const TaskId owner = rebound_out->owner_task_id;
    if (owner.space() == TaskId::Space::PARAM) {
        const int32_t param_index = owner.local_id();
        if (param_index < 0 || param_index >= execution.boundary_tensor_count) return false;
        const simpler::hbg::Tensor &boundary = execution.boundary_tensors[param_index];
        rebound_out->buffer = boundary.buffer;
        rebound_out->start_offset += boundary.start_offset;
        rebound_out->address_space = boundary.address_space;
    } else {
        rebound_out->buffer.addr += execution.heap_base;
    }
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
    const simpler::hbg::Tensor &operand, const GraphPredicate &predicate, DispatchPredicate *resolved_out
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
    const uint64_t element_size = get_element_size(operand.dtype);
    if (element_size != 1 && element_size != 2 && element_size != 4 && element_size != 8) return false;
    if (predicate.elem_size != element_size || predicate.elem_offset >= operand.extent_elem_cache) return false;
    // The recorder bounded elem_offset by the operand's own extent. That the extent itself
    // lies inside the operand's buffer is a property of the well-formed tensor the caller
    // passed, so the scaled sum stays inside the buffer this resolves against.
    const uint64_t byte_offset = (operand.start_offset + predicate.elem_offset) * element_size;

    resolved_out->addr = operand.buffer.addr + byte_offset;
    resolved_out->target = predicate.target;
    resolved_out->elem_size = predicate.elem_size;
    resolved_out->op = static_cast<PredicateOp>(predicate.op);
    return true;
}

}  // namespace

GraphExecution *graph_execution_localize(ChipTaskSlotState &outer_slot, const GraphImageView &image) {
    if (outer_slot.task_kind != TaskKind::GRAPH || outer_slot.to_descriptor().packed_buffer_base == nullptr ||
        outer_slot.to_descriptor().packed_buffer_end == nullptr) {
        return nullptr;
    }

    // The descriptor names the Definition by its offset inside this run's
    // section; the bytes are reached through the caller's own view of that
    // section and never through an address any other thread could hold.
    GraphDefinitionValue object{};
    if (!graph_definition_decode_framed(image, outer_slot.to_descriptor().graph_definition_offset, &object)) {
        return nullptr;
    }
    const GraphDefinition &definition = object.definition;
    TaskPayload &payload = outer_slot.to_payload();
    if (definition.total_bytes == 0 || definition.task_count <= 0 || definition.task_count > SUB_TASK_MAX_NUM ||
        payload.tensor_count != definition.boundary_tensor_count ||
        payload.scalar_count != definition.boundary_scalar_count ||
        (payload.tensor_count != 0 && payload.tensor_data() == nullptr) ||
        (payload.scalar_count != 0 && payload.scalar_data() == nullptr)) {
        return nullptr;
    }

    const uintptr_t outer_base = reinterpret_cast<uintptr_t>(outer_slot.to_descriptor().packed_buffer_base);
    const uintptr_t outer_end = reinterpret_cast<uintptr_t>(outer_slot.to_descriptor().packed_buffer_end);
    if (outer_end < outer_base || definition.required_heap > UINTPTR_MAX - outer_base ||
        definition.execution_storage_bytes > outer_end - outer_base ||
        definition.required_heap > outer_end - outer_base - definition.execution_storage_bytes) {
        return nullptr;
    }
    GraphExecution *execution = acquire_execution_storage(
        outer_base + definition.required_heap, definition.execution_storage_bytes, definition.task_count,
        definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count
    );
    if (execution == nullptr) return nullptr;

    execution->definition_offset = object.image_offset;
    execution->outer_slot = &outer_slot;
    // Checked just above: required_heap fits between outer_base and outer_end, so every
    // offset a body tensor carries resolves inside the region this Graph was given.
    execution->heap_base = outer_base;
    execution->boundary_tensors = payload.tensor_data();
    execution->boundary_tensor_count = payload.tensor_count;
    execution->boundary_scalars = payload.scalar_data();
    execution->boundary_scalar_count = payload.scalar_count;
    if (!bind_graph_topology(*execution, image, object)) {
        execution->retired_tasks.store(execution->task_count, std::memory_order_relaxed);
        graph_execution_mark_completed(*execution);
        return nullptr;
    }
    outer_slot.graph_context = execution;
    return execution;
}

GraphMaterializeResult graph_execution_materialize_slice(
    ChipTaskSlotState &outer_slot, GraphExecution &execution, const GraphImageView &image, int32_t max_tasks,
    int32_t *tasks_materialized
) {
    if (tasks_materialized != nullptr) *tasks_materialized = 0;
    if (outer_slot.task_kind != TaskKind::GRAPH || outer_slot.to_descriptor().packed_buffer_base == nullptr ||
        max_tasks <= 0 || execution.definition_offset == 0 || execution.task_storage == nullptr) {
        return GraphMaterializeResult::INVALID;
    }
    // Decoded again here, from this thread's own view: a slice may run on a
    // thread that did not localize this Graph, and the framing is what makes the
    // offset shared state carries usable against a section this thread holds.
    GraphDefinitionValue object{};
    if (!graph_definition_decode_framed(image, execution.definition_offset, &object)) {
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

    // Every section below is read one element at a time out of `image`, each read
    // bounded against its array, this Definition and the whole section, so no
    // array pointer into the package exists to outlive this call.
    const GraphDefinition &definition = object.definition;
    if (definition.task_count != execution.task_count) {
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

        task.task_id = TaskId::make_sub_task(outer_slot.to_descriptor().task_id.local_id(), i);
        SubTaskDefinition source{};
        uint64_t task_offset = 0;
        if (!graph_definition_load_element<SubTaskDefinition>(
                image, object, definition.off_sub_tasks, definition.task_count, i, &source
            ) ||
            !graph_definition_load_element<uint64_t>(
                image, object, definition.off_sub_task_offsets, definition.task_count, i, &task_offset
            )) {
            execution.materialize_busy.store(0, std::memory_order_release);
            return GraphMaterializeResult::INVALID;
        }
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
        slot.sub_task_local_id = i;
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
            simpler::hbg::TensorData record{};
            if (!graph_definition_load_element<simpler::hbg::TensorData>(
                    image, object, definition.off_tensors, definition.tensor_arg_count, tensor_index, &record
                ) ||
                !graph_rebind_tensor(execution, record, &task_tensors[j])) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
            execution.consumed_tensor_args++;
        }
        uint64_t *task_scalars = payload.scalar_data();
        for (int32_t j = 0; j < source.scalar_count; ++j) {
            const int32_t scalar_index = source.scalar_offset + j;
            GraphScalarInheritance ref{};
            if (!graph_definition_load_element<GraphScalarInheritance>(
                    image, object, definition.off_scalar_inheritance, definition.scalar_arg_count, scalar_index, &ref
                )) {
                execution.materialize_busy.store(0, std::memory_order_release);
                return GraphMaterializeResult::INVALID;
            }
            if (!ref.inherited()) {
                if (!graph_definition_load_element<uint64_t>(
                        image, object, definition.off_scalars, definition.scalar_arg_count, scalar_index,
                        &task_scalars[j]
                    )) {
                    execution.materialize_busy.store(0, std::memory_order_release);
                    return GraphMaterializeResult::INVALID;
                }
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
            simpler::hbg::Tensor operand;
            // The consuming task's own output is a valid source for a tensor arg but never
            // for an operand: it would bind the predicate to the buffer this task has yet
            // to write, so the dispatch decision would read whatever the heap last held.
            // The recorder refuses it; so does the image reader, reading the same owner the
            // rebind resolves against.
            GraphPredicate predicate{};
            const bool predicate_loaded =
                predicate_index < definition.predicate_count &&
                graph_definition_load_element<GraphPredicate>(
                    image, object, definition.off_predicates, definition.predicate_count, predicate_index, &predicate
                );
            const TaskId operand_owner = predicate_loaded ? predicate.operand.owner_task_id : TaskId::invalid();
            if (!predicate_loaded ||
                (operand_owner.space() == TaskId::Space::SUB_TASK && operand_owner.local_id() == i) ||
                !graph_rebind_tensor(execution, predicate.operand, &operand) ||
                !graph_predicate_resolve(operand, predicate, &payload.predicate)) {
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
