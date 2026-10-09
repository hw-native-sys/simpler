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

#include "host_build_graph/graph_submit.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "assert_compat.h"
#include "common/host_phase_kind.h"
#include "common/unified_log.h"
#include "host_build_graph/dep_compute.h"
#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/graph_boundary_match.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/graph_host_state.h"
#include "host_build_graph/graph_recording.h"
#include "host_build_graph/host_phase_trace.h"
#include "host_build_graph/orch_profiling.h"
#include "host_build_graph/orchestrator_internal.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"
#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"
#include "tensor.h"

namespace {

GraphHostState *graph_state_from(OrchestratorState *orch) {
    return orch == nullptr ? nullptr : static_cast<GraphHostState *>(orch->graph_host_state);
}

// The in-flight entry this thread is recording into, bound by graph_prepare and
// cleared by graph_end / graph_abort.
thread_local GraphInflightRecording *g_active_graph_entry = nullptr;
thread_local GraphRecording *g_active_graph_recording = nullptr;
// The recording a thread holds belongs to one GraphHostState. Recording into it
// from a different orchestrator would silently mix two graphs, so the owner is
// part of the thread-local identity rather than implied by it.
thread_local GraphHostState *g_active_graph_owner = nullptr;

// Drop this thread's recording's view of an in-flight entry's boundary. Called where a
// recording ends, because the entry does not outlive graph_commit while the storage does.
void unbind_recorder_boundary() {
    if (g_active_graph_recording != nullptr) g_active_graph_recording->boundary = nullptr;
}

// Bind this thread's storage to one in-flight entry and empty it. Returns false when the
// hazard map or the tensor pool cannot be stood up.
bool graph_recording_reset(GraphRecording &recording, const GraphInflightRecording &entry) {
    // A body over SUB_TASK_MAX_NUM is abandoned, but it still grew every array to its real
    // size while it ran. Handing that to the next recording would retain storage for a
    // Definition that can never be published, unbounded, for the process's life -- so an
    // over-cap recording gives its storage back instead of passing it on. This is what
    // makes the bound documented on GraphRecording::task_count true rather than nominal.
    if (recording.tasks.size() > static_cast<size_t>(SUB_TASK_MAX_NUM)) {
        recording = GraphRecording{};
    }
    if (!graph_recording_stand_up(recording)) {
        return false;
    }
    recording.tensor_map.reset();
    recording.full_key = entry.full_key;
    recording.boundary = &entry.boundary;
    recording.next_virtual_offset = 0;
    recording.unsupported = false;
    recording.scope_stack_top = -1;
    recording.manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    // clear() keeps each array's capacity, and the stand-up above reserved every one of
    // them to what a body at the cap needs, so no body a thread records can grow one.
    // tasks is deliberately not cleared: see GraphRecording::task_count.
    recording.task_count = 0;
    recording.task_tensor_cursor = 0;
    recording.scalars.clear();
    recording.scalar_inheritance.clear();
    recording.internal_fanins.clear();
    recording.predicates.clear();
    return true;
}

}  // namespace

GraphRecording *active_graph_recording(OrchestratorState *orch) {
    GraphHostState *state = graph_state_from(orch);
    if (state == nullptr || state != g_active_graph_owner) return nullptr;
    return g_active_graph_recording;
}

namespace {

// The one-line diagnosis of a scalar mismatch. Shared by both match paths: they differ in
// where they hold the recorded boundary, not in what a mismatch there means.
//
// The index is what makes this actionable -- a boundary of a dozen scalars gives the author
// nowhere to look without it.
void graph_boundary_warn_scalar_mismatch(
    const GraphBoundaryScalarMatch &recorded, const GraphTaskArgs &args, int32_t index
) {
    if (args.scalar_dynamic(index) != recorded.dynamic) {
        LOG_WARN(
            "[GraphExecution] boundary scalar %d declaration differs from recording; using ordinary path: "
            "recorded=%s actual=%s",
            index, recorded.dynamic ? "dynamic" : "static", args.scalar_dynamic(index) ? "dynamic" : "static"
        );
        return;
    }
    // Reaching here means the declarations agree and the values did not, which
    // graph_boundary_scalar_mismatch only reports for a slot it found static. That is what
    // makes the scalar<uint64_t> read below legal: Arg::scalar<T> refuses a dynamic
    // parameter, so widening this branch to cover a declaration mismatch would trip it.
    LOG_WARN(
        "[GraphExecution] boundary scalar %d is declared static and its value moved; using ordinary path: "
        "recorded=%llu actual=%llu",
        index, static_cast<unsigned long long>(recorded.value),
        static_cast<unsigned long long>(args.scalar<uint64_t>(index))
    );
}

bool graph_boundary_matches(
    const GraphDefinition &definition, const GraphDefinitionRecord &record, const GraphTaskArgs &args
) {
    if (args.scalar_count() != definition.boundary_scalar_count || args.explicit_dep_count() != 0 ||
        args.tensor_count() != definition.boundary_tensor_count) {
        LOG_WARN(
            "[GraphExecution] fixed boundary contract mismatch: tensors=%d/%d scalars=%d/%d explicit_deps=%u",
            args.tensor_count(), definition.boundary_tensor_count, args.scalar_count(),
            definition.boundary_scalar_count, args.explicit_dep_count()
        );
        return false;
    }
    // Before the tensors, because it is the cheaper half and the likelier reject: a static
    // scalar whose value moved is a declared fallback, where a tensor whose geometry moved is
    // a contract this runtime does not support. That asymmetry is also why no debug_assert
    // sits beside this one.
    //
    // The count checked above is the image's copy while the array walked here is the host
    // record's, so state the invariant locally: both are filled from the same
    // bound_boundary().params.scalar_count() when the Definition is published, which is two
    // hops away from either reader. The tensor half below indexes the same way.
    debug_assert(
        record.boundary_match_info.scalars.size() == static_cast<size_t>(definition.boundary_scalar_count) &&
        "a published record's scalar match array is sized by the boundary the image counted"
    );
    if (const int32_t i = graph_boundary_scalar_mismatch(record.boundary_match_info.scalars.data(), args); i >= 0) {
        graph_boundary_warn_scalar_mismatch(record.boundary_match_info.scalars[i], args, i);
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const simpler::hbg::Tensor &actual = args.tensor(i).ref();
        if (actual.ndims > MAX_TENSOR_DIMS) {
            debug_assert(
                actual.ndims <= MAX_TENSOR_DIMS && "Graph boundary simpler::hbg::Tensor rank is not supported"
            );
            LOG_WARN(
                "[GraphExecution] simpler::hbg::Tensor rank %u exceeds the fixed Graph boundary limit",
                static_cast<unsigned>(actual.ndims)
            );
            return false;
        }
        const GraphBoundaryTensorMatch &expected = record.boundary_match_info.tensors[i];
        if (!graph_boundary_tensor_matches(expected, actual, args.tag(i))) {
            // Logged before the assertion, which aborts a debug build: an assertion that
            // fires first takes the one line saying what actually moved down with it.
            //
            // start_offset is not among these: an argument may slide, and what that costs
            // is checked as an arrangement below rather than per parameter.
            LOG_WARN(
                "[GraphExecution] fixed tensor metadata mismatch at boundary arg %d; using ordinary path: "
                "size=%llu/%llu ndims=%u/%u dtype=%u/%u tag=%u/%u manual_dep=%d/%d contiguous=%d/%d",
                i, static_cast<unsigned long long>(actual.buffer.size),
                static_cast<unsigned long long>(expected.buffer_size), static_cast<unsigned>(actual.ndims),
                static_cast<unsigned>(expected.ndims), static_cast<unsigned>(actual.dtype),
                static_cast<unsigned>(expected.dtype), static_cast<unsigned>(args.tag(i)),
                static_cast<unsigned>(expected.tag), static_cast<int>(actual.manual_dep),
                static_cast<int>(expected.manual_dep), static_cast<int>(actual.is_contiguous),
                static_cast<int>(expected.is_contiguous)
            );
            debug_assert(false && "Variable Graph boundary tensor shape/metadata is not supported");
            return false;
        }
    }
    if (!graph_boundary_arrangement_matches(record.boundary_match_info.tensors.data(), args)) {
        LOG_WARN(
            "%s", "[GraphExecution] boundary alias partition or intra-partition offsets differ from recording; "
                  "using ordinary path"
        );
        debug_assert(false && "Changing the Graph boundary arrangement is not supported");
        return false;
    }
    return true;
}

bool graph_boundary_matches(const GraphBoundary &boundary, const GraphTaskArgs &args) {
    if (args.scalar_count() != boundary.params.scalar_count() || args.explicit_dep_count() != 0 ||
        args.tensor_count() != boundary.params.tensor_count()) {
        return false;
    }
    // The one diagnosed mismatch on this path. The structural ones stay silent because a
    // same-key submission whose shape differs is an ordinary outcome of the key folding only
    // construction parameters, while a static scalar whose value moved is the author's
    // declaration failing to hold -- the same reason the published path words it.
    if (const int32_t i = graph_boundary_scalar_mismatch(boundary.match_info.scalars.data(), args); i >= 0) {
        graph_boundary_warn_scalar_mismatch(boundary.match_info.scalars[i], args, i);
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const simpler::hbg::Tensor &actual = args.tensor(i).ref();
        if (actual.ndims > MAX_TENSOR_DIMS ||
            !graph_boundary_tensor_matches(boundary.match_info.tensors[i], actual, args.tag(i))) {
            return false;
        }
    }
    return graph_boundary_arrangement_matches(boundary.match_info.tensors.data(), args);
}

void graph_reset_outer_payload(TaskPayload &payload) {
    payload.tensor_count = 0;
    payload.scalar_count = 0;
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (auto &word : payload.staged_core_mask)
        word.store(0, std::memory_order_relaxed);
    payload.published_block_count.store(0, std::memory_order_relaxed);
    payload.early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload.running_slot_count.store(0, std::memory_order_relaxed);
    payload.early_sync_drain_state.store(EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

bool graph_submit_outer(
    OrchestratorState *orch, GraphHostState *state, uint64_t full_key, int32_t owned_heap, bool defer_heap,
    const GraphTaskArgs &args, TaskId *submitted_id
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit Graph outside a scope");
    auto &allocator = orch->task_allocator;
    if (allocator.active_count() >= allocator.capacity() ||
        (!defer_heap && static_cast<uint64_t>(owned_heap) > allocator.heap_available())) {
        LOG_WARN("%s", "[GraphExecution] task-capacity/heap preflight failed; using ordinary path");
        return false;
    }

    // The argument pools hold MAX_TENSOR_ARGS ChipTensors and MAX_SCALAR_ARGS scalars
    // per task slot, a budget no CoreTaskArgs task can exceed. A Graph boundary is
    // GraphTaskArgs-wide, so a wide one draws more than the single slot it occupies is
    // worth — GraphBoundaryPool.WidestBoundaryExceedsOneSlotBudget pins how much. The
    // cursors bump through a fixed mirror whose last segment is the scalar pool, so an
    // overdraw writes past that mirror rather than merely exhausting a quota. Test it
    // ahead of the slot claim and decline the Graph path, which leaves the caller to
    // replay the block as ordinary tasks.
    const uint64_t max_tasks = static_cast<uint64_t>(orch->task_allocator.capacity());
    const int32_t tensor_slots = args.tensor_count();
    const int32_t scalar_span = CHIP_ALIGN_UP(args.scalar_count(), ARG_POOL_ALIGN / (int32_t)sizeof(uint64_t));
    if (static_cast<uint64_t>(orch->tensor_pool_cursor) + tensor_slots > max_tasks * MAX_TENSOR_ARGS ||
        static_cast<uint64_t>(orch->scalar_pool_cursor) + scalar_span > max_tasks * MAX_SCALAR_ARGS) {
        LOG_WARN("%s", "[GraphExecution] boundary exceeds the argument pools; using ordinary path");
        return false;
    }

    GraphPendingUpload pending;
    pending.full_key = full_key;
    pending.deferred_heap = defer_heap;

    DepInputs boundary_inputs{
        args.tensor_count(), args.tensor_data(), args.tag_data(), 0, nullptr,
    };
    const int32_t tensormap_needed = count_registrable_outputs(boundary_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) return false;
    const TaskAllocResult allocation = allocator.alloc(defer_heap ? 0 : owned_heap);
    if (allocation.failed()) {
        orch_mark_fatal(orch, SIMPLER_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }
    const TaskId task_id = TaskId::make_global(allocation.task_id);
    SharedMemoryTaskHeader &tasks = orch->sm_header->tasks;
    ChipTaskStorage &storage = tasks.storage_at(allocation.task_id);
    TaskDescriptor &task = storage.task;
    TaskPayload &payload = storage.payload;
    ChipTaskSlotState &slot = storage.slot;

    // Init-on-write, as in prepare_task: this slot's dynamic scheduling fields and
    // progress state are established here, at the claim, because nothing else
    // writes them. A stale wake_list_head of WAKE_LIST_SENTINEL would close the
    // list against every consumer, and a stale progress state would report the
    // Graph done before it ran.
    slot.reset_for_reuse();
    tasks.reset_task_state(allocation.task_id);

    // Graph boundaries use the same compact argument pools as ordinary tasks. The
    // outer payload carries the invocation data; graph_context only names the
    // shared Definition until device initialization replaces it with GraphExecution.
    // The preflight bounds both spans, so these cursors stay inside their pools.
    payload.bind_regions(
        orch->tensor_pool + orch->tensor_pool_cursor, orch->scalar_pool + orch->scalar_pool_cursor,
        orch->fanin_pool + orch->fanin_pool_cursor
    );
    orch->tensor_pool_cursor += tensor_slots;
    orch->scalar_pool_cursor += scalar_span;
    slot.active_mask = ActiveMask{};
    slot.task_attrs = TaskAttrs{};
    slot.total_required_subtasks = 0;
    // A shell places no block, but this must stay positive: an early-released
    // shell sits in EARLY_DISPATCH_STAGING, so its readiness runs through
    // try_early_dispatch_release, which returns next_block_idx >= this. At zero
    // that is true for a shell's never-advanced cursor, push_ready_routed
    // returns before its TaskKind::GRAPH branch, and the Graph silently never
    // activates — visible only as SIMPLER_ERROR_SCHEDULER_TIMEOUT.
    slot.logical_block_num = 1;
    slot.task_kind = TaskKind::GRAPH;

    task.task_id = task_id;
    std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
    task.packed_buffer_base = allocation.packed_base;
    task.packed_buffer_end = allocation.packed_end;
    graph_reset_outer_payload(payload);
    payload.tensor_count = args.tensor_count();
    payload.scalar_count = args.scalar_count();
    auto *boundary_tensors = payload.tensor_data();
    for (int32_t i = 0; i < args.tensor_count(); ++i)
        boundary_tensors[i] = args.tensor(i).ref();
    if (args.scalar_count() != 0) {
        // Resolved, not copied: this is the boundary the device patches BOUNDARY-sourced
        // sub-task scalars from, so it has to hold values. Only scalar_count entries
        // are written; the region's alignment padding keeps whatever it held.
        args.pack_scalars(payload.scalar_data());
    }

    // graph_reset_outer_payload above zeroed the count; the region delta is resolved
    // once here.
    next_fanin_seen_epoch(orch);
    int32_t *fanin_slots = payload.fanin_data();
    auto emit = [&](TaskId producer_id) -> bool {
        return append_fanin_or_fail(*orch, producer_id, fanin_slots, payload.fanin_count);
    };
    // An outer GRAPH task is an ordinary task, so the dependency graph
    // has to carry it: without this the whole Graph — and every edge into it —
    // is absent from deps.json, leaving a run of 40 replays described by only its
    // handful of non-Graph tasks. It dispatches no kernel of its own and the
    // sub-DAG it replays owns no task slots, so what is captured is its boundary:
    // the args it consumes and the edges those produce.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, SUBTASK_SLOT_COUNT> kernel_ids_capture{
            INVALID_KERNEL_ID,
            INVALID_KERNEL_ID,
            INVALID_KERNEL_ID,
        };
        dep_gen_host_graph_begin_task(
            task_id, orch->in_manual_scope(), /*early_dispatch=*/false, kernel_ids_capture.data(),
            slot.logical_block_num, args.tensor_count(), args.tensor_data(), args.tag_data()
        );
        const bool ok =
            compute_task_fanin(boundary_inputs, orch->tensor_map, orch->in_manual_scope(), emit, DepGraphAnnotate{});
        // The task's last capture point, so the entry closes whether or not the
        // fanin computation succeeded.
        dep_gen_host_graph_end_task();
        if (!ok) return false;
    } else if (!compute_task_fanin(boundary_inputs, orch->tensor_map, orch->in_manual_scope(), emit)) {
        return false;
    }
    register_task_outputs(boundary_inputs, task_id, orch->tensor_map, orch->in_manual_scope());
    // The region's length is settled, so the cursor closes it at the real count. The
    // equality holds only while nothing between the bind and here bound another fanin
    // region, which is what makes the deferred advance safe.
    debug_assert(orch->fanin_pool_cursor == static_cast<int32_t>(payload.fanin_data() - orch->fanin_pool));
    orch->fanin_pool_cursor += CHIP_ALIGN_UP(payload.fanin_count, ARG_POOL_ALIGN / (int32_t)sizeof(int32_t));

    // Early-dispatch qualification for the shell. Its fanin is an ordinary
    // inline row of GLOBAL producers, so the rule is the top-level one, minus
    // the terms that describe dispatching a task to cores: a shell has no
    // predicate, no resource shape, and never occupies a core itself. What its
    // release does instead is admit the body's roots, which is why a shell
    // qualifies on producers alone.
    //
    // A GRAPH producer still disqualifies, as it does at top level: a shell
    // publishes no placement of its own, so there is nothing for a consumer to
    // bet on. That is the graph-as-producer direction, deliberately left out.
    int32_t *const shell_fanin = payload.fanin_data();
    bool shell_candidate = payload.fanin_count > 0;
    for (int32_t i = 0; shell_candidate && i < payload.fanin_count; i++) {
        const ChipTaskSlotState &producer = orch->sm_header->tasks.get_slot_state_by_task_id(shell_fanin[i]);
        if (producer.task_kind == TaskKind::GRAPH || !producer.task_attrs.allow_early_resolve()) {
            shell_candidate = false;
        }
    }
    if (shell_candidate) {
        std::sort(shell_fanin, shell_fanin + payload.fanin_count);
        slot.ed_flags |= ED_FLAG_CANDIDATE;
        for (int32_t i = 0; i < payload.fanin_count; i++) {
            orch->sm_header->tasks.get_slot_state_by_task_id(shell_fanin[i]).ed_flags |= ED_FLAG_TRACKED;
        }
    }

    pending.outer_slot = &slot;
    state->pending_uploads.push_back(pending);
    if (submitted_id != nullptr) *submitted_id = task_id;
#if SIMPLER_DFX
    orch->tasks_submitted++;
#endif
    return true;
}

bool graph_submit_definition(
    OrchestratorState *orch, GraphHostState *state, const GraphDefinition *definition,
    const GraphDefinitionRecord &record, const GraphTaskArgs &args, TaskId *submitted_id
) {
    if (definition == nullptr || !graph_boundary_matches(*definition, record, args) ||
        definition->execution_storage_bytes == 0 ||
        definition->required_heap > UINT64_MAX - definition->execution_storage_bytes) {
        return false;
    }
    const uint64_t owned_heap = definition->required_heap + definition->execution_storage_bytes;
    if (owned_heap > static_cast<uint64_t>(INT32_MAX)) return false;
    return graph_submit_outer(
        orch, state, definition->full_key, static_cast<int32_t>(owned_heap), false, args, submitted_id
    );
}

bool graph_submit_pending_definition(
    OrchestratorState *orch, GraphHostState *state, uint64_t full_key, const GraphTaskArgs &args, TaskId *submitted_id
) {
    return graph_submit_outer(orch, state, full_key, 0, true, args, submitted_id);
}

bool graph_finalize_pending_submissions(OrchestratorState *orch, GraphHostState *state, uint64_t *failed_key) {
    for (GraphPendingUpload &pending : state->pending_uploads) {
        if (!pending.deferred_heap) continue;
        auto definition_it = state->definitions.find(pending.full_key);
        const GraphDefinition *definition = definition_it == state->definitions.end() ?
                                                nullptr :
                                                graph_record_definition(*state, definition_it->second);
        if (definition == nullptr || definition->execution_storage_bytes == 0 ||
            definition->required_heap > UINT64_MAX - definition->execution_storage_bytes ||
            pending.outer_slot == nullptr || pending.outer_slot->task_kind != TaskKind::GRAPH) {
            if (failed_key != nullptr) *failed_key = pending.full_key;
            return false;
        }
        const uint64_t owned_heap = definition->required_heap + definition->execution_storage_bytes;
        if (owned_heap > static_cast<uint64_t>(INT32_MAX)) {
            if (failed_key != nullptr) *failed_key = pending.full_key;
            return false;
        }
        void *packed_base = nullptr;
        void *packed_end = nullptr;
        if (!orch->task_allocator.reserve_deferred_heap(static_cast<int32_t>(owned_heap), &packed_base, &packed_end)) {
            if (failed_key != nullptr) *failed_key = pending.full_key;
            return false;
        }
        TaskDescriptor &outer_task = pending.outer_slot->to_descriptor();
        outer_task.packed_buffer_base = packed_base;
        outer_task.packed_buffer_end = packed_end;
        pending.deferred_heap = false;
    }
    return true;
}

}  // namespace

TaskOutputTensors graph_record_submit_sub_task(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    ORCH_PHASE_START();
    TaskOutputTensors result;
    GraphRecording &recording = *active_graph_recording(orch);

    const int32_t task_index = recording.task_count;
    // A recorded task lives in the SUB_TASK id space, so an id the body hands
    // around says which of the two kinds of thing it names without any arithmetic:
    // a SUB_TASK id is a task of this body, indexed by its low field; a GLOBAL id is
    // a task submitted before the Graph, which nothing in the body may depend on.
    const TaskId task_id = TaskId::make_sub_task(GRAPH_RECORD_NO_OWNING_GRAPH, task_index);
    result.set_task_id(task_id);

    if (task_index >= SUB_TASK_MAX_NUM || args.has_error()) {
        recording.unsupported = true;
    }

    const OutputLayout layout = calculate_output_layout(args);
    const uint64_t aligned_output = CHIP_ALIGN_UP(static_cast<uint64_t>(layout.total_output_size), CHIP_ALIGN_SIZE);
    // Outputs are bumped past the parameter windows: both regions belong to one simulated
    // heap, so the bound counts them together. That is deliberately conservative -- the
    // parameter windows stand for the caller's own buffers and never reach the graph heap
    // a replay commits -- and costs nothing at this scale.
    const uint64_t param_used = recording.bound_boundary().param_used_heap_size;
    const uint64_t reserved = param_used + aligned_output;
    if (reserved > MAX_HEAP_CAPACITY || recording.next_virtual_offset > MAX_HEAP_CAPACITY - reserved) {
        recording.unsupported = true;
        return result;
    }
    const uintptr_t packed_base_addr = GRAPH_RECORD_BASE + param_used + recording.next_virtual_offset;
    recording.next_virtual_offset += aligned_output;

    // The task is filled in place, in the slot it will keep, and reset() puts the rest of
    // the slot back to a freshly recorded task's state. An over-cap body grows `tasks`,
    // which moves the slots, but the addresses handed to the caller live in the recording's
    // tensor pool rather than in a slot, so a move cannot invalidate them.
    if (task_index >= static_cast<int32_t>(recording.tasks.size())) recording.tasks.emplace_back();
    RecordedSubTask &task = recording.tasks[task_index];
    task.reset();
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.active_mask = active_mask;
    // The recorded copy keeps the caller's early-resolve intent, which is the input
    // graph_fill_definition qualifies this body's early dispatch against. The bit is
    // cleared on the way into the Definition instead, so no sub-task reaches
    // the device carrying it.
    task.task_attrs = task_attrs;
    task.logical_block_num = args.launch_spec.block_num();
    // Mirror prepare_task's contract: block_num must be positive and the subtask
    // count must fit int16_t. An out-of-contract value marks asynchronous
    // recording unsupported and makes commit fail-fast, rather than baking a
    // truncated or negative count into the cached Definition (which the device
    // would expand into a sub-task that never completes).
    const int32_t required_subtasks =
        static_cast<int32_t>(task.logical_block_num) * __builtin_popcount(active_mask.core_mask());
    if (task.logical_block_num <= 0 || required_subtasks > std::numeric_limits<int16_t>::max()) {
        recording.unsupported = true;
        task.total_required_subtasks = 0;
    } else {
        task.total_required_subtasks = static_cast<int16_t>(required_subtasks);
    }
    task.record_packed_base = packed_base_addr;
    task.total_output_size = aligned_output;

    // Build the tensor list exactly as TaskPayload::init: inputs/inouts copy
    // the caller's simpler::hbg::Tensor; outputs materialize from the create-info onto the
    // scratch buffer and carry the recorded task's owner id.
    const int32_t tensor_count = args.tensor_count();
    // Claim this task's slice of the pool. The cursor is a pure bump, so slices abut and a
    // body holds its tensors in the bytes they need; nothing is ever returned to it, since
    // the whole pool is reset by the next recording.
    if (static_cast<size_t>(recording.task_tensor_cursor) + static_cast<size_t>(tensor_count) >
        GRAPH_RECORD_TENSOR_POOL_ELEMS) {
        recording.unsupported = true;
        return result;
    }
    task.tensor_offset = recording.task_tensor_cursor;
    task.tensor_count = tensor_count;
    recording.task_tensor_cursor += tensor_count;
    simpler::hbg::Tensor *task_tensors = recording.task_tensors(task);
    // Value-initialized before the fill, not merely claimed: the slice holds whatever the
    // previous body left in it, and simpler::hbg::Tensor::init_from writes strides only up
    // to the new tensor's ndims, so a narrower tensor would inherit a wider one's trailing
    // strides.
    std::fill_n(task_tensors, tensor_count, simpler::hbg::Tensor{});
    for (int32_t i = 0; i < tensor_count; ++i) {
        simpler::hbg::Tensor &slot_tensor = task_tensors[i];
        if (args.tag(i) != TensorArgType::OUTPUT) {
            slot_tensor.copy(args.tensor(i).ref());
        } else {
            init_tensor_from_create_info(
                slot_tensor, args.tensor(i).create_info(),
                reinterpret_cast<void *>(packed_base_addr + layout.offsets[i]), layout.buffer_sizes[i]
            );
            slot_tensor.owner_task_id = task_id;
        }
    }
    // The addresses handed out here are into the pool, which never moves, so they stay
    // valid for the rest of this recording.
    for (int32_t i = 0; i < tensor_count; ++i) {
        if (args.tag(i) == TensorArgType::OUTPUT) result.materialize_output(task_tensors[i]);
    }
    task.scalar_offset = static_cast<int32_t>(recording.scalars.size());
    task.scalar_count = args.scalar_count();
    // Resolved values, not slots: an inherited slot's word is a host pointer, and this is
    // the recording's working copy of what the task passed. What a BOUNDARY-sourced slot
    // contributes to the Definition is overwritten with a placeholder at build time.
    recording.scalars.resize(recording.scalars.size() + static_cast<size_t>(task.scalar_count));
    args.pack_scalars(recording.scalars.data() + task.scalar_offset);
#if SIMPLER_DFX
    task.dump_metadata.dump_arg_mask = args.dump_arg_mask();
    task.dump_metadata.dump_arg_flags = args.dump_arg_index_ambiguous_mask();
    memcpy(task.dump_metadata.scalar_dtypes, args.scalar_dtypes(), args.scalar_count() * sizeof(uint8_t));
#endif

    // Classify each scalar's source: a slot holding its own value is static Definition
    // data, while an inherited slot names a boundary parameter refreshed on every replay.
    recording.scalar_inheritance.resize(task.scalar_offset + task.scalar_count);
    graph_classify_scalars(recording, args, task.scalar_offset);

    for (int32_t i = 0; i < tensor_count; ++i) {
        if (!graph_classify_tensor(recording, task_index, task_tensors[i])) {
            recording.unsupported = true;
        }
    }
    // A dispatch predicate resolves to an absolute GM address at submit, which a
    // Definition replayed against fresh buffers cannot carry. Record the operand
    // the same way a tensor arg is recorded — classified source plus the element
    // index within that tensor — and let materialize resolve the pair. The
    // predicate creates no dependency here any more than it does on the ordinary
    // path: the caller declares one, and the explicit-dep loop below records it.
    //
    // Gated on the recorded attribute, not on args: a kernel-less task never
    // dispatches, so submit_dummy_task and alloc_tensors drop the predicate the
    // caller set. Reading args here instead would record a predicate the task's
    // own attribute denies, and materialize rejects a Definition whose two halves
    // disagree.
    if (task.task_attrs.has_predicate()) {
        const CoreTaskPredicate &pred = args.predicate();
        GraphRecordedPredicate recorded;
        recorded.op = pred.op;
        recorded.target = pred.target;
        const simpler::hbg::Tensor *operand = pred.operand.tensor;
        // A predicate on the consuming task's own output would read the buffer that task
        // has yet to write, so it names no value the predicate could be evaluating. An
        // index vector that leaves the operand's extent is caught here too: materialize
        // would otherwise reject the baked offset on the device, where the failure is a
        // Scheduler fatal rather than a named unsupported construct.
        const uint64_t flat_offset =
            operand == nullptr ? 0 : operand->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
        if (operand == nullptr || operand->address_space != AddressSpace::DEVICE || operand->ndims > MAX_TENSOR_DIMS ||
            pred.operand.ndims > operand->ndims || flat_offset < operand->start_offset ||
            flat_offset - operand->start_offset >= operand->extent_elem_cache ||
            !graph_classify_tensor(recording, task_index, *operand) ||
            (operand->owner_task_id.space() == TaskId::Space::SUB_TASK &&
             operand->owner_task_id.local_id() == task_index)) {
            recording.unsupported = true;
        } else {
            recorded.operand.copy(*operand);
            recorded.elem_offset = flat_offset - operand->start_offset;
            recorded.elem_size = static_cast<uint8_t>(get_element_size(operand->dtype));
        }
        task.predicate_index = static_cast<int32_t>(recording.predicates.size());
        recording.predicates.push_back(recorded);
    }

    task.fanin_offset = static_cast<int32_t>(recording.internal_fanins.size());
    // Dedup within this task's own range: the flat array's earlier entries belong
    // to the body's earlier tasks.
    auto add_fanin = [&recording, &task](int32_t producer) {
        const auto begin = recording.internal_fanins.begin() + task.fanin_offset;
        if (std::find(begin, recording.internal_fanins.end(), producer) == recording.internal_fanins.end()) {
            recording.internal_fanins.push_back(producer);
        }
    };
    // An argument produced by an earlier task of this body is an edge. The owner says which
    // task that is; the consuming task's own outputs are not edges, which is why this asks
    // for a strictly earlier one.
    for (int32_t i = 0; i < tensor_count; ++i) {
        const TaskId owner = task_tensors[i].owner_task_id;
        if (owner.space() != TaskId::Space::SUB_TASK) continue;
        const int32_t producer = owner.local_id();
        if (producer >= 0 && producer < task_index) add_fanin(producer);
    }

    // Inferred hazards, on the same terms as the ordinary path. The loop above only
    // names the sub-task that ALLOCATED each buffer; every write-then-read through a
    // buffer someone else allocated — an alloc_tensors output written in place
    // with add_inout, or a view of a boundary tensor — needs the last-writer
    // lookup compute_task_fanin performs. Running the very same function against
    // the recording's own map is what keeps a Definition's edge set equal to the
    // one the body gets when the ordinary path submits its tasks one at a time.
    //
    // Producers outside the recording window are dropped: they are reached
    // through boundary tensors, and the outer Graph shell already carries those
    // args, so the shell's own fanin orders the whole body behind them.
    {
        const DepInputs dep_inputs{
            tensor_count,
            args.tensor_data(),
            args.tag_data(),
            static_cast<int32_t>(args.explicit_dep_count()),
            args.explicit_deps_data(),
        };
        const bool manual_scope = recording.in_manual_scope();
        if (!recording.storage_ready || task_index >= SUB_TASK_MAX_NUM) {
            // An over-cap body is already abandoned, and its task ids have run past
            // the low field TaskId::make_sub_task packs them into, so registering one
            // would key the map outside its task chains.
            recording.unsupported = true;
        } else if (recording.tensor_map.free_entries() < count_registrable_outputs(dep_inputs, manual_scope)) {
            // Recording one more task would assert inside new_entry(). Abandon the
            // Definition instead, so the run fails by name at graph_commit rather
            // than on a hard assert here.
            LOG_WARN(
                "[GraphExecution] recording hazard map exhausted at sub-task %d (%d entries); Graph abandoned",
                task_index, GRAPH_RECORD_TENSORMAP_POOL_SIZE
            );
            recording.unsupported = true;
        } else {
            auto emit_inferred = [&add_fanin, task_index](TaskId producer) -> bool {
                // Only a task of this body can be an edge in the Definition. A GLOBAL
                // producer is a task submitted before the Graph, and a PARAM "producer" is
                // a boundary parameter, which names no task at all -- the outer shell was
                // submitted through the ordinary path against this same boundary, so its
                // own fanin already orders the whole body behind whatever produced it, and
                // the Definition carries no edge of its own. Reading either one's low bits
                // as a task index would invent an edge onto whichever task happens to sit
                // there.
                if (producer.space() != TaskId::Space::SUB_TASK) return true;
                const int32_t producer_index = producer.local_id();
                if (producer_index < task_index) {
                    add_fanin(producer_index);
                }
                return true;
            };
            (void)compute_task_fanin(dep_inputs, recording.tensor_map, manual_scope, emit_inferred);
            register_task_outputs(dep_inputs, task_id, recording.tensor_map, manual_scope);
        }
    }
    for (uint32_t i = 0; i < args.explicit_dep_count(); ++i) {
        const TaskId dep = args.explicit_dep(i);
        if (!dep.is_valid()) {
            recording.unsupported = true;
            continue;
        }
        if (dep.space() == TaskId::Space::PARAM) {
            // A dependency named off a boundary parameter -- the shape an orchestration
            // writes as set_dependencies({param.owner_task_id}). It orders the body behind
            // whatever produced that argument, and the outer shell already carries exactly
            // that ordering through its own args, so the Definition needs no edge of its
            // own. Admitted without one, like the GLOBAL case below.
            debug_assert(
                dep.local_id() < recording.bound_boundary().params.tensor_count() &&
                "a PARAM id names a parameter of the boundary it was stamped from"
            );
            continue;
        }
        if (dep.is_global()) {
            // A body reaches a GLOBAL id only through a variable it did not receive: every
            // tensor it can name carries a PARAM owner, handled above. So this is the
            // explicit-dependency form of using something that never came through the
            // boundary, and it is refused for the same reason -- the Definition has no edge
            // that could express it.
            LOG_WARN(
                "[GraphExecution] sub-task %d depends on a task outside the Graph; order the body behind it by "
                "passing that task's output as a GraphTaskArgs parameter",
                task_index
            );
            recording.unsupported = true;
            continue;
        }
        const int32_t dep_index = dep.local_id();
        if (dep_index >= task_index) {
            // A task of this body that is not yet recorded: the Definition's edges are
            // acyclic by construction, so a forward reference cannot be expressed.
            recording.unsupported = true;
            continue;
        }
        add_fanin(dep_index);
    }

    task.fanin_count = static_cast<int32_t>(recording.internal_fanins.size() - task.fanin_offset);
    // Published last: until this advances, the slot is not part of the recording, so
    // nothing that scans the recorded tasks can see the task being built.
    recording.task_count = task_index + 1;
    ORCH_PHASE_END(HostPhaseKind::OrchRecordSubTask, TaskId::to_uint64(task_id));
    return result;
}

GraphScopeResult OrchestratorState::graph_begin(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash) {
    if (!require_device_arguments(this, args)) return {};
    ORCH_PHASE_START_SPANNING();
    const GraphScopeResult result = graph_begin_inner(graph_key, args, callable_hash);
    ORCH_PHASE_END_SPANNING(HostPhaseKind::OrchGraphBegin, graph_key);
    return result;
}

GraphScopeResult
OrchestratorState::graph_begin_inner(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash) {
    auto *orch = this;
    GraphScopeResult result;
    GraphHostState *state = graph_state_from(orch);
    if (state == nullptr || !rt_graph_args_cacheable(args) || args.explicit_dep_count() != 0) {
        debug_assert(args.explicit_dep_count() == 0 && "Graph boundary explicit dependencies are not supported");
        return result;
    }
    if (GraphRecording *active = active_graph_recording(orch); active != nullptr) {
        active->unsupported = true;
        debug_assert(active == nullptr && "Nested Graph recording is not supported");
        LOG_WARN("%s", "[GraphExecution] nested Graph recording is not supported");
        return result;
    }

    const uint64_t full_key = graph_full_key(callable_hash, graph_key);
    std::unique_lock<std::mutex> lock(state->recording_mutex);

    // A published Definition is immutable, so the cache lookup comes first and
    // answers regardless of what else is recording. Gating it on an idle recorder
    // would make an already-built Definition wait for an unrelated one.
    auto definition_it = state->definitions.find(full_key);
    if (definition_it != state->definitions.end()) {
        TaskId submitted = TaskId::invalid();
        ORCH_PHASE_START();
        if (graph_submit_definition(
                orch, state, graph_record_definition(*state, definition_it->second), definition_it->second, args,
                &submitted
            )) {
            result.execute_block = false;
            result.task_id = submitted;
            ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, TaskId::to_uint64(submitted));
#if SIMPLER_DFX
            g_orch_submit_idx++;
#if SIMPLER_ORCH_PROFILING
            g_orch_submit_count++;
#endif
#endif
        }
        return result;
    }

    // This key is already recording: publish another zero-heap shell against it.
    // A recording that ended has its Definition in the cache, so reaching here
    // with a spent status means the recording failed and this key is spent.
    auto inflight_it = state->inflight.find(full_key);
    if (inflight_it != state->inflight.end()) {
        GraphInflightRecording &entry = *inflight_it->second;
        if (entry.status() != GraphRecordingStatus::RECORDING || !graph_boundary_matches(entry.boundary, args)) {
            return result;
        }
        TaskId submitted = TaskId::invalid();
        ORCH_PHASE_START();
        if (graph_submit_pending_definition(orch, state, full_key, args, &submitted)) {
            result.execute_block = false;
            result.task_id = submitted;
            ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, TaskId::to_uint64(submitted));
#if SIMPLER_DFX
            g_orch_submit_idx++;
#if SIMPLER_ORCH_PROFILING
            g_orch_submit_count++;
#endif
#endif
        }
        return result;
    }

    if (state->claimed_definitions() >= GRAPH_MAX_DEFINITIONS) {
        debug_assert(
            state->claimed_definitions() < GRAPH_MAX_DEFINITIONS &&
            "Graph Definition cache exceeds the supported per-worker limit"
        );
        LOG_WARN(
            "[GraphExecution] Definition cache is full (%zu published, %zu in flight); using ordinary path",
            state->definitions.size(), state->inflight.size()
        );
        return result;
    }

    // Only the boundary is captured here. The recorded body's storage belongs to the
    // recorder thread that picks the job up (bound at graph_prepare), so this path
    // allocates the boundary copy and nothing else — a megabyte-scale hazard map stood
    // up here would sit on the submitting thread, between two outer shells.
    auto entry = std::make_unique<GraphInflightRecording>();
    entry->full_key = full_key;
    // The boundary is built once, here, and only read afterwards -- by the recorder that
    // picks this entry up, and by later same-key submissions comparing against it.
    //
    // Tensors are filled before any TensorRef is made to point at them: `tensors` is a
    // fixed-size array precisely so those pointers cannot move, but a slot must still hold
    // its value before args names it.
    GraphBoundary &boundary = entry->boundary;
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        boundary.tensors[i] = args.tensor(i).ref();
    }
    // Each parameter is named by reference, so the tensors below stay the storage `params`
    // reads through.
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        simpler::hbg::Tensor &owned = boundary.tensors[i];
        switch (args.tag(i)) {
        case TensorArgType::INPUT:
            boundary.params.add_input(owned);
            break;
        case TensorArgType::OUTPUT_EXISTING:
            boundary.params.add_output(owned);
            break;
        case TensorArgType::INOUT:
            boundary.params.add_inout(owned);
            break;
        case TensorArgType::NO_DEP:
            boundary.params.add_no_dep(owned);
            break;
        case TensorArgType::OUTPUT:
            // GraphTaskArgs::add_output rejects a TensorCreateInfo at compile time, so no
            // boundary carries this tag. The case exists because the switch is exhaustive.
            debug_assert(false && "a Graph boundary cannot hold a runtime-allocated output");
            break;
        }
    }
    // Values resolved, declarations carried over. A dynamic parameter names itself, so
    // `params` stays the basis recording resolves against.
    boundary.params.gen_scalar_params_from_args(args);
    boundary.match_info.scalars.resize(static_cast<size_t>(boundary.params.scalar_count()));
    graph_boundary_capture_scalars(boundary.match_info.scalars.data(), boundary.params);
    boundary.params.launch_spec = args.launch_spec;
    boundary.params.set_allow_early_resolve(args.allow_early_resolve());
    if (args.task_timing_slot() != TASK_TIMING_SLOT_NONE) {
        boundary.params.set_task_timing_slot(args.task_timing_slot());
    }
    // Before anything reads these tensors: the parameters move into the recording's own
    // address space here, so every reader downstream -- the body, the recorder, a later
    // same-key submission -- sees one consistent set of addresses. `params` already names
    // them, and by reference, so it carries the move across.
    //
    // A boundary this runtime cannot represent takes the ordinary path, like a structural
    // mismatch does: the entry is dropped before it is published, the body runs against the
    // caller's own arguments, and the run stays correct without a Definition.
    if (!graph_boundary_relocate_params(boundary)) {
        LOG_WARN(
            "%s", "[GraphExecution] boundary tensors must name non-empty buffers, and two at one address must "
                  "name one size; using ordinary path"
        );
        return result;
    }
    boundary.params.set_predicate(args.predicate());
    GraphInflightRecording *entry_ptr = entry.get();
    state->inflight.emplace(full_key, std::move(entry));
    state->inflight_count.store(state->inflight.size(), std::memory_order_release);

    TaskId submitted = TaskId::invalid();
    ORCH_PHASE_START();
    if (graph_submit_pending_definition(orch, state, full_key, args, &submitted)) {
        result.execute_block = false;
        result.recording = true;
        result.recording_handle = entry_ptr;
        result.params = &entry_ptr->boundary.params;
        result.task_id = submitted;
        ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, TaskId::to_uint64(submitted));
#if SIMPLER_DFX
        g_orch_submit_idx++;
#if SIMPLER_ORCH_PROFILING
        g_orch_submit_count++;
#endif
#endif
    } else {
        state->inflight.erase(full_key);
        state->inflight_count.store(state->inflight.size(), std::memory_order_release);
    }
    return result;
}

// The parameter list is the entry's own -- graph_begin published it as
// GraphScopeResult::params, and both the queued job and the synchronous fallback forward
// that same object here, which is why it arrives unnamed: there is no second boundary for
// it to agree with. A later same-key submission does arrive with the caller's own args,
// and graph_begin_inner compares those against this entry before publishing a shell
// against it.
bool OrchestratorState::graph_prepare(void *recording_handle, const GraphTaskArgs &) {
    GraphHostState *state = graph_state_from(this);
    if (state == nullptr || recording_handle == nullptr || g_active_graph_recording != nullptr) return false;
    auto *entry = static_cast<GraphInflightRecording *>(recording_handle);
    // graph_begin published this entry before the private job was enqueued, and
    // the entry's address is stable for as long as the recording lives, so the
    // recording thread reaches its own state without searching for it. Until this
    // thread calls graph_end/graph_abort, later graph_begin calls only read the
    // boundary under recording_mutex, and only this thread writes the
    // fields it binds below. Taking that mutex here lets the main thread's
    // same-key submit burst starve prepare and collapse the intended overlap, so
    // the status read goes through the atomic instead.
    if (entry->status() != GraphRecordingStatus::RECORDING) {
        return false;
    }
    // This thread's own storage, emptied rather than allocated -- see
    // recorder_recording(). Failure is reachable only on this thread's first recording,
    // where the hazard map is stood up; the caller then aborts the recording, and the
    // outer shell it already submitted replays nothing.
    GraphRecording &recording = recorder_recording();
    if (!graph_recording_reset(recording, *entry)) {
        LOG_WARN("%s", "[GraphExecution] recording hazard map allocation failed; recording abandoned");
        return false;
    }
    g_active_graph_entry = entry;
    g_active_graph_recording = &recording;
    g_active_graph_owner = state;
    return true;
}

void OrchestratorState::graph_abort(void *recording_handle) {
    GraphHostState *state = graph_state_from(this);
    auto *entry = static_cast<GraphInflightRecording *>(recording_handle);
    if (state == nullptr || entry == nullptr) return;
    {
        std::scoped_lock lock(state->recording_mutex);
        entry->set_status(GraphRecordingStatus::FAILED);
    }
    // The storage outlives the entry it was bound to, and graph_commit destroys the
    // entries, so leaving the pointer behind parks a stale one in thread_local state for
    // the rest of the process. The next graph_prepare rebinds before anything reads it,
    // which is why this is hygiene rather than a fix -- but clearing it turns a dangling
    // boundary into a null one, and null is the only state bound_boundary()'s assertion
    // can catch a read outside a recording by.
    unbind_recorder_boundary();
    g_active_graph_entry = nullptr;
    g_active_graph_recording = nullptr;
    g_active_graph_owner = nullptr;
    state->recording_cv.notify_all();
}

// Finish the background recording and publish the Definition. The main
// thread finalizes the already-submitted outer Graph tasks in graph_commit.
//
// Retires the entry it bound on every path below that has one, so a caller never has
// to pair a `false` return with an abort — and must not: graph_commit frees a drained
// entry after releasing recording_mutex, so a second abort would take that mutex and
// still touch freed memory.
bool OrchestratorState::graph_end() {
    GraphHostState *state = graph_state_from(this);
    GraphRecording *recording = active_graph_recording(this);
    GraphInflightRecording *entry = g_active_graph_entry;
    if (state == nullptr || recording == nullptr || entry == nullptr) return false;

    // A fatal latched anywhere in the run ends this pass with no Definition, but the
    // entry still has to leave RECORDING: graph_commit's drain blocks until every
    // in-flight entry has, and nothing else transitions this one. Returning early
    // instead would park the entry — and this thread's recorder thread_locals — for
    // the rest of the process, and hang the bind that is already failing.
    if (is_fatal()) {
        graph_abort(entry);
        return false;
    }

    ORCH_PHASE_START();
    std::optional<GraphDefinition> layout = graph_layout_definition(*recording);
    // The claim is what decides where this thread writes, so it precedes the fill
    // and never moves the arena: a Definition the retained capacity cannot hold is
    // built in a buffer of its own and copied at upload instead, which keeps the
    // run correct while the next one's arena is sized for it.
    GraphDefinitionRecord record;
    std::byte *image = nullptr;
    if (layout.has_value()) {
        record.bytes = layout->total_bytes;
        if (std::optional<size_t> offset = state->reserve_object(layout->total_bytes); offset.has_value()) {
            record.object_offset = *offset;
            image = state->image_at(*offset);
        } else {
            record.spill.assign(layout->total_bytes, std::byte{0});
            image = record.spill.data();
        }
    }
    const bool built = layout.has_value() && graph_fill_definition(*recording, *layout, image);
    if (built) {
        // The boundary the reuse condition is checked against. Copied because the entry
        // holding it is drained at graph_commit while the Definition stays in the cache.
        // The counts are not copied: they are in the image, where the device checks this
        // invocation's argument counts against them.
        record.boundary_match_info = recording->bound_boundary().match_info;
        ORCH_PHASE_END(HostPhaseKind::OrchBuildDefinition, recording->task_count);
    }
    const GraphDefinition *header = built ? graph_record_definition(*state, record) : nullptr;
    if (header == nullptr) {
        debug_assert(false && "The recorded Graph contains a construct that Graph Execution does not support");
        LOG_WARN("%s", "[GraphExecution] asynchronous recording produced an unsupported Graph");
        graph_abort(entry);
        return false;
    }
    LOG_DEBUG(
        "[GraphExecution] define key=0x%llx tasks=%d bytes=%u", static_cast<unsigned long long>(header->full_key),
        header->task_count, header->total_bytes
    );
    bool ready = false;
    {
        std::scoped_lock lock(state->recording_mutex);
        if (entry->status() != GraphRecordingStatus::RECORDING || entry->full_key != header->full_key) {
            entry->set_status(GraphRecordingStatus::FAILED);
        } else {
            state->definitions.emplace(header->full_key, std::move(record));
            entry->set_status(GraphRecordingStatus::READY);
        }
        ready = entry->status() == GraphRecordingStatus::READY;
    }
    unbind_recorder_boundary();
    g_active_graph_entry = nullptr;
    g_active_graph_recording = nullptr;
    g_active_graph_owner = nullptr;
    state->recording_cv.notify_all();
    return ready;
}

// Join every recording in flight and back-patch all deferred shells in submit
// order. Orchestration completion is the only normal-path barrier.
void OrchestratorState::graph_commit() {
    ORCH_PHASE_START_SPANNING();
    graph_commit_inner();
    ORCH_PHASE_END_SPANNING(HostPhaseKind::OrchGraphCommit, 0);
}

void OrchestratorState::graph_commit_inner() {
    if (active_graph_recording(this) != nullptr) return;
    GraphHostState *state = graph_state_from(this);
    if (state == nullptr || state->inflight_count.load(std::memory_order_acquire) == 0) return;

    std::unordered_map<uint64_t, std::unique_ptr<GraphInflightRecording>> drained;
    {
        std::unique_lock<std::mutex> lock(state->recording_mutex);
        if (state->inflight.empty()) return;
        {
            ORCH_PHASE_START();
            state->recording_cv.wait(lock, [&]() {
                return !state->any_recording();
            });
            ORCH_PHASE_END(HostPhaseKind::OrchRecordingWait, state->inflight.size());
        }
        drained.swap(state->inflight);
        state->inflight_count.store(0, std::memory_order_release);
    }

    uint64_t failed_key = 0;
    bool failed = false;
    for (const auto &[key, entry] : drained) {
        auto definition_it = state->definitions.find(key);
        if (entry->status() == GraphRecordingStatus::READY && definition_it != state->definitions.end() &&
            graph_record_definition(*state, definition_it->second) != nullptr) {
            continue;
        }
        if (!failed) failed_key = key;
        failed = true;
    }
    if (!failed && !graph_finalize_pending_submissions(this, state, &failed_key)) failed = true;
    if (failed) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "failed to finalize asynchronous Graph key=%#llx",
            static_cast<unsigned long long>(failed_key)
        );
    }
}
