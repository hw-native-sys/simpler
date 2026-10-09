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

#include <stdint.h>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "common/args_dump_task_metadata.h"
#include "host_build_graph/graph_boundary_match.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/submit_types.h"
#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"
#include "task_interface/assert_compat.h"

// The vocabulary of a Graph recording: what a body is captured into, and what the capture
// is captured from. A recording is thread-owned storage reset per body rather than
// allocated per body, so the types here are sized to the caps once and reused.

// A recorded task's dispatch predicate, held as the operand tensor plus the element
// index within it rather than the absolute address submit would resolve. The tensor is
// copied because the caller only lends it for the duration of the submit call.
struct GraphRecordedPredicate {
    simpler::hbg::Tensor operand;
    uint64_t elem_offset{0};
    int64_t target{0};
    uint8_t elem_size{0};
    PredicateOp op{PredicateOp::NONE};
};

struct RecordedSubTask {
    std::array<int32_t, SUBTASK_SLOT_COUNT> kernel_ids{};
    ActiveMask active_mask{};
    TaskAttrs task_attrs{};
    int16_t logical_block_num{1};
    int16_t total_required_subtasks{0};
    size_t total_output_size{0};
    uintptr_t record_packed_base{0};
    // This task's slice of the recording's tensor pool, as an offset so the task
    // carries no address into storage the recording owns. The element addresses are handed
    // to the caller through TaskOutputTensors and have to stay valid while the rest of the
    // body records, which the pool satisfies by being allocated at the cap and never growing.
    int32_t tensor_offset{0};
    int32_t tensor_count{0};
    int32_t scalar_offset{0};
    int32_t scalar_count{0};
    int32_t fanin_offset{0};
    int32_t fanin_count{0};
    // Index into the recording's predicates, or -1 when the task carries none.
    int32_t predicate_index{-1};
    ArgsDumpTaskMetadata dump_metadata;

    // Restore the state a freshly recorded task has, in a slot the previous body left
    // behind. A reused slot that keeps any field of the previous body
    // records a Definition that body never had, and predicate_index and dump_metadata are
    // written only on the paths that have one, so neither can be left to the fill.
    //
    // Field by field rather than `*this = RecordedSubTask{}`: the latter is immune to
    // fields added later, but it costs a second write of the whole struct on every recorded
    // task and measured 350-700 us per bind on dsv4's 1679 tasks. The static_assert below is
    // the cheap half of that guarantee -- adding a field breaks the build here, which is
    // where the reader is told to extend this function.
    void reset() {
        kernel_ids = {};
        active_mask = {};
        task_attrs = {};
        logical_block_num = 1;
        total_required_subtasks = 0;
        total_output_size = 0;
        record_packed_base = 0;
        tensor_offset = 0;
        tensor_count = 0;
        scalar_offset = 0;
        scalar_count = 0;
        fanin_offset = 0;
        fanin_count = 0;
        predicate_index = -1;
        dump_metadata = {};
    }
};

// reset() above lists this struct's fields by hand, and a field it forgets is carried
// from the previous body into the next recording -- silently, as a Definition that body
// never had. Adding a field changes this size, so the build stops here instead.
static_assert(
    sizeof(RecordedSubTask) == 104, "RecordedSubTask gained or lost a field: extend reset() to match, then "
                                    "update this size"
);

// The Graph boundary as the submitting thread captured it, deep-copied because the caller
// only lends its arguments for the duration of the submit call.
//
// This is the boundary: the recorded body reads it, later same-key submissions compare
// against it, and a task slot that follows one of its parameters names a slot in `params`.
// Everything is written once, by the submitting thread in graph_begin, and only read
// afterwards -- the recorder never writes here.
//
// `tensors` is the storage `params` points into: a TensorRef holds a Tensor*, so holding a
// GraphTaskArgs is not the same as owning its tensor data. The array is fixed-size so
// those pointers cannot be invalidated by a reallocation.
struct GraphBoundary {
    // Deliberately user-provided rather than `= default`: a defaulted constructor here is
    // trivial, so make_unique<GraphInflightRecording>()'s value-initialization would zero
    // all 13.3 KB of `tensors` on the submitting thread. A user-provided one leaves the
    // array default-initialized -- Tensor is trivially default constructible, so those
    // pages cost nothing until a Graph writes the tensors it actually has.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    GraphBoundary() {}

    // Each tensor carries a recording-space address and an owner of
    // TaskId::Space::PARAM naming its own index; graph_begin writes both over the caller's
    // after copying the geometry, which is what makes a recording a closed address space.
    // Every other field is the caller's, so a later same-key submission compares against
    // these directly -- size, origin, geometry and flags are all still what the argument
    // had, and the alias partition is computed within each set rather than across them.
    std::array<simpler::hbg::Tensor, GRAPH_MAX_TENSOR_ARGS> tensors;
    // A dynamic parameter here names itself, a static one names nothing -- so this list is
    // the basis recording resolves against: scalar(i) folds to &params.scalars_[i] either
    // way, and graph_classify_scalars turns that into the index i. A parameter that named
    // the caller's variable instead would hand out an address outside this array, the task
    // slot following it would be recorded as static, and it would silently stop being
    // refreshed on replay.
    GraphTaskArgs params;

    // What a later invocation of this key is checked against. The match path walks the
    // whole of it on every same-key submission, so it carries the compared fields directly
    // and each half is sized to the boundary, not to the cap. Held apart from `params`
    // because a published Definition outlives this boundary and needs its own copy.
    GraphBoundaryMatchInfo match_info;
    // Bytes the parameters occupy in the recording's space. A sub-task's outputs are
    // bumped from GRAPH_RECORD_BASE + this, so the two regions of one simulated heap do
    // not overlap.
    uint64_t param_used_heap_size{0};
};

// Storage for one recorded body, owned by the recorder thread and reset per
// recording rather than allocated per recording — see recorder_recording().
struct GraphRecording {
    uint64_t full_key{0};
    uint64_t next_virtual_offset{0};
    // The in-flight entry's boundary copy, bound at graph_prepare and valid until
    // graph_end/graph_abort. Not owned here: the submitting thread reads it for
    // boundary matching while this thread records.
    const GraphBoundary *boundary{nullptr};
    bool unsupported{false};
    std::vector<RecordedSubTask> tasks;
    // How many of `tasks` this recording has filled. The array itself is never cleared
    // and graph_recording_reserve_storage sizes it to the sub-task cap, so a body is
    // recorded into slots that already exist: a recorded task makes no allocation at all.
    // An over-cap body is marked unsupported but keeps recording so it can finish, so this
    // is not bounded by SUB_TASK_MAX_NUM while it runs -- what bounds it is the storage
    // each further task claims, which exhausts memory long before the counter's range.
    int32_t task_count{0};
    // Every recorded task's tensor arguments, packed end to end in one region this
    // recording bumps through, and the reason a task holds an offset rather than its own
    // buffer: a body's tensors then occupy the bytes they need instead of a page per task
    // (a per-task buffer at the cap is 32 x 128 B = exactly one page, so dsv4's 1679 tasks
    // touched 1679 pages to hold ~210 KB). Allocated once per thread at the cap and never
    // grown, which is what keeps a task's borrowed element addresses valid for the rest of
    // the recording. `new[]` default-initializes a trivially-default-constructible Tensor,
    // so the region costs no page until a body writes one; each element a task uses is
    // value-initialized before it is filled.
    std::unique_ptr<simpler::hbg::Tensor[]> task_tensor_pool;
    int32_t task_tensor_cursor{0};
    // Flat per-task arrays, indexed by the ranges on RecordedSubTask. Held here rather
    // than on each recorded task so recording a graph pays no allocation per task per array,
    // and reserved to the sub-task cap by graph_recording_reserve_storage so it pays no
    // growth either.
    std::vector<uint64_t> scalars;
    // The wire form directly: recording resolves an inherited slot into a boundary
    // parameter index as it classifies, so there is no host-side kind left to translate.
    std::vector<GraphScalarInheritance> scalar_inheritance;
    std::vector<int32_t> internal_fanins;
    // Indexed by RecordedSubTask::predicate_index; only predicated tasks
    // contribute an entry.
    std::vector<GraphRecordedPredicate> predicates;
    // Hazard state for the recorded body, owned per recorder thread because
    // several graphs record at once, each on its own thread.
    //
    // The ordinary submit path reads a task's producers out of orch->tensor_map
    // (compute_task_fanin, STEP 3) and publishes the task's writes back into it
    // (register_task_outputs, STEP 4). The shadow-record path replaces
    // submit_task_common wholesale, so without a map of its own the recorder
    // can only see the edges tensor-source classification yields — and that
    // classification answers "which recorded task's packed window holds these bytes",
    // i.e. who ALLOCATED the buffer, never who wrote it last. A body that
    // allocates once with alloc_tensors and then writes in place with add_inout
    // (the shape every generated orchestration uses) would therefore record an
    // sub-task with no edge to its actual producer, and the Definition would replay
    // a DAG the same body never had when submitted task by task.
    ChipTensorMap tensor_map{};
    // Set once both the hazard map and the tensor pool are up, and only then: the two
    // allocate, so a flag set by the first would let a thread whose second allocation
    // failed skip the stand-up on its next recording and record through a null pool.
    bool storage_ready{false};
    // Scope depth as the body sees it. begin_scope/end_scope leave the real
    // orchestrator stack untouched while recording (a Graph replays flat), but
    // the manual-scope flag still has to follow the body: a manual scope
    // suppresses inference on the ordinary path, so it must suppress it here too.
    int32_t scope_stack_top{-1};
    int32_t manual_begin_depth{CHIP_MAX_SCOPE_DEPTH};

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    simpler::hbg::Tensor *task_tensors(const RecordedSubTask &task) const {
        return task_tensor_pool.get() + task.tensor_offset;
    }

    // The entry's boundary this recording is bound to. graph_prepare binds it and
    // graph_end/graph_abort clears it, so every read below sits between those two points:
    // a body only runs once graph_prepare has succeeded, and graph_layout_definition runs
    // before the unbind.
    const GraphBoundary &bound_boundary() const {
        debug_assert(boundary != nullptr && "a recording reads its boundary only while bound");
        return *boundary;
    }
};

// Entry capacity for one recorded body's hazard map. A Definition is capped at
// SUB_TASK_MAX_NUM tasks and each recorded task registers at most its INOUT/OUTPUT_EXISTING
// args, so this bounds the worst realistic body while staying a small fraction of
// the ordinary path's whole-orchestration pool (CHIP_TENSORMAP_POOL_SIZE). Exhausting
// it marks the recording unsupported, which graph_commit reports as
// SIMPLER_ERROR_INVALID_ARGS -- the outer shell is already submitted by then, so there
// is no ordinary-path fallback left to take.
constexpr int32_t GRAPH_RECORD_TENSORMAP_POOL_SIZE = 16384;

// Elements in the recording's tensor pool: every sub-task a body can hold, times
// every tensor argument one such task can carry. An in-cap body therefore always fits, and
// the bump cursor is checked anyway because a body that overshoots SUB_TASK_MAX_NUM keeps
// recording so it can finish.
constexpr size_t GRAPH_RECORD_TENSOR_POOL_ELEMS =
    static_cast<size_t>(SUB_TASK_MAX_NUM) * static_cast<size_t>(CORE_MAX_TENSOR_ARGS);

// The graph_local_id a recorded task's SUB_TASK id carries. A recorded task belongs
// to no Graph task yet -- every shell replaying the Definition re-mints the id with
// its own local id at materialize -- so record time names a task by its index alone,
// and the id's low field is that index and nothing else. That is what keeps the
// index inside the SUB_TASK_MAX_NUM task chains the recording's hazard map is
// dimensioned for.
constexpr int32_t GRAPH_RECORD_NO_OWNING_GRAPH = 0;

// Turn one task's scalar slots into wire source refs. A slot is BOUNDARY exactly when it
// inherits a parameter of THIS recording's boundary, which the subtraction below both
// converts to an index and proves; everything else is static Definition data.
//
// Recording is where this must happen: an origin is only valid while the body runs, and
// the Definition it feeds has to be position-independent.
template <typename ArgT>
void graph_classify_scalars(GraphRecording &recording, const ArgT &args, int32_t scalar_offset) {
    const GraphTaskArgs &params = recording.bound_boundary().params;
    const uintptr_t base = reinterpret_cast<uintptr_t>(params.scalar_slot_base());
    const uintptr_t span = static_cast<uintptr_t>(params.scalar_count()) * sizeof(uint64_t);
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        GraphScalarInheritance &ref = recording.scalar_inheritance[scalar_offset + i];
        if (!args.scalar_dynamic(i)) {
            ref = GraphScalarInheritance::self_value();
            continue;
        }
        // Integer arithmetic, not pointer comparison: relational operators on pointers are
        // only defined within one array object, and an origin outside this boundary's slot
        // array is admitted here. The origin is never dereferenced -- it may already point
        // at a caller local that has gone out of scope.
        const uintptr_t origin = reinterpret_cast<uintptr_t>(args.scalar_origin(i));
        if (origin < base || origin - base >= span || (origin - base) % sizeof(uint64_t) != 0) {
            // A parameter whose origin is not one of this boundary's own is static
            // Definition data: the Definition's inheritance entries index this boundary
            // alone, so no other slot can be refreshed on replay, and the value the handle
            // already carried into this slot is what the image should hold.
            // GRAPH_EXECUTION.md states this as part of the boundary-scalar contract.
            ref = GraphScalarInheritance::self_value();
            continue;
        }
        ref = GraphScalarInheritance::from_boundary(static_cast<uint16_t>((origin - base) / sizeof(uint64_t)));
    }
}

bool graph_boundary_relocate_params(GraphBoundary &boundary);

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key);

GraphRecording &recorder_recording();

bool graph_recording_stand_up(GraphRecording &recording);

/**
 * Stand the calling thread's recording storage up — hazard map, sub-task slots, the
 * flat per-task arrays and the task tensor pool — without recording anything.
 *
 * graph_recording_stand_up() against this thread's own recorder_recording(), with the
 * failure counted rather than only returned.
 *
 * A recorder worker calls this once as it starts, so the allocations land at callable
 * registration rather than inside the first bind that worker serves, and a failure is
 * reported where the caller can still act on it. It is an optimization, not the only
 * stand-up point: a worker the pool creates after prewarm, and a thread whose storage was
 * dropped for overshooting the sub-task cap, still stand up lazily on their next recording.
 *
 * @return false when an allocation failed; the failure is also counted for
 *         graph_recorder_storage_failures(), which is how the host notices across the
 *         .so boundary that carries no return value.
 */
bool graph_recorder_stand_up_storage();

/** Stand-up failures since the process started. Monotonic. */
size_t graph_recorder_storage_failures();

bool graph_classify_tensor(const GraphRecording &recording, int32_t task_index, const simpler::hbg::Tensor &tensor);

std::optional<GraphDefinition> graph_layout_definition(const GraphRecording &recording);

bool graph_fill_definition(const GraphRecording &recording, GraphDefinition definition, std::byte *image);

const GraphDefinition *graph_definition(const std::byte *data, size_t bytes);
