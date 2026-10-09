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

#include "host_build_graph/graph_recording.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>
#include <vector>

#include "common/unified_log.h"
#include "host_build_graph/graph_cache.h"
#include "host_build_graph/graph_host_state.h"

namespace {

// Whether a tensor the body derived from boundary parameter `param_index` is one this
// parameter can stand for.
//
// Against that one parameter and no other: the owner says which parameter this came from,
// and a view op copies `buffer` wholesale while changing only `start_offset`, so anything
// legitimately derived from a parameter still carries that parameter's buffer. Comparing
// the buffer is therefore the bounds check -- an address that moved cannot have come from
// a view, and replay resolves the buffer through this parameter alone.
//
// The origin needs no comparison: it travels absolutely, and the argument's own origin is
// part of the reuse condition, so a call that shifted it does not
// reach a replay at all.
bool graph_tensor_from_boundary(
    const GraphRecording &recording, const simpler::hbg::Tensor &tensor, int32_t param_index
) {
    const GraphTaskArgs &params = recording.bound_boundary().params;
    if (param_index < 0 || param_index >= params.tensor_count()) return false;
    const simpler::hbg::Tensor &param_tensor = params.tensor(param_index).ref();
    return tensor.buffer.addr == param_tensor.buffer.addr && tensor.buffer.size == param_tensor.buffer.size;
}

// Stand the recording's hazard map up on its own allocation. Failure is
// reported to the caller, which abandons the recording rather than producing a
// Definition with inferred edges missing.
bool graph_recording_init_tensor_map(GraphRecording &recording) {
    return recording.tensor_map.init(CHIP_TENSORMAP_NUM_BUCKETS, GRAPH_RECORD_TENSORMAP_POOL_SIZE, SUB_TASK_MAX_NUM);
}

// Stand this thread's retained storage up at the cap, so no body it records grows any of
// it and no recorded task allocates.
//
// Capacity kept across recordings is otherwise the high-water mark of the bodies this
// thread happened to record, and which body a thread gets is decided by one FIFO the
// recorder pool's workers all wait on (GraphAsyncRecordingState::start notifies one
// worker; whichever wakes takes the next job). So the assignment is not stable across
// binds: a thread that recorded a narrow body first extends its slots and reallocates
// every array the first time a wider one lands on it, on whatever bind that happens to
// be. Measured on dsv4, whose eight Definitions differ in size, a warm bind still created
// 1336 of the 1679 `tasks` slots that body needed. Standing everything up at the cap makes a
// thread's storage independent of the order it saw bodies in.
//
// Each bound is a per-task cap times the sub-task cap, so these are the recorded
// body's own limits rather than a worst case invented here: the tensor pool is one entry
// per tensor argument (CORE_MAX_TENSOR_ARGS), the two scalar
// arrays one per scalar argument (CORE_MAX_SCALAR_ARGS), and predicates at
// most one per task.
//
// internal_fanins is the one array left growing, and the reason is the size it grows to
// rather than the bound it could reach. It has no per-sub-task cap: CHIP_MAX_FANIN
// bounds a global task's inline fanin, but a sub-task's producers travel in the
// Definition's own CSR, which the scheduler reads directly, so the only limits are uint16
// producer indices and each producer being an earlier task of the same body — a structural
// 1024 x 1023 / 2 edges, 4.2 MB.
// What decides whether growth costs anything is not that bound but whether a reallocation
// crosses glibc's mmap threshold, since a freed block below it is reused off the heap
// without re-faulting (see the entry cited above). A dsv4 body holds ~630 edges, 5 KB, two
// orders of magnitude under the threshold — so buying 4.2 MB of address space per recorder
// thread for it would be sizing an array to a worst case, which is the opposite of what
// the reservations above do. Re-decide this with a measurement if a workload's bodies ever
// get dense enough to push it past ~128 KB.
//
// The tensor pool is 4 MB and the rest ~1.3 MB per recorder thread, next to the 2.17 MB
// hazard map. Only the pool's used prefix ever becomes resident: it is default-initialized
// and a body writes the bytes its tensors need, contiguously.
//
// Returns false when the pool cannot be allocated, which the caller treats like a hazard
// map it could not stand up.
bool graph_recording_reserve_storage(GraphRecording &recording) {
    constexpr size_t kSubTaskCap = static_cast<size_t>(SUB_TASK_MAX_NUM);
    recording.task_tensor_pool.reset(new (std::nothrow) simpler::hbg::Tensor[GRAPH_RECORD_TENSOR_POOL_ELEMS]);
    if (recording.task_tensor_pool == nullptr) return false;
    recording.tasks.resize(kSubTaskCap);
    recording.scalars.reserve(kSubTaskCap * static_cast<size_t>(CORE_MAX_SCALAR_ARGS));
    recording.scalar_inheritance.reserve(kSubTaskCap * static_cast<size_t>(CORE_MAX_SCALAR_ARGS));
    recording.predicates.reserve(kSubTaskCap);
    return true;
}

// A recorded tensor as the relocation record graph_execution.h describes: the two fields
// replay has a base for are relative here, and which base follows the owner.
//
// A parameter's buffer is replaced wholesale by this call's argument, so its recorded
// address means nothing outside the recording that issued it and leaves as zero; its
// origin becomes the view's offset inside that parameter, which replay adds the
// argument's own origin to. A body tensor's buffer address becomes an offset into the
// recording's output region -- an affine image of the heap a replay commits -- and its
// origin is already its own.
//
// Between them the image holds no address from the recording's own space, and none from
// the caller's either: it is readable knowing only the heap it will be bound to and the
// arguments it will be bound against.
void graph_tensor_relocation_record(
    const GraphRecording &recording, const simpler::hbg::Tensor &tensor, simpler::hbg::TensorData *record
) {
    record->init_from(tensor);
    const TaskId owner = tensor.owner_task_id;
    if (owner.space() == TaskId::Space::PARAM) {
        record->buffer.addr = 0;
        // No view op lowers start_offset, so a tensor owned by parameter i has an origin at
        // or after that parameter's own. The subtraction below is unsigned.
        const uint64_t param_origin = recording.bound_boundary().params.tensor(owner.local_id()).ref().start_offset;
        debug_assert(record->start_offset >= param_origin);
        record->start_offset -= param_origin;
    } else {
        record->buffer.addr -= GRAPH_RECORD_BASE + recording.bound_boundary().param_used_heap_size;
    }
}

template <typename T>
bool graph_layout_section(size_t count, size_t *cursor, uint32_t *offset) {
    if (count == 0) {
        *offset = 0;
        return true;
    }
    if (*cursor > UINT32_MAX || count > UINT32_MAX / sizeof(T)) return false;
    const size_t aligned = (*cursor + alignof(T) - 1) & ~(alignof(T) - 1);
    const size_t bytes = count * sizeof(T);
    if (aligned > UINT32_MAX || bytes > UINT32_MAX - aligned) return false;
    *offset = static_cast<uint32_t>(aligned);
    *cursor = aligned + bytes;
    return true;
}

template <typename T>
T *graph_image_section(std::byte *image, uint32_t offset) {
    return offset == 0 ? nullptr : reinterpret_cast<T *>(image + offset);
}

}  // namespace

// Move a captured boundary's parameters into the recording's own address space: each
// tensor takes a recording-space address and an owner naming its own parameter index.
// Returns false when the boundary is not one this runtime can represent. No tensor is
// rewritten on that path, so the boundary still holds the caller's own addresses and owners.
//
// This is what closes the recording: after it, no address or provenance the caller owns is
// reachable from the body, so a Definition recorded from it is a faithful relocatable
// image. The caller's real addresses reach the device through the outer shell's own
// arguments instead.
//
// Parameters over one buffer land on one address, and parameters over different buffers
// on different ones. The shadow tensor map infers WAR/WAW edges by buffer address equality,
// so splitting one buffer across two addresses drops the edges between its views and
// replays a DAG the body never had, while merging two buffers onto one invents edges it
// never had. The partition graph_alias_partition settles is what makes both impossible.
//
// Runs on the submitting thread, before the entry is published, which is what keeps
// GraphBoundary written once and read-only thereafter.
bool graph_boundary_relocate_params(GraphBoundary &boundary) {
    boundary.param_used_heap_size = 0;
    const int32_t tensor_count = boundary.params.tensor_count();
    if (tensor_count <= 0) return true;

    // Built before the rewrite below, so it holds the caller's own geometry: this is what
    // a later invocation's arguments are compared against, and the rewrite only replaces
    // the buffer address and the owner, neither of which is compared.
    boundary.match_info.tensors.resize(static_cast<size_t>(tensor_count));
    for (int32_t i = 0; i < tensor_count; ++i) {
        boundary.match_info.tensors[i] = graph_boundary_tensor_match_of(boundary.tensors[i], boundary.params.tag(i));
    }

    std::array<uint16_t, GRAPH_MAX_TENSOR_ARGS> alias_rep{};
    if (!graph_alias_partition(boundary.params, alias_rep.data())) return false;

    // A parameter takes as much of the recording space as its argument takes of the real
    // heap, so the space stays a mirror of the heap a replay will need. Only a
    // representative claims room; the rest of its partition joins it at the same address.
    std::array<uint64_t, GRAPH_MAX_TENSOR_ARGS> base_of_rep{};
    uint64_t cursor = 0;
    for (int32_t i = 0; i < tensor_count; ++i) {
        simpler::hbg::Tensor &owned = boundary.tensors[i];
        const int32_t rep = alias_rep[i];
        boundary.match_info.tensors[i].alias_rep = alias_rep[i];
        if (rep == i) {
            base_of_rep[rep] = GRAPH_RECORD_BASE + cursor;
            cursor += CHIP_ALIGN_UP(owned.buffer.size, PACKED_OUTPUT_ALIGN);
        }
        owned.buffer.addr = base_of_rep[rep];
        owned.owner_task_id = TaskId::make_param(i);
    }
    boundary.param_used_heap_size = cursor;
    return true;
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

// The recorder thread's own storage for the body it is recording, and the reason none
// of this is allocated per recording.
//
// A recorder thread outlives the bind that first used it: the pool parks
// kPrewarmedWorkerCount threads at callable registration and keeps them across runs
// (GraphAsyncRecordingState in orchestration_api.h). Its storage now does too. What a
// recording needs is a hazard map of 2.17 MB and seven flat arrays; standing those up
// per recording made the first touch of every page a minor fault, and paid it again on
// every bind because the memory went back to the kernel in between — with the
// allocation itself sitting on the submitting thread inside graph_begin, between two
// outer shells. Held per thread instead, the pages fault once in the process's life and
// a recording starts by resetting what is already resident.
//
// This retains no content across recordings: reset() clears every array and empties the
// map. Only the pages stay.
//
// Safe as a thread_local because a thread records one body at a time: graph_prepare
// refuses to bind while this thread already has a recording active, so a Graph nested
// inside a recorded body takes the ordinary path rather than claiming this storage
// twice.
GraphRecording &recorder_recording() {
    static thread_local GraphRecording storage;
    return storage;
}

// Stand this thread's storage up once, or report that it could not be. Idempotent.
//
// Either allocation failing drops whatever the other one took, so the next attempt starts
// from nothing instead of finding the flag set and one of the two regions missing — the
// record path guards on that same flag, so a half-built storage would be recorded through.
bool graph_recording_stand_up(GraphRecording &recording) {
    if (recording.storage_ready) return true;
    try {
        if (!graph_recording_init_tensor_map(recording) || !graph_recording_reserve_storage(recording)) {
            recording = GraphRecording{};
            return false;
        }
    } catch (const std::bad_alloc &) {
        // The tensor pool is a nothrow new, but the flat arrays are vectors whose
        // resize/reserve throw. This also runs on a recorder worker as it starts, where an
        // escaping exception terminates the process instead of letting the pool's prewarm
        // report the failure.
        recording = GraphRecording{};
        return false;
    }
    recording.storage_ready = true;
    return true;
}

// Whether a tensor the body used is one this recording can represent, and against the
// thing it actually came from: the parameter its owner names, or the producing task's own
// output block. Both containment tests live here rather than at replay -- the recorder has
// those objects in front of it, and the device is then handed a tensor it can rebase
// without looking anything up.
bool graph_classify_tensor(const GraphRecording &recording, int32_t task_index, const simpler::hbg::Tensor &tensor) {
    const uintptr_t tensor_addr = static_cast<uintptr_t>(tensor.buffer.addr);
    const TaskId owner = tensor.owner_task_id;

    // Provenance decides which classification applies, not the address. A recording's
    // space starts just above zero, and a real device address is 48-bit, so the two
    // overlap: an object that entered the body without passing through the boundary can
    // hold an address inside the parameter region, and an address test would attribute it
    // to a parameter and bake a Definition that rebinds it to someone else's buffer.
    // owner_task_id is stamped by this recording -- PARAM on the boundary tensors, SUB_TASK
    // on what the body produced -- and views propagate it, so it answers the question the
    // address cannot.
    if (!owner.is_valid() || owner.is_global()) {
        LOG_WARN(
            "[GraphExecution] sub-task %d uses a tensor (addr=0x%llx) that did not come through the Graph "
            "boundary; pass it as a GraphTaskArgs parameter instead",
            task_index, static_cast<unsigned long long>(tensor_addr)
        );
        return false;
    }
    if (owner.space() == TaskId::Space::PARAM) {
        return graph_tensor_from_boundary(recording, tensor, owner.local_id());
    }

    // A task of this body, which may be the consuming task itself reading back an output it
    // is about to write -- its slot is already filled, so both resolve the same way. What
    // neither may do is name a task that is not recorded yet.
    const int32_t producer_index = owner.local_id();
    if (producer_index < 0 || producer_index > task_index) {
        LOG_WARN(
            "[GraphExecution] sub-task %d uses a tensor (addr=0x%llx) owned by task %d, which it cannot depend "
            "on: a Definition's edges run from tasks already recorded",
            task_index, static_cast<unsigned long long>(tensor_addr), producer_index
        );
        return false;
    }
    const RecordedSubTask &producer = recording.tasks[producer_index];
    // Compared as a subtraction on unsigned values so a block ending past the address space
    // cannot wrap the bound.
    if (producer.total_output_size == 0 || tensor_addr < producer.record_packed_base ||
        tensor_addr - producer.record_packed_base >= producer.total_output_size) {
        LOG_WARN(
            "[GraphExecution] sub-task %d uses a tensor (addr=0x%llx) addressed outside the output of task %d "
            "that owns it; a Graph body may only use a boundary parameter, a view of one, or a task's output",
            task_index, static_cast<unsigned long long>(tensor_addr), producer_index
        );
        return false;
    }
    return true;
}

// Counts, section offsets and total_bytes for the image this recording produces,
// settled without writing any of it so the destination can be claimed at the
// exact size. required_heap comes from the fill, which is the pass that walks the
// tasks in order.
std::optional<GraphDefinition> graph_layout_definition(const GraphRecording &recording) {
    if (recording.unsupported || recording.task_count == 0 || recording.task_count > SUB_TASK_MAX_NUM ||
        recording.bound_boundary().params.tensor_count() <= 0 ||
        recording.bound_boundary().params.tensor_count() > UINT16_MAX) {
        return std::nullopt;
    }

    int32_t total_tensors = 0;
    int32_t total_scalars = 0;
    int32_t total_fanins = 0;
    int32_t root_count = 0;
    int32_t predicate_count = 0;
    // task_count, not tasks.size(): the array keeps the slots a longer body left behind,
    // and those are not part of this recording.
    // The flat arrays a recorded task indexes into. Each is grown only by the recorder,
    // bounded by SUB_TASK_MAX_NUM times a per-task constant, so its length is an
    // int32 and every range test below stays in the offsets' own domain.
    const int32_t recorded_scalars = static_cast<int32_t>(recording.scalars.size());
    const int32_t recorded_scalar_inheritance = static_cast<int32_t>(recording.scalar_inheritance.size());
    const int32_t recorded_fanins = static_cast<int32_t>(recording.internal_fanins.size());
    for (int32_t i = 0; i < recording.task_count; ++i) {
        const RecordedSubTask &source = recording.tasks[i];
        // Negative first, so every count and offset below is a valid length by the time
        // it is compared and each remaining-span subtraction is non-negative. The
        // running-total tests bound each accumulator at the width its Definition field
        // carries rather than at the accumulator's own.
        if (source.tensor_count < 0 || source.scalar_count < 0 || source.fanin_count < 0 || source.scalar_offset < 0 ||
            source.fanin_offset < 0 || source.tensor_count > INT32_MAX - total_tensors ||
            source.scalar_count > INT32_MAX - total_scalars || source.fanin_count > INT32_MAX - total_fanins ||
            source.scalar_offset > recorded_scalars || source.scalar_count > recorded_scalars - source.scalar_offset ||
            source.scalar_offset > recorded_scalar_inheritance ||
            source.scalar_count > recorded_scalar_inheritance - source.scalar_offset ||
            source.fanin_offset > recorded_fanins || source.fanin_count > recorded_fanins - source.fanin_offset) {
            return std::nullopt;
        }
        total_tensors += source.tensor_count;
        total_scalars += source.scalar_count;
        total_fanins += source.fanin_count;
        root_count += source.fanin_count == 0 ? 1 : 0;
        predicate_count += source.predicate_index >= 0 ? 1 : 0;
    }
    if (predicate_count > UINT16_MAX) return std::nullopt;

    GraphDefinition definition{};
    definition.full_key = recording.full_key;
    definition.task_count = recording.task_count;
    definition.edge_count = total_fanins;
    definition.root_count = root_count;
    definition.boundary_tensor_count = recording.bound_boundary().params.tensor_count();
    definition.boundary_scalar_count = recording.bound_boundary().params.scalar_count();
    definition.tensor_arg_count = total_tensors;
    definition.scalar_arg_count = total_scalars;
    definition.predicate_count = predicate_count;
    size_t execution_storage_bytes = 0;
    // The execution keeps its own copy of the fanin CSR, so its rows and indices
    // are part of the storage the outer task's heap tail has to hold. Sized here,
    // where the counts are decided, and checked against this value on both ends
    // of the bind.
    if (!graph_execution_storage_bytes(
            definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
            &execution_storage_bytes
        ) ||
        execution_storage_bytes > UINT32_MAX) {
        return std::nullopt;
    }
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);

    size_t image_bytes = sizeof(GraphDefinition);
    if (!graph_layout_section<int32_t>(recording.task_count + 1, &image_bytes, &definition.off_fanout_offsets) ||
        !graph_layout_section<uint16_t>(total_fanins, &image_bytes, &definition.off_fanout_indices) ||
        !graph_layout_section<int32_t>(recording.task_count + 1, &image_bytes, &definition.off_fanin_offsets) ||
        !graph_layout_section<uint16_t>(total_fanins, &image_bytes, &definition.off_fanin_indices) ||
        !graph_layout_section<uint16_t>(root_count, &image_bytes, &definition.off_root_indices) ||
        !graph_layout_section<uint64_t>(recording.task_count, &image_bytes, &definition.off_sub_task_offsets) ||
        !graph_layout_section<SubTaskDefinition>(recording.task_count, &image_bytes, &definition.off_sub_tasks) ||
        !graph_layout_section<simpler::hbg::TensorData>(total_tensors, &image_bytes, &definition.off_tensors) ||
        !graph_layout_section<uint64_t>(total_scalars, &image_bytes, &definition.off_scalars) ||
        !graph_layout_section<GraphScalarInheritance>(
            total_scalars, &image_bytes, &definition.off_scalar_inheritance
        ) ||
        !graph_layout_section<GraphPredicate>(predicate_count, &image_bytes, &definition.off_predicates)) {
        return std::nullopt;
    }
    definition.total_bytes = static_cast<uint32_t>(image_bytes);
    return definition;
}

// Write the image of `recording` at `image`, which must be graph_layout_definition's
// total_bytes and aligned for every section type it laid out. `definition` is that
// layout; the fill settles required_heap and writes the header.
//
// Every section is written in full here, so the destination's prior content does not
// reach the device — with one exception, `fanout_offsets`, which is accumulated
// rather than assigned and is therefore zeroed below before its first increment.
// Two kinds of byte are written by nobody and read by nobody: the alignment slack
// between sections, and `SubTaskDefinition`'s interior padding, which the
// per-field assignment below cannot reach and a static_assert on that struct's
// size pins against further growth.
bool graph_fill_definition(const GraphRecording &recording, GraphDefinition definition, std::byte *image) {
    if (image == nullptr) return false;
    always_assert(
        reinterpret_cast<uintptr_t>(image) % GRAPH_DEFINITION_OBJECT_ALIGN == 0 &&
        "a Definition image base must carry the alignment its section offsets assume"
    );
    const size_t total_tensors = definition.tensor_arg_count;
    const size_t total_scalars = definition.scalar_arg_count;
    const size_t total_fanins = definition.edge_count;
    const size_t root_count = definition.root_count;
    const size_t predicate_count = definition.predicate_count;
    auto *fanout_offsets = graph_image_section<int32_t>(image, definition.off_fanout_offsets);
    auto *fanout_indices = graph_image_section<uint16_t>(image, definition.off_fanout_indices);
    auto *fanin_offsets = graph_image_section<int32_t>(image, definition.off_fanin_offsets);
    auto *fanin_indices = graph_image_section<uint16_t>(image, definition.off_fanin_indices);
    auto *roots = graph_image_section<uint16_t>(image, definition.off_root_indices);
    auto *sub_task_offsets = graph_image_section<uint64_t>(image, definition.off_sub_task_offsets);
    auto *tasks = graph_image_section<SubTaskDefinition>(image, definition.off_sub_tasks);
    auto *tensors = graph_image_section<simpler::hbg::TensorData>(image, definition.off_tensors);
    auto *scalars = graph_image_section<uint64_t>(image, definition.off_scalars);
    auto *scalar_inheritance = graph_image_section<GraphScalarInheritance>(image, definition.off_scalar_inheritance);
    auto *predicates = graph_image_section<GraphPredicate>(image, definition.off_predicates);
    uint64_t required_heap = 0;
    size_t tensor_cursor = 0;
    size_t scalar_cursor = 0;
    size_t fanin_cursor = 0;
    size_t root_cursor = 0;
    size_t predicate_cursor = 0;
    // A producer's fanout count is accumulated across the consumer walk below and then
    // prefix-summed in place, so every entry has to start at zero — including [0],
    // which nothing else writes and which the device checks is zero.
    std::fill_n(fanout_offsets, recording.task_count + 1, 0);
    fanin_offsets[0] = 0;
    for (int32_t i = 0; i < recording.task_count; ++i) {
        const RecordedSubTask &source = recording.tasks[i];
        if (source.total_output_size > static_cast<size_t>(INT32_MAX) || source.fanin_count > UINT16_MAX) {
            return false;
        }
        sub_task_offsets[i] = required_heap;
        const uint64_t output_bytes = CHIP_ALIGN_UP(source.total_output_size, CHIP_ALIGN_SIZE);
        if (required_heap > UINT64_MAX - output_bytes) return false;
        required_heap += output_bytes;

        if (source.fanin_count == 0) roots[root_cursor++] = static_cast<uint16_t>(i);
        // Early-dispatch qualification, carrying the top-level submit path's
        // conjunction over to the Definition's own data. An internal fanin of at
        // least one excludes a body root, whose real gate is the outer shell's
        // activation rather than this CSR. A body records once per shape and every
        // execution replays it, so the verdict costs nothing per invocation.
        //
        // The top-level conjunction's Graph-shell term has no counterpart here:
        // graph_begin refuses a nested recording, so no producer in a body is a
        // Graph shell.
        bool ed_candidate = source.fanin_count > 0 && !source.task_attrs.has_predicate() &&
                            source.active_mask.to_shape() != ResourceShape::DUMMY;
        const size_t row_begin = fanin_cursor;
        for (int32_t f = 0; f < source.fanin_count; ++f) {
            const int32_t producer = recording.internal_fanins[source.fanin_offset + f];
            if (producer >= i) return false;
            fanin_indices[fanin_cursor++] = static_cast<uint16_t>(producer);
            fanout_offsets[producer + 1]++;
            if (!recording.tasks[producer].task_attrs.allow_early_resolve()) ed_candidate = false;
        }
        fanin_offsets[i + 1] = static_cast<int32_t>(fanin_cursor);
        if (ed_candidate) {
            // A candidate's row is sorted ascending, and the builder emits a
            // producer before its consumers, so the row's tail names its deepest
            // producer — the entry the completion chain's tail-first scan bets on.
            // Every other row keeps its record order, which is the consumer's
            // deduplicated operand order and names no depth.
            std::sort(fanin_indices + row_begin, fanin_indices + fanin_cursor);
            // Every producer of a candidate must record its publication state.
            // Producers precede consumers, so each entry here was assigned its own
            // ed_flags at its own iteration and this only ever adds to it.
            for (size_t f = row_begin; f < fanin_cursor; ++f) {
                tasks[fanin_indices[f]].ed_flags |= ED_FLAG_TRACKED;
            }
        }

        SubTaskDefinition &task = tasks[i];
        std::copy(source.kernel_ids.begin(), source.kernel_ids.end(), std::begin(task.kernel_id));
        task.active_mask = source.active_mask.raw();
        // The recorded attrs carry the caller's early-resolve intent, which the
        // qualification above consumes; no sub-task reaches the device with
        // that bit set.
        TaskAttrs wire_attrs = source.task_attrs;
        wire_attrs.set_early_resolve(false);
        task.task_attrs = wire_attrs.raw();
        // Assignment, not accumulation: this entry may hold a previous body's
        // verdict, and only a consumer recorded later can add ED_FLAG_TRACKED to
        // it, which happens after this write.
        task.ed_flags = ed_candidate ? ED_FLAG_CANDIDATE : 0;
        task.reserved = 0;
        task.logical_block_num = source.logical_block_num;
        task.total_required_subtasks = source.total_required_subtasks;
        task.tensor_count = source.tensor_count;
        task.scalar_count = source.scalar_count;
        task.total_output_size = static_cast<int32_t>(source.total_output_size);
        task.tensor_offset = static_cast<int32_t>(tensor_cursor);
        task.scalar_offset = static_cast<int32_t>(scalar_cursor);
        task.dump_metadata = source.dump_metadata;
        task.predicate_slot = 0;
        if (source.predicate_index >= 0) {
            if (static_cast<size_t>(source.predicate_index) >= recording.predicates.size()) return false;
            const GraphRecordedPredicate &recorded = recording.predicates[source.predicate_index];
            if (recorded.operand.ndims > MAX_TENSOR_DIMS) return false;
            GraphPredicate packed{};
            graph_tensor_relocation_record(recording, recorded.operand, &packed.operand);
            packed.elem_offset = recorded.elem_offset;
            packed.target = recorded.target;
            packed.elem_size = recorded.elem_size;
            packed.op = static_cast<uint8_t>(recorded.op);
            predicates[predicate_cursor] = packed;
            task.predicate_slot = static_cast<uint16_t>(++predicate_cursor);
        }
        const simpler::hbg::Tensor *source_tensors = recording.task_tensors(source);
        for (int32_t t = 0; t < source.tensor_count; ++t) {
            if (source_tensors[t].ndims > MAX_TENSOR_DIMS) return false;
            graph_tensor_relocation_record(recording, source_tensors[t], &tensors[tensor_cursor]);
            tensor_cursor++;
        }
        for (int32_t scalar_index = 0; scalar_index < source.scalar_count; ++scalar_index) {
            const GraphScalarInheritance &inheritance =
                recording.scalar_inheritance[source.scalar_offset + scalar_index];
            // The last guard before the device indexes this: classification accounting
            // being right does not prove the image's own layout is.
            if (inheritance.inherited() &&
                inheritance.boundary_index() >= recording.bound_boundary().params.scalar_count()) {
                return false;
            }
            scalar_inheritance[scalar_cursor] = inheritance;
            // An inherited slot's Definition value is a placeholder: materialize overwrites
            // it from the invocation's own boundary.
            scalars[scalar_cursor++] =
                inheritance.inherited() ? 0 : recording.scalars[source.scalar_offset + scalar_index];
        }
    }
    if (tensor_cursor != total_tensors || scalar_cursor != total_scalars || fanin_cursor != total_fanins ||
        root_cursor != root_count || predicate_cursor != predicate_count) {
        return false;
    }
    definition.required_heap = required_heap;
    for (int32_t i = 0; i < recording.task_count; ++i)
        fanout_offsets[i + 1] += fanout_offsets[i];
    std::vector<int32_t> cursors(fanout_offsets, fanout_offsets + recording.task_count);
    for (int32_t consumer = 0; consumer < recording.task_count; ++consumer) {
        for (int32_t f = fanin_offsets[consumer]; f < fanin_offsets[consumer + 1]; ++f) {
            const int32_t producer = fanin_indices[f];
            fanout_indices[cursors[producer]++] = static_cast<uint16_t>(consumer);
        }
    }
    const GraphBoundary &boundary = recording.bound_boundary();
    const GraphTaskArgs &boundary_params = boundary.params;
    for (int32_t i = 0; i < boundary_params.tensor_count(); ++i) {
        if (boundary_params.tensor(i).ref().ndims > MAX_TENSOR_DIMS) return false;
    }
    std::memcpy(image, &definition, sizeof(definition));
    return true;
}

// The image at `data`, once it is a Definition of exactly `bytes`. Rejects a
// region whose own total_bytes disagrees, which is what makes a record's
// bookkeeping and the bytes it points at one fact rather than two.
const GraphDefinition *graph_definition(const std::byte *data, size_t bytes) {
    if (data == nullptr || bytes < sizeof(GraphDefinition)) return nullptr;
    const auto *definition = reinterpret_cast<const GraphDefinition *>(data);
    return definition->total_bytes == bytes ? definition : nullptr;
}

// Counted rather than returned across the .so boundary: the orch .so's prewarm entry has
// no return value, and giving it one would make an orch .so built before this change
// report whatever its x0 held.
std::atomic<size_t> g_recorder_storage_failures{0};

bool graph_recorder_stand_up_storage() {
    if (graph_recording_stand_up(recorder_recording())) return true;
    g_recorder_storage_failures.fetch_add(1, std::memory_order_relaxed);
    return false;
}

size_t graph_recorder_storage_failures() { return g_recorder_storage_failures.load(std::memory_order_relaxed); }
