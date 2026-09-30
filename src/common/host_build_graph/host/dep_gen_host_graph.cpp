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

/**
 * @file dep_gen_host_graph.cpp
 * @brief Host-side dep_gen graph capture for host_build_graph.
 *
 * Edge model (the schema in docs/dfx/dep-gen.md):
 *   explicit  — declared via Arg::set_dependencies (STEP 1); no tensor context.
 *   creator   — creator retention on an existing tensor (STEP 3 Step A).
 *   tensormap — a producer whose written slice overlaps what this task reads
 *               (STEP 3 Step B); carries both slices' geometry.
 *
 * Per-task producer dedup mirrors append_fanin_or_fail, which collapses all three
 * sources into one fanin list: the first edge to name a producer is kept. tensormap
 * edges are exempt — a second producer slice for the same task is a distinct fact
 * about the data flow, and viewers rely on seeing every overlap.
 *
 * The graph accumulates directly in the shape a background writer owns
 * (`simpler::dfx::host_graph::HostGraphExport`), so handing it over is a move of
 * five vectors rather than a copy. The serializer lives in that same header.
 */

#include "host_build_graph/dep_gen_host_graph.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arg_direction.h"  // CORE_MAX_TENSOR_ARGS
#include "common/unified_log.h"
#include "data_type.h"
#include "host/host_graph_runs.h"

namespace hg = simpler::dfx::host_graph;

// The export carries the argument tag as a raw byte, because a shared platform
// header must not depend on whichever `tensor.h` a bare include resolves to.
// This is the one translation unit that sees both, so this is where the two
// stay in step.
static_assert(hg::kArgInput == static_cast<uint8_t>(TensorArgType::INPUT));
static_assert(hg::kArgOutput == static_cast<uint8_t>(TensorArgType::OUTPUT));
static_assert(hg::kArgInout == static_cast<uint8_t>(TensorArgType::INOUT));
static_assert(hg::kArgOutputExisting == static_cast<uint8_t>(TensorArgType::OUTPUT_EXISTING));

namespace {

// FNV-1a 64-bit hash of (buffer_addr, version) — stable tensor identity across
// runs (no time-dependent inputs).
uint64_t make_tensor_id(uint64_t buffer_addr, int32_t version) {
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;
    uint64_t h = FNV_OFFSET;
    const uint8_t *p;
    p = reinterpret_cast<const uint8_t *>(&buffer_addr);
    for (size_t i = 0; i < sizeof(buffer_addr); i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    uint32_t v = static_cast<uint32_t>(version);
    p = reinterpret_cast<const uint8_t *>(&v);
    for (size_t i = 0; i < sizeof(v); i++) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

hg::Overlap to_export_overlap(OverlapStatus s) {
    switch (s) {
    case OverlapStatus::COVERED:
        return hg::Overlap::Covered;
    case OverlapStatus::OTHER:
        return hg::Overlap::Other;
    case OverlapStatus::NO_OVERLAP:
        return hg::Overlap::NoOverlap;
    }
    return hg::Overlap::NoOverlap;
}

// Copy a tensor's slice description (shape + start_offset + stride) into an
// edge's consumer_* fields.
void fill_consumer(hg::EdgeEntry &e, const simpler::hbg::Tensor &t) {
    e.consumer_dtype = static_cast<uint8_t>(t.dtype);
    e.consumer_ndims = t.ndims;
    e.consumer_start_offset = t.start_offset;
    for (uint32_t i = 0; i < t.ndims && i < MAX_TENSOR_DIMS; i++) {
        e.consumer_shape[i] = t.shapes[i];
        e.consumer_strides[i] = t.strides[i];
    }
}

// Copy a ChipTensorMapEntry's slice description into an edge's producer_*
// fields. Only called from the tensormap path.
void fill_producer(hg::EdgeEntry &e, const ChipTensorMapEntry &entry) {
    e.producer_ndims = entry.ndims;
    e.producer_start_offset = entry.start_offset;
    for (uint32_t i = 0; i < entry.ndims && i < MAX_TENSOR_DIMS; i++) {
        e.producer_shape[i] = entry.shapes[i];
        e.producer_strides[i] = entry.strides[i];
    }
}

// ---------------------------------------------------------------------------
// Capture state — thread-local, handed over on the same thread
// ---------------------------------------------------------------------------

struct HostGraphState {
    bool enabled = false;
    /**
     * An orchestration started on this thread since the last hand-off.
     *
     * This — not `captured` — is what says the graph belongs to this thread. An
     * orchestration that submits nothing never reaches `begin_task`, so
     * `captured` cannot tell that legitimate empty graph from a capture that
     * never ran here, and the two need different answers.
     */
    bool capture_open = false;
    /** At least one task was recorded. Says nothing about which thread. */
    bool captured = false;
    hg::HostGraphExport graph;
    std::unordered_map<uint64_t, size_t> tensor_index;  // tensor_id → tensors[] idx
    // Producers already named for the task currently being submitted.
    std::unordered_set<TaskId> task_preds;
    uint64_t current_task_id = 0;
    bool in_task = false;

    // Releases the previous graph's memory rather than clear()ing it: a captured
    // graph is proportional to the task count, and a process that ran capture
    // once would otherwise hold that peak for its lifetime.
    void reset() {
        graph.release();
        std::unordered_map<uint64_t, size_t>{}.swap(tensor_index);
        std::unordered_set<TaskId>{}.swap(task_preds);
        captured = false;
        capture_open = false;
        current_task_id = 0;
        in_task = false;
    }
};

// Thread-local, not process-global: see the isolation note in the header.
HostGraphState &state() {
    static thread_local HostGraphState s;
    return s;
}

// Register a tensor in the tensors[] table on first sight of (addr, version).
// buffer_numel describes the underlying storage size in elements; per-edge
// fields describe the slice via (start_offset, strides[]).
uint64_t register_tensor(HostGraphState &s, const simpler::hbg::Tensor &t) {
    uint64_t id = make_tensor_id(t.buffer.addr, t.version);
    if (s.tensor_index.find(id) != s.tensor_index.end()) {
        return id;
    }
    hg::TensorEntry e;
    e.tensor_id = id;
    e.buffer_addr = t.buffer.addr;
    e.version = t.version;
    e.dtype = static_cast<uint8_t>(t.dtype);
    const uint64_t elem_size = get_element_size(t.dtype);
    e.buffer_numel = (elem_size == 0) ? 0 : (t.buffer.size / elem_size);
    s.tensor_index[id] = s.graph.tensors.size();
    s.graph.tensors.push_back(e);
    return id;
}

/**
 * Write one graph to `path`, truncating whatever is there.
 *
 * The stream is flushed and closed before its state is read: a graph held
 * entirely in the userspace buffer would otherwise report success and lose its
 * bytes to a write error at close, which `ofstream`'s destructor performs after
 * the check.
 */
bool write_graph_truncating(const char *path, const hg::HostGraphExport &graph) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        LOG_ERROR("dep_gen host graph: failed to open '%s' for write", path);
        return false;
    }
    hg::write_host_graph_body(out, graph);
    out.flush();
    out.close();
    if (!out) {
        LOG_ERROR("dep_gen host graph: writing '%s' failed while flushing or closing it", path);
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Capture surface
// ---------------------------------------------------------------------------

bool dep_gen_host_graph_enabled() { return state().enabled; }

void dep_gen_host_graph_begin_capture() {
    HostGraphState &s = state();
    s.reset();
    // Only when armed: a thread that ran an orchestration with capture off has
    // no graph to claim, and must not be mistaken for one that does.
    s.capture_open = s.enabled;
}

void dep_gen_host_graph_begin_task(
    TaskId task_id, bool in_manual_scope, bool early_dispatch, const int32_t kernel_ids[3], int32_t block_num,
    int32_t tensor_count, const TensorRef *tensors, const TensorArgType *arg_types
) {
    HostGraphState &s = state();
    if (!s.enabled) {
        return;
    }
    s.task_preds.clear();
    s.current_task_id = TaskId::to_uint64(task_id);
    s.in_task = true;
    s.captured = true;

    // Args live in one flat block addressed by per-task offsets, so the whole
    // graph is five vectors a hand-off can move. The leading zero is the first
    // task's start.
    if (s.graph.task_arg_offsets.empty()) s.graph.task_arg_offsets.push_back(0);

    hg::TaskEntry entry;
    entry.task_id = s.current_task_id;
    entry.in_manual_scope = in_manual_scope;
    entry.early_dispatch = early_dispatch;
    entry.kernel_id[0] = kernel_ids != nullptr ? kernel_ids[0] : -1;
    entry.kernel_id[1] = kernel_ids != nullptr ? kernel_ids[1] : -1;
    entry.kernel_id[2] = kernel_ids != nullptr ? kernel_ids[2] : -1;
    entry.block_num = block_num > 0 ? static_cast<uint32_t>(block_num) : 1u;

    int32_t tc = tensor_count;
    if (tc < 0 || tensors == nullptr || arg_types == nullptr) {
        tc = 0;
    } else if (tc > CORE_MAX_TENSOR_ARGS) {
        tc = CORE_MAX_TENSOR_ARGS;
    }
    for (int32_t i = 0; i < tc; i++) {
        hg::TaskArgEntry slot{};
        slot.idx = i;
        slot.arg_type = static_cast<uint8_t>(arg_types[i]);
        if (arg_types[i] == TensorArgType::OUTPUT) {
            // OUTPUT slots carry create_info, not a simpler::hbg::Tensor, until the runtime
            // materializes the buffer. Viewers render this as a placeholder
            // "alloc" output slot.
            slot.has_tensor_info = false;
        } else {
            const simpler::hbg::Tensor &t = tensors[i].ref();
            slot.tensor_id = register_tensor(s, t);
            slot.has_tensor_info = true;
            slot.dtype = static_cast<uint8_t>(t.dtype);
            slot.ndims = t.ndims;
            slot.start_offset = t.start_offset;
            for (uint32_t d = 0; d < t.ndims && d < MAX_TENSOR_DIMS; d++) {
                slot.shape[d] = t.shapes[d];
                slot.strides[d] = t.strides[d];
            }
        }
        s.graph.task_args.push_back(slot);
    }
    s.graph.tasks.push_back(entry);
    s.graph.task_arg_offsets.push_back(static_cast<uint32_t>(s.graph.task_args.size()));
}

void dep_gen_host_graph_end_task() { state().in_task = false; }

void dep_gen_host_graph_add_explicit_edge(TaskId producer) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    if (!s.task_preds.insert(producer).second) {
        return;
    }
    hg::EdgeEntry e{};
    e.pred = TaskId::to_uint64(producer);
    e.succ = s.current_task_id;
    e.consumer_arg_idx = -1;
    e.kind = hg::EdgeKind::Explicit;
    s.graph.edges.push_back(e);
}

void dep_gen_host_graph_add_creator_edge(TaskId producer, int32_t arg_idx, const simpler::hbg::Tensor &consumer) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    if (!s.task_preds.insert(producer).second) {
        return;
    }
    hg::EdgeEntry e{};
    e.pred = TaskId::to_uint64(producer);
    e.succ = s.current_task_id;
    e.consumer_arg_idx = arg_idx;
    e.kind = hg::EdgeKind::Creator;
    e.tensor_id = register_tensor(s, consumer);
    fill_consumer(e, consumer);
    s.graph.edges.push_back(e);
}

void dep_gen_host_graph_add_tensormap_edge(
    TaskId producer, int32_t arg_idx, const simpler::hbg::Tensor &consumer, const ChipTensorMapEntry &entry,
    OverlapStatus overlap
) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    // Every overlapping producer slice is its own edge; the pred set is still
    // updated so a later creator/explicit edge for the same producer collapses.
    s.task_preds.insert(producer);
    hg::EdgeEntry e{};
    e.pred = TaskId::to_uint64(producer);
    e.succ = s.current_task_id;
    e.consumer_arg_idx = arg_idx;
    e.kind = hg::EdgeKind::Tensormap;
    e.overlap = to_export_overlap(overlap);
    e.tensor_id = make_tensor_id(entry.buffer_addr, entry.version);
    fill_consumer(e, consumer);
    fill_producer(e, entry);
    s.graph.edges.push_back(e);
}

// ---------------------------------------------------------------------------
// Control surface
// ---------------------------------------------------------------------------

extern "C" void dep_gen_host_graph_set_enabled(bool enable) { state().enabled = enable; }

extern "C" bool dep_gen_host_graph_active() { return true; }

extern "C" int dep_gen_host_graph_take(hg::HostGraphExport *out) {
    HostGraphState &s = state();
    if (out == nullptr) return static_cast<int>(hg::TakeOutcome::NotCaptured);
    if (!s.capture_open) {
        // Either capture was never armed for this run, or this run's
        // orchestration ran on a different thread than this call.
        return static_cast<int>(hg::TakeOutcome::NotCaptured);
    }
    const bool task_left_open = s.in_task;
    // Moved, not copied: after this the thread-local holds empty vectors and
    // the next `begin_capture` has nothing of this graph to overwrite. The
    // caller owns every byte the writer will read.
    out->tasks = std::move(s.graph.tasks);
    out->task_arg_offsets = std::move(s.graph.task_arg_offsets);
    out->task_args = std::move(s.graph.task_args);
    out->tensors = std::move(s.graph.tensors);
    out->edges = std::move(s.graph.edges);
    s.graph.release();
    std::unordered_map<uint64_t, size_t>{}.swap(s.tensor_index);
    std::unordered_set<TaskId>{}.swap(s.task_preds);
    s.capture_open = false;
    s.captured = false;
    s.in_task = false;
    s.current_task_id = 0;
    // A task opened and never closed means the graph is missing whatever edges
    // that task would have contributed, and the format has nowhere to say so.
    return static_cast<int>(task_left_open ? hg::TakeOutcome::Incomplete : hg::TakeOutcome::Complete);
}

extern "C" int dep_gen_host_graph_emit(const char *deps_json_path) {
    if (deps_json_path == nullptr) {
        LOG_ERROR("dep_gen host graph: null deps_json_path");
        return -1;
    }
    hg::HostGraphExport graph;
    const int outcome = dep_gen_host_graph_take(&graph);
    if (outcome != static_cast<int>(hg::TakeOutcome::Complete)) {
        LOG_ERROR(
            "dep_gen host graph: %s — deps.json not written to %s",
            outcome == static_cast<int>(hg::TakeOutcome::Incomplete) ? "a task was left open" :
                                                                       "no capture on this thread",
            deps_json_path
        );
        return -3;
    }
    if (graph.tasks.empty()) {
        // This path's existing answer for a graph of no tasks, kept so a clean
        // default run's set of produced files does not change. The retained path
        // publishes the empty graph instead, where it is reportable either way.
        LOG_ERROR("dep_gen host graph: no capture on this thread — deps.json not written to %s", deps_json_path);
        return -3;
    }
    if (!write_graph_truncating(deps_json_path, graph)) {
        return -2;
    }
    LOG_INFO(
        "dep_gen host graph: wrote deps.json to %s (tasks=%zu, tensors=%zu, edges=%zu)", deps_json_path,
        graph.tasks.size(), graph.tensors.size(), graph.edges.size()
    );
    return 0;
}
