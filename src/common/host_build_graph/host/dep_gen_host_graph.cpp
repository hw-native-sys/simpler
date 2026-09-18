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
 * @brief Host-side dep_gen graph capture + deps.json writer for host_build_graph.
 *
 * Edge model (unchanged from the schema in docs/dfx/dep-gen.md):
 *   explicit  — declared via Arg::set_dependencies (STEP 1); no tensor context.
 *   creator   — creator retention on an existing tensor (STEP 3 Step A).
 *   tensormap — a producer whose written slice overlaps what this task reads
 *               (STEP 3 Step B); carries both slices' geometry.
 *
 * Per-task producer dedup mirrors append_fanin_or_fail, which collapses all three
 * sources into one fanin list: the first edge to name a producer is kept. tensormap
 * edges are exempt — a second producer slice for the same task is a distinct fact
 * about the data flow, and viewers rely on seeing every overlap.
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

namespace {

// ---------------------------------------------------------------------------
// Graph tables (serialized as tasks[] / tensors[] / edges[])
// ---------------------------------------------------------------------------

// Edge categories — matches the three places a runtime fanin edge is born.
enum class EdgeSource { EXPLICIT, CREATOR, TENSORMAP };

const char *edge_source_str(EdgeSource s) {
    switch (s) {
    case EdgeSource::EXPLICIT:
        return "explicit";
    case EdgeSource::CREATOR:
        return "creator";
    case EdgeSource::TENSORMAP:
        return "tensormap";
    }
    return "unknown";
}

const char *overlap_status_str(OverlapStatus s) {
    switch (s) {
    case OverlapStatus::COVERED:
        return "covered";
    case OverlapStatus::OTHER:
        return "other";
    case OverlapStatus::NO_OVERLAP:
        return "no_overlap";
    }
    return "unknown";
}

const char *arg_type_str(TensorArgType t) {
    switch (t) {
    case TensorArgType::INPUT:
        return "INPUT";
    case TensorArgType::OUTPUT:
        return "OUTPUT";
    case TensorArgType::INOUT:
        return "INOUT";
    case TensorArgType::OUTPUT_EXISTING:
        return "OUTPUT_EXISTING";
    }
    return "UNKNOWN";
}

// One annotated edge. consumer_* always populated. producer_* populated for
// TENSORMAP source only — the explicit/creator paths have no matched tensormap
// entry to copy from.
//
// Slice description follows the strided simpler::hbg::Tensor model: (start_offset, strides[])
// in element units. Byte offset of element coords[] is
//   (start_offset + Σ coords[i] · strides[i]) · dtype_bytes
struct EdgeAnnot {
    uint64_t pred;
    uint64_t succ;
    int32_t consumer_arg_idx;  // -1 for EXPLICIT (not tied to a tensor arg)
    EdgeSource source;
    OverlapStatus overlap;  // only meaningful for TENSORMAP
    uint64_t tensor_id;     // 0 for EXPLICIT
    // Consumer side (the simpler::hbg::Tensor the submitting task is reading).
    uint8_t consumer_dtype;
    uint32_t consumer_ndims;
    uint32_t consumer_shape[MAX_TENSOR_DIMS];
    uint64_t consumer_start_offset;  // 1D element offset
    uint32_t consumer_strides[MAX_TENSOR_DIMS];
    // Producer side (the slice the producer wrote, from the tensormap entry).
    uint32_t producer_ndims;
    uint32_t producer_shape[MAX_TENSOR_DIMS];
    uint64_t producer_start_offset;
    uint32_t producer_strides[MAX_TENSOR_DIMS];
};

// One entry in the tensors[] table: the underlying storage, keyed by
// (buffer_addr, version). buffer_numel is the storage element count;
// per-edge fields describe the slice (start_offset + stride).
struct TensorTableEntry {
    uint64_t tensor_id;
    uint64_t buffer_addr;
    uint64_t buffer_numel;  // storage size in elements (= buffer.size / dtype_bytes)
    int32_t version;
    uint8_t dtype;
};

// One arg slot of a task, captured for the `tasks[].args[]` block so downstream
// viewers can render per-task input / output compartments without scanning every
// edge. `has_tensor_info` is false only for OUTPUT slots: the runtime has not
// materialized a simpler::hbg::Tensor for them at submit_task time.
struct TaskArgEntry {
    int32_t idx;
    TensorArgType arg_type;
    bool has_tensor_info;
    uint64_t tensor_id;
    uint8_t dtype;
    uint32_t ndims;
    uint32_t shape[MAX_TENSOR_DIMS];
    uint64_t start_offset;  // 1D element offset
    uint32_t strides[MAX_TENSOR_DIMS];
};

struct TaskTableEntry {
    uint64_t task_id;
    bool in_manual_scope;
    bool early_dispatch;
    int32_t kernel_id[3];  // per-subslot {AIC, AIV0, AIV1}, -1 = inactive
    uint32_t block_num;
    std::vector<TaskArgEntry> args;
};

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

// Copy a simpler::hbg::Tensor's slice description (shape + start_offset + stride) into an
// EdgeAnnot's consumer_* fields.
void fill_consumer(EdgeAnnot &e, const simpler::hbg::Tensor &t) {
    e.consumer_dtype = static_cast<uint8_t>(t.dtype);
    e.consumer_ndims = t.ndims;
    e.consumer_start_offset = t.start_offset;
    for (uint32_t i = 0; i < t.ndims && i < MAX_TENSOR_DIMS; i++) {
        e.consumer_shape[i] = t.shapes[i];
        e.consumer_strides[i] = t.strides[i];
    }
}

// Copy a ChipTensorMapEntry's slice description into an EdgeAnnot's producer_*
// fields. Only called from the TENSORMAP path.
void fill_producer(EdgeAnnot &e, const ChipTensorMapEntry &entry) {
    e.producer_ndims = entry.ndims;
    e.producer_start_offset = entry.start_offset;
    for (uint32_t i = 0; i < entry.ndims && i < MAX_TENSOR_DIMS; i++) {
        e.producer_shape[i] = entry.shapes[i];
        e.producer_strides[i] = entry.strides[i];
    }
}

// ---------------------------------------------------------------------------
// Capture state — thread-local, built and emitted on the same thread
// ---------------------------------------------------------------------------

struct HostGraphState {
    bool enabled = false;
    // A graph was captured since the last reset. Distinguishes "this
    // orchestration submitted nothing" from a missing capture/adoption handoff.
    bool captured = false;
    std::vector<TaskTableEntry> tasks;
    std::vector<TensorTableEntry> tensors;
    std::unordered_map<uint64_t, size_t> tensor_index;  // tensor_id → tensors[] idx
    std::vector<EdgeAnnot> edges;
    // Producers already named for the task currently being submitted.
    std::unordered_set<uint64_t> task_preds;
    uint64_t current_task_id = 0;
    bool in_task = false;

    // Releases the previous graph's memory rather than clear()ing it: a captured
    // graph is proportional to the task count (one heap-allocated arg vector per
    // task), and a process that ran capture once would otherwise hold that peak
    // for its lifetime.
    void reset() {
        std::vector<TaskTableEntry>{}.swap(tasks);
        std::vector<TensorTableEntry>{}.swap(tensors);
        std::unordered_map<uint64_t, size_t>{}.swap(tensor_index);
        std::vector<EdgeAnnot>{}.swap(edges);
        std::unordered_set<uint64_t>{}.swap(task_preds);
        captured = false;
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
    TensorTableEntry e;
    e.tensor_id = id;
    e.buffer_addr = t.buffer.addr;
    e.version = t.version;
    e.dtype = static_cast<uint8_t>(t.dtype);
    const uint64_t elem_size = get_element_size(t.dtype);
    e.buffer_numel = (elem_size == 0) ? 0 : (t.buffer.size / elem_size);
    s.tensor_index[id] = s.tensors.size();
    s.tensors.push_back(e);
    return id;
}

// ---------------------------------------------------------------------------
// JSON writer
// ---------------------------------------------------------------------------

void write_uint_array(std::ofstream &out, const uint32_t *data, uint32_t n) {
    out << '[';
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) out << ',';
        out << data[i];
    }
    out << ']';
}

bool write_deps_json(
    const char *path, const std::vector<TaskTableEntry> &tasks, const std::vector<TensorTableEntry> &tensors,
    const std::vector<EdgeAnnot> &edges
) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        LOG_ERROR("dep_gen host graph: failed to open '%s' for write", path);
        return false;
    }
    // Strided tensor representation. tensors[].buffer_numel is the underlying
    // storage element count; tasks[].args[] and edges[] carry per-slice
    // geometry as (start_offset uint64, strides[] uint32 — runtime invariant
    // forbids zero / negative strides, see runtime/tensor.h).
    // "runtime" names the TaskId layout every id below carries. A task id encodes
    // whichever layout its runtime uses and nothing in the value says which, so a
    // reader that decodes one has to be told. The literal rather than
    // SIMPLER_RUNTIME_NAME: this file only ever builds into host_build_graph, and the
    // unit tests that compile it define no such macro.
    out << "{\"runtime\":\"host_build_graph\",\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); i++) {
        if (i > 0) out << ',';
        const auto &t = tasks[i];
        // uint64 fields are quoted as strings — task_id/tensor_id/buffer_addr/
        // pred/succ can exceed Number.MAX_SAFE_INTEGER (2^53-1), silently
        // losing precision in JS-based JSON parsers. Python consumers already
        // pass these through int(...) and don't care which form they receive.
        out << "{\"task_id\":\"" << t.task_id << '"';
        out << ",\"scope\":\"" << (t.in_manual_scope ? "manual" : "auto") << '"';
        out << ",\"early_dispatch\":" << (t.early_dispatch ? "true" : "false");
        // Per-subslot kernel ids {AIC, AIV0, AIV1}; INVALID_KERNEL_ID = -1 for
        // inactive subslots. Emitted as a plain int triple — downstream viewers
        // (and the swimlane host post-processor) use it to resolve task_id →
        // kernel without the AICore record carrying the field itself.
        out << ",\"kernel_ids\":[" << t.kernel_id[0] << ',' << t.kernel_id[1] << ',' << t.kernel_id[2] << ']';
        out << ",\"block_num\":" << t.block_num;
        out << ",\"args\":[";
        for (size_t a = 0; a < t.args.size(); a++) {
            if (a > 0) out << ',';
            const auto &arg = t.args[a];
            out << "{\"idx\":" << arg.idx;
            out << ",\"type\":\"" << arg_type_str(arg.arg_type) << '"';
            if (arg.has_tensor_info) {
                out << ",\"tensor_id\":\"" << arg.tensor_id << '"';
                out << ",\"dtype\":\"" << get_dtype_name(static_cast<DataType>(arg.dtype)) << '"';
                out << ",\"shape\":";
                write_uint_array(out, arg.shape, arg.ndims);
                out << ",\"start_offset\":\"" << arg.start_offset << '"';
                out << ",\"strides\":";
                write_uint_array(out, arg.strides, arg.ndims);
            }
            out << '}';
        }
        out << "]}";
    }
    out << ']';

    out << ",\"tensors\":[";
    for (size_t i = 0; i < tensors.size(); i++) {
        if (i > 0) out << ',';
        const auto &t = tensors[i];
        out << "{\"tensor_id\":\"" << t.tensor_id << '"';
        out << ",\"buffer_addr\":\"" << t.buffer_addr << '"';
        out << ",\"version\":" << t.version;
        out << ",\"dtype\":\"" << get_dtype_name(static_cast<DataType>(t.dtype)) << '"';
        out << ",\"buffer_numel\":\"" << t.buffer_numel << '"';
        out << '}';
    }
    out << ']';

    out << ",\"edges\":[";
    for (size_t i = 0; i < edges.size(); i++) {
        if (i > 0) out << ',';
        const auto &e = edges[i];
        out << "{\"pred\":\"" << e.pred << "\",\"succ\":\"" << e.succ << '"';
        out << ",\"arg\":" << e.consumer_arg_idx;
        out << ",\"source\":\"" << edge_source_str(e.source) << '"';
        if (e.source == EdgeSource::TENSORMAP) {
            out << ",\"overlap\":\"" << overlap_status_str(e.overlap) << '"';
        }
        if (e.source != EdgeSource::EXPLICIT) {
            out << ",\"tensor_id\":\"" << e.tensor_id << '"';
            out << ",\"consumer_dtype\":\"" << get_dtype_name(static_cast<DataType>(e.consumer_dtype)) << '"';
            out << ",\"consumer_shape\":";
            write_uint_array(out, e.consumer_shape, e.consumer_ndims);
            out << ",\"consumer_start_offset\":\"" << e.consumer_start_offset << '"';
            out << ",\"consumer_strides\":";
            write_uint_array(out, e.consumer_strides, e.consumer_ndims);
        }
        if (e.source == EdgeSource::TENSORMAP) {
            out << ",\"producer_shape\":";
            write_uint_array(out, e.producer_shape, e.producer_ndims);
            out << ",\"producer_start_offset\":\"" << e.producer_start_offset << '"';
            out << ",\"producer_strides\":";
            write_uint_array(out, e.producer_strides, e.producer_ndims);
        }
        out << '}';
    }
    out << "]}\n";
    return static_cast<bool>(out);
}

}  // namespace

// ---------------------------------------------------------------------------
// Capture surface
// ---------------------------------------------------------------------------

bool dep_gen_host_graph_enabled() { return state().enabled; }

void dep_gen_host_graph_begin_capture() { state().reset(); }

void dep_gen_host_graph_begin_task(
    uint64_t task_id_raw, bool in_manual_scope, bool early_dispatch, const int32_t kernel_ids[3], int32_t block_num,
    int32_t tensor_count, const TensorRef *tensors, const TensorArgType *arg_types
) {
    HostGraphState &s = state();
    if (!s.enabled) {
        return;
    }
    s.task_preds.clear();
    s.current_task_id = task_id_raw;
    s.in_task = true;
    s.captured = true;

    TaskTableEntry entry;
    entry.task_id = task_id_raw;
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
    entry.args.reserve(static_cast<size_t>(tc));
    for (int32_t i = 0; i < tc; i++) {
        TaskArgEntry slot{};
        slot.idx = i;
        slot.arg_type = arg_types[i];
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
        entry.args.push_back(slot);
    }
    s.tasks.push_back(std::move(entry));
}

void dep_gen_host_graph_end_task() { state().in_task = false; }

void dep_gen_host_graph_add_explicit_edge(uint64_t producer_raw) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    if (!s.task_preds.insert(producer_raw).second) {
        return;
    }
    EdgeAnnot e{};
    e.pred = producer_raw;
    e.succ = s.current_task_id;
    e.consumer_arg_idx = -1;
    e.source = EdgeSource::EXPLICIT;
    s.edges.push_back(e);
}

void dep_gen_host_graph_add_creator_edge(uint64_t producer_raw, int32_t arg_idx, const simpler::hbg::Tensor &consumer) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    if (!s.task_preds.insert(producer_raw).second) {
        return;
    }
    EdgeAnnot e{};
    e.pred = producer_raw;
    e.succ = s.current_task_id;
    e.consumer_arg_idx = arg_idx;
    e.source = EdgeSource::CREATOR;
    e.tensor_id = register_tensor(s, consumer);
    fill_consumer(e, consumer);
    s.edges.push_back(e);
}

void dep_gen_host_graph_add_tensormap_edge(
    uint64_t producer_raw, int32_t arg_idx, const simpler::hbg::Tensor &consumer, const ChipTensorMapEntry &entry,
    OverlapStatus overlap
) {
    HostGraphState &s = state();
    if (!s.enabled || !s.in_task) {
        return;
    }
    // Every overlapping producer slice is its own edge; the pred set is still
    // updated so a later creator/explicit edge for the same producer collapses.
    s.task_preds.insert(producer_raw);
    EdgeAnnot e{};
    e.pred = producer_raw;
    e.succ = s.current_task_id;
    e.consumer_arg_idx = arg_idx;
    e.source = EdgeSource::TENSORMAP;
    e.overlap = overlap;
    e.tensor_id = make_tensor_id(entry.buffer_addr, entry.version);
    fill_consumer(e, consumer);
    fill_producer(e, entry);
    s.edges.push_back(e);
}

// ---------------------------------------------------------------------------
// Control surface
// ---------------------------------------------------------------------------

extern "C" void dep_gen_host_graph_set_enabled(bool enable) { state().enabled = enable; }

extern "C" bool dep_gen_host_graph_active() { return true; }

extern "C" int dep_gen_host_graph_emit(const char *deps_json_path) {
    if (deps_json_path == nullptr) {
        LOG_ERROR("dep_gen host graph: null deps_json_path");
        return -1;
    }
    HostGraphState &s = state();
    if (!s.captured) {
        // An empty graph here is not "the orchestration submitted nothing" —
        // begin_task() would have set captured even for a graph of one task.
        // The graph is thread-local, so it means either capture was never armed
        // or this run's orchestration ran on a different thread than this emit.
        LOG_ERROR("dep_gen host graph: no capture on this thread — deps.json not written to %s", deps_json_path);
        return -3;
    }
    if (!write_deps_json(deps_json_path, s.tasks, s.tensors, s.edges)) {
        return -2;
    }
    LOG_INFO(
        "dep_gen host graph: wrote deps.json to %s (tasks=%zu, tensors=%zu, edges=%zu)", deps_json_path, s.tasks.size(),
        s.tensors.size(), s.edges.size()
    );
    return 0;
}
