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
 * @file host_graph_runs.h
 * @brief One host-built dependency graph, owned independently of the thread that
 *        built it, plus the deps.json writer that consumes it.
 *
 * A host-orchestrating runtime builds its graph into thread-local capture state
 * and, today, serializes it on that same thread. A background writer cannot read
 * that state — it would see its own empty thread-local — so the graph is *moved*
 * into a `HostGraphExport` on the capturing thread and the writer owns that.
 *
 * Runtime-agnostic on purpose, and self-contained: task ids arrive already
 * encoded (`TaskId::to_uint64`) and every tag arrives as the small integer the
 * writer switches on, so this header names no runtime's `TaskId` layout, needs
 * no task-argument header, and compiles the same under either runtime. The
 * argument tag is the raw `TensorArgType` byte, which is how
 * `common/platform/include/common/dep_gen.h` already carries it one layer down.
 *
 * The JSON is the schema in docs/dfx/dep-gen.md — byte for byte what the
 * capturing thread wrote before. The writer lives here, inline, so the capture
 * translation unit and the platform's exporter share one implementation without
 * either gaining a link edge on the other.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

#include "data_type.h"  // DataType, MAX_TENSOR_DIMS, get_dtype_name

namespace simpler::dfx::host_graph {

/** Where a fanin edge was born. Mirrors the three runtime dependency steps. */
enum class EdgeKind : uint8_t { Explicit = 0, Creator = 1, Tensormap = 2 };

/**
 * The `TensorArgType` bytes this writer renders by name.
 *
 * The enum itself is a task-argument concept one layer up, and naming it here
 * would make a shared platform header depend on whichever `tensor.h` a bare
 * include resolves to. So the byte travels, as it does in `dep_gen.h`, and the
 * capture asserts each constant against its enumerator where that enum is in
 * scope. A byte outside this set renders `UNKNOWN`, which is what the
 * synchronous writer has always done — `NO_DEP` included.
 */
inline constexpr uint8_t kArgInput = 0;
inline constexpr uint8_t kArgOutput = 1;
inline constexpr uint8_t kArgInout = 2;
inline constexpr uint8_t kArgOutputExisting = 3;

/** How a tensormap producer's slice met the consumer's. Tensormap edges only. */
enum class Overlap : uint8_t { Covered = 0, Other = 1, NoOverlap = 2 };

/**
 * One arg slot of a task.
 *
 * `has_tensor_info` is false only for OUTPUT slots: the runtime has not
 * materialized a tensor for them at submit time, so they carry no geometry.
 */
struct TaskArgEntry {
    int32_t idx{0};
    uint8_t arg_type{kArgInput};  // a TensorArgType byte
    bool has_tensor_info{false};
    uint64_t tensor_id{0};
    uint8_t dtype{0};
    uint32_t ndims{0};
    uint32_t shape[MAX_TENSOR_DIMS]{};
    uint64_t start_offset{0};
    uint32_t strides[MAX_TENSOR_DIMS]{};
};

/**
 * One task.
 *
 * Its args are not held here but in the export's flat `task_args`, addressed by
 * `task_arg_offsets`: a per-task vector would make the export one allocation per
 * task and its charged size unknowable from a single `capacity()`.
 */
struct TaskEntry {
    uint64_t task_id{0};  // encoded, TaskId::to_uint64
    bool in_manual_scope{false};
    bool early_dispatch{false};
    int32_t kernel_id[3]{-1, -1, -1};  // {AIC, AIV0, AIV1}, -1 = inactive
    uint32_t block_num{1};
};

/** The underlying storage behind a tensor id, keyed by (addr, version). */
struct TensorEntry {
    uint64_t tensor_id{0};
    uint64_t buffer_addr{0};
    uint64_t buffer_numel{0};  // storage element count
    int32_t version{0};
    uint8_t dtype{0};
};

/**
 * One annotated edge. Consumer geometry is always present; producer geometry
 * only for a tensormap edge, which is the only kind with a matched entry.
 */
struct EdgeEntry {
    uint64_t pred{0};  // encoded
    uint64_t succ{0};  // encoded
    int32_t consumer_arg_idx{-1};
    EdgeKind kind{EdgeKind::Explicit};
    Overlap overlap{Overlap::NoOverlap};
    uint64_t tensor_id{0};
    uint8_t consumer_dtype{0};
    uint32_t consumer_ndims{0};
    uint32_t consumer_shape[MAX_TENSOR_DIMS]{};
    uint64_t consumer_start_offset{0};
    uint32_t consumer_strides[MAX_TENSOR_DIMS]{};
    uint32_t producer_ndims{0};
    uint32_t producer_shape[MAX_TENSOR_DIMS]{};
    uint64_t producer_start_offset{0};
    uint32_t producer_strides[MAX_TENSOR_DIMS]{};
};

/** The destination a graph is published under, including its file name. */
inline constexpr size_t kMaxOutputDirBytes = 4096;
/**
 * Headroom inside the allowance for the file name a publication appends.
 *
 * `/deps.json.tmp` is 14 bytes; the same 64 the device-side collector reserves,
 * so both refuse the same destinations.
 */
inline constexpr size_t kOutputDirNameHeadroom = 64;

/** True when a destination leaves room for the file names appended to it. */
inline bool output_dir_fits(size_t length) { return length + kOutputDirNameHeadroom <= kMaxOutputDirBytes; }

/**
 * One run's graph, owned independently of the capture that produced it.
 *
 * The writer reads nothing a later run touches: the five vectors were moved out
 * of the thread-local state, and the identity and destination are the ones the
 * run itself supplied rather than whatever the runner's config says by the time
 * the writer gets here. The destination is fixed-capacity so the bytes an
 * unpublished export holds are known before it is accepted.
 */
struct HostGraphExport {
    uint64_t run_epoch{0};
    char output_dir[kMaxOutputDirBytes]{};
    uint32_t output_dir_len{0};
    std::vector<TaskEntry> tasks;
    std::vector<uint32_t> task_arg_offsets;  // size() == tasks.size() + 1
    std::vector<TaskArgEntry> task_args;     // tasks[i]'s args are [off[i], off[i + 1])
    std::vector<TensorEntry> tensors;
    std::vector<EdgeEntry> edges;

    /** The bytes the five vectors hold, from their own capacity. */
    size_t payload_bytes() const {
        return tasks.capacity() * sizeof(TaskEntry) + task_arg_offsets.capacity() * sizeof(uint32_t) +
               task_args.capacity() * sizeof(TaskArgEntry) + tensors.capacity() * sizeof(TensorEntry) +
               edges.capacity() * sizeof(EdgeEntry);
    }

    /** Drop the payload and report nothing: the charge is the caller's record. */
    void release() noexcept {
        std::vector<TaskEntry>{}.swap(tasks);
        std::vector<uint32_t>{}.swap(task_arg_offsets);
        std::vector<TaskArgEntry>{}.swap(task_args);
        std::vector<TensorEntry>{}.swap(tensors);
        std::vector<EdgeEntry>{}.swap(edges);
    }
};

/** What a hand-off found in the capturing thread's state. */
enum class TakeOutcome : int {
    /** The orchestration ran on this thread and closed every task it opened. */
    Complete = 0,
    /** Capture was never armed, or the orchestration ran on another thread. */
    NotCaptured = 1,
    /** A task was opened and never closed, so the graph is missing edges. */
    Incomplete = 2,
};

// ---------------------------------------------------------------------------
// The deps.json writer
// ---------------------------------------------------------------------------

namespace detail {

inline const char *edge_kind_name(EdgeKind k) {
    switch (k) {
    case EdgeKind::Explicit:
        return "explicit";
    case EdgeKind::Creator:
        return "creator";
    case EdgeKind::Tensormap:
        return "tensormap";
    }
    return "unknown";
}

inline const char *overlap_name(Overlap o) {
    switch (o) {
    case Overlap::Covered:
        return "covered";
    case Overlap::Other:
        return "other";
    case Overlap::NoOverlap:
        return "no_overlap";
    }
    return "unknown";
}

inline const char *arg_type_name(uint8_t t) {
    switch (t) {
    case kArgInput:
        return "INPUT";
    case kArgOutput:
        return "OUTPUT";
    case kArgInout:
        return "INOUT";
    case kArgOutputExisting:
        return "OUTPUT_EXISTING";
    default:
        return "UNKNOWN";
    }
}

inline void write_uint_array(std::ostream &out, const uint32_t *data, uint32_t n) {
    out << '[';
    for (uint32_t i = 0; i < n; i++) {
        if (i > 0) out << ',';
        out << data[i];
    }
    out << ']';
}

}  // namespace detail

/**
 * Serialize one graph, in the schema docs/dfx/dep-gen.md documents.
 *
 * Strided tensor representation. `tensors[].buffer_numel` is the underlying
 * storage element count; `tasks[].args[]` and `edges[]` carry per-slice geometry
 * as (start_offset uint64, strides[] uint32 — the runtime invariant forbids zero
 * and negative strides).
 *
 * uint64 fields are quoted as strings: task_id / tensor_id / buffer_addr / pred
 * / succ can exceed Number.MAX_SAFE_INTEGER (2^53-1) and would silently lose
 * precision in a JS-based JSON parser. Python consumers pass them through
 * `int(...)` and do not care which form they receive.
 *
 * "runtime" names the TaskId layout every id below carries. A task id encodes
 * whichever layout its runtime uses and nothing in the value says which, so a
 * reader that decodes one has to be told.
 */
inline void write_host_graph_body(std::ostream &out, const HostGraphExport &g) {
    out << "{\"runtime\":\"host_build_graph\",\"tasks\":[";
    for (size_t i = 0; i < g.tasks.size(); i++) {
        if (i > 0) out << ',';
        const TaskEntry &t = g.tasks[i];
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
        const uint32_t first = g.task_arg_offsets[i];
        const uint32_t last = g.task_arg_offsets[i + 1];
        for (uint32_t a = first; a < last; a++) {
            if (a > first) out << ',';
            const TaskArgEntry &arg = g.task_args[a];
            out << "{\"idx\":" << arg.idx;
            out << ",\"type\":\"" << detail::arg_type_name(arg.arg_type) << '"';
            if (arg.has_tensor_info) {
                out << ",\"tensor_id\":\"" << arg.tensor_id << '"';
                out << ",\"dtype\":\"" << get_dtype_name(static_cast<DataType>(arg.dtype)) << '"';
                out << ",\"shape\":";
                detail::write_uint_array(out, arg.shape, arg.ndims);
                out << ",\"start_offset\":\"" << arg.start_offset << '"';
                out << ",\"strides\":";
                detail::write_uint_array(out, arg.strides, arg.ndims);
            }
            out << '}';
        }
        out << "]}";
    }
    out << ']';

    out << ",\"tensors\":[";
    for (size_t i = 0; i < g.tensors.size(); i++) {
        if (i > 0) out << ',';
        const TensorEntry &t = g.tensors[i];
        out << "{\"tensor_id\":\"" << t.tensor_id << '"';
        out << ",\"buffer_addr\":\"" << t.buffer_addr << '"';
        out << ",\"version\":" << t.version;
        out << ",\"dtype\":\"" << get_dtype_name(static_cast<DataType>(t.dtype)) << '"';
        out << ",\"buffer_numel\":\"" << t.buffer_numel << '"';
        out << '}';
    }
    out << ']';

    out << ",\"edges\":[";
    for (size_t i = 0; i < g.edges.size(); i++) {
        if (i > 0) out << ',';
        const EdgeEntry &e = g.edges[i];
        out << "{\"pred\":\"" << e.pred << "\",\"succ\":\"" << e.succ << '"';
        out << ",\"arg\":" << e.consumer_arg_idx;
        out << ",\"source\":\"" << detail::edge_kind_name(e.kind) << '"';
        if (e.kind == EdgeKind::Tensormap) {
            out << ",\"overlap\":\"" << detail::overlap_name(e.overlap) << '"';
        }
        if (e.kind != EdgeKind::Explicit) {
            out << ",\"tensor_id\":\"" << e.tensor_id << '"';
            out << ",\"consumer_dtype\":\"" << get_dtype_name(static_cast<DataType>(e.consumer_dtype)) << '"';
            out << ",\"consumer_shape\":";
            detail::write_uint_array(out, e.consumer_shape, e.consumer_ndims);
            out << ",\"consumer_start_offset\":\"" << e.consumer_start_offset << '"';
            out << ",\"consumer_strides\":";
            detail::write_uint_array(out, e.consumer_strides, e.consumer_ndims);
        }
        if (e.kind == EdgeKind::Tensormap) {
            out << ",\"producer_shape\":";
            detail::write_uint_array(out, e.producer_shape, e.producer_ndims);
            out << ",\"producer_start_offset\":\"" << e.producer_start_offset << '"';
            out << ",\"producer_strides\":";
            detail::write_uint_array(out, e.producer_strides, e.producer_ndims);
        }
        out << '}';
    }
    out << "]}\n";
}

}  // namespace simpler::dfx::host_graph
