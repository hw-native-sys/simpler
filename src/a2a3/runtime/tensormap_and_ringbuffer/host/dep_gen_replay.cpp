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
 * @file dep_gen_replay.cpp
 * @brief Replay in-memory DepGenRecord stream → deps.json (strided tensor
 *        representation, tensor-annotated) via a host-resident PTO2TensorMap,
 *        with a differential check against the runtime template `compute_task_fanin`.
 *
 * Two passes run per record against two parallel PTO2TensorMap instances that
 * evolve in lockstep:
 *
 *   ORACLE pass (read-only contract):
 *     Drives `compute_task_fanin` (the same template the device orchestrator
 *     uses in pto_orchestrator.cpp:submit_task) against `tm_oracle`. Emits
 *     only PTO2TaskId values — the canonical set of producer IDs the runtime
 *     would have wired. We never widen this template's emit signature: this
 *     pass IS the contract, and any future change to `compute_task_fanin`
 *     automatically refreshes the oracle.
 *
 *   ANNOT pass (this file's feature):
 *     Inlines the same STEP A (creator retention) + STEP B (tensormap lookup)
 *     against `tm_annot`, but the callback fires with the full
 *     `PTO2TensorMapEntry&` + the consumer Tensor* + the arg index, so the
 *     replay can record per-edge tensor metadata (producer/consumer
 *     shape/offset, dtype, version).
 *
 * After both passes finish per record, we compare the producer-ID set the
 * oracle emitted to the producer-ID set the annot pass emitted. They MUST
 * match. If they diverge, deps.json is not written and the function returns
 * non-zero — this is the "no shotgun modifications" guarantee: anyone who
 * changes `compute_task_fanin` will trip this gate immediately and know to
 * mirror the change in the annot pass.
 *
 * STEP 1 (explicit_deps) is emitted at the call site (per pto_dep_compute.h's
 * "kept at call site" note); both passes run the same explicit-deps loop, so
 * the comparison covers it too.
 *
 * STEP 4 (`register_task_outputs`) runs on BOTH tensor maps after both passes
 * complete, keeping `tm_oracle` and `tm_annot` bit-equivalent for the next
 * record's INOUT+COVERED `remove_entry` mutations.
 *
 * Pool sizing: replay never advances last_task_alive, so each tensor map's
 * entry pool must accommodate every output write across the whole trace. We
 * scan the record buffer once to count INOUT + OUTPUT_EXISTING slots and size
 * the pool accordingly. Both maps get the same size.
 */

#include "dep_gen_replay.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "data_type.h"
#include "pto_dep_compute.h"
#include "pto_task_id.h"
#include "pto_tensormap.h"
#include "tensor.h"
#include "tensor_arg.h"

namespace {

int32_t ceil_pow2(int32_t v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// Count INOUT + OUTPUT_EXISTING slots across the record buffer —
// register_task_outputs only inserts those, and skips entries with manual_dep
// set. Counting both without inspecting manual_dep is a conservative upper
// bound (manual_dep is rare; the small over-allocation pays for itself in
// avoided pool exhaustion).
int32_t count_outputs(const DepGenRecord *records, size_t n) {
    int32_t total = 0;
    for (size_t i = 0; i < n; i++) {
        const DepGenRecord &r = records[i];
        // Overflow chain slots are reinterpret_cast views with no tensor data;
        // their `tensor_count` bytes are actually the overflow `dep_count` field,
        // which would mislead the loop below if read as a tensor count.
        if (r.flags & DEP_GEN_FLAG_OVERFLOW) continue;
        for (uint16_t j = 0; j < r.tensor_count; j++) {
            auto t = static_cast<TensorArgType>(r.arg_types[j]);
            if (t == TensorArgType::INOUT || t == TensorArgType::OUTPUT_EXISTING) {
                total++;
            }
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// JSON output accumulators (in-memory tables that get serialized at the end)
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

// One annotated edge. consumer_* always populated. producer_* populated for
// TENSORMAP source only — the explicit/creator emit paths don't have a
// matched tensormap entry to copy from.
//
// Slice description follows the strided Tensor model: (start_offset, strides[])
// in element units. Byte offset of element coords[] is
//   (start_offset + Σ coords[i] · strides[i]) · dtype_bytes
struct EdgeAnnot {
    uint64_t pred;
    uint64_t succ;
    int32_t consumer_arg_idx;  // -1 for EXPLICIT (not tied to a tensor arg)
    EdgeSource source;
    OverlapStatus overlap;  // only meaningful for TENSORMAP
    uint64_t tensor_id;     // 0 for EXPLICIT
    // Consumer side (the Tensor the submitting task is reading).
    uint8_t consumer_dtype;
    uint32_t consumer_ndims;
    uint32_t consumer_shape[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t consumer_start_offset;  // 1D element offset
    uint32_t consumer_strides[RUNTIME_MAX_TENSOR_DIMS];
    // Producer side (the slice the producer wrote, from the tensormap entry).
    // Only populated when source == TENSORMAP.
    uint32_t producer_ndims;
    uint32_t producer_shape[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t producer_start_offset;
    uint32_t producer_strides[RUNTIME_MAX_TENSOR_DIMS];
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

// One arg slot of a task, captured for the `tasks[].args[]` block so
// downstream viewers can render per-task input / output compartments without
// having to scan every edge. `has_tensor_info` is false only for OUTPUT slots:
// the runtime hasn't materialized a Tensor yet at submit_task time, so the
// captured blob is zeroed.
struct TaskArgEntry {
    int32_t idx;
    TensorArgType arg_type;
    bool has_tensor_info;
    uint64_t tensor_id;
    uint8_t dtype;
    uint32_t ndims;
    uint32_t shape[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t start_offset;  // 1D element offset
    uint32_t strides[RUNTIME_MAX_TENSOR_DIMS];
};

struct TaskTableEntry {
    uint64_t task_id;
    bool in_manual_scope;
    std::vector<TaskArgEntry> args;
};

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

// FNV-1a 64-bit hash of (buffer_addr, version) — stable tensor identity
// across runs (no time-dependent inputs).
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

// Register a tensor in the tensors[] table on first sight of (addr,
// version). buffer_numel describes the underlying storage size in elements;
// per-edge fields describe the slice via (start_offset, strides[]). Subsequent
// sightings of the same (addr, version) are no-ops.
uint64_t register_tensor(
    std::unordered_map<uint64_t, size_t> &index_by_id, std::vector<TensorTableEntry> &table, const Tensor &t
) {
    uint64_t id = make_tensor_id(t.buffer.addr, t.version);
    auto it = index_by_id.find(id);
    if (it != index_by_id.end()) {
        return id;
    }
    TensorTableEntry e;
    e.tensor_id = id;
    e.buffer_addr = t.buffer.addr;
    e.version = t.version;
    e.dtype = static_cast<uint8_t>(t.dtype);
    const uint64_t elem_size = get_element_size(t.dtype);
    e.buffer_numel = (elem_size == 0) ? 0 : (t.buffer.size / elem_size);
    index_by_id[id] = table.size();
    table.push_back(e);
    return id;
}

// Copy a Tensor's slice description (shape + start_offset + stride) into an
// EdgeAnnot's consumer_* fields.
void fill_consumer(EdgeAnnot &e, const Tensor &t) {
    e.consumer_dtype = static_cast<uint8_t>(t.dtype);
    e.consumer_ndims = t.ndims;
    e.consumer_start_offset = t.start_offset;
    for (uint32_t i = 0; i < t.ndims && i < RUNTIME_MAX_TENSOR_DIMS; i++) {
        e.consumer_shape[i] = t.shapes[i];
        e.consumer_strides[i] = t.strides[i];
    }
}

// Copy a PTO2TensorMapEntry's slice description into an EdgeAnnot's producer_*
// fields. Only called from the TENSORMAP emit path.
void fill_producer(EdgeAnnot &e, const PTO2TensorMapEntry &entry) {
    e.producer_ndims = entry.ndims;
    e.producer_start_offset = entry.start_offset;
    for (uint32_t i = 0; i < entry.ndims && i < RUNTIME_MAX_TENSOR_DIMS; i++) {
        e.producer_shape[i] = entry.shapes[i];
        e.producer_strides[i] = entry.strides[i];
    }
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
        LOG_ERROR("dep_gen replay: failed to open '%s' for write", path);
        return false;
    }
    // Strided tensor representation. tensors[].buffer_numel is the underlying
    // storage element count; tasks[].args[] and edges[] carry per-slice
    // geometry as (start_offset uint64, strides[] uint32 — runtime invariant
    // forbids zero / negative strides, see runtime/tensor.h).
    out << "{\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); i++) {
        if (i > 0) out << ',';
        const auto &t = tasks[i];
        // uint64 fields are quoted as strings — task_id/tensor_id/buffer_addr/
        // pred/succ can exceed Number.MAX_SAFE_INTEGER (2^53-1), silently
        // losing precision in JS-based JSON parsers. Python consumers already
        // pass these through int(...) and don't care which form they receive.
        out << "{\"task_id\":\"" << t.task_id << '"';
        out << ",\"scope\":\"" << (t.in_manual_scope ? "manual" : "auto") << '"';
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

// ---------------------------------------------------------------------------
// Annot pass — mirrors compute_task_fanin step-by-step against tm_annot.
// Must stay bit-equivalent to pto_dep_compute.h::compute_task_fanin in terms
// of which producer IDs are emitted (the differential check enforces this).
// ---------------------------------------------------------------------------

template <typename EmitTM, typename EmitCreator>
void annot_pass(
    const DepInputs &inputs, PTO2TensorMap &tensor_map, bool in_manual_scope, EmitCreator emit_creator,
    EmitTM emit_tensormap
) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::OUTPUT) {
            continue;
        }
        const Tensor *tensor = inputs.tensors[i].ptr;

        // STEP A: creator retention.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            emit_creator(owner, i, *tensor);
        }

        // STEP B: tensormap lookup (only INPUT/INOUT, skip manual_dep).
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            emit_tensormap(entry.producer_task_id, i, *tensor, entry, overlap_status);
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                tensor_map.remove_entry(entry);
            }
            return true;
        });
    }
}

}  // namespace

extern "C" int
dep_gen_replay_emit_deps_json(const DepGenRecord *records, size_t num_records, const char *deps_json_path) {
    if (deps_json_path == nullptr) {
        LOG_ERROR("dep_gen replay: null deps_json_path");
        return -1;
    }
    if (num_records > 0 && records == nullptr) {
        LOG_ERROR("dep_gen replay: num_records=%zu but records pointer is null", num_records);
        return -1;
    }
    LOG_INFO_V0("dep_gen replay: processing %zu in-memory records (dual-pass)", num_records);

    // Per-ring task window sizes — tensormap masks slot indices and requires
    // each to be a power of two. Auto-size from the records themselves so each
    // ring's window comfortably covers its observed max local_id (no slot
    // aliasing during INOUT+COVERED remove_from_task). Same sizes feed both
    // maps so they stay in lockstep.
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint32_t max_local[PTO2_MAX_RING_DEPTH] = {0};
    for (size_t i = 0; i < num_records; i++) {
        PTO2TaskId tid{records[i].task_id};
        uint8_t ring = tid.ring();
        uint32_t local = tid.local();
        if (ring < PTO2_MAX_RING_DEPTH && local > max_local[ring]) {
            max_local[ring] = local;
        }
    }
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t need = static_cast<int32_t>(max_local[r] + 1);
        task_window_sizes[r] = ceil_pow2(need < 16 ? 16 : need);
    }

    int32_t output_count = count_outputs(records, num_records);
    int32_t pool_size = output_count + (output_count / 10) + 64;
    if (pool_size < PTO2_TENSORMAP_POOL_SIZE) {
        pool_size = PTO2_TENSORMAP_POOL_SIZE;
    }

    PTO2TensorMap tm_oracle;
    PTO2TensorMap tm_annot;
    std::memset(&tm_oracle, 0, sizeof(tm_oracle));
    std::memset(&tm_annot, 0, sizeof(tm_annot));

    // Libc-backed arena (default ctor) that owns both replay tensormaps'
    // storage. Released by the arena destructor when this function returns.
    DeviceArena replay_arena;

    auto oracle_layout =
        PTO2TensorMap::reserve_layout(replay_arena, PTO2_TENSORMAP_NUM_BUCKETS, pool_size, task_window_sizes);
    auto annot_layout =
        PTO2TensorMap::reserve_layout(replay_arena, PTO2_TENSORMAP_NUM_BUCKETS, pool_size, task_window_sizes);
    if (replay_arena.commit() == nullptr || !tm_oracle.init_data_from_layout(oracle_layout, replay_arena) ||
        !tm_annot.init_data_from_layout(annot_layout, replay_arena)) {
        LOG_ERROR("dep_gen replay: tensormap.init failed (buckets=%d, pool=%d)", PTO2_TENSORMAP_NUM_BUCKETS, pool_size);
        return -3;
    }
    // Replay tensormaps live entirely on host; only arena-internal pointer
    // fields need wiring (no parent-orch back-reference exists anymore).
    tm_oracle.wire_arena_pointers(oracle_layout, replay_arena);
    tm_annot.wire_arena_pointers(annot_layout, replay_arena);

    // JSON output accumulators.
    std::vector<TaskTableEntry> task_table;
    std::vector<TensorTableEntry> tensor_table;
    std::unordered_map<uint64_t, size_t> tensor_index;  // tensor_id → table idx
    std::vector<EdgeAnnot> annot_edges;
    annot_edges.reserve(num_records * 2);

    TensorRef tref_buf[CORE_MAX_TENSOR_ARGS];
    TensorArgType atype_buf[CORE_MAX_TENSOR_ARGS];

    // Per-record dedup of producer IDs — must match runtime's
    // PTO2FaninBuilder::append_fanin_or_fail semantics, which collapses STEP 1
    // (explicit_deps) + STEP A (creator retention) + STEP B (tensormap lookup)
    // into a single per-task fanin list. Both oracle and annot use this same
    // semantics so the divergence check is meaningful.
    std::unordered_set<uint64_t> oracle_preds;
    std::unordered_set<uint64_t> annot_preds;

    // Scratch buffer for assembling full dep lists across overflow chains.
    // Declared outside the loop so it can be reused (clear() keeps capacity).
    std::vector<uint64_t> full_deps_buf;

    for (size_t rec_i = 0; rec_i < num_records; rec_i++) {
        const DepGenRecord &rec = records[rec_i];

        // Overflow chain records are consumed by the preceding base; skip
        // them in the main scan so we don't double-process or read the
        // overflow's reinterpreted bytes as tensor/dep info.
        if (rec.flags & DEP_GEN_FLAG_OVERFLOW) continue;

        PTO2TaskId task_id{rec.task_id};
        bool in_manual_scope = (rec.flags & DEP_GEN_FLAG_IN_MANUAL_SCOPE) != 0;

        oracle_preds.clear();
        annot_preds.clear();

        int32_t tc = static_cast<int32_t>(rec.tensor_count);
        if (tc > CORE_MAX_TENSOR_ARGS) {
            tc = CORE_MAX_TENSOR_ARGS;
        }
        for (int32_t i = 0; i < tc; i++) {
            tref_buf[i].ptr = reinterpret_cast<const Tensor *>(&rec.tensors[i][0]);
            atype_buf[i] = static_cast<TensorArgType>(rec.arg_types[i]);
        }

        // Assemble the full dep list. Fast path: ≤ DEP_GEN_MAX_EXPLICIT_DEPS,
        // no chain, point straight at rec.explicit_deps. Slow path: gather
        // base + chain into full_deps_buf and point at the buffer.
        //
        // `explicit_dep_count` / `over->dep_count` originate from device
        // shared memory and are bounded by the writer to the array sizes, but
        // we clamp on read too so a corrupted record never drives an OOB read
        // off the end of rec.explicit_deps[64] / over->deps[326].
        const uint64_t *deps_data;
        int32_t dc;
        if (rec.flags & DEP_GEN_FLAG_HAS_OVERFLOW) {
            full_deps_buf.clear();
            uint16_t base_dc = rec.explicit_dep_count;
            if (base_dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
                LOG_ERROR(
                    "dep_gen replay: clamping base explicit_dep_count %u > %d at rec_idx=%zu (task_id=%" PRIu64 ")",
                    base_dc, DEP_GEN_MAX_EXPLICIT_DEPS, rec_i, rec.task_id
                );
                base_dc = DEP_GEN_MAX_EXPLICIT_DEPS;
            }
            full_deps_buf.reserve(static_cast<size_t>(base_dc) + DEP_GEN_OVERFLOW_DEPS_PER_RECORD);
            full_deps_buf.insert(full_deps_buf.end(), rec.explicit_deps, rec.explicit_deps + base_dc);
            bool chain_complete = false;
            for (size_t j = rec_i + 1; j < num_records; j++) {
                const DepGenRecord &maybe = records[j];
                if (!(maybe.flags & DEP_GEN_FLAG_OVERFLOW)) {
                    LOG_ERROR(
                        "dep_gen replay: unterminated overflow chain at rec_idx=%zu (task_id=%" PRIu64 ")", rec_i,
                        rec.task_id
                    );
                    break;
                }
                if (maybe.task_id != rec.task_id) {
                    LOG_ERROR(
                        "dep_gen replay: orphan overflow at rec_idx=%zu (expected task_id=%" PRIu64 ", found %" PRIu64
                        ")",
                        j, rec.task_id, maybe.task_id
                    );
                    break;
                }
                const auto *over = reinterpret_cast<const DepGenOverflowRecord *>(&maybe);
                uint16_t over_dc = over->dep_count;
                if (over_dc > DEP_GEN_OVERFLOW_DEPS_PER_RECORD) {
                    LOG_ERROR(
                        "dep_gen replay: clamping overflow dep_count %u > %d at rec_idx=%zu (task_id=%" PRIu64 ")",
                        over_dc, DEP_GEN_OVERFLOW_DEPS_PER_RECORD, j, rec.task_id
                    );
                    over_dc = DEP_GEN_OVERFLOW_DEPS_PER_RECORD;
                }
                full_deps_buf.insert(full_deps_buf.end(), over->deps, over->deps + over_dc);
                if (over->flags & DEP_GEN_FLAG_LAST_OVERFLOW) {
                    chain_complete = true;
                    break;
                }
            }
            if (!chain_complete) {
                LOG_ERROR(
                    "dep_gen replay: chain for task_id=%" PRIu64 " missing LAST_OVERFLOW marker — "
                    "using partial dep list (%zu deps)",
                    rec.task_id, full_deps_buf.size()
                );
            }
            deps_data = full_deps_buf.data();
            dc = static_cast<int32_t>(full_deps_buf.size());
        } else {
            deps_data = rec.explicit_deps;
            uint16_t base_dc = rec.explicit_dep_count;
            if (base_dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
                LOG_ERROR(
                    "dep_gen replay: clamping no-chain explicit_dep_count %u > %d at rec_idx=%zu (task_id=%" PRIu64 ")",
                    base_dc, DEP_GEN_MAX_EXPLICIT_DEPS, rec_i, rec.task_id
                );
                base_dc = DEP_GEN_MAX_EXPLICIT_DEPS;
            }
            dc = static_cast<int32_t>(base_dc);
        }

        DepInputs inputs;
        inputs.tensor_count = tc;
        inputs.tensors = tref_buf;
        inputs.arg_types = atype_buf;
        inputs.explicit_dep_count = dc;
        inputs.explicit_deps = reinterpret_cast<const PTO2TaskId *>(deps_data);

        // Register tasks[] entry (with per-arg slot info) and any unseen
        // tensors[] entries up-front. Tensors are registered from the
        // consumer-side blob so raw_shapes / dtype are populated (the
        // producer-side PTO2TensorMapEntry drops raw_shapes to fit in two
        // cache lines).
        TaskTableEntry task_entry;
        task_entry.task_id = rec.task_id;
        task_entry.in_manual_scope = in_manual_scope;
        task_entry.args.reserve(tc);
        for (int32_t i = 0; i < tc; i++) {
            TaskArgEntry slot{};
            slot.idx = i;
            slot.arg_type = atype_buf[i];
            if (atype_buf[i] == TensorArgType::OUTPUT) {
                // OUTPUT blob is zero at submit time (writer has no Tensor
                // yet); leave has_tensor_info=false. Viewers render this as
                // a placeholder "alloc" output slot.
                slot.has_tensor_info = false;
            } else {
                const Tensor &t = *tref_buf[i].ptr;
                register_tensor(tensor_index, tensor_table, t);
                slot.has_tensor_info = true;
                slot.tensor_id = make_tensor_id(t.buffer.addr, t.version);
                slot.dtype = static_cast<uint8_t>(t.dtype);
                slot.ndims = t.ndims;
                slot.start_offset = t.start_offset;
                for (uint32_t d = 0; d < t.ndims && d < RUNTIME_MAX_TENSOR_DIMS; d++) {
                    slot.shape[d] = t.shapes[d];
                    slot.strides[d] = t.strides[d];
                }
            }
            task_entry.args.push_back(slot);
        }
        task_table.push_back(std::move(task_entry));

        // ============ STEP 1 — explicit_deps (call-site emit) ============
        // Same loop on both passes; they MUST produce identical sets here
        // because they read the same record. Annot records explicit edges
        // with consumer_arg_idx = -1 (not tied to any tensor arg). Reads
        // from deps_data (base record's explicit_deps[] on fast path, the
        // gathered base+chain buffer on overflow path).
        for (int32_t i = 0; i < dc; i++) {
            uint64_t pred_raw = deps_data[i];
            if (oracle_preds.insert(pred_raw).second) {
                // First time this pred is seen at runtime call site.
            }
            if (annot_preds.insert(pred_raw).second) {
                EdgeAnnot e{};
                e.pred = pred_raw;
                e.succ = rec.task_id;
                e.consumer_arg_idx = -1;
                e.source = EdgeSource::EXPLICIT;
                annot_edges.push_back(e);
            }
        }

        // ============ ORACLE pass — drive compute_task_fanin ============
        bool ok = compute_task_fanin(inputs, tm_oracle, in_manual_scope, [&](PTO2TaskId producer) -> bool {
            oracle_preds.insert(producer.raw);
            return true;
        });
        if (!ok) {
            LOG_ERROR("dep_gen replay: compute_task_fanin returned fatal at task_id=%" PRIu64, rec.task_id);
            tm_oracle.destroy();
            tm_annot.destroy();
            return -4;
        }

        // ============ ANNOT pass — inline mirror, full entry capture ============
        annot_pass(
            inputs, tm_annot, in_manual_scope,
            // emit_creator(producer, arg_idx, consumer_tensor)
            [&](PTO2TaskId producer, int32_t arg_idx, const Tensor &consumer) {
                if (!annot_preds.insert(producer.raw).second) {
                    return;  // already covered by an earlier emit on this record
                }
                EdgeAnnot e{};
                e.pred = producer.raw;
                e.succ = rec.task_id;
                e.consumer_arg_idx = arg_idx;
                e.source = EdgeSource::CREATOR;
                e.tensor_id = make_tensor_id(consumer.buffer.addr, consumer.version);
                fill_consumer(e, consumer);
                annot_edges.push_back(e);
            },
            // emit_tensormap(producer, arg_idx, consumer_tensor, entry, status)
            [&](PTO2TaskId producer, int32_t arg_idx, const Tensor &consumer, const PTO2TensorMapEntry &entry,
                OverlapStatus status) {
                // Per-(succ, arg_idx, producer_buffer_addr, producer_version)
                // dedup gives us "the same producer slice fired twice for the
                // same consumer arg" collapse — but two distinct slices from
                // the same producer (different version), or two different
                // producers, both yield their own edges. The producer-id-set
                // comparison below uses annot_preds, which dedups by pred
                // only, matching runtime PTO2FaninBuilder semantics.
                annot_preds.insert(producer.raw);
                EdgeAnnot e{};
                e.pred = producer.raw;
                e.succ = rec.task_id;
                e.consumer_arg_idx = arg_idx;
                e.source = EdgeSource::TENSORMAP;
                e.overlap = status;
                e.tensor_id = make_tensor_id(entry.buffer_addr, entry.version);
                fill_consumer(e, consumer);
                fill_producer(e, entry);
                annot_edges.push_back(e);
            }
        );

        // ============ Differential check ============
        if (oracle_preds != annot_preds) {
            LOG_ERROR(
                "dep_gen replay: DIVERGENCE at task_id=%" PRIu64 " (rec_idx=%zu): oracle has %zu preds, annot has %zu",
                rec.task_id, rec_i, oracle_preds.size(), annot_preds.size()
            );
            // Log the symmetric difference for debugging.
            for (uint64_t p : oracle_preds) {
                if (annot_preds.find(p) == annot_preds.end()) {
                    LOG_ERROR("  only-in-oracle pred: %" PRIu64, p);
                }
            }
            for (uint64_t p : annot_preds) {
                if (oracle_preds.find(p) == oracle_preds.end()) {
                    LOG_ERROR("  only-in-annot  pred: %" PRIu64, p);
                }
            }
            tm_oracle.destroy();
            tm_annot.destroy();
            return -6;
        }

        // ============ STEP 4 — publish outputs on BOTH maps ============
        register_task_outputs(inputs, task_id, tm_oracle, in_manual_scope);
        register_task_outputs(inputs, task_id, tm_annot, in_manual_scope);
    }

    tm_oracle.destroy();
    tm_annot.destroy();

    if (!write_deps_json(deps_json_path, task_table, tensor_table, annot_edges)) {
        return -5;
    }
    LOG_INFO_V0(
        "dep_gen replay: wrote deps.json to %s (tasks=%zu, tensors=%zu, edges=%zu)", deps_json_path, task_table.size(),
        tensor_table.size(), annot_edges.size()
    );
    return 0;
}
