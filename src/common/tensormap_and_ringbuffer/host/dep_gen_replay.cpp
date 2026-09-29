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
 *        representation, tensor-annotated) via a host-resident ChipTensorMap,
 *        with a differential check against the runtime template `compute_task_fanin`.
 *
 * Two passes run per record against two parallel ChipTensorMap instances that
 * evolve in lockstep:
 *
 *   ORACLE pass (read-only contract):
 *     Drives `compute_task_fanin` (the same template the device orchestrator
 *     uses in orchestrator.cpp:submit_task) against `tm_oracle`. Its emit
 *     fires with (TaskId, DepFlags) — the canonical (producer, WAIT/RETAIN)
 *     mapping the runtime would have wired, OR-accumulated per producer. This
 *     pass IS the contract, and any future change to `compute_task_fanin`
 *     automatically refreshes the oracle.
 *
 *   ANNOT pass (this file's feature):
 *     Inlines the same STEP A (creator retention) + STEP B (tensormap lookup)
 *     against `tm_annot`, but the callback fires with the full
 *     `ChipTensorMapEntry&` + the consumer simpler::tmr::Tensor* + the arg index, so the
 *     replay can record per-edge tensor metadata (producer/consumer
 *     shape/offset, dtype, version).
 *
 * After both passes finish per record, we compare the (producer -> DepFlags)
 * mapping the oracle emitted to the one the annot pass emitted. They MUST
 * match on both the producer set and each producer's accumulated flags. If they
 * diverge, deps.json is not written and the function returns non-zero — this is
 * the "no shotgun modifications" guarantee: anyone who changes
 * `compute_task_fanin`'s producers or edge flags trips this gate immediately and
 * knows to mirror the change in the annot pass.
 *
 * STEP 1 (explicit_deps) is emitted at the call site (per dep_compute.h's
 * "kept at call site" note). Both passes seed explicit edges from the same
 * captured dep/kind arrays, so the differential check includes their effect
 * but cannot independently validate those bytes. Scene tests validate the
 * capture/replay round-trip for explicit kinds.
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
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "data_type.h"
#include "dep_compute.h"
#include "tensormap_and_ringbuffer/task_id.h"
#include "tensormap.h"
#include "tensor.h"

namespace {

// ---------------------------------------------------------------------------
// Charged allocation
// ---------------------------------------------------------------------------

/**
 * The budget in force for this replay. Null means unbounded.
 *
 * Passed by pointer through every container's allocator rather than read from a
 * global: the replay is re-entrant in principle and a global would tie two
 * concurrent publications together.
 */
struct Charger {
    const DepGenReplayBudget *budget{nullptr};

    bool charge(size_t bytes) const {
        if (budget == nullptr || budget->charge == nullptr) return true;
        return budget->charge(budget->ctx, bytes);
    }
    void credit(size_t bytes) const {
        if (budget == nullptr || budget->credit == nullptr) return;
        budget->credit(budget->ctx, bytes);
    }
};

/**
 * Allocator that charges before allocating and credits after freeing.
 *
 * Every container below is instantiated with this, so growth, rehash and the
 * bucket directory are all accounted without any hand-written carving: a
 * reallocation charges the new block while the old one is still charged, which
 * is the transient the caller's bound has to cover, and the standard library
 * is the one deciding sizes.
 *
 * A refused charge throws `std::bad_alloc`, which the entry point catches and
 * reports as a refusal — no container is left partially grown.
 */
template <typename T>
class ChargedAlloc {
public:
    using value_type = T;

    explicit ChargedAlloc(const Charger *charger) noexcept :
        charger_(charger) {}
    template <typename U>
    ChargedAlloc(const ChargedAlloc<U> &other) noexcept :
        charger_(other.charger()) {}

    const Charger *charger() const noexcept { return charger_; }

    // Bytes one element occupies. A rebound allocator's value type is often
    // itself a pointer — a node-based container's bucket array is an array of
    // pointers — so this is deliberately the size of `T` and not of whatever
    // `T` points at: N of them really do occupy N * sizeof(T) bytes, which is
    // the figure the budget has to hold. Named once so the justified
    // diagnostic is suppressed in one place rather than at every use.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    static constexpr size_t kElementBytes = sizeof(T);

    T *allocate(size_t n) {
        if (n != 0 && n > std::numeric_limits<size_t>::max() / kElementBytes) throw std::bad_alloc();
        const size_t bytes = n * kElementBytes;
        if (charger_ != nullptr && !charger_->charge(bytes)) throw std::bad_alloc();
        void *p = ::operator new(bytes, std::nothrow);
        if (p == nullptr) {
            if (charger_ != nullptr) charger_->credit(bytes);
            throw std::bad_alloc();
        }
        return static_cast<T *>(p);
    }

    void deallocate(T *p, size_t n) noexcept {
        ::operator delete(static_cast<void *>(p));
        if (charger_ != nullptr) charger_->credit(n * kElementBytes);
    }

    template <typename U>
    bool operator==(const ChargedAlloc<U> &other) const noexcept {
        return charger_ == other.charger();
    }
    template <typename U>
    bool operator!=(const ChargedAlloc<U> &other) const noexcept {
        return !(*this == other);
    }

private:
    const Charger *charger_{nullptr};
};

/** The arena backend the tensormaps allocate through, so they are charged too. */
void *charged_arena_alloc(void *ctx, size_t size) noexcept {
    auto *charger = static_cast<const Charger *>(ctx);
    if (charger != nullptr && !charger->charge(size)) return nullptr;
    void *p = std::malloc(size);
    if (p == nullptr && charger != nullptr) charger->credit(size);
    return p;
}

/**
 * Frees the arena's one allocation and credits it.
 *
 * `DeviceArena` does not hand the size back, so the charged figure is carried
 * beside it in the `Charger` the arena was constructed with.
 */
struct ArenaCharge {
    Charger charger;
    size_t bytes{0};
};

void *arena_alloc_recording(void *ctx, size_t size) noexcept {
    auto *rec = static_cast<ArenaCharge *>(ctx);
    void *p = charged_arena_alloc(&rec->charger, size);
    if (p != nullptr) rec->bytes = size;
    return p;
}

void arena_free_recording(void *ctx, void *ptr) noexcept {
    auto *rec = static_cast<ArenaCharge *>(ctx);
    std::free(ptr);
    if (rec->bytes != 0) {
        rec->charger.credit(rec->bytes);
        rec->bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Checked arithmetic and record-layout validation
// ---------------------------------------------------------------------------

bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    *out = a * b;
    return true;
}

/**
 * Largest `local_id` a task window may be sized from.
 *
 * `ceil_pow2` below is `int32_t`: above 2^30 the bit smear yields `0x80000000`,
 * negative as `int32_t` and ~1.8e19 once cast to the `size_t` an arena reserve
 * takes. The counts are device-written, so the domain is checked.
 */
constexpr int32_t kMaxTaskLocalId = (1 << 30) - 1;

/**
 * Validate the overflow-chain structure of the whole trace.
 *
 * `record_layout_valid` below checks one slot at a time, which cannot see that
 * a chain is broken: a base marked `HAS_OVERFLOW` whose continuation is
 * missing, mis-owned or unterminated leaves every individual record in range.
 * The replay's chain walk used to log that and carry on with whatever prefix it
 * had, and the dual-pass check cannot catch it — both passes are fed the same
 * partial dependency list, so they agree. A graph built from a truncated
 * dependency list is not this run's graph, and `deps.json` has nowhere to say
 * so, which is why this rejects instead.
 *
 * Walked once, consuming each chain, so any overflow record still reached at
 * the top level is an orphan nothing claims.
 */
bool chain_structure_valid(const DepGenRecord *records, size_t num_records) {
    size_t i = 0;
    while (i < num_records) {
        const DepGenRecord &base = records[i];
        if (base.flags & DEP_GEN_FLAG_OVERFLOW) {
            LOG_ERROR(
                "dep_gen replay: record %zu is an overflow slot no base record claims (task_id=0x%" PRIx64 ")", i,
                TaskId::to_uint64(base.task_id)
            );
            return false;
        }
        if (base.flags & DEP_GEN_FLAG_LAST_OVERFLOW) {
            LOG_ERROR(
                "dep_gen replay: record %zu marks LAST_OVERFLOW without being an overflow slot (task_id=0x%" PRIx64 ")",
                i, TaskId::to_uint64(base.task_id)
            );
            return false;
        }
        if (!(base.flags & DEP_GEN_FLAG_HAS_OVERFLOW)) {
            i++;
            continue;
        }
        // A chain is a contiguous run of overflow slots carrying the base's own
        // task id and ending in LAST_OVERFLOW. Several legitimate segments are
        // exactly that, so a multi-segment chain walks through here unchanged.
        size_t j = i + 1;
        bool terminated = false;
        while (j < num_records) {
            const DepGenRecord &link = records[j];
            if (!(link.flags & DEP_GEN_FLAG_OVERFLOW)) break;
            if (link.flags & DEP_GEN_FLAG_HAS_OVERFLOW) {
                LOG_ERROR("dep_gen replay: record %zu is both an overflow slot and a chain owner", j);
                return false;
            }
            if (link.task_id != base.task_id) break;
            j++;
            if (link.flags & DEP_GEN_FLAG_LAST_OVERFLOW) {
                terminated = true;
                break;
            }
        }
        if (!terminated) {
            LOG_ERROR(
                "dep_gen replay: the chain owned by record %zu (task_id=0x%" PRIx64
                ") is not terminated by a matching LAST_OVERFLOW slot — this run's dependency list is incomplete",
                i, TaskId::to_uint64(base.task_id)
            );
            return false;
        }
        i = j;
    }
    return true;
}

/**
 * Validate one slot against the layout its own flags select.
 *
 * An overflow slot is a reinterpret view whose `tensor_count` bytes are the
 * overflow `dep_count`, so reading it as a base record would validate the
 * wrong field. Each kind is parsed as itself.
 */
bool record_layout_valid(const DepGenRecord &r, size_t index) {
    const TaskId tid = r.task_id;
    if (tid.ring() >= CHIP_MAX_RING_DEPTH) {
        LOG_ERROR(
            "dep_gen replay: record %zu names ring %u, outside the %d ring domain", index,
            static_cast<unsigned>(tid.ring()), CHIP_MAX_RING_DEPTH
        );
        return false;
    }
    const int32_t local = tid.local_id();
    if (local < 0 || local > kMaxTaskLocalId) {
        LOG_ERROR("dep_gen replay: record %zu names local id %d, outside [0, %d]", index, local, kMaxTaskLocalId);
        return false;
    }
    if (r.flags & DEP_GEN_FLAG_OVERFLOW) {
        const auto *over = reinterpret_cast<const DepGenOverflowRecord *>(&r);
        if (over->dep_count > DEP_GEN_OVERFLOW_DEPS_PER_RECORD) {
            LOG_ERROR(
                "dep_gen replay: overflow slot %zu claims %u deps, above the %d it can hold", index,
                static_cast<unsigned>(over->dep_count), DEP_GEN_OVERFLOW_DEPS_PER_RECORD
            );
            return false;
        }
        return true;
    }
    if (r.tensor_count > CORE_MAX_TENSOR_ARGS) {
        LOG_ERROR(
            "dep_gen replay: record %zu claims %u tensor args, above the %d it can hold", index,
            static_cast<unsigned>(r.tensor_count), CORE_MAX_TENSOR_ARGS
        );
        return false;
    }
    if (r.explicit_dep_count > DEP_GEN_MAX_EXPLICIT_DEPS) {
        LOG_ERROR(
            "dep_gen replay: record %zu claims %u inline deps, above the %d it can hold", index,
            static_cast<unsigned>(r.explicit_dep_count), DEP_GEN_MAX_EXPLICIT_DEPS
        );
        return false;
    }
    return true;
}

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

// JSON array of the DepFlags bits set on an edge, e.g. ["wait","retain"].
void write_dep_flags(std::ostream &out, DepFlags flags) {
    out << '[';
    bool first = true;
    if (dep_has_wait(flags)) {
        out << "\"wait\"";
        first = false;
    }
    if (dep_has_retain(flags)) {
        if (!first) out << ',';
        out << "\"retain\"";
    }
    out << ']';
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
// Slice description follows the strided simpler::tmr::Tensor model: (start_offset, strides[])
// in element units. Byte offset of element coords[] is
//   (start_offset + Σ coords[i] · strides[i]) · dtype_bytes
struct EdgeAnnot {
    TaskId pred;
    TaskId succ;
    int32_t consumer_arg_idx;  // -1 for EXPLICIT (not tied to a tensor arg)
    EdgeSource source;
    DepFlags flags;         // per-edge WAIT/RETAIN semantics carried into deps.json
    OverlapStatus overlap;  // only meaningful for TENSORMAP
    uint64_t tensor_id;     // 0 for EXPLICIT
    // Consumer side (the simpler::tmr::Tensor the submitting task is reading).
    uint8_t consumer_dtype;
    uint32_t consumer_ndims;
    uint32_t consumer_shape[MAX_TENSOR_DIMS];
    uint64_t consumer_start_offset;  // 1D element offset
    uint32_t consumer_strides[MAX_TENSOR_DIMS];
    // Producer side (the slice the producer wrote, from the tensormap entry).
    // Only populated when source == TENSORMAP.
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

// One arg slot of a task, captured for the `tasks[].args[]` block so
// downstream viewers can render per-task input / output compartments without
// having to scan every edge. `has_tensor_info` is false only for OUTPUT slots:
// the runtime hasn't materialized a simpler::tmr::Tensor yet at submit_task time, so the
// captured blob is zeroed.
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

// Every table the replay grows is charged. The aliases exist so the allocator
// cannot be left off one of them by accident.
template <typename T>
using ChargedVec = std::vector<T, ChargedAlloc<T>>;
template <typename K, typename V>
using ChargedMap = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, ChargedAlloc<std::pair<const K, V>>>;

struct TaskTableEntry {
    TaskId task_id;
    bool in_manual_scope;
    bool early_dispatch;
    int32_t kernel_id[3];  // per-subslot {AIC, AIV0, AIV1}, -1 = inactive
    uint32_t block_num;
    ChargedVec<TaskArgEntry> args;
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
    ChargedMap<uint64_t, size_t> &index_by_id, ChargedVec<TensorTableEntry> &table, const simpler::tmr::Tensor &t
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

// Copy a simpler::tmr::Tensor's slice description (shape + start_offset + stride) into an
// EdgeAnnot's consumer_* fields.
void fill_consumer(EdgeAnnot &e, const simpler::tmr::Tensor &t) {
    e.consumer_dtype = static_cast<uint8_t>(t.dtype);
    e.consumer_ndims = t.ndims;
    e.consumer_start_offset = t.start_offset;
    for (uint32_t i = 0; i < t.ndims && i < MAX_TENSOR_DIMS; i++) {
        e.consumer_shape[i] = t.shapes[i];
        e.consumer_strides[i] = t.strides[i];
    }
}

// Copy a ChipTensorMapEntry's slice description into an EdgeAnnot's producer_*
// fields. Only called from the TENSORMAP emit path.
void fill_producer(EdgeAnnot &e, const ChipTensorMapEntry &entry) {
    e.producer_ndims = entry.ndims;
    e.producer_start_offset = entry.start_offset;
    for (uint32_t i = 0; i < entry.ndims && i < MAX_TENSOR_DIMS; i++) {
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
    const char *path, const ChargedVec<TaskTableEntry> &tasks, const ChargedVec<TensorTableEntry> &tensors,
    const ChargedVec<EdgeAnnot> &edges
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
    // "runtime" names the TaskId layout every id below carries. A task id encodes
    // whichever layout its runtime uses and nothing in the value says which, so a
    // reader that decodes one has to be told. The literal rather than
    // SIMPLER_RUNTIME_NAME: this file only ever builds into
    // tensormap_and_ringbuffer, and the unit tests that compile it define no such
    // macro.
    out << "{\"runtime\":\"tensormap_and_ringbuffer\",\"tasks\":[";
    for (size_t i = 0; i < tasks.size(); i++) {
        if (i > 0) out << ',';
        const auto &t = tasks[i];
        // uint64 fields are quoted as strings — task_id/tensor_id/buffer_addr/
        // pred/succ can exceed Number.MAX_SAFE_INTEGER (2^53-1), silently
        // losing precision in JS-based JSON parsers. Python consumers already
        // pass these through int(...) and don't care which form they receive.
        out << "{\"task_id\":\"" << TaskId::to_uint64(t.task_id) << '"';
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
        out << "{\"pred\":\"" << TaskId::to_uint64(e.pred) << "\",\"succ\":\"" << TaskId::to_uint64(e.succ) << '"';
        out << ",\"arg\":" << e.consumer_arg_idx;
        out << ",\"source\":\"" << edge_source_str(e.source) << '"';
        out << ",\"flags\":";
        write_dep_flags(out, e.flags);
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
    // The stream's own destructor would flush and close after this function had
    // already returned its verdict, so a small graph held entirely in the
    // userspace buffer could report success and then lose its bytes to a write
    // error at close. Flush and close here, and let the state afterwards be the
    // answer: link publication controls the name, not the content.
    out.flush();
    out.close();
    if (!out) {
        LOG_ERROR("dep_gen replay: writing '%s' failed while flushing or closing it", path);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Annot pass — mirrors compute_task_fanin step-by-step against tm_annot.
// Must stay bit-equivalent to dep_compute.h::compute_task_fanin in terms
// of which producer IDs are emitted (the differential check enforces this).
// ---------------------------------------------------------------------------

template <typename EmitTM, typename EmitCreator>
void annot_pass(
    const DepInputs &inputs, ChipTensorMap &tensor_map, bool in_manual_scope, EmitCreator emit_creator,
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
        const simpler::tmr::Tensor *tensor = &inputs.tensors[i].ref();

        // STEP A: creator retention.
        TaskId owner = tensor->owner_task_id;
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

        tensor_map.lookup(*tensor, [&](ChipTensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            emit_tensormap(entry.producer_task_id, i, *tensor, entry, overlap_status);
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                tensor_map.remove_entry(entry);
            }
            return true;
        });
    }
}

/**
 * The body, so the entry points can turn a refused charge into a return code.
 *
 * `bad_alloc` is the only way a charge refusal or an allocation failure leaves
 * here, and a caller must see a code rather than an exception: the budgeted
 * caller is a background writer whose thread has no boundary of its own, and
 * the unbudgeted one is reached from a C entry.
 */
int emit_deps_json_body(
    const DepGenRecord *records, size_t num_records, const char *deps_json_path, const Charger &charger
) {
    // Every count below is device-written, so each slot is validated against
    // the layout its own flags select before anything is sized or indexed from
    // it. This precedes count_outputs(), which indexes arg_types[] by
    // tensor_count with no bound of its own.
    for (size_t i = 0; i < num_records; i++) {
        if (!record_layout_valid(records[i], i)) return -5;
    }
    // Chain structure is a property of the sequence, not of any one slot, so
    // it is checked separately and before anything is sized from the trace.
    if (!chain_structure_valid(records, num_records)) return -5;

    // Per-ring task window sizes — tensormap masks slot indices and requires
    // each to be a power of two. Auto-size from the records themselves so each
    // ring's window comfortably covers its observed max local_id (no slot
    // aliasing during INOUT+COVERED remove_from_task). Same sizes feed both
    // maps so they stay in lockstep.
    // Every ring and local id is in domain by the validation above, so
    // ceil_pow2 cannot be handed a value it turns negative.
    int32_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    int32_t max_local[CHIP_MAX_RING_DEPTH] = {0};
    for (size_t i = 0; i < num_records; i++) {
        TaskId tid = records[i].task_id;
        uint8_t ring = tid.ring();
        int32_t local = tid.local_id();
        if (local > max_local[ring]) {
            max_local[ring] = local;
        }
    }
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        int32_t need = max_local[r] + 1;
        task_window_sizes[r] = ceil_pow2(need < 16 ? 16 : need);
    }

    // Widened deliberately: output_count is bounded by num_records x
    // CORE_MAX_TENSOR_ARGS, which overflows int32_t for a large enough trace,
    // and the sum would wrap before the comparison meant to raise it to the
    // floor.
    const int32_t output_count = count_outputs(records, num_records);
    uint64_t pool_wide = static_cast<uint64_t>(output_count) + static_cast<uint64_t>(output_count) / 10 + 64;
    if (pool_wide < static_cast<uint64_t>(CHIP_TENSORMAP_POOL_SIZE)) {
        pool_wide = static_cast<uint64_t>(CHIP_TENSORMAP_POOL_SIZE);
    }
    if (pool_wide > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        LOG_ERROR("dep_gen replay: tensormap pool size %" PRIu64 " exceeds the addressable pool", pool_wide);
        return -5;
    }
    const int32_t pool_size = static_cast<int32_t>(pool_wide);

    ChipTensorMap tm_oracle;
    ChipTensorMap tm_annot;
    std::memset(&tm_oracle, 0, sizeof(tm_oracle));
    std::memset(&tm_annot, 0, sizeof(tm_annot));

    // Arena owning both replay tensormaps' storage, allocating through the
    // caller's budget so the floor these two maps cost is charged like
    // everything else. Released by the arena destructor on return, which
    // credits the same figure.
    ArenaCharge arena_charge{charger, 0};
    DeviceArena replay_arena(&arena_alloc_recording, &arena_free_recording, &arena_charge);

    auto oracle_layout =
        ChipTensorMap::reserve_layout(replay_arena, CHIP_TENSORMAP_NUM_BUCKETS, pool_size, task_window_sizes);
    auto annot_layout =
        ChipTensorMap::reserve_layout(replay_arena, CHIP_TENSORMAP_NUM_BUCKETS, pool_size, task_window_sizes);
    // reserve() asserts rather than returns on exceeding kMaxRegions, and
    // asserts are compiled out of a release build, so the count is a
    // compile-time fact here: two maps take 2 x (4 + 2 x CHIP_MAX_RING_DEPTH).
    static_assert(
        2 * (4 + 2 * CHIP_MAX_RING_DEPTH) <= static_cast<int>(DeviceArena::kMaxRegions),
        "the two replay tensormaps must fit one arena's region table"
    );
    if (replay_arena.commit() == nullptr || !tm_oracle.init_data_from_layout(oracle_layout, replay_arena) ||
        !tm_annot.init_data_from_layout(annot_layout, replay_arena)) {
        LOG_ERROR(
            "dep_gen replay: tensormap init failed or was refused (buckets=%d, pool=%d, bytes=%zu)",
            CHIP_TENSORMAP_NUM_BUCKETS, pool_size, replay_arena.total_size()
        );
        return -3;
    }
    // Replay tensormaps live entirely on host; only arena-internal pointer
    // fields need wiring (no parent-orch back-reference exists anymore).
    tm_oracle.wire_arena_pointers(oracle_layout, replay_arena);
    tm_annot.wire_arena_pointers(annot_layout, replay_arena);

    // JSON output accumulators. Every one is charged, and the reservation
    // below is an opening size rather than a bound: one tensor argument can
    // name several producers, so the edge count follows the graph and each
    // growth charges as it happens.
    ChargedVec<TaskTableEntry> task_table{ChargedAlloc<TaskTableEntry>(&charger)};
    ChargedVec<TensorTableEntry> tensor_table{ChargedAlloc<TensorTableEntry>(&charger)};
    ChargedMap<uint64_t, size_t> tensor_index{ChargedAlloc<std::pair<const uint64_t, size_t>>(&charger)};
    ChargedVec<EdgeAnnot> annot_edges{ChargedAlloc<EdgeAnnot>(&charger)};
    size_t opening_edges = 0;
    if (checked_mul(num_records, 2, &opening_edges)) annot_edges.reserve(opening_edges);

    TensorRef tref_buf[CORE_MAX_TENSOR_ARGS];
    TensorArgType atype_buf[CORE_MAX_TENSOR_ARGS];

    // Per-record producer ID -> accumulated DepFlags — must match runtime's
    // FaninBuilder::append_fanin_or_fail semantics, which collapses STEP 1
    // (explicit_deps) + STEP A (creator retention) + STEP B (tensormap lookup)
    // into a single per-task fanin edge and OR-accumulates its flags. Both oracle
    // and annot use this same semantics so the divergence check compares the
    // (producer, flags) mapping rather than the producer-ID set alone.
    ChargedMap<TaskId, DepFlags> oracle_preds{ChargedAlloc<std::pair<const TaskId, DepFlags>>(&charger)};
    ChargedMap<TaskId, DepFlags> annot_preds{ChargedAlloc<std::pair<const TaskId, DepFlags>>(&charger)};
    ChargedMap<TaskId, size_t> explicit_edge_index{ChargedAlloc<std::pair<const TaskId, size_t>>(&charger)};

    // Scratch buffer for assembling full dep lists across overflow chains.
    // Declared outside the loop so it can be reused (clear() keeps capacity).
    ChargedVec<TaskId> full_deps_buf{ChargedAlloc<TaskId>(&charger)};
    ChargedVec<uint8_t> full_kinds_buf{ChargedAlloc<uint8_t>(&charger)};

    for (size_t rec_i = 0; rec_i < num_records; rec_i++) {
        const DepGenRecord &rec = records[rec_i];

        // Overflow chain records are consumed by the preceding base; skip
        // them in the main scan so we don't double-process or read the
        // overflow's reinterpreted bytes as tensor/dep info.
        if (rec.flags & DEP_GEN_FLAG_OVERFLOW) continue;

        TaskId task_id = rec.task_id;
        bool in_manual_scope = (rec.flags & DEP_GEN_FLAG_IN_MANUAL_SCOPE) != 0;

        oracle_preds.clear();
        annot_preds.clear();
        explicit_edge_index.clear();

        int32_t tc = static_cast<int32_t>(rec.tensor_count);
        if (tc > CORE_MAX_TENSOR_ARGS) {
            tc = CORE_MAX_TENSOR_ARGS;
        }
        for (int32_t i = 0; i < tc; i++) {
            tref_buf[i] = reinterpret_cast<const simpler::tmr::Tensor *>(&rec.tensors[i][0]);
            atype_buf[i] = static_cast<TensorArgType>(rec.arg_types[i]);
        }

        // Assemble the full dep list. Fast path: ≤ DEP_GEN_MAX_EXPLICIT_DEPS,
        // no chain, point straight at rec.explicit_deps. Slow path: gather
        // base + chain into full_deps_buf/full_kinds_buf and point at the buffers.
        //
        // `explicit_dep_count` / `over->dep_count` originate from device
        // shared memory and are bounded by the writer to the array sizes, but
        // we clamp on read too so a corrupted record never drives an OOB read
        // off the end of rec.explicit_deps[64] / over->deps[524].
        const TaskId *deps_data;
        const uint8_t *kinds_data;
        int32_t dc;
        if (rec.flags & DEP_GEN_FLAG_HAS_OVERFLOW) {
            full_deps_buf.clear();
            full_kinds_buf.clear();
            uint16_t base_dc = rec.explicit_dep_count;
            if (base_dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
                LOG_ERROR(
                    "dep_gen replay: clamping base explicit_dep_count %u > %d at rec_idx=%zu (task_id=0x%" PRIx64 ")",
                    base_dc, DEP_GEN_MAX_EXPLICIT_DEPS, rec_i, TaskId::to_uint64(rec.task_id)
                );
                base_dc = DEP_GEN_MAX_EXPLICIT_DEPS;
            }
            full_deps_buf.reserve(static_cast<size_t>(base_dc) + DEP_GEN_OVERFLOW_DEPS_PER_RECORD);
            full_kinds_buf.reserve(static_cast<size_t>(base_dc) + DEP_GEN_OVERFLOW_DEPS_PER_RECORD);
            full_deps_buf.insert(full_deps_buf.end(), rec.explicit_deps, rec.explicit_deps + base_dc);
            full_kinds_buf.insert(full_kinds_buf.end(), rec.explicit_dep_kinds, rec.explicit_dep_kinds + base_dc);
            bool chain_complete = false;
            for (size_t j = rec_i + 1; j < num_records; j++) {
                const DepGenRecord &maybe = records[j];
                if (!(maybe.flags & DEP_GEN_FLAG_OVERFLOW)) {
                    LOG_ERROR(
                        "dep_gen replay: unterminated overflow chain at rec_idx=%zu (task_id=0x%" PRIx64 ")", rec_i,
                        TaskId::to_uint64(rec.task_id)
                    );
                    break;
                }
                if (maybe.task_id != rec.task_id) {
                    LOG_ERROR(
                        "dep_gen replay: orphan overflow at rec_idx=%zu (expected task_id=0x%" PRIx64
                        ", found 0x%" PRIx64 ")",
                        j, TaskId::to_uint64(rec.task_id), TaskId::to_uint64(maybe.task_id)
                    );
                    break;
                }
                const auto *over = reinterpret_cast<const DepGenOverflowRecord *>(&maybe);
                uint16_t over_dc = over->dep_count;
                if (over_dc > DEP_GEN_OVERFLOW_DEPS_PER_RECORD) {
                    LOG_ERROR(
                        "dep_gen replay: clamping overflow dep_count %u > %d at rec_idx=%zu (task_id=0x%" PRIx64 ")",
                        over_dc, DEP_GEN_OVERFLOW_DEPS_PER_RECORD, j, TaskId::to_uint64(rec.task_id)
                    );
                    over_dc = DEP_GEN_OVERFLOW_DEPS_PER_RECORD;
                }
                full_deps_buf.insert(full_deps_buf.end(), over->deps, over->deps + over_dc);
                full_kinds_buf.insert(full_kinds_buf.end(), over->kinds, over->kinds + over_dc);
                if (over->flags & DEP_GEN_FLAG_LAST_OVERFLOW) {
                    chain_complete = true;
                    break;
                }
            }
            if (!chain_complete) {
                // Unreachable: `chain_structure_valid` rejected the trace
                // before this loop could see a broken chain. Kept as a refusal
                // rather than a log so the two can never disagree about what a
                // partial dependency list means.
                LOG_ERROR(
                    "dep_gen replay: chain for task_id=0x%" PRIx64 " lost its LAST_OVERFLOW marker after validation",
                    TaskId::to_uint64(rec.task_id)
                );
                return -5;
            }
            deps_data = full_deps_buf.data();
            kinds_data = full_kinds_buf.data();
            dc = static_cast<int32_t>(full_deps_buf.size());
        } else {
            deps_data = rec.explicit_deps;
            kinds_data = rec.explicit_dep_kinds;
            uint16_t base_dc = rec.explicit_dep_count;
            if (base_dc > DEP_GEN_MAX_EXPLICIT_DEPS) {
                LOG_ERROR(
                    "dep_gen replay: clamping no-chain explicit_dep_count %u > %d at rec_idx=%zu (task_id=0x%" PRIx64
                    ")",
                    base_dc, DEP_GEN_MAX_EXPLICIT_DEPS, rec_i, TaskId::to_uint64(rec.task_id)
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
        inputs.explicit_deps = deps_data;

        // Register tasks[] entry (with per-arg slot info) and any unseen
        // tensors[] entries up-front. ChipTensors are registered from the
        // consumer-side blob so raw_shapes / dtype are populated (the
        // producer-side ChipTensorMapEntry drops raw_shapes to fit in two
        // cache lines).
        TaskTableEntry task_entry{{},        false, false,
                                  {0, 0, 0}, 0u,    ChargedVec<TaskArgEntry>(ChargedAlloc<TaskArgEntry>(&charger))};
        task_entry.task_id = rec.task_id;
        task_entry.in_manual_scope = in_manual_scope;
        task_entry.early_dispatch = (rec.flags & DEP_GEN_FLAG_EARLY_DISPATCH) != 0;
        task_entry.kernel_id[0] = rec.kernel_id[0];
        task_entry.kernel_id[1] = rec.kernel_id[1];
        task_entry.kernel_id[2] = rec.kernel_id[2];
        task_entry.block_num = rec.block_num > 0 ? rec.block_num : 1u;
        task_entry.args.reserve(tc);
        for (int32_t i = 0; i < tc; i++) {
            TaskArgEntry slot{};
            slot.idx = i;
            slot.arg_type = atype_buf[i];
            if (atype_buf[i] == TensorArgType::OUTPUT) {
                // OUTPUT blob is zero at submit time (writer has no simpler::tmr::Tensor
                // yet); leave has_tensor_info=false. Viewers render this as
                // a placeholder "alloc" output slot.
                slot.has_tensor_info = false;
            } else {
                const simpler::tmr::Tensor &t = tref_buf[i].ref();
                register_tensor(tensor_index, tensor_table, t);
                slot.has_tensor_info = true;
                slot.tensor_id = make_tensor_id(t.buffer.addr, t.version);
                slot.dtype = static_cast<uint8_t>(t.dtype);
                slot.ndims = t.ndims;
                slot.start_offset = t.start_offset;
                for (uint32_t d = 0; d < t.ndims && d < MAX_TENSOR_DIMS; d++) {
                    slot.shape[d] = t.shapes[d];
                    slot.strides[d] = t.strides[d];
                }
            }
            task_entry.args.push_back(slot);
        }
        task_table.push_back(std::move(task_entry));

        // ============ STEP 1 — explicit_deps (call-site emit) ============
        // Both passes seed explicit edges from the same captured record. The
        // differential check therefore covers their interaction with creator
        // and tensormap edges, while scene tests independently validate the
        // captured kind bytes. Annot records explicit edges with
        // consumer_arg_idx = -1 (not tied to any tensor arg). deps_data comes
        // from the base record on the fast path or the gathered base+chain
        // buffer on overflow; kinds_data is its parallel semantics array.
        for (int32_t i = 0; i < dc; i++) {
            const TaskId pred = deps_data[i];
            const uint8_t raw_kind = kinds_data[i];
            constexpr uint8_t kKnownDepFlags = static_cast<uint8_t>(DEP_WAIT | DEP_RETAIN);
            if ((raw_kind & static_cast<uint8_t>(~kKnownDepFlags)) != 0) {
                // Unknown semantics make the artifact untrustworthy. Reject
                // the trace instead of emitting a plausible but altered graph.
                LOG_ERROR(
                    "dep_gen replay: invalid explicit dep flags 0x%02x at task_id=0x%" PRIx64 " dep_idx=%d", raw_kind,
                    TaskId::to_uint64(rec.task_id), i
                );
                tm_oracle.destroy();
                tm_annot.destroy();
                return -7;
            }
            const DepFlags kind = static_cast<DepFlags>(raw_kind);
            oracle_preds[pred] |= kind;
            bool first = annot_preds.find(pred) == annot_preds.end();
            annot_preds[pred] |= kind;
            if (first) {
                EdgeAnnot e{};
                e.pred = pred;
                e.succ = rec.task_id;
                e.consumer_arg_idx = -1;
                e.source = EdgeSource::EXPLICIT;
                e.flags = kind;
                explicit_edge_index.emplace(pred, annot_edges.size());
                annot_edges.push_back(e);
            } else {
                annot_edges[explicit_edge_index.at(pred)].flags |= kind;
            }
        }

        // ============ ORACLE pass — drive compute_task_fanin ============
        bool ok = compute_task_fanin(inputs, tm_oracle, in_manual_scope, [&](TaskId producer, DepFlags kind) -> bool {
            oracle_preds[producer] |= kind;
            return true;
        });
        if (!ok) {
            LOG_ERROR(
                "dep_gen replay: compute_task_fanin returned fatal at task_id=0x%" PRIx64,
                TaskId::to_uint64(rec.task_id)
            );
            tm_oracle.destroy();
            tm_annot.destroy();
            return -4;
        }

        // ============ ANNOT pass — inline mirror, full entry capture ============
        annot_pass(
            inputs, tm_annot, in_manual_scope,
            // emit_creator(producer, arg_idx, consumer_tensor)
            [&](TaskId producer, int32_t arg_idx, const simpler::tmr::Tensor &consumer) {
                bool first = annot_preds.find(producer) == annot_preds.end();
                annot_preds[producer] |= (DEP_WAIT | DEP_RETAIN);
                if (!first) {
                    auto explicit_it = explicit_edge_index.find(producer);
                    if (explicit_it != explicit_edge_index.end()) {
                        annot_edges[explicit_it->second].flags |= (DEP_WAIT | DEP_RETAIN);
                    }
                    return;  // already covered by an earlier emit on this record
                }
                EdgeAnnot e{};
                e.pred = producer;
                e.succ = rec.task_id;
                e.consumer_arg_idx = arg_idx;
                e.source = EdgeSource::CREATOR;
                e.flags = DEP_WAIT | DEP_RETAIN;
                e.tensor_id = make_tensor_id(consumer.buffer.addr, consumer.version);
                fill_consumer(e, consumer);
                annot_edges.push_back(e);
            },
            // emit_tensormap(producer, arg_idx, consumer_tensor, entry, status)
            [&](TaskId producer, int32_t arg_idx, const simpler::tmr::Tensor &consumer, const ChipTensorMapEntry &entry,
                OverlapStatus status) {
                // Per-(succ, arg_idx, producer_buffer_addr, producer_version)
                // dedup gives us "the same producer slice fired twice for the
                // same consumer arg" collapse — but two distinct slices from
                // the same producer (different version), or two different
                // producers, both yield their own edges. The producer-id-set
                // comparison below uses annot_preds, which dedups by pred
                // only, matching runtime FaninBuilder semantics.
                annot_preds[producer] |= DEP_WAIT;
                EdgeAnnot e{};
                e.pred = producer;
                e.succ = rec.task_id;
                e.consumer_arg_idx = arg_idx;
                e.source = EdgeSource::TENSORMAP;
                e.flags = DEP_WAIT;
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
                "dep_gen replay: DIVERGENCE at task_id=0x%" PRIx64
                " (rec_idx=%zu): oracle has %zu preds, annot has %zu",
                TaskId::to_uint64(rec.task_id), rec_i, oracle_preds.size(), annot_preds.size()
            );
            // Log the symmetric difference (missing preds and flag mismatches).
            for (const auto &[p, f] : oracle_preds) {
                auto it = annot_preds.find(p);
                if (it == annot_preds.end()) {
                    LOG_ERROR(
                        "  only-in-oracle pred: 0x%" PRIx64 " flags=%u", TaskId::to_uint64(p), static_cast<unsigned>(f)
                    );
                } else if (it->second != f) {
                    LOG_ERROR(
                        "  flags mismatch pred: 0x%" PRIx64 " oracle=%u annot=%u", TaskId::to_uint64(p),
                        static_cast<unsigned>(f), static_cast<unsigned>(it->second)
                    );
                }
            }
            for (const auto &[p, f] : annot_preds) {
                if (oracle_preds.find(p) == oracle_preds.end()) {
                    LOG_ERROR(
                        "  only-in-annot  pred: 0x%" PRIx64 " flags=%u", TaskId::to_uint64(p), static_cast<unsigned>(f)
                    );
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

    // Its own code: -7 already means an invalid explicit dep-flag byte, and a
    // caller that cannot tell "the graph was rejected" from "the file could
    // not be written" has nothing to act on.
    if (!write_deps_json(deps_json_path, task_table, tensor_table, annot_edges)) {
        return -9;
    }
    LOG_INFO(
        "dep_gen replay: wrote deps.json to %s (tasks=%zu, tensors=%zu, edges=%zu)", deps_json_path, task_table.size(),
        tensor_table.size(), annot_edges.size()
    );
    return 0;
}

}  // namespace

extern "C" int dep_gen_replay_emit_deps_json_budgeted(
    const DepGenRecord *records, size_t num_records, const char *deps_json_path, const DepGenReplayBudget *budget
) {
    if (deps_json_path == nullptr) {
        LOG_ERROR("dep_gen replay: null deps_json_path");
        return -1;
    }
    if (num_records > 0 && records == nullptr) {
        LOG_ERROR("dep_gen replay: num_records=%zu but records pointer is null", num_records);
        return -1;
    }
    LOG_INFO(
        "dep_gen replay: processing %zu in-memory records (dual-pass, %s)", num_records,
        budget == nullptr ? "unbudgeted" : "budgeted"
    );
    const Charger charger{budget};
    try {
        return emit_deps_json_body(records, num_records, deps_json_path, charger);
    } catch (const std::bad_alloc &) {
        // Either a charge was refused or the allocation behind it failed. Both
        // leave no file: a graph missing edges is not this run's graph, and the
        // format has nowhere to say so.
        LOG_ERROR("dep_gen replay: storage for this graph was refused — deps.json not produced");
        return -8;
    } catch (...) {
        LOG_ERROR("dep_gen replay: an unexpected host failure stopped this graph");
        return -8;
    }
}

extern "C" int
dep_gen_replay_emit_deps_json(const DepGenRecord *records, size_t num_records, const char *deps_json_path) {
    return dep_gen_replay_emit_deps_json_budgeted(records, num_records, deps_json_path, nullptr);
}
