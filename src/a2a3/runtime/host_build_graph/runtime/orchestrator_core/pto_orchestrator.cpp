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
 * host_build_graph orchestrator implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <limits>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "dep_gen_host_graph.h"
#include "pto_dep_compute.h"
#include "graph_execution.h"
#include "graph_host_state.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Weak fallbacks: host/dep_gen_host_graph.cpp provides the strong symbols in the
// HOST build, where the orchestrator runs and the graph is captured. The AICPU
// build has no host graph and links these no-op stubs so the runtime translation
// unit is self-contained. Visibility is hidden so the HOST .so doesn't export
// them into the global dynamic symbol table where they'd shadow the strong
// symbols (same pattern as get_sys_cnt_aicpu / chip_swimlane_aicpu_record_orch_phase
// below).
__attribute__((weak, visibility("hidden"))) bool dep_gen_host_graph_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_begin_task(
    uint64_t, bool, bool, const int32_t[3], int32_t, int32_t, const TensorRef *, const TensorArgType *
) {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_end_task() {}

// Raises the two edge kinds compute_task_fanin can discover, for the capture
// instantiation. Shared by the ordinary submit path and the outer GRAPH task so
// both describe an edge the same way.
struct DepGraphAnnotate {
    void creator(int32_t arg_idx, const ChipTensor &consumer, PTO2TaskId producer) const {
        dep_gen_host_graph_add_creator_edge(producer.raw, arg_idx, consumer);
    }
    void tensormap(
        int32_t arg_idx, const ChipTensor &consumer, const PTO2TensorMapEntry &entry, OverlapStatus overlap
    ) const {
        dep_gen_host_graph_add_tensormap_edge(entry.producer_task_id.raw, arg_idx, consumer, entry, overlap);
    }
};
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_explicit_edge(uint64_t) {}
__attribute__((weak, visibility("hidden"))) void
dep_gen_host_graph_add_creator_edge(uint64_t, int32_t, const ChipTensor &) {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_tensormap_edge(
    uint64_t, int32_t, const ChipTensor &, const PTO2TensorMapEntry &, OverlapStatus
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// dep_gen_host_graph_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// AICore register accessor (aicpu/platform_regs.h). The host orchestrator's
// route_ready_once path transitively ODR-uses the early-dispatch doorbell inline
// (pto_scheduler.h ring_one_doorbell), but no core is gated during host
// graph-build, so the doorbell never fires and this weak host fallback only
// satisfies the linker. The AICPU build links the strong definition from
// platform/.../platform_regs.cpp; hidden so the HOST .so does not shadow it.
__attribute__((weak, visibility("hidden"))) volatile uint32_t *get_reg_ptr(uint64_t, RegId) {
    static volatile uint32_t sink = 0;
    return &sink;
}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
// Weak fallback for builds that don't link chip_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log. Per-event records are a separate channel on a
// separate clock — see ORCH_PHASE_END below.
#define CYCLE_COUNT_START()                  \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1; \
    (void)_t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() {
    // Host fallback: monotonic wall-clock in AICPU cycle units so the host-orch
    // deadlock/timeout backstops fire at their intended wall-clock (see the
    // detailed rationale on the same fallback in pto_runtime2.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
// submit_idx tags a record with its position in the pass's submit order.
static uint32_t g_orch_submit_idx = 0;
// The per-sub-step accumulators exist only in an ORCH_PROFILING build, so at this
// level there is nothing to time.
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#endif

// Host phase record sink. The host build links a strong definition that folds the
// record into its kind's counters and appends it to the platform's record pool;
// every other build keeps this no-op. Kind values are HostPhaseKind, passed as a
// plain integer because this file is also compiled for the AICPU, where the
// platform's host headers are absent.
__attribute__((weak, visibility("hidden"))) void host_phase_record(uint64_t, uint64_t, uint32_t, uint64_t, uint32_t) {}

// Host monotonic clock shared with the `[STRACE]` span tree, so a record nests
// under chip.run.bind.host_orch without any clock conversion. The host build
// links the strong definition in host_phase_trace.cpp; this fallback keeps
// non-host builds linking, where the recorder above is a no-op anyway.
__attribute__((weak, visibility("hidden"))) uint64_t host_phase_now_ns() { return 0; }

#if SIMPLER_DFX
// Kinds this file records, spelled as the HostPhaseKind values the host side
// reads. Only the host orchestrator reaches these sites, so the bind-stage kinds
// that precede them in the enum are not named here.
enum class HostOrchPhase : uint32_t {
    Submit = 12,           // submit_task_common: one ordinary ring task
    Prepare = 13,          // prepare_task: one alloc_tensors slot
    RecordNode = 14,       // graph_record_submit_node: one recorded Graph node
    GraphSubmit = 15,      // graph_submit_definition: one outer GRAPH task
    BuildDefinition = 16,  // graph_build_definition: nodes compacted into the image
};
#define ORCH_PHASE_START() const uint64_t _orch_phase_t0 = host_phase_now_ns()
#define ORCH_PHASE_END(phase, detail)                                                                         \
    do {                                                                                                      \
        host_phase_record(                                                                                    \
            _orch_phase_t0, host_phase_now_ns(), static_cast<uint32_t>(phase), static_cast<uint64_t>(detail), \
            g_orch_submit_idx                                                                                 \
        );                                                                                                    \
    } while (0)
#else
#define ORCH_PHASE_START()
#define ORCH_PHASE_END(phase, detail) \
    do {                              \
    } while (0)
#endif

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if SIMPLER_DFX
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    std::array<char, 1024> message{};
    vsnprintf(message.data(), message.size(), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message.data());
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message.data());
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

enum class GraphRecordedTensorSource : uint8_t {
    BOUNDARY_EXACT,
    BOUNDARY_VIEW,
    INTERNAL,
    OWN_OUTPUT,
};

struct GraphRecordedTensorSourceRef {
    GraphRecordedTensorSource source{GraphRecordedTensorSource::BOUNDARY_EXACT};
    size_t source_index{0};
    uint64_t packed_offset{0};
};

enum class GraphRecordedScalarSource : uint8_t {
    STATIC_VALUE,
    BOUNDARY,
    INVALIDATED_BOUNDARY,
};

struct GraphRecordedScalarSourceRef {
    GraphRecordedScalarSource source{GraphRecordedScalarSource::STATIC_VALUE};
    size_t source_index{0};
};

// A node's dispatch predicate, held as the operand tensor plus the element index
// within it rather than the absolute address submit would resolve. The tensor is
// copied because the caller only lends it for the duration of the submit call.
struct GraphRecordedPredicate {
    ChipTensor operand;
    GraphRecordedTensorSourceRef source;
    uint64_t elem_offset{0};
    int64_t target{0};
    uint8_t elem_size{0};
    PredicateOp op{PredicateOp::NONE};
};

struct GraphRecordedNode {
    std::array<int32_t, PTO2_SUBTASK_SLOT_COUNT> kernel_ids{};
    ActiveMask active_mask{};
    TaskAttrs task_attrs{};
    int16_t logical_block_num{1};
    int16_t total_required_subtasks{0};
    size_t total_output_size{0};
    uintptr_t record_packed_base{0};
    // Element addresses are handed to the caller through TaskOutputTensors, so
    // this one stays per node: its heap buffer has to outlive every later node's
    // recording. The arrays whose addresses are never borrowed live flat on the
    // recording instead — see GraphRecording.
    std::vector<ChipTensor> tensors;
    // Ranges into the recording's flat arrays. tensor_sources has one entry per
    // tensor, so tensors.size() is its count.
    uint32_t tensor_source_offset{0};
    uint32_t scalar_offset{0};
    uint32_t scalar_count{0};
    uint32_t fanin_offset{0};
    uint32_t fanin_count{0};
    // Index into the recording's predicates, or -1 when the node carries none.
    int32_t predicate_index{-1};
    ArgsDumpTaskMetadata dump_metadata;
};

// One recorded node's scratch output window. reserve_heap_scratch is a pure bump
// and a node stores the aligned size it advanced by, so consecutive windows abut:
// held in record order these are sorted and disjoint, which is what lets an
// address lookup binary search instead of walking every producer. A node with no
// output advances nothing and owns no entry.
struct GraphRecordedOutputRange {
    uintptr_t begin;
    uintptr_t end;
    uint32_t node_index;
};

struct GraphRecording {
    uint64_t full_key{0};
    int32_t start_local_task_id{0};
    uint64_t next_virtual_offset{0};
    // Worker-owned deep copy of the first Graph boundary. It stays valid from
    // graph_prepare through graph_end and anchors boundary scalar sources while
    // the main thread submits outer shells from later invocation arguments.
    const GraphTaskArgs *boundary_args{nullptr};
    int32_t boundary_scalar_count{0};
    bool unsupported{false};
    std::vector<ChipTensor> boundary_tensors;
    std::vector<TensorArgType> boundary_types;
    std::vector<GraphRecordedNode> nodes;
    // Flat per-node arrays, indexed by the ranges on GraphRecordedNode. Held
    // here rather than on each node so recording a graph pays a handful of
    // amortized growths instead of one allocation per node per array.
    std::vector<GraphRecordedTensorSourceRef> tensor_sources;
    std::vector<uint64_t> scalars;
    std::vector<GraphRecordedScalarSourceRef> scalar_sources;
    std::vector<size_t> internal_fanins;
    std::vector<GraphRecordedOutputRange> output_ranges;
    // Indexed by GraphRecordedNode::predicate_index; only predicated nodes
    // contribute an entry.
    std::vector<GraphRecordedPredicate> predicates;
    // Hazard state for the recorded body, owned per recording because several
    // graphs record at once, each on its own thread.
    //
    // The ordinary submit path reads a task's producers out of orch->tensor_map
    // (compute_task_fanin, STEP 3) and publishes the task's writes back into it
    // (register_task_outputs, STEP 4). The shadow-record path replaces
    // submit_task_common wholesale, so without a map of its own the recorder
    // can only see the edges tensor-source classification yields — and that
    // classification answers "which node's packed window holds these bytes",
    // i.e. who ALLOCATED the buffer, never who wrote it last. A body that
    // allocates once with alloc_tensors and then writes in place with add_inout
    // (the shape every generated orchestration uses) would therefore record a
    // node with no edge to its actual producer, and the Definition would replay
    // a DAG the same body never had when submitted task by task.
    DeviceArena tensor_map_arena;
    PTO2TensorMapLayout tensor_map_layout{};
    PTO2TensorMap tensor_map{};
    bool tensor_map_ready{false};
    // Scope depth as the body sees it. begin_scope/end_scope leave the real
    // orchestrator stack untouched while recording (a Graph replays flat), but
    // the manual-scope flag still has to follow the body: a manual scope
    // suppresses inference on the ring, so it must suppress it here too.
    int32_t scope_stack_top{-1};
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }
};

struct GraphPendingUpload {
    PTO2TaskSlotState *outer_slot{nullptr};
    std::vector<std::byte> image;
    bool deferred_heap{false};
};

enum class GraphRecordingStatus : uint8_t { RECORDING = 0, READY = 1, FAILED = 2 };

// One Definition being recorded. Entries are keyed by Graph key and held by
// unique_ptr, so a rehash of the owning map never moves one: the recording
// thread is handed this address at graph_begin and dereferences it without
// taking recording_mutex.
struct GraphInflightRecording {
    uint64_t full_key{0};
    std::unique_ptr<GraphRecording> recording;
    // Atomic because graph_prepare reads it on the recording thread without
    // taking recording_mutex, by design: acquiring the mutex there lets a
    // main-thread burst of same-key submissions starve the thread before it can
    // bind its private recording state. Every other access holds the mutex.
    std::atomic<GraphRecordingStatus> recording_status{GraphRecordingStatus::RECORDING};

    GraphRecordingStatus status() const { return recording_status.load(std::memory_order_acquire); }
    void set_status(GraphRecordingStatus next) { recording_status.store(next, std::memory_order_release); }
};

struct GraphHostState {
    std::unordered_map<uint64_t, std::vector<std::byte>> definitions;
    // Recordings in flight, at most one per Graph key. Several record at once,
    // each on its own thread; graph_commit drains and finalizes all of them.
    std::unordered_map<uint64_t, std::unique_ptr<GraphInflightRecording>> inflight;
    std::vector<GraphPendingUpload> pending_uploads;
    std::mutex recording_mutex;
    std::condition_variable recording_cv;
    // Mirrors inflight.size() so orchestration completion answers the common
    // "nothing is recording" case without taking recording_mutex.
    std::atomic<size_t> inflight_count{0};

    // Definitions this run can still admit: published plus in flight, against
    // the per-worker cache limit.
    size_t claimed_definitions() const { return definitions.size() + inflight.size(); }
    bool any_recording() const {
        for (const auto &entry : inflight) {
            if (entry.second->status() == GraphRecordingStatus::RECORDING) return true;
        }
        return false;
    }
};

namespace {

GraphHostState *graph_state_from(PTO2OrchestratorState *orch) {
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

GraphRecording *active_graph_recording(PTO2OrchestratorState *orch) {
    GraphHostState *state = graph_state_from(orch);
    if (state == nullptr || state != g_active_graph_owner) return nullptr;
    return g_active_graph_recording;
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

bool graph_tensor_exact(const ChipTensor &lhs, const ChipTensor &rhs) {
    if (lhs.ndims > MAX_TENSOR_DIMS || rhs.ndims > MAX_TENSOR_DIMS || lhs.buffer.addr != rhs.buffer.addr ||
        lhs.buffer.size != rhs.buffer.size || lhs.start_offset != rhs.start_offset || lhs.version != rhs.version ||
        lhs.ndims != rhs.ndims || lhs.dtype != rhs.dtype || lhs.manual_dep != rhs.manual_dep ||
        lhs.is_contiguous != rhs.is_contiguous || lhs.address_space != rhs.address_space) {
        return false;
    }
    return std::equal(std::begin(lhs.shapes), std::begin(lhs.shapes) + lhs.ndims, std::begin(rhs.shapes)) &&
           std::equal(std::begin(lhs.strides), std::begin(lhs.strides) + lhs.ndims, std::begin(rhs.strides));
}

bool graph_tensor_from_boundary(
    const GraphRecording &recording, const ChipTensor &tensor, GraphRecordedTensorSourceRef *source
) {
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        if (!graph_tensor_exact(tensor, recording.boundary_tensors[i])) continue;
        source->source = GraphRecordedTensorSource::BOUNDARY_EXACT;
        source->source_index = i;
        source->packed_offset = 0;
        return true;
    }
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        const ChipTensor &boundary = recording.boundary_tensors[i];
        if (tensor.buffer.addr != boundary.buffer.addr || tensor.buffer.size != boundary.buffer.size ||
            tensor.start_offset < boundary.start_offset) {
            continue;
        }
        source->source = GraphRecordedTensorSource::BOUNDARY_VIEW;
        source->source_index = i;
        source->packed_offset = tensor.start_offset - boundary.start_offset;
        return true;
    }
    return false;
}

template <typename ArgT>
GraphRecordedScalarSourceRef
graph_classify_scalar(const GraphRecording &recording, const ArgT &args, int32_t scalar_index) {
    if (recording.boundary_args == nullptr) return {};
    // Identity, not type: an internal node's Arg and the boundary Arg have
    // different capacities, so compare the addresses through void.
    if (static_cast<const void *>(&args) == static_cast<const void *>(recording.boundary_args) &&
        scalar_index < recording.boundary_args->scalar_count()) {
        return GraphRecordedScalarSourceRef{GraphRecordedScalarSource::BOUNDARY, static_cast<size_t>(scalar_index)};
    }

    const void *source = args.scalar_source(scalar_index);
    const void *invalidated_source = args.invalidated_scalar_source(scalar_index);
    if (source == nullptr && invalidated_source == nullptr) return {};
    for (int32_t i = 0; i < recording.boundary_args->scalar_count(); ++i) {
        const void *boundary_source = static_cast<const void *>(&recording.boundary_args->scalar(i));
        if (source == boundary_source) {
            return GraphRecordedScalarSourceRef{GraphRecordedScalarSource::BOUNDARY, static_cast<size_t>(i)};
        }
        if (invalidated_source == boundary_source) {
            return GraphRecordedScalarSourceRef{
                GraphRecordedScalarSource::INVALIDATED_BOUNDARY, static_cast<size_t>(i)
            };
        }
    }
    return {};
}

// Entry capacity for one recorded body's hazard map. A Definition is capped at
// GRAPH_MAX_NODES nodes and each node registers at most its INOUT/OUTPUT_EXISTING
// args, so this bounds the worst realistic body while staying a small fraction of
// the ring path's whole-orchestration pool (PTO2_TENSORMAP_POOL_SIZE). Exhausting
// it marks the recording unsupported, which graph_commit reports as
// PTO2_ERROR_INVALID_ARGS -- the outer shell is already submitted by then, so there
// is no ordinary-path fallback left to take.
constexpr int32_t GRAPH_RECORD_TENSORMAP_POOL_SIZE = 16384;

// Stand the recording's hazard map up on its own host allocation. Failure is
// reported to the caller, which is still before the outer shell is submitted and
// so can still take the ordinary path, rather than producing a Definition with
// inferred edges missing.
bool graph_recording_init_tensor_map(GraphRecording &recording) {
    recording.tensor_map_layout = PTO2TensorMap::reserve_layout(
        recording.tensor_map_arena, PTO2_TENSORMAP_NUM_BUCKETS, GRAPH_RECORD_TENSORMAP_POOL_SIZE, GRAPH_MAX_NODES
    );
    if (recording.tensor_map_arena.commit() == nullptr) return false;
    if (!recording.tensor_map.init_data_from_layout(recording.tensor_map_layout, recording.tensor_map_arena)) {
        return false;
    }
    recording.tensor_map.wire_arena_pointers(recording.tensor_map_layout, recording.tensor_map_arena);
    recording.tensor_map_ready = true;
    return true;
}

bool graph_classify_tensor(
    const GraphRecording &recording, const GraphRecordedNode &current, int32_t task_index, const ChipTensor &tensor,
    GraphRecordedTensorSourceRef *source
) {
    if (graph_tensor_from_boundary(recording, tensor, source)) return true;
    const uintptr_t tensor_addr = static_cast<uintptr_t>(tensor.buffer.addr);
    // The node being recorded is not in output_ranges yet — its entry is appended
    // once its own tensors are classified — so its window is tested here, and a
    // hit is OWN_OUTPUT rather than a dependency.
    if (current.record_packed_base != 0 && current.total_output_size != 0 &&
        current.total_output_size <= UINTPTR_MAX - current.record_packed_base) {
        const uintptr_t begin = current.record_packed_base;
        if (tensor_addr >= begin && tensor_addr < begin + current.total_output_size) {
            source->source = GraphRecordedTensorSource::OWN_OUTPUT;
            source->source_index = static_cast<size_t>(task_index);
            source->packed_offset = tensor_addr - begin;
            return true;
        }
    }
    // Sorted and disjoint, so the only window that can hold the address is the
    // last one starting at or below it.
    const auto after = std::upper_bound(
        recording.output_ranges.begin(), recording.output_ranges.end(), tensor_addr,
        [](uintptr_t addr, const GraphRecordedOutputRange &range) {
            return addr < range.begin;
        }
    );
    if (after == recording.output_ranges.begin()) return false;
    const GraphRecordedOutputRange &range = *(after - 1);
    if (tensor_addr >= range.end) return false;
    source->source = GraphRecordedTensorSource::INTERNAL;
    source->source_index = range.node_index;
    source->packed_offset = tensor_addr - range.begin;
    return true;
}

GraphBoundarySignature graph_boundary_signature(const ChipTensor &tensor, TensorArgType type, uint16_t alias_rep) {
    GraphBoundarySignature signature{};
    signature.buffer_size = tensor.buffer.size;
    std::copy(std::begin(tensor.shapes), std::end(tensor.shapes), std::begin(signature.shapes));
    std::copy(std::begin(tensor.strides), std::end(tensor.strides), std::begin(signature.strides));
    signature.alias_rep = alias_rep;
    signature.ndims = static_cast<uint8_t>(tensor.ndims);
    signature.dtype = static_cast<uint8_t>(tensor.dtype);
    signature.tag = static_cast<uint8_t>(type);
    signature.manual_dep = tensor.manual_dep ? 1 : 0;
    signature.is_contiguous = tensor.is_contiguous ? 1 : 0;
    return signature;
}

template <typename T>
uint32_t graph_append_section(std::vector<std::byte> *image, const std::vector<T> &values) {
    if (values.empty()) return 0;
    if (image->size() > UINT32_MAX || values.size() > UINT32_MAX / sizeof(T)) return 0;
    const size_t aligned = PTO2_ALIGN_UP(image->size(), alignof(T));
    const size_t bytes = values.size() * sizeof(T);
    if (aligned > UINT32_MAX || bytes > UINT32_MAX - aligned) return 0;
    image->resize(aligned + bytes);
    std::memcpy(image->data() + aligned, values.data(), bytes);
    return static_cast<uint32_t>(aligned);
}

std::optional<GraphTensorSourceRef> graph_pack_tensor_source(const GraphRecordedTensorSourceRef &source) {
    if (source.source_index > UINT16_MAX) return std::nullopt;

    GraphTensorSourceRef packed{};
    switch (source.source) {
    case GraphRecordedTensorSource::BOUNDARY_EXACT:
        packed.source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_EXACT);
        break;
    case GraphRecordedTensorSource::BOUNDARY_VIEW:
        packed.source = static_cast<uint8_t>(GraphTensorSource::BOUNDARY_VIEW);
        break;
    case GraphRecordedTensorSource::INTERNAL:
        packed.source = static_cast<uint8_t>(GraphTensorSource::INTERNAL);
        break;
    case GraphRecordedTensorSource::OWN_OUTPUT:
        packed.source = static_cast<uint8_t>(GraphTensorSource::OWN_OUTPUT);
        break;
    }
    packed.source_index = static_cast<uint16_t>(source.source_index);
    packed.packed_offset = source.packed_offset;
    return packed;
}

std::optional<GraphScalarSourceRef> graph_pack_scalar_source(const GraphRecordedScalarSourceRef &source) {
    if (source.source == GraphRecordedScalarSource::INVALIDATED_BOUNDARY || source.source_index > UINT16_MAX) {
        return std::nullopt;
    }

    GraphScalarSourceRef packed{};
    packed.source = source.source == GraphRecordedScalarSource::BOUNDARY ?
                        static_cast<uint8_t>(GraphScalarSource::BOUNDARY) :
                        static_cast<uint8_t>(GraphScalarSource::STATIC_VALUE);
    packed.source_index = static_cast<uint16_t>(source.source_index);
    return packed;
}

bool graph_build_definition(const GraphRecording &recording, std::vector<std::byte> *image) {
    if (image == nullptr || recording.unsupported || recording.nodes.empty() ||
        recording.nodes.size() > GRAPH_MAX_NODES || recording.boundary_tensors.size() > UINT16_MAX ||
        recording.boundary_tensors.size() != recording.boundary_types.size() || recording.boundary_args == nullptr ||
        std::any_of(recording.boundary_tensors.begin(), recording.boundary_tensors.end(), [](const ChipTensor &tensor) {
            return tensor.ndims > MAX_TENSOR_DIMS;
        })) {
        return false;
    }

    // Every per-node array is grown by push_back below, and the recording
    // already knows how many entries each node contributes, so size them once
    // here. Left unreserved, packing 277 nodes' args reallocates each vector a
    // dozen times and copies its whole contents each time.
    size_t total_tensors = 0;
    for (const GraphRecordedNode &source : recording.nodes) {
        total_tensors += source.tensors.size();
    }
    const size_t total_scalars = recording.scalars.size();
    const size_t total_fanins = recording.internal_fanins.size();
    std::vector<uint32_t> fanout_counts(recording.nodes.size(), 0);
    std::vector<uint32_t> fanin_offsets(recording.nodes.size() + 1, 0);
    std::vector<uint16_t> fanin_indices;
    std::vector<uint16_t> roots;
    std::vector<uint64_t> node_offsets(recording.nodes.size(), 0);
    std::vector<GraphNodeDefinition> nodes(recording.nodes.size());
    std::vector<GraphTensor> tensors;
    std::vector<GraphTensorSourceRef> tensor_sources;
    std::vector<uint64_t> scalars;
    std::vector<GraphScalarSourceRef> scalar_sources;
    std::vector<GraphPredicate> predicates;
    predicates.reserve(recording.predicates.size());
    fanin_indices.reserve(total_fanins);
    roots.reserve(recording.nodes.size());
    tensors.reserve(total_tensors);
    tensor_sources.reserve(total_tensors);
    scalars.reserve(total_scalars);
    scalar_sources.reserve(total_scalars);

    uint64_t required_heap = 0;
    uint32_t edge_count = 0;
    for (size_t i = 0; i < recording.nodes.size(); ++i) {
        const GraphRecordedNode &source = recording.nodes[i];
        if (source.total_output_size > static_cast<size_t>(INT32_MAX) ||
            source.tensors.size() > static_cast<size_t>(INT32_MAX) ||
            source.scalar_count > static_cast<uint32_t>(INT32_MAX) || source.fanin_count > UINT16_MAX ||
            tensors.size() > UINT32_MAX - source.tensors.size() ||
            tensor_sources.size() > UINT32_MAX - source.tensors.size() ||
            scalars.size() > UINT32_MAX - source.scalar_count ||
            scalar_sources.size() > UINT32_MAX - source.scalar_count ||
            std::any_of(source.tensors.begin(), source.tensors.end(), [](const ChipTensor &tensor) {
                return tensor.ndims > MAX_TENSOR_DIMS;
            })) {
            return false;
        }
        node_offsets[i] = required_heap;
        const uint64_t output_bytes = PTO2_ALIGN_UP(source.total_output_size, PTO2_ALIGN_SIZE);
        if (required_heap > UINT64_MAX - output_bytes) return false;
        required_heap += output_bytes;

        fanin_offsets[i + 1] = fanin_offsets[i] + source.fanin_count;
        if (source.fanin_count == 0) roots.push_back(static_cast<uint16_t>(i));
        for (uint32_t f = 0; f < source.fanin_count; ++f) {
            const size_t producer = recording.internal_fanins[source.fanin_offset + f];
            if (producer >= i) return false;
            fanout_counts[producer]++;
            fanin_indices.push_back(static_cast<uint16_t>(producer));
            edge_count++;
        }

        GraphNodeDefinition &node = nodes[i];
        std::copy(source.kernel_ids.begin(), source.kernel_ids.end(), std::begin(node.kernel_id));
        node.active_mask = source.active_mask.raw();
        node.task_attrs = source.task_attrs.raw();
        node.logical_block_num = source.logical_block_num;
        node.total_required_subtasks = source.total_required_subtasks;
        node.tensor_count = static_cast<int32_t>(source.tensors.size());
        node.scalar_count = static_cast<int32_t>(source.scalar_count);
        node.total_output_size = static_cast<int32_t>(source.total_output_size);
        node.tensor_offset = static_cast<uint32_t>(tensors.size());
        node.scalar_offset = static_cast<uint32_t>(scalars.size());
        node.dump_metadata = source.dump_metadata;
        node.predicate_slot = 0;
        if (source.predicate_index >= 0) {
            if (static_cast<size_t>(source.predicate_index) >= recording.predicates.size() ||
                predicates.size() >= static_cast<size_t>(UINT16_MAX)) {
                return false;
            }
            const GraphRecordedPredicate &recorded = recording.predicates[source.predicate_index];
            std::optional<GraphTensorSourceRef> packed_source = graph_pack_tensor_source(recorded.source);
            if (!packed_source.has_value() || recorded.operand.ndims > MAX_TENSOR_DIMS) return false;
            GraphPredicate packed{};
            packed.operand = graph_tensor_pack(recorded.operand);
            packed.operand_source = *packed_source;
            packed.elem_offset = recorded.elem_offset;
            packed.target = recorded.target;
            packed.elem_size = recorded.elem_size;
            packed.op = static_cast<uint8_t>(recorded.op);
            node.predicate_slot = static_cast<uint16_t>(predicates.size() + 1);
            predicates.push_back(packed);
        }
        for (const ChipTensor &tensor : source.tensors)
            tensors.push_back(graph_tensor_pack(tensor));
        for (size_t t = 0; t < source.tensors.size(); ++t) {
            std::optional<GraphTensorSourceRef> packed_source =
                graph_pack_tensor_source(recording.tensor_sources[source.tensor_source_offset + t]);
            if (!packed_source.has_value()) return false;
            tensor_sources.push_back(*packed_source);
        }
        for (size_t scalar_index = 0; scalar_index < source.scalar_count; ++scalar_index) {
            std::optional<GraphScalarSourceRef> packed_source =
                graph_pack_scalar_source(recording.scalar_sources[source.scalar_offset + scalar_index]);
            if (!packed_source.has_value() ||
                (packed_source->source == static_cast<uint8_t>(GraphScalarSource::BOUNDARY) &&
                 packed_source->source_index >= recording.boundary_args->scalar_count())) {
                return false;
            }
            scalar_sources.push_back(*packed_source);
            scalars.push_back(
                packed_source->source == static_cast<uint8_t>(GraphScalarSource::BOUNDARY) ?
                    0 :
                    recording.scalars[source.scalar_offset + scalar_index]
            );
        }
    }

    std::vector<uint32_t> fanout_offsets(recording.nodes.size() + 1, 0);
    for (size_t i = 0; i < recording.nodes.size(); ++i)
        fanout_offsets[i + 1] = fanout_offsets[i] + fanout_counts[i];
    std::vector<uint16_t> fanout_indices(edge_count);
    std::vector<uint32_t> cursors(fanout_offsets.begin(), fanout_offsets.end() - 1);
    for (size_t consumer = 0; consumer < recording.nodes.size(); ++consumer) {
        const GraphRecordedNode &consumer_node = recording.nodes[consumer];
        for (uint32_t f = 0; f < consumer_node.fanin_count; ++f) {
            const size_t producer = recording.internal_fanins[consumer_node.fanin_offset + f];
            fanout_indices[cursors[producer]++] = static_cast<uint16_t>(consumer);
        }
    }

    std::vector<GraphBoundarySignature> signatures;
    signatures.reserve(recording.boundary_tensors.size());
    for (size_t i = 0; i < recording.boundary_tensors.size(); ++i) {
        uint16_t alias_rep = static_cast<uint16_t>(i);
        for (size_t j = 0; j < i; ++j) {
            if (recording.boundary_tensors[j].buffer.addr == recording.boundary_tensors[i].buffer.addr &&
                recording.boundary_tensors[j].buffer.size == recording.boundary_tensors[i].buffer.size) {
                alias_rep = static_cast<uint16_t>(j);
                break;
            }
        }
        signatures.push_back(
            graph_boundary_signature(recording.boundary_tensors[i], recording.boundary_types[i], alias_rep)
        );
    }

    image->assign(sizeof(GraphDefinition), std::byte{0});
    GraphDefinition definition{};
    definition.full_key = recording.full_key;
    definition.required_heap = required_heap;
    definition.task_count = static_cast<uint32_t>(nodes.size());
    definition.edge_count = edge_count;
    definition.root_count = static_cast<uint32_t>(roots.size());
    definition.boundary_count = static_cast<uint32_t>(signatures.size());
    definition.boundary_scalar_count = static_cast<uint32_t>(recording.boundary_scalar_count);
    definition.tensor_arg_count = static_cast<uint32_t>(tensors.size());
    definition.scalar_arg_count = static_cast<uint32_t>(scalars.size());
    definition.predicate_count = static_cast<uint32_t>(predicates.size());
    // The widest node sets the stride every entry of this Definition's execution
    // storage gets; nodes[] carries each node's declared tensor count.
    int32_t widest_node_tensor_count = 0;
    for (const GraphNodeDefinition &node : nodes) {
        if (node.tensor_count > widest_node_tensor_count) widest_node_tensor_count = node.tensor_count;
    }
    const size_t node_stride = graph_node_stride(widest_node_tensor_count);
    size_t execution_storage_bytes = 0;
    if (node_stride > UINT32_MAX ||
        !graph_execution_storage_bytes(
            static_cast<int32_t>(definition.task_count), node_stride, &execution_storage_bytes
        ) ||
        execution_storage_bytes > UINT32_MAX) {
        return false;
    }
    definition.node_stride = static_cast<uint32_t>(node_stride);
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);
    // Every section's byte count is known now, so the image grows once instead of
    // resizing eleven times. Each append aligns its start, hence the per-section
    // alignment slack.
    auto section_bytes = [](size_t count, size_t elem_size, size_t align) {
        return count == 0 ? size_t{0} : (align - 1) + count * elem_size;
    };
    image->reserve(
        image->size() + section_bytes(fanout_offsets.size(), sizeof(uint32_t), alignof(uint32_t)) +
        section_bytes(fanout_indices.size(), sizeof(uint16_t), alignof(uint16_t)) +
        section_bytes(fanin_offsets.size(), sizeof(uint32_t), alignof(uint32_t)) +
        section_bytes(fanin_indices.size(), sizeof(uint16_t), alignof(uint16_t)) +
        section_bytes(roots.size(), sizeof(uint16_t), alignof(uint16_t)) +
        section_bytes(node_offsets.size(), sizeof(uint64_t), alignof(uint64_t)) +
        section_bytes(nodes.size(), sizeof(GraphNodeDefinition), alignof(GraphNodeDefinition)) +
        section_bytes(tensors.size(), sizeof(GraphTensor), alignof(GraphTensor)) +
        section_bytes(tensor_sources.size(), sizeof(GraphTensorSourceRef), alignof(GraphTensorSourceRef)) +
        section_bytes(scalars.size(), sizeof(uint64_t), alignof(uint64_t)) +
        section_bytes(scalar_sources.size(), sizeof(GraphScalarSourceRef), alignof(GraphScalarSourceRef)) +
        section_bytes(signatures.size(), sizeof(GraphBoundarySignature), alignof(GraphBoundarySignature)) +
        section_bytes(predicates.size(), sizeof(GraphPredicate), alignof(GraphPredicate))
    );
    definition.off_fanout_offsets = graph_append_section(image, fanout_offsets);
    definition.off_fanout_indices = graph_append_section(image, fanout_indices);
    definition.off_fanin_offsets = graph_append_section(image, fanin_offsets);
    definition.off_fanin_indices = graph_append_section(image, fanin_indices);
    definition.off_root_indices = graph_append_section(image, roots);
    definition.off_node_offsets = graph_append_section(image, node_offsets);
    definition.off_nodes = graph_append_section(image, nodes);
    definition.off_tensors = graph_append_section(image, tensors);
    definition.off_tensor_sources = graph_append_section(image, tensor_sources);
    definition.off_scalars = graph_append_section(image, scalars);
    definition.off_scalar_sources = graph_append_section(image, scalar_sources);
    definition.off_boundary_signatures = graph_append_section(image, signatures);
    definition.off_predicates = graph_append_section(image, predicates);
    if (definition.off_fanout_offsets == 0 || definition.off_fanin_offsets == 0 || definition.off_node_offsets == 0 ||
        definition.off_nodes == 0 || definition.off_boundary_signatures == 0 ||
        (!tensors.empty() && definition.off_tensors == 0) ||
        (!tensor_sources.empty() && definition.off_tensor_sources == 0) ||
        (!scalars.empty() && definition.off_scalars == 0) ||
        (!scalar_sources.empty() && definition.off_scalar_sources == 0) ||
        (!fanout_indices.empty() && definition.off_fanout_indices == 0) ||
        (!fanin_indices.empty() && definition.off_fanin_indices == 0) ||
        (!roots.empty() && definition.off_root_indices == 0) ||
        (!predicates.empty() && definition.off_predicates == 0)) {
        return false;
    }
    definition.total_bytes = static_cast<uint32_t>(image->size());
    std::memcpy(image->data(), &definition, sizeof(definition));
    definition.content_hash = graph_hash_bytes(1469598103934665603ULL, image->data(), image->size());
    std::memcpy(image->data(), &definition, sizeof(definition));
    return true;
}

const GraphDefinition *graph_definition(const std::vector<std::byte> &image) {
    if (image.size() < sizeof(GraphDefinition)) return nullptr;
    const auto *definition = reinterpret_cast<const GraphDefinition *>(image.data());
    return definition->total_bytes == image.size() ? definition : nullptr;
}

}  // namespace

GraphHostStatePtr make_graph_host_state() { return GraphHostStatePtr{new (std::nothrow) GraphHostState{}}; }

void GraphHostStateDeleter::operator()(GraphHostState *state) const noexcept { delete state; }

size_t graph_host_upload_count(const GraphHostState &state) { return state.pending_uploads.size(); }

std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index) {
    if (index >= state.pending_uploads.size()) return std::nullopt;
    GraphPendingUpload &upload = state.pending_uploads[index];
    if (upload.outer_slot == nullptr || upload.image.empty()) return std::nullopt;
    return GraphHostUpload{upload.outer_slot, upload.image.data(), upload.image.size()};
}

GraphHostDefinitionList graph_host_definitions(GraphHostState &state) {
    GraphHostDefinitionList list;
    list.entries.reserve(state.definitions.size());
    for (auto &[key, image] : state.definitions) {
        const GraphDefinition *header = graph_definition(image);
        if (header != nullptr && header->total_bytes == image.size()) {
            list.entries.push_back(GraphHostDefinition{key, image.data(), image.size()});
        }
    }
    return list;
}

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(orch->fanin_seen_epoch, 0, static_cast<size_t>(orch->sm_header->ring.task_capacity) * sizeof(uint32_t));
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

// Polling: fanin is a flat array of position-independent producer local ids on
// the payload (no dep-pool spill, no producer pointers). The builder writes them
// directly into payload->fanin_local_ids as producers are appended, deduping by
// slot and hard-capping at PTO2_MAX_FANIN. self_local is this task's own local id
// (the consumer), used to bump each producer's last_consumer_local_id (the
// reclaim gate the host wait_for_consumers polls via completed_watermark).
struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2OrchestratorState *orch, PTO2TaskPayload *payload, int32_t self_local, uint32_t seen_epoch) :
        count(0),
        orch(orch),
        seen_epoch(seen_epoch),
        self_local(self_local),
        payload(payload) {}
    int32_t count{0};
    PTO2OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    int32_t self_local{0};
    PTO2TaskPayload *payload{nullptr};

    bool mark_seen(int32_t prod_slot) {
        uint32_t *seen = orch->fanin_seen_epoch;
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }
};

static bool
append_fanin_or_fail(PTO2OrchestratorState *orch, PTO2TaskId producer_task_id, PTO2FaninBuilder *fanin_builder) {
    if (!producer_task_id.is_valid() || producer_task_id.ring() != 0) {
        orch->report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "HBG dependencies require valid ring-zero task ids");
        return false;
    }

    PTO2SharedMemoryRingHeader &ring = orch->sm_header->ring;
    const int32_t producer_local = static_cast<int32_t>(producer_task_id.local());
    const int32_t prod_slot = ring.get_slot_by_task_id(producer_local);
    PTO2TaskSlotState *prod_state = &ring.get_slot_state_by_slot(prod_slot);

    // Skip a stale/reused producer slot: the cached owner id no longer resolves
    // to this producer (defensive — whole-graph-resident hbg does not reuse slots
    // at build time). A COMPLETED producer IS a real fanin edge under polling (its
    // completion_flags byte is set), so it is not skipped.
    if (prod_state->task == nullptr || prod_state->task->task_id != producer_task_id) {
        return true;
    }
    // Dedup by slot; HBG has one task-capacity domain.
    if (fanin_builder->mark_seen(prod_slot)) {
        return true;
    }
    if (fanin_builder->count >= PTO2_MAX_FANIN) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Fanin Capacity Exhausted!");
        LOG_ERROR("========================================");
        LOG_ERROR("HBG stores every producer dependency inline on the consumer task.");
        LOG_ERROR("  Fanin:     used=%d/%d", fanin_builder->count, PTO2_MAX_FANIN);
        LOG_ERROR("  Requested: at least %d distinct producer dependencies", fanin_builder->count + 1);
        LOG_ERROR("Solution:");
        LOG_ERROR("  Reduce the task fanin to at most PTO2_MAX_FANIN=%d.", PTO2_MAX_FANIN);
        LOG_ERROR("  HBG has no dependency spill pool; PTO2_RING_DEP_POOL does not apply.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    fanin_builder->payload->fanin_local_ids[fanin_builder->count++] = static_cast<int32_t>(producer_task_id.local());

    // Reclaim gate: record this task as a consumer of the producer. The producer
    // slot retires once completed_watermark reaches this consumer id.
    if (fanin_builder->self_local > prod_state->last_consumer_local_id) {
        prod_state->last_consumer_local_id = fanin_builder->self_local;
    }
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout calculate_output_layout(const CoreTaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PTO2PreparedTask *out
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");
    auto &allocator = orch->task_allocator;

    int16_t block_num = args.launch_spec.block_num();
    int32_t active_subtasks_per_block = __builtin_popcount(active_mask.core_mask());
    int32_t total_required_subtasks = static_cast<int32_t>(block_num) * active_subtasks_per_block;
    if (block_num <= 0 || total_required_subtasks > std::numeric_limits<int16_t>::max()) {
        orch->report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "block_num=%d with %d active slots requires %d subtasks; expected block_num >= 1 and total <= %d",
            block_num, active_subtasks_per_block, total_required_subtasks, std::numeric_limits<int16_t>::max()
        );
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(0, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->ring.get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->ring.task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->ring.task_payloads[out->alloc_result.slot];

    // Init-on-write: this slot's dynamic scheduling fields and completion flag are
    // initialized here, as the orchestrator claims the slot. whole-graph-resident
    // hbg claims slots [0, total_tasks) exactly once and the device reads no slot
    // past total_tasks, so this claim-time write is the only per-slot SM reset and
    // the unclaimed tail is neither initialized nor read.
    out->slot_state->reset_for_reuse();
    orch->sm_header->ring.completion_flags[out->alloc_result.slot].store(0, std::memory_order_relaxed);

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(task_capacity) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in PTO2TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by the reset_for_reuse() above:
    //   wake_list_head=nullptr, next_in_wake_list=nullptr,
    //   any_subtask_deferred=false, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state is set to PENDING here as the orchestrator populates the slot
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    out->slot_state->total_required_subtasks = static_cast<int16_t>(total_required_subtasks);
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    out->slot_state->task_kind = active_mask ? TaskKind::KERNEL : TaskKind::DUMMY;
    // Reclaim gate: seed last_consumer to self, so a producer with no consumers
    // is retirable once completed_watermark >= its own id. Each fanin edge bumps
    // it in append_fanin_or_fail. completion_flags for this slot were cleared
    // above (whole-graph-resident hbg never reuses a slot).
    out->slot_state->last_consumer_local_id = static_cast<int32_t>(out->task_id.local());
    // payload.fanin_count is set in submit_task_common's STEP 6.
    scope_tasks_push(orch, out->slot_state);

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        // scope_tasks lives in the per-Worker arena (single backing allocation),
        // so realloc is not legal. Capacity is the total in-flight slot budget
        // (the runtime task capacity; see reserve_layout). Once saturated, no
        // further push can succeed regardless of buffer growth.
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__, "scope_tasks buffer saturated at %d entries",
            orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // A Graph replays as a flat DAG with no scope structure: scope boundaries only
    // shape scheduling on the ring, and the shadow-record path submits no ring
    // tasks. So a scope inside a Graph body must not touch the real scope stack.
    // Its manual/auto mode still matters, though — the recorder infers a node's
    // producers with the same compute_task_fanin the ring path uses, and that
    // inference is suppressed inside a manual scope — so the depth is tracked on
    // the recording instead.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        // Reject what the ring rejects. An auto scope inside a manual one is a
        // fatal below; accepting it here would let a Graph record and replay a
        // body that ordinary submission refuses. The push still happens so
        // end_scope stays balanced -- the recording is doomed either way, since
        // graph_commit turns an unsupported recording into PTO2_ERROR_INVALID_ARGS.
        if (recording->scope_stack_top >= PTO2_MAX_SCOPE_DEPTH - 1 ||
            (mode == PTO2ScopeMode::AUTO && recording->in_manual_scope())) {
            recording->unsupported = true;
        }
        if (recording->scope_stack_top < PTO2_MAX_SCOPE_DEPTH - 1) {
            ++recording->scope_stack_top;
            if (mode == PTO2ScopeMode::MANUAL && !recording->in_manual_scope()) {
                recording->manual_begin_depth = recording->scope_stack_top;
            }
        }
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        // Task and heap allocation frontiers are forward-only, so their reclaim ends stay at 0.
        scope_stats_begin(
            ring_id, /*task_start=*/0, alloc.task_head(), /*heap_start=*/0, alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // Matches begin_scope: a scope inside a Graph body never touches the real
    // scope stack, only the recording's own manual-scope depth.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        if (recording->scope_stack_top >= 0) {
            if (recording->manual_begin_depth == recording->scope_stack_top) {
                recording->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
            }
            --recording->scope_stack_top;
        }
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->task_allocator;
        // Polling: no dep_pool to report (readiness is via completion_flags).
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        // Task and heap allocation frontiers are forward-only, so their reclaim ends stay at 0.
        scope_stats_end(
            ring_id, /*task_start=*/0, alloc.task_head(), /*heap_start=*/0, alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif

#if SIMPLER_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;
    if (ending_manual_scope) {
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if SIMPLER_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. Device completion never reclaims TensorMap
// entries; only synchronous dependency computation can remove a covered
// producer before this check. A pool that is still short here therefore cannot
// become large enough while the host waits: latch
// PTO2_ERROR_TENSORMAP_OVERFLOW and bail rather than letting new_entry()'s hard
// assert fire mid-registration. Returns false when the pool is exhausted or a
// fatal is already latched by another party.
static bool ensure_tensormap_capacity(PTO2OrchestratorState *orch, int32_t needed) {
    PTO2TensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }
    if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
        return false;
    }

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: TensorMap Entry Pool Exhausted!");
    LOG_ERROR("========================================");
    LOG_ERROR("Device completion does not reclaim HBG TensorMap entries.");
    LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
    LOG_ERROR("  - Free:        %d entries", tm.free_entries());
    LOG_ERROR("  - Needed:      %d entries", needed);
    LOG_ERROR("Solution:");
    LOG_ERROR("  Increase PTO2_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, PTO2_ERROR_TENSORMAP_OVERFLOW);
    return false;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    ORCH_PHASE_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: open this task's graph entry before its dependency
    // steps run, so the edges STEP 1 / STEP 3 discover attach to it. The graph
    // is recorded from the dependency path itself, which makes it the runtime's
    // own answer rather than a reconstruction — the sole source of truth for
    // fanout now that the swimlane hot path no longer records it.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, PTO2_SUBTASK_SLOT_COUNT> kernel_ids_capture{
            aic_kernel_id,
            aiv0_kernel_id,
            aiv1_kernel_id,
        };
        dep_gen_host_graph_begin_task(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), kernel_ids_capture.data(),
            args.launch_spec.block_num(), args.tensor_count(), args.tensor_data(), args.tag_data()
        );
    }

    PTO2FaninBuilder fanin_builder(orch, &payload, static_cast<int32_t>(task_id.local()), next_fanin_seen_epoch(orch));

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid() || dep_task_id.ring() != 0) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid ring-zero task ids"
            );
            return result;
        }
        if (capture_dep_graph) {
            dep_gen_host_graph_add_explicit_edge(dep_task_id.raw);
        }
        if (!append_fanin_or_fail(orch, dep_task_id, &fanin_builder)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        return append_fanin_or_fail(orch, producer_task_id, &fanin_builder);
    };

    // The capture branch instantiates compute_task_fanin with a live Annotate;
    // the plain branch keeps the un-annotated instantiation the hot path had.
    if (capture_dep_graph) {
        const bool ok =
            compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit, DepGraphAnnotate{});
        // STEP 3 is this task's last capture point, so the entry closes here
        // whether or not the fanin computation succeeded.
        dep_gen_host_graph_end_task();
        if (!ok) {
            return result;
        }
    } else {
        if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
            return result;
        }
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering, so an
    // exhausted pool reports here rather than tripping new_entry()'s hard assert
    // mid-registration.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    CYCLE_COUNT_LAP(g_orch_insert_cycle);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // append_fanin_or_fail wrote each producer's local id straight into
    // payload.fanin_local_ids and bumped its last_consumer_local_id; the count is
    // published in STEP 6 below. payload.init does not touch the fanin region.
    payload.init(args, result, prepared.alloc_result, layout);

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/ChipTensor access. Both branches write predicate.op explicitly because
    // payload slots are ring-reused; op == NONE means "always dispatch".
    {
        const CoreTaskPredicate &pred = args.predicate();
        if (pred.op != PredicateOp::NONE && pred.operand.tensor != nullptr && pred.operand.tensor->buffer.addr != 0) {
            uint64_t elem_size = get_element_size(pred.operand.tensor->dtype);
            uint64_t flat_offset = pred.operand.tensor->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
            payload.predicate.addr = pred.operand.tensor->buffer.addr + flat_offset * elem_size;
            payload.predicate.target = pred.target;
            payload.predicate.elem_size = static_cast<uint8_t>(elem_size);
            payload.predicate.op = pred.op;
        } else {
            payload.predicate.addr = 0;
            payload.predicate.op = PredicateOp::NONE;
        }
    }
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    // === STEP 6: publish the inline fanin count (device boot classifies) ===
    // Polling + host-orch: append_fanin_or_fail already wrote each producer's
    // local id into payload.fanin_local_ids and bumped its last_consumer_local_id.
    // All that remains is to record how many. There is NO fanout adjacency, NO
    // dep_pool, and NO ready routing here — the initial device boot scan classifies
    // each task once. A -1 result from classify_fanin_state routes the task through
    // push_ready_routed; otherwise the returned index selects the producer passed
    // to register_wake. Wake retargeting in register_wake may reclassify a task
    // when the selected producer is already complete.
    // The initial scan happens before the scheduler dispatch loop starts. Fanin is
    // a flat array of position-independent integers, so it crosses to the device
    // unchanged.
    payload.fanin_count = fanin_builder.count;
    (void)sched;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    ORCH_PHASE_END(HostOrchPhase::Submit, task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

namespace {

bool graph_boundary_matches(const GraphDefinition &definition, const GraphTaskArgs &args) {
    if (args.scalar_count() != static_cast<int32_t>(definition.boundary_scalar_count) ||
        args.explicit_dep_count() != 0 || args.tensor_count() != static_cast<int32_t>(definition.boundary_count)) {
        LOG_WARN(
            "[GraphExecution] fixed boundary contract mismatch: tensors=%d/%u scalars=%d/%u explicit_deps=%u",
            args.tensor_count(), definition.boundary_count, args.scalar_count(), definition.boundary_scalar_count,
            args.explicit_dep_count()
        );
        return false;
    }
    const auto *signatures = graph_definition_array<GraphBoundarySignature>(
        definition, definition.off_boundary_signatures, definition.boundary_count
    );
    if (signatures == nullptr) return false;

    bool alias_mismatch = false;
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const ChipTensor &tensor = args.tensor(i).ref();
        const GraphBoundarySignature &signature = signatures[i];
        if (tensor.ndims > MAX_TENSOR_DIMS) {
            debug_assert(tensor.ndims <= MAX_TENSOR_DIMS && "Graph boundary ChipTensor rank is not supported");
            LOG_WARN("[GraphExecution] ChipTensor rank %u exceeds the fixed Graph boundary limit", tensor.ndims);
            return false;
        }
        const auto shape_end = std::begin(tensor.shapes) + tensor.ndims;
        const auto stride_end = std::begin(tensor.strides) + tensor.ndims;
        const bool metadata_match = tensor.buffer.size == signature.buffer_size && tensor.ndims == signature.ndims &&
                                    static_cast<uint8_t>(tensor.dtype) == signature.dtype &&
                                    static_cast<uint8_t>(args.tag(i)) == signature.tag &&
                                    static_cast<uint8_t>(tensor.manual_dep ? 1 : 0) == signature.manual_dep &&
                                    static_cast<uint8_t>(tensor.is_contiguous ? 1 : 0) == signature.is_contiguous &&
                                    std::equal(std::begin(tensor.shapes), shape_end, std::begin(signature.shapes)) &&
                                    std::equal(std::begin(tensor.strides), stride_end, std::begin(signature.strides));
        if (!metadata_match) {
            debug_assert(metadata_match && "Variable Graph boundary tensor shape/metadata is not supported");
            LOG_WARN(
                "[GraphExecution] fixed tensor shape/metadata mismatch at boundary arg %d; using ordinary path", i
            );
            return false;
        }
        uint16_t alias_rep = static_cast<uint16_t>(i);
        for (int32_t j = 0; j < i; ++j) {
            const ChipTensor &other = args.tensor(j).ref();
            if (other.buffer.addr == tensor.buffer.addr && other.buffer.size == tensor.buffer.size) {
                alias_rep = static_cast<uint16_t>(j);
                break;
            }
        }
        alias_mismatch |= alias_rep != signature.alias_rep;
    }
    if (alias_mismatch) {
        debug_assert(!alias_mismatch && "Changing the Graph boundary alias partition is not supported");
        LOG_WARN("%s", "[GraphExecution] boundary alias partition differs from recording; using ordinary path");
        return false;
    }
    return true;
}

bool graph_recording_boundary_matches(const GraphRecording &recording, const GraphTaskArgs &args) {
    if (args.scalar_count() != recording.boundary_scalar_count || args.explicit_dep_count() != 0 ||
        args.tensor_count() != static_cast<int32_t>(recording.boundary_tensors.size()) ||
        recording.boundary_tensors.size() != recording.boundary_types.size()) {
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const ChipTensor &expected = recording.boundary_tensors[static_cast<size_t>(i)];
        const ChipTensor &actual = args.tensor(i).ref();
        if (actual.ndims > MAX_TENSOR_DIMS || actual.buffer.size != expected.buffer.size ||
            actual.ndims != expected.ndims || actual.dtype != expected.dtype ||
            args.tag(i) != recording.boundary_types[static_cast<size_t>(i)] ||
            actual.manual_dep != expected.manual_dep || actual.is_contiguous != expected.is_contiguous ||
            !std::equal(
                std::begin(actual.shapes), std::begin(actual.shapes) + actual.ndims, std::begin(expected.shapes)
            ) ||
            !std::equal(
                std::begin(actual.strides), std::begin(actual.strides) + actual.ndims, std::begin(expected.strides)
            )) {
            return false;
        }
        uint16_t expected_alias = static_cast<uint16_t>(i);
        uint16_t actual_alias = static_cast<uint16_t>(i);
        for (int32_t j = 0; j < i; ++j) {
            const ChipTensor &expected_other = recording.boundary_tensors[static_cast<size_t>(j)];
            if (expected_other.buffer.addr == expected.buffer.addr &&
                expected_other.buffer.size == expected.buffer.size) {
                expected_alias = static_cast<uint16_t>(j);
                break;
            }
        }
        for (int32_t j = 0; j < i; ++j) {
            const ChipTensor &actual_other = args.tensor(j).ref();
            if (actual_other.buffer.addr == actual.buffer.addr && actual_other.buffer.size == actual.buffer.size) {
                actual_alias = static_cast<uint16_t>(j);
                break;
            }
        }
        if (actual_alias != expected_alias) return false;
    }
    return true;
}

void graph_reset_outer_payload(PTO2TaskPayload &payload) {
    payload.tensor_count = 0;
    payload.scalar_count = 0;
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (auto &word : payload.staged_core_mask)
        word.store(0, std::memory_order_relaxed);
    payload.dispatch_fanin.store(0, std::memory_order_relaxed);
    payload.dispatch_propagated.store(0, std::memory_order_relaxed);
    payload.published_block_count.store(0, std::memory_order_relaxed);
    payload.early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload.running_slot_count.store(0, std::memory_order_relaxed);
    payload.early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

bool graph_prepare_submission_image(
    uint64_t full_key, const GraphTaskArgs &args, std::vector<std::byte> *submission_image
) {
    if (submission_image == nullptr) return false;
    const size_t tensors_offset = PTO2_ALIGN_UP(sizeof(GraphSubmission), alignof(GraphTensor));
    const size_t tensor_bytes = static_cast<size_t>(args.tensor_count()) * sizeof(GraphTensor);
    if (tensors_offset > UINT32_MAX || tensors_offset > UINT32_MAX - tensor_bytes) {
        return false;
    }
    const size_t tensors_end = tensors_offset + tensor_bytes;
    const size_t scalar_bytes = static_cast<size_t>(args.scalar_count()) * sizeof(uint64_t);
    const size_t scalars_offset = args.scalar_count() == 0 ? 0 : PTO2_ALIGN_UP(tensors_end, alignof(uint64_t));
    const size_t total_bytes = args.scalar_count() == 0 ? tensors_end : scalars_offset + scalar_bytes;
    if ((args.scalar_count() != 0 && (scalars_offset > UINT32_MAX || scalars_offset > UINT32_MAX - scalar_bytes)) ||
        total_bytes > UINT32_MAX) {
        return false;
    }
    submission_image->assign(total_bytes, std::byte{0});
    auto *tensors = reinterpret_cast<GraphTensor *>(submission_image->data() + tensors_offset);
    for (int32_t i = 0; i < args.tensor_count(); ++i)
        tensors[i] = graph_tensor_pack(args.tensor(i).ref());
    if (args.scalar_count() != 0) {
        std::memcpy(
            submission_image->data() + scalars_offset, args.scalar_data(),
            static_cast<size_t>(args.scalar_count()) * sizeof(uint64_t)
        );
    }

    GraphSubmission submission{};
    submission.graph_key = full_key;
    submission.total_bytes = static_cast<uint32_t>(submission_image->size());
    submission.tensors_offset = static_cast<uint32_t>(tensors_offset);
    submission.tensor_count = static_cast<uint32_t>(args.tensor_count());
    submission.scalars_offset = static_cast<uint32_t>(scalars_offset);
    submission.scalar_count = static_cast<uint32_t>(args.scalar_count());
    std::memcpy(submission_image->data(), &submission, sizeof(submission));
    return true;
}

bool graph_submit_outer(
    PTO2OrchestratorState *orch, GraphHostState *state, uint64_t full_key, uint64_t definition_hash, int32_t owned_heap,
    bool defer_heap, const GraphTaskArgs &args, PTO2TaskId *submitted_id
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit Graph outside a scope");
    auto &allocator = orch->task_allocator;
    if (allocator.active_count() >= allocator.task_capacity() ||
        (!defer_heap && static_cast<uint64_t>(owned_heap) > allocator.heap_available())) {
        LOG_WARN("%s", "[GraphExecution] task-capacity/heap preflight failed; using ordinary path");
        return false;
    }

    GraphPendingUpload pending;
    if (!graph_prepare_submission_image(full_key, args, &pending.image)) return false;
    reinterpret_cast<GraphSubmission *>(pending.image.data())->definition_hash = definition_hash;
    pending.deferred_heap = defer_heap;

    DepInputs boundary_inputs{
        args.tensor_count(), args.tensor_data(), args.tag_data(), 0, nullptr,
    };
    const int32_t tensormap_needed = count_registrable_outputs(boundary_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) return false;
    const PTO2TaskAllocResult allocation = allocator.alloc(defer_heap ? 0 : owned_heap);
    if (allocation.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }
    const PTO2TaskId task_id = PTO2TaskId::make(0, static_cast<uint32_t>(allocation.task_id));
    PTO2SharedMemoryRingHeader &ring = orch->sm_header->ring;
    PTO2TaskDescriptor &task = ring.task_descriptors[allocation.slot];
    PTO2TaskPayload &payload = ring.task_payloads[allocation.slot];
    PTO2TaskSlotState &slot = ring.get_slot_state_by_slot(allocation.slot);

    // Init-on-write, as in prepare_task: this slot's dynamic scheduling fields and
    // completion flag are established here, at the claim, because nothing else
    // writes them. A stale wake_list_head of WAKE_LIST_SENTINEL would close the
    // list against every consumer, and a stale completion flag would report the
    // Graph done before it ran.
    slot.reset_for_reuse();
    ring.completion_flags[allocation.slot].store(0, std::memory_order_relaxed);

    slot.bind_buffers(&payload, &task);
    slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    slot.last_consumer_local_id = static_cast<int32_t>(task_id.local());
    slot.active_mask = ActiveMask{};
    slot.task_attrs = TaskAttrs{};
    slot.total_required_subtasks = 0;
    slot.logical_block_num = 1;
    slot.task_kind = TaskKind::GRAPH;
    scope_tasks_push(orch, &slot);

    task.task_id = task_id;
    std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
    task.packed_buffer_base = allocation.packed_base;
    task.packed_buffer_end = allocation.packed_end;
    graph_reset_outer_payload(payload);

    PTO2FaninBuilder fanin_builder(orch, &payload, static_cast<int32_t>(task_id.local()), next_fanin_seen_epoch(orch));
    auto emit = [&](PTO2TaskId producer_id) -> bool {
        return append_fanin_or_fail(orch, producer_id, &fanin_builder);
    };
    // An outer GRAPH task is a ring task like any other, so the dependency graph
    // has to carry it: without this the whole Graph — and every edge into it —
    // is absent from deps.json, leaving a run of 40 replays described by only its
    // handful of non-Graph tasks. It dispatches no kernel of its own and the
    // sub-DAG it replays owns no ring slots, so what is captured is its boundary:
    // the args it consumes and the edges those produce.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, PTO2_SUBTASK_SLOT_COUNT> kernel_ids_capture{
            INVALID_KERNEL_ID,
            INVALID_KERNEL_ID,
            INVALID_KERNEL_ID,
        };
        dep_gen_host_graph_begin_task(
            task_id.raw, orch->in_manual_scope(), /*early_dispatch=*/false, kernel_ids_capture.data(),
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
    payload.fanin_count = fanin_builder.count;

    pending.outer_slot = &slot;
    state->pending_uploads.push_back(std::move(pending));
    if (submitted_id != nullptr) *submitted_id = task_id;
#if SIMPLER_DFX
    orch->tasks_submitted++;
#endif
    return true;
}

bool graph_submit_definition(
    PTO2OrchestratorState *orch, GraphHostState *state, const std::vector<std::byte> &definition_image,
    const GraphTaskArgs &args, PTO2TaskId *submitted_id
) {
    const GraphDefinition *definition = graph_definition(definition_image);
    if (definition == nullptr || !graph_boundary_matches(*definition, args) ||
        definition->execution_storage_bytes == 0 ||
        definition->required_heap > UINT64_MAX - definition->execution_storage_bytes) {
        return false;
    }
    const uint64_t owned_heap = definition->required_heap + definition->execution_storage_bytes;
    if (owned_heap > static_cast<uint64_t>(INT32_MAX)) return false;
    return graph_submit_outer(
        orch, state, definition->full_key, definition->content_hash, static_cast<int32_t>(owned_heap), false, args,
        submitted_id
    );
}

bool graph_submit_pending_definition(
    PTO2OrchestratorState *orch, GraphHostState *state, uint64_t full_key, const GraphTaskArgs &args,
    PTO2TaskId *submitted_id
) {
    return graph_submit_outer(orch, state, full_key, 0, 0, true, args, submitted_id);
}

bool graph_finalize_pending_submissions(PTO2OrchestratorState *orch, GraphHostState *state, uint64_t *failed_key) {
    for (GraphPendingUpload &pending : state->pending_uploads) {
        if (!pending.deferred_heap) continue;
        if (pending.image.size() < sizeof(GraphSubmission)) return false;
        auto *submission = reinterpret_cast<GraphSubmission *>(pending.image.data());
        auto definition_it = state->definitions.find(submission->graph_key);
        const GraphDefinition *definition =
            definition_it == state->definitions.end() ? nullptr : graph_definition(definition_it->second);
        if (definition == nullptr || definition->execution_storage_bytes == 0 ||
            definition->required_heap > UINT64_MAX - definition->execution_storage_bytes ||
            pending.outer_slot == nullptr || pending.outer_slot->task == nullptr ||
            pending.outer_slot->task_kind != TaskKind::GRAPH ||
            !graph_submission_wire_size_valid(*submission, pending.image.size())) {
            if (failed_key != nullptr) *failed_key = submission->graph_key;
            return false;
        }
        const uint64_t owned_heap = definition->required_heap + definition->execution_storage_bytes;
        if (owned_heap > static_cast<uint64_t>(INT32_MAX)) {
            if (failed_key != nullptr) *failed_key = submission->graph_key;
            return false;
        }
        void *packed_base = nullptr;
        void *packed_end = nullptr;
        if (!orch->task_allocator.reserve_deferred_heap(static_cast<int32_t>(owned_heap), &packed_base, &packed_end)) {
            if (failed_key != nullptr) *failed_key = submission->graph_key;
            return false;
        }
        pending.outer_slot->task->packed_buffer_base = packed_base;
        pending.outer_slot->task->packed_buffer_end = packed_end;
        submission->definition_hash = definition->content_hash;
        pending.deferred_heap = false;
    }
    return true;
}

// Record one internal Graph node during the recording pass without consuming a
// shared task slot. Builds the node's metadata and materialized outputs
// exactly as submit_task_common would, but assigns output buffers from the
// bit-63 virtual address range and derives internal fanins from tensor-source
// classification — so no shared task slot, tensormap entry, fanin-pool entry, or upload
// is produced for the node. The resulting Definition is later attached to the
// outer GRAPH shells already submitted by the main thread. The returned
// TaskOutputTensors borrow the node's own tensor storage; moving the node into
// recording.nodes keeps those addresses valid because the inner buffer is
// transferred, not copied.
TaskOutputTensors graph_record_submit_node(
    PTO2OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    ORCH_PHASE_START();
    TaskOutputTensors result;
    GraphRecording &recording = *active_graph_recording(orch);

    const size_t node_index = recording.nodes.size();
    // A recorded node's index equals its local task id minus the recording
    // baseline, so its synthetic id keeps classification and explicit-dep
    // arithmetic identical to the ordinary path.
    const PTO2TaskId task_id =
        PTO2TaskId::make(0, static_cast<uint32_t>(recording.start_local_task_id) + static_cast<uint32_t>(node_index));
    result.set_task_id(task_id);

    if (node_index >= GRAPH_MAX_NODES || args.has_error) {
        recording.unsupported = true;
    }

    const PTO2OutputLayout layout = calculate_output_layout(args);
    const uint64_t aligned_output =
        layout.total_output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(layout.total_output_size), PTO2_ALIGN_SIZE) :
                                       0;
    if (recording.next_virtual_offset > GRAPH_RECORD_VIRTUAL_BASE - aligned_output) {
        recording.unsupported = true;
        return result;
    }
    const uintptr_t packed_base_addr = GRAPH_RECORD_VIRTUAL_BASE + recording.next_virtual_offset;
    recording.next_virtual_offset += aligned_output;

    GraphRecordedNode node;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    node.kernel_ids[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    node.active_mask = active_mask;
    node.task_attrs = task_attrs;
    node.task_attrs.set_early_resolve(false);
    node.logical_block_num = args.launch_spec.block_num();
    // Mirror prepare_task's contract: block_num must be positive and the subtask
    // count must fit int16_t. An out-of-contract value marks asynchronous
    // recording unsupported and makes commit fail-fast, rather than baking a
    // truncated or negative count into the cached Definition (which the device
    // would expand into a node that never completes).
    const int32_t required_subtasks =
        static_cast<int32_t>(node.logical_block_num) * __builtin_popcount(active_mask.core_mask());
    if (node.logical_block_num <= 0 || required_subtasks > std::numeric_limits<int16_t>::max()) {
        recording.unsupported = true;
        node.total_required_subtasks = 0;
    } else {
        node.total_required_subtasks = static_cast<int16_t>(required_subtasks);
    }
    node.record_packed_base = packed_base_addr;
    node.total_output_size = aligned_output;

    // Build the tensor list exactly as PTO2TaskPayload::init: inputs/inouts copy
    // the caller's ChipTensor; outputs materialize from the create-info onto the
    // scratch buffer and carry this node's owner id.
    const int32_t tensor_count = args.tensor_count();
    node.tensors.resize(static_cast<size_t>(tensor_count));
    for (int32_t i = 0; i < tensor_count; ++i) {
        ChipTensor &slot_tensor = node.tensors[static_cast<size_t>(i)];
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
    // Materialize output refs only after node.tensors is fully sized so the
    // borrowed addresses stay stable across the move into recording.nodes.
    for (int32_t i = 0; i < tensor_count; ++i) {
        if (args.tag(i) == TensorArgType::OUTPUT) result.materialize_output(node.tensors[static_cast<size_t>(i)]);
    }
    node.scalar_offset = static_cast<uint32_t>(recording.scalars.size());
    node.scalar_count = static_cast<uint32_t>(args.scalar_count());
    recording.scalars.insert(recording.scalars.end(), args.scalars(), args.scalars() + args.scalar_count());
#if SIMPLER_DFX
    node.dump_metadata.dump_arg_mask = args.dump_arg_mask();
    node.dump_metadata.dump_arg_flags = args.dump_arg_index_ambiguous_mask();
    memcpy(node.dump_metadata.scalar_dtypes, args.scalar_dtypes(), args.scalar_count() * sizeof(uint8_t));
#endif

    // Classify each scalar's source: a plain literal is static Definition data,
    // while a value copied from a boundary scalar is refreshed on replay. A
    // mutable tracked boundary scalar is not supported and falls back.
    recording.scalar_sources.resize(static_cast<size_t>(node.scalar_offset) + node.scalar_count);
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        GraphRecordedScalarSourceRef source = graph_classify_scalar(recording, args, i);
        if (source.source == GraphRecordedScalarSource::INVALIDATED_BOUNDARY) recording.unsupported = true;
        recording.scalar_sources[static_cast<size_t>(node.scalar_offset) + static_cast<size_t>(i)] = source;
    }

    // Classify each tensor's source, then derive internal fanins from the
    // INTERNAL classifications plus any explicit internal dependency.
    node.tensor_source_offset = static_cast<uint32_t>(recording.tensor_sources.size());
    recording.tensor_sources.resize(static_cast<size_t>(node.tensor_source_offset) + tensor_count);
    for (int32_t i = 0; i < tensor_count; ++i) {
        // The out-pointer is used only for the duration of the call, so pointing
        // it into the flat array is safe even though a later node grows that array.
        if (!graph_classify_tensor(
                recording, node, static_cast<int32_t>(node_index), node.tensors[static_cast<size_t>(i)],
                &recording.tensor_sources[static_cast<size_t>(node.tensor_source_offset) + static_cast<size_t>(i)]
            )) {
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
    // Gated on the recorded attribute, not on args: a kernel-less node never
    // dispatches, so submit_dummy_task and alloc_tensors drop the predicate the
    // caller set. Reading args here instead would record a predicate the node's
    // own attribute denies, and materialize rejects a Definition whose two halves
    // disagree.
    if (node.task_attrs.has_predicate()) {
        const CoreTaskPredicate &pred = args.predicate();
        GraphRecordedPredicate recorded;
        recorded.op = pred.op;
        recorded.target = pred.target;
        const ChipTensor *operand = pred.operand.tensor;
        // OWN_OUTPUT would read the node's own output before the node runs, so it
        // names no value the predicate could be evaluating. An index vector that
        // leaves the operand's extent is caught here too: materialize would
        // otherwise reject the baked offset on the device, where the failure is a
        // Scheduler fatal rather than a named unsupported construct.
        const uint64_t flat_offset =
            operand == nullptr ? 0 : operand->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
        if (operand == nullptr || operand->ndims > MAX_TENSOR_DIMS || pred.operand.ndims > operand->ndims ||
            flat_offset < operand->start_offset || flat_offset - operand->start_offset >= operand->extent_elem_cache ||
            !graph_classify_tensor(recording, node, static_cast<int32_t>(node_index), *operand, &recorded.source) ||
            recorded.source.source == GraphRecordedTensorSource::OWN_OUTPUT) {
            recording.unsupported = true;
        } else {
            recorded.operand.copy(*operand);
            recorded.elem_offset = flat_offset - operand->start_offset;
            recorded.elem_size = static_cast<uint8_t>(get_element_size(operand->dtype));
        }
        node.predicate_index = static_cast<int32_t>(recording.predicates.size());
        recording.predicates.push_back(recorded);
    }

    node.fanin_offset = static_cast<uint32_t>(recording.internal_fanins.size());
    // Dedup within this node's own range: the flat array's earlier entries belong
    // to earlier nodes.
    auto add_fanin = [&recording, &node](size_t producer) {
        const auto begin = recording.internal_fanins.begin() + node.fanin_offset;
        if (std::find(begin, recording.internal_fanins.end(), producer) == recording.internal_fanins.end()) {
            recording.internal_fanins.push_back(producer);
        }
    };
    for (uint32_t i = 0; i < static_cast<uint32_t>(tensor_count); ++i) {
        const GraphRecordedTensorSourceRef &source = recording.tensor_sources[node.tensor_source_offset + i];
        if (source.source == GraphRecordedTensorSource::INTERNAL) add_fanin(source.source_index);
    }

    // Inferred hazards, on the same terms as the ring path. The loop above only
    // names the node that ALLOCATED each buffer; every write-then-read through a
    // buffer someone else allocated — an alloc_tensors output written in place
    // with add_inout, or a view of a boundary tensor — needs the last-writer
    // lookup compute_task_fanin performs. Running the very same function against
    // the recording's own map is what keeps a Definition's edge set equal to the
    // one the body gets when its tasks are submitted individually.
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
        if (!recording.tensor_map_ready) {
            recording.unsupported = true;
        } else if (recording.tensor_map.free_entries() < count_registrable_outputs(dep_inputs, manual_scope)) {
            // Recording one more node would assert inside new_entry(). Abandon the
            // Definition instead, so the run fails by name at graph_commit rather
            // than on a hard assert here.
            LOG_WARN(
                "[GraphExecution] recording hazard map exhausted at node %zu (%d entries); Graph abandoned", node_index,
                GRAPH_RECORD_TENSORMAP_POOL_SIZE
            );
            recording.unsupported = true;
        } else {
            auto emit_inferred = [&recording, &add_fanin, node_index](PTO2TaskId producer) -> bool {
                const int32_t producer_index = static_cast<int32_t>(producer.local()) - recording.start_local_task_id;
                if (producer.ring() == 0 && producer_index >= 0 && producer_index < static_cast<int32_t>(node_index)) {
                    add_fanin(static_cast<size_t>(producer_index));
                }
                return true;
            };
            (void)compute_task_fanin(dep_inputs, recording.tensor_map, manual_scope, emit_inferred);
            register_task_outputs(dep_inputs, task_id, recording.tensor_map, manual_scope);
        }
    }
    for (uint32_t i = 0; i < args.explicit_dep_count(); ++i) {
        const PTO2TaskId dep = args.explicit_dep(i);
        const int32_t dep_index = static_cast<int32_t>(dep.local()) - recording.start_local_task_id;
        if (!dep.is_valid() || dep.ring() != 0 || dep_index >= static_cast<int32_t>(node_index)) {
            recording.unsupported = true;
            continue;
        }
        if (dep_index < 0) {
            const bool represented_by_boundary = std::any_of(
                recording.boundary_tensors.begin(), recording.boundary_tensors.end(), [dep](const ChipTensor &tensor) {
                    return tensor.owner_task_id == dep;
                }
            );
            if (!represented_by_boundary) recording.unsupported = true;
        } else {
            add_fanin(static_cast<size_t>(dep_index));
        }
    }

    node.fanin_count = static_cast<uint32_t>(recording.internal_fanins.size() - node.fanin_offset);
    if (node.record_packed_base != 0 && node.total_output_size != 0 &&
        node.total_output_size <= UINTPTR_MAX - node.record_packed_base) {
        const uintptr_t begin = node.record_packed_base;
        const uintptr_t end = begin + node.total_output_size;
        // The sorted-and-disjoint property the lookup depends on, checked rather
        // than assumed: a mid-recording heap rollback would break it, and today
        // the only rollback is graph_end's, after every node is recorded.
        always_assert(recording.output_ranges.empty() || recording.output_ranges.back().end <= begin);
        recording.output_ranges.push_back({begin, end, static_cast<uint32_t>(node_index)});
    }
    recording.nodes.push_back(std::move(node));
    ORCH_PHASE_END(HostOrchPhase::RecordNode, task_id.raw);
    return result;
}

}  // namespace

GraphScopeResult
PTO2OrchestratorState::graph_begin(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash) {
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
        PTO2TaskId submitted = PTO2TaskId::invalid();
        ORCH_PHASE_START();
        if (graph_submit_definition(orch, state, definition_it->second, args, &submitted)) {
            result.execute_block = false;
            result.task_id = submitted;
            ORCH_PHASE_END(HostOrchPhase::GraphSubmit, submitted.raw);
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
    // with no recording object means the pass failed and this key is spent.
    auto inflight_it = state->inflight.find(full_key);
    if (inflight_it != state->inflight.end()) {
        GraphInflightRecording &entry = *inflight_it->second;
        if (entry.status() != GraphRecordingStatus::RECORDING || entry.recording == nullptr ||
            !graph_recording_boundary_matches(*entry.recording, args)) {
            return result;
        }
        PTO2TaskId submitted = PTO2TaskId::invalid();
        ORCH_PHASE_START();
        if (graph_submit_pending_definition(orch, state, full_key, args, &submitted)) {
            result.execute_block = false;
            result.task_id = submitted;
            ORCH_PHASE_END(HostOrchPhase::GraphSubmit, submitted.raw);
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

    auto recording = std::make_unique<GraphRecording>();
    recording->full_key = full_key;
    recording->start_local_task_id = orch->task_allocator.active_count();
    if (!graph_recording_init_tensor_map(*recording)) {
        LOG_WARN("[GraphExecution] recording hazard map allocation failed; using ordinary path");
        return result;
    }
    recording->boundary_scalar_count = args.scalar_count();
    recording->boundary_tensors.reserve(static_cast<size_t>(args.tensor_count()));
    recording->boundary_types.reserve(static_cast<size_t>(args.tensor_count()));
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        recording->boundary_tensors.push_back(args.tensor(i).ref());
        recording->boundary_types.push_back(args.tag(i));
    }
    auto entry = std::make_unique<GraphInflightRecording>();
    entry->full_key = full_key;
    entry->recording = std::move(recording);
    GraphInflightRecording *entry_ptr = entry.get();
    state->inflight.emplace(full_key, std::move(entry));
    state->inflight_count.store(state->inflight.size(), std::memory_order_release);

    PTO2TaskId submitted = PTO2TaskId::invalid();
    ORCH_PHASE_START();
    if (graph_submit_pending_definition(orch, state, full_key, args, &submitted)) {
        result.execute_block = false;
        result.recording = true;
        result.recording_handle = entry_ptr;
        result.task_id = submitted;
        ORCH_PHASE_END(HostOrchPhase::GraphSubmit, submitted.raw);
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

bool PTO2OrchestratorState::graph_prepare(void *recording_handle, const GraphTaskArgs &args) {
    GraphHostState *state = graph_state_from(this);
    if (state == nullptr || recording_handle == nullptr || g_active_graph_recording != nullptr) return false;
    auto *entry = static_cast<GraphInflightRecording *>(recording_handle);
    // graph_begin published this entry before the private job was enqueued, and
    // the entry's address is stable for as long as the recording lives, so the
    // recording thread reaches its own state without searching for it. Until this
    // thread calls graph_end/graph_abort, later graph_begin calls only read the
    // boundary vectors under recording_mutex, and only this thread writes the
    // fields it binds below. Taking that mutex here lets the main thread's
    // same-key submit burst starve prepare and collapse the intended overlap, so
    // the status read goes through the atomic instead.
    if (entry->status() != GraphRecordingStatus::RECORDING || entry->recording == nullptr) {
        return false;
    }
    // The recording was created from this very boundary at graph_begin, and the
    // handle names that recording rather than being searched for, so a mismatch
    // here is unreachable. The comparison walks up to 128 ChipTensor descriptors on
    // the thread whose start-up latency this path exists to keep short, so it is an
    // assertion: debug builds still catch a boundary that stopped matching, release
    // builds compile it out.
    debug_assert(
        graph_recording_boundary_matches(*entry->recording, args) &&
        "the recording thread's boundary copy must match the boundary graph_begin recorded"
    );
    args.anchor_scalar_sources();
    entry->recording->boundary_args = &args;
    g_active_graph_entry = entry;
    g_active_graph_recording = entry->recording.get();
    g_active_graph_owner = state;
    return true;
}

void PTO2OrchestratorState::graph_abort(void *recording_handle) {
    GraphHostState *state = graph_state_from(this);
    auto *entry = static_cast<GraphInflightRecording *>(recording_handle);
    if (state == nullptr || entry == nullptr) return;
    {
        std::lock_guard<std::mutex> lock(state->recording_mutex);
        entry->recording.reset();
        entry->set_status(GraphRecordingStatus::FAILED);
    }
    g_active_graph_entry = nullptr;
    g_active_graph_recording = nullptr;
    g_active_graph_owner = nullptr;
    state->recording_cv.notify_all();
}

// Finish the background recording pass and publish the Definition. The main
// thread finalizes the already-submitted outer Graph tasks in graph_commit.
bool PTO2OrchestratorState::graph_end() {
    GraphHostState *state = graph_state_from(this);
    GraphRecording *recording = active_graph_recording(this);
    GraphInflightRecording *entry = g_active_graph_entry;
    if (state == nullptr || recording == nullptr || entry == nullptr) return false;

    std::vector<std::byte> definition;
    ORCH_PHASE_START();
    const bool built = graph_build_definition(*recording, &definition);
    if (built) {
        ORCH_PHASE_END(HostOrchPhase::BuildDefinition, recording->nodes.size());
    }
    const GraphDefinition *header = built ? graph_definition(definition) : nullptr;
    if (header == nullptr) {
        debug_assert(false && "The recorded Graph contains a construct that Graph Execution does not support");
        LOG_WARN("%s", "[GraphExecution] asynchronous recording produced an unsupported Graph");
        graph_abort(entry);
        return false;
    }
    LOG_DEBUG(
        "[GraphExecution] define key=0x%llx nodes=%u bytes=%u", static_cast<unsigned long long>(header->full_key),
        header->task_count, header->total_bytes
    );
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(state->recording_mutex);
        if (entry->status() != GraphRecordingStatus::RECORDING || entry->full_key != header->full_key) {
            entry->set_status(GraphRecordingStatus::FAILED);
        } else {
            state->definitions.emplace(header->full_key, std::move(definition));
            entry->set_status(GraphRecordingStatus::READY);
        }
        ready = entry->status() == GraphRecordingStatus::READY;
        entry->recording.reset();
    }
    g_active_graph_entry = nullptr;
    g_active_graph_recording = nullptr;
    g_active_graph_owner = nullptr;
    state->recording_cv.notify_all();
    return ready;
}

// Join every recording in flight and back-patch all deferred shells in submit
// order. Orchestration completion is the only normal-path barrier.
void PTO2OrchestratorState::graph_commit() {
    if (active_graph_recording(this) != nullptr) return;
    GraphHostState *state = graph_state_from(this);
    if (state == nullptr || state->inflight_count.load(std::memory_order_acquire) == 0) return;

    std::unordered_map<uint64_t, std::unique_ptr<GraphInflightRecording>> drained;
    {
        std::unique_lock<std::mutex> lock(state->recording_mutex);
        if (state->inflight.empty()) return;
        state->recording_cv.wait(lock, [&]() {
            return !state->any_recording();
        });
        drained.swap(state->inflight);
        state->inflight_count.store(0, std::memory_order_release);
    }

    uint64_t failed_key = 0;
    bool failed = false;
    for (const auto &[key, entry] : drained) {
        auto definition_it = state->definitions.find(key);
        if (entry->status() == GraphRecordingStatus::READY && definition_it != state->definitions.end() &&
            graph_definition(definition_it->second) != nullptr) {
            continue;
        }
        if (!failed) failed_key = key;
        failed = true;
    }
    if (!failed && !graph_finalize_pending_submissions(this, state, &failed_key)) failed = true;
    if (failed) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "failed to finalize asynchronous Graph key=%#llx",
            static_cast<unsigned long long>(failed_key)
        );
    }
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    if (!static_cast<bool>(active_mask)) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__,
            "MixedKernels names no active slot; set at least one of aic/aiv0/aiv1 kernel_id"
        );
        return TaskOutputTensors{};
    }

    int16_t block_num = args.launch_spec.block_num();

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    // sync_start is only meaningful for tasks with block_num > 1.
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        task_attrs.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        task_attrs.set_predicate();
    }

    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_node(
            orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
            normalized.aiv1_kernel_id
        );
    }

    return submit_task_common(
        orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
        normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const CoreTaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_node(
            orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const CoreTaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();
    ORCH_PHASE_START();

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    // A Graph body may allocate. The allocation records as a kernel-less node —
    // the same shape submit_dummy_task records — and replay reserves the
    // intermediate heap for every internal node anyway, so the outputs land at
    // addresses the replayed Definition derives for itself.
    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_node(
            orch, args, ActiveMask{}, TaskAttrs{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time attributes. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;  // hidden-alloc tasks have no producer dependencies
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        //
        // Flag the creator so it does NOT suppress its consumers' early-dispatch.
        // Under the direct-only model an unflagged producer disqualifies its
        // consumer, and a pre-completed producer only seeds dispatch_fanin when
        // flagged. A buffer allocation is pure memory whose output is ready at
        // creation — it should always be transparent, never a barrier. Unlike a
        // codegen task there is no Arg-driven hint to honor here, so mark it
        // unconditionally.
        prepared.slot_state->task_attrs.set_early_resolve(true);
        prepared.slot_state->mark_completed();  // host-visible task_state mirror
        // Polling: pre-set the device-visible completion_flags byte in the H2D
        // image. Consumers poll completion_flags (not task_state), so a hidden-alloc
        // producer completed here on the host must publish its flag too — otherwise
        // every consumer register_wakes on a producer that never runs on device and
        // the run hangs. (The device watermark walk transparently steps past this
        // pre-set flag when a later on-device task completes.)
        PTO2SharedMemoryRingHeader &done_ring = orch->sm_header->ring;
        int32_t done_local = static_cast<int32_t>(prepared.task_id.local());
        done_ring.set_completion_flag(done_local);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    ORCH_PHASE_END(HostOrchPhase::Prepare, prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    int32_t total_tasks = orch->task_allocator.active_count();
    if (total_tasks > 0) {
        LOG_DEBUG("=== [Orchestrator] total_tasks=%d ===", total_tasks);
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
PTO2OrchProfilingData orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_args_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
