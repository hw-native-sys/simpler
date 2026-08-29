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

#include "host_build_graph/orchestrator.h"

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

#include "common/host_phase_kind.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/dep_compute.h"
#include "graph_execution.h"
#include "graph_host_state.h"
#include "host_build_graph/task_id_encoding.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"
#include "tensor.h"

#if SIMPLER_DFX
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
    void creator(int32_t arg_idx, const simpler::hbg::Tensor &consumer, TaskId producer) const {
        dep_gen_host_graph_add_creator_edge(producer.raw, arg_idx, consumer);
    }
    void tensormap(
        int32_t arg_idx, const simpler::hbg::Tensor &consumer, const ChipTensorMapEntry &entry, OverlapStatus overlap
    ) const {
        dep_gen_host_graph_add_tensormap_edge(entry.producer_task_id.raw, arg_idx, consumer, entry, overlap);
    }
};
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_explicit_edge(uint64_t) {}
__attribute__((weak, visibility("hidden"))) void
dep_gen_host_graph_add_creator_edge(uint64_t, int32_t, const simpler::hbg::Tensor &) {}
__attribute__((weak, visibility("hidden"))) void dep_gen_host_graph_add_tensormap_edge(
    uint64_t, int32_t, const simpler::hbg::Tensor &, const ChipTensorMapEntry &, OverlapStatus
) {}

// AICore register accessor (aicpu/platform_regs.h). The host orchestrator's
// route_ready_once path transitively ODR-uses the early-dispatch doorbell inline
// (scheduler.h ring_one_doorbell), but no core is gated during host
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
    // detailed rationale on the same fallback in runtime_core.cpp).
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
static uint64_t g_orch_alloc_cycle = 0;   // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;    // param copy
static uint64_t g_orch_lookup_cycle = 0;  // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;  // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;   // fanin list + early-return check
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_args_atomic_count = 0;
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
    // detailed rationale on the same fallback in runtime_core.cpp).
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Scale sec and nsec separately (divisor is the constant 1e9): avoids a
    // div-by-zero when PLATFORM_PROF_SYS_CNT_FREQ >= 1 GHz and the truncation
    // error a `1e9 / FREQ` divisor would introduce for non-dividing frequencies.
    return static_cast<uint64_t>(ts.tv_sec) * PLATFORM_PROF_SYS_CNT_FREQ +
           static_cast<uint64_t>(ts.tv_nsec) * PLATFORM_PROF_SYS_CNT_FREQ / 1000000000ull;
}
// submit_idx tags a record with its position in the orchestration's submit order.
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
// Only the host orchestrator reaches these sites, so this file names only the Orch*
// half of HostPhaseKind; the bind kinds never appear here.
#define ORCH_PHASE_START() const uint64_t _orch_phase_t0 = host_phase_now_ns()
#define ORCH_PHASE_END(phase, detail)                                                                         \
    do {                                                                                                      \
        host_phase_record(                                                                                    \
            _orch_phase_t0, host_phase_now_ns(), static_cast<uint32_t>(phase), static_cast<uint64_t>(detail), \
            g_orch_submit_idx                                                                                 \
        );                                                                                                    \
    } while (0)
// For a phase that spans a submission rather than sitting inside one: the index
// advances during the span, so the group is taken at the start or the record
// files itself under the next submission.
#define ORCH_PHASE_START_SPANNING() \
    ORCH_PHASE_START();             \
    const uint32_t _orch_phase_group = g_orch_submit_idx
#define ORCH_PHASE_END_SPANNING(phase, detail)                                                                \
    do {                                                                                                      \
        host_phase_record(                                                                                    \
            _orch_phase_t0, host_phase_now_ns(), static_cast<uint32_t>(phase), static_cast<uint64_t>(detail), \
            _orch_phase_group                                                                                 \
        );                                                                                                    \
    } while (0)
#else
#define ORCH_PHASE_START()
#define ORCH_PHASE_END(phase, detail) \
    do {                              \
    } while (0)
#define ORCH_PHASE_START_SPANNING()
#define ORCH_PHASE_END_SPANNING(phase, detail) \
    do {                                       \
    } while (0)
#endif

static int32_t orch_mark_fatal(OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == SIMPLER_ERROR_NONE || orch->sm_header == nullptr) {
        return SIMPLER_ERROR_NONE;
    }

    int32_t expected = SIMPLER_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != SIMPLER_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    std::array<char, 1024> message{};
    vsnprintf(message.data(), message.size(), fmt, args);
    if (latched_code != SIMPLER_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message.data());
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message.data());
}

void OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
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

// A recorded task's dispatch predicate, held as the operand tensor plus the element
// index within it rather than the absolute address submit would resolve. The tensor is
// copied because the caller only lends it for the duration of the submit call.
struct GraphRecordedPredicate {
    simpler::hbg::Tensor operand;
    GraphRecordedTensorSourceRef source;
    uint64_t elem_offset{0};
    int64_t target{0};
    uint8_t elem_size{0};
    PredicateOp op{PredicateOp::NONE};
};

struct RecordedInGraphTask {
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
    uint32_t tensor_offset{0};
    uint32_t tensor_count{0};
    // Ranges into the recording's flat arrays. tensor_sources has one entry per
    // tensor, so tensor_count is its count too.
    uint32_t tensor_source_offset{0};
    uint32_t scalar_offset{0};
    uint32_t scalar_count{0};
    uint32_t fanin_offset{0};
    uint32_t fanin_count{0};
    // Index into the recording's predicates, or -1 when the task carries none.
    int32_t predicate_index{-1};
    ArgsDumpTaskMetadata dump_metadata;

    // Restore the state a freshly recorded task has, in a slot the previous body left
    // behind. A reused slot that keeps any field of the previous body
    // records a Definition that body never had, and predicate_index and dump_metadata are
    // written only on the paths that have one, so neither can be left to the fill.
    //
    // Field by field rather than `*this = RecordedInGraphTask{}`: the latter is immune to
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
        tensor_source_offset = 0;
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
    sizeof(RecordedInGraphTask) == 104, "RecordedInGraphTask gained or lost a field: extend reset() to match, then "
                                        "update this size"
);

// One recorded task's scratch output window. reserve_heap_scratch is a pure bump
// and a task stores the aligned size it advanced by, so consecutive windows abut:
// held in record order these are sorted and disjoint, which is what lets an
// address lookup binary search instead of walking every producer. A task with no
// output advances nothing and owns no entry.
struct GraphRecordedOutputRange {
    uintptr_t begin;
    uintptr_t end;
    uint32_t task_index;
};

// The Graph boundary as the submitting thread captured it, deep-copied because the
// caller only lends its arguments for the duration of the submit call. It anchors
// boundary scalar sources while the main thread submits outer shells from later
// invocation arguments, and later same-key submissions compare against it under
// recording_mutex — so it belongs to the in-flight entry, not to the recorder's
// storage, which the submitting thread must never touch.
struct GraphBoundary {
    const GraphTaskArgs *args{nullptr};
    int32_t scalar_count{0};
    std::vector<simpler::hbg::Tensor> tensors;
    std::vector<TensorArgType> types;
};

// Words per row of GraphRecording::task_reach — one bit per task a body may hold.
inline constexpr size_t GRAPH_REACH_WORDS = (MAX_IN_GRAPH_TASKS + 63) / 64;

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
    std::vector<RecordedInGraphTask> tasks;
    // How many of `tasks` this recording has filled. The array itself is never cleared
    // and graph_recording_reserve_storage sizes it to the in-graph task cap, so a body is
    // recorded into slots that already exist: a recorded task makes no allocation at all.
    size_t task_count{0};
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
    uint32_t task_tensor_cursor{0};
    // Flat per-task arrays, indexed by the ranges on RecordedInGraphTask. Held here rather
    // than on each recorded task so recording a graph pays no allocation per task per array,
    // and reserved to the in-graph task cap by graph_recording_reserve_storage so it pays no
    // growth either.
    std::vector<GraphRecordedTensorSourceRef> tensor_sources;
    std::vector<uint64_t> scalars;
    std::vector<GraphRecordedScalarSourceRef> scalar_sources;
    std::vector<size_t> internal_fanins;
    std::vector<GraphRecordedOutputRange> output_ranges;
    // Exact ancestor closure per recorded task, GRAPH_REACH_WORDS words each, indexed
    // by task index within the body. Bit j of task i's row is set when task j reaches
    // task i through the body's internal edges.
    //
    // This is a full bitset rather than the one-word window the global submit path
    // carries, and the two limits are what make that affordable here: a body holds at
    // most MAX_IN_GRAPH_TASKS tasks, so a row is a fixed 128 B and the whole array
    // 128 KiB next to this recording's 4 MB tensor pool; and a Graph is recorded once
    // and then replayed, so the fold is paid once for every replay that reads the
    // reduced edge set. No window means no edge is kept merely because its producer
    // sat too far back.
    std::vector<uint64_t> task_reach;
    // Edges this body's reduction removed, reported against the shipped edge count
    // when the Definition is laid out. A Graph is recorded once, so this is where a
    // body's redundancy is read off a run.
    size_t reduced_edges{0};
    // Indexed by RecordedInGraphTask::predicate_index; only predicated tasks
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
    // in-graph task with no edge to its actual producer, and the Definition would replay
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

    simpler::hbg::Tensor *task_tensors(const RecordedInGraphTask &task) const {
        return task_tensor_pool.get() + task.tensor_offset;
    }

    const GraphTaskArgs *boundary_args() const { return boundary == nullptr ? nullptr : boundary->args; }
    int32_t boundary_scalar_count() const { return boundary == nullptr ? 0 : boundary->scalar_count; }
    const std::vector<simpler::hbg::Tensor> &boundary_tensors() const { return boundary->tensors; }
    const std::vector<TensorArgType> &boundary_types() const { return boundary->types; }
};

struct GraphPendingUpload {
    ChipTaskSlotState *outer_slot{nullptr};
    uint64_t full_key{0};
    bool deferred_heap{false};
};

enum class GraphRecordingStatus : uint8_t { RECORDING = 0, READY = 1, FAILED = 2 };

// One Definition being recorded. Entries are keyed by Graph key and held by
// unique_ptr, so a rehash of the owning map never moves one: the recording
// thread is handed this address at graph_begin and dereferences it without
// taking recording_mutex.
//
// The entry carries the boundary and the status, not the recorded body: the body's
// storage belongs to the recorder thread that will fill it (recorder_recording()), so
// nothing here is sized by the graph.
struct GraphInflightRecording {
    uint64_t full_key{0};
    GraphBoundary boundary;
    // Atomic because graph_prepare reads it on the recording thread without
    // taking recording_mutex, by design: acquiring the mutex there lets a
    // main-thread burst of same-key submissions starve the thread before it can
    // bind its private recording state. Every other access holds the mutex.
    std::atomic<GraphRecordingStatus> recording_status{GraphRecordingStatus::RECORDING};

    GraphRecordingStatus status() const { return recording_status.load(std::memory_order_acquire); }
    void set_status(GraphRecordingStatus next) { recording_status.store(next, std::memory_order_release); }
};

// One published Definition image. It lives in the run's arena at `object_offset`
// — an offset, not an address, so the arena can be reallocated between
// publication and upload — unless the arena had no room, in which case `spill`
// holds the image and `object_offset` is GRAPH_NO_OBJECT_OFFSET.
struct GraphDefinitionRecord {
    size_t object_offset{GRAPH_NO_OBJECT_OFFSET};
    size_t bytes{0};
    std::vector<std::byte> spill;
};

struct GraphHostState {
    explicit GraphHostState(const GraphDefinitionArena &arena) :
        arena(arena) {}

    std::unordered_map<uint64_t, GraphDefinitionRecord> definitions;
    // Recordings in flight, at most one per Graph key. Several record at once,
    // each on its own thread; graph_commit drains and finalizes all of them.
    std::unordered_map<uint64_t, std::unique_ptr<GraphInflightRecording>> inflight;
    std::vector<GraphPendingUpload> pending_uploads;
    std::mutex recording_mutex;
    std::condition_variable recording_cv;
    // Mirrors inflight.size() so orchestration completion answers the common
    // "nothing is recording" case without taking recording_mutex.
    std::atomic<size_t> inflight_count{0};
    // Fixed for the run: recorder threads hold addresses inside it, so it must
    // not move while any of them is filling an image.
    GraphDefinitionArena arena;
    std::atomic<size_t> arena_cursor{0};

    // Claim room for one object of `image_bytes`, padded so the next object
    // starts aligned too. Returns the object offset, or nullopt when the run has
    // outgrown the retained arena — the caller then builds into its own buffer.
    // Several recording threads reserve at once and a losing exchange retries
    // rather than advancing the cursor past the capacity, so an object that does
    // not fit costs the run its own slot and no one else's.
    std::optional<size_t> reserve_object(size_t image_bytes) {
        if (arena.base == nullptr || arena.object_align == 0) return std::nullopt;
        if (image_bytes > SIZE_MAX - arena.object_prefix_bytes) return std::nullopt;
        const size_t object_bytes = arena.object_prefix_bytes + image_bytes;
        if (object_bytes > SIZE_MAX - (arena.object_align - 1)) return std::nullopt;
        const size_t claimed = (object_bytes + arena.object_align - 1) & ~(arena.object_align - 1);
        size_t offset = arena_cursor.load(std::memory_order_relaxed);
        while (true) {
            if (claimed > arena.capacity - offset) return std::nullopt;
            if (arena_cursor.compare_exchange_weak(
                    offset, offset + claimed, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                return offset;
            }
        }
    }

    // Where an object's image starts, for a record the arena holds.
    std::byte *image_at(size_t object_offset) const { return arena.base + object_offset + arena.object_prefix_bytes; }

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

GraphRecording *active_graph_recording(OrchestratorState *orch) {
    GraphHostState *state = graph_state_from(orch);
    if (state == nullptr || state != g_active_graph_owner) return nullptr;
    return g_active_graph_recording;
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

bool graph_tensor_exact(const simpler::hbg::Tensor &lhs, const simpler::hbg::Tensor &rhs) {
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
    const GraphRecording &recording, const simpler::hbg::Tensor &tensor, GraphRecordedTensorSourceRef *source
) {
    for (size_t i = 0; i < recording.boundary_tensors().size(); ++i) {
        if (!graph_tensor_exact(tensor, recording.boundary_tensors()[i])) continue;
        source->source = GraphRecordedTensorSource::BOUNDARY_EXACT;
        source->source_index = i;
        source->packed_offset = 0;
        return true;
    }
    for (size_t i = 0; i < recording.boundary_tensors().size(); ++i) {
        const simpler::hbg::Tensor &boundary = recording.boundary_tensors()[i];
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
    if (recording.boundary_args() == nullptr) return {};
    // Identity, not type: an in-graph task's Arg and the boundary Arg have
    // different capacities, so compare the addresses through void.
    if (static_cast<const void *>(&args) == static_cast<const void *>(recording.boundary_args()) &&
        scalar_index < recording.boundary_args()->scalar_count()) {
        return GraphRecordedScalarSourceRef{GraphRecordedScalarSource::BOUNDARY, static_cast<size_t>(scalar_index)};
    }

    const void *source = args.scalar_source(scalar_index);
    const void *invalidated_source = args.invalidated_scalar_source(scalar_index);
    if (source == nullptr && invalidated_source == nullptr) return {};
    for (int32_t i = 0; i < recording.boundary_args()->scalar_count(); ++i) {
        const void *boundary_source = static_cast<const void *>(&recording.boundary_args()->scalar(i));
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
// MAX_IN_GRAPH_TASKS tasks and each recorded task registers at most its INOUT/OUTPUT_EXISTING
// args, so this bounds the worst realistic body while staying a small fraction of
// the ordinary path's whole-orchestration pool (CHIP_TENSORMAP_POOL_SIZE). Exhausting
// it marks the recording unsupported, which graph_commit reports as
// SIMPLER_ERROR_INVALID_ARGS -- the outer shell is already submitted by then, so there
// is no ordinary-path fallback left to take.
constexpr int32_t GRAPH_RECORD_TENSORMAP_POOL_SIZE = 16384;

// Elements in the recording's tensor pool: every in-graph task a body can hold, times
// every tensor argument one such task can carry. An in-cap body therefore always fits, and
// the bump cursor is checked anyway because a body that overshoots MAX_IN_GRAPH_TASKS keeps
// recording so it can finish.
constexpr size_t GRAPH_RECORD_TENSOR_POOL_ELEMS =
    static_cast<size_t>(MAX_IN_GRAPH_TASKS) * static_cast<size_t>(CORE_MAX_TENSOR_ARGS);

// The graph_local_id a recorded task's IN_GRAPH id carries. A recorded task belongs
// to no Graph task yet -- every shell replaying the Definition re-mints the id with
// its own local id at materialize -- so record time names a task by its index alone,
// and the id's low field is that index and nothing else. That is what keeps the
// index inside the MAX_IN_GRAPH_TASKS task chains the recording's hazard map is
// dimensioned for.
constexpr uint32_t GRAPH_RECORD_NO_OWNING_GRAPH = 0;

// Stand the recording's hazard map up on its own allocation. Failure is
// reported to the caller, which abandons the recording rather than producing a
// Definition with inferred edges missing.
bool graph_recording_init_tensor_map(GraphRecording &recording) {
    return recording.tensor_map.init(CHIP_TENSORMAP_NUM_BUCKETS, GRAPH_RECORD_TENSORMAP_POOL_SIZE, MAX_IN_GRAPH_TASKS);
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

// Drop this thread's recording's view of an in-flight entry's boundary. Called where a
// recording ends, because the entry does not outlive graph_commit while the storage does.
void unbind_recorder_boundary() {
    if (g_active_graph_recording != nullptr) g_active_graph_recording->boundary = nullptr;
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
// Each bound is a per-task cap times the in-graph task cap, so these are the recorded
// body's own limits rather than a worst case invented here: the tensor pool and
// tensor_sources are one entry per tensor argument (CORE_MAX_TENSOR_ARGS), the two scalar
// arrays one per scalar argument (CORE_MAX_SCALAR_ARGS), and predicates and output_ranges at
// most one per task.
//
// internal_fanins is the one array left growing, and the reason is the size it grows to
// rather than the bound it could reach. It has no per-in-graph-task cap: CHIP_MAX_FANIN
// bounds a global task's inline fanin, but an in-graph task's producers travel in the
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
    constexpr size_t kInGraphTaskCap = MAX_IN_GRAPH_TASKS;
    recording.task_tensor_pool.reset(new (std::nothrow) simpler::hbg::Tensor[GRAPH_RECORD_TENSOR_POOL_ELEMS]);
    if (recording.task_tensor_pool == nullptr) return false;
    recording.tasks.resize(kInGraphTaskCap);
    recording.tensor_sources.reserve(kInGraphTaskCap * static_cast<size_t>(CORE_MAX_TENSOR_ARGS));
    recording.scalars.reserve(kInGraphTaskCap * static_cast<size_t>(CORE_MAX_SCALAR_ARGS));
    recording.scalar_sources.reserve(kInGraphTaskCap * static_cast<size_t>(CORE_MAX_SCALAR_ARGS));
    recording.output_ranges.reserve(kInGraphTaskCap);
    recording.predicates.reserve(kInGraphTaskCap);
    // Sized rather than reserved: a row is read by index, and every row a body can
    // reach must exist before the first task is recorded.
    recording.task_reach.assign(kInGraphTaskCap * GRAPH_REACH_WORDS, 0);
    return true;
}

// Bind this thread's storage to one in-flight entry and empty it. Returns false when the
// hazard map or the tensor pool cannot be stood up, which is only reachable on the
// thread's first recording.
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

// Bind this thread's storage to one in-flight entry and empty it. Returns false when the
// hazard map or the tensor pool cannot be stood up.
bool graph_recording_reset(GraphRecording &recording, const GraphInflightRecording &entry) {
    // A body over MAX_IN_GRAPH_TASKS is abandoned, but it still grew every array to its real
    // size while it ran. Handing that to the next recording would retain storage for a
    // Definition that can never be published, unbounded, for the process's life -- so an
    // over-cap recording gives its storage back instead of passing it on. This is what
    // makes the bound documented on GraphRecording::task_count true rather than nominal.
    if (recording.tasks.size() > MAX_IN_GRAPH_TASKS) {
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
    recording.tensor_sources.clear();
    recording.scalars.clear();
    recording.scalar_sources.clear();
    recording.internal_fanins.clear();
    recording.output_ranges.clear();
    recording.predicates.clear();
    // Cleared whole rather than per task, so a row left by the previous body cannot be
    // read as this one's: a task that bails out mid-record never writes its own row,
    // and a later task of the same body would otherwise fold a stale ancestor set.
    std::fill(recording.task_reach.begin(), recording.task_reach.end(), uint64_t{0});
    recording.reduced_edges = 0;
    return true;
}

bool graph_classify_tensor(
    const GraphRecording &recording, const RecordedInGraphTask &current, int32_t task_index,
    const simpler::hbg::Tensor &tensor, GraphRecordedTensorSourceRef *source
) {
    if (graph_tensor_from_boundary(recording, tensor, source)) return true;
    const uintptr_t tensor_addr = static_cast<uintptr_t>(tensor.buffer.addr);
    // The task being recorded is not in output_ranges yet — its entry is appended
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
    source->source_index = range.task_index;
    source->packed_offset = tensor_addr - range.begin;
    return true;
}

GraphBoundarySignature
graph_boundary_signature(const simpler::hbg::Tensor &tensor, TensorArgType type, uint16_t alias_rep) {
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

// Counts, section offsets and total_bytes for the image this recording produces,
// settled without writing any of it so the destination can be claimed at the
// exact size. required_heap comes from the fill, which is the pass that walks the
// tasks in order.
std::optional<GraphDefinition> graph_layout_definition(const GraphRecording &recording) {
    if (recording.unsupported || recording.task_count == 0 || recording.task_count > MAX_IN_GRAPH_TASKS ||
        recording.boundary_tensors().empty() || recording.boundary_tensors().size() > UINT16_MAX ||
        recording.boundary_tensors().size() != recording.boundary_types().size() ||
        recording.boundary_args() == nullptr) {
        return std::nullopt;
    }

    size_t total_tensors = 0;
    size_t total_scalars = 0;
    size_t total_fanins = 0;
    size_t root_count = 0;
    size_t predicate_count = 0;
    // task_count, not tasks.size(): the array keeps the slots a longer body left behind,
    // and those are not part of this recording.
    for (size_t i = 0; i < recording.task_count; ++i) {
        const RecordedInGraphTask &source = recording.tasks[i];
        if (source.tensor_count > UINT32_MAX - total_tensors || source.scalar_count > UINT32_MAX - total_scalars ||
            source.fanin_count > UINT32_MAX - total_fanins ||
            source.tensor_source_offset > recording.tensor_sources.size() ||
            source.tensor_count > recording.tensor_sources.size() - source.tensor_source_offset ||
            source.scalar_offset > recording.scalars.size() ||
            source.scalar_count > recording.scalars.size() - source.scalar_offset ||
            source.scalar_offset > recording.scalar_sources.size() ||
            source.scalar_count > recording.scalar_sources.size() - source.scalar_offset ||
            source.fanin_offset > recording.internal_fanins.size() ||
            source.fanin_count > recording.internal_fanins.size() - source.fanin_offset) {
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
    definition.task_count = static_cast<uint32_t>(recording.task_count);
    definition.edge_count = static_cast<uint32_t>(total_fanins);
    definition.root_count = static_cast<uint32_t>(root_count);
    definition.boundary_count = static_cast<uint32_t>(recording.boundary_tensors().size());
    definition.boundary_scalar_count = static_cast<uint32_t>(recording.boundary_scalar_count());
    definition.tensor_arg_count = static_cast<uint32_t>(total_tensors);
    definition.scalar_arg_count = static_cast<uint32_t>(total_scalars);
    definition.predicate_count = static_cast<uint32_t>(predicate_count);
    size_t execution_storage_bytes = 0;
    if (!graph_execution_storage_bytes(
            static_cast<int32_t>(definition.task_count), definition.tensor_arg_count, definition.scalar_arg_count,
            &execution_storage_bytes
        ) ||
        execution_storage_bytes > UINT32_MAX) {
        return std::nullopt;
    }
    definition.execution_storage_bytes = static_cast<uint32_t>(execution_storage_bytes);

    size_t image_bytes = sizeof(GraphDefinition);
    if (!graph_layout_section<uint32_t>(recording.task_count + 1, &image_bytes, &definition.off_fanout_offsets) ||
        !graph_layout_section<uint16_t>(total_fanins, &image_bytes, &definition.off_fanout_indices) ||
        !graph_layout_section<uint32_t>(recording.task_count + 1, &image_bytes, &definition.off_fanin_offsets) ||
        !graph_layout_section<uint16_t>(total_fanins, &image_bytes, &definition.off_fanin_indices) ||
        !graph_layout_section<uint16_t>(root_count, &image_bytes, &definition.off_root_indices) ||
        !graph_layout_section<uint64_t>(recording.task_count, &image_bytes, &definition.off_in_graph_task_offsets) ||
        !graph_layout_section<InGraphTaskDefinition>(
            recording.task_count, &image_bytes, &definition.off_in_graph_tasks
        ) ||
        !graph_layout_section<GraphTensor>(total_tensors, &image_bytes, &definition.off_tensors) ||
        !graph_layout_section<GraphTensorSourceRef>(total_tensors, &image_bytes, &definition.off_tensor_sources) ||
        !graph_layout_section<uint64_t>(total_scalars, &image_bytes, &definition.off_scalars) ||
        !graph_layout_section<GraphScalarSourceRef>(total_scalars, &image_bytes, &definition.off_scalar_sources) ||
        !graph_layout_section<GraphBoundarySignature>(
            recording.boundary_tensors().size(), &image_bytes, &definition.off_boundary_signatures
        ) ||
        !graph_layout_section<GraphPredicate>(predicate_count, &image_bytes, &definition.off_predicates)) {
        return std::nullopt;
    }
    definition.total_bytes = static_cast<uint32_t>(image_bytes);
    LOG_DEBUG(
        "[GraphExecution] Definition key=%#llx: %u tasks, %u edges shipped, %zu reduced away",
        static_cast<unsigned long long>(definition.full_key), definition.task_count, definition.edge_count,
        recording.reduced_edges
    );
    return definition;
}

// Write the image of `recording` at `image`, which must be graph_layout_definition's
// total_bytes and aligned for every section type it laid out. `definition` is that
// layout; the fill settles required_heap and writes the header.
//
// Every section is written in full here, so the destination's prior content does not
// reach the device — with one exception, `fanout_offsets`, which is accumulated
// rather than assigned and is therefore zeroed below before its first increment.
// The alignment slack between sections is written by nobody and read by nobody.
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
    auto *fanout_offsets = graph_image_section<uint32_t>(image, definition.off_fanout_offsets);
    auto *fanout_indices = graph_image_section<uint16_t>(image, definition.off_fanout_indices);
    auto *fanin_offsets = graph_image_section<uint32_t>(image, definition.off_fanin_offsets);
    auto *fanin_indices = graph_image_section<uint16_t>(image, definition.off_fanin_indices);
    auto *roots = graph_image_section<uint16_t>(image, definition.off_root_indices);
    auto *in_graph_task_offsets = graph_image_section<uint64_t>(image, definition.off_in_graph_task_offsets);
    auto *tasks = graph_image_section<InGraphTaskDefinition>(image, definition.off_in_graph_tasks);
    auto *tensors = graph_image_section<GraphTensor>(image, definition.off_tensors);
    auto *tensor_sources = graph_image_section<GraphTensorSourceRef>(image, definition.off_tensor_sources);
    auto *scalars = graph_image_section<uint64_t>(image, definition.off_scalars);
    auto *scalar_sources = graph_image_section<GraphScalarSourceRef>(image, definition.off_scalar_sources);
    auto *signatures = graph_image_section<GraphBoundarySignature>(image, definition.off_boundary_signatures);
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
    std::fill_n(fanout_offsets, recording.task_count + 1, 0U);
    fanin_offsets[0] = 0;
    for (size_t i = 0; i < recording.task_count; ++i) {
        const RecordedInGraphTask &source = recording.tasks[i];
        if (source.total_output_size > static_cast<size_t>(INT32_MAX) ||
            source.tensor_count > static_cast<uint32_t>(INT32_MAX) ||
            source.scalar_count > static_cast<uint32_t>(INT32_MAX) || source.fanin_count > UINT16_MAX) {
            return false;
        }
        in_graph_task_offsets[i] = required_heap;
        const uint64_t output_bytes = CHIP_ALIGN_UP(source.total_output_size, CHIP_ALIGN_SIZE);
        if (required_heap > UINT64_MAX - output_bytes) return false;
        required_heap += output_bytes;

        if (source.fanin_count == 0) roots[root_cursor++] = static_cast<uint16_t>(i);
        for (uint32_t f = 0; f < source.fanin_count; ++f) {
            const size_t producer = recording.internal_fanins[source.fanin_offset + f];
            if (producer >= i) return false;
            fanin_indices[fanin_cursor++] = static_cast<uint16_t>(producer);
            fanout_offsets[producer + 1]++;
        }
        fanin_offsets[i + 1] = static_cast<uint32_t>(fanin_cursor);

        InGraphTaskDefinition &task = tasks[i];
        std::copy(source.kernel_ids.begin(), source.kernel_ids.end(), std::begin(task.kernel_id));
        task.active_mask = source.active_mask.raw();
        task.task_attrs = source.task_attrs.raw();
        task.logical_block_num = source.logical_block_num;
        task.total_required_subtasks = source.total_required_subtasks;
        task.tensor_count = static_cast<int32_t>(source.tensor_count);
        task.scalar_count = static_cast<int32_t>(source.scalar_count);
        task.total_output_size = static_cast<int32_t>(source.total_output_size);
        task.tensor_offset = static_cast<uint32_t>(tensor_cursor);
        task.scalar_offset = static_cast<uint32_t>(scalar_cursor);
        task.dump_metadata = source.dump_metadata;
        task.predicate_slot = 0;
        if (source.predicate_index >= 0) {
            if (static_cast<size_t>(source.predicate_index) >= recording.predicates.size()) return false;
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
            predicates[predicate_cursor] = packed;
            task.predicate_slot = static_cast<uint16_t>(++predicate_cursor);
        }
        const simpler::hbg::Tensor *source_tensors = recording.task_tensors(source);
        for (size_t t = 0; t < source.tensor_count; ++t) {
            if (source_tensors[t].ndims > MAX_TENSOR_DIMS) return false;
            tensors[tensor_cursor] = graph_tensor_pack(source_tensors[t]);
            std::optional<GraphTensorSourceRef> packed_source =
                graph_pack_tensor_source(recording.tensor_sources[source.tensor_source_offset + t]);
            if (!packed_source.has_value()) return false;
            tensor_sources[tensor_cursor] = *packed_source;
            tensor_cursor++;
        }
        for (size_t scalar_index = 0; scalar_index < source.scalar_count; ++scalar_index) {
            std::optional<GraphScalarSourceRef> packed_source =
                graph_pack_scalar_source(recording.scalar_sources[source.scalar_offset + scalar_index]);
            if (!packed_source.has_value() ||
                (packed_source->source == static_cast<uint8_t>(GraphScalarSource::BOUNDARY) &&
                 packed_source->source_index >= recording.boundary_args()->scalar_count())) {
                return false;
            }
            scalar_sources[scalar_cursor] = *packed_source;
            scalars[scalar_cursor++] = packed_source->source == static_cast<uint8_t>(GraphScalarSource::BOUNDARY) ?
                                           0 :
                                           recording.scalars[source.scalar_offset + scalar_index];
        }
    }
    if (tensor_cursor != total_tensors || scalar_cursor != total_scalars || fanin_cursor != total_fanins ||
        root_cursor != root_count || predicate_cursor != predicate_count) {
        return false;
    }
    definition.required_heap = required_heap;
    for (size_t i = 0; i < recording.task_count; ++i)
        fanout_offsets[i + 1] += fanout_offsets[i];
    std::vector<uint32_t> cursors(fanout_offsets, fanout_offsets + recording.task_count);
    for (size_t consumer = 0; consumer < recording.task_count; ++consumer) {
        for (uint32_t f = fanin_offsets[consumer]; f < fanin_offsets[consumer + 1]; ++f) {
            const size_t producer = fanin_indices[f];
            fanout_indices[cursors[producer]++] = static_cast<uint16_t>(consumer);
        }
    }
    for (size_t i = 0; i < recording.boundary_tensors().size(); ++i) {
        const simpler::hbg::Tensor &tensor = recording.boundary_tensors()[i];
        if (tensor.ndims > MAX_TENSOR_DIMS) return false;
        uint16_t alias_rep = static_cast<uint16_t>(i);
        for (size_t j = 0; j < i; ++j) {
            if (recording.boundary_tensors()[j].buffer.addr == tensor.buffer.addr &&
                recording.boundary_tensors()[j].buffer.size == tensor.buffer.size) {
                alias_rep = static_cast<uint16_t>(j);
                break;
            }
        }
        signatures[i] = graph_boundary_signature(tensor, recording.boundary_types()[i], alias_rep);
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

// Where a published record's image is: in the arena at its object offset, or in
// the buffer it spilled to.
const GraphDefinition *graph_record_definition(const GraphHostState &state, const GraphDefinitionRecord &record) {
    if (record.object_offset == GRAPH_NO_OBJECT_OFFSET) {
        return graph_definition(record.spill.data(), record.spill.size());
    }
    if (state.arena.base == nullptr) return nullptr;
    return graph_definition(state.image_at(record.object_offset), record.bytes);
}

}  // namespace

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

GraphHostStatePtr make_graph_host_state(const GraphDefinitionArena &arena) {
    return GraphHostStatePtr{new (std::nothrow) GraphHostState{arena}};
}

void GraphHostStateDeleter::operator()(GraphHostState *state) const noexcept { delete state; }

size_t graph_host_upload_count(const GraphHostState &state) { return state.pending_uploads.size(); }

std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index) {
    if (index >= state.pending_uploads.size()) return std::nullopt;
    GraphPendingUpload &upload = state.pending_uploads[index];
    if (upload.outer_slot == nullptr) return std::nullopt;
    return GraphHostUpload{upload.outer_slot, upload.full_key};
}

size_t graph_host_arena_used(const GraphHostState &state) { return state.arena_cursor.load(std::memory_order_acquire); }

GraphHostDefinitionList graph_host_definitions(GraphHostState &state) {
    GraphHostDefinitionList list;
    list.entries.reserve(state.definitions.size());
    for (const auto &[key, record] : state.definitions) {
        if (graph_record_definition(state, record) == nullptr) continue;
        const bool spilled = record.object_offset == GRAPH_NO_OBJECT_OFFSET;
        list.entries.push_back(
            GraphHostDefinition{
                key, record.object_offset, spilled ? record.spill.data() : nullptr,
                spilled ? record.spill.size() : record.bytes
            }
        );
    }
    return list;
}

// Advances the epoch fanin_mark_seen keys against, so a mark left by an earlier task
// never reads as a repeat. The table is cleared only on wraparound.
static void next_fanin_seen_epoch(OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(
            orch->fanin_seen_epoch.get(), 0, static_cast<size_t>(orch->task_allocator.capacity()) * sizeof(uint32_t)
        );
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
}

// True when this producer was already appended under the current epoch. A negative
// local id cannot index the epoch table, so it is reported as not-yet-seen and the
// caller appends it rather than rejecting it; append_fanin_or_fail has already
// established that the producer is a GLOBAL task whose local id is a valid table
// entry.
static bool fanin_mark_seen(OrchestratorState &orch, TaskId producer_task_id) {
    const int32_t prod_local = static_cast<int32_t>(simpler::hbg::task_local_id(producer_task_id));
    if (prod_local < 0) {
        return false;
    }
    uint32_t *seen = orch.fanin_seen_epoch.get();
    uint32_t slot = static_cast<uint32_t>(prod_local);
    if (seen[slot] == orch.fanin_seen_current_epoch) {
        return true;
    }
    seen[slot] = orch.fanin_seen_current_epoch;
    return false;
}

// Polling: fanin is a flat array of position-independent producer local ids in the
// payload's own fanin region (no dep-pool spill, no producer pointers), deduped
// against the current fanin_seen epoch and hard-capped at CHIP_MAX_FANIN.
// fanin_slots and fanin_count are that region and payload.fanin_count itself: the
// region is named by a SelfRelativePtr delta, so the caller resolves it once, and
// the count accumulates in place instead of being copied back at the end.
static bool append_fanin_or_fail(
    OrchestratorState &orch, TaskId producer_task_id, TaskId self_task_id, int32_t *fanin_slots, int32_t &fanin_count
) {
    // Only a GLOBAL producer has an entry in the task table. An IN_GRAPH id's low
    // bits are a packed (Graph task, task index) pair, so using them as a table
    // index names an unrelated task — or, since get_slot_state_by_task_id does not
    // bounds-check, no task at all. A recorded task's id is IN_GRAPH and the
    // recorder resolves it against its own body, never here, so a foreign space
    // reaching this point is an id that escaped its Graph — a caller error, not a
    // case to tolerate.
    //
    // The table lookup lives here, after this check, rather than at the three call
    // sites: that keeps the id-space invariant in one place and makes it impossible
    // for a caller to form the out-of-bounds slot reference before reaching it.
    if (!simpler::hbg::is_global_task(producer_task_id)) {
        orch.report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "producer task %#llx is in id space %u, not GLOBAL; host_build_graph resolves every fanin edge against "
            "its one task table",
            static_cast<unsigned long long>(producer_task_id.raw),
            static_cast<unsigned int>(simpler::hbg::task_id_space(producer_task_id))
        );
        return false;
    }
    ChipTaskSlotState *prod_state = &orch.sm_header->tasks.get_slot_state_by_task_id(
        static_cast<int32_t>(simpler::hbg::task_local_id(producer_task_id))
    );
    // Skip a stale/reused producer slot: the cached owner id no longer resolves
    // to this producer (defensive — whole-graph-resident hbg does not reuse slots
    // at build time). A COMPLETED producer IS a real fanin edge under polling (its
    // completion_flags byte is set), so it is not skipped.
    if (prod_state->task == nullptr ||
        simpler::hbg::task_local_id(prod_state->task->task_id) != simpler::hbg::task_local_id(producer_task_id)) {
        return true;
    }
    // Dedup by producer local id, which is also its task-table slot.
    if (fanin_mark_seen(orch, producer_task_id)) {
        return true;
    }
    if (fanin_count >= CHIP_MAX_FANIN) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Fanin Capacity Exhausted!");
        LOG_ERROR("========================================");
        LOG_ERROR("HBG stores every producer dependency in the consumer task's fanin region.");
        LOG_ERROR("  Fanin:     used=%d/%d", fanin_count, CHIP_MAX_FANIN);
        LOG_ERROR("  Requested: at least %d distinct producer dependencies", fanin_count + 1);
        LOG_ERROR("Solution:");
        LOG_ERROR("  Reduce the task fanin to at most CHIP_MAX_FANIN=%d.", CHIP_MAX_FANIN);
        LOG_ERROR("  HBG has no dependency spill pool; runtime_env.ring_dep_pool does not apply.");
        LOG_ERROR("========================================");
        orch_mark_fatal(&orch, SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
        return false;
    }
    fanin_slots[fanin_count++] = static_cast<int32_t>(simpler::hbg::task_local_id(producer_task_id));

    // Reclaim gate: record this task as a consumer of the producer. The producer
    // slot retires once the completed_watermark reaches this consumer id.
    const int32_t self_local = static_cast<int32_t>(simpler::hbg::task_local_id(self_task_id));
    if (self_local > prod_state->last_consumer_local_id) {
        prod_state->last_consumer_local_id = self_local;
    }
    return true;
}

// Bounded transitive reduction of one task's fanin, run once per submit on the
// fully-appended edge list and before the count is published.
//
// Pass 1 folds two words over the edges. `direct` carries one bit per producer, at
// its distance d = self - producer; `via` is the union of each producer's own
// ancestor word shifted by that same d, which is distance addition — an ancestor
// a hops behind producer p is a + d hops behind this task. Their union is this
// task's ancestor set, published for its own consumers to walk. Pass 2 then drops
// every producer whose bit appears in `via`: some other producer already reaches
// it, so the direct edge orders nothing the chain does not, and the reduced edge
// list has the same reachability closure as the full one.
//
// Two properties of host_build_graph make the walk trivially sound, and both are
// why this is one word of state rather than the pinning protocol a ring runtime
// needs. A task id is its slot index, handed out by a forward-only bump allocator
// and never reclaimed, so it doubles as the global submission order — the distance
// is a subtraction, with no sequence number to carry. And a producer's entry was
// published by its own submit and is never rewritten, so reading it needs no proof
// that the slot still holds the task that wrote it.
//
// Dropping an edge does not shorten any buffer's lifetime. Retention rides
// last_consumer_local_id, which append_fanin_or_fail already raised to this task
// when the edge was appended and which nothing here lowers: a producer whose edge
// is dropped still waits for this task before the host may overwrite it.
//
// A producer further back than FANIN_REACH_WINDOW keeps its edge and contributes
// nothing — it and all its ancestors are unrepresentable in the window. At exactly
// FANIN_REACH_WINDOW only the direct bit is set: the shift would be undefined, and
// every ancestor of that producer already lies outside the window.
static void reduce_redundant_fanin(OrchestratorState *orch, TaskId self_task_id, int32_t *fanin_slots, int32_t &count) {
    const int32_t self = static_cast<int32_t>(simpler::hbg::task_local_id(self_task_id));
    uint64_t *reach = orch->fanin_reach.get();

    uint64_t direct = 0;
    uint64_t via = 0;
    for (int32_t i = 0; i < count; i++) {
        // A producer is always submitted before its consumer, so the distance is
        // positive; a non-positive one would name this task or an unsubmitted slot,
        // and is skipped rather than indexed.
        const int32_t d = self - fanin_slots[i];
        debug_assert(d > 0 && "a fanin producer is submitted before its consumer");
        if (d <= 0 || d > FANIN_REACH_WINDOW) continue;
        direct |= uint64_t{1} << (d - 1);
        if (d < FANIN_REACH_WINDOW) via |= reach[fanin_slots[i]] << d;
    }
    // A via bit lands at index (i + d) for ancestor bit i >= 0 of a producer at
    // distance d >= 1, so index 0 is unreachable: the immediately preceding task can
    // never be proven redundant. A set bit 0 means the shift-merge has drifted (the
    // classic form shifts by d - 1) and the pass is about to drop an edge nothing
    // covers.
    always_assert((via & uint64_t{1}) == 0 && "via bit 0 set: a distance-1 producer is not reducible");
    reach[self] = direct | via;

#if SIMPLER_DFX
    orch->fanin_edges_seen += count;
#endif
    if (via == 0) return;

    // Compact in place, preserving order: classify_fanin_state scans the region from
    // the back for the latest-submitted unmet producer, so the surviving edges must
    // stay in the order they were appended.
    int32_t kept = 0;
    for (int32_t i = 0; i < count; i++) {
        const int32_t d = self - fanin_slots[i];
        if (d > 0 && d <= FANIN_REACH_WINDOW && (via & (uint64_t{1} << (d - 1))) != 0) continue;
        fanin_slots[kept++] = fanin_slots[i];
    }
    // Every dropped producer is reachable from some other producer, and that covering
    // producer's local id is strictly larger, so following the cover relation up
    // terminates at one that nothing covers. A task with producers therefore always
    // keeps at least one edge. Emptying the region instead would make the device's
    // boot scan classify this task as a root and dispatch it against its unfinished
    // producers — a data race, not a hang, so it is worth catching here.
    always_assert(kept > 0 && "reduction emptied a non-empty fanin: no producer survived as a maximal element");
#if SIMPLER_DFX
    orch->fanin_edges_reduced += count - kept;
#endif
    count = kept;
}

struct PreparedTask {
    TaskId task_id = TaskId::invalid();
    TaskAllocResult alloc_result = {-1, nullptr, nullptr};
    TaskDescriptor *task = nullptr;
    TaskPayload *payload = nullptr;
    ChipTaskSlotState *slot_state = nullptr;
};

static OutputLayout calculate_output_layout(const CoreTaskArgs &args) {
    OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] = CHIP_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool prepare_task(
    OrchestratorState *orch, const CoreTaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PreparedTask *out
) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");
    auto &allocator = orch->task_allocator;

    int16_t block_num = args.launch_spec.block_num();
    int32_t active_subtasks_per_block = __builtin_popcount(active_mask.core_mask());
    int32_t total_required_subtasks = static_cast<int32_t>(block_num) * active_subtasks_per_block;
    if (block_num <= 0 || total_required_subtasks > std::numeric_limits<int16_t>::max()) {
        orch->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "block_num=%d with %d active slots requires %d subtasks; expected block_num >= 1 and total <= %d",
            block_num, active_subtasks_per_block, total_required_subtasks, std::numeric_limits<int16_t>::max()
        );
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, SIMPLER_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = simpler::hbg::make_global_task(static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->tasks.get_slot_state_by_task_id(out->alloc_result.task_id);
    out->task = &orch->sm_header->tasks.task_descriptors[out->alloc_result.task_id];
    out->payload = &orch->sm_header->tasks.task_payloads[out->alloc_result.task_id];

    // Bind the three argument regions before prefetch() and init(), both of which
    // dereference them. The scalar cursor advances in whole cache lines because init()
    // rounds its scalar memcpy up to one; a packed advance would let that rounding
    // write into the next task's region. A tensor region is aligned for any count,
    // simpler::hbg::Tensor being two cache lines. The fanin cursor advances at publish, not
    // here — see the comment where it does.
    const uint64_t max_tasks = static_cast<uint64_t>(orch->task_allocator.capacity());
    const int32_t scalar_span = CHIP_ALIGN_UP(args.scalar_count(), ARG_POOL_ALIGN / (int32_t)sizeof(uint64_t));
    debug_assert(static_cast<uint64_t>(orch->tensor_pool_cursor) + args.tensor_count() <= max_tasks * MAX_TENSOR_ARGS);
    debug_assert(static_cast<uint64_t>(orch->scalar_pool_cursor) + scalar_span <= max_tasks * MAX_SCALAR_ARGS);
    debug_assert(static_cast<uint64_t>(orch->fanin_pool_cursor) + CHIP_MAX_FANIN <= max_tasks * CHIP_MAX_FANIN);
    out->payload->bind_regions(
        orch->tensor_pool + orch->tensor_pool_cursor, orch->scalar_pool + orch->scalar_pool_cursor,
        orch->fanin_pool + orch->fanin_pool_cursor
    );
    orch->tensor_pool_cursor += args.tensor_count();
    orch->scalar_pool_cursor += scalar_span;

    // Init-on-write: this slot's dynamic scheduling fields and completion flag are
    // initialized here, as the orchestrator claims the slot. whole-graph-resident
    // hbg claims slots [0, total_tasks) exactly once and the device reads no slot
    // past total_tasks, so this claim-time write is the only per-slot SM reset and
    // the unclaimed tail is neither initialized nor read.
    out->slot_state->reset_for_reuse();
    orch->sm_header->tasks.completion_flags[out->alloc_result.task_id].store(0, std::memory_order_relaxed);

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets TaskHeaderView::init() skip the O(max_tasks) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by the reset_for_reuse() above:
    //   wake_list_head=nullptr, next_in_wake_list=nullptr,
    //   any_subtask_deferred=false, completed_subtasks=0, next_block_idx=0
    // task_state is set to PENDING here as the orchestrator populates the slot
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->task_state.store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    out->slot_state->total_required_subtasks = static_cast<int16_t>(total_required_subtasks);
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    out->slot_state->task_kind = active_mask.is_dummy() ? TaskKind::DUMMY : TaskKind::KERNEL;
    // Reclaim gate: seed last_consumer to self, so a producer with no consumers
    // is retirable once completed_watermark >= its own id. Each fanin edge bumps
    // it in append_fanin_or_fail. completion_flags for this slot were cleared
    // above (whole-graph-resident hbg never reuses a slot).
    out->slot_state->last_consumer_local_id = static_cast<int32_t>(simpler::hbg::task_local_id(out->task_id));
    // payload.fanin_count is left untouched here: submit_task_common zeroes it before
    // its fanin appends, which accumulate into it in place.

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

void OrchestratorState::begin_scope(ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // A Graph replays as a flat DAG with no scope structure: scope boundaries only
    // shape scheduling on the ordinary path, and the shadow-record path submits no
    // ordinary tasks. So a scope inside a Graph body must not touch the real scope
    // stack. Its manual/auto mode still matters, though — the recorder infers a recorded
    // task's producers with the same compute_task_fanin the ordinary path uses, and
    // that inference is suppressed inside a manual scope — so the depth is tracked on
    // the recording instead.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        // Reject what the ordinary path rejects. An auto scope inside a manual one is a
        // fatal below; accepting it here would let a Graph record and replay a
        // body that ordinary submission refuses. The push still happens so
        // end_scope stays balanced -- the recording is doomed either way, since
        // graph_commit turns an unsupported recording into SIMPLER_ERROR_INVALID_ARGS.
        if (recording->scope_stack_top >= CHIP_MAX_SCOPE_DEPTH - 1 ||
            (mode == ScopeMode::AUTO && recording->in_manual_scope())) {
            recording->unsupported = true;
        }
        if (recording->scope_stack_top < CHIP_MAX_SCOPE_DEPTH - 1) {
            ++recording->scope_stack_top;
            if (mode == ScopeMode::MANUAL && !recording->in_manual_scope()) {
                recording->manual_begin_depth = recording->scope_stack_top;
            }
        }
        return;
    }
    assert(orch->scope_stack_top < CHIP_MAX_SCOPE_DEPTH - 1 && "Scope stack overflow");
    if (mode == ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported"
        );
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    if (mode == ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
}

void OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    // Matches begin_scope: a scope inside a Graph body never touches the real
    // scope stack, only the recording's own manual-scope depth.
    if (GraphRecording *recording = active_graph_recording(orch); recording != nullptr) {
        if (recording->scope_stack_top >= 0) {
            if (recording->manual_begin_depth == recording->scope_stack_top) {
                recording->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
            }
            --recording->scope_stack_top;
        }
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    if (orch->scope_stack_top == orch->manual_begin_depth) {
        orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    }
    --orch->scope_stack_top;
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. Device completion never reclaims TensorMap
// entries; only synchronous dependency computation can remove a covered
// producer before this check. A pool that is still short here therefore cannot
// become large enough while the host waits: latch
// SIMPLER_ERROR_TENSORMAP_OVERFLOW and bail rather than letting new_entry()'s hard
// assert fire mid-registration. Returns false when the pool is exhausted or a
// fatal is already latched by another party.
static bool ensure_tensormap_capacity(OrchestratorState *orch, int32_t needed) {
    ChipTensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }
    if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != SIMPLER_ERROR_NONE) {
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
    LOG_ERROR("  Increase CHIP_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, SIMPLER_ERROR_TENSORMAP_OVERFLOW);
    return false;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    ORCH_PHASE_START();
    TaskOutputTensors result;
    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    SchedulerState *sched = orch->scheduler;
    TaskId task_id = prepared.task_id;
    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: open this task's graph entry before its dependency
    // steps run, so the edges STEP 1 / STEP 3 discover attach to it. The graph
    // is recorded from the dependency path itself, which makes it the runtime's
    // own answer rather than a reconstruction — the sole source of truth for
    // fanout now that the swimlane hot path no longer records it.
    const bool capture_dep_graph = dep_gen_host_graph_enabled();
    if (capture_dep_graph) {
        const std::array<int32_t, SUBTASK_SLOT_COUNT> kernel_ids_capture{
            aic_kernel_id,
            aiv0_kernel_id,
            aiv1_kernel_id,
        };
        dep_gen_host_graph_begin_task(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), kernel_ids_capture.data(),
            args.launch_spec.block_num(), args.tensor_count(), args.tensor_data(), args.tag_data()
        );
    }

    // The region delta is resolved once here, after prepare_task bound the regions.
    // Zeroing the count gives the appends their starting point, and is this
    // device-read field's only write on the submit path — hbg never zero-fills the
    // task table.
    next_fanin_seen_epoch(orch);
    int32_t *fanin_slots = payload.fanin_data();
    payload.fanin_count = 0;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        if (capture_dep_graph) {
            dep_gen_host_graph_add_explicit_edge(dep_task_id.raw);
        }
        if (!append_fanin_or_fail(*orch, dep_task_id, task_id, fanin_slots, payload.fanin_count)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](TaskId producer_task_id) -> bool {
        return append_fanin_or_fail(*orch, producer_task_id, task_id, fanin_slots, payload.fanin_count);
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
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // append_fanin_or_fail wrote every producer's local id into the payload's fanin
    // region, counted them in payload.fanin_count, and bumped each producer's
    // last_consumer_local_id. payload.init writes tensor_count/scalar_count only and
    // must not touch either, or it would discard that.
    payload.init(args, result, prepared.alloc_result, layout);

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/simpler::hbg::Tensor access. Both branches write predicate.op explicitly because
    // a payload slot is raw shared memory with no constructor; op == NONE means
    // "always dispatch".
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

    // === STEP 6: close the fanin region (device boot classifies) ===
    // Polling + host-orch: append_fanin_or_fail already wrote each producer's local
    // id into the payload's fanin region, counted them in payload.fanin_count, and
    // bumped each producer's last_consumer_local_id. There is NO fanout adjacency, NO
    // dep_pool, and NO ready routing here — the initial device boot scan classifies
    // each task once. A -1 result from classify_fanin_state routes the task through
    // push_ready_routed; otherwise the returned index selects the producer passed
    // to register_wake. Wake retargeting in register_wake may reclassify a task
    // when the selected producer is already complete.
    // The initial scan happens before the scheduler dispatch loop starts. Fanin is
    // a flat array of position-independent integers, so it crosses to the device
    // unchanged.
    //
    // Reduction runs first, on the complete edge list and before anything reads the
    // count: it both publishes this task's ancestor word for later submits and
    // settles which edges the region actually holds.
    reduce_redundant_fanin(orch, task_id, fanin_slots, payload.fanin_count);
    // The region's length is settled, so the cursor closes it at the real count. The
    // equality holds only while nothing between the bind and here bound another fanin
    // region, which is what makes the deferred advance safe.
    debug_assert(orch->fanin_pool_cursor == static_cast<int32_t>(payload.fanin_data() - orch->fanin_pool));
    orch->fanin_pool_cursor += CHIP_ALIGN_UP(payload.fanin_count, ARG_POOL_ALIGN / (int32_t)sizeof(int32_t));

    (void)sched;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    ORCH_PHASE_END(HostPhaseKind::OrchSubmitTask, task_id.raw);

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
        const simpler::hbg::Tensor &tensor = args.tensor(i).ref();
        const GraphBoundarySignature &signature = signatures[i];
        if (tensor.ndims > MAX_TENSOR_DIMS) {
            debug_assert(
                tensor.ndims <= MAX_TENSOR_DIMS && "Graph boundary simpler::hbg::Tensor rank is not supported"
            );
            LOG_WARN(
                "[GraphExecution] simpler::hbg::Tensor rank %u exceeds the fixed Graph boundary limit", tensor.ndims
            );
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
            const simpler::hbg::Tensor &other = args.tensor(j).ref();
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

bool graph_boundary_matches(const GraphBoundary &boundary, const GraphTaskArgs &args) {
    if (args.scalar_count() != boundary.scalar_count || args.explicit_dep_count() != 0 ||
        args.tensor_count() != static_cast<int32_t>(boundary.tensors.size()) ||
        boundary.tensors.size() != boundary.types.size()) {
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const simpler::hbg::Tensor &expected = boundary.tensors[static_cast<size_t>(i)];
        const simpler::hbg::Tensor &actual = args.tensor(i).ref();
        if (actual.ndims > MAX_TENSOR_DIMS || actual.buffer.size != expected.buffer.size ||
            actual.ndims != expected.ndims || actual.dtype != expected.dtype ||
            args.tag(i) != boundary.types[static_cast<size_t>(i)] || actual.manual_dep != expected.manual_dep ||
            actual.is_contiguous != expected.is_contiguous ||
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
            const simpler::hbg::Tensor &expected_other = boundary.tensors[static_cast<size_t>(j)];
            if (expected_other.buffer.addr == expected.buffer.addr &&
                expected_other.buffer.size == expected.buffer.size) {
                expected_alias = static_cast<uint16_t>(j);
                break;
            }
        }
        for (int32_t j = 0; j < i; ++j) {
            const simpler::hbg::Tensor &actual_other = args.tensor(j).ref();
            if (actual_other.buffer.addr == actual.buffer.addr && actual_other.buffer.size == actual.buffer.size) {
                actual_alias = static_cast<uint16_t>(j);
                break;
            }
        }
        if (actual_alias != expected_alias) return false;
    }
    return true;
}

void graph_reset_outer_payload(TaskPayload &payload) {
    payload.tensor_count = 0;
    payload.scalar_count = 0;
    payload.fanin_count = 0;
    payload.predicate = DispatchPredicate{};
    payload.early_dispatch_state.store(EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (auto &word : payload.staged_core_mask)
        word.store(0, std::memory_order_relaxed);
    payload.dispatch_fanin.store(0, std::memory_order_relaxed);
    payload.dispatch_propagated.store(0, std::memory_order_relaxed);
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
    const int32_t tensor_slots =
        static_cast<int32_t>(graph_boundary_tensor_pool_slots(static_cast<uint32_t>(args.tensor_count())));
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
    const TaskId task_id = simpler::hbg::make_global_task(static_cast<uint32_t>(allocation.task_id));
    SharedMemoryTaskHeader &tasks = orch->sm_header->tasks;
    TaskDescriptor &task = tasks.task_descriptors[allocation.task_id];
    TaskPayload &payload = tasks.task_payloads[allocation.task_id];
    ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(allocation.task_id);

    // Init-on-write, as in prepare_task: this slot's dynamic scheduling fields and
    // completion flag are established here, at the claim, because nothing else
    // writes them. A stale wake_list_head of WAKE_LIST_SENTINEL would close the
    // list against every consumer, and a stale completion flag would report the
    // Graph done before it ran.
    slot.reset_for_reuse();
    tasks.completion_flags[allocation.task_id].store(0, std::memory_order_relaxed);

    slot.bind_buffers(&payload, &task);
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
    slot.task_state.store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    slot.last_consumer_local_id = static_cast<int32_t>(simpler::hbg::task_local_id(task_id));
    slot.active_mask = ActiveMask{};
    slot.task_attrs = TaskAttrs{};
    slot.total_required_subtasks = 0;
    slot.logical_block_num = 1;
    slot.task_kind = TaskKind::GRAPH;

    task.task_id = task_id;
    std::fill(std::begin(task.kernel_id), std::end(task.kernel_id), INVALID_KERNEL_ID);
    task.packed_buffer_base = allocation.packed_base;
    task.packed_buffer_end = allocation.packed_end;
    graph_reset_outer_payload(payload);
    payload.tensor_count = args.tensor_count();
    payload.scalar_count = args.scalar_count();
    auto *boundary_tensors = reinterpret_cast<GraphTensor *>(payload.tensor_data());
    for (int32_t i = 0; i < args.tensor_count(); ++i)
        new (&boundary_tensors[i]) GraphTensor{graph_tensor_pack(args.tensor(i).ref())};
    if (args.scalar_count() != 0) {
        std::memcpy(
            payload.scalar_data(), args.scalar_data(),
            CHIP_ALIGN_UP(static_cast<size_t>(args.scalar_count()) * sizeof(uint64_t), ARG_POOL_ALIGN)
        );
    }

    // graph_reset_outer_payload above zeroed the count; the region delta is resolved
    // once here.
    next_fanin_seen_epoch(orch);
    int32_t *fanin_slots = payload.fanin_data();
    auto emit = [&](TaskId producer_id) -> bool {
        return append_fanin_or_fail(*orch, producer_id, task_id, fanin_slots, payload.fanin_count);
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
    // An outer Graph task takes reduction on the same terms as an ordinary one: it
    // completes only once its whole replayed body has, so it composes with the
    // ancestor walk exactly as a single task does.
    reduce_redundant_fanin(orch, task_id, fanin_slots, payload.fanin_count);
    // The region's length is settled, so the cursor closes it at the real count. The
    // equality holds only while nothing between the bind and here bound another fanin
    // region, which is what makes the deferred advance safe.
    debug_assert(orch->fanin_pool_cursor == static_cast<int32_t>(payload.fanin_data() - orch->fanin_pool));
    orch->fanin_pool_cursor += CHIP_ALIGN_UP(payload.fanin_count, ARG_POOL_ALIGN / (int32_t)sizeof(int32_t));

    pending.outer_slot = &slot;
    state->pending_uploads.push_back(pending);
    if (submitted_id != nullptr) *submitted_id = task_id;
#if SIMPLER_DFX
    orch->tasks_submitted++;
#endif
    return true;
}

bool graph_submit_definition(
    OrchestratorState *orch, GraphHostState *state, const GraphDefinition *definition, const GraphTaskArgs &args,
    TaskId *submitted_id
) {
    if (definition == nullptr || !graph_boundary_matches(*definition, args) ||
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
            pending.outer_slot == nullptr || pending.outer_slot->task == nullptr ||
            pending.outer_slot->task_kind != TaskKind::GRAPH) {
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
        pending.outer_slot->task->packed_buffer_base = packed_base;
        pending.outer_slot->task->packed_buffer_end = packed_end;
        pending.deferred_heap = false;
    }
    return true;
}

// Exact transitive reduction of one recorded task's internal fanin, run once the
// task's producers are all appended and before its fanin_count is taken.
//
// Same idea as the global submit path's reduction, at full resolution. `via` is the
// union of the producers' own ancestor rows: every task reachable through one of
// them, and therefore already ordered before this task by the chain. A producer
// whose bit `via` carries adds no ordering of its own and is dropped. The published
// row is `via` plus a bit for every producer — kept or dropped, since dropping an
// edge does not stop the producer being an ancestor.
//
// The full bitset is what a recorded body buys over the global path's one-word
// window: a producer arbitrarily far back in the body is still covered, and because
// each row is already a closure, a chain of any length collapses in this one pass.
// A Graph is recorded once and replayed thereafter, so the fold is amortized over
// every replay that reads the shortened CSR.
//
// This rewrites readiness only. A body's buffers come out of the Graph's own heap
// and are released when the Graph completes, not per task, so there is no lifetime
// an edge could have been holding.
//
// An over-cap task index, or a recording whose storage never stood up, has already
// been marked unsupported and its Definition will be refused; either is left alone
// rather than indexed past the row array.
void graph_reduce_recorded_fanin(GraphRecording &recording, size_t task_index, size_t fanin_offset) {
    if (task_index >= MAX_IN_GRAPH_TASKS ||
        recording.task_reach.size() < static_cast<size_t>(MAX_IN_GRAPH_TASKS) * GRAPH_REACH_WORDS) {
        return;
    }
    const size_t end = recording.internal_fanins.size();
    if (end == fanin_offset) {
        return;  // a body root: nothing to fold, and its row is already clear
    }

    uint64_t via[GRAPH_REACH_WORDS] = {};
    uint64_t direct[GRAPH_REACH_WORDS] = {};
    for (size_t i = fanin_offset; i < end; ++i) {
        const uint64_t *producer_row = &recording.task_reach[recording.internal_fanins[i] * GRAPH_REACH_WORDS];
        for (size_t w = 0; w < GRAPH_REACH_WORDS; ++w) {
            via[w] |= producer_row[w];
        }
    }

    // Compact in place, preserving order: the surviving producers keep the order the
    // recording appended them in, which is the order materialize writes to the CSR.
    size_t kept = fanin_offset;
    for (size_t i = fanin_offset; i < end; ++i) {
        const size_t producer = recording.internal_fanins[i];
        direct[producer / 64] |= uint64_t{1} << (producer % 64);
        if ((via[producer / 64] >> (producer % 64) & uint64_t{1}) != 0) {
            continue;
        }
        recording.internal_fanins[kept++] = producer;
    }
    // Each dropped producer is reached from another producer whose index is strictly
    // larger, so following the cover relation up terminates at one nothing covers: a
    // task with producers always keeps an edge. Emptying the list instead would make
    // materialize record this task as a body root and replay it against unfinished
    // producers.
    always_assert(kept > fanin_offset && "reduction emptied a recorded task's fanin");
    recording.reduced_edges += end - kept;
    recording.internal_fanins.resize(kept);

    uint64_t *self = &recording.task_reach[task_index * GRAPH_REACH_WORDS];
    for (size_t w = 0; w < GRAPH_REACH_WORDS; ++w) {
        self[w] = via[w] | direct[w];
    }
}

// Record one in-graph task while recording, without consuming a task-table
// slot. Builds the task's metadata and materialized outputs exactly as
// submit_task_common would, but assigns output buffers from the bit-63 virtual
// address range and derives internal fanins from tensor-source classification — so
// no task-table slot, tensormap entry, fanin-pool entry, or upload is produced for
// it. The resulting Definition is later attached to the outer GRAPH shells already
// submitted by the main thread. The returned TaskOutputTensors point into the
// recording's tensor pool, which is allocated at the cap and never grows, so they
// stay valid for the rest of the recording.
TaskOutputTensors graph_record_submit_in_graph_task(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    ORCH_PHASE_START();
    TaskOutputTensors result;
    GraphRecording &recording = *active_graph_recording(orch);

    const size_t task_index = recording.task_count;
    // A recorded task lives in the IN_GRAPH id space, so an id the body hands
    // around says which of the two kinds of thing it names without any arithmetic:
    // an IN_GRAPH id is a task of this body, indexed by its low field; a GLOBAL id is
    // a task submitted before the Graph, which nothing in the body may depend on.
    const TaskId task_id =
        simpler::hbg::make_in_graph_task(GRAPH_RECORD_NO_OWNING_GRAPH, static_cast<uint32_t>(task_index));
    result.set_task_id(task_id);

    if (task_index >= MAX_IN_GRAPH_TASKS || args.has_error) {
        recording.unsupported = true;
    }

    const OutputLayout layout = calculate_output_layout(args);
    const uint64_t aligned_output =
        layout.total_output_size > 0 ? CHIP_ALIGN_UP(static_cast<uint64_t>(layout.total_output_size), CHIP_ALIGN_SIZE) :
                                       0;
    if (recording.next_virtual_offset > GRAPH_RECORD_VIRTUAL_BASE - aligned_output) {
        recording.unsupported = true;
        return result;
    }
    const uintptr_t packed_base_addr = GRAPH_RECORD_VIRTUAL_BASE + recording.next_virtual_offset;
    recording.next_virtual_offset += aligned_output;

    // The task is filled in place, in the slot it will keep, and reset() puts the rest of
    // the slot back to a freshly recorded task's state. An over-cap body grows `tasks`,
    // which moves the slots, but the addresses handed to the caller live in the recording's
    // tensor pool rather than in a slot, so a move cannot invalidate them.
    if (task_index >= recording.tasks.size()) recording.tasks.emplace_back();
    RecordedInGraphTask &task = recording.tasks[task_index];
    task.reset();
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_ids[static_cast<int>(SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.active_mask = active_mask;
    task.task_attrs = task_attrs;
    task.task_attrs.set_early_resolve(false);
    task.logical_block_num = args.launch_spec.block_num();
    // Mirror prepare_task's contract: block_num must be positive and the subtask
    // count must fit int16_t. An out-of-contract value marks asynchronous
    // recording unsupported and makes commit fail-fast, rather than baking a
    // truncated or negative count into the cached Definition (which the device
    // would expand into an in-graph task that never completes).
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
    task.tensor_count = static_cast<uint32_t>(tensor_count);
    recording.task_tensor_cursor += static_cast<uint32_t>(tensor_count);
    simpler::hbg::Tensor *task_tensors = recording.task_tensors(task);
    // Value-initialized before the fill, not merely claimed: the slice holds whatever the
    // previous body left in it, and simpler::hbg::Tensor::init_from writes strides only up
    // to the new tensor's ndims, so a narrower tensor would inherit a wider one's trailing
    // strides.
    std::fill_n(task_tensors, static_cast<size_t>(tensor_count), simpler::hbg::Tensor{});
    for (int32_t i = 0; i < tensor_count; ++i) {
        simpler::hbg::Tensor &slot_tensor = task_tensors[static_cast<size_t>(i)];
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
        if (args.tag(i) == TensorArgType::OUTPUT) result.materialize_output(task_tensors[static_cast<size_t>(i)]);
    }
    task.scalar_offset = static_cast<uint32_t>(recording.scalars.size());
    task.scalar_count = static_cast<uint32_t>(args.scalar_count());
    recording.scalars.insert(recording.scalars.end(), args.scalars(), args.scalars() + args.scalar_count());
#if SIMPLER_DFX
    task.dump_metadata.dump_arg_mask = args.dump_arg_mask();
    task.dump_metadata.dump_arg_flags = args.dump_arg_index_ambiguous_mask();
    memcpy(task.dump_metadata.scalar_dtypes, args.scalar_dtypes(), args.scalar_count() * sizeof(uint8_t));
#endif

    // Classify each scalar's source: a plain literal is static Definition data,
    // while a value copied from a boundary scalar is refreshed on replay. A
    // mutable tracked boundary scalar is not supported and falls back.
    recording.scalar_sources.resize(static_cast<size_t>(task.scalar_offset) + task.scalar_count);
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        GraphRecordedScalarSourceRef source = graph_classify_scalar(recording, args, i);
        if (source.source == GraphRecordedScalarSource::INVALIDATED_BOUNDARY) recording.unsupported = true;
        recording.scalar_sources[static_cast<size_t>(task.scalar_offset) + static_cast<size_t>(i)] = source;
    }

    // Classify each tensor's source, then derive internal fanins from the
    // INTERNAL classifications plus any explicit internal dependency.
    task.tensor_source_offset = static_cast<uint32_t>(recording.tensor_sources.size());
    recording.tensor_sources.resize(static_cast<size_t>(task.tensor_source_offset) + tensor_count);
    for (int32_t i = 0; i < tensor_count; ++i) {
        // The out-pointer is used only for the duration of the call, so pointing
        // it into the flat array is safe even though a later task of this body grows it.
        if (!graph_classify_tensor(
                recording, task, static_cast<int32_t>(task_index), task_tensors[static_cast<size_t>(i)],
                &recording.tensor_sources[static_cast<size_t>(task.tensor_source_offset) + static_cast<size_t>(i)]
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
        // OWN_OUTPUT would read the task's own output before the task runs, so it
        // names no value the predicate could be evaluating. An index vector that
        // leaves the operand's extent is caught here too: materialize would
        // otherwise reject the baked offset on the device, where the failure is a
        // Scheduler fatal rather than a named unsupported construct.
        const uint64_t flat_offset =
            operand == nullptr ? 0 : operand->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
        if (operand == nullptr || operand->ndims > MAX_TENSOR_DIMS || pred.operand.ndims > operand->ndims ||
            flat_offset < operand->start_offset || flat_offset - operand->start_offset >= operand->extent_elem_cache ||
            !graph_classify_tensor(recording, task, static_cast<int32_t>(task_index), *operand, &recorded.source) ||
            recorded.source.source == GraphRecordedTensorSource::OWN_OUTPUT) {
            recording.unsupported = true;
        } else {
            recorded.operand.copy(*operand);
            recorded.elem_offset = flat_offset - operand->start_offset;
            recorded.elem_size = static_cast<uint8_t>(get_element_size(operand->dtype));
        }
        task.predicate_index = static_cast<int32_t>(recording.predicates.size());
        recording.predicates.push_back(recorded);
    }

    task.fanin_offset = static_cast<uint32_t>(recording.internal_fanins.size());
    // Dedup within this task's own range: the flat array's earlier entries belong
    // to the body's earlier tasks.
    auto add_fanin = [&recording, &task](size_t producer) {
        const auto begin = recording.internal_fanins.begin() + task.fanin_offset;
        if (std::find(begin, recording.internal_fanins.end(), producer) == recording.internal_fanins.end()) {
            recording.internal_fanins.push_back(producer);
        }
    };
    for (uint32_t i = 0; i < static_cast<uint32_t>(tensor_count); ++i) {
        const GraphRecordedTensorSourceRef &source = recording.tensor_sources[task.tensor_source_offset + i];
        if (source.source == GraphRecordedTensorSource::INTERNAL) add_fanin(source.source_index);
    }

    // Inferred hazards, on the same terms as the ordinary path. The loop above only
    // names the in-graph task that ALLOCATED each buffer; every write-then-read through a
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
        if (!recording.storage_ready || task_index >= MAX_IN_GRAPH_TASKS) {
            // An over-cap body is already abandoned, and its task ids have run past
            // the index field make_in_graph_task packs them into, so registering one
            // would key the map outside its task chains.
            recording.unsupported = true;
        } else if (recording.tensor_map.free_entries() < count_registrable_outputs(dep_inputs, manual_scope)) {
            // Recording one more task would assert inside new_entry(). Abandon the
            // Definition instead, so the run fails by name at graph_commit rather
            // than on a hard assert here.
            LOG_WARN(
                "[GraphExecution] recording hazard map exhausted at in-graph task %zu (%d entries); Graph abandoned",
                task_index, GRAPH_RECORD_TENSORMAP_POOL_SIZE
            );
            recording.unsupported = true;
        } else {
            auto emit_inferred = [&add_fanin, task_index](TaskId producer) -> bool {
                // A GLOBAL producer is a task submitted before the Graph. The outer shell
                // was submitted through the ordinary path against this same boundary, so
                // its own fanin already orders the whole body behind that task and the
                // Definition carries no edge of its own.
                if (simpler::hbg::is_global_task(producer)) return true;
                const uint32_t producer_index = simpler::hbg::task_local_id(producer);
                if (producer_index < static_cast<uint32_t>(task_index)) {
                    add_fanin(static_cast<size_t>(producer_index));
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
        if (simpler::hbg::is_global_task(dep)) {
            // Only the outer shell can order the body behind a pre-Graph task, and it
            // does so through its boundary args -- so a dep no boundary tensor carries
            // has no edge in the Definition and the body cannot be recorded.
            const bool represented_by_boundary = std::any_of(
                recording.boundary_tensors().begin(), recording.boundary_tensors().end(),
                [dep](const simpler::hbg::Tensor &tensor) {
                    return tensor.owner_task_id == dep;
                }
            );
            if (!represented_by_boundary) recording.unsupported = true;
            continue;
        }
        const uint32_t dep_index = simpler::hbg::task_local_id(dep);
        if (dep_index >= static_cast<uint32_t>(task_index)) {
            // A task of this body that is not yet recorded: the Definition's edges are
            // acyclic by construction, so a forward reference cannot be expressed.
            recording.unsupported = true;
            continue;
        }
        add_fanin(static_cast<size_t>(dep_index));
    }

    // Runs on the complete producer list and before the count is taken, so the count
    // and the CSR materialize writes both describe the reduced edge set.
    graph_reduce_recorded_fanin(recording, task_index, task.fanin_offset);
    task.fanin_count = static_cast<uint32_t>(recording.internal_fanins.size() - task.fanin_offset);
    if (task.record_packed_base != 0 && task.total_output_size != 0 &&
        task.total_output_size <= UINTPTR_MAX - task.record_packed_base) {
        const uintptr_t begin = task.record_packed_base;
        const uintptr_t end = begin + task.total_output_size;
        // The sorted-and-disjoint property the lookup depends on, checked rather
        // than assumed: a mid-recording heap rollback would break it, and today
        // the only rollback is graph_end's, after every task of the body is recorded.
        always_assert(recording.output_ranges.empty() || recording.output_ranges.back().end <= begin);
        recording.output_ranges.push_back({begin, end, static_cast<uint32_t>(task_index)});
    }
    // Published last: until this advances, the slot is not part of the recording, so
    // nothing that scans the recorded tasks can see the task being built.
    recording.task_count = task_index + 1;
    ORCH_PHASE_END(HostPhaseKind::OrchRecordInGraphTask, task_id.raw);
    return result;
}

}  // namespace

GraphScopeResult OrchestratorState::graph_begin(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash) {
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
                orch, state, graph_record_definition(*state, definition_it->second), args, &submitted
            )) {
            result.execute_block = false;
            result.task_id = submitted;
            ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, submitted.raw);
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
            ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, submitted.raw);
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
    entry->boundary.scalar_count = args.scalar_count();
    entry->boundary.tensors.reserve(static_cast<size_t>(args.tensor_count()));
    entry->boundary.types.reserve(static_cast<size_t>(args.tensor_count()));
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        entry->boundary.tensors.push_back(args.tensor(i).ref());
        entry->boundary.types.push_back(args.tag(i));
    }
    GraphInflightRecording *entry_ptr = entry.get();
    state->inflight.emplace(full_key, std::move(entry));
    state->inflight_count.store(state->inflight.size(), std::memory_order_release);

    TaskId submitted = TaskId::invalid();
    ORCH_PHASE_START();
    if (graph_submit_pending_definition(orch, state, full_key, args, &submitted)) {
        result.execute_block = false;
        result.recording = true;
        result.recording_handle = entry_ptr;
        result.task_id = submitted;
        ORCH_PHASE_END(HostPhaseKind::OrchGraphSubmit, submitted.raw);
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

bool OrchestratorState::graph_prepare(void *recording_handle, const GraphTaskArgs &args) {
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
    if (entry->status() != GraphRecordingStatus::RECORDING) {
        return false;
    }
    // The entry was created from this very boundary at graph_begin, and the handle names
    // that entry rather than being searched for, so a mismatch here is unreachable. The
    // comparison walks up to 128 simpler::hbg::Tensor descriptors on the thread whose start-up
    // latency this path exists to keep short, so it is an assertion: debug builds still
    // catch a boundary that stopped matching, release builds compile it out.
    debug_assert(
        graph_boundary_matches(entry->boundary, args) &&
        "the entry's boundary copy must match the boundary graph_begin recorded"
    );
    // This thread's own storage, emptied rather than allocated -- see
    // recorder_recording(). Failure is reachable only on this thread's first recording,
    // where the hazard map is stood up; the caller then aborts the recording, and the
    // outer shell it already submitted replays nothing.
    GraphRecording &recording = recorder_recording();
    if (!graph_recording_reset(recording, *entry)) {
        LOG_WARN("%s", "[GraphExecution] recording hazard map allocation failed; recording abandoned");
        return false;
    }
    args.anchor_scalar_sources();
    entry->boundary.args = &args;
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
    // which is why this is hygiene rather than a fix -- but boundary_tensors() does not
    // null-check, so a future reader outside a recording would follow it.
    unbind_recorder_boundary();
    g_active_graph_entry = nullptr;
    g_active_graph_recording = nullptr;
    g_active_graph_owner = nullptr;
    state->recording_cv.notify_all();
}

// Finish the background recording and publish the Definition. The main
// thread finalizes the already-submitted outer Graph tasks in graph_commit.
bool OrchestratorState::graph_end() {
    GraphHostState *state = graph_state_from(this);
    GraphRecording *recording = active_graph_recording(this);
    GraphInflightRecording *entry = g_active_graph_entry;
    if (state == nullptr || recording == nullptr || entry == nullptr) return false;

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
        "[GraphExecution] define key=0x%llx tasks=%u bytes=%u", static_cast<unsigned long long>(header->full_key),
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

TaskOutputTensors OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
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
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    if (!static_cast<bool>(active_mask)) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "MixedKernels names no active slot; set at least one of aic/aiv0/aiv1 kernel_id"
        );
        return TaskOutputTensors{};
    }

    int16_t block_num = args.launch_spec.block_num();

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(SUBTASK_MASK_AIV1);
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
        ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                SIMPLER_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
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
        return graph_record_submit_in_graph_task(
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
TaskOutputTensors OrchestratorState::submit_dummy_task(const CoreTaskArgs &args) {
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
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_in_graph_task(
            orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors OrchestratorState::alloc_tensors(const CoreTaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
        );
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();
    ORCH_PHASE_START();

    if (args.has_error) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    // A Graph body may allocate. The allocation records as a kernel-less in-graph
    // task — the same shape submit_dummy_task records — and replay reserves the
    // intermediate heap for every in-graph task anyway, so the outputs land at
    // addresses the replayed Definition derives for itself.
    if (active_graph_recording(orch) != nullptr) {
        return graph_record_submit_in_graph_task(
            orch, args, ActiveMask{}, TaskAttrs{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
        );
    }

    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time attributes. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_count = 0;  // hidden-alloc tasks have no producer dependencies
    // With no producers there is nothing to reduce, but the slot is still a producer
    // for later submits, so its ancestor word has to say so: empty, not whatever the
    // allocation left there.
    orch->fanin_reach[simpler::hbg::task_local_id(prepared.task_id)] = 0;
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required — a consumer reads this slot's task_attrs and completion
        // mirror, both set below — but worker dispatch fields are never
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
        SharedMemoryTaskHeader &done_tasks = orch->sm_header->tasks;
        int32_t done_local = static_cast<int32_t>(simpler::hbg::task_local_id(prepared.task_id));
        done_tasks.set_completion_flag(done_local);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    ORCH_PHASE_END(HostPhaseKind::OrchAllocTensors, prepared.task_id.raw);

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

void OrchestratorState::mark_done() {
    auto *orch = this;
    int32_t total_tasks = orch->task_allocator.active_count();
    if (total_tasks > 0) {
        LOG_DEBUG("=== [Orchestrator] total_tasks=%d ===", total_tasks);
#if SIMPLER_DFX
        LOG_DEBUG(
            "=== [Orchestrator] fanin edges: %lld built, %lld reduced (window=%d) ===",
            static_cast<long long>(orch->fanin_edges_seen), static_cast<long long>(orch->fanin_edges_reduced),
            FANIN_REACH_WINDOW
        );
#endif
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
OrchProfilingData orchestrator_get_profiling() {
    OrchProfilingData d;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.submit_count = g_orch_submit_count;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.args_atomic_count = g_orch_args_atomic_count;

    // Reset
    g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_args_atomic_count = 0;
    return d;
}
#endif
