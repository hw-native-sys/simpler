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
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <new>
#include <type_traits>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_graph_execution.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Verify the captured Tensor blob size in DepGenRecord matches the runtime
// Tensor layout. The platform header defines DEP_GEN_TENSOR_SIZE without
// including runtime/tensor.h, so this check lives at the orch callsite.
static_assert(sizeof(Tensor) == DEP_GEN_TENSOR_SIZE, "DepGenRecord::tensors slot size out of sync with sizeof(Tensor)");
// DEP_GEN_MAX_EXPLICIT_DEPS is a diagnostic-side capture cap only; the runtime
// imposes no hard cap on explicit dep count. If a submit exceeds this cap,
// dep_gen_aicpu_record_submit() logs and truncates — runtime correctness is
// unaffected, only the captured replay record is truncated.

// Weak fallbacks: dep_gen_collector_aicpu.cpp provides the strong symbols in
// AICPU builds. Host builds (host_build_graph runtime, future dep_gen replay)
// link these no-op stubs so the runtime translation unit is self-contained.
// Visibility is hidden so the HOST .so doesn't export them into the global
// dynamic symbol table where they'd shadow the AICPU .so's strong symbols
// (same pattern as get_sys_cnt_aicpu / l2_swimlane_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(
    uint64_t, bool, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, int, const int32_t[3]
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// is_dep_gen_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// Heap-ring wrap report, called from the allocator (pto_ring_buffer.h) on each
// wrap. Strong definition lives in the AICPU collector; host builds fall back to
// this weak no-op so the runtime translation unit stays self-contained.
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}

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
#if SIMPLER_DFX
namespace {
void record_host_orch_phase(
    PTO2OrchestratorState *orch, uint64_t start_time, uint64_t end_time, uint64_t task_id, uint32_t submit_idx
) {
    if (orch == nullptr || orch->host_orch_phase_records == nullptr ||
        orch->host_orch_phase_count >= orch->host_orch_phase_capacity) {
        return;
    }
    auto &record = orch->host_orch_phase_records[orch->host_orch_phase_count++];
    record.start_time = start_time;
    record.end_time = end_time;
    record.task_id = task_id;
    record.submit_idx = submit_idx;
    record._pad = 0;
}

}  // namespace
#endif

#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
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
// Weak fallback for builds that don't link l2_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t start, uint64_t end, uint64_t task_id, uint32_t submit_idx) {
    (void)start;
    (void)end;
    (void)task_id;
    (void)submit_idx;
}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                       \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                \
    do {                                                                                   \
        if (_prof_active) {                                                                \
            record_host_orch_phase(orch, _submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                  \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
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
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t start, uint64_t end, uint64_t task_id, uint32_t submit_idx) {
    (void)start;
    (void)end;
    (void)task_id;
    (void)submit_idx;
}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                \
    do {                                                                                   \
        if (_prof_active) {                                                                \
            _t1 = get_sys_cnt_aicpu();                                                     \
            record_host_orch_phase(orch, _submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                  \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
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

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

namespace {

constexpr int32_t PTO2_GRAPH_DEFINITION_CAP = 16;
constexpr int32_t PTO2_GRAPH_MAX_TASKS = 1024;
constexpr int32_t PTO2_GRAPH_MAX_FANIN_PER_TASK = 128;
constexpr int32_t PTO2_GRAPH_MAX_INTERNAL_EDGES = PTO2_GRAPH_MAX_TASKS * PTO2_GRAPH_MAX_FANIN_PER_TASK;

enum class PTO2GraphTensorSource : uint8_t {
    BOUNDARY_EXACT = 0,
    BOUNDARY_VIEW = 1,
    INTERNAL = 2,
    OWN_OUTPUT = 3,
};

enum class PTO2GraphFaninSource : uint8_t {
    INTERNAL = 0,
    EXTERNAL_LOCAL_DELTA = 1,
};

struct PTO2GraphTensorSourceRef {
    PTO2GraphTensorSource source{PTO2GraphTensorSource::BOUNDARY_EXACT};
    uint16_t source_index{0};
    uint64_t packed_offset{0};
};

struct PTO2GraphFaninRef {
    PTO2GraphFaninSource source{PTO2GraphFaninSource::INTERNAL};
    int32_t value{0};
};

// Capture-only representation. Its fixed arrays make recording allocation-free
// but are never copied wholesale into the process-local Definition cache.
struct PTO2GraphRecordedNode {
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT]{INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    ActiveMask active_mask{};
    int16_t logical_block_num{1};
    int16_t total_required_subtasks{0};
    int32_t task_timing_slot{TASK_TIMING_SLOT_NONE};
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t total_output_size{0};
    uint64_t record_packed_base{0};
    Tensor tensors[MAX_TENSOR_ARGS];
    PTO2GraphTensorSourceRef tensor_sources[MAX_TENSOR_ARGS];
    TensorArgType tensor_arg_types[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    uint16_t scalar_source_indices[MAX_SCALAR_ARGS];
    int32_t fanin_count{0};
    PTO2GraphFaninRef fanins[PTO2_GRAPH_MAX_FANIN_PER_TASK];
};

static_assert(std::is_trivially_copyable_v<PTO2GraphRecordedNode>, "Graph capture nodes must remain byte-copyable");

// Cache/runtime representation. Variable-length tensors, sources, scalars,
// and scalar-source indices live in Definition-wide packed arrays.
struct PTO2GraphNodeDefinition {
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];
    ActiveMask active_mask;
    int16_t logical_block_num;
    int16_t total_required_subtasks;
    int32_t task_timing_slot;
    int32_t tensor_count;
    int32_t scalar_count;
    int32_t total_output_size;
    uint32_t tensor_offset;
    uint32_t scalar_offset;
};

static_assert(
    std::is_trivially_copyable_v<PTO2GraphNodeDefinition>, "Compact Graph Definition nodes must remain byte-copyable"
);

struct PTO2GraphReplayPlan {
    uint64_t required_heap{0};
    int32_t boundary_count{0};
    uint32_t required_tensor_count{0};
    uint32_t required_scalar_count{0};
    uint16_t boundary_indices[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
    TensorArgType boundary_types[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
};

struct PTO2GraphRecordingDefinition {
    bool in_use{false};
    uint64_t full_key{0};
    int32_t task_count{0};
    int32_t edge_count{0};
    int32_t root_count{0};
    PTO2GraphReplayPlan replay_plan;
    uint32_t fanout_offsets[PTO2_GRAPH_MAX_TASKS + 1]{};
    uint16_t fanout_indices[PTO2_GRAPH_MAX_INTERNAL_EDGES]{};
    uint16_t fanin_counts[PTO2_GRAPH_MAX_TASKS]{};
    uint16_t root_indices[PTO2_GRAPH_MAX_TASKS]{};
    uint64_t node_offsets[PTO2_GRAPH_MAX_TASKS]{};
    PTO2GraphRecordedNode tasks[PTO2_GRAPH_MAX_TASKS];
};

struct PTO2GraphDefinition {
    bool in_use{false};
    uint64_t full_key{0};
    int32_t task_count{0};
    int32_t edge_count{0};
    int32_t root_count{0};
    PTO2GraphReplayPlan replay_plan;
    size_t storage_bytes{0};
    void *storage{nullptr};
    uint32_t *fanout_offsets{nullptr};
    uint16_t *fanout_indices{nullptr};
    uint16_t *fanin_counts{nullptr};
    uint16_t *root_indices{nullptr};
    uint64_t *node_offsets{nullptr};
    PTO2GraphNodeDefinition *tasks{nullptr};
    Tensor *tensors{nullptr};
    PTO2GraphTensorSourceRef *tensor_sources{nullptr};
    uint64_t *scalars{nullptr};
    uint16_t *scalar_source_indices{nullptr};
};

struct PTO2GraphRecordingState {
    bool active{false};
    bool pending_finalize{false};
    bool unsupported{false};
    int32_t unsupported_reason{0};
    int32_t unsupported_task_index{-1};
    int32_t unsupported_tensor_index{-1};
    uint64_t full_key{0};
    int32_t start_local_task_id{0};
    int32_t current_task_index{-1};
    int32_t current_fanin_count{0};
    PTO2GraphFaninRef current_fanins[PTO2_GRAPH_MAX_FANIN_PER_TASK];
    ChipStorageTaskArgs args;
    PTO2GraphRecordingDefinition temp;
};

// Definitions survive repeated simpler_run calls while the AICPU runtime DSO
// remains loaded for the DeviceRunner lifetime.
PTO2GraphDefinition g_graph_definitions[PTO2_GRAPH_DEFINITION_CAP];
int32_t g_graph_next_replace = 0;
PTO2GraphRecordingState g_graph_recording;

enum PTO2GraphUnsupportedReason : int32_t {
    PTO2_GRAPH_UNSUPPORTED_NONE = 0,
    PTO2_GRAPH_UNSUPPORTED_TASK_WINDOW = 1,
    PTO2_GRAPH_UNSUPPORTED_NULL_PRODUCER = 2,
    PTO2_GRAPH_UNSUPPORTED_EXTERNAL_PRODUCER = 3,
    PTO2_GRAPH_UNSUPPORTED_FANIN_OVERFLOW = 4,
    PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_ORDER = 5,
    PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_NULL = 6,
    PTO2_GRAPH_UNSUPPORTED_ARG_OVERFLOW = 7,
    PTO2_GRAPH_UNSUPPORTED_TENSOR_SOURCE = 8,
    PTO2_GRAPH_UNSUPPORTED_NESTED_SCOPE = 9,
    PTO2_GRAPH_UNSUPPORTED_EXTERNAL_EXPLICIT_DEP = 10,
    PTO2_GRAPH_UNSUPPORTED_PREDICATE = 11,
};

void graph_mark_unsupported(PTO2GraphUnsupportedReason reason, int32_t task_index = -1, int32_t tensor_index = -1) {
    if (!g_graph_recording.unsupported) {
        g_graph_recording.unsupported_reason = static_cast<int32_t>(reason);
        g_graph_recording.unsupported_task_index = task_index;
        g_graph_recording.unsupported_tensor_index = tensor_index;
    }
    g_graph_recording.unsupported = true;
}

void reset_graph_definition_header(PTO2GraphRecordingDefinition *templ) {
    templ->in_use = false;
    templ->full_key = 0;
    templ->task_count = 0;
    templ->edge_count = 0;
    templ->root_count = 0;
    templ->replay_plan = PTO2GraphReplayPlan{};
}

void reset_graph_recording() {
    g_graph_recording.active = false;
    g_graph_recording.pending_finalize = false;
    g_graph_recording.unsupported = false;
    g_graph_recording.unsupported_reason = PTO2_GRAPH_UNSUPPORTED_NONE;
    g_graph_recording.unsupported_task_index = -1;
    g_graph_recording.unsupported_tensor_index = -1;
    g_graph_recording.full_key = 0;
    g_graph_recording.start_local_task_id = 0;
    g_graph_recording.current_task_index = -1;
    g_graph_recording.current_fanin_count = 0;
    g_graph_recording.args.clear();
    reset_graph_definition_header(&g_graph_recording.temp);
}

uint64_t graph_full_key(uint64_t callable_hash, uint64_t graph_key) {
    uint64_t h = 1469598103934665603ULL;
    h = pto2_graph_hash_bytes(h, &PTO2_GRAPH_CACHE_SCHEMA_VERSION, sizeof(PTO2_GRAPH_CACHE_SCHEMA_VERSION));
    h = pto2_graph_hash_bytes(h, &callable_hash, sizeof(callable_hash));
    return pto2_graph_hash_bytes(h, &graph_key, sizeof(graph_key));
}

bool graph_snapshot_args(const L2TaskArgs &source, ChipStorageTaskArgs *snapshot) {
    if (snapshot == nullptr || source.has_error ||
        source.tensor_count() > static_cast<int32_t>(PTO2_GRAPH_MAX_TENSOR_ARGS) ||
        source.scalar_count() > static_cast<int32_t>(PTO2_GRAPH_MAX_SCALAR_ARGS)) {
        return false;
    }
    snapshot->clear();
    for (int32_t i = 0; i < source.tensor_count(); ++i) {
        if (source.tag(i) == TensorArgType::OUTPUT) return false;
        snapshot->add_tensor(source.tensor(i).ref());
    }
    for (int32_t i = 0; i < source.scalar_count(); ++i) {
        snapshot->add_scalar(source.scalar(i));
    }
    return true;
}

PTO2GraphDefinition *find_graph_definition(uint64_t full_key) {
    for (int32_t i = 0; i < PTO2_GRAPH_DEFINITION_CAP; ++i) {
        if (g_graph_definitions[i].in_use && g_graph_definitions[i].full_key == full_key) {
            return &g_graph_definitions[i];
        }
    }
    return nullptr;
}

bool graph_definition_append_storage(
    size_t *cursor, size_t count, size_t element_size, size_t alignment, size_t *offset
) {
    if (cursor == nullptr || offset == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    if (*cursor > SIZE_MAX - (alignment - 1)) return false;
    size_t aligned = (*cursor + alignment - 1) & ~(alignment - 1);
    if (count > 0 && element_size > SIZE_MAX / count) return false;
    size_t bytes = count * element_size;
    if (aligned > SIZE_MAX - bytes) return false;
    *offset = aligned;
    *cursor = aligned + bytes;
    return true;
}

void release_graph_definition(PTO2GraphDefinition *definition) {
    if (definition == nullptr) return;
    free(definition->storage);
    *definition = PTO2GraphDefinition{};
}

bool compact_graph_definition(PTO2GraphDefinition *definition, const PTO2GraphRecordingDefinition &recorded) {
    if (definition == nullptr || recorded.task_count <= 0 || recorded.edge_count < 0 || recorded.root_count < 0) {
        return false;
    }

    size_t tensor_arg_count = 0;
    size_t scalar_arg_count = 0;
    for (int32_t i = 0; i < recorded.task_count; ++i) {
        const PTO2GraphRecordedNode &node = recorded.tasks[i];
        if (node.tensor_count < 0 || node.tensor_count > MAX_TENSOR_ARGS || node.scalar_count < 0 ||
            node.scalar_count > MAX_SCALAR_ARGS) {
            return false;
        }
        tensor_arg_count += static_cast<size_t>(node.tensor_count);
        scalar_arg_count += static_cast<size_t>(node.scalar_count);
    }
    if (tensor_arg_count > UINT32_MAX || scalar_arg_count > UINT32_MAX) return false;

    size_t cursor = 0;
    size_t fanout_offsets_offset = 0;
    size_t fanout_indices_offset = 0;
    size_t fanin_counts_offset = 0;
    size_t root_indices_offset = 0;
    size_t node_offsets_offset = 0;
    size_t tasks_offset = 0;
    size_t tensors_offset = 0;
    size_t tensor_sources_offset = 0;
    size_t scalars_offset = 0;
    size_t scalar_source_indices_offset = 0;
    if (!graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.task_count) + 1, sizeof(uint32_t), alignof(uint32_t),
            &fanout_offsets_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.edge_count), sizeof(uint16_t), alignof(uint16_t),
            &fanout_indices_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.task_count), sizeof(uint16_t), alignof(uint16_t), &fanin_counts_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.root_count), sizeof(uint16_t), alignof(uint16_t), &root_indices_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.task_count), sizeof(uint64_t), alignof(uint64_t), &node_offsets_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, static_cast<size_t>(recorded.task_count), sizeof(PTO2GraphNodeDefinition),
            alignof(PTO2GraphNodeDefinition), &tasks_offset
        ) ||
        !graph_definition_append_storage(&cursor, tensor_arg_count, sizeof(Tensor), alignof(Tensor), &tensors_offset) ||
        !graph_definition_append_storage(
            &cursor, tensor_arg_count, sizeof(PTO2GraphTensorSourceRef), alignof(PTO2GraphTensorSourceRef),
            &tensor_sources_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, scalar_arg_count, sizeof(uint64_t), alignof(uint64_t), &scalars_offset
        ) ||
        !graph_definition_append_storage(
            &cursor, scalar_arg_count, sizeof(uint16_t), alignof(uint16_t), &scalar_source_indices_offset
        )) {
        return false;
    }

    void *storage = nullptr;
    if (posix_memalign(&storage, alignof(Tensor), cursor) != 0) return false;
    auto *base = static_cast<char *>(storage);

    PTO2GraphDefinition compact;
    compact.in_use = true;
    compact.full_key = recorded.full_key;
    compact.task_count = recorded.task_count;
    compact.edge_count = recorded.edge_count;
    compact.root_count = recorded.root_count;
    compact.replay_plan = recorded.replay_plan;
    compact.storage_bytes = cursor;
    compact.storage = storage;
    compact.fanout_offsets = reinterpret_cast<uint32_t *>(base + fanout_offsets_offset);
    compact.fanout_indices = reinterpret_cast<uint16_t *>(base + fanout_indices_offset);
    compact.fanin_counts = reinterpret_cast<uint16_t *>(base + fanin_counts_offset);
    compact.root_indices = reinterpret_cast<uint16_t *>(base + root_indices_offset);
    compact.node_offsets = reinterpret_cast<uint64_t *>(base + node_offsets_offset);
    compact.tasks = reinterpret_cast<PTO2GraphNodeDefinition *>(base + tasks_offset);
    compact.tensors = reinterpret_cast<Tensor *>(base + tensors_offset);
    compact.tensor_sources = reinterpret_cast<PTO2GraphTensorSourceRef *>(base + tensor_sources_offset);
    compact.scalars = reinterpret_cast<uint64_t *>(base + scalars_offset);
    compact.scalar_source_indices = reinterpret_cast<uint16_t *>(base + scalar_source_indices_offset);

    memcpy(
        compact.fanout_offsets, recorded.fanout_offsets,
        sizeof(uint32_t) * (static_cast<size_t>(recorded.task_count) + 1)
    );
    memcpy(
        compact.fanout_indices, recorded.fanout_indices, sizeof(uint16_t) * static_cast<size_t>(recorded.edge_count)
    );
    memcpy(compact.fanin_counts, recorded.fanin_counts, sizeof(uint16_t) * static_cast<size_t>(recorded.task_count));
    memcpy(compact.root_indices, recorded.root_indices, sizeof(uint16_t) * static_cast<size_t>(recorded.root_count));
    memcpy(compact.node_offsets, recorded.node_offsets, sizeof(uint64_t) * static_cast<size_t>(recorded.task_count));

    uint32_t tensor_cursor = 0;
    uint32_t scalar_cursor = 0;
    for (int32_t i = 0; i < recorded.task_count; ++i) {
        const PTO2GraphRecordedNode &src = recorded.tasks[i];
        PTO2GraphNodeDefinition &dst = compact.tasks[i];
        for (int32_t k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
            dst.kernel_id[k] = src.kernel_id[k];
        dst.active_mask = src.active_mask;
        dst.logical_block_num = src.logical_block_num;
        dst.total_required_subtasks = src.total_required_subtasks;
        dst.task_timing_slot = src.task_timing_slot;
        dst.tensor_count = src.tensor_count;
        dst.scalar_count = src.scalar_count;
        dst.total_output_size = src.total_output_size;
        dst.tensor_offset = tensor_cursor;
        dst.scalar_offset = scalar_cursor;

        memcpy(compact.tensors + tensor_cursor, src.tensors, sizeof(Tensor) * static_cast<size_t>(src.tensor_count));
        memcpy(
            compact.tensor_sources + tensor_cursor, src.tensor_sources,
            sizeof(PTO2GraphTensorSourceRef) * static_cast<size_t>(src.tensor_count)
        );
        memcpy(compact.scalars + scalar_cursor, src.scalars, sizeof(uint64_t) * static_cast<size_t>(src.scalar_count));
        memcpy(
            compact.scalar_source_indices + scalar_cursor, src.scalar_source_indices,
            sizeof(uint16_t) * static_cast<size_t>(src.scalar_count)
        );
        tensor_cursor += static_cast<uint32_t>(src.tensor_count);
        scalar_cursor += static_cast<uint32_t>(src.scalar_count);
    }

    release_graph_definition(definition);
    *definition = compact;
    return true;
}

bool store_graph_definition(const PTO2GraphRecordingDefinition &templ) {
    int32_t target = -1;
    for (int32_t i = 0; i < PTO2_GRAPH_DEFINITION_CAP; ++i) {
        if (!g_graph_definitions[i].in_use) {
            target = i;
            break;
        }
    }
    if (target < 0) {
        target = g_graph_next_replace;
        g_graph_next_replace = (g_graph_next_replace + 1) % PTO2_GRAPH_DEFINITION_CAP;
    }
    if (target < 0) return false;
    if (!compact_graph_definition(&g_graph_definitions[target], templ)) return false;
    return true;
}

void graph_record_begin_task(PTO2TaskId task_id) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    int32_t idx = static_cast<int32_t>(task_id.local()) - g_graph_recording.start_local_task_id;
    if (idx < 0 || idx >= PTO2_GRAPH_MAX_TASKS || idx != g_graph_recording.temp.task_count) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_TASK_WINDOW, idx);
        return;
    }
    g_graph_recording.current_task_index = idx;
    g_graph_recording.current_fanin_count = 0;
}

void graph_record_note_fanin(PTO2TaskSlotState *producer) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    if (producer == nullptr || producer->task == nullptr) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_NULL_PRODUCER, g_graph_recording.current_task_index);
        return;
    }
    int32_t producer_local = static_cast<int32_t>(producer->task->task_id.local());
    int32_t producer_index = producer_local - g_graph_recording.start_local_task_id;
    if (producer_index >= g_graph_recording.current_task_index) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_EXTERNAL_PRODUCER, g_graph_recording.current_task_index);
        return;
    }
    if (g_graph_recording.current_fanin_count >= PTO2_GRAPH_MAX_FANIN_PER_TASK) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_FANIN_OVERFLOW, g_graph_recording.current_task_index);
        return;
    }
    PTO2GraphFaninRef &fanin = g_graph_recording.current_fanins[g_graph_recording.current_fanin_count++];
    if (producer_index >= 0) {
        fanin.source = PTO2GraphFaninSource::INTERNAL;
        fanin.value = producer_index;
    } else {
        fanin.source = PTO2GraphFaninSource::EXTERNAL_LOCAL_DELTA;
        fanin.value = producer_local - g_graph_recording.start_local_task_id;
    }
}

}  // namespace

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        memset(
            orch->fanin_seen_epoch, 0, static_cast<size_t>(orch->sm_header->ring.task_window_size) * sizeof(uint32_t)
        );
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2OrchestratorState *orch, PTO2FaninPool &spill_pool, uint32_t seen_epoch) :
        count(0),
        spill_start(0),
        orch(orch),
        seen_epoch(seen_epoch),
        spill_pool(spill_pool) {}
    int32_t count{0};
    int32_t spill_start{0};
    PTO2OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    PTO2FaninPool &spill_pool;
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP];

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool mark_seen(uint8_t prod_ring, int32_t prod_slot) {
        if (prod_ring >= PTO2_MAX_RING_DEPTH || prod_slot < 0) {
            return false;
        }
        uint32_t *seen = orch->fanin_seen_epoch;
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }
};

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, uint8_t prod_ring, int32_t prod_slot, PTO2TaskSlotState *prod_state,
    PTO2TaskId producer_task_id, PTO2FaninBuilder *fanin_builder
) {
    // Decide-and-claim under the producer's fanout_lock. Two conditions make this
    // resolved slot a non-dependency, and both must be checked together with the
    // fanout_count++ so the producer cannot slip from live to consumed/reused in
    // between:
    //   (1) Generation mismatch — the producer was CONSUMED, its slot
    //       reset_for_reuse'd and rebound to a newer task. The cached
    //       owner_task_id still resolves to this slot, but it no longer holds our
    //       producer; ++'ing it would corrupt an unrelated task.
    //   (2) Already CONSUMED in place — finished, output ready, no real edge.
    // In either case, adding it to the fanin and bumping fanout_count would leave
    // a stale ++/release pair (Orch-side wiring drops the fanout edge but keeps
    // the fanin slot, so on_task_release still release_producer()'s it) that
    // desyncs the slot's refcount (rc != fc) and wedges in-order reclaim. Claiming a live
    // producer under the lock pins it: fanout_count now counts us, so it cannot
    // reach CONSUMED (rc == fc) until we release it in on_task_release, keeping the
    // slot's generation stable until then. check_and_handle_consumed flips
    // COMPLETED->CONSUMED under the same lock, so the check and the ++ are atomic
    // against the consume. fanout_count is lock-protected per the
    // PTO2TaskSlotState contract.
    //
    // Dedup (mark_seen) happens HERE, gated on a live producer — NOT before the
    // gone check. mark_seen keys only on (ring, slot); a stale owner that resolves
    // to a reused slot must not record it as seen, or a later dependency on the
    // live generation in the same submission would hit mark_seen and be skipped
    // without claiming it (dropped edge). Marking only when !gone keeps the dedup
    // keyed to the live producer, and doing it before the ++ still suppresses a
    // double-count for a producer named twice in one submission.
    prod_state->lock_fanout();
    bool gone = prod_state->task == nullptr || prod_state->task->task_id.local() != producer_task_id.local() ||
                prod_state->task_state.load(std::memory_order_acquire) == PTO2_TASK_CONSUMED;
    bool claim = !gone && !fanin_builder->mark_seen(prod_ring, prod_slot);
    if (claim) {
        // Low bits hold the consumer count; bit31 is the scope ref. The consumer
        // count must never carry into bit31 (would corrupt the scope-release
        // flag) — true for any sane fanout (<< 2^31).
        assert(
            (prod_state->fanout_count & ~PTO2_FANOUT_SCOPE_BIT) < (PTO2_FANOUT_SCOPE_BIT - 1) &&
            "fanout consumer count overflow into scope bit"
        );
        prod_state->fanout_count++;
    }
    prod_state->unlock_fanout();
#if SIMPLER_ORCH_PROFILING
    // lock + unlock always; one fanout_count store when we actually claim.
    g_orch_args_atomic_count += claim ? 3 : 2;
#endif
    // gone (stale/consumed) or an already-seen duplicate live producer: no new
    // fanin edge either way.
    if (!claim) {
        return true;
    }

    graph_record_note_fanin(prod_state);

    if (fanin_builder->count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder->inline_slots[fanin_builder->count++] = prod_state;
        return true;
    }

    PTO2FaninPool &fanin_pool = fanin_builder->spill_pool;
    if (!fanin_pool.ensure_space(orch->sm_header->ring, 1)) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    int32_t spill_idx = fanin_pool.top;
    PTO2FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    if (fanin_builder->count == PTO2_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    entry->slot_state = prod_state;
    fanin_builder->count++;
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

static bool all_claimed_fanin_completed(const PTO2FaninBuilder &fanin_builder) {
    if (fanin_builder.count == 0) return true;
    return fanin_builder.for_each([](PTO2TaskSlotState *producer) -> bool {
        return producer != nullptr && producer->task_state.load(std::memory_order_acquire) >= PTO2_TASK_COMPLETED;
    });
}

static bool all_claimed_fanin_allow_early_resolve(const PTO2FaninBuilder &fanin_builder) {
    if (fanin_builder.count == 0) return true;
    return fanin_builder.for_each([](PTO2TaskSlotState *producer) -> bool {
        return producer != nullptr && producer->allow_early_resolve;
    });
}

// Record the dep_pool top reached after this slot's Orch-side wiring, so the
// scheduler can bound its reclaim scan to the entries this task allocated.
static void orch_mark_dep_pool_position(PTO2OrchestratorState *orch, PTO2TaskSlotState &slot_state) {
    auto &rss = orch->scheduler->ring_sched_state;
    slot_state.dep_pool_mark = rss.dep_pool.top;
#if SIMPLER_DFX
    if (is_scope_stats_enabled()) {
        rss.publish_dep_pool_snapshot();
    }
#endif
}

// Wire a task with at least one live producer: prepend a fanout link on every
// live producer under its fanout_lock, seed dispatch_fanin for pre-completed
// codegen-flagged producers, then publish readiness via route_ready_once once
// the fanin refcount is already satisfied.
static void orch_wire_fanin_task(PTO2OrchestratorState *orch, PTO2TaskSlotState &slot_state, int32_t wfanin) {
    PTO2SchedulerState *sched = orch->scheduler;
    auto &rss = sched->ring_sched_state;
    PTO2TaskPayload *payload = slot_state.payload;
    slot_state.fanin_count = wfanin + 1;

    int32_t early_finished = 0;
    bool early_disqualified = false;
    for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer) {
        producer->lock_fanout();
        int32_t pstate = producer->task_state.load(std::memory_order_acquire);
        if (!early_disqualified && !producer->allow_early_resolve) {
            early_disqualified = true;
        }
        if (pstate >= PTO2_TASK_COMPLETED) {
            early_finished++;
        } else {
            producer->fanout_head = rss.dep_pool.prepend(producer->fanout_head, &slot_state);
        }
        producer->unlock_fanout();
    });

    // Pre-completed producers will not dispatch again. Seed dispatch_fanin only
    // when every producer is codegen-flagged; one unflagged producer makes the
    // direct-only early-dispatch candidate count unreachable by design.
    if (!early_disqualified && early_finished != 0) {
        payload->dispatch_fanin.fetch_add(early_finished, std::memory_order_acq_rel);
    }

    int32_t init_rc = early_finished + 1;
    int32_t new_rc = slot_state.fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
    orch_mark_dep_pool_position(orch, slot_state);
    if (new_rc >= slot_state.fanin_count) {
        sched->route_ready_once(slot_state);
    }
}

static bool orch_wire_live_fanin_task(PTO2OrchestratorState *orch, PTO2TaskSlotState &slot_state, int32_t wfanin) {
    auto &rss = orch->scheduler->ring_sched_state;

    // dep_pool is orchestrator-exclusive during host orchestration (no lock).
    // The pool is sized for the whole graph (no reclaim during host
    // orchestration, exactly like the task window and GM heap), so an exhausted
    // pool latches PTO2_ERROR_DEP_POOL_OVERFLOW and reports the deadlock with the
    // same structural + wall-clock logic the heap/task-window allocator uses. A
    // false return also covers a fatal already latched elsewhere.
    if (!rss.dep_pool.ensure_space(*rss.ring, wfanin)) {
        orch->fatal = true;
        return false;
    }

    orch_wire_fanin_task(orch, slot_state, wfanin);
    return true;
}

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

namespace {

static bool graph_tensor_metadata_equal(const Tensor &lhs, const Tensor &rhs) {
    if (lhs.buffer.addr != rhs.buffer.addr || lhs.buffer.size != rhs.buffer.size ||
        lhs.start_offset != rhs.start_offset || lhs.version != rhs.version || lhs.ndims != rhs.ndims ||
        lhs.dtype != rhs.dtype || lhs.manual_dep != rhs.manual_dep || lhs.is_contiguous != rhs.is_contiguous ||
        lhs.child_memory != rhs.child_memory) {
        return false;
    }
    return memcmp(lhs.shapes, rhs.shapes, sizeof(uint32_t) * lhs.ndims) == 0 &&
           memcmp(lhs.strides, rhs.strides, sizeof(uint32_t) * lhs.ndims) == 0;
}

static bool
graph_tensor_from_args(const Tensor &tensor, const ChipStorageTaskArgs &args, PTO2GraphTensorSourceRef *source_ref) {
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        if (graph_tensor_metadata_equal(tensor, args.tensor(i))) {
            source_ref->source = PTO2GraphTensorSource::BOUNDARY_EXACT;
            source_ref->source_index = static_cast<uint16_t>(i);
            source_ref->packed_offset = 0;
            return true;
        }
    }
    uint16_t best_index = UINT16_MAX;
    uint64_t best_offset = UINT64_MAX;
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        const Tensor &boundary = args.tensor(i);
        if (tensor.buffer.addr == boundary.buffer.addr && tensor.buffer.size == boundary.buffer.size &&
            tensor.start_offset >= boundary.start_offset) {
            uint64_t offset = tensor.start_offset - boundary.start_offset;
            if (offset < best_offset) {
                best_index = static_cast<uint16_t>(i);
                best_offset = offset;
            }
        }
    }
    if (best_index == UINT16_MAX) return false;
    source_ref->source = PTO2GraphTensorSource::BOUNDARY_VIEW;
    source_ref->source_index = best_index;
    source_ref->packed_offset = best_offset;
    return true;
}

static bool
graph_classify_tensor(PTO2GraphRecordedNode *task, int32_t tensor_index, const Tensor &tensor, int32_t task_index) {
    PTO2GraphTensorSourceRef &source_ref = task->tensor_sources[tensor_index];
    if (graph_tensor_from_args(tensor, g_graph_recording.args, &source_ref)) return true;

    uint64_t tensor_addr = tensor.buffer.addr;
    for (int32_t producer_index = task_index; producer_index >= 0; --producer_index) {
        const PTO2GraphRecordedNode &producer =
            (producer_index == task_index) ? *task : g_graph_recording.temp.tasks[producer_index];
        if (producer.record_packed_base == 0 || producer.total_output_size <= 0) continue;
        uint64_t producer_begin = producer.record_packed_base;
        uint64_t producer_end = producer_begin + static_cast<uint64_t>(producer.total_output_size);
        if (tensor_addr < producer_begin || tensor_addr >= producer_end) continue;
        source_ref.source =
            producer_index == task_index ? PTO2GraphTensorSource::OWN_OUTPUT : PTO2GraphTensorSource::INTERNAL;
        source_ref.source_index = static_cast<uint16_t>(producer_index);
        source_ref.packed_offset = tensor_addr - producer_begin;
        return true;
    }
    return false;
}

static void graph_record_task(const PTO2PreparedTask &prepared, const L0TaskArgs &args) {
    if (!g_graph_recording.active || g_graph_recording.unsupported) return;
    int32_t task_index = static_cast<int32_t>(prepared.task_id.local()) - g_graph_recording.start_local_task_id;
    if (task_index < 0 || task_index >= PTO2_GRAPH_MAX_TASKS || task_index != g_graph_recording.temp.task_count) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_ORDER, task_index);
        return;
    }
    if (prepared.task == nullptr || prepared.payload == nullptr || prepared.slot_state == nullptr) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_RECORD_TASK_NULL, task_index);
        return;
    }

    PTO2GraphRecordedNode &task = g_graph_recording.temp.tasks[task_index];
    task = PTO2GraphRecordedNode{};
    for (int i = 0; i < PTO2_SUBTASK_SLOT_COUNT; ++i)
        task.kernel_id[i] = prepared.task->kernel_id[i];
    task.active_mask = prepared.slot_state->active_mask;
    task.logical_block_num = prepared.slot_state->logical_block_num;
    task.total_required_subtasks = prepared.slot_state->total_required_subtasks;
    task.task_timing_slot = prepared.task->task_timing_slot;
    task.tensor_count = prepared.payload->tensor_count;
    task.scalar_count = prepared.payload->scalar_count;
    task.total_output_size = static_cast<int32_t>(
        reinterpret_cast<uintptr_t>(prepared.task->packed_buffer_end) -
        reinterpret_cast<uintptr_t>(prepared.task->packed_buffer_base)
    );
    task.record_packed_base = reinterpret_cast<uint64_t>(prepared.task->packed_buffer_base);
    if (task.tensor_count < 0 || task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 ||
        task.scalar_count > MAX_SCALAR_ARGS) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_ARG_OVERFLOW, task_index);
        return;
    }

    for (int32_t i = 0; i < task.tensor_count; ++i) {
        task.tensors[i].copy(prepared.payload->tensors[i]);
        task.tensor_arg_types[i] = args.tag(i);
        if (!graph_classify_tensor(&task, i, task.tensors[i], task_index)) {
            graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_TENSOR_SOURCE, task_index, i);
            return;
        }
    }
    memcpy(task.scalars, prepared.payload->scalars, sizeof(uint64_t) * static_cast<size_t>(task.scalar_count));
    for (int32_t i = 0; i < task.scalar_count; ++i)
        task.scalar_source_indices[i] = args.scalar_source_index(i);
    for (uint32_t i = 0; i < args.explicit_dep_count(); ++i) {
        PTO2TaskId dependency = args.explicit_dep(i);
        int32_t dependency_index = static_cast<int32_t>(dependency.local()) - g_graph_recording.start_local_task_id;
        if (!dependency.is_valid() || dependency.ring() != 0 || dependency_index < 0 ||
            dependency_index >= task_index) {
            graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_EXTERNAL_EXPLICIT_DEP, task_index);
            return;
        }
    }
    if (args.predicate().op != PredicateOp::NONE) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_PREDICATE, task_index);
        return;
    }
    task.fanin_count = g_graph_recording.current_fanin_count;
    for (int32_t i = 0; i < task.fanin_count; ++i)
        task.fanins[i] = g_graph_recording.current_fanins[i];

    g_graph_recording.temp.task_count++;
    g_graph_recording.current_task_index = -1;
}

static void graph_reset_payload(PTO2TaskPayload *payload) {
    payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
    for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; ++w) {
        payload->staged_core_mask[w].store(0, std::memory_order_relaxed);
    }
    payload->dispatch_fanin.store(0, std::memory_order_relaxed);
    payload->dispatch_propagated.store(0, std::memory_order_relaxed);
    payload->published_block_count.store(0, std::memory_order_relaxed);
    payload->early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
    payload->running_slot_count.store(0, std::memory_order_relaxed);
    payload->early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
}

static TensorArgType graph_boundary_type(bool reads, bool writes, bool no_dep) {
    if (reads && writes) return TensorArgType::INOUT;
    if (writes) return TensorArgType::OUTPUT_EXISTING;
    if (reads) return TensorArgType::INPUT;
    return no_dep ? TensorArgType::NO_DEP : TensorArgType::INPUT;
}

static bool graph_build_definition(PTO2GraphRecordingDefinition *templ, const ChipStorageTaskArgs &args) {
    if (templ == nullptr || templ->task_count <= 0 || templ->task_count > PTO2_GRAPH_MAX_TASKS) return false;
    PTO2GraphReplayPlan plan;
    bool boundary_seen[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
    bool boundary_reads[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
    bool boundary_writes[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
    bool boundary_no_dep[PTO2_GRAPH_MAX_TENSOR_ARGS]{};
    uint32_t fanout_counts[PTO2_GRAPH_MAX_TASKS]{};

    for (int32_t i = 0; i < templ->task_count; ++i) {
        const PTO2GraphRecordedNode &task = templ->tasks[i];
        if (task.tensor_count < 0 || task.tensor_count > MAX_TENSOR_ARGS || task.scalar_count < 0 ||
            task.scalar_count > MAX_SCALAR_ARGS || task.fanin_count < 0 ||
            task.fanin_count > PTO2_GRAPH_MAX_FANIN_PER_TASK || task.total_output_size < 0) {
            return false;
        }
        templ->node_offsets[i] = plan.required_heap;
        uint64_t aligned_output = PTO2_ALIGN_UP(static_cast<uint64_t>(task.total_output_size), PTO2_ALIGN_SIZE);
        if (plan.required_heap > UINT64_MAX - aligned_output) return false;
        plan.required_heap += aligned_output;

        int32_t internal_fanin_count = 0;
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const PTO2GraphFaninRef &fanin = task.fanins[e];
            if (fanin.source != PTO2GraphFaninSource::INTERNAL) continue;
            if (fanin.value < 0 || fanin.value >= i) return false;
            if (templ->edge_count >= PTO2_GRAPH_MAX_INTERNAL_EDGES) return false;
            fanout_counts[fanin.value]++;
            templ->edge_count++;
            internal_fanin_count++;
        }
        templ->fanin_counts[i] = static_cast<uint16_t>(internal_fanin_count);
        if (internal_fanin_count == 0) {
            templ->root_indices[templ->root_count++] = static_cast<uint16_t>(i);
        }

        for (int32_t j = 0; j < task.tensor_count; ++j) {
            const PTO2GraphTensorSourceRef &source_ref = task.tensor_sources[j];
            bool is_boundary = source_ref.source == PTO2GraphTensorSource::BOUNDARY_EXACT ||
                               source_ref.source == PTO2GraphTensorSource::BOUNDARY_VIEW;
            if (is_boundary && source_ref.source_index >= args.tensor_count()) return false;
            if (source_ref.source == PTO2GraphTensorSource::INTERNAL && source_ref.source_index >= i) return false;
            if (is_boundary) {
                uint16_t index = source_ref.source_index;
                uint32_t required = static_cast<uint32_t>(index) + 1;
                if (required > plan.required_tensor_count) plan.required_tensor_count = required;
                TensorArgType type = task.tensor_arg_types[j];
                boundary_seen[index] = true;
                boundary_reads[index] |= type == TensorArgType::INPUT || type == TensorArgType::INOUT;
                boundary_writes[index] |= type == TensorArgType::INOUT || type == TensorArgType::OUTPUT_EXISTING;
                boundary_no_dep[index] |= type == TensorArgType::NO_DEP;
            }
        }
        for (int32_t j = 0; j < task.scalar_count; ++j) {
            uint16_t source_index = task.scalar_source_indices[j];
            if (source_index != PTO2_TASK_ARG_STATIC && source_index >= args.scalar_count()) return false;
            if (source_index != PTO2_TASK_ARG_STATIC) {
                uint32_t required = static_cast<uint32_t>(source_index) + 1;
                if (required > plan.required_scalar_count) plan.required_scalar_count = required;
            }
        }
    }

    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        if (!boundary_seen[i]) continue;
        TensorArgType type = graph_boundary_type(boundary_reads[i], boundary_writes[i], boundary_no_dep[i]);
        plan.boundary_indices[plan.boundary_count] = static_cast<uint16_t>(i);
        plan.boundary_types[plan.boundary_count] = type;
        plan.boundary_count++;
    }

    templ->fanout_offsets[0] = 0;
    for (int32_t i = 0; i < templ->task_count; ++i) {
        templ->fanout_offsets[i + 1] = templ->fanout_offsets[i] + fanout_counts[i];
    }
    uint32_t fanout_cursors[PTO2_GRAPH_MAX_TASKS];
    memcpy(fanout_cursors, templ->fanout_offsets, sizeof(uint32_t) * static_cast<size_t>(templ->task_count));
    for (int32_t consumer_index = 0; consumer_index < templ->task_count; ++consumer_index) {
        const PTO2GraphRecordedNode &task = templ->tasks[consumer_index];
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const PTO2GraphFaninRef &fanin = task.fanins[e];
            if (fanin.source != PTO2GraphFaninSource::INTERNAL) continue;
            templ->fanout_indices[fanout_cursors[fanin.value]++] = static_cast<uint16_t>(consumer_index);
        }
    }

    templ->replay_plan = plan;
    return templ->fanout_offsets[templ->task_count] == static_cast<uint32_t>(templ->edge_count);
}

static bool graph_submission_preflight(
    PTO2OrchestratorState *orch, const PTO2GraphDefinition &templ, const ChipStorageTaskArgs &args
) {
    const PTO2GraphReplayPlan &plan = templ.replay_plan;
    if (args.tensor_count() < static_cast<int32_t>(plan.required_tensor_count) ||
        args.scalar_count() < static_cast<int32_t>(plan.required_scalar_count)) {
        return false;
    }
    auto &allocator = orch->ring.task_allocator;
    if (allocator.active_count() + 1 >= allocator.window_size()) return false;
    if (plan.required_heap > allocator.heap_available() || plan.required_heap > INT32_MAX) return false;
    int32_t tensormap_entries = 0;
    for (int32_t i = 0; i < plan.boundary_count; ++i) {
        uint16_t arg_index = plan.boundary_indices[i];
        if (arg_index >= args.tensor_count()) return false;
        TensorArgType type = plan.boundary_types[i];
        if ((type == TensorArgType::INOUT || type == TensorArgType::OUTPUT_EXISTING) &&
            !args.tensor(arg_index).manual_dep) {
            tensormap_entries++;
        }
    }
    if (tensormap_entries > orch->tensor_map.free_entries()) return false;
    return true;
}

static size_t graph_definition_image_bytes(const PTO2GraphDefinition &templ) {
    size_t header_bytes = PTO2_ALIGN_UP(sizeof(PTO2GraphDefinition), alignof(Tensor));
    if (templ.storage_bytes > SIZE_MAX - header_bytes) return 0;
    return header_bytes + templ.storage_bytes;
}

template <typename T>
static bool graph_rebase_cloned_pointer(T *&ptr, uintptr_t source_begin, size_t source_bytes, intptr_t delta) {
    if (ptr == nullptr) return true;
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    if (value < source_begin || value >= source_begin + source_bytes) return false;
    ptr = reinterpret_cast<T *>(static_cast<intptr_t>(value) + delta);
    return true;
}

static PTO2GraphDefinition *
graph_clone_definition_image(void *definition_storage, size_t definition_capacity, const PTO2GraphDefinition &templ) {
    if (definition_storage == nullptr || templ.storage == nullptr) return nullptr;
    const size_t header_bytes = PTO2_ALIGN_UP(sizeof(PTO2GraphDefinition), alignof(Tensor));
    if (definition_capacity < header_bytes + templ.storage_bytes) return nullptr;

    auto *copy = new (definition_storage) PTO2GraphDefinition(templ);
    void *copy_storage = static_cast<char *>(definition_storage) + header_bytes;
    memcpy(copy_storage, templ.storage, templ.storage_bytes);
    const uintptr_t source_begin = reinterpret_cast<uintptr_t>(templ.storage);
    const intptr_t delta =
        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(copy_storage)) - static_cast<intptr_t>(source_begin);
    copy->storage = copy_storage;
    if (!graph_rebase_cloned_pointer(copy->fanout_offsets, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->fanout_indices, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->fanin_counts, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->root_indices, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->node_offsets, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->tasks, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->tensors, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->tensor_sources, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->scalars, source_begin, templ.storage_bytes, delta) ||
        !graph_rebase_cloned_pointer(copy->scalar_source_indices, source_begin, templ.storage_bytes, delta)) {
        return nullptr;
    }
    return copy;
}

static bool graph_clone_definition(PTO2GraphSubmission *submission, const PTO2GraphDefinition &templ) {
    if (submission == nullptr) return false;
    submission->graph_definition =
        graph_clone_definition_image(submission->definition_storage, submission->definition_capacity, templ);
    return submission->graph_definition != nullptr;
}

static bool graph_clone_definition(PTO2GraphExecution *execution, const PTO2GraphDefinition &templ) {
    if (execution == nullptr) return false;
    auto *copy = graph_clone_definition_image(execution->definition_storage, execution->definition_capacity, templ);
    if (copy == nullptr) return false;
    execution->graph_definition = copy;
    execution->topology.edge_count = copy->edge_count;
    execution->topology.root_count = copy->root_count;
    execution->topology.fanout_offsets = copy->fanout_offsets;
    execution->topology.fanout_indices = copy->fanout_indices;
    execution->topology.fanin_counts = copy->fanin_counts;
    execution->topology.root_indices = copy->root_indices;
    return true;
}

static bool graph_submit_definition(
    PTO2OrchestratorState *orch, const PTO2GraphDefinition &templ, const ChipStorageTaskArgs &args,
    uint64_t *orch_record_task_id
) {
    if (orch_record_task_id != nullptr) *orch_record_task_id = PTO2TaskId::invalid().raw;
    if (!graph_submission_preflight(orch, templ, args)) return false;

    const size_t definition_bytes = graph_definition_image_bytes(templ);
    if (definition_bytes == 0) return false;
    PTO2GraphSubmission *submission = pto2_graph_submission_create(templ.task_count, templ.full_key, definition_bytes);
    if (submission == nullptr || !graph_clone_definition(submission, templ)) {
        pto2_graph_submission_discard(submission);
        return false;
    }
    submission->args = args;

    const PTO2GraphReplayPlan &plan = templ.replay_plan;
    PTO2TaskAllocResult outer_alloc = orch->ring.task_allocator.alloc(static_cast<int32_t>(plan.required_heap));
    if (outer_alloc.failed()) {
        pto2_graph_submission_discard(submission);
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    PTO2TaskId outer_task_id = PTO2TaskId::make(0, static_cast<uint32_t>(outer_alloc.task_id));
    PTO2SharedMemoryRingHeader &ring = orch->sm_header->ring;
    PTO2TaskDescriptor &outer_task = ring.task_descriptors[outer_alloc.slot];
    PTO2TaskPayload &outer_payload = ring.task_payloads[outer_alloc.slot];
    PTO2TaskSlotState &outer_slot = ring.get_slot_state_by_slot(outer_alloc.slot);
    outer_task.task_id = outer_task_id;
    for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
        outer_task.kernel_id[k] = INVALID_KERNEL_ID;
    outer_task.task_timing_slot = TASK_TIMING_SLOT_NONE;
    outer_task.packed_buffer_base = outer_alloc.packed_base;
    outer_task.packed_buffer_end = outer_alloc.packed_end;
    outer_task.graph_execution = submission;
    outer_task.graph_node_index = -1;
    outer_task.kind = PTO2TaskKind::GRAPH;
    outer_payload.tensor_count = 0;
    outer_payload.scalar_count = 0;
    outer_payload.fanin_actual_count = 0;
    outer_payload.fanin_spill_start = 0;
    outer_payload.fanin_spill_pool = &orch->ring.fanin_pool;
    outer_payload.predicate = DispatchPredicate{};
    graph_reset_payload(&outer_payload);
    outer_slot.bind_buffers(&outer_payload, &outer_task);
    outer_slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    outer_slot.active_mask = ActiveMask{};
    outer_slot.set_allow_early_resolve(false);
    outer_slot.total_required_subtasks = 0;
    outer_slot.logical_block_num = 1;
    scope_tasks_push(orch, &outer_slot);

    submission->outer_slot = &outer_slot;

    TensorRef boundary_tensors[PTO2_GRAPH_MAX_TENSOR_ARGS];
    TensorArgType boundary_types[PTO2_GRAPH_MAX_TENSOR_ARGS];
    for (int32_t i = 0; i < plan.boundary_count; ++i) {
        boundary_tensors[i] = &args.tensor(plan.boundary_indices[i]);
        boundary_types[i] = plan.boundary_types[i];
    }
    DepInputs boundary_inputs{plan.boundary_count, boundary_tensors, boundary_types, 0, nullptr};
    PTO2FaninBuilder fanin_builder(orch, orch->ring.fanin_pool, next_fanin_seen_epoch(orch));
    orch->tensor_map.sync_tensormap(outer_task_id, ring.fc.last_task_alive.load(std::memory_order_acquire));
    auto boundary_emit = [&](PTO2TaskId producer_task_id) -> bool {
        uint8_t producer_ring_id = producer_task_id.ring();
        int32_t producer_local = static_cast<int32_t>(producer_task_id.local());
        int32_t producer_slot = ring.get_slot_by_task_id(producer_local);
        PTO2TaskSlotState *producer = &ring.get_slot_state_by_slot(producer_slot);
        return append_fanin_or_fail(orch, producer_ring_id, producer_slot, producer, producer_task_id, &fanin_builder);
    };
    if (!compute_task_fanin(boundary_inputs, orch->tensor_map, orch->in_manual_scope(), boundary_emit)) {
        pto2_graph_submission_discard(submission);
        return false;
    }
    register_task_outputs(boundary_inputs, outer_task_id, orch->tensor_map, orch->in_manual_scope());

    outer_payload.fanin_actual_count = fanin_builder.count;
    outer_payload.fanin_spill_start = fanin_builder.spill_start;
    outer_payload.fanin_spill_pool = &fanin_builder.spill_pool;
    const int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    for (int32_t i = 0; i < inline_count; ++i)
        outer_payload.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];

    if (fanin_builder.count == 0) {
        outer_slot.fanin_count = 1;
        outer_slot.fanin_refcount.store(1, std::memory_order_release);
        orch_mark_dep_pool_position(orch, outer_slot);
        orch->scheduler->push_ready_routed(&outer_slot);
    } else if (all_claimed_fanin_completed(fanin_builder)) {
        int32_t ready_seed = fanin_builder.count + 1;
        outer_slot.fanin_count = ready_seed;
        outer_slot.fanin_refcount.store(ready_seed, std::memory_order_release);
        orch_mark_dep_pool_position(orch, outer_slot);
        orch->scheduler->push_ready_routed(&outer_slot);
    } else if (!orch_wire_live_fanin_task(orch, outer_slot, fanin_builder.count)) {
        pto2_graph_submission_discard(submission);
        return false;
    }

    if (orch_record_task_id != nullptr) *orch_record_task_id = outer_task_id.raw;
    if (!orch->scheduler->graph_prepare_queue.push_tagged(&outer_slot, outer_task_id.raw)) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        pto2_graph_submission_discard(submission);
        return false;
    }
    submission->upload_next = orch->pending_graph_uploads;
    orch->pending_graph_uploads = submission;
#if SIMPLER_DFX
    orch->tasks_submitted++;
#endif
    return true;
}

}  // namespace

bool pto2_graph_definition_relocate_for_upload(PTO2GraphSubmission *submission, void *device_base) {
    if (submission == nullptr || device_base == nullptr || submission->graph_definition == nullptr) return false;
    auto *definition =
        const_cast<PTO2GraphDefinition *>(static_cast<const PTO2GraphDefinition *>(submission->graph_definition));
    const uintptr_t host_begin = reinterpret_cast<uintptr_t>(submission);
    const intptr_t delta =
        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(device_base)) - static_cast<intptr_t>(host_begin);
    const size_t bytes = submission->allocation_bytes;
    return graph_rebase_cloned_pointer(definition->storage, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->fanout_offsets, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->fanout_indices, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->fanin_counts, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->root_indices, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->node_offsets, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->tasks, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->tensors, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->tensor_sources, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->scalars, host_begin, bytes, delta) &&
           graph_rebase_cloned_pointer(definition->scalar_source_indices, host_begin, bytes, delta);
}

PTO2GraphExecution *pto2_graph_execution_localize(PTO2TaskSlotState &outer_slot) {
    if (outer_slot.task == nullptr || outer_slot.task->kind != PTO2TaskKind::GRAPH) return nullptr;
    PTO2GraphSubmission *submission = pto2_graph_submission_from_task(*outer_slot.task);
    if (submission == nullptr || submission->graph_definition == nullptr) return nullptr;

    PTO2GraphExecution *execution = submission->local_execution.load(std::memory_order_acquire);
    if (execution != nullptr) return execution;

    // Reclaim completed local blocks before allocating.  This pool is owned by
    // the Scheduler DSO and survives run boundaries, so later invocations can
    // reuse both the allocation and graph-affine static node contents.
    pto2_graph_execution_collect_retired();
    const auto &templ = *static_cast<const PTO2GraphDefinition *>(submission->graph_definition);
    const size_t definition_bytes = graph_definition_image_bytes(templ);
    execution = pto2_graph_execution_create(submission->node_count, submission->graph_key, definition_bytes);
    if (execution == nullptr) return nullptr;

    if (!execution->definition_affine_reuse && !graph_clone_definition(execution, templ)) {
        pto2_graph_execution_discard(execution);
        return nullptr;
    }
    execution->args = submission->args;
    execution->outer_slot = &outer_slot;

    PTO2GraphExecution *expected = nullptr;
    if (!submission->local_execution.compare_exchange_strong(
            expected, execution, std::memory_order_release, std::memory_order_acquire
        )) {
        pto2_graph_execution_discard(execution);
        return expected;
    }
    pto2_graph_execution_publish(execution);
    return execution;
}

bool PTO2OrchestratorState::finalize_pending_graph_definition() {
    if (!g_graph_recording.pending_finalize) return true;

    bool built = graph_build_definition(&g_graph_recording.temp, g_graph_recording.args);
    bool stored = built && store_graph_definition(g_graph_recording.temp);
    if (stored) {
        PTO2GraphDefinition *definition = find_graph_definition(g_graph_recording.full_key);
        LOG_INFO_V0(
            "[GraphExecution] define key=0x%llx nodes=%d bytes=%zu",
            static_cast<unsigned long long>(g_graph_recording.full_key), g_graph_recording.temp.task_count,
            definition == nullptr ? 0 : definition->storage_bytes
        );
    }
    reset_graph_recording();
    return stored;
}

PTO2GraphMaterializeResult pto2_graph_execution_materialize_slice(
    PTO2TaskSlotState &outer_slot, PTO2GraphExecution &execution_ref, int32_t max_nodes, int32_t *nodes_materialized
) {
    if (nodes_materialized != nullptr) *nodes_materialized = 0;
    if (outer_slot.task == nullptr || outer_slot.task->kind != PTO2TaskKind::GRAPH || max_nodes <= 0) {
        return PTO2GraphMaterializeResult::INVALID;
    }
    PTO2GraphExecution *execution = &execution_ref;
    if (execution == nullptr || execution->graph_definition == nullptr || execution->node_storage == nullptr) {
        return PTO2GraphMaterializeResult::INVALID;
    }

    PTO2GraphExecutionState state = execution->state.load(std::memory_order_acquire);
    if (state >= PTO2GraphExecutionState::PREPARED) return PTO2GraphMaterializeResult::PREPARED;

    uint8_t expected_busy = 0;
    if (!execution->materialize_busy.compare_exchange_strong(
            expected_busy, 1, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return PTO2GraphMaterializeResult::BUSY;
    }

    state = execution->state.load(std::memory_order_acquire);
    if (state == PTO2GraphExecutionState::SUBMITTED) {
        if (!pto2_graph_execution_begin_materialize(*execution)) {
            execution->materialize_busy.store(0, std::memory_order_release);
            state = execution->state.load(std::memory_order_acquire);
            return state >= PTO2GraphExecutionState::PREPARED ? PTO2GraphMaterializeResult::PREPARED :
                                                                PTO2GraphMaterializeResult::BUSY;
        }
    } else if (state != PTO2GraphExecutionState::MATERIALIZING) {
        execution->materialize_busy.store(0, std::memory_order_release);
        return state >= PTO2GraphExecutionState::PREPARED ? PTO2GraphMaterializeResult::PREPARED :
                                                            PTO2GraphMaterializeResult::INVALID;
    }

    const auto &templ = *static_cast<const PTO2GraphDefinition *>(execution->graph_definition);
    const ChipStorageTaskArgs &args = execution->args;
    const bool reuse_static = execution->definition_affine_reuse;
    const int32_t first_node = execution->materialized_nodes;
    if (first_node == 0 && !reuse_static) {
        execution->materialized_graph_key = 0;
        execution->materialized_node_count = 0;
    }
    const int32_t last_node = std::min(templ.task_count, first_node + max_nodes);
    for (int32_t i = first_node; i < last_node; ++i) {
        const PTO2GraphNodeDefinition &src = templ.tasks[i];
        PTO2GraphNodeStorage *node = &execution->node_storage[i];
        if (i >= execution->constructed_nodes) {
            // Default-initialize object lifetimes without value-initializing
            // the 4 KiB payload arrays. Every live descriptor/slot field and
            // every tensor/scalar below the published counts is overwritten
            // before PREPARED is release-published.
            node = new (node) PTO2GraphNodeStorage;
            execution->constructed_nodes++;
        }
        execution->materialized_nodes++;
        PTO2TaskDescriptor &task = node->task;
        PTO2TaskPayload &payload = node->payload;
        PTO2TaskSlotState &slot = node->slot;

        uint32_t synthetic_local =
            (static_cast<uint32_t>(outer_slot.task->task_id.local()) << 10) | static_cast<uint32_t>(i);
        task.task_id = PTO2TaskId::make(1, synthetic_local);
        task.packed_buffer_base = static_cast<char *>(outer_slot.task->packed_buffer_base) + templ.node_offsets[i];
        task.packed_buffer_end = static_cast<char *>(task.packed_buffer_base) +
                                 PTO2_ALIGN_UP(static_cast<uint64_t>(src.total_output_size), PTO2_ALIGN_SIZE);
        if (!reuse_static) {
            for (int k = 0; k < PTO2_SUBTASK_SLOT_COUNT; ++k)
                task.kernel_id[k] = src.kernel_id[k];
            task.task_timing_slot = src.task_timing_slot;
            task.graph_execution = execution;
            task.graph_node_index = i;
            task.kind = PTO2TaskKind::GRAPH_NODE;
        }

        slot.reset_for_reuse();
        if (!reuse_static) slot.bind_buffers(&payload, &task);
        slot.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        slot.fanin_refcount.store(1, std::memory_order_relaxed);
        if (!reuse_static) {
            slot.fanin_count = static_cast<int32_t>(templ.fanin_counts[i]) + 1;
            slot.active_mask = src.active_mask;
            // Graph topology is released directly, so ordinary fanout-based
            // early propagation is intentionally disabled for Graph nodes.
            slot.set_allow_early_resolve(false);
            slot.total_required_subtasks = src.total_required_subtasks;
            slot.logical_block_num = src.logical_block_num;
        }

        if (!reuse_static) {
            payload.tensor_count = src.tensor_count;
            payload.scalar_count = src.scalar_count;
        }
        payload.fanin_actual_count = 0;
        payload.fanin_spill_start = 0;
        payload.fanin_spill_pool = nullptr;
        payload.predicate = DispatchPredicate{};
        for (int32_t j = 0; j < src.tensor_count; ++j) {
            const uint32_t tensor_index = src.tensor_offset + static_cast<uint32_t>(j);
            Tensor &tensor = payload.tensors[j];
            if (!reuse_static) tensor.copy(templ.tensors[tensor_index]);
            const PTO2GraphTensorSourceRef &source_ref = templ.tensor_sources[tensor_index];
            if (source_ref.source == PTO2GraphTensorSource::BOUNDARY_EXACT) {
                tensor.copy(args.tensor(source_ref.source_index));
            } else if (source_ref.source == PTO2GraphTensorSource::BOUNDARY_VIEW) {
                const Tensor &boundary = args.tensor(source_ref.source_index);
                tensor.buffer = boundary.buffer;
                tensor.owner_task_id = boundary.owner_task_id;
                tensor.start_offset = boundary.start_offset + source_ref.packed_offset;
                tensor.version = boundary.version;
                tensor.child_memory = boundary.child_memory;
            } else {
                int32_t producer_index = source_ref.source == PTO2GraphTensorSource::OWN_OUTPUT ?
                                             i :
                                             static_cast<int32_t>(source_ref.source_index);
                PTO2TaskDescriptor &producer = execution->node_storage[producer_index].task;
                tensor.buffer.addr = reinterpret_cast<uint64_t>(producer.packed_buffer_base) + source_ref.packed_offset;
                tensor.owner_task_id = producer.task_id;
            }
        }
        for (int32_t j = 0; j < src.scalar_count; ++j) {
            const uint32_t scalar_index = src.scalar_offset + static_cast<uint32_t>(j);
            uint16_t source_index = templ.scalar_source_indices[scalar_index];
            if (!reuse_static || source_index != PTO2_TASK_ARG_STATIC) {
                payload.scalars[j] =
                    source_index == PTO2_TASK_ARG_STATIC ? templ.scalars[scalar_index] : args.scalar(source_index);
            }
        }
        graph_reset_payload(&payload);
    }
    if (nodes_materialized != nullptr) *nodes_materialized = last_node - first_node;

    if (execution->materialized_nodes < templ.task_count) {
        execution->materialize_busy.store(0, std::memory_order_release);
        return PTO2GraphMaterializeResult::PENDING;
    }

    execution->nodes = execution->node_storage;
    execution->materialized_graph_key = execution->graph_key;
    execution->materialized_node_count = execution->node_count;
    pto2_graph_execution_publish_materialized(*execution);
    execution->materialize_busy.store(0, std::memory_order_release);
    return PTO2GraphMaterializeResult::PREPARED;
}

static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args) {
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

static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
    LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
    LOG_ERROR("  no slots can be reclaimed -> deadlock.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
) {
    uint8_t ring_id = 0;
    auto &allocator = orch->ring.task_allocator;

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->ring.get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->ring.task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->ring.task_payloads[out->alloc_result.slot];

    graph_record_begin_task(out->task_id);

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(window_size) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in PTO2TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already zeroed by reset_for_reuse() at slot init:
    //   fanout_lock=0, fanout_count=PTO2_FANOUT_SCOPE_BIT, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state is set to PENDING here as the orchestrator populates the slot
    // (host_build_graph does not recycle slots at runtime, so there is no
    // post-CONSUMED reset path).
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    // fanin_count is set during Orch-side wiring
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
        // (the runtime task window; see reserve_layout) — hitting it means the
        // ring is saturated, so no further push could succeed regardless of
        // buffer growth.
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
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
        auto &alloc = orch->ring.task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_state.read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_begin(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
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
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = 0;
        auto &alloc = orch->ring.task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_state.read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_end(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
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
// registers this task's outputs. The pool is watermark-reclaimed like the
// task/heap/fanin pools — retired tasks' entries free once last_task_alive
// advances — so an exhausted pool is back-pressure, not a hard error. Reclaim
// against the single ring's watermark; if still short,
// spin until reclaim actually frees entries, with the same 500 ms wall-clock
// backstop as the task allocator and fanin spill pool. A pool that stays full
// (no entry freed) is a genuine deadlock: latch PTO2_ERROR_TENSORMAP_OVERFLOW
// and bail. Returns false on deadlock or on a fatal already latched by another
// party. Cold path — the fast path returns immediately when the pool has room.
static bool ensure_tensormap_capacity(PTO2OrchestratorState *orch, int32_t needed) {
    PTO2TensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }

    int32_t alive;
    auto read_alive = [&]() {
        // Relaxed: a self-correcting poll re-read every reclaim tick, so a stale
        // watermark only defers reclaim one tick and never over-frees.
        alive = orch->sm_header->ring.fc.last_task_alive.load(std::memory_order_relaxed);
    };

    read_alive();
    int64_t cur_alive_sum = tm.reclaim_retired_all(alive);  // kept for the deadlock diagnostic
    int32_t prev_free = tm.free_entries();
    if (prev_free >= needed) {
        return true;
    }

    int spin_count = 0;
    uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
    bool block_timing = false;  // false until the first no-reclaim-progress tick
    while (tm.free_entries() < needed) {
        spin_count++;

        // Reclaim (and the all-ring watermark reads it needs) is the costly part of
        // this spin and the only path that frees entries; gate it to a periodic tick.
        // Cold path, but the spin itself is tight.
        if ((spin_count & 31) == 0) {
            read_alive();
            cur_alive_sum = tm.reclaim_retired_all(alive);
            int32_t cur_free = tm.free_entries();
            if (cur_free >= needed) {
                return true;
            }
            // Progress is entries actually freed, NOT watermark movement: a ring can
            // retire zero-output tasks (count_registrable_outputs == 0), advancing
            // last_task_alive without freeing any entry. Gating the backstop on
            // free_entries() keeps a wedged pool from dodging the timeout while some
            // unrelated ring keeps draining.
            if (cur_free > prev_free) {
                spin_count = 0;
                prev_free = cur_free;
                block_timing = false;
            }
        }

        if ((spin_count & 1023) == 0) {
            // A fatal latched elsewhere breaks this otherwise-unbounded spin.
            if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                return false;
            }
            // Absolute-time backstop, matching the task allocator: stable across
            // chips/contention, unlike a fixed spin count. get_sys_cnt_aicpu()
            // is an MMIO read, so sample it only once per 1024 spins.
            uint64_t now = get_sys_cnt_aicpu();
            if (!block_timing) {
                block_cycle0 = now;
                block_timing = true;
            } else if (now - block_cycle0 >= PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: TensorMap Entry Pool Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("TensorMap entry pool freed no entries for ~500 ms while a task waits.");
                LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
                LOG_ERROR("  - Needed:      %d entries", needed);
                LOG_ERROR("  - last_task_alive: %" PRId64, cur_alive_sum);
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  No retiring task is freeing tensormap entries (last_task_alive may");
                LOG_ERROR("  still move on rings with no registered outputs). Check TaskRing");
                LOG_ERROR("  diagnostics for the stalled producer.");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Increase PTO2_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
                LOG_ERROR("========================================");
                orch_mark_fatal(orch, PTO2_ERROR_TENSORMAP_OVERFLOW);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->ring.fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph — the sole
    // source of truth for fanout now that the swimlane hot path no longer
    // records it.
    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        // TensorArgType is `enum class : int32_t` (4 bytes); the on-disk record
        // packs arg_types as uint8_t[16] (5-value enum fits in a byte). Narrow
        // each tag here rather than letting the AICPU writer reinterpret a
        // 4×-wider array as bytes — that path silently lost two of every three
        // tags on little-endian and synthesized phantom self-edges in replay.
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        // Clamp to MAX_TENSOR_ARGS even though the Arg builder caps adds at
        // MAX_TENSOR_ARGS: defensive against any future builder bypass /
        // shared-memory bit-flip that could otherwise overrun the two
        // MAX_TENSOR_ARGS-sized stack buffers above.
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            // OUTPUT slots carry create_info (not yet a Tensor); skip them —
            // they have no producer to look up and replay's per-tensor loop
            // also skips OUTPUT.
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            args.launch_spec.block_num(), kernel_ids_capture
        );
    }

    PTO2FaninBuilder fanin_builder(orch, orch->ring.fanin_pool, next_fanin_seen_epoch(orch));

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    CYCLE_COUNT_LAP(g_orch_sync_cycle);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        PTO2TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        uint8_t dep_ring_id = dep_task_id.ring();
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->ring;
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
        if (!append_fanin_or_fail(orch, dep_ring_id, dep_slot, producer_slot_state, dep_task_id, &fanin_builder)) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        uint8_t prod_ring = producer_task_id.ring();
        PTO2SharedMemoryRingHeader &producer_ring = orch->sm_header->ring;
        int32_t prod_slot = producer_ring.get_slot_by_task_id(static_cast<int32_t>(producer_task_id.local()));
        PTO2TaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
        return append_fanin_or_fail(orch, prod_ring, prod_slot, prod_state, producer_task_id, &fanin_builder);
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
        return result;
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering. The pool
    // is reclaimed as last_task_alive advances; an
    // exhausted pool back-pressures here (and detects a wedged watermark) rather
    // than tripping new_entry()'s hard assert mid-registration.
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
    task.task_timing_slot = args.task_timing_slot();
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;
    task.graph_execution = nullptr;
    task.graph_node_index = -1;
    task.kind = static_cast<bool>(active_mask) ? PTO2TaskKind::KERNEL : PTO2TaskKind::DUMMY;

    // fanout_count was already incremented per live producer inside
    // append_fanin_or_fail, atomically with the consumed/generation check under
    // the producer's fanout_lock. Doing it there (rather than a separate pass
    // here) is what prevents a producer from transitioning to CONSUMED between
    // the dependency decision and the claim.
    int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    // Store fanin metadata in payload for scheduler to iterate
    payload.fanin_actual_count = fanin_builder.count;
    payload.fanin_spill_start = fanin_builder.spill_start;
    payload.fanin_spill_pool = &fanin_builder.spill_pool;
    for (int i = 0; i < inline_count; i++) {
        payload.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
    }

    payload.init(args, result, prepared.alloc_result, layout);
    cur_slot_state.set_allow_early_resolve(args.allow_early_resolve());

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/Tensor access. Both branches write predicate.op explicitly because
    // payload slots are ring-reused; op == NONE means "always dispatch".
    {
        const L0TaskPredicate &pred = args.predicate();
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
    graph_record_task(prepared, args);
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Selective vs full dump is latched at dump_args_init from DumpDataHeader
        // (host-decided before any dispatch), so it is race-free regardless of
        // submission order. Here we only record each marked task's arg mask and
        // metadata flags, which selective collection consults.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);

    // === STEP 6: wire on the orchestrator side and publish readiness ===
    // host_build_graph host-orch: the orchestrator runs to completion on the host
    // before the device boots scheduler-only, so wire the fanout adjacency here —
    // lock each producer, allocate dep_pool entries, and publish readiness —
    // directly in submit instead of deferring it to a device-side wiring-queue
    // drain. Zero-fanin tasks and tasks whose claimed producers are already
    // completed need no fanout links or dep_pool entries and go straight to the
    // ready queue; tasks with live producers allocate fanout links first. The
    // dep_pool is sized for the whole graph (no reclaim during host
    // orchestration, exactly like the task window and GM heap), so an exhausted
    // pool latches PTO2_ERROR_DEP_POOL_OVERFLOW via dep_pool.ensure_space() — the
    // same failure class as a task-window/heap overflow. The resulting
    // fanout_head / dep-entry / ready-queue pointers are host-DDR addresses;
    // runtime_maker relocates them to device addresses before H2D.
    if (fanin_builder.count == 0) {
        cur_slot_state.fanin_count = 1;
        cur_slot_state.fanin_refcount.store(1, std::memory_order_release);
        orch_mark_dep_pool_position(orch, cur_slot_state);
        sched->push_ready_routed(&cur_slot_state);
    } else if (all_claimed_fanin_completed(fanin_builder)) {
        int32_t ready_seed = fanin_builder.count + 1;
        cur_slot_state.fanin_count = ready_seed;
        if (all_claimed_fanin_allow_early_resolve(fanin_builder)) {
            payload.dispatch_fanin.store(fanin_builder.count, std::memory_order_release);
        }
        cur_slot_state.fanin_refcount.store(ready_seed, std::memory_order_release);
        orch_mark_dep_pool_position(orch, cur_slot_state);
        sched->push_ready_routed(&cur_slot_state);
    } else {
        if (!orch_wire_live_fanin_task(orch, cur_slot_state, fanin_builder.count)) {
            return result;
        }
    }

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
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
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

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

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
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
        active_mask.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        active_mask.set_has_predicate();
    }

    return submit_task_common(
        orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const L0TaskArgs &args) {
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

    return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const L0TaskArgs &args) {
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

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, &prepared)) {
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
    // alloc_tensors builds a kernel-less descriptor that never dispatches; keep
    // the slot untagged so a recycled ring slot cannot leak a stale tag.
    task.task_timing_slot = TASK_TIMING_SLOT_NONE;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;
    task.graph_execution = nullptr;
    task.graph_node_index = -1;
    task.kind = PTO2TaskKind::DUMMY;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_actual_count = 0;
    payload.fanin_spill_start = 0;
    payload.fanin_spill_pool = &orch->ring.fanin_pool;
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
        prepared.slot_state->allow_early_resolve = true;
        prepared.slot_state->mark_completed();
    }
    graph_record_task(prepared, args);
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

PTO2GraphScopeResult
PTO2OrchestratorState::graph_begin(uint64_t graph_key, const L2TaskArgs &args, uint64_t callable_hash) {
    PTO2GraphScopeResult result;
    auto *orch = this;
    if (g_graph_recording.pending_finalize) finalize_pending_graph_definition();

    ChipStorageTaskArgs args_snapshot;
    if (orch->fatal || !graph_snapshot_args(args, &args_snapshot)) return result;
    if (g_graph_recording.active) {
        graph_mark_unsupported(PTO2_GRAPH_UNSUPPORTED_NESTED_SCOPE);
        return result;
    }

    uint64_t full_key = graph_full_key(callable_hash, graph_key);
    PTO2GraphDefinition *cached = find_graph_definition(full_key);
    if (cached != nullptr) {
#if SIMPLER_DFX
        bool record_replay_orch = orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES;
        uint64_t replay_start_ts = record_replay_orch ? get_sys_cnt_aicpu() : 0;
#endif
        uint64_t replay_task_id = PTO2TaskId::invalid().raw;
        if (graph_submit_definition(orch, *cached, args_snapshot, &replay_task_id)) {
#if SIMPLER_DFX
            if (record_replay_orch) {
                record_host_orch_phase(orch, replay_start_ts, get_sys_cnt_aicpu(), replay_task_id, g_orch_submit_idx++);
            }
#endif
            LOG_INFO_V0(
                "[GraphExecution] submit key=0x%llx nodes=%d", static_cast<unsigned long long>(full_key),
                cached->task_count
            );
            result.execute_block = false;
            result.recording = false;
            result.task_id = PTO2TaskId{replay_task_id};
            return result;
        }
        if (orch->fatal) return result;
    }

    reset_graph_recording();
    g_graph_recording.active = true;
    g_graph_recording.full_key = full_key;
    g_graph_recording.start_local_task_id = orch->ring.task_allocator.task_head();
    g_graph_recording.args = args_snapshot;
    reset_graph_definition_header(&g_graph_recording.temp);
    g_graph_recording.temp.full_key = full_key;
    result.execute_block = true;
    result.recording = true;
    return result;
}

void PTO2OrchestratorState::graph_end() {
    if (!g_graph_recording.active) return;
    if (g_graph_recording.unsupported || g_graph_recording.temp.task_count <= 0) {
        if (g_graph_recording.unsupported) {
            LOG_WARN(
                "Graph Execution definition skipped: reason=%d task_index=%d tensor_index=%d recorded_tasks=%d",
                g_graph_recording.unsupported_reason, g_graph_recording.unsupported_task_index,
                g_graph_recording.unsupported_tensor_index, g_graph_recording.temp.task_count
            );
        }
        reset_graph_recording();
        return;
    }
    g_graph_recording.active = false;
    g_graph_recording.pending_finalize = true;
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    if (!orch->fatal) finalize_pending_graph_definition();
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->ring.task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_INFO_V0("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &fanin_pool = orch->ring.fanin_pool;
        if (fanin_pool.top > 1) {
            LOG_INFO_V0(
                "=== [FaninPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, fanin_pool.top,
                fanin_pool.tail, fanin_pool.top - fanin_pool.tail, fanin_pool.high_water, fanin_pool.capacity
            );
        }
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
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
