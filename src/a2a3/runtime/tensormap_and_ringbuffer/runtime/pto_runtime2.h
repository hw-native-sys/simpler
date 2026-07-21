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

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"
#include "pto_orchestrator.h"
#include "aicore_completion_mailbox.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "aicpu/device_time.h"
#include "common/platform_config.h"  // PLATFORM_PROF_SYS_CNT_FREQ (data-wait deadline)
#include "common/unified_log.h"

__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu();

// FREQ-scaled cycle count for the tensor-data wait timeout. Derived here, not
// in pto_runtime2_types.h: that header is included by orchestrations which
// define PLATFORM_PROF_SYS_CNT_FREQ locally, causing a redefinition conflict.
// Mirrors the upstream/main approach in pto_runtime2.cpp pre-polling-squash.
static constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_CYCLES =
    (PTO2_TENSOR_DATA_TIMEOUT_MS * PLATFORM_PROF_SYS_CNT_FREQ) / 1000;

enum PTO2RuntimeMode {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures

struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration).
    // ABI-aligned with pto_orchestration_api.h's PTO2RuntimeOps: log_error,
    // log_warn, log_debug, log_info_v in this exact order. Mismatched layout
    // here causes the orch SO to call wrong function pointers via rt->ops,
    // which manifests as silent hangs in the dlopen'd orchestration code.
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);
    // INFO with explicit verbosity tier (v ∈ [0, 9]; gating done inside).
    void (*log_info_v)(const char *func, int v, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(
        PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const L0TaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const L0TaskArgs &args);
    // Stash the call-site captured by PTO2ScopeGuard into the [ScopeStats]
    // collector. Always present in the struct to keep ops-table layout stable
    // across SIMPLER_DFX settings; set to nullptr at SIMPLER_DFX=0.
    void (*scope_set_site)(const char *file, int line);
};

/**
 * Sizing half of the runtime-arena layout: the capacities that *define* the
 * layout (the input to runtime_reserve_layout). Stable per (callable_id, ring
 * config); re-read at AICPU boot to reconstruct ring/heap/dep-pool capacities.
 */
struct ArenaSizingKey {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]{};
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]{};
    int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH]{};
};

/**
 * Offset half of the runtime-arena layout: every sub-region offset
 * (sm_handle wrapper / orchestrator / scheduler / runtime header / AICore
 * mailbox) plus the committed arena byte size. The *output* of
 * runtime_reserve_layout; consumed by runtime_init_data_from_layout and
 * runtime_wire_arena_pointers (the AICPU re-wires arena-internal pointers
 * from these after rtMemcpy).
 */
struct ArenaOffsets {
    size_t off_sm_handle{0};
    PTO2OrchestratorLayout orch;
    PTO2SchedulerLayout sched;
    size_t off_runtime{0};
    size_t off_mailbox{0};

    // Total arena byte size post-commit. Used by host to size the prebuilt
    // image buffer and as the rtMemcpy length.
    size_t arena_size{0};
};

/**
 * Layout descriptor for the prebuilt runtime arena. Two named halves with
 * distinct lifetimes/semantics: `sizing` is the layout-defining input
 * (capacities), `offsets` is the computed sub-region offsets + arena size.
 * Produced once on the host by runtime_reserve_layout(); consumed by
 * runtime_init_data_from_layout and runtime_wire_arena_pointers.
 */
struct PTO2RuntimeArenaLayout {
    ArenaSizingKey sizing;
    ArenaOffsets offsets;
};

struct PTO2Runtime {
    // Ops table (first field — used by orchestration .so via function pointers)
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;

    // Components
    PTO2SharedMemoryHandle *sm_handle;
    PTO2OrchestratorState orchestrator;
    PTO2SchedulerState scheduler;
    AICoreCompletionMailbox *aicore_mailbox;

    // GM Heap for output buffers
    void *gm_heap;
    uint64_t gm_heap_size;
    bool gm_heap_owned;  // True if we allocated it

    // Mode
    PTO2RuntimeMode mode;

    // Statistics
    int64_t total_cycles;

    PTO2RuntimeArenaLayout prebuilt_layout;
};

// Canonical per-ring form (matches upstream a5 signature).
inline PTO2RuntimeArenaLayout runtime_reserve_layout(
    DeviceArena &arena, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH], const int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH]
) {
    PTO2RuntimeArenaLayout layout{};
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        layout.sizing.task_window_sizes[r] = task_window_sizes[r];
        layout.sizing.heap_sizes[r] = heap_sizes[r];
        layout.sizing.dep_pool_capacities[r] = dep_pool_capacities[r];
    }

    int32_t task_window_sizes_i32[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        task_window_sizes_i32[r] = static_cast<int32_t>(task_window_sizes[r]);

    layout.offsets.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.offsets.orch = PTO2OrchestratorState::reserve_layout(arena, task_window_sizes_i32, dep_pool_capacities[0]);
    layout.offsets.sched = PTO2SchedulerState::reserve_layout(arena, dep_pool_capacities[0]);
    layout.offsets.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.offsets.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.offsets.arena_size = arena.total_size();
    return layout;
}

// Single-size adapter: broadcasts the scalar to every ring. Defined after the
// per-ring overload so name lookup sees both at the call site.
inline PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    uint64_t per_ring_task_window[PTO2_MAX_RING_DEPTH];
    uint64_t per_ring_heap[PTO2_MAX_RING_DEPTH];
    int32_t per_ring_dep_pool[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        per_ring_task_window[r] = task_window_size;
        per_ring_heap[r] = 0;  // Heap default; caller may set separately via runtime_init_data_from_layout.
        per_ring_dep_pool[r] = dep_pool_capacity;
    }
    return runtime_reserve_layout(arena, per_ring_task_window, per_ring_heap, per_ring_dep_pool);
}

inline PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t,
    void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.offsets.off_runtime));
    memset(rt, 0, sizeof(*rt));

    auto *sm_wrap = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.offsets.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    if (!rt->orchestrator.init_data_from_layout(
            layout.offsets.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.sizing.task_window_sizes[0]
        ))
        return nullptr;
    if (!rt->scheduler.init_data_from_layout(layout.offsets.sched, arena, sm_dev_base)) return nullptr;

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.offsets.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

// Per-ring overload (matches upstream a5 signature with sm_size + heap_sizes[]).
inline PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t sm_size,
    void *gm_heap_dev_base, const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    return runtime_init_data_from_layout(arena, layout, mode, sm_dev_base, sm_size, gm_heap_dev_base, heap_sizes[0]);
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt);

inline void runtime_destroy(PTO2Runtime *rt) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}

// Upstream-compatible overload: arena is ignored (arena lifetime is owned by
// the caller in the polling design too).
inline void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) { runtime_destroy(rt); }

// Upstream arena-reuse path (#1234). On cache hits the host skips the
// arena re-upload, so the AICPU-side reset here is the only thing that
// scrubs the previous run's orchestrator/scheduler state. Currently
// re-runs init_data_from_layout on each sub-region followed by
// wire_arena_pointers (init_data_from_layout wipes the struct via
// *state = {}, so the wired pointers must be re-set). This adds ~2 ms of
// Device wall vs upstream's surgical reset_for_reuse; a fully surgical
// polling version is deferred as follow-up work (see the reset_for_reuse
// methods added on PTO2OrchestratorState / PTO2SchedulerState /
// PTO2TensorMap / PTO2TaskAllocator / PTO2ReadyQueue / PTO2SpscQueue for
// the scaffolding — the last-mile issue is that ready_queue's
// reset_for_reuse is a no-op and something in the surgical path leaves
// state that trips a scheduler stall on the second run).
void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt);

bool runtime_reset_for_reuse(DeviceArena & /*arena*/, const PTO2RuntimeArenaLayout & /*layout*/, PTO2Runtime *rt);

void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode);

void rt_scope_begin(PTO2Runtime *rt);

inline void rt_scope_end(PTO2Runtime *rt) { rt->orchestrator.end_scope(); }

inline void rt_orchestration_done(PTO2Runtime *rt) { rt->orchestrator.mark_done(); }

void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

// Orchestration-side logging dispatchers: orchestration .so calls
// LOG_*(fmt, ...) which routes through these ops into the unified log.
// Verbosity gates live inside the unified_log_* primitives.
inline void rt_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    unified_log_error(func, "%s", message);
}
inline void rt_log_warn(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    unified_log_warn(func, "%s", message);
}
inline void rt_log_debug(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    unified_log_debug(func, "%s", message);
}
inline void rt_log_info_v(const char *func, int v, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    unified_log_info_v(func, v, "%s", message);
}

MAYBE_UNINITIALIZED_BEGIN
bool wait_for_tensor_ready(PTO2Runtime *rt, const Tensor &tensor, bool wait_for_consumers, const char *caller);

MAYBE_UNINITIALIZED_END

uint64_t get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);

void set_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value);

// Function-pointer ops table backing — moved from pto_runtime2.cpp so that
// the runtime_finalize_after_wire above can refer to it.

TaskOutputTensors submit_task_impl(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const L0TaskArgs &args);

TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const L0TaskArgs &args);

TaskOutputTensors submit_dummy_task_impl(PTO2Runtime *rt, const L0TaskArgs &args);

inline bool is_fatal_impl(PTO2Runtime *rt) { return rt->orchestrator.fatal; }

inline const PTO2RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .report_fatal = rt_report_fatal,
    .log_error = rt_log_error,
    .log_warn = rt_log_warn,
    .log_debug = rt_log_debug,
    .log_info_v = rt_log_info_v,
    .get_tensor_data = get_tensor_data,
    .set_tensor_data = set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
    .submit_dummy_task = submit_dummy_task_impl,
    .scope_set_site = nullptr,
};

void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count);

#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif
