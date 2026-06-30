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
 * PTO Runtime2 - Main Interface
 *
 * This is the main header for the PTO Runtime2 system.
 * It provides a unified API for task graph construction and execution.
 *
 * Key Features:
 * - Ring buffer based memory management (zero allocation overhead)
 * - Lazy invalidation TensorMap for dependency discovery
 * - Scope-based buffer lifecycle management
 * - Per-task spinlocks for concurrent fanout updates
 * - Orchestrator-Scheduler decoupling via shared memory
 *
 * Usage:
 *   1. Create runtime: PTO2Runtime create methods
 *   2. Build task graph in orchestration function:
 *      - begin_scope() / end_scope()
 *      - submit_task()
 *   3. Mark orchestration complete: mark_done()
 *   4. Destroy runtime
 *
 * Based on: docs/RUNTIME_LOGIC.md
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

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum PTO2RuntimeMode {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

/**
 * Function-pointer ops table for runtime operations.
 *
 * The orchestration .so calls runtime functions through this table
 * (via pto_orchestration_api.h inline wrappers), so it has zero link
 * dependencies on runtime .cpp files.
 */
typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures

struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);
    // INFO with explicit verbosity tier (v ∈ [0,9]; gating done inside).
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
    // across PTO2_PROFILING settings; set to nullptr at PTO2_PROFILING=0.
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
 * (capacities + scheduler timeout), `offsets` is the computed sub-region
 * offsets + arena size. Produced once on the host by runtime_reserve_layout();
 * consumed by runtime_init_data_from_layout and runtime_wire_arena_pointers.
 */
struct PTO2RuntimeArenaLayout {
    ArenaSizingKey sizing;
    ArenaOffsets offsets;
};

/**
 * PTO Runtime2 context
 *
 * Contains all state for orchestration and scheduling.
 * In simulated mode, runs in single process with shared address space.
 */
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

    // Prebuilt-arena fast path metadata. Carries every offset
    // wire_arena_pointers needs at AICPU boot so the AICPU can reconstruct
    // all arena-internal pointer fields without re-running init_data. The
    // device base of the runtime arena travels separately on the host-side
    // Runtime (Runtime::prebuilt_arena_base_), since the AICPU needs it
    // *before* dereferencing this image. Populated on host by
    // runtime_init_data_from_layout + runtime_wire_arena_pointers; read by
    // aicpu_executor.cpp.
    PTO2RuntimeArenaLayout prebuilt_layout;
};

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Phase 1 — declare every sub-region (sm_handle wrapper, orchestrator /
 * scheduler / tensor_map / mailbox / PTO2Runtime header) on the supplied
 * arena. Pure arithmetic; does not touch device memory and may run on host.
 * Returns the layout descriptor; caller commits/attaches the arena before
 * Phase 2/3.
 */
PTO2RuntimeArenaLayout runtime_reserve_layout(
    DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
);
PTO2RuntimeArenaLayout runtime_reserve_layout(
    DeviceArena &arena, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH], const int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH]
);

/**
 * Phase 2 — write the data half of the runtime arena: standalone fields,
 * memset'd arena regions, sub-structure initializers, and SM-side device
 * pointers. The arena must already be committed (or attached); writes go
 * into arena.base() + sub-region offsets.
 *
 * `sm_dev_base` / `gm_heap_dev_base` are device addresses; we only store
 * them (never dereference). Safe to run on a host arena that owns a host
 * mirror of the runtime image — the resulting buffer is rtMemcpy-ready.
 *
 * Returns the PTO2Runtime* that sits at layout.off_runtime within the arena.
 * Caller must follow up with runtime_wire_arena_pointers; rt->ops and the
 * AICore-side count fields are left untouched and must be filled by the
 * AICPU at boot.
 */
PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t sm_size,
    void *gm_heap_dev_base, uint64_t heap_size
);
PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t sm_size,
    void *gm_heap_dev_base, const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
);

/**
 * Phase 3 — wire every arena-internal pointer field (rt->sm_handle,
 * rt->aicore_mailbox, orchestrator.{scope_tasks, scope_begins, scheduler,
 * tensor_map.*, rings[].fanin_pool.base}, scheduler.{ready_queues, dep_pool,
 * wiring.queue}) so each holds arena.base() + offset. Idempotent — runs on
 * both host (writing host-mirror addresses) and AICPU (writing device
 * addresses) sides.
 */
void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt);

/**
 * AICPU-only Phase 4 — fill in the few fields the host could not know at
 * prebuilt-image build time: the ops table (s_runtime_ops is a device-side
 * file-local global, host cannot resolve its device address) and the
 * orchestrator's core counts (depend on the executor's scheduler context).
 * Call once per boot after runtime_wire_arena_pointers.
 */
void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count);

/**
 * Destroy runtime. With the prebuilt-arena fast path the arena buffer is
 * pooled across runs by DeviceRunner, so we never call arena.release()
 * here — the destructor only forgets sub-structure pointers (idempotent
 * cleanup).
 */
void runtime_destroy(PTO2Runtime *rt, DeviceArena &arena);

/**
 * Set execution mode
 */
void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode);

// =============================================================================
// Orchestration API (called by orchestration function)
// =============================================================================

/**
 * Begin a new scope
 *
 * All tasks submitted within this scope will have their lifetime
 * bounded by the scope. When scope_end() is called, the scope
 * releases its reference to all enclosed tasks.
 */
void rt_scope_begin(PTO2Runtime *rt);

/**
 * End current scope
 *
 * Releases scope reference for all tasks submitted since scope_begin().
 * Tasks whose refcount reaches zero will have their buffers released.
 */
void rt_scope_end(PTO2Runtime *rt);

/**
 * Mark orchestration as complete
 *
 * Signals that no more tasks will be submitted.
 */
void rt_orchestration_done(PTO2Runtime *rt);

/**
 * Enter fatal state explicitly from orchestration.
 */
void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

/**
 * Cross-layer data access: read a tensor value by waiting for its producer.
 */
uint64_t get_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);

/**
 * Cross-layer data access: write a value to a tensor at given indices.
 * Waits for producer completion (WAW) and all consumers (WAR) via TensorMap.
 * See set_tensor_data in pto_orchestration_api.h for full documentation.
 */
void set_tensor_data(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value);

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with pto_orchestration_api.h (same layout, guarded).
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif
