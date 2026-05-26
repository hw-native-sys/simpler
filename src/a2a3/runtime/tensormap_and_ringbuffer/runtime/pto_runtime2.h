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

#include "device_arena.h"
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
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args);
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
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const Arg &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const Arg &args);
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
};

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Create runtime from caller-provided GM SM buffer + GM heap.
 *
 * All AICPU-side runtime state (PTO2SharedMemoryHandle wrapper, PTO2Runtime,
 * AICoreCompletionMailbox, plus the orchestrator/scheduler/tensor_map
 * sub-regions) is laid out on the supplied arena and committed in a single
 * backing allocation — including the SM handle wrapper itself. The arena is
 * owned by the caller (typically the per-Worker AicpuExecutor);
 * runtime_destroy() calls arena.release() once to free the lot.
 *
 * `sm_base` / `sm_size` describe the SM buffer that host has already placed
 * for the runtime to use; the SM handle wrapper is constructed in-place on
 * an arena-reserved region pointing at that buffer.
 *
 * @param mode             Execution mode
 * @param sm_base          Pre-allocated SM buffer base (host-owned)
 * @param sm_size          Size of the SM buffer in bytes
 * @param task_window_size Per-ring task window size used to lay out SM
 * @param gm_heap          GM heap base for output buffers (or NULL if not used)
 * @param heap_size        GM heap size in bytes
 * @param arena            Caller-owned arena that sources all runtime sub-regions.
 *                         Must be freshly constructed (no prior commit) —
 *                         runtime_create_from_sm reserves + commits internally.
 * @return Runtime context, or NULL on failure
 */
PTO2Runtime *runtime_create_from_sm(
    PTO2RuntimeMode mode, void *sm_base, uint64_t sm_size, uint64_t task_window_size, void *gm_heap, uint64_t heap_size,
    DeviceArena &arena, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
);

/**
 * Destroy runtime and free all resources. arena.release() is the actual
 * memory free; the rt pointer is no longer valid afterward.
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
