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
 * PTO Runtime2 - Orchestrator Interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_graph_cache.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

/**
 * Layout descriptor produced by PTO2OrchestratorState::reserve_layout(). Holds
 * arena offsets for the dependency pool, fanin dedup table, and nested
 * TensorMap layout.
 */
struct PTO2OrchestratorLayout {
    size_t off_fanin_seen_epoch;
    size_t off_dep_pool_entries;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacity;
    uint64_t scope_stack_capacity;
};

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === WHOLE-GRAPH RESOURCES ===
    PTO2TaskAllocator task_allocator;
    PTO2DepListPool dep_pool;
    uint32_t *fanin_seen_epoch;
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Scopes retain only nesting/manual-dependency semantics. They do not own
    // task lists because nothing is reclaimed within a replay.
    int32_t scope_stack_top;        // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;  // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    // === SCHEDULER STATE ACCESS ===
    // Same runtime-arena scheduler object; Orch-side wiring mutates dep_pool
    // and publishes ready tasks through it before scheduler workers dispatch.
    PTO2SchedulerState *scheduler;

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if SIMPLER_DFX
    // L2 swimlane_level copied from get_l2_swimlane_level().
    L2SwimlaneLevel l2_swimlane_level{L2SwimlaneLevel::DISABLED};
#endif

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // The executor adds this count into its completed_tasks_ progress counter
    // after orchestration finishes so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // Boundary publication normally sees no completed tasks because every
    // scheduler task is held behind its publication gate. Hidden alloc tasks
    // are the one exception: they complete synchronously on the Orch thread.
    // Track them incrementally per arena so rt_graph_boundary() stays O(1).
    int32_t inline_completed_by_buffer[PTO2_REPLAY_GRAPH_BUFFER_COUNT]{};

    // === STATISTICS ===
#if SIMPLER_DFX
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API (defined in pto_orchestrator.cpp) ===

    static PTO2OrchestratorLayout
    reserve_layout(DeviceArena &arena, int32_t task_window_size, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // sm_dev_base is the SM device address (only stored, never dereferenced);
    // task_window_size feeds the per-ring SM address arithmetic. Safe to call
    // on a host arena that holds the prebuilt image.
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );
    bool reset_for_reuse(
        const PTO2OrchestratorLayout &layout, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );

    // Phase 3b: wire dependency storage, initial-ready handoff, TensorMap, and scheduler.
    // Idempotent — host runs once on the image, AICPU runs once after attach.
    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler);

    // Forget pointers; arena owns the backing buffers.
    void destroy();
    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    TaskOutputTensors submit_dummy_task(const L0TaskArgs &args);
    TaskOutputTensors alloc_tensors(const L0TaskArgs &args);
    PTO2GraphScopeResult graph_begin(uint64_t graph_key, const L2TaskArgs &args, uint64_t callable_hash);
    void graph_end(PTO2GraphCacheStats *stats);
    bool finalize_pending_graph_definition();
    void mark_done();
};

size_t pto2_graph_definition_cache_bytes();
void pto2_graph_definition_destroy_all();

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if SIMPLER_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    int64_t submit_count;
};

PTO2OrchProfilingData orchestrator_get_profiling();
#endif

#endif  // PTO_ORCHESTRATOR_H
