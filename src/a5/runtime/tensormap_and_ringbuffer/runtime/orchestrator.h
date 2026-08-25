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
 * tensormap_and_ringbuffer orchestrator interface
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

#pragma once

#include "common/chip_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "ring_buffer.h"
#include "runtime_types.h"
#include "submit_types.h"
#include "scheduler/scheduler.h"
#include "shared_memory.h"
#include "tensormap.h"
#include "types.h"

/**
 * Layout descriptor produced by OrchestratorState::reserve_layout(). Holds
 * arena offsets for every sub-region the orchestrator owns (per-ring fanin
 * pools, scope arrays, plus the nested ChipTensorMap layout).
 */
struct OrchestratorLayout {
    size_t off_fanin_pool[CHIP_MAX_RING_DEPTH];
    size_t off_fanin_seen_epoch[CHIP_MAX_RING_DEPTH];
    size_t off_wait_reach[CHIP_MAX_RING_DEPTH];
    size_t off_scope_tasks;
    size_t off_scope_begins;
    ChipTensorMapLayout tensor_map;
    int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH];
    int32_t scope_tasks_cap;
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
struct OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    SharedMemoryHeader *sm_header;

    // === PER-RING RESOURCES ===
    ChipRingSet rings[CHIP_MAX_RING_DEPTH];
    uint32_t *fanin_seen_epoch[CHIP_MAX_RING_DEPTH];
    uint32_t fanin_seen_current_epoch{1};

    // Per-slot frozen WAIT-ancestor reachability (bitmap + submit seq),
    // indexed [ring][slot]. Orchestrator-private runtime-arena storage, never
    // read by scheduler threads. A slot's entry is valid only while the slot
    // holds the task that published it: every consumer that reads a producer's
    // entry holds that producer's submit-claim pin (fanout_count), so the slot
    // cannot be rebound under the read.
    WaitReachEntry *wait_reach[CHIP_MAX_RING_DEPTH];

    // Global submission sequence. Assigned once per prepared task across all
    // rings; unsigned subtraction preserves recent distances across uint64 wrap.
    uint64_t submit_seq{0};

    // === TENSOR MAP (Private) ===
    ChipTensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    ChipTaskSlotState **scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;     // Allocated capacity of scope_tasks
    int32_t *scope_begins;            // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;          // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;    // Max nesting depth (CHIP_MAX_SCOPE_DEPTH)
    int32_t manual_begin_depth{CHIP_MAX_SCOPE_DEPTH};

    // === SCHEDULER STATE ACCESS ===
    // Same runtime-arena scheduler object; Orch-side wiring mutates dep_pool
    // and publishes ready tasks through it before scheduler workers dispatch.
    SchedulerState *scheduler;

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if SIMPLER_DFX
    // chip swimlane_level copied from get_chip_swimlane_level().
    ChipSwimlaneLevel chip_swimlane_level{ChipSwimlaneLevel::DISABLED};
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

    // === STATISTICS ===
#if SIMPLER_DFX
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    /**
     * Get current ring index from scope depth.
     * Maps scope depth to ring_id: min(scope_depth, CHIP_MAX_RING_DEPTH - 1)
     */
    uint8_t current_ring_id() const {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < CHIP_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : CHIP_MAX_RING_DEPTH - 1;
    }

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API (defined in orchestrator.cpp) ===

    // Phase 1: declare every sub-region (per-ring fanin pool, scope arrays,
    // tensor_map sub-layout) on the supplied arena. task_window_sizes feeds
    // the nested tensor_map layout. Returned layout is consumed by
    // init_data_from_layout.
    static OrchestratorLayout reserve_layout(
        DeviceArena &arena, const int32_t task_window_sizes[CHIP_MAX_RING_DEPTH],
        int32_t dep_pool_capacity = CHIP_DEP_LIST_POOL_SIZE
    );
    static OrchestratorLayout reserve_layout(
        DeviceArena &arena, const int32_t task_window_sizes[CHIP_MAX_RING_DEPTH],
        const int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH]
    );

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // sm_dev_base is the SM device address (only stored, never dereferenced);
    // task_window_size feeds the per-ring SM address arithmetic. Safe to call
    // on a host arena that holds the prebuilt image.
    bool init_data_from_layout(
        const OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );
    bool init_data_from_layout(
        const OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap,
        const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]
    );
    bool reset_for_reuse(
        const OrchestratorLayout &layout, void *sm_dev_base, void *gm_heap,
        const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]
    );

    // Phase 3b: write the arena-internal pointer fields (scope_tasks,
    // scope_begins, rings[].fanin_pool.base, tensor_map.{buckets,entry_pool,
    // free_entry_list,task_entry_heads}, scheduler reference).
    // Idempotent — host runs once on the image, AICPU runs once after attach.
    void wire_arena_pointers(const OrchestratorLayout &layout, DeviceArena &arena, SchedulerState *scheduler);

    // Forget pointers; arena owns the backing buffers.
    void destroy();
    void set_scheduler(SchedulerState *scheduler);
    void mark_dep_pool_position(ChipTaskSlotState &slot_state);
    void wire_fanin_task(ChipTaskSlotState &slot_state, int32_t wfanin);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(ScopeMode mode = ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    TaskOutputTensors submit_dummy_task(const CoreTaskArgs &args);
    TaskOutputTensors alloc_tensors(const CoreTaskArgs &args);
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if SIMPLER_ORCH_PROFILING
struct OrchProfilingData {
    uint64_t sync_cycle;
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t alloc_wait_cycle;  // Cycles spent waiting in unified alloc
    uint64_t fanin_wait_cycle;  // Cycles spent waiting in fanout_lock
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
    uint64_t args_atomic_count;
    uint64_t scope_end_atomic_count;
};

OrchProfilingData orchestrator_get_profiling();
#endif
