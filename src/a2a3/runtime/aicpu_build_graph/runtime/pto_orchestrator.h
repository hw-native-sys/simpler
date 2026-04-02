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
 * PTO Runtime2 - Orchestrator Interface (Explicit Dependency Variant)
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph via explicit add_dependency calls
 * 5. Managing buffer scopes for lifecycle control
 *
 * Key differences from the tensormap_and_ringbuffer variant:
 * - No TensorMap: dependencies are explicitly specified by orchestration code
 * - Scope-end batch publish: tasks are invisible to the scheduler until scope_end
 * - submit_task returns PTO2TaskId for use in add_dependency calls
 */

#ifndef SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_ORCHESTRATOR_H_
#define SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_ORCHESTRATOR_H_

#include "pto_ring_buffer.h"     // NOLINT(build/include_subdir)
#include "pto_runtime2_types.h"  // NOLINT(build/include_subdir)
#include "pto_scheduler.h"       // NOLINT(build/include_subdir)
#include "pto_shared_memory.h"   // NOLINT(build/include_subdir)
#include "pto_submit_types.h"    // NOLINT(build/include_subdir)
#include "pto_types.h"           // NOLINT(build/include_subdir)

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 * No TensorMap — dependencies are added explicitly via pto2_add_dependency().
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHandle *sm_handle;

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    PTO2TaskSlotState **scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;     // Allocated capacity of scope_tasks
    int32_t *scope_begins;            // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;          // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;    // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only
#if PTO2_PROFILING
    // Runtime profiling switch copied from Runtime::enable_profiling.
    bool enable_profiling;
#endif

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    /**
     * Get current ring index from scope depth.
     * Maps scope depth to ring_id: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
     */
    uint8_t current_ring_id() const {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < PTO2_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : PTO2_MAX_RING_DEPTH - 1;
    }

    /**
     * Allocate packed output buffer from current ring's heap
     */
    void *pto2_alloc_packed_buffer(int32_t total_size) {
        if (total_size <= 0) {
            return NULL;
        }

        uint8_t rid = current_ring_id();
        void *buffer = rings[rid].heap_ring.pto2_heap_ring_alloc(total_size);

#if PTO2_PROFILING
        buffers_allocated++;
        bytes_allocated += total_size;
#endif

        return buffer;
    }
};

// =============================================================================
// Orchestrator API
// =============================================================================

/**
 * Initialize orchestrator state
 *
 * @param orch       Orchestrator state to initialize
 * @param sm_handle  Shared memory handle
 * @param gm_heap    GM heap memory for output buffers
 * @param heap_size  Size of GM heap
 * @return true on success
 */
bool pto2_orchestrator_init(
    PTO2OrchestratorState *orch, PTO2SharedMemoryHandle *sm_handle, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
);

/**
 * Destroy orchestrator state and free resources
 */
void pto2_orchestrator_destroy(PTO2OrchestratorState *orch);

/**
 * Set scheduler reference (for simulated mode)
 */
void pto2_orchestrator_set_scheduler(PTO2OrchestratorState *orch, PTO2SchedulerState *scheduler);

// =============================================================================
// Scope Management
// =============================================================================

/**
 * Begin a new scope
 *
 * Pushes a new empty task list onto the scope stack.
 * Tasks submitted while this scope is at the top of the stack are
 * owned by it and have their fanout_count initialized to 1.
 */
void pto2_scope_begin(PTO2OrchestratorState *orch);

/**
 * End current scope
 *
 * Batch-publishes all tasks in the scope:
 * 1. For each task, releases the "+1 redundance" in fanin_refcount
 * 2. Tasks with all deps satisfied are pushed to the ready queue
 * 3. Releases the scope's fanout reference (enables CONSUMED transition)
 *
 * This is the scope-end batch publish mechanism: tasks are invisible
 * to the scheduler until this point.
 */
void pto2_scope_end(PTO2OrchestratorState *orch);

// =============================================================================
// Task Submission
// =============================================================================

/**
 * Submit a task with InCore function and parameters
 *
 * Simplified flow (no TensorMap):
 * 1. Allocates task slot from TaskRing (blocks until available)
 * 2. Allocates packed output buffer from HeapRing (blocks until available)
 * 3. Writes task descriptor and payload
 * 4. Initializes fanin with +1 redundance (released at scope_end)
 *
 * The task is NOT visible to the scheduler until scope_end.
 * Dependencies must be added via pto2_add_dependency() before scope_end.
 *
 * @param orch        Orchestrator state
 * @param mixed_kernels  Kernel IDs for AIC/AIV0/AIV1 slots
 * @param args      Aggregated tensor and scalar parameters
 * @return PTO2TaskId for use in pto2_add_dependency()
 */
SubmitResult pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args);

// =============================================================================
// Explicit Dependency Management
// =============================================================================

/**
 * Add a dependency edge: producer -> consumer
 *
 * The consumer task will not become ready until the producer completes.
 * Both tasks must have been created via pto2_submit_mixed_task().
 *
 * For cross-scope dependencies (producer from a previous scope that is
 * already visible to the scheduler), this uses the fanout_lock for
 * thread safety and handles the case where the producer has already
 * completed (early-finish optimization).
 *
 * @param orch      Orchestrator state
 * @param producer  Producer task ID (must complete before consumer starts)
 * @param consumer  Consumer task ID (depends on producer)
 */
void pto2_add_dependency(PTO2OrchestratorState *orch, PTO2TaskId producer, PTO2TaskId consumer);

// =============================================================================
// Flow Control
// =============================================================================

/**
 * Mark orchestration as complete
 *
 * Signals to scheduler that no more tasks will be submitted.
 */
void pto2_orchestrator_done(PTO2OrchestratorState *orch);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print orchestrator statistics
 */
void pto2_orchestrator_print_stats(PTO2OrchestratorState *orch);

/**
 * Print scope stack state
 */
void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState *orch);

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t alloc_cycle;
    uint64_t args_cycle;
    uint64_t heap_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t alloc_wait_cycle;  // Cycles spent waiting in task_ring_alloc
    uint64_t heap_wait_cycle;   // Cycles spent waiting in heap_ring_alloc
    uint64_t fanin_wait_cycle;  // Cycles spent waiting in fanout_lock
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
    uint64_t args_atomic_count;
    uint64_t heap_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t scope_end_atomic_count;
};

/**
 * Get and reset orchestrator profiling data.
 * Returns accumulated profiling data and resets counters.
 */
PTO2OrchProfilingData pto2_orchestrator_get_profiling();
#endif

#endif  // SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_ORCHESTRATOR_H_
