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
 * Orchestrator interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Tracking scope nesting, which selects TensorMap or explicit dependencies
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <memory>

#include "common/chip_swimlane_profiling.h"
#include "host_build_graph/task_allocator.h"
#include "graph_cache.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/tensormap.h"
#include "host_build_graph/types.h"

struct GraphHostState;

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 *
 * host_build_graph runs the orchestrator on the host and ships the shared-memory
 * image it produces, so this whole object is host-only: it owns its scratch
 * arrays outright and no device code reads any of them. RuntimeContext therefore
 * holds it by pointer — a by-value member would put non-trivially-copyable state
 * inside the struct bind copies to the device.
 */
struct OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    SharedMemoryHeader *sm_header;

    // === TASK / HEAP ALLOCATION ===
    // hbg has one task table, so one allocator covers the whole graph.
    TaskAllocator task_allocator;
    std::unique_ptr<uint32_t[]> fanin_seen_epoch;
    uint32_t fanin_seen_current_epoch{1};

    // Frozen dependency-ancestor reachability, one word per task, indexed by local
    // id. Bit i of entry `t` is set when the task with local id `t - i - 1` reaches
    // `t` through a chain of fanin edges. Published once by t's own submit and never
    // rewritten: a task id is also its slot index and hbg reclaims neither, so an
    // entry describes the same task for the whole run. Orchestrator scratch like
    // fanin_seen_epoch — host-only, and nothing on the device reads it.
    std::unique_ptr<uint64_t[]> fanin_reach;

    // === TENSOR MAP (Private) ===
    ChipTensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Depth only. A scope decides whether a submit takes its fanin from TensorMap
    // discovery or from CoreTaskArgs::set_dependencies, and submit_task requires
    // one to be open; it bounds no task or buffer lifetime, so there is no
    // per-scope task list to walk at scope end.
    int32_t scope_stack_top{-1};  // Current top of stack (-1 = no scope open)
    int32_t manual_begin_depth{CHIP_MAX_SCOPE_DEPTH};

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    SchedulerState *scheduler;  // For simulated mode only

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // rt_orchestration_done publishes this into RuntimeContext, which is the copy
    // the device-side executor adds into its completed_tasks_ progress counter
    // so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // Host-only, like everything else here.
    GraphHostState *graph_host_state{nullptr};

    // === ARGUMENT POOLS (host-only) ===
    // The mirror's three argument pools and the next free element in each. A submit
    // binds its region at the cursor and advances it by what the task uses, so the
    // pools stay packed and the bind path reads the cursors as the image's pool
    // extents. Nothing is ever returned, and the pools are dimensioned for the worst
    // case (max_tasks tasks each at their full cap), so a bump cannot overflow.
    // The bases live here, not in the task header: nothing on the device resolves one.
    //
    // The fanin cursor does not advance at bind time. append_fanin_or_fail appends and
    // dedups producers afterwards, so the region's length is known only once the last
    // producer is appended, and it advances there.
    int32_t *fanin_pool{nullptr};
    simpler::hbg::Tensor *tensor_pool{nullptr};
    uint64_t *scalar_pool{nullptr};
    int32_t fanin_pool_cursor{0};
    int32_t tensor_pool_cursor{0};
    int32_t scalar_pool_cursor{0};

    // === STATISTICS ===
#if SIMPLER_DFX
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
    // Fanin edges transitive reduction dropped, and the edges that reached it. Both
    // count the whole run and are reported once by mark_done, which is how a change
    // to the window or to a workload's shape is read off a run.
    int64_t fanin_edges_seen;
    int64_t fanin_edges_reduced;
#endif

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API (defined in orchestrator.cpp) ===

    // Allocate the scratch arrays (fanin epoch table, scope arrays, tensor map)
    // and bind this orchestrator to one shared-memory mirror, GM heap and
    // scheduler. sm_base is the base of the mirror this orchestrator writes; it
    // is dereferenced, so a host-orch pass passes its host mirror rather than a
    // device address. `max_tasks` is the slot count that mirror is dimensioned
    // for — the bind's resolved ring_task_window — and the cap alloc() enforces.
    //
    // Returns false when an allocation fails; the caller then has no hazard map
    // and must not orchestrate.
    bool init(void *sm_base, void *gm_heap, uint64_t heap_size, uint64_t max_tasks, SchedulerState *scheduler);

    void set_scheduler(SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(ScopeMode mode = ScopeMode::AUTO);
    void end_scope();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    TaskOutputTensors submit_dummy_task(const CoreTaskArgs &args);
    TaskOutputTensors alloc_tensors(const CoreTaskArgs &args);
    GraphScopeResult graph_begin(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash);
    bool graph_prepare(void *recording_handle, const GraphTaskArgs &args);
    void graph_abort(void *recording_handle);
    bool graph_end();
    void graph_commit();
    // Bodies of the two above. Both have several early returns, so the phase
    // record that measures them wraps the call instead of every exit.
    GraphScopeResult graph_begin_inner(uint64_t graph_key, const GraphTaskArgs &args, uint64_t callable_hash);
    void graph_commit_inner();
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if SIMPLER_ORCH_PROFILING
struct OrchProfilingData {
    uint64_t alloc_cycle;  // Combined task slot + heap allocation
    uint64_t args_cycle;
    uint64_t lookup_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    int64_t submit_count;
    // Wait time tracking for blocking phases
    uint64_t fanin_wait_cycle;  // Legacy (wiring): fanout_lock wait; polling has no such lock
    // Atomic operation counts per phase
    uint64_t args_atomic_count;
};

OrchProfilingData orchestrator_get_profiling();
#endif
