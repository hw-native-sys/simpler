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

#include <atomic>

#include "common/l2_swimlane_profiling.h"
#include "utils/device_arena.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

/**
 * Layout descriptor produced by PTO2OrchestratorState::reserve_layout(). Holds
 * arena offsets for every sub-region the orchestrator owns (per-ring fanin
 * pools, scope arrays, plus the nested PTO2TensorMap layout).
 */
struct PTO2OrchestratorLayout {
    size_t off_fanin_pool[PTO2_MAX_RING_DEPTH];
    size_t off_fanin_seen_epoch[PTO2_MAX_RING_DEPTH];
    size_t off_scope_tasks;
    size_t off_scope_begins;
    PTO2TensorMapLayout tensor_map;
    int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH];
    int32_t scope_tasks_cap;
    uint64_t scope_stack_capacity;
};

constexpr int32_t PTO2_SUBMIT_PIPELINE_QUEUE_CAP = 64;
constexpr int32_t PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP = 64;
constexpr int32_t PTO2_SUBMIT_PIPELINE_SCOPE_INLINE_CAP = 64;
constexpr int32_t PTO2_SUBMIT_PIPELINE_MAX_COMMIT_STAGES = 1;
constexpr int32_t PTO2_SUBMIT_PIPELINE_CONTROL_IDLE = 0;
constexpr int32_t PTO2_SUBMIT_PIPELINE_CONTROL_WORK = 1;
constexpr int32_t PTO2_SUBMIT_PIPELINE_CONTROL_STOP = 2;
constexpr int32_t PTO2_SUBMIT_PIPELINE_HEAP_GUARD_OUTPUT_BYTES = 64 * 1024;

enum class PTO2SubmitPipelineRecordKind : int32_t {
    TASK_DEFERRED = 0,
    SCOPE_END = 1,
};

struct PTO2SubmitCommitRecord {
    PTO2SubmitPipelineRecordKind kind{PTO2SubmitPipelineRecordKind::TASK_DEFERRED};
    PTO2TaskPayload *payload{nullptr};
    PTO2TaskSlotState *slot_state{nullptr};
    PTO2SchedulerState *scheduler{nullptr};
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskId explicit_deps[PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP]{};
    PTO2TaskSlotState *scope_task_slot_states[PTO2_SUBMIT_PIPELINE_SCOPE_INLINE_CAP]{};
    TensorArgType arg_types[MAX_TENSOR_ARGS]{};
    int32_t explicit_dep_count{0};
    int32_t scope_task_count{0};
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT]{INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID};
    bool in_manual_scope{false};
};

struct PTO2SubmitPipelineQueue {
    std::atomic<uint64_t> tail{0};
    std::atomic<uint64_t> head{0};
    std::atomic<int32_t> slot_state[PTO2_SUBMIT_PIPELINE_QUEUE_CAP];
    PTO2SubmitCommitRecord records[PTO2_SUBMIT_PIPELINE_QUEUE_CAP];
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

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];
    uint32_t *fanin_seen_epoch[PTO2_MAX_RING_DEPTH];
    uint32_t fanin_seen_current_epoch{1};

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

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
    int32_t manual_begin_depth{PTO2_MAX_SCOPE_DEPTH};

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only

    // === SUBMIT COMMIT PIPELINE ===
    // Strategy2 uses a second orchestrator worker to commit dependency lookup,
    // TensorMap registration, and ready/wiring publication after O1 has
    // materialized the task payload returned to orchestration.
    bool submit_pipeline_enabled{false};
    bool submit_pipeline_defer_dependencies{false};
    bool submit_pipeline_signal_scheduler_drain{false};
    bool submit_pipeline_compact_deferred_records{false};
    int32_t submit_pipeline_commit_stages{0};
    std::atomic<bool> submit_pipeline_stop{false};
    std::atomic<bool> submit_pipeline_work_available{false};
    std::atomic<int32_t> submit_pipeline_control{PTO2_SUBMIT_PIPELINE_CONTROL_IDLE};
    std::atomic<uint64_t> submit_pipeline_completed{0};
    std::atomic<bool> submit_pipeline_stage_done[PTO2_SUBMIT_PIPELINE_MAX_COMMIT_STAGES];
    PTO2SubmitPipelineQueue submit_pipeline_queues[PTO2_SUBMIT_PIPELINE_MAX_COMMIT_STAGES];
#if PTO2_PROFILING
    uint64_t submit_pipeline_task_enqueue_cycles{0};
    uint64_t submit_pipeline_scope_enqueue_cycles{0};
    uint64_t submit_pipeline_flush_cycles{0};
    uint64_t submit_pipeline_deferred_dep_cycles{0};
    uint64_t submit_pipeline_deferred_fanin_cycles{0};
    uint64_t submit_pipeline_publish_cycles{0};
    uint64_t submit_pipeline_scope_release_cycles{0};
    uint64_t submit_pipeline_publish_spins{0};
    uint32_t submit_pipeline_scheduler_drain_hint_count{0};
    uint32_t submit_pipeline_task_enqueue_count{0};
    uint32_t submit_pipeline_scope_enqueue_count{0};
    uint32_t submit_pipeline_flush_count{0};
    uint32_t submit_pipeline_deferred_commit_count{0};
    uint32_t submit_pipeline_scope_record_count{0};
    uint32_t submit_pipeline_dep_explicit_count{0};
    uint32_t submit_pipeline_dep_register_count{0};
    uint32_t submit_pipeline_dep_fanin_actual_count{0};
#endif

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if PTO2_PROFILING
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

    bool in_manual_scope() const { return scope_stack_top >= manual_begin_depth; }

    // === Cold-path API (defined in pto_orchestrator.cpp) ===

    // Phase 1: declare every sub-region (per-ring fanin pool, scope arrays,
    // tensor_map sub-layout) on the supplied arena. task_window_sizes feeds
    // the nested tensor_map layout. Returned layout is consumed by
    // init_from_layout.
    static PTO2OrchestratorLayout reserve_layout(
        DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH],
        int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
    );
    static PTO2OrchestratorLayout reserve_layout(
        DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH],
        const int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH]
    );

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // sm_dev_base is the SM device address (only stored, never dereferenced);
    // task_window_size feeds the per-ring SM address arithmetic. Safe to call
    // on a host arena that holds the prebuilt image.
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap, uint64_t heap_size,
        uint64_t task_window_size
    );
    bool init_data_from_layout(
        const PTO2OrchestratorLayout &layout, DeviceArena &arena, void *sm_dev_base, void *gm_heap,
        const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]
    );
    bool reset_for_reuse(
        const PTO2OrchestratorLayout &layout, void *sm_dev_base, void *gm_heap,
        const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH], const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]
    );

    // Phase 3b: write the arena-internal pointer fields (scope_tasks,
    // scope_begins, rings[].fanin_pool.base, tensor_map.{buckets,entry_pool,
    // free_entry_list,task_entry_heads}, scheduler reference).
    // Idempotent — host runs once on the image, AICPU runs once after attach.
    void wire_arena_pointers(const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SchedulerState *scheduler);

    // Forget pointers; arena owns the backing buffers.
    void destroy();
    void set_scheduler(PTO2SchedulerState *scheduler);
    void report_fatal(int32_t error_code, const char *func, const char *fmt, ...);
    void begin_scope(PTO2ScopeMode mode = PTO2ScopeMode::AUTO);
    void end_scope();
    void enable_submit_pipeline(
        int32_t orchestrator_threads, bool enqueue_submit_records = false, bool defer_submit_dependencies = false,
        bool signal_scheduler_drain = false, bool compact_deferred_records = false
    );
    uint64_t run_submit_pipeline_worker(int32_t stage_idx, int32_t phase_thread_idx = -1);
    void flush_submit_task_batch();
    void flush_submit_pipeline();
    void log_submit_pipeline_diagnostics(int32_t thread_idx) const;
    void log_four_stage_diagnostics(int32_t thread_idx) const;
    void log_active_detail_diagnostics(
        int32_t thread_idx, uint64_t active_cycles, uint64_t bind_cycles, uint64_t p_bind_cycles,
        uint64_t outer_scope_begin_cycles, uint64_t p_func_cycles, uint64_t outer_scope_end_cycles
    ) const;
    void log_submit_detail_diagnostics(int32_t thread_idx) const;
    void stop_submit_pipeline();
    TaskOutputTensors submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args);
    TaskOutputTensors submit_dummy_task(const L0TaskArgs &args);
    TaskOutputTensors alloc_tensors(const L0TaskArgs &args);
    void mark_done();
};

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
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

PTO2OrchProfilingData orchestrator_get_profiling();
#endif

#endif  // PTO_ORCHESTRATOR_H
