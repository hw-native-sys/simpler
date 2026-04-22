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
 * @file l2_perf_collector_aicpu.h
 * @brief AICPU performance data collection interface
 *
 * Provides performance profiling management interface for AICPU side.
 * Handles buffer initialization and per-record completion. In the memcpy-based
 * collection design, Host pre-allocates one L2PerfBuffer per core and one
 * PhaseBuffer per thread; AICPU writes directly into them until full, after
 * which further records are silently dropped.
 */

#ifndef PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_

#include "common/l2_perf_profiling.h"
#include "runtime.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_aicpu.h based on platform:
// Both provide unified get_sys_cnt_aicpu() interface
#include "device_time.h"

// ============= Public Interface =============

/**
 * Initialize performance profiling
 *
 * Sets up double buffers for each core and initializes tracking state.
 *
 * @param runtime Runtime instance pointer
 */
void l2_perf_aicpu_init_profiling(Runtime *runtime);

/**
 * Complete a L2PerfRecord with AICPU-side metadata after AICore task completion
 *
 * Reads l2_perf_buf->count, validates task_id match against the latest record,
 * and fills all AICPU-side fields. Returns -1 and silently drops the record
 * when the buffer is full (count >= PLATFORM_PROF_BUFFER_SIZE). Callers must
 * pre-extract fanout into a plain uint64_t array (platform layer cannot depend
 * on runtime linked-list types).
 *
 * @param l2_perf_buf              L2PerfBuffer pointer (from handshake l2_perf_records_addr)
 * @param expected_reg_task_id  Register dispatch token (low 32 bits) to validate
 * @param task_id               Task identifier to write (PTO2 encoding or plain id)
 * @param func_id               Kernel function identifier
 * @param core_type             Core type (AIC/AIV)
 * @param dispatch_time         AICPU timestamp when task was dispatched
 * @param finish_time           AICPU timestamp when task completion was observed
 * @param fanout                Pre-extracted successor task ID array (nullptr if none)
 * @param fanout_count          Number of entries in fanout array (0 if none)
 */
int l2_perf_aicpu_complete_record(
    L2PerfBuffer *l2_perf_buf, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time, const uint64_t *fanout, int32_t fanout_count
);

/**
 * Update total task count in performance header
 *
 * Allows dynamic update of total_tasks as orchestrator makes progress.
 * Used by tensormap_and_ringbuffer runtime where task count grows incrementally.
 *
 * @param runtime Runtime instance pointer
 * @param total_tasks Current total task count
 */
void l2_perf_aicpu_update_total_tasks(Runtime *runtime, uint32_t total_tasks);

/**
 * Initialize AICPU phase profiling
 *
 * Sets up AicpuPhaseHeader and clears per-thread phase record buffers.
 * Must be called once from thread 0 after l2_perf_aicpu_init_profiling().
 *
 * @param runtime Runtime instance pointer
 * @param num_sched_threads Number of scheduler threads
 */
void l2_perf_aicpu_init_phase_profiling(Runtime *runtime, int num_sched_threads);

/**
 * Record a single scheduler phase
 *
 * Appends an AicpuPhaseRecord to the specified thread's buffer.
 * Silently drops records when the buffer is full.
 *
 * @param thread_idx Scheduler thread index
 * @param phase_id Phase identifier
 * @param start_time Phase start timestamp
 * @param end_time Phase end timestamp
 * @param loop_iter Current loop iteration number
 * @param tasks_processed Number of tasks processed in this batch (scheduler phases), or
 *                        full PTO2 task_id encoding (ring_id << 32) | local_id (orchestrator
 *                        phases in multi-ring runtimes: tensormap_and_ringbuffer, aicpu_build_graph)
 */
void l2_perf_aicpu_record_phase(
    int thread_idx, AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint64_t tasks_processed
);

/**
 * Write orchestrator cumulative summary
 *
 * Writes the orchestrator's accumulated profiling data to shared memory
 * for host-side collection.
 *
 * @param src Pointer to populated AicpuOrchSummary (magic field is set internally)
 */
void l2_perf_aicpu_write_orch_summary(const AicpuOrchSummary *src);

/**
 * Set orchestrator thread index for per-task phase recording
 *
 * Must be called once from the orchestrator thread before any
 * l2_perf_aicpu_record_orch_phase() calls.
 *
 * @param thread_idx Thread index for the orchestrator (typically num_sched_threads)
 */
void l2_perf_aicpu_set_orch_thread_idx(int thread_idx);

/**
 * Record a single orchestrator phase
 *
 * Appends an AicpuPhaseRecord for one sub-step of pto2_submit_task().
 * Uses the orchestrator's dedicated buffer slot (set via set_orch_thread_idx).
 *
 * @param phase_id Orchestrator phase identifier (ORCH_SYNC..ORCH_SCOPE_END)
 * @param start_time Phase start timestamp
 * @param end_time Phase end timestamp
 * @param submit_idx Task submission index (acts as loop_iter)
 * @param task_id Task identifier. For multi-ring runtimes (tensormap_and_ringbuffer, aicpu_build_graph), this is the
 * full PTO2 encoding: (ring_id << 32) | local_id, enabling cross-view correlation between orchestrator and scheduler
 * swimlanes.
 */
void l2_perf_aicpu_record_orch_phase(
    AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t submit_idx, uint64_t task_id
);

/**
 * Write core-to-thread assignment mapping to shared memory.
 *
 * Callers invoke `l2_perf_aicpu_init_core_assignments(total_cores)` once, then
 * `l2_perf_aicpu_write_core_assignments_for_thread(t, ids, n)` for every
 * scheduler thread.
 */
void l2_perf_aicpu_init_core_assignments(int total_cores);
void l2_perf_aicpu_write_core_assignments_for_thread(int thread_idx, const int *core_ids, int core_num);

#endif  // PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_
