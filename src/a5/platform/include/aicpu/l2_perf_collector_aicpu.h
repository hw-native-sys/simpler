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
 * Handles buffer initialization, switching, and flushing.
 */

#ifndef PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_

#include "common/core_type.h"
#include "common/l2_perf_profiling.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_aicpu.h based on platform:
// Both provide unified get_sys_cnt_aicpu() interface
#include "device_time.h"

// ============= Public Interface =============

/**
 * L2 perf platform setters — called by the host (sim) or the AICPU kernel
 * entry (onboard) before `l2_perf_aicpu_init()` so AICPU code can read perf
 * state without reaching into the generic `Runtime` struct.
 */
extern "C" void set_platform_l2_perf_base(uint64_t l2_perf_data_base);
extern "C" uint64_t get_platform_l2_perf_base();
extern "C" void set_l2_swimlane_enabled(bool enable);
extern "C" bool is_l2_swimlane_enabled();

/**
 * Initialize performance profiling for `worker_count` cores.
 *
 * Caches per-core BufferState (including stable AICore staging-ring
 * pointers `state.aicore_ring_ptr`) and pops the initial L2PerfBuffer from
 * each free_queue. Reads the perf device-base pointer published via
 * `set_platform_l2_perf_base()`. Does **not** write any Handshake field —
 * profiling state lives in `KernelArgs` + AICore platform-owned slots.
 *
 * @param worker_count Number of active AICore workers
 */
void l2_perf_aicpu_init(int worker_count);

/**
 * Complete a L2PerfRecord with AICPU-side metadata after AICore task completion
 *
 * Reads from the per-core L2PerfAicoreRing dual-issue slot
 * (`s_perf_aicore_rings[core_id]->dual_issue_slots[reg_task_id & ...]`),
 * validates task_id match, and commits the record into
 * `state->current_buf_ptr->records[count++]`. Callers must pre-extract
 * fanout into a plain uint64_t array (platform layer cannot depend on
 * runtime linked-list types).
 *
 * @param core_id               Core ID owning the destination buffer (resolved via s_perf_buffer_states)
 * @param expected_reg_task_id  Register dispatch token (low 32 bits) to validate
 * @param task_id               Task identifier to write (PTO2 encoding or plain id)
 * @param func_id               Kernel function identifier
 * @param core_type             Core type (AIC/AIV)
 * @param dispatch_time         AICPU timestamp when task was dispatched
 * @param finish_time           AICPU timestamp when task completion was observed
 * @param fanout                Pre-extracted successor task ID array (nullptr if none)
 * @param fanout_count          Number of entries in fanout array (0 if none)
 *
 * Routes the destination via state->current_buf_ptr (not Handshake), so that
 * flush()-clearing current_buf_ptr deterministically halts subsequent commits
 * (they take the dropped path). Same shape as a2a3.
 */
int l2_perf_aicpu_complete_record(
    int core_id, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time, const uint64_t *fanout, int32_t fanout_count
);

/**
 * Flush remaining performance data
 *
 * Marks non-empty buffers as ready and enqueues them for host collection.
 *
 * @param thread_idx Thread index
 * @param cur_thread_cores Array of core IDs managed by this thread
 * @param core_num Number of cores managed by this thread
 */
void l2_perf_aicpu_flush_buffers(int thread_idx, const int *cur_thread_cores, int core_num);

/**
 * Initialize AICPU phase profiling
 *
 * Sets up AicpuPhaseHeader and clears per-thread phase record buffers.
 * Must be called once from thread 0 after l2_perf_aicpu_init().
 *
 * @param worker_count Number of AICore workers (used to locate phase region)
 * @param num_sched_threads Number of scheduler threads
 */
void l2_perf_aicpu_init_phase(int worker_count, int num_sched_threads);

/**
 * Record a single scheduler phase
 *
 * Appends an AicpuPhaseRecord to the specified thread's buffer.
 * When the buffer is full, switches to a new buffer via FreeQueue.
 *
 * @param thread_idx Scheduler thread index
 * @param phase_id Phase identifier
 * @param start_time Phase start timestamp
 * @param end_time Phase end timestamp
 * @param loop_iter Current loop iteration number
 * @param tasks_processed Number of tasks processed in this batch (scheduler phases), or
 *                        full PTO2 task_id encoding (ring_id << 32) | local_id (orchestrator
 *                        phases in tensormap_and_ringbuffer)
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
 * Appends an AicpuPhaseRecord for one sub-step of submit_task().
 * Uses the orchestrator's dedicated buffer slot (set via set_orch_thread_idx).
 *
 * @param phase_id Orchestrator phase identifier (ORCH_SYNC..ORCH_SCOPE_END)
 * @param start_time Phase start timestamp
 * @param end_time Phase end timestamp
 * @param submit_idx Task submission index (acts as loop_iter)
 * @param task_id Task identifier. For tensormap_and_ringbuffer, this is the full PTO2 encoding:
 * (ring_id << 32) | local_id, enabling cross-view correlation between orchestrator and scheduler swimlanes.
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

/**
 * Flush remaining phase records for a thread
 *
 * Marks the current WRITING phase buffer as READY and enqueues it
 * for host collection. Called at thread exit (analogous to l2_perf_aicpu_flush_buffers).
 *
 * @param thread_idx Thread index (scheduler thread or orchestrator)
 */
void l2_perf_aicpu_flush_phase_buffers(int thread_idx);

#endif  // PLATFORM_AICPU_L2_PERF_COLLECTOR_AICPU_H_
