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

#include "common/l2_perf_profiling.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_aicpu.h based on platform:
// Both provide unified get_sys_cnt_aicpu() interface
#include "device_time.h"

// ============= Public Interface =============

/**
 * L2 perf handshake setters — called by the host (sim) or the AICPU kernel
 * entry (onboard) before `l2_perf_aicpu_init()` so AICPU code can read perf
 * state without reaching into the generic `Runtime` struct.
 *
 * Two-channel level transport (mirrors the PMU pattern):
 *   - binary on/off — `enable_profiling_flag` bit1 → `set_l2_swimlane_enabled(bool)`
 *     at kernel entry; queried via `is_l2_swimlane_enabled()`.
 *   - granular L2PerfLevel — `L2PerfDataHeader::l2_perf_level`
 *     (shared memory); read in `l2_perf_aicpu_init` and cached, then queried
 *     via `get_l2_perf_level()` for
 *     `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates.
 */
extern "C" void set_platform_l2_perf_base(uint64_t l2_perf_data_base);
extern "C" uint64_t get_platform_l2_perf_base();
extern "C" void set_l2_swimlane_enabled(bool enable);
extern "C" bool is_l2_swimlane_enabled();

// AICore rotation-table device pointer (= KernelArgs::aicore_ring_addr).
// Published by the host before AICPU init runs; AICPU init fills the table
// with the per-core `&L2PerfAicoreBufferState::rotation` device addresses so
// AICore can index `aicore_ring_addr[block_idx]` to find its rotation channel.
// Moved from host into AICPU so the host stays decoupled from the AICore-side
// shared-memory layout (host previously did host-to-device address translation
// + reached into get_aicore_buffer_state to fill this).
extern "C" void set_platform_aicore_rotation_table(uint64_t table_addr);
extern "C" uint64_t get_platform_aicore_rotation_table();

// Typed getter for the granular perf_level (promoted from the shared-memory
// header inside l2_perf_aicpu_init). Gate sites should use this so the
// comparison RHS is a named L2PerfLevel constant.
L2PerfLevel get_l2_perf_level();

/**
 * Initialize performance profiling
 *
 * Sets up the AICPU buffer pool for each core and initializes tracking state.
 * Reads the perf device-base pointer published via `set_platform_l2_perf_base()`.
 *
 * Also primes the per-core AICore rotation channel: pops the initial
 * L2PerfAicoreBuffer from L2PerfAicoreBufferState::free_queue and writes its
 * address into the AicoreRotation channel that AICore polls per task.
 *
 * @param worker_count  Number of AICore workers (cores) to initialize
 */
void l2_perf_aicpu_init(int worker_count);

/**
 * Rotate the AICore buffer for a given core, if needed.
 *
 * Called from the dispatch path (scheduler_dispatch in tensormap_and_ringbuffer,
 * aicpu_executor in host_build_graph) immediately before write_reg(DATA_MAIN_BASE)
 * for each task. Increments the per-core dispatch counter and, when it crosses
 * a PLATFORM_AICORE_BUFFER_SIZE boundary, enqueues the current AICore buffer
 * to the ready queue (kind=2) and pops the next one from free_queue.
 *
 * Race safety: rotation happens BEFORE the dispatch register write, so by the
 * runtime's completion-before-dispatch invariant all prior tasks have FIN'd
 * (and AICore has finished writing their records into the old buffer) before
 * the old buffer enters the ready queue.
 *
 * Called regardless of l2_perf_level — internally gates on AICORE_TIMING.
 *
 * @param core_id     Core index
 * @param thread_idx  Owning AICPU thread (target ready-queue)
 */
void l2_perf_aicpu_maybe_rotate_aicore(int core_id, int thread_idx);

/**
 * Complete a L2PerfRecord with AICPU-side metadata after AICore task completion
 *
 * AICore-as-producer: AICore writes start/end/task_id directly into the
 * per-core L2PerfAicoreBuffer at `records[reg_task_id % SIZE]`. AICPU does
 * NOT read that buffer on the hot path — it only writes AICPU-owned fields
 * (task_id, reg_task_id, func_id, core_type, dispatch_time, finish_time)
 * here, leaving start/end as zero. The host post-processor joins the AICore
 * stream into the L2PerfRecord stream by `reg_task_id` at flush time.
 *
 * Per-core counter accounting:
 *   total_record_count++       — every commit attempt (success or failure)
 *   dropped_record_count++     — capacity-driven drop (no free buffer / queue
 *                                full); actionable via
 *                                PLATFORM_PROF_BUFFERS_PER_CORE
 *
 * @param core_id               Core index — used to resolve buffer state and update counters
 * @param thread_idx            Owning AICPU thread (used when rotating records buffer)
 * @param expected_reg_task_id  Register dispatch token (low 32 bits) — written
 *                              into L2PerfRecord.reg_task_id as the join key
 * @param task_id               Task identifier to write (PTO2 encoding or plain id)
 * @param func_id               Kernel function identifier
 * @param core_type             Core type (AIC/AIV)
 * @param dispatch_time         AICPU timestamp when task was dispatched
 * @param finish_time           AICPU timestamp when task completion was observed
 */
int l2_perf_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t expected_reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type,
    uint64_t dispatch_time, uint64_t finish_time
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
 * @param worker_count       Number of AICore workers (cores) — used to resolve
 *                           the phase region's offset relative to the L2Perf base
 * @param num_sched_threads  Number of scheduler threads
 */
void l2_perf_aicpu_init_phase(int worker_count, int num_sched_threads);

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
 *                        phases in tensormap_and_ringbuffer)
 * @param extra1, extra2  Phase-specific delta counters (see AicpuPhaseRecord doc).
 *                        SCHED_DISPATCH uses extra1=pop_hit, extra2=pop_miss; other
 *                        phases pass 0.
 */
void l2_perf_aicpu_record_phase(
    int thread_idx, AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint64_t tasks_processed, uint32_t extra1 = 0, uint32_t extra2 = 0
);

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
 * Record one orchestrator submit envelope
 *
 * Appends an AicpuPhaseRecord covering an entire submit_task() / alloc_tensors()
 * call. Uses the orchestrator's dedicated buffer slot (set via
 * set_orch_thread_idx). Per-sub-step phase records (ORCH_SYNC..ORCH_FANIN)
 * were dropped — the per-step cumulatives (`g_orch_*_cycle`) in the
 * cold-path log carry the breakdown that those records were duplicating.
 *
 * @param phase_id Always AicpuPhaseId::ORCH_SUBMIT. (Param kept for API
 *                 stability; legacy values are ignored by the host parser.)
 * @param start_time Submit start timestamp
 * @param end_time Submit end timestamp
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
