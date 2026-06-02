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
 * @file l2_swimlane_collector_aicpu.h
 * @brief AICPU performance data collection interface
 *
 * Provides performance profiling management interface for AICPU side.
 * Handles buffer initialization, switching, and flushing.
 */

#ifndef PLATFORM_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_

#include "common/l2_swimlane_profiling.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_aicpu.h based on platform:
// Both provide unified get_sys_cnt_aicpu() interface
#include "aicpu/device_time.h"

// ============= Public Interface =============

/**
 * L2 swimlane handshake setters — called by the host (sim) or the AICPU kernel
 * entry (onboard) before `l2_swimlane_aicpu_init()` so AICPU code can read perf
 * state without reaching into the generic `Runtime` struct.
 *
 * Two-channel level transport (mirrors the PMU pattern):
 *   - binary on/off — `enable_profiling_flag` bit1 → `set_l2_swimlane_enabled(bool)`
 *     at kernel entry; queried via `is_l2_swimlane_enabled()`.
 *   - granular L2SwimlaneLevel — `L2SwimlaneDataHeader::l2_swimlane_level`
 *     (shared memory); read in `l2_swimlane_aicpu_init` and cached, then queried
 *     via `get_l2_swimlane_level()` for
 *     `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates.
 */
extern "C" void set_platform_l2_swimlane_base(uint64_t l2_swimlane_data_base);
extern "C" uint64_t get_platform_l2_swimlane_base();
extern "C" void set_l2_swimlane_enabled(bool enable);
extern "C" bool is_l2_swimlane_enabled();

// AICore head-table device pointer (= KernelArgs::l2_swimlane_aicore_rotation_table).
// Published by the host before AICPU init runs; AICPU init fills the table
// with the per-core `&L2SwimlaneAicoreTaskPool::head` device addresses so
// AICore can index `l2_swimlane_aicore_rotation_table[block_idx]` to find its
// active-head cache line. Moved from host into AICPU so the host stays
// decoupled from the AICore-side shared-memory layout (host previously did
// host-to-device address translation + reached into get_aicore_buffer_state
// to fill this).
extern "C" void set_platform_l2_swimlane_aicore_rotation_table(uint64_t table_addr);
extern "C" uint64_t get_platform_l2_swimlane_aicore_rotation_table();

// Typed getter for the granular perf_level (promoted from the shared-memory
// header inside l2_swimlane_aicpu_init). Gate sites should use this so the
// comparison RHS is a named L2SwimlaneLevel constant.
L2SwimlaneLevel get_l2_swimlane_level();

/**
 * Initialize performance profiling
 *
 * Sets up the AICPU buffer pool for each core and initializes tracking state.
 * Reads the perf device-base pointer published via `set_platform_l2_swimlane_base()`.
 *
 * Also primes the per-core AICore rotation channel: pops the initial
 * L2SwimlaneAicoreTaskBuffer from L2SwimlaneAicoreTaskPool::free_queue and writes its
 * address into the L2SwimlaneActiveHead channel that AICore polls per task.
 *
 * @param worker_count  Number of AICore workers (cores) to initialize
 */
void l2_swimlane_aicpu_init(int worker_count);

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
 * Called regardless of l2_swimlane_level — internally gates on AICORE_TIMING.
 *
 * @param core_id     Core index
 * @param thread_idx  Owning AICPU thread (target ready-queue)
 */
void l2_swimlane_aicpu_maybe_rotate_aicore(int core_id, int thread_idx);

/**
 * Complete a L2SwimlaneAicpuTaskRecord with AICPU-side metadata after AICore task completion
 *
 * AICore-as-producer: AICore writes start/end/task_id directly into the
 * per-core L2SwimlaneAicoreTaskBuffer at `records[reg_task_id % SIZE]`. AICPU does
 * NOT read that buffer on the hot path — it only writes AICPU-owned fields
 * (task_id, reg_task_id, func_id, core_type, dispatch_time, finish_time)
 * here, leaving start/end as zero. The host post-processor joins the AICore
 * stream into the L2SwimlaneAicpuTaskRecord stream by `reg_task_id` at flush time.
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
 *                              into L2SwimlaneAicpuTaskRecord.reg_task_id as the join key
 * @param task_id               Task identifier to write (PTO2 encoding or plain id)
 * @param func_id               Kernel function identifier
 * @param core_type             Core type (AIC/AIV)
 * @param dispatch_time         AICPU timestamp when task was dispatched
 * @param finish_time           AICPU timestamp when task completion was observed
 */
int l2_swimlane_aicpu_complete_task(
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
void l2_swimlane_aicpu_flush(int thread_idx, const int *cur_thread_cores, int core_num);

/**
 * Initialize AICPU phase profiling
 *
 * Writes phase metadata (num_sched_phase_threads, num_orch_phase_threads,
 * num_phase_cores, core_to_thread[]) into L2SwimlaneDataHeader and primes
 * per-thread sched and orch phase pools. Must be called once from thread 0
 * after l2_swimlane_aicpu_init().
 *
 * @param worker_count             Number of AICore workers (cores) — used to
 *                                 resolve the phase region's offset relative
 *                                 to the L2Swimlane base
 * @param num_sched_phase_threads  Number of sched-phase pools to prime
 * @param num_orch_phase_threads   Number of orch-phase pools to prime
 *                                 (typically 1; in orch_to_sched mode =
 *                                 num_aicpu_threads)
 */
void l2_swimlane_aicpu_init_phase(int worker_count, int num_sched_phase_threads, int num_orch_phase_threads);

/**
 * Record a single scheduler phase
 *
 * Appends an L2SwimlaneAicpuSchedPhaseRecord to the specified thread's sched
 * pool. Silently drops records when the buffer is full or the pool was not
 * primed (init failed for this thread).
 *
 * @param thread_idx       Scheduler thread index
 * @param kind             Complete or Dispatch
 * @param start_time       Phase start timestamp
 * @param end_time         Phase end timestamp
 * @param loop_iter        Current scheduler-loop iteration number
 * @param tasks_processed  Tasks processed in this phase batch
 * @param pop_hit          Dispatch delta since last emit (0 for Complete)
 * @param pop_miss         Dispatch delta since last emit (0 for Complete)
 */
void l2_swimlane_aicpu_record_sched_phase(
    int thread_idx, L2SwimlaneSchedPhaseKind kind, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint32_t tasks_processed, uint32_t pop_hit = 0, uint32_t pop_miss = 0
);

/**
 * Set orchestrator thread index for per-task phase recording
 *
 * Must be called once from the orchestrator thread before any
 * l2_swimlane_aicpu_record_orch_phase() calls.
 *
 * @param thread_idx Thread index for the orchestrator (typically num_sched_threads;
 *                   in orch_to_sched mode each scheduler thread sets its own)
 */
void l2_swimlane_aicpu_set_orch_thread_idx(int thread_idx);

/**
 * Record one orchestrator submit envelope
 *
 * Appends an L2SwimlaneAicpuOrchPhaseRecord covering an entire submit_task()
 * / alloc_tensors() call. Uses the orchestrator's dedicated orch-phase pool
 * (chosen via set_orch_thread_idx).
 *
 * @param start_time  Submit start timestamp
 * @param end_time    Submit end timestamp
 * @param task_id     Task identifier. For tensormap_and_ringbuffer, full PTO2
 *                    encoding: (ring_id << 32) | local_id, enabling
 *                    cross-view correlation between orchestrator and
 *                    scheduler swimlanes.
 * @param submit_idx  Monotonic submit counter
 */
void l2_swimlane_aicpu_record_orch_phase(uint64_t start_time, uint64_t end_time, uint64_t task_id, uint32_t submit_idx);

/**
 * Write core-to-thread assignment mapping to shared memory.
 *
 * Callers invoke `l2_swimlane_aicpu_init_core_assignments(total_cores)` once, then
 * `l2_swimlane_aicpu_write_core_assignments_for_thread(t, ids, n)` for every
 * scheduler thread.
 */
void l2_swimlane_aicpu_init_core_assignments(int total_cores);
void l2_swimlane_aicpu_write_core_assignments_for_thread(int thread_idx, const int *core_ids, int core_num);

/**
 * Flush the remaining scheduler-phase records for a scheduler thread.
 *
 * Marks the thread's current WRITING sched-phase buffer as READY and enqueues
 * it for host collection. Called at scheduler-thread exit.
 *
 * @param thread_idx Scheduler thread index (= sched pool index = ready queue)
 */
void l2_swimlane_aicpu_flush_sched_phase_buffer(int thread_idx);

/**
 * Flush the remaining orchestrator-phase records (single orch instance, pool
 * ordinal 0). Called once by the orchestrator thread at orchestration end.
 *
 * @param thread_idx Calling (orchestrator) AICPU thread index — selects the
 *                   ready queue to enqueue into. The pool/lane tag is ordinal 0.
 */
void l2_swimlane_aicpu_flush_orch_phase_buffer(int thread_idx);

#endif  // PLATFORM_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_
