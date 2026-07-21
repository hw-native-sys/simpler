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

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_

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
 * Pre-dispatch hook for AICore buffer rotation and per-pool stats.
 *
 * Called from the dispatch path (scheduler_dispatch in tensormap_and_ringbuffer,
 * aicpu_executor in host_build_graph) immediately BEFORE `write_reg(DATA_MAIN_BASE)`
 * for each AICore task. Two responsibilities:
 *
 *   1. Maintain the per-core AICPU-side dispatch count.
 *   2. Rotate the AICore buffer when the count is about to cross a
 *      PLATFORM_AICORE_BUFFER_SIZE boundary — publish the next buffer from
 *      free_queue to AICore and STASH the just-filled buffer for deferred,
 *      ACK-gated release (see l2_swimlane_aicpu_on_aicore_ack).
 *   3. Bump the AICore pool's `total_record_count` so host reconcile
 *      (total == collected + dropped) stays accurate at all levels —
 *      including AICORE_TIMING (level=1), where `complete_task` is bypassed.
 *
 * Race safety: rotation runs BEFORE the dispatch register write. The
 * completion-before-dispatch invariant proves prior tasks FIN'd, but
 * tensormap_and_ringbuffer may write FIN before the swimlane record, so the old
 * buffer's tail record may not have drained at rotation time. The old buffer is
 * therefore not enqueued here; it is released once AICore ACKs the boundary
 * dispatch (whose token is `reg_task_id`) or, when a runtime does not wire the
 * ACK hook, by the next-rotation / run-end backstop.
 *
 * No-op if l2_swimlane is disabled or `core_id` is out of range.
 *
 * @param core_id      Core index this dispatch targets
 * @param thread_idx   Owning AICPU thread (target ready-queue for rotation)
 * @param reg_task_id  Per-core dispatch token of this dispatch; when it is the
 *                     boundary dispatch it becomes the ACK gate for releasing
 *                     the just-filled buffer
 */
void l2_swimlane_aicpu_on_aicore_dispatch(int core_id, int thread_idx, uint32_t reg_task_id);

/**
 * Post-completion hook: release an ACK-gated AICore buffer.
 *
 * Called from the completion path on every matched AICore COND event (ACK or
 * FIN). When a rotation has stashed the previous buffer with a gate equal to
 * `reg_task_id`, that event proves AICore reached at least the new buffer's
 * first-task ACK — hence passed the old buffer's last record dcci+dsb — so the
 * stashed buffer is enqueued to the ready queue. No-op when nothing is stashed
 * or the token does not match the gate. Runtimes that write the record before
 * FIN need not wire this; the next-rotation / run-end backstop covers them.
 *
 * @param core_id      Core index the COND event was observed on
 * @param thread_idx   Owning AICPU thread (target ready-queue)
 * @param reg_task_id  Task token extracted from the observed COND value
 */
void l2_swimlane_aicpu_on_aicore_ack(int core_id, int thread_idx, uint32_t reg_task_id);

/**
 * Commit an AICPU-side timing record for one completed task.
 *
 * AICore-as-producer: identity (task_token_raw) and AICore-side timing
 * (start/end) live in the per-core L2SwimlaneAicoreTaskRecord stream;
 * core_type is published once by the host into the collector
 * (L2SwimlaneCollector::set_core_types); func_id is resolved post-process
 * from deps.json. This function therefore only needs to record the two
 * AICPU-only timestamps plus the host-side join key.
 *
 * Per-core counter accounting:
 *   total_record_count++       — every commit attempt (success or failure)
 *   dropped_record_count++     — capacity-driven drop (no free buffer /
 *                                queue full); actionable via
 *                                PLATFORM_PROF_BUFFERS_PER_CORE
 *
 * @param core_id               Core index — used to resolve buffer state and update counters
 * @param thread_idx            Owning AICPU thread (used when rotating records buffer)
 * @param reg_task_id           Per-core dispatch token (low 32 bits) — host join
 *                              key against the AICore record stream
 * @param dispatch_time         AICPU timestamp when task was dispatched
 * @param finish_time           AICPU timestamp when task completion was observed
 */
int l2_swimlane_aicpu_complete_task(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t dispatch_time, uint64_t finish_time
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
 *                                 (typically 1)
 */
void l2_swimlane_aicpu_init_phase(int worker_count, int num_sched_phase_threads, int num_orch_phase_threads);

/**
 * Record a single scheduler phase
 *
 * Appends an L2SwimlaneAicpuSchedPhaseRecord to the specified thread's sched
 * pool. Silently drops records when the buffer is full or the pool was not
 * primed (init failed for this thread).
 *
 * Queue-depth snapshots record the per-shape shared ready-queue occupancy at
 * phase boundaries. Pass nullptr for either array when not capturing (the
 * record's corresponding slot is zero-filled).
 *
 * @param thread_idx       Scheduler thread index
 * @param kind             Scheduler phase kind; DummyTask uses l2_swimlane_aicpu_record_dummy_task
 * @param start_time       Phase start timestamp
 * @param end_time         Phase end timestamp
 * @param loop_iter        Current scheduler-loop iteration number
 * @param tasks_processed  Tasks processed in this phase batch
 * @param pop_hit          Ready-queue hit delta since the previous Dispatch emit; ignored for other kinds
 *                         (Complete, Release, Dummy, EarlyDispatch, Resolve, Drain, DrainPrepare, DrainPublish)
 * @param pop_miss         Ready-queue miss delta since the previous Dispatch emit; ignored for other kinds
 *                         (Complete, Release, Dummy, EarlyDispatch, Resolve, Drain, DrainPrepare, DrainPublish)
 * @param shared_at_start  Per-shape sched.ready_queues[shape].size() at phase start (may be nullptr)
 * @param shared_at_end    Per-shape sched.ready_queues[shape].size() at phase end (may be nullptr)
 */
void l2_swimlane_aicpu_record_sched_phase(
    int thread_idx, L2SwimlaneSchedPhaseKind kind, uint64_t start_time, uint64_t end_time, uint32_t loop_iter,
    uint32_t tasks_processed, uint32_t pop_hit = 0, uint32_t pop_miss = 0, const int16_t *shared_at_start = nullptr,
    const int16_t *shared_at_end = nullptr
);

/**
 * Record the completion point of one dependency-only dummy task.
 *
 * @param thread_idx     Scheduler thread that completed the task
 * @param complete_time  Timestamp sampled immediately before completion propagation
 * @param loop_iter      Current scheduler-loop iteration number
 * @param task_id        Full PTO2 task identity
 */
void l2_swimlane_aicpu_record_dummy_task(int thread_idx, uint64_t complete_time, uint32_t loop_iter, uint64_t task_id);

/**
 * Set orchestrator thread index for per-task phase recording
 *
 * Must be called once from the orchestrator thread before any
 * l2_swimlane_aicpu_record_orch_phase() calls.
 *
 * @param thread_idx Thread index for the orchestrator (typically num_sched_threads)
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

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_L2_SWIMLANE_COLLECTOR_AICPU_H_
