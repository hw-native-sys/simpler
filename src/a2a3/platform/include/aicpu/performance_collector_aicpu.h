/**
 * @file performance_collector_aicpu.h
 * @brief AICPU performance data collection interface
 *
 * Provides performance profiling management interface for AICPU side.
 * Handles buffer initialization, switching, and flushing.
 */

#ifndef PLATFORM_AICPU_PERFORMANCE_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_PERFORMANCE_COLLECTOR_AICPU_H_

#include "common/perf_profiling.h"
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
void perf_aicpu_init_profiling(Runtime* runtime);

/**
 * Record dispatch and finish timestamps
 *
 * Updates task record with AICPU-side timing information.
 *
 * @param record PerfRecord pointer to update
 * @param dispatch_time Dispatch timestamp
 * @param finish_time Finish timestamp
 */
void perf_aicpu_record_dispatch_and_finish_time(PerfRecord* record,
                                                 uint64_t dispatch_time,
                                                 uint64_t finish_time);

/**
 * Switch performance buffer when current buffer is full
 *
 * Checks buffer capacity and switches to alternate buffer if needed.
 *
 * @param runtime Runtime instance pointer
 * @param core_id Core ID
 * @param thread_idx Thread index
 */
void perf_aicpu_switch_buffer(Runtime* runtime, int core_id, int thread_idx);

/**
 * Flush remaining performance data
 *
 * Marks non-empty buffers as ready and enqueues them for host collection.
 *
 * @param runtime Runtime instance pointer
 * @param thread_idx Thread index
 * @param cur_thread_cores Array of core IDs managed by this thread
 * @param core_num Number of cores managed by this thread
 */
void perf_aicpu_flush_buffers(Runtime* runtime,
                               int thread_idx,
                               const int* cur_thread_cores,
                               int core_num);

/**
 * Update total task count in performance header
 *
 * Allows dynamic update of total_tasks as orchestrator makes progress.
 * Used by tensormap_and_ringbuffer runtime where task count grows incrementally.
 *
 * @param runtime Runtime instance pointer
 * @param total_tasks Current total task count
 */
void perf_aicpu_update_total_tasks(Runtime* runtime, uint32_t total_tasks);

/**
 * Initialize AICPU phase profiling
 *
 * Sets up AicpuPhaseHeader and clears per-thread phase record buffers.
 * Must be called once from thread 0 after perf_aicpu_init_profiling().
 *
 * @param runtime Runtime instance pointer
 * @param num_sched_threads Number of scheduler threads
 */
void perf_aicpu_init_phase_profiling(Runtime* runtime, int num_sched_threads);

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
 * @param tasks_processed Number of tasks processed in this phase
 */
void perf_aicpu_record_phase(int thread_idx,
                              AicpuPhaseId phase_id,
                              uint64_t start_time, uint64_t end_time,
                              uint32_t loop_iter, uint32_t tasks_processed);

/**
 * Write orchestrator cumulative summary
 *
 * Writes the orchestrator's accumulated profiling data to shared memory
 * for host-side collection.
 *
 * @param src Pointer to populated AicpuOrchSummary (magic field is set internally)
 */
void perf_aicpu_write_orch_summary(const AicpuOrchSummary* src);

/**
 * Set orchestrator thread index for per-task phase recording
 *
 * Must be called once from the orchestrator thread before any
 * perf_aicpu_record_orch_phase() calls.
 *
 * @param thread_idx Thread index for the orchestrator (typically num_sched_threads)
 */
void perf_aicpu_set_orch_thread_idx(int thread_idx);

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
 */
void perf_aicpu_record_orch_phase(AicpuPhaseId phase_id,
                                   uint64_t start_time, uint64_t end_time,
                                   uint32_t submit_idx);

#endif  // PLATFORM_AICPU_PERFORMANCE_COLLECTOR_AICPU_H_
