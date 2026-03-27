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
 * Complete a PerfRecord with AICPU-side metadata after AICore task completion
 *
 * Reads perf_buf->count, validates task_id match against the latest record,
 * and fills all AICPU-side fields. Callers must pre-extract fanout into a
 * plain int32_t array (platform layer cannot depend on runtime linked-list types).
 *
 * @param perf_buf         PerfBuffer pointer (from handshake perf_records_addr)
 * @param expected_task_id Task ID to validate against the latest record
 * @param func_id          Kernel function identifier
 * @param core_type        Core type (AIC/AIV)
 * @param dispatch_time    AICPU timestamp when task was dispatched
 * @param finish_time      AICPU timestamp when task completion was observed
 * @param ring_id          Ring layer (0 for single-ring / host_build_graph)
 * @param fanout           Pre-extracted successor task ID array (nullptr if none)
 * @param fanout_count     Number of entries in fanout array (0 if none)
 */
int perf_aicpu_complete_record(PerfBuffer* perf_buf,
    uint32_t expected_task_id,
    uint32_t func_id,
    CoreType core_type,
    uint64_t dispatch_time,
    uint64_t finish_time,
    uint8_t ring_id,
    const int32_t* fanout,
    int32_t fanout_count);

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
void perf_aicpu_flush_buffers(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num);

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
 * @param num_orch_threads Number of orchestrator threads (may become schedulers after transition)
 */
void perf_aicpu_init_phase_profiling(Runtime* runtime, int num_sched_threads, int num_orch_threads = 1);

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
    uint64_t start_time,
    uint64_t end_time,
    uint32_t loop_iter,
    uint32_t tasks_processed);

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
 * @param task_id Task ID (stored in tasks_processed field for task tracking)
 */
void perf_aicpu_record_orch_phase(
    AicpuPhaseId phase_id, uint64_t start_time, uint64_t end_time, uint32_t submit_idx, uint32_t task_id);

/**
 * Write core-to-thread assignment mapping to shared memory
 *
 * Records which scheduler thread manages each core_id.
 * Called once after orchestration completes (not on the scheduler hot path).
 *
 * @param core_assignments 2D array [thread_idx][i] = core_id
 * @param core_counts Per-thread core count array
 * @param num_threads Number of scheduler threads
 * @param total_cores Total number of cores
 */
void perf_aicpu_write_core_assignments(const int core_assignments[][PLATFORM_MAX_CORES_PER_THREAD],
    const int* core_counts,
    int num_threads,
    int total_cores);

/**
 * Flush remaining phase records for a thread
 *
 * Marks the current WRITING phase buffer as READY and enqueues it
 * for host collection. Called at thread exit (analogous to perf_aicpu_flush_buffers).
 *
 * @param thread_idx Thread index (scheduler thread or orchestrator)
 */
void perf_aicpu_flush_phase_buffers(int thread_idx);

#endif  // PLATFORM_AICPU_PERFORMANCE_COLLECTOR_AICPU_H_
