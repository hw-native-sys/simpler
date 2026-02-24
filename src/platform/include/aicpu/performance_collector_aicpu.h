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

#endif  // PLATFORM_AICPU_PERFORMANCE_COLLECTOR_AICPU_H_
