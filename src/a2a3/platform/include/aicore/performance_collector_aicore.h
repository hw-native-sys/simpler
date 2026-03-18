/**
 * @file performance_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#ifndef PLATFORM_AICORE_PERFORMANCE_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_PERFORMANCE_COLLECTOR_AICORE_H_

#include "common/perf_profiling.h"
#include "aicore/aicore.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/platform/a2a3/aicore/inner_kernel.h (real hardware)
// - src/platform/a2a3sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * Record task execution performance data
 *
 * Writes performance metrics to the provided buffer. Buffer management
 * and status tracking are handled by AICPU.
 *
 * AICore records task_id and timestamps only. AICPU fills func_id and
 * core_type at completion time from TaskDescriptor.
 *
 * @param perf_buf Performance buffer pointer
 * @param task_id Task ID
 * @param start_time Start timestamp
 * @param end_time End timestamp
 * @param kernel_ready_time Kernel ready timestamp
 */
__aicore__ __attribute__((always_inline))
static inline void perf_aicore_record_task(
    __gm__ PerfBuffer* perf_buf,
    uint32_t task_id,
    uint64_t start_time,
    uint64_t end_time,
    uint64_t kernel_ready_time) {

    // Read current buffer count
    dcci(&perf_buf->count, SINGLE_CACHE_LINE);
    uint32_t idx = perf_buf->count;

    if (idx >= PLATFORM_PROF_BUFFER_SIZE) {
        return;
    }

    __gm__ PerfRecord* record = &perf_buf->records[idx];

    // Write record data (func_id and core_type filled by AICPU at completion)
    record->start_time = start_time;
    record->end_time = end_time;
    record->kernel_ready_time = kernel_ready_time;
    record->mixed_task_id = task_id;
    record->fanout_count = 0;
    record->fanout_filled = 0;

    perf_buf->count = idx + 1;

    // Flush cache to make data visible
    dcci(&perf_buf->count, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(record, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}

#endif  // PLATFORM_AICORE_PERFORMANCE_COLLECTOR_AICORE_H_
