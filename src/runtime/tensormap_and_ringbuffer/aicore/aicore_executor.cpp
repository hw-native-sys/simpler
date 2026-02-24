#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"
#include "common/perf_profiling.h"
#include "common/memory_barrier.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * @brief Record task execution performance data
 *
 * This function records the performance metrics of a task execution to the profiling buffer.
 * It writes the task timing data, metadata, and marks the buffer as full when needed.
 *
 * Note: Fanout information is filled by AICPU after task completion (not recorded here).
 *
 * @param my_hank Pointer to the handshake structure containing perf buffer info
 * @param payload Pointer to the PTO2DispatchPayload structure
 * @param start_time Task start timestamp
 * @param end_time Task end timestamp
 * @param block_idx AICore block index
 * @param core_type Core type (AIC or AIV)
 * @param kernel_ready_time Kernel ready timestamp (when AICore entered main loop)
 */
__aicore__ __attribute__((always_inline)) static void record_task_performance(
    __gm__ Handshake* my_hank,
    __gm__ PTO2DispatchPayload* payload,
    uint64_t start_time,
    uint64_t end_time,
    int block_idx,
    CoreType core_type,
    uint64_t kernel_ready_time) {

    // Check if buffer is available for writing
    if (my_hank->perf_buffer_status != 0) {
        return;  // Buffer full, skip recording
    }

    // Get current performance buffer pointer
    __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;

    // Get current count (no atomic operation needed - single writer)
    rmb();
    uint32_t idx = perf_buf->count;

    // Check if buffer has space
    if (idx < PLATFORM_PROF_BUFFER_SIZE) {
        // Get pointer to the record slot
        __gm__ PerfRecord* record = (__gm__ PerfRecord*)&perf_buf->records[idx];

        // Write record data (only essential fields, fanout filled by AICPU)
        record->start_time = start_time;
        record->end_time = end_time;
        record->kernel_ready_time = kernel_ready_time;
        record->task_id = payload->task_id;      // Use payload->task_id
        record->func_id = payload->kernel_id;    // Use payload->kernel_id
        record->core_id = block_idx;
        record->core_type = core_type;

        // Increment count after writing record
        perf_buf->count = idx + 1;

        // Write memory barrier: ensure performance data is visible to Host
        wmb();

        // Check if buffer is full after this write
        if (perf_buf->count >= PLATFORM_PROF_BUFFER_SIZE) {
            my_hank->perf_buffer_status = 1;  // Notify AICPU: buffer full
        }
    } else {
        // Buffer is already full
        my_hank->perf_buffer_status = 1;
    }
}

/**
 * Execute task from PTO2DispatchPayload.
 *
 * Directly accesses PTO2DispatchPayload fields for task execution,
 * matching ref_runtime implementation for a2a3 compatibility.
 *
 * @param task_ptr Pointer to PTO2DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ void* task_ptr) {
    __gm__ PTO2DispatchPayload* payload = reinterpret_cast<__gm__ PTO2DispatchPayload*>(task_ptr);
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(payload->args));
}

/**
 * AICore main execution loop
 *
 * Implements the AICPU-AICore handshake protocol:
 * 1. Wait for AICPU ready signal
 * 2. Signal AICore ready
 * 3. Poll for tasks and execute until quit signal
 *
 * Task dispatch uses PTO2DispatchPayload from PTO2 shared memory.
 * Supports performance profiling when runtime->enable_profiling is true.
 *
 * @param runtime Pointer to Runtime in global memory
 * @param block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready and report core type
    my_hank->core_type = core_type;        // Report core type to AICPU
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)

    // Check if profiling is enabled
    bool profiling_enabled = runtime->enable_profiling;

    // Record kernel ready time (before entering main loop)
    // This timestamp represents when the AICore is ready to execute tasks
    // but hasn't started executing any task yet.
    // Used for: 1) Startup overhead analysis, 2) Cross-core time alignment
    uint64_t kernel_ready_time = 0;
    if (profiling_enabled) {
        kernel_ready_time = get_sys_cnt();
    }

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0)
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ PTO2DispatchPayload* payload =
                reinterpret_cast<__gm__ PTO2DispatchPayload*>(my_hank->task);

            // Performance profiling: record start time
            uint64_t start_time = 0;
            if (profiling_enabled) {
                start_time = get_sys_cnt();
            }

            // Execute the task
            execute_task(reinterpret_cast<__gm__ void*>(my_hank->task));

            // Performance profiling: record task execution
            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt();
                record_task_performance(my_hank, payload, start_time, end_time,
                                      block_idx, core_type, kernel_ready_time);
            }

            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}
