#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"
#include "common/perf_profiling.h"
#include "aicore/performance_collector_aicore.h"
#include "common/platform_config.h"  // Register-based communication (RegId, AICoreStatus)

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

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

    // Ensure all memory writes are visible to other cores
    pipe_barrier(PIPE_ALL);
}

/**
 * AICore main execution loop
 *
 * Implements the AICPU-AICore register-based dispatch protocol:
 * 1. Wait for AICPU ready signal via handshake buffer
 * 2. Report physical core ID and core type, signal AICore ready
 * 3. Poll DATA_MAIN_BASE register for task dispatch until exit signal
 *
 * Task dispatch uses PTO2DispatchPayload from per-core payload array.
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
        dcci(my_hank, SINGLE_CACHE_LINE);
    }

    // Phase 2: Report physical core ID and core type, signal ready
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Report initial idle status via register
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // Read per-core payload address from hank->task (written by AICPU before aicpu_ready)
    __gm__ PTO2DispatchPayload* my_payload =
        reinterpret_cast<__gm__ PTO2DispatchPayload*>(my_hank->task);

    bool profiling_enabled = runtime->enable_profiling;
    uint64_t kernel_ready_time = 0;
    if (profiling_enabled) {
        kernel_ready_time = get_sys_cnt_aicore();
    }

    // Phase 3: Main execution loop - poll register for tasks until exit signal
    volatile uint32_t task_id = 0;
    volatile uint32_t last_task_id = 0;

    while (true) {
        task_id = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (task_id == AICORE_EXIT_SIGNAL) {
            break;
        }

        // Execute task if new (task_id encoding: 0=idle, task_id+1=task)
        if (task_id == 0 || task_id == last_task_id) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            // Invalidate cache to read fresh payload written by AICPU
            dcci(my_payload, ENTIRE_DATA_CACHE);

            __gm__ PTO2DispatchPayload* payload = my_payload;

            write_reg(RegId::COND, MAKE_ACK_VALUE(payload->task_id));

            // Performance profiling: record start time
            uint64_t start_time = 0;
            if (profiling_enabled) {
                start_time = get_sys_cnt_aicore();
            }

            // Execute the task
            execute_task(reinterpret_cast<__gm__ void*>(payload));

            // Performance profiling: record task execution
            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;
                perf_aicore_record_task(perf_buf, payload->task_id, payload->kernel_id,
                                       start_time, end_time, kernel_ready_time,
                                       core_type);
            }

            last_task_id = task_id;
            write_reg(RegId::COND, MAKE_FIN_VALUE(payload->task_id));
        }
    }
}
