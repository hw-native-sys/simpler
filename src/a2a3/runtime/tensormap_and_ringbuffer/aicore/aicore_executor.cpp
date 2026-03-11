#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"
#include "common/perf_profiling.h"
#include "aicore/performance_collector_aicore.h"
#include "common/platform_config.h"  // Register-based communication

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
 * Reads function_bin_addr and args from the dispatch payload.
 *
 * @param payload Pointer to PTO2DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ PTO2DispatchPayload* payload) {
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
 * Task dispatch reads PTO2DispatchPayload address from Handshake.task.
 * Task ID is derived from the register value (task_id + 1 encoding).
 *
 * @param runtime Pointer to Runtime in global memory
 * @param block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // In multi-round execution the DeviceRunner singleton keeps AICore threads alive
    // across rounds. DATA_MAIN_BASE still holds the EXIT_SIGNAL from the previous
    // round, so clear it before the handshake wait. Clearing after the wait would
    // race with AICPU, which may finish all tasks and write a new EXIT_SIGNAL while
    // this thread is descheduled between the wait and the clear.
    write_reg(RegId::DATA_MAIN_BASE, 0);

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

    bool profiling_enabled = runtime->enable_profiling;
    uint64_t kernel_ready_time = 0;
    if (profiling_enabled) {
        kernel_ready_time = get_sys_cnt_aicore();
    }

    // Phase 3: Main execution loop - poll register for tasks until exit signal
    // Register encoding: 0=idle, task_id+1=task, AICORE_EXIT_SIGNAL=exit
    uint32_t reg_val = 0;
    uint32_t last_reg_val = 0;

    while (true) {
        reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            break;
        }

        // Execute task if new (reg_val encoding: 0=idle, task_id+1=task)
        if (reg_val == 0 || reg_val == last_reg_val) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            uint32_t task_id = reg_val - 1;  // Decode: register holds task_id + 1

            // Invalidate entire data cache to read fresh payload and hank->task
            dcci(my_hank, ENTIRE_DATA_CACHE);

            // Read per-task dispatch payload address (updated by AICPU each dispatch)
            __gm__ PTO2DispatchPayload* payload =
                reinterpret_cast<__gm__ PTO2DispatchPayload*>(my_hank->task);

            write_reg(RegId::COND, MAKE_ACK_VALUE(task_id));

            // Performance profiling: record start time
            uint64_t start_time = 0;
            if (profiling_enabled) {
                start_time = get_sys_cnt_aicore();
            }

            // Execute the task
            execute_task(payload);

            // Performance profiling: record task execution
            // (func_id and core_type are filled by AICPU at completion time)
            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;
                perf_aicore_record_task(perf_buf, task_id,
                                       start_time, end_time, kernel_ready_time);
            }

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}
