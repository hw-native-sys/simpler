#include "aicore/aicore.h"
#include "runtime.h"
#include "common/perf_profiling.h"
#include "common/platform_config.h"  // Platform configuration (C/C++ compatible)

typedef void (*KernelFunc)(__gm__ int64_t*);

/**
 * @brief Record task execution performance data
 *
 * This function records the performance metrics of a task execution to the profiling buffer.
 * It writes the task timing data, metadata, and marks the buffer as full when needed.
 *
 * @param my_hank Pointer to the handshake structure containing perf buffer info
 * @param task_ptr Pointer to the executed task
 * @param start_time Task start timestamp
 * @param end_time Task end timestamp
 * @param block_idx AICore block index
 * @param core_type Core type (AIC or AIV)
 * @param kernel_ready_time Kernel ready timestamp (when AICore entered main loop)
 */
__aicore__ __attribute__((always_inline)) static void record_task_performance(
    __gm__ Handshake* my_hank,
    __gm__ Task* task_ptr,
    uint64_t start_time,
    uint64_t end_time,
    int block_idx,
    CoreType core_type,
    uint64_t kernel_ready_time) {

    // dcci() for handshake visibility during profiling
    dcci((__gm__ uint32_t*)&my_hank->perf_buffer_status, SINGLE_CACHE_LINE, CACHELINE_OUT);

    if (my_hank->perf_buffer_status != 0) {
        return;
    }

    __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;
    uint32_t idx = perf_buf->count;

    if (idx < PLATFORM_PROF_BUFFER_SIZE) {
        __gm__ PerfRecord* record = (__gm__ PerfRecord*)&perf_buf->records[idx];

        record->start_time = start_time;
        record->end_time = end_time;
        record->kernel_ready_time = kernel_ready_time;
        record->task_id = task_ptr->task_id;
        record->func_id = task_ptr->func_id;
        record->core_id = block_idx;
        record->core_type = core_type;

        perf_buf->count = idx + 1;
        dcci(record, ENTIRE_DATA_CACHE, CACHELINE_OUT);
        if (perf_buf->count >= PLATFORM_PROF_BUFFER_SIZE) {
            my_hank->perf_buffer_status = 1;
        }
    } else {
        my_hank->perf_buffer_status = 1;
    }
    // dcci() for handshake visibility during profiling
    dcci((__gm__ uint32_t*)&my_hank->perf_buffer_status, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    if (task->function_bin_addr == 0) {
        return;
    }
    KernelFunc kernel = (KernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));

    // Ensure all memory writes are visible to other cores
    pipe_barrier(PIPE_ALL);
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Report physical core ID and core type for AICPU
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    my_hank->aicore_done = block_idx + 1;

    dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

    // Report initial idle status for task dispatch
    write_reg(RegId::COND, static_cast<uint64_t>(AICoreStatus::IDLE));

    bool profiling_enabled = runtime->enable_profiling;
    uint64_t kernel_ready_time = get_sys_cnt();

    // Main loop: poll DATA_MAIN_BASE for task_id
    volatile uint32_t task_id = 0;
    volatile uint32_t last_task_id = 0;

    while (true) {
        task_id = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (task_id == AICORE_EXIT_SIGNAL) {
            break;
        }

        // Execute task if new (task_id encoding: 0=idle, task_id+1=task)
        if (task_id != 0 && task_id != last_task_id) {
            write_reg(RegId::COND, static_cast<uint64_t>(AICoreStatus::BUSY));
            __gm__ Task* task_ptr = &(runtime->tasks[task_id - 1]);
            uint64_t start_time = get_sys_cnt();
            
            execute_task(task_ptr);

            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt();
                record_task_performance(my_hank, task_ptr, start_time, end_time,
                                      block_idx, core_type, kernel_ready_time);
            }

            last_task_id = task_id;
            write_reg(RegId::COND, static_cast<uint64_t>(AICoreStatus::IDLE));
        }
    }
}
