#include "aicore/aicore.h"
#include "runtime.h"
#include "common/perf_profiling.h"
#include "aicore/performance_collector_aicore.h"
#include "common/platform_config.h"  // Platform configuration (C/C++ compatible)

typedef void (*KernelFunc)(__gm__ int64_t*);

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

    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    bool profiling_enabled = runtime->enable_profiling;
    uint64_t kernel_ready_time = get_sys_cnt_aicore();

    volatile uint32_t task_id = 0;
    volatile uint32_t last_task_id = 0;

    while (true) {
        task_id = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (task_id == AICORE_EXIT_SIGNAL) {
            break;
        }

        if (task_id == 0 || task_id == last_task_id) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            uint32_t actual_task_id = task_id - 1;
            write_reg(RegId::COND, MAKE_ACK_VALUE(actual_task_id));

            __gm__ Task* task_ptr = &(runtime->tasks[actual_task_id]);
            uint64_t start_time = get_sys_cnt_aicore();

            execute_task(task_ptr);

            if (profiling_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ PerfBuffer* perf_buf = (__gm__ PerfBuffer*)my_hank->perf_records_addr;
                perf_aicore_record_task(perf_buf, task_ptr->task_id, task_ptr->func_id,
                                      start_time, end_time, kernel_ready_time,
                                      core_type);
            }

            last_task_id = task_id;

            write_reg(RegId::COND, MAKE_FIN_VALUE(actual_task_id));
        }
    }
}
