#include "aicore/aicore.h"
#include "runtime.h"

typedef void (*KernelFunc)(__gm__ int64_t*);

__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    if (task->function_bin_addr == 0) {
        return;
    }
    KernelFunc kernel = (KernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready and report core type
    my_hank->core_type = core_type;        // Report core type to AICPU
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid task pointer)
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            execute_task(reinterpret_cast<__gm__ Task*>(my_hank->task));
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}
