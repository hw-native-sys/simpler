#include "aicore.h"
#include "runtime.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t*);

/**
 * Task execution wrapper - dispatches tasks using function pointers
 *
 * This function demonstrates the runtime function pointer dispatch pattern.
 * Following the production system flow:
 * - function_bin_addr points to compiled kernel code in device GM memory
 * - The address is cast to a function pointer: UnifiedKernelFunc kernel =
 * (UnifiedKernelFunc)function_bin_addr
 * - The kernel is invoked: kernel(task->args)
 *
 * This is the KEY difference from compile-time linking:
 * - OLD: extern "C" declarations, resolved at link time
 * - NEW: function_bin_addr from GM memory, cast at runtime
 *
 * With unified kernel signature, no switch statement is needed.
 * All kernels unpack their own arguments from the args array.
 *
 * @param task Pointer to task in global memory (null during initialization)
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task* task) {
    // Null task pointer indicates no work assigned (initialization state)
    if (task == nullptr) {
        return;
    }

    // Check for valid function_bin_addr
    if (task->function_bin_addr == 0) {
        // Invalid address - skip execution
        return;
    }

    // Cast function_bin_addr to unified function pointer and invoke
    // All kernels have signature: void kernel(__gm__ int64_t* args)
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t*>(task->args));
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime* runtime, int block_idx, int core_type) {
    __gm__ Handshake* my_hank = (__gm__ Handshake*)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    }

    // Phase 2: Signal AICore is ready (use core_id + 1 to avoid 0)
    my_hank->aicore_done = block_idx + 1;

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0 means valid Task* pointer)
        if (my_hank->task != 0) {
            __gm__ Task* task_ptr = reinterpret_cast<__gm__ Task*>(my_hank->task);
            execute_task(task_ptr);
            // Mark task as complete (task_status: 0=idle, 1=busy)
            my_hank->task_status = 0;
        }
    }
}