#include "aicore/aicore.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"

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

    // Phase 3: Main execution loop - poll for tasks until quit signal
    while (true) {
        dcci(my_hank, ENTIRE_DATA_CACHE, CACHELINE_OUT);

        // Check for quit command from AICPU
        if (my_hank->control == 1) {
            break;  // Exit kernel
        }

        // Execute task if assigned (task != 0)
        if (my_hank->task_status == 1 && my_hank->task != 0) {
            execute_task(reinterpret_cast<__gm__ void*>(my_hank->task));
            my_hank->task_status = 0;
        }
    }
}
