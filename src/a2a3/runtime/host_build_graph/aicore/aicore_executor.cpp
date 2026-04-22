/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "aicore/aicore.h"
#include "aicore/l2_perf_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"  // Platform configuration (C/C++ compatible)
#include "runtime.h"

typedef void (*KernelFunc)(__gm__ int64_t *);

__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ Task *task) {
    if (task->function_bin_addr == 0) {
        return;
    }
    KernelFunc kernel = (KernelFunc)task->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(task->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
    }

    // Phase 2: Report physical core ID, signal ready
    my_hank->physical_core_id = get_physical_core_id();
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_regs_ready = 1;
    dcci(&my_hank->aicore_regs_ready, SINGLE_CACHE_LINE, CACHELINE_OUT);
    while (my_hank->aicpu_regs_ready == 0) {
        dcci(&my_hank->aicpu_regs_ready, SINGLE_CACHE_LINE);
    }
    // Report initial idle status via register
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // Phase 3: Report core type, signal ready
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    bool l2_perf_enabled = runtime->enable_l2_swimlane;
    bool dump_tensor_enabled = GET_PROFILING_FLAG(my_hank->enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    bool pmu_enabled = GET_PROFILING_FLAG(my_hank->enable_profiling_flag, PROFILING_FLAG_PMU);

    volatile uint32_t task_id = AICPU_IDLE_TASK_ID;
    volatile uint32_t last_task_id = AICPU_IDLE_TASK_ID;

    while (true) {
        task_id = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (task_id == AICORE_EXIT_SIGNAL) {
            // Signal exit acknowledgment to AICPU
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }

        if (task_id == AICPU_IDLE_TASK_ID || task_id == last_task_id) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            uint32_t actual_task_id = task_id;
            write_reg(RegId::COND, MAKE_ACK_VALUE(actual_task_id));

            __gm__ Task *task_ptr = &(runtime->tasks[actual_task_id]);
            uint64_t start_time = get_sys_cnt_aicore();

            if (pmu_enabled) {
                pmu_aicore_begin();
            }

            execute_task(task_ptr);

            if (pmu_enabled) {
                pmu_aicore_end();
            }

            if (dump_tensor_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            if (l2_perf_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ L2PerfBuffer *l2_perf_buf = (__gm__ L2PerfBuffer *)my_hank->l2_perf_records_addr;
                l2_perf_aicore_record_task(l2_perf_buf, actual_task_id, start_time, end_time);
            }

            last_task_id = task_id;

            write_reg(RegId::COND, MAKE_FIN_VALUE(actual_task_id));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
