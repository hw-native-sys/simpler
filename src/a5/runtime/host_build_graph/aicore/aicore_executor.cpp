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
#include "aicore/aicore_profiling_state.h"
#include "aicore/l2_perf_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"  // Platform configuration (C/C++ compatible)
#include "common/pmu_profiling.h"
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

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int core_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[core_idx]);

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
    my_hank->aicore_done = core_idx + 1;

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Cache profiling state once after Phase 3. The L2 / PMU rings are
    // stable for the entire run (host-allocated, never reassigned), so they
    // are safe to cache here. The PMU reg base, by contrast, lives in a
    // table whose entries the AICPU fills during pmu_aicpu_init *after*
    // handshake_all_cores returns — i.e., possibly after this Phase-3
    // cache. We therefore fetch pmu_reg_base on demand inside the loop;
    // by the time the first PMU record fires, AICPU has dispatched a task
    // (via DATA_MAIN_BASE), which serializes after the table fill.
    uint32_t profiling_flag = get_aicore_profiling_flag();
    bool l2_perf_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    bool dump_tensor_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    bool pmu_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_PMU);
    __gm__ L2PerfAicoreRing *l2_perf_ring = l2_perf_enabled ? get_aicore_l2_perf_ring() : nullptr;
    __gm__ PmuAicoreRing *pmu_ring = pmu_enabled ? get_aicore_pmu_ring() : nullptr;

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
                pmu_aicore_record_task(pmu_ring, get_aicore_pmu_reg_base(), actual_task_id);
            }

            if (dump_tensor_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            if (l2_perf_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                l2_perf_aicore_record_task(l2_perf_ring, actual_task_id, start_time, end_time);
            }

            last_task_id = task_id;

            write_reg(RegId::COND, MAKE_FIN_VALUE(actual_task_id));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
