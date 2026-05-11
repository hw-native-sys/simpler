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
#ifndef A5_TM_RB_AICORE_EXECUTOR_H
#define A5_TM_RB_AICORE_EXECUTOR_H

#include "aicore/aicore.h"
#include "aicore/l2_perf_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"  // Register-based communication
#include "common/pmu_profiling.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

// Defined inline in this header so both the legacy AICore kernel TU
// (platform/onboard/aicore/kernel.cpp, compiled with --cce-aicore-only)
// and the chevron launch TU
// (platform/onboard/aicore/chevron_launch.cpp, compiled with bisheng
// -xcce as a host+device single TU) can pull in the same body without a
// separate .cpp file that needs to be co-linked. Both TUs include this
// header and get their own instantiation; only one launch path is active
// per host SO, so the device-side duplication is benign.

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

__aicore__ __attribute__((always_inline)) inline static void
aicore_executor_run_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

inline __aicore__ void aicore_execute(__gm__ Runtime *runtime, int s_block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[s_block_idx]);

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
    my_hank->aicore_done = s_block_idx + 1;  // Signal ready (use s_block_idx + 1 to avoid 0)

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Cache per-core dispatch payload pointer (set by AICPU before aicpu_ready)
    __gm__ PTO2DispatchPayload *payload = reinterpret_cast<__gm__ PTO2DispatchPayload *>(my_hank->task);

    bool l2_perf_enabled = GET_PROFILING_FLAG(my_hank->enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    bool dump_tensor_enabled = GET_PROFILING_FLAG(my_hank->enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    bool pmu_enabled = GET_PROFILING_FLAG(my_hank->enable_profiling_flag, PROFILING_FLAG_PMU);

    // Phase 4: Main execution loop - poll register for tasks until exit signal
    // Register encoding: AICPU_IDLE_TASK_ID=idle, task_id=task, AICORE_EXIT_SIGNAL=exit
    uint32_t reg_val = AICPU_IDLE_TASK_ID;
    uint32_t last_reg_val = AICPU_IDLE_TASK_ID;

    while (true) {
        reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            // Signal exit acknowledgment to AICPU
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }

        // Execute task if new (reg_val encoding: AICPU_IDLE_TASK_ID=idle, task_id=task)
        if (reg_val == AICPU_IDLE_TASK_ID || reg_val == last_reg_val) {
            SPIN_WAIT_HINT();
            continue;
        }

        {
            uint32_t task_id = reg_val;  // Decode: register holds task_id directly

            // Select dual-buffer slot: same bit as AICPU used when writing payload
            __gm__ PTO2DispatchPayload *exec_payload = payload + (task_id & 1u);

            // Invalidate payload buffer (AICPU updates its content each dispatch)
            dcci(exec_payload, ENTIRE_DATA_CACHE);

            write_reg(RegId::COND, MAKE_ACK_VALUE(task_id));

            // Performance profiling: record start time
            uint64_t start_time = get_sys_cnt_aicore();

            if (pmu_enabled) {
                pmu_aicore_begin();
            }

            // Execute the task
            aicore_executor_run_task(exec_payload);

            if (pmu_enabled) {
                pmu_aicore_end();
                // Read pmu_buffer_addr / pmu_reg_base per-task (mirrors how
                // perf_records_addr is read below): by the time AICPU dispatches
                // a real task_id, pmu_aicpu_init has already published these.
                __gm__ PmuBuffer *pmu_buf = reinterpret_cast<__gm__ PmuBuffer *>(my_hank->pmu_buffer_addr);
                pmu_aicore_record_task(pmu_buf, my_hank->pmu_reg_base, task_id);
            }

            if (dump_tensor_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            // Performance profiling: record task execution
            if (l2_perf_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                __gm__ L2PerfBuffer *l2_perf_buf = (__gm__ L2PerfBuffer *)my_hank->l2_perf_records_addr;
                l2_perf_aicore_record_task(l2_perf_buf, task_id, start_time, end_time);
            }

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

#endif  // A5_TM_RB_AICORE_EXECUTOR_H
