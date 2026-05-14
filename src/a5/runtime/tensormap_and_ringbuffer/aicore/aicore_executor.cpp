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
#include "common/platform_config.h"  // Register-based communication
#include "common/pmu_profiling.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

/**
 * Execute task from PTO2DispatchPayload.
 *
 * Reads function_bin_addr and args from the dispatch payload.
 *
 * @param payload Pointer to PTO2DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

/**
 * AICore main execution loop
 *
 * Implements the AICPU-AICore register-based dispatch protocol:
 * 1. Wait for AICPU ready signal via handshake buffer
 * 2. Report physical core ID and core type, signal AICore ready
 * 3. Cache per-core PTO2DispatchPayload pointer from hank->task
 * 4. Poll DATA_MAIN_BASE register for task dispatch until exit signal
 *
 * AICPU writes &s_payload_per_core[i] to hank->task before setting
 * aicpu_ready=1. AICore caches this pointer and reads function_bin_addr +
 * args pointer from it on each dispatch. reg_val is a monotonically
 * increasing task ID used only for dispatch signaling and ACK/FIN protocol.
 *
 * @param runtime Pointer to Runtime in global memory
 * @param s_block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int s_block_idx, CoreType core_type) {
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

    // Cache profiling state once after Phase 3. The L2 / PMU rings are
    // stable for the entire run; pmu_reg_base is read per-record (see
    // host_build_graph executor for why).
    uint32_t profiling_flag = get_aicore_profiling_flag();
    bool l2_perf_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_L2_SWIMLANE);
    bool dump_tensor_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_DUMP_TENSOR);
    bool pmu_enabled = GET_PROFILING_FLAG(profiling_flag, PROFILING_FLAG_PMU);
    __gm__ L2PerfAicoreRing *l2_perf_ring = l2_perf_enabled ? get_aicore_l2_perf_ring() : nullptr;
    __gm__ PmuAicoreRing *pmu_ring = pmu_enabled ? get_aicore_pmu_ring() : nullptr;

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
            execute_task(exec_payload);

            if (pmu_enabled) {
                pmu_aicore_end();
                pmu_aicore_record_task(pmu_ring, get_aicore_pmu_reg_base(), task_id);
            }

            if (dump_tensor_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            // Performance profiling: record task execution
            if (l2_perf_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                l2_perf_aicore_record_task(l2_perf_ring, task_id, start_time, end_time);
            }

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
