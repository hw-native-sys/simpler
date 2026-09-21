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
#include "aicore/chip_swimlane_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"  // Register-based communication
#include "common/pmu_profiling.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

/**
 * Execute task from DispatchPayload.
 *
 * Reads function_bin_addr and args from the dispatch payload.
 *
 * @param payload Pointer to DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

using simpler::tmr::TmrCoreReport;
using simpler::tmr::TmrKernelContextDescriptor;
using simpler::tmr::TmrLaunchControl;

__aicore__ static void execute_dispatch_loop(__gm__ Handshake *my_hank) {
    dcci(my_hank, SINGLE_CACHE_LINE);
    __gm__ DispatchPayload *payload = reinterpret_cast<__gm__ DispatchPayload *>(my_hank->task);

    // Cache profiling state once after window-open. The L2 / PMU rings and the
    // PMU MMIO base are all stable for the entire run (host-resolved at
    // AICore kernel entry from KernelArgs::regs[physical_core_id]), so
    // they are safe to cache here.
    uint32_t profiling_flag = get_aicore_profiling_flag();
    bool chip_swimlane_enabled = SIMPLER_GET_DFX_FLAG(profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    bool dump_args_enabled = SIMPLER_GET_DFX_FLAG(profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
    bool pmu_enabled = SIMPLER_GET_DFX_FLAG(profiling_flag, SIMPLER_DFX_FLAG_PMU);
    // This executor chooses first-dispatch lazy resolution. The rotation
    // channel is safe to resolve after the wrapper observes window-open.
    __gm__ ChipSwimlaneActiveHead *chip_swimlane_head = nullptr;
    ChipSwimlaneAicoreLocalState chip_swimlane_local = {nullptr, UINT32_MAX, 0};
    __gm__ PmuAicoreRing *pmu_ring = pmu_enabled ? get_aicore_pmu_ring() : nullptr;
    uint64_t pmu_reg_base = pmu_enabled ? get_aicore_pmu_reg_base() : 0;

    // Register encoding: AICPU_IDLE_TASK_ID=idle, task_id=task, AICORE_EXIT_SIGNAL=exit
    uint32_t reg_val = AICPU_IDLE_TASK_ID;
    uint32_t last_reg_val = AICPU_IDLE_TASK_ID;
    bool exiting = false;

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
            // receive_time is captured the instant DATA_MAIN_BASE returned a
            // new task_id, BEFORE the per-task dcci + ack pair. Paired with
            // start_time (captured after dcci + ack) it lets DFX split head_OH
            // into the AICPU→AICore NoC propagation (dispatch_ts → receive_time,
            // hardware-bound) and the AICore-local dcci+ack cost
            // (receive_time → start_time, software-tunable). Stored in the
            // record as a 32-bit delta `start_time - receive_time`.
            //
            // Early-dispatch (src_payload != 0): receive_time stays at pickup —
            // before the doorbell wait — so it may precede the producer's end.
            uint64_t receive_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            uint32_t task_id = reg_val;  // Decode: register holds task_id directly

            // First-task lazy resolve of the rotation channel.
            if (chip_swimlane_enabled && chip_swimlane_head == nullptr) {
                chip_swimlane_head = get_chip_swimlane_aicore_head();
            }

            // Select dual-buffer slot: same bit as AICPU used when writing payload
            __gm__ DispatchPayload *exec_payload = payload + (task_id & 1u);

            // Invalidate payload buffer (AICPU updates its content each dispatch)
            dcci(exec_payload, ENTIRE_DATA_CACHE);

            // Early-dispatch gate. A gated task was staged on this core before its
            // dependencies resolved; wait until AICPU rings the doorbell
            // (DATA_MAIN_BASE high 32 == task_id) before executing. The ACK is
            // deferred until AFTER the gate so the scheduler keeps the core
            // off-limits (pending_occupied stays set) while the task is gated.
            // src_payload == 0 (ready path) skips this; a non-zero src_payload is
            // both the gate flag and the source TaskPayload.
            if (exec_payload->src_payload != 0) {
                __gm__ char *src = reinterpret_cast<__gm__ char *>(exec_payload->src_payload);
                int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(src + TASKPAYLOAD_TENSOR_COUNT_OFFSET);
                int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(src + TASKPAYLOAD_SCALAR_COUNT_OFFSET);
                __gm__ uint64_t *src_scalars = reinterpret_cast<__gm__ uint64_t *>(src + TASKPAYLOAD_SCALARS_OFFSET);
                int n = 0;
                for (int32_t i = 0; i < tensor_count; i++) {
                    exec_payload->args[n++] =
                        reinterpret_cast<uint64_t>(src + TASKPAYLOAD_TENSORS_OFFSET + i * TASKPAYLOAD_TENSOR_STRIDE);
                }
                for (int32_t i = 0; i < scalar_count; i++) {
                    exec_payload->args[n++] = src_scalars[i];
                }
                OUT_OF_ORDER_STORE_BARRIER();
                while (true) {
                    if (read_dmb_high32() == task_id) {
                        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
                            exiting = true;
                        }
                        break;
                    }
                    if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
                        exiting = true;
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (exiting) {
                    write_reg(RegId::COND, AICORE_EXITED_VALUE);
                    break;
                }
            }

            // Bind this task to the currently-published L2 buffer generation
            // before ACK makes progress visible to AICPU. The PMU staging ring
            // is stable for the full run and does not participate in rotation.
            __gm__ ChipSwimlaneAicoreTaskRecord *chip_swimlane_record = nullptr;
            if (chip_swimlane_enabled) {
                chip_swimlane_record =
                    chip_swimlane_aicore_reserve_task_record(chip_swimlane_head, &chip_swimlane_local);
            }

            write_reg(RegId::COND, MAKE_ACK_VALUE(task_id));

            uint64_t start_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            if (pmu_enabled) {
                pmu_aicore_begin();
            }

            // Execute the task
            execute_task(exec_payload);

            // Keep start_time -> end_time scoped to AICore execution.
            uint64_t end_time = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

            if (pmu_enabled) {
                pmu_aicore_end();
                // PMU's stable per-core staging slot must be visible before FIN:
                // the AICPU completion path consumes it as soon as FIN arrives.
                pmu_aicore_record_task(pmu_ring, pmu_reg_base, task_id);
            }

            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));

            if (dump_args_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            // Performance profiling: record task execution. task_token_raw is
            // the task identity (already in AICore cache from the dispatch
            // payload); reg_task_id is the per-core dispatch token AICore just
            // read. Host uses reg_task_id as join key vs the AICPU stream.
            if (chip_swimlane_enabled) {
                uint64_t task_token_raw = exec_payload->local_context.async_ctx.task_token.raw;
                chip_swimlane_aicore_commit_task_record(
                    chip_swimlane_record, task_token_raw, task_id, receive_time, start_time, end_time
                );
            }
        }
    }
}

__aicore__ static void
execute_worker(__gm__ Runtime *runtime, __gm__ Handshake *my_hank, int block_idx, CoreType core_type) {
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Each launch resets this SPR; the AICPU publishes task before opening it.
    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    write_reg(RegId::COND, AICORE_IDLE_VALUE);
    execute_dispatch_loop(my_hank);
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
    (void)runtime;
}

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    execute_worker(runtime, &runtime->dev.workers[block_idx], block_idx, core_type);
}

__aicore__ __attribute__((weak)) void aicore_execute_kernel(
    __gm__ Runtime *runtime, __gm__ const TmrKernelContextDescriptor *context, int block_idx, CoreType core_type
) {
    if (block_idx < 0 || block_idx >= context->worker_count || context->worker_count > RUNTIME_MAX_WORKER ||
        context->reports_bytes != static_cast<uint64_t>(context->worker_count) * sizeof(TmrCoreReport) ||
        context->reports_address == 0 || context->reports_address > UINT64_MAX - context->reports_bytes ||
        context->reports_address % alignof(TmrCoreReport) != 0)
        return;
    auto *reports = reinterpret_cast<__gm__ Handshake *>(context->reports_address);
    execute_worker(runtime, &reports[block_idx], block_idx, core_type);
}
