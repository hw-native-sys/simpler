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
#include "aicore/l2_swimlane_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"  // Register-based communication
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
 * 1. Report physical core ID and core type, signal aicore_done (no AICPU wait)
 * 2. Wait for the AICPU to open our register window (DATA_MAIN_BASE != 0)
 * 3. Cache per-core PTO2DispatchPayload pointer from hank->task
 * 4. Poll DATA_MAIN_BASE register for task dispatch until exit signal
 *
 * AICore reports on launch; the AICPU writes &s_payload_per_core[i] to
 * hank->task and then opens the register window (DATA_MAIN_BASE = IDLE), which
 * is itself the acknowledgement. AICore caches this pointer and reads
 * function_bin_addr + args pointer from it on each dispatch. reg_val is a
 * monotonically increasing task ID used only for dispatch signaling and
 * ACK/FIN protocol.
 *
 * Profiling state (enable flag, L2 swimlane rotation channel) is published into the platform
 * via set_aicore_profiling_flag / set_aicore_l2_swimlane_ring at kernel entry —
 * this routine reads it through the matching getters, so neither Handshake
 * nor this signature carry profiling fields.
 *
 * @param runtime Pointer to Runtime in global memory
 * @param block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[block_idx]);

    // Phase 1: report physical core ID + core type and signal done in one write,
    // with no wait for the AICPU — both fields are self-known. The AICPU opens
    // this core's register window only after it observes aicore_done, so a single
    // report suffices. The host clears aicore_done before this kernel launches,
    // so the value the AICPU reads is this run's report, never a stale prior one.
    my_hank->physical_core_id = get_physical_core_id();
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = block_idx + 1;  // Signal ready (use block_idx + 1 to avoid 0)
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Phase 2: Wait for the AICPU to open our register window. A kernel launch
    // resets DATA_MAIN_BASE to 0 (verified on a2a3 silicon); the AICPU writes
    // DATA_MAIN_BASE = AICPU_IDLE_TASK_ID (non-zero) as it opens FAST_PATH, so a
    // non-zero read means the window is open and reads/writes are valid. The
    // AICPU runs assign_cores_to_threads (µs) between opening the window and the
    // first dispatch, so this IDLE is observed long before any task_id lands —
    // the poll cannot miss it and mistake a later task for the reset value.
    // Window-open is the sync point for everything the AICPU publishes (task
    // pointer, swimlane head): the AICPU writes those before opening the window.
    while (read_reg(RegId::DATA_MAIN_BASE) == 0) {
        SPIN_WAIT_HINT();
    }
    // Report initial idle status via register (FAST_PATH is now open).
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // The AICPU writes task after observing our report (so our CACHELINE_OUT flush
    // above cannot clobber it) and before opening the window; dcci to read its
    // fresh value here.
    dcci(my_hank, SINGLE_CACHE_LINE);
    __gm__ PTO2DispatchPayload *payload = reinterpret_cast<__gm__ PTO2DispatchPayload *>(my_hank->task);

    uint32_t enable_profiling_flag = get_aicore_profiling_flag();
    bool l2_swimlane_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_L2_SWIMLANE);
    bool dump_tensor_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_TENSOR);
    bool pmu_enabled = SIMPLER_GET_DFX_FLAG(enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);

    // Per-core L2SwimlaneActiveHead channel. AICPU completes
    // `l2_swimlane_aicpu_init` (in pre_handshake_init) before any thread opens a
    // register window in `handshake_partition`, and Phase 2 above has already
    // observed our window open, so the rotation-table slot is populated and the
    // first deref is safe here — off the dispatch→start critical path.
    __gm__ L2SwimlaneActiveHead *l2_swimlane_head = l2_swimlane_enabled ? get_l2_swimlane_aicore_head() : nullptr;
    // cached_buf_seq must start != AICPU's initial head.current_buf_seq (0)
    // so the first record_task observes a mismatch and loads the buffer ptr.
    L2SwimlaneAicoreLocalState l2_swimlane_local = {nullptr, UINT32_MAX, 0};

    // Phase 4: Main execution loop - poll register for tasks until exit signal
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
            // receive_time marks the moment AICPU's full "task is ready to
            // execute" signal landed on this core. Paired with start_time
            // (captured after the per-task dcci + ack pair) it lets DFX split
            // head_OH into the AICPU→AICore-ready propagation (dispatch_ts →
            // receive_time, hardware + scheduling-bound) and the AICore-local
            // critical-path prep (receive_time → start_time, software-tunable).
            // Stored in the record as a 32-bit delta `start_time - receive_time`.
            //
            // For the common path (src_payload == 0) the new task_id on
            // DATA_MAIN_BASE is itself the ready signal, so receive_time is
            // stamped immediately and local_setup covers dcci + ack.
            //
            // For the speculative early-dispatch path (src_payload != 0) the
            // dcci ran BEFORE the dependency-wait spin, so its cost is hidden
            // behind the doorbell-wait — not on the critical path between
            // "task genuinely ready" and "kernel begins". receive_time is
            // re-stamped after the doorbell arrives, so propagation absorbs
            // both the original NoC delivery AND any speculation overshoot,
            // while local_setup stays the pure ack-on-critical-path cost. This
            // makes local_setup the clean "AICore prep we can't hide" figure
            // for both paths.
            uint64_t receive_time = get_sys_cnt_aicore();

            uint32_t task_id = reg_val;  // Decode: register holds task_id directly

            // Select dual-buffer slot: same bit as AICPU used when writing payload
            __gm__ PTO2DispatchPayload *exec_payload = payload + (task_id & 1u);

            // Invalidate payload buffer (AICPU updates its content each dispatch)
            dcci(exec_payload, ENTIRE_DATA_CACHE);

            // Speculative early-dispatch gate. A not-ready task was staged on
            // this core before its dependencies resolved; wait until AICPU rings
            // the doorbell (DATA_MAIN_BASE high 32 == task_id) before executing.
            // The ACK is deferred until AFTER the gate so the scheduler keeps the
            // core off-limits (pending_occupied stays set, no ACK->pending_freed)
            // while the task is gated — preventing a real task from being
            // dual-issued behind it. The kernel's own input dcci runs inside
            // execute_task() below — strictly AFTER this gate — so predecessor
            // outputs are visible. src_payload == 0 (the common path) skips this;
            // a non-zero src_payload is both the gate flag and the source
            // PTO2TaskPayload.
            if (exec_payload->src_payload != 0) {
                // AICPU staged only src_payload, not the arg vector — fill
                // args[0..num_args) ourselves now, while we are idle waiting for
                // the doorbell. The whole-cache dcci(ENTIRE_DATA_CACHE) above
                // already invalidated src's lines, so tensor_count/scalar_count/
                // scalars read coherently with the orchestrator's submit writes.
                // args[SPMD_LOCAL_CONTEXT_INDEX]/[SPMD_GLOBAL_CONTEXT_INDEX] are
                // still written by the AICPU (num_args <= 48 never reaches them).
                __gm__ char *src = reinterpret_cast<__gm__ char *>(exec_payload->src_payload);
                int32_t tensor_count = *reinterpret_cast<__gm__ int32_t *>(src + PTO2_TASKPAYLOAD_TENSOR_COUNT_OFFSET);
                int32_t scalar_count = *reinterpret_cast<__gm__ int32_t *>(src + PTO2_TASKPAYLOAD_SCALAR_COUNT_OFFSET);
                __gm__ uint64_t *src_scalars =
                    reinterpret_cast<__gm__ uint64_t *>(src + PTO2_TASKPAYLOAD_SCALARS_OFFSET);
                int n = 0;
                for (int32_t i = 0; i < tensor_count; i++) {
                    exec_payload->args[n++] = reinterpret_cast<uint64_t>(
                        src + PTO2_TASKPAYLOAD_TENSORS_OFFSET + i * PTO2_TASKPAYLOAD_TENSOR_STRIDE
                    );
                }
                for (int32_t i = 0; i < scalar_count; i++) {
                    exec_payload->args[n++] = src_scalars[i];
                }
                OUT_OF_ORDER_STORE_BARRIER();
                while (true) {
                    // Honor teardown: shutdown overwrites the low half with EXIT.
                    // Check it on the doorbell-match iteration too, so an EXIT that
                    // races in right after the matching doorbell still wins over
                    // executing the gated task.
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
                // Re-stamp receive_time at the moment the doorbell landed: the
                // dcci above ran during the speculative-staging window
                // (overlapped with the dependency wait, off the critical path).
                // Propagation now absorbs the speculation overshoot; local_setup
                // = start - receive stays the pure ack-on-critical-path cost.
                receive_time = get_sys_cnt_aicore();
            }

            write_reg(RegId::COND, MAKE_ACK_VALUE(task_id));

            // Performance profiling: record start time
            uint64_t start_time = get_sys_cnt_aicore();

            // PMU: start counting window around kernel execution
            if (pmu_enabled) {
                pmu_aicore_begin();
            }

            // Execute the task
            execute_task(exec_payload);

            if (pmu_enabled) {
                pmu_aicore_end();
            }

            if (dump_tensor_enabled) {
                pipe_barrier(PIPE_ALL);
            }

            // Performance profiling: record task execution.
            // Two identity fields go into the record (different roles):
            //   - task_token_raw (PTO2 ring/local) is pulled from the dispatch
            //     payload's LocalContext.async_ctx — already in AICore cache
            //     from the just-completed task, no extra GM load. Host uses
            //     it as the canonical task identity for JSON output / ring
            //     decoding.
            //   - reg_task_id is `task_id` (= reg_val, the per-core dispatch
            //     token AICore just read from DATA_MAIN_BASE). Per-dispatch
            //     unique within this core; host uses it as the join key
            //     against the AICPU record stream. Required for correctness
            //     under SPMD (block_num > num_cores) and MIX cluster spread,
            //     where multiple dispatches of the same task share the same
            //     task_token_raw.
            last_reg_val = reg_val;
            write_reg(RegId::COND, MAKE_FIN_VALUE(task_id));

            // Sample end_time AFTER the FIN write so the op-event end marks the
            // moment the AICPU can first observe completion — any compute-end ->
            // FIN gap (epilogue / write-back) shows directly on the bar instead
            // of being inferred. The record write itself stays off the critical
            // path (it runs after FIN, so it no longer delays completion).
            if (l2_swimlane_enabled) {
                uint64_t end_time = get_sys_cnt_aicore();
                uint64_t task_token_raw = exec_payload->local_context.async_ctx.task_token.raw;
                l2_swimlane_aicore_record_task(
                    l2_swimlane_head, &l2_swimlane_local, task_token_raw, task_id, receive_time, start_time, end_time
                );
            }
        }
    }

    // Flush all dirty cache lines to HBM before kernel exit.
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
