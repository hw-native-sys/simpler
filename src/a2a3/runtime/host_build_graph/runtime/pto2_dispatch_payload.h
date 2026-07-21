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

/**
 * @file pto2_dispatch_payload.h
 * @brief Per-core dispatch payload for AICore kernel execution
 *
 * PTO2DispatchPayload holds the kernel function address, a per-core args[]
 * array, and embedded SPMD context (LocalContext + GlobalContext).  AICPU
 * maintains a static array of these (one per core).
 *
 * GlobalContext (sub_block_id) is initialized once at runtime startup via
 * init_global_context() and never modified afterwards.
 *
 * LocalContext (block_idx, block_num) and args[] are rebuilt by build_payload()
 * before each dispatch.  Both context struct pointers are written into the
 * args[] suffix on every dispatch (since args[] is rebuilt entirely each time).
 *
 * AICore caches a pointer to its per-core slot at startup and reads from
 * it on each dispatch.  The struct is cache-line aligned to avoid false
 * sharing across concurrently dispatched cores.
 *
 * The DATA_MAIN_BASE register protocol is unchanged from the base runtime:
 * a monotonically increasing reg_task_id signals new work to AICore.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "arg_direction.h"
#include "intrinsic.h"

/** Max dispatch arguments: 16 scalars + up to 32 tensor pointers + ext params */
#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + PTO2_EXT_PARAMS_COUNT)
#endif

#ifndef PTO2_ALIGN_UP
#define PTO2_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

// Verify hardcoded indices in intrinsic.h match the computed values.
static_assert(
    (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) == SPMD_LOCAL_CONTEXT_INDEX, "LOCAL_CONTEXT_INDEX out of sync with intrinsic.h"
);
static_assert(
    (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + 1) == SPMD_GLOBAL_CONTEXT_INDEX,
    "GLOBAL_CONTEXT_INDEX out of sync with intrinsic.h"
);

// Byte offsets into PTO2TaskPayload used by AICore to materialize args[] on the
// not_ready (early-dispatch) path. The AICore .o does not include
// pto_runtime2_types.h, so it reads tensor_count / scalar_count / tensors[] /
// scalars[] through these constants. The AICPU side (scheduler_dispatch.cpp)
// static_asserts them against offsetof where the full struct is visible, so any
// PTO2TaskPayload layout drift fails the build rather than corrupting args[].
constexpr uint32_t PTO2_TASKPAYLOAD_TENSOR_COUNT_OFFSET = 0;
constexpr uint32_t PTO2_TASKPAYLOAD_SCALAR_COUNT_OFFSET = 4;
constexpr uint32_t PTO2_TASKPAYLOAD_TENSORS_OFFSET = 576;
constexpr uint32_t PTO2_TASKPAYLOAD_SCALARS_OFFSET = 4672;
constexpr uint32_t PTO2_TASKPAYLOAD_TENSOR_STRIDE = 128;  // sizeof(Tensor)

/**
 * Per-core dispatch payload: function address + args[] + SPMD context.
 *
 * AICPU maintains a static array s_payload_per_core[RUNTIME_MAX_WORKER].
 * AICore caches a pointer to its per-core slot at startup (via Handshake.task)
 * and reads from it on each dispatch.
 *
 * The struct is cache-line aligned to prevent false sharing across
 * concurrently dispatched cores.
 */
struct alignas(64) PTO2DispatchPayload {
    // === Cache line 0 (64B): control block, the only line written per dispatch ===
    // function_bin_addr, local_context.{block_idx,block_num,async_ctx.task_token}
    // and src_payload are the per-dispatch writes; async_ctx's slab pointers +
    // capacity are cold (prefilled once at init) but ride this hot line for free.
    // Sized to exactly 64B so both dispatch paths write one control line: the
    // ready path (src_payload = 0) then also fills args[0..num_args); the gated
    // path (src_payload = &PTO2TaskPayload) leaves args[] to the idle AICore.
    uint64_t function_bin_addr; /**< Kernel entry address in GM (set by Scheduler). */

    /** Per-dispatch context: block_idx/block_num (hot) + async_ctx (task_token hot,
     *  slab pointers + capacity prefilled once at init). args[SPMD_LOCAL_CONTEXT_INDEX]
     *  points here. */
    LocalContext local_context;

    /** Early-dispatch gate AND source pointer, folded into one field. 0 = ready:
     *  AICore executes on pickup (args[] already filled by the AICPU). Non-zero =
     *  gated: the value is the source PTO2TaskPayload address; the AICore fills
     *  args[0..num_args) from it, then waits for the doorbell (DATA_MAIN_BASE
     *  high 32 == this dispatch's reg_task_id) before executing. A payload address
     *  is never 0, so the gate flag is lossless. */
    volatile uint64_t src_payload;

    // === Cache lines 1..7: kernel argument vector ===
    /** [0..num_args) = GM tensor pointers + scalar values; [SPMD_LOCAL_CONTEXT_INDEX]
     *  = &local_context, [SPMD_GLOBAL_CONTEXT_INDEX] = &global_context (both prefilled
     *  once at init). On the gated path the AICPU leaves [0..num_args) unwritten and
     *  the idle AICore fills them from src_payload during its gate wait. */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];

    /** Per-core global context: sub_block_id (AIV lane identity). Cold: written once
     *  at init, never per dispatch, so it lives in the tail (not the CL0 control
     *  block). args[SPMD_GLOBAL_CONTEXT_INDEX] points here. */
    GlobalContext global_context;
    // No explicit tail padding: alignas(64) rounds sizeof up to 512 (8 cache lines).

    static_assert(sizeof(args[0]) == 8);
    static_assert(
        PTO2_ALIGN_UP((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]), 64) ==
        (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0])
    );
};

static_assert(sizeof(PTO2DispatchPayload) == 512, "PTO2DispatchPayload hardware ABI size drift");
static_assert(offsetof(PTO2DispatchPayload, args) == 64, "args[] must start at cache line 1 (control block = CL0)");
static_assert(offsetof(PTO2DispatchPayload, src_payload) < 64, "src_payload (gate) must live on the CL0 control block");
