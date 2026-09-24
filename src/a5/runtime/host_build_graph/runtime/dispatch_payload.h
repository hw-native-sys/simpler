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
 * @file dispatch_payload.h
 * @brief Per-core dispatch payload for AICore kernel execution
 *
 * DispatchPayload holds the kernel function address, a per-core args[]
 * array, and embedded SPMD context (LocalContext + GlobalContext).  AICPU
 * maintains a static array of these (one per core).
 *
 * GlobalContext (sub_block_id) is populated by the scheduler path. The legacy
 * scheduler seeds each per-core slot at startup; resident graph materialization
 * writes the selected task subslot before every dispatch publication.
 *
 * build_payload() refreshes the function address, LocalContext's hot fields,
 * and either the ready-path argument prefix or the gated-path source pointer.
 * Context pointers and async-completion slab metadata are initialized once per
 * core and payload buffer.
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
#ifndef DISPATCH_MAX_ARGS
#define DISPATCH_MAX_ARGS (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + EXT_PARAMS_COUNT)
#endif

#ifndef CHIP_ALIGN_UP
#define CHIP_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

// Verify hardcoded indices in intrinsic.h match the computed values.
static_assert(
    (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) == SPMD_LOCAL_CONTEXT_INDEX, "LOCAL_CONTEXT_INDEX out of sync with intrinsic.h"
);
static_assert(
    (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + 1) == SPMD_GLOBAL_CONTEXT_INDEX,
    "GLOBAL_CONTEXT_INDEX out of sync with intrinsic.h"
);

// Byte offsets into TaskPayload used by AICore to materialize args[] on the
// not_ready (early-dispatch) path. The AICore .o does not include
// runtime_types.h, so it reads tensor_count / scalar_count / tensors[] /
// scalars[] through these constants. The AICPU side (scheduler_dispatch.cpp)
// static_asserts them against offsetof where the full struct is visible, so any
// TaskPayload layout drift fails the build rather than corrupting args[].
constexpr uint32_t TASKPAYLOAD_TENSOR_COUNT_OFFSET = 0;
constexpr uint32_t TASKPAYLOAD_SCALAR_COUNT_OFFSET = 4;
// Offsets of the fields that NAME the argument regions, not of the regions themselves:
// a region's address is (src + <field offset>) + <the int32 delta stored there>.
constexpr uint32_t TASKPAYLOAD_TENSORS_DELTA_OFFSET = 12;
constexpr uint32_t TASKPAYLOAD_SCALARS_DELTA_OFFSET = 16;
constexpr uint32_t TASKPAYLOAD_FANIN_DELTA_OFFSET = 20;
// Cache line 9 (byte 576) holds the AICPU-only DispatchPredicate. Scalars follow it,
// then tensors — the tensor array is last because it is the only region whose used
// extent varies per task, so a task's read set is a contiguous prefix.
constexpr uint32_t TASKPAYLOAD_TENSOR_STRIDE = 128;  // sizeof(simpler::hbg::Tensor)

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
struct alignas(64) DispatchPayload {
    // === Cache line 0 (64B): control block ===
    // The legacy path writes function_bin_addr, local_context's block fields and
    // task token, and src_payload per dispatch. The SSBUF ready path initializes
    // its fixed fields in the per-run image and updates function_bin_addr (and
    // src_payload for inline tasks) during dispatch.
    // async_ctx's slab pointers + capacity are cold but ride this hot line.
    // Sized to exactly 64B so both dispatch paths write one control line: the
    // ready path (src_payload = 0) then also fills args[0..num_args); the gated
    // path (src_payload = &TaskPayload) leaves args[] to the idle AICore.
    uint64_t function_bin_addr; /**< Kernel entry address in GM (set by Scheduler). */

    /** Context for the dispatched kernel. The legacy path updates block fields
     *  and task_token per dispatch; the SSBUF path keeps fixed values per run.
     *  Slab pointers and capacity are prefilled once at init.
     *  args[SPMD_LOCAL_CONTEXT_INDEX] points here. */
    LocalContext local_context;

    /** Early-dispatch gate AND source pointer, folded into one field. 0 = ready:
     *  AICore executes on pickup (args[] already filled by the AICPU). Non-zero =
     *  gated: the value is the source TaskPayload address; the AICore fills
     *  args[0..num_args) from it, then waits for the doorbell (DATA_MAIN_BASE
     *  high 32 == this dispatch's reg_task_id) before executing. A payload address
     *  is never 0, so the gate flag is lossless. */
    volatile uint64_t src_payload;

    // === Cache lines 1..7: kernel argument vector ===
    /** [0..num_args) = GM tensor pointers + scalar values; [SPMD_LOCAL_CONTEXT_INDEX]
     *  = &local_context, [SPMD_GLOBAL_CONTEXT_INDEX] = &global_context (both prefilled
     *  once at init). On the gated path the AICPU leaves [0..num_args) unwritten and
     *  the idle AICore fills them from src_payload during its gate wait. */
    uint64_t args[DISPATCH_MAX_ARGS];

    /** Per-dispatch global context: sub_block_id identifies the selected AIV
     *  task subslot. The AICore Scheduler writes it during materialization;
     *  the legacy scheduler seeds the equivalent per-core value at startup.
     *  args[SPMD_GLOBAL_CONTEXT_INDEX] points here. */
    GlobalContext global_context;
    // No explicit tail padding: alignas(64) rounds sizeof up to 512 (8 cache lines).

    static_assert(sizeof(args[0]) == 8);
    static_assert(
        CHIP_ALIGN_UP((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]), 64) ==
        (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0])
    );
};

static_assert(sizeof(DispatchPayload) == 512, "DispatchPayload hardware ABI size drift");
static_assert(offsetof(DispatchPayload, args) == 64, "args[] must start at cache line 1 (control block = CL0)");
static_assert(offsetof(DispatchPayload, src_payload) < 64, "src_payload (gate) must live on the CL0 control block");
