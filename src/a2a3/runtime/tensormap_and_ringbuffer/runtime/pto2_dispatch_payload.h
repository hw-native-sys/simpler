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

#include <stdint.h>

#include "intrinsic.h"
#include "pto_types.h"

/** Max dispatch arguments: 32 scalars + up to 16 tensor pointers + ext params */
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

/**
 * Per-core dispatch payload: function address + args[] + SPMD context.
 *
 * AICPU maintains a static array s_pto2_payload_per_core[RUNTIME_MAX_WORKER].
 * AICore caches a pointer to its per-core slot at startup (via Handshake.task)
 * and reads from it on each dispatch.
 *
 * The struct is cache-line aligned to prevent false sharing across
 * concurrently dispatched cores.
 */
struct alignas(64) PTO2DispatchPayload {
    uint64_t function_bin_addr;            /**< Kernel entry address in GM (set by Scheduler) */
    uint64_t args[PTO2_DISPATCH_MAX_ARGS]; /**< Kernel arguments (GM pointers + scalars + ext params) */

    /** Per-dispatch context: block_idx and block_num.
     *  Written by build_payload() before each dispatch.
     *  args[SPMD_LOCAL_CONTEXT_INDEX] points here. */
    LocalContext local_context;

    /** Per-core global context: sub_block_id (AIV lane identity).
     *  Initialized once by init_global_context() at runtime startup.
     *  args[SPMD_GLOBAL_CONTEXT_INDEX] points here. */
    GlobalContext global_context;

    uint8_t reserved_payload_abi_pad[8];

    static_assert(sizeof(args[0]) == 8);
    static_assert(
        PTO2_ALIGN_UP((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]), 64) ==
        (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0])
    );
};

static_assert(sizeof(PTO2DispatchPayload) == 512, "PTO2DispatchPayload hardware ABI size drift");
