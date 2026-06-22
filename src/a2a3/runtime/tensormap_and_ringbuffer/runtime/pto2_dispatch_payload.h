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

#pragma once

#include <stdint.h>

#include "intrinsic.h"
#include "pto_types.h"

#ifndef PTO2_DISPATCH_MAX_ARGS
#define PTO2_DISPATCH_MAX_ARGS (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + PTO2_EXT_PARAMS_COUNT)
#endif

#ifndef PTO2_ALIGN_UP
#define PTO2_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

// Verify hardcoded indices in intrinsic.h match the computed values.
static_assert((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) == SPMD_LOCAL_CONTEXT_INDEX, "LOCAL_CONTEXT_INDEX out of sync with intrinsic.h");
static_assert((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + 1) == SPMD_GLOBAL_CONTEXT_INDEX, "GLOBAL_CONTEXT_INDEX out of sync with intrinsic.h");

struct alignas(64) PTO2DispatchPayload
{
    uint64_t function_bin_addr;
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];

    LocalContext local_context;

    GlobalContext global_context;

    /** Speculative early-dispatch gate. 0 = ready: AICore executes on pickup.
     *  1 = not-ready: AICore waits until AICPU rings the doorbell
     *  (DATA_MAIN_BASE high 32 == this dispatch's reg_task_id) before executing. */
    volatile uint32_t not_ready;
    uint8_t reserved_payload_abi_pad[4];

    static_assert(sizeof(args[0]) == 8);
    static_assert(PTO2_ALIGN_UP((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]), 64) == (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]));
};

static_assert(sizeof(PTO2DispatchPayload) == 512, "PTO2DispatchPayload hardware ABI size drift");
