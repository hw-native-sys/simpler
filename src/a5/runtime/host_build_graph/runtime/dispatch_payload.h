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

/** Per-slot kernel entry, fixed argument vector and invocation contexts. */

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

// Byte offsets into TaskPayload used by resident workers to materialize args[]. The AICore .o does not include
// runtime_types.h, so it reads tensor_count / scalar_count / tensors[] /
// scalars[] through these constants. The Host runtime maker
// static_asserts them against offsetof where the full struct is visible, so any
// TaskPayload layout drift fails the build rather than corrupting args[].
constexpr uint32_t TASKPAYLOAD_TENSOR_COUNT_OFFSET = 0;
constexpr uint32_t TASKPAYLOAD_SCALAR_COUNT_OFFSET = 4;
// Offsets of the fields that NAME the argument regions, not of the regions themselves:
// a region's address is (src + <field offset>) + <the int32 delta stored there>.
constexpr uint32_t TASKPAYLOAD_TENSORS_DELTA_OFFSET = 12;
constexpr uint32_t TASKPAYLOAD_SCALARS_DELTA_OFFSET = 16;
constexpr uint32_t TASKPAYLOAD_FANIN_DELTA_OFFSET = 20;
constexpr uint32_t TASKPAYLOAD_TENSOR_STRIDE = 128;  // sizeof(simpler::hbg::Tensor)

struct alignas(64) DispatchPayload {
    uint64_t function_bin_addr;
    LocalContext local_context;

    // Context suffix pointers remain stable for the lifetime of this slot.
    alignas(64) uint64_t args[DISPATCH_MAX_ARGS];

    GlobalContext global_context;

    static_assert(sizeof(args[0]) == 8);
    static_assert(
        CHIP_ALIGN_UP((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0]), 64) ==
        (MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0])
    );
};

static_assert(sizeof(DispatchPayload) == 512, "DispatchPayload hardware ABI size drift");
static_assert(offsetof(DispatchPayload, args) == 64, "args[] must start at cache line 1 (control block = CL0)");
