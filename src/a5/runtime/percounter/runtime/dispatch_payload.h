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

/** Percounter's mutable per-lane invocation payload. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel_args.h"

struct alignas(64) DispatchPayload {
    uint64_t function_bin_addr;
    LocalContext local_context;

    // Context suffix pointers remain stable for the lifetime of this slot.
    alignas(64) uint64_t args[DISPATCH_MAX_ARGS];

    GlobalContext global_context;

    static_assert(sizeof(args[0]) == 8);
    static_assert(((MAX_TENSOR_ARGS + MAX_SCALAR_ARGS) * sizeof(args[0])) % 64 == 0);
};

static_assert(sizeof(DispatchPayload) == 512, "DispatchPayload hardware ABI size drift");
static_assert(offsetof(DispatchPayload, args) == 64, "args[] must start at cache line 1 (control block = CL0)");

static_assert(std::is_standard_layout_v<DispatchPayload> && std::is_trivially_copyable_v<DispatchPayload>);
