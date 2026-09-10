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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel_invocation_header.h"

// CANN deep-copies this prefix and the following payload into each launch.
// residency_address is the issuing context's stable device slot descriptor,
// supplied by the binder, never an address supplied by a tensor/callable image.
// Its allocation stays alive until every referencing graph is destroyed and
// all executions have completed. Slot updates require external quiescence.
struct SimplerKernelDispatchArgs {
    uint64_t packet_bytes;
    uint64_t residency_address;
    SimplerKernelInvocationHeader invocation;
};

// Direct AICPU entry return values, not host C API or latched runtime codes.
// CANN propagates nonzero entry failure through the caller's synchronization.
enum class KernelDispatchStatus : int32_t {
    Success = 0,
    InvalidArgs = 1,
    NotResident = 2,
    Stale = 3,
    UnsupportedPayload = 4,
};

static_assert(
    std::is_trivially_copyable_v<SimplerKernelDispatchArgs> && std::is_standard_layout_v<SimplerKernelDispatchArgs>
);

extern "C" int simpler_aicpu_kernel_exec(void *args);
