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

#include "callable_protocol.h"
#include "kernel_invocation_header.h"

// Immutable device-resident descriptor, published before registration. The
// invocation consumer validates this comparand on every execution/replay.
struct KernelCallableDeviceResidency {
    uint64_t generation;
    uint64_t device_address;
    uint64_t bytes;
    int32_t callable_id;
    uint32_t reserved;
};

inline bool kernel_callable_residency_matches(
    const SimplerKernelInvocationHeader &invocation, const KernelCallableDeviceResidency &resident
) {
    return invocation.mode == SIMPLER_MODE_KERNEL && invocation.callable_id >= 0 &&
           invocation.callable_id < MAX_REGISTERED_CALLABLE_IDS && invocation.callable_id == resident.callable_id &&
           invocation.generation != 0 && invocation.generation == resident.generation && resident.device_address != 0 &&
           resident.bytes != 0 && resident.reserved == 0;
}

static_assert(
    std::is_trivially_copyable_v<KernelCallableDeviceResidency> &&
    std::is_standard_layout_v<KernelCallableDeviceResidency>
);
