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
#include <cstring>
#include <limits>

#include "kernel_dispatch_args.h"
#include "callable_protocol.h"
#include "arg_direction.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/kernel_invocation_consumer.h"

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_kernel_exec(void *arg) {
    if (arg == nullptr || reinterpret_cast<uintptr_t>(arg) % alignof(SimplerKernelDispatchArgs) != 0)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    // The packet prefix and declared packet_bytes must describe the actual
    // CANN argument allocation. The entry ABI exposes no independent length.
    const auto &args = *static_cast<const SimplerKernelDispatchArgs *>(arg);
    const auto &invocation = args.invocation;
    if (args.packet_bytes < sizeof(args) || args.packet_bytes > std::numeric_limits<size_t>::max() ||
        invocation.payload_bytes != args.packet_bytes - sizeof(args) || invocation.mode != SIMPLER_MODE_KERNEL ||
        invocation.callable_id < 0 || invocation.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        invocation.generation == 0 || invocation.tensor_count < 0 || invocation.tensor_count > CHIP_MAX_TENSOR_ARGS ||
        invocation.scalar_count < 0 || invocation.scalar_count > CHIP_MAX_SCALAR_ARGS ||
        invocation.tensor_count > CHIP_MAX_TENSOR_ARGS - invocation.scalar_count ||
        invocation.host_copy_tensor_count != 0)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    if (args.residency_address == 0 || args.residency_address % alignof(KernelCallableDeviceResidency) != 0 ||
        args.residency_address > std::numeric_limits<uintptr_t>::max() - sizeof(KernelCallableDeviceResidency))
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);

    const auto *descriptor = reinterpret_cast<const KernelCallableDeviceResidency *>(args.residency_address);
    // Read the current slot on every invocation, including graph replay. The
    // invalidation completes before the snapshot; no concurrent slot writer
    // is permitted while any execution can use this residency.
    cache_invalidate_range(descriptor, sizeof(*descriptor));
    KernelCallableDeviceResidency resident;
    std::memcpy(&resident, descriptor, sizeof(resident));
    if (resident.callable_id != invocation.callable_id || resident.generation == 0 || resident.device_address == 0 ||
        resident.bytes == 0 || resident.reserved != 0)
        return static_cast<int>(KernelDispatchStatus::NotResident);
    if (!kernel_callable_residency_matches(invocation, resident)) return static_cast<int>(KernelDispatchStatus::Stale);

    const auto *payload = static_cast<const unsigned char *>(arg) + sizeof(args);
    return consume_kernel_invocation(invocation, resident, payload, static_cast<size_t>(invocation.payload_bytes));
}
