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
#include <limits>

#include "kernel_dispatch_args.h"
#include "kernel_invocation_validation.h"
#include "callable_protocol.h"
#include "arg_direction.h"
#include "aicpu/kernel_invocation_consumer.h"

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_kernel_exec(void *arg) {
    if (arg == nullptr || reinterpret_cast<uintptr_t>(arg) % alignof(SimplerKernelDispatchArgs) != 0)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    // The packet prefix and declared packet_bytes must describe the actual
    // CANN argument allocation. The entry ABI exposes no independent length.
    // Before runtime admission, binding_address has no independently trusted
    // owner. A rejected prefix must not dereference it for AICore cancellation.
    const auto &args = *static_cast<const SimplerKernelDispatchArgs *>(arg);
    const auto &invocation = args.invocation;
    if (args.packet_bytes < sizeof(args) ||
        args.packet_bytes > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(arg) ||
        invocation.payload_bytes != args.packet_bytes - sizeof(args) || invocation.mode != SIMPLER_MODE_KERNEL ||
        invocation.callable_id < 0 || invocation.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        invocation.tensor_count < 0 || invocation.tensor_count > CHIP_MAX_TENSOR_ARGS || invocation.scalar_count < 0 ||
        invocation.scalar_count > CHIP_MAX_SCALAR_ARGS ||
        invocation.tensor_count > CHIP_MAX_TENSOR_ARGS - invocation.scalar_count ||
        !simpler::kernel::valid_host_copy_tensor_count(invocation.tensor_count, invocation.host_copy_tensor_count) ||
        invocation.reserved_ != 0)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    if (args.chip_callable_address == 0 || args.chip_callable_address % alignof(ChipCallable) != 0 ||
        args.chip_callable_bytes < sizeof(ChipCallable) ||
        args.chip_callable_bytes > std::numeric_limits<uintptr_t>::max() - args.chip_callable_address)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);

    const auto *payload = static_cast<const unsigned char *>(arg) + sizeof(args);
    return consume_kernel_invocation(
        args, *reinterpret_cast<const ChipCallable *>(args.chip_callable_address),
        static_cast<size_t>(args.chip_callable_bytes), payload, static_cast<size_t>(invocation.payload_bytes)
    );
}
