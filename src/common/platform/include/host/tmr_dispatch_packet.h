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
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "kernel_callable_residency.h"
#include "kernel_dispatch_args.h"
#include "worker/tmr_kernel_invocation.h"

namespace simpler::tmr {

// residency_address comes from the issuing cache's resolved descriptor, not
// from the invocation. The owner keeps that device allocation alive through
// all referring tasks/graphs. This helper only adds the transport envelope;
// it neither dereferences the address nor supplies missing provider metadata.
inline InvocationStatus make_tmr_dispatch_packet(
    const TmrEncodingCandidate &candidate, const PreparedInvocationView &callable,
    const TmrExecutionBindingView &binding, uint64_t residency_address, std::vector<uint8_t> *out
) {
    if (out == nullptr) return InvocationStatus::InvalidArgument;
    const auto status = validate_tmr_submission(candidate, callable, binding);
    if (status != InvocationStatus::Ok) return status;
    if (residency_address == 0 || residency_address % alignof(KernelCallableDeviceResidency) != 0 ||
        residency_address > std::numeric_limits<uintptr_t>::max() - sizeof(KernelCallableDeviceResidency))
        return InvocationStatus::InvalidBinding;
    constexpr size_t prefix_bytes = offsetof(SimplerKernelDispatchArgs, invocation);
    static_assert(sizeof(SimplerKernelDispatchArgs) == prefix_bytes + sizeof(SimplerKernelInvocationHeader));
    const auto packet = candidate.packet();
    if (packet.size > std::numeric_limits<uint32_t>::max() - prefix_bytes) return InvocationStatus::InvalidSize;
    try {
        SimplerKernelDispatchArgs envelope{};
        envelope.packet_bytes = prefix_bytes + packet.size;
        envelope.residency_address = residency_address;
        std::vector<uint8_t> encoded(static_cast<size_t>(envelope.packet_bytes));
        std::memcpy(encoded.data(), &envelope, prefix_bytes);
        std::memcpy(encoded.data() + prefix_bytes, packet.data, packet.size);
        *out = std::move(encoded);
        return InvocationStatus::Ok;
    } catch (const std::bad_alloc &) {
        return InvocationStatus::AllocationFailure;
    }
}

}  // namespace simpler::tmr
