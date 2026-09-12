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

#include "arg_direction.h"
#include "callable_protocol.h"
#include "kernel_invocation_header.h"

namespace simpler::kernel {

struct ByteSpan {
    const uint8_t *data{nullptr};
    size_t size{0};
};

enum class InvocationStatus {
    Ok,
    InvalidArgument,
    InvalidHeader,
    InvalidCounts,
    InvalidSignature,
    InvalidSize,
    StaleCallable,
    InvalidBinding,
    InvalidTensor,
    AllocationFailure,
};

// Values come from an independently validated, live prepared registration.
// The resource owner protects publication and borrowing through consumption.
// slot_generation versions callable residency, not its execution context.
struct PreparedInvocationView {
    int32_t callable_id;
    int32_t tensor_count;
    int32_t scalar_count;
    uint64_t slot_generation;
};

inline bool valid_invocation_counts(int32_t tensors, int32_t scalars) noexcept {
    return tensors >= 0 && tensors <= CHIP_MAX_TENSOR_ARGS && scalars >= 0 && scalars <= CHIP_MAX_SCALAR_ARGS &&
           tensors + scalars <= CHIP_MAX_TENSOR_ARGS;
}

// signature names a readable, aligned array of sig_count entries; its owning
// callable's flexible-array bounds are validated before this function is called.
inline InvocationStatus derive_invocation_counts(
    const ArgDirection *signature, int32_t sig_count, int32_t cached_scalars, int32_t *tensors, int32_t *scalars
) noexcept {
    if (tensors == nullptr || scalars == nullptr || (signature == nullptr && sig_count != 0))
        return InvocationStatus::InvalidArgument;
    if (sig_count < 0 || sig_count > CHIP_MAX_TENSOR_ARGS || cached_scalars < 0 ||
        cached_scalars > CHIP_MAX_SCALAR_ARGS)
        return InvocationStatus::InvalidCounts;
    int32_t scalar_count = 0;
    for (int32_t i = 0; i < sig_count; ++i) {
        switch (signature[i]) {
        case ArgDirection::SCALAR:
            ++scalar_count;
            break;
        case ArgDirection::IN:
        case ArgDirection::OUT:
        case ArgDirection::INOUT:
            if (scalar_count != 0) return InvocationStatus::InvalidSignature;
            break;
        default:
            return InvocationStatus::InvalidSignature;
        }
    }
    if (!valid_invocation_counts(sig_count - scalar_count, scalar_count) ||
        (cached_scalars != 0 && cached_scalars != scalar_count))
        return InvocationStatus::InvalidSignature;
    *tensors = sig_count - scalar_count;
    *scalars = scalar_count;
    return InvocationStatus::Ok;
}

inline bool valid_prepared_invocation(const PreparedInvocationView &view) noexcept {
    return view.callable_id >= 0 && view.callable_id < MAX_REGISTERED_CALLABLE_IDS && view.slot_generation != 0 &&
           valid_invocation_counts(view.tensor_count, view.scalar_count);
}

inline InvocationStatus validate_invocation_header(
    ByteSpan packet, const PreparedInvocationView &trusted, SimplerKernelInvocationHeader *out
) noexcept {
    if (out == nullptr || packet.data == nullptr) return InvocationStatus::InvalidArgument;
    if (packet.size < sizeof(SimplerKernelInvocationHeader)) return InvocationStatus::InvalidSize;
    SimplerKernelInvocationHeader header{};
    std::memcpy(&header, packet.data, sizeof(header));
    if (header.mode != SIMPLER_MODE_KERNEL || header.callable_id < 0 ||
        header.callable_id >= MAX_REGISTERED_CALLABLE_IDS || header.generation == 0)
        return InvocationStatus::InvalidHeader;
    if (!valid_invocation_counts(header.tensor_count, header.scalar_count) || header.host_copy_tensor_count != 0)
        return InvocationStatus::InvalidCounts;
    if (header.payload_bytes != packet.size - sizeof(header)) return InvocationStatus::InvalidSize;
    if (!valid_prepared_invocation(trusted)) return InvocationStatus::InvalidArgument;
    if (header.callable_id != trusted.callable_id || header.generation != trusted.slot_generation)
        return InvocationStatus::StaleCallable;
    if (header.tensor_count != trusted.tensor_count || header.scalar_count != trusted.scalar_count)
        return InvocationStatus::InvalidCounts;
    *out = header;
    return InvocationStatus::Ok;
}

}  // namespace simpler::kernel
