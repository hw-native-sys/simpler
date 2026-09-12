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

#include <cstring>
#include <limits>
#include <type_traits>

#include "task_interface/kernel_invocation_validation.h"
#include "task_interface/tensor.h"

namespace simpler::tmr {

using kernel::ByteSpan;
using kernel::InvocationStatus;
using kernel::PreparedInvocationView;

// Borrowed from the owner of an immutable execution binding. This view does
// not allocate, publish, retain or release the referenced device resources.
// The provider owns the binding format and capacity validation. The address
// is opaque here; context_generation is independent of callable residency.
struct TmrExecutionBindingView {
    uint64_t device_binding_addr;
    uint64_t context_generation;
};

struct TmrBindingRef {
    uint64_t device_binding_addr;
    uint64_t context_generation;
};
static_assert(std::is_trivially_copyable_v<TmrBindingRef> && std::is_standard_layout_v<TmrBindingRef>);
static_assert(sizeof(TmrBindingRef) == 16);
static_assert(offsetof(TmrBindingRef, device_binding_addr) == 0);
static_assert(offsetof(TmrBindingRef, context_generation) == 8);

inline InvocationStatus tmr_invocation_size(const PreparedInvocationView &callable, size_t *out) noexcept {
    if (out == nullptr || !kernel::valid_prepared_invocation(callable)) return InvocationStatus::InvalidArgument;
    // Validated counts bound each term and the sum on all supported hosts.
    *out = sizeof(SimplerKernelInvocationHeader) + sizeof(TmrBindingRef) +
           static_cast<size_t>(callable.tensor_count) * sizeof(ChipTensor) +
           static_cast<size_t>(callable.scalar_count) * sizeof(uint64_t);
    return InvocationStatus::Ok;
}

inline InvocationStatus normalize_invocation_tensor(const ChipTensor &input, ChipTensor *out) noexcept {
    if (out == nullptr) return InvocationStatus::InvalidArgument;
    const uint64_t element_bytes = get_element_size(input.dtype);
    if (input.address_space != AddressSpace::DEVICE || element_bytes == 0 || input.ndims == 0 ||
        input.ndims > MAX_TENSOR_DIMS)
        return InvocationStatus::InvalidTensor;
    bool empty = false;
    for (uint32_t i = 0; i < input.ndims; ++i)
        empty |= input.shapes[i] == 0;
    if (!empty) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        uint64_t elements = 1;
        uint64_t extent = 1;
        for (uint32_t i = 0; i < input.ndims; ++i) {
            if (input.strides[i] == 0 || elements > limit / input.shapes[i]) return InvocationStatus::InvalidTensor;
            elements *= input.shapes[i];
            const uint64_t term = static_cast<uint64_t>(input.shapes[i] - 1) * input.strides[i];
            if (term > limit - extent) return InvocationStatus::InvalidTensor;
            extent += term;
        }
        if (elements > limit / element_bytes || input.buffer.addr == 0 ||
            input.buffer.size > limit - input.buffer.addr || input.start_offset > limit - extent ||
            input.start_offset + extent > input.buffer.size / element_bytes)
            return InvocationStatus::InvalidTensor;
    }
    // Empty views capture no bytes. Canonical row-major empty shapes can have
    // zero strides; neither an address nor backing bytes are required.
    ChipTensor normalized;
    std::memset(&normalized, 0, sizeof(normalized));
    normalized.buffer = input.buffer;
    normalized.start_offset = input.start_offset;
    normalized.ndims = input.ndims;
    normalized.dtype = input.dtype;
    normalized.address_space = input.address_space;
    for (uint32_t i = 0; i < input.ndims; ++i) {
        normalized.shapes[i] = input.shapes[i];
        normalized.strides[i] = input.strides[i];
    }
    std::memcpy(out, &normalized, sizeof(normalized));
    return InvocationStatus::Ok;
}

class TmrInvocationView;
inline InvocationStatus decode_tmr_invocation(
    ByteSpan packet, const PreparedInvocationView &trusted_callable, const TmrExecutionBindingView &trusted_binding,
    TmrInvocationView *out
) noexcept;

// The immutable packet storage outlives this view and every read through it.
class TmrInvocationView {
public:
    int32_t tensor_count() const noexcept { return header_.tensor_count; }
    int32_t scalar_count() const noexcept { return header_.scalar_count; }
    bool valid() const noexcept { return packet_.data != nullptr; }

    bool tensor(int32_t index, ChipTensor *out) const noexcept {
        if (out == nullptr || !valid() || index < 0 || index >= tensor_count()) return false;
        const size_t offset = sizeof(header_) + sizeof(TmrBindingRef) + static_cast<size_t>(index) * sizeof(*out);
        std::memcpy(out, packet_.data + offset, sizeof(*out));
        return true;
    }

    bool scalar(int32_t index, uint64_t *out) const noexcept {
        if (out == nullptr || !valid() || index < 0 || index >= scalar_count()) return false;
        const size_t offset = sizeof(header_) + sizeof(TmrBindingRef) +
                              static_cast<size_t>(tensor_count()) * sizeof(ChipTensor) +
                              static_cast<size_t>(index) * sizeof(*out);
        std::memcpy(out, packet_.data + offset, sizeof(*out));
        return true;
    }

private:
    ByteSpan packet_{};
    SimplerKernelInvocationHeader header_{};
    friend InvocationStatus decode_tmr_invocation(
        ByteSpan, const PreparedInvocationView &, const TmrExecutionBindingView &, TmrInvocationView *
    ) noexcept;
};

inline InvocationStatus decode_tmr_invocation(
    ByteSpan packet, const PreparedInvocationView &trusted_callable, const TmrExecutionBindingView &trusted_binding,
    TmrInvocationView *out
) noexcept {
    if (out == nullptr) return InvocationStatus::InvalidArgument;
    TmrInvocationView candidate;
    auto status = kernel::validate_invocation_header(packet, trusted_callable, &candidate.header_);
    if (status != InvocationStatus::Ok) return status;
    size_t expected_bytes = 0;
    status = tmr_invocation_size(trusted_callable, &expected_bytes);
    if (status != InvocationStatus::Ok) return status;
    if (packet.size != expected_bytes) return InvocationStatus::InvalidSize;
    TmrBindingRef binding{};
    std::memcpy(&binding, packet.data + sizeof(SimplerKernelInvocationHeader), sizeof(binding));
    if (trusted_binding.device_binding_addr == 0 || trusted_binding.context_generation == 0 ||
        binding.device_binding_addr != trusted_binding.device_binding_addr ||
        binding.context_generation != trusted_binding.context_generation)
        return InvocationStatus::InvalidBinding;
    candidate.packet_ = packet;
    for (int32_t i = 0; i < candidate.tensor_count(); ++i) {
        ChipTensor tensor{};
        ChipTensor normalized{};
        candidate.tensor(i, &tensor);
        status = normalize_invocation_tensor(tensor, &normalized);
        if (status != InvocationStatus::Ok) return status;
    }
    *out = candidate;
    return InvocationStatus::Ok;
}

}  // namespace simpler::tmr
