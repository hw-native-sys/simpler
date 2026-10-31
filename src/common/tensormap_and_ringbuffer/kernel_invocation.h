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
#include <type_traits>

#include "task_interface/callable_protocol.h"
#include "task_interface/kernel_capture.h"
#include "task_interface/kernel_invocation_header.h"
#include "task_interface/task_args.h"

namespace simpler::tmr {

struct KernelCallableIdentity {
    int32_t callable_id{-1};
    uint64_t generation{0};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
};

struct EagerKernelBinding {
    uint64_t runtime_address{0};
    uint64_t run_epoch{0};
    KernelCaptureReference reference{};
};
static_assert(std::is_trivially_copyable_v<EagerKernelBinding> && std::is_standard_layout_v<EagerKernelBinding>);
static_assert(sizeof(EagerKernelBinding) == 40);
static_assert(offsetof(EagerKernelBinding, reference) == 16);

inline bool valid_kernel_reference(const KernelCaptureReference &reference) noexcept {
    return reference.control_address != 0 && reference.control_address % alignof(KernelCaptureControl) == 0 &&
           reference.capture_generation != 0 && reference.context_generation != 0;
}

inline bool unbound_kernel_reference(const KernelCaptureReference &reference) noexcept {
    return reference.control_address == 0 && reference.capture_generation == 0 && reference.context_generation == 0;
}

inline bool same_kernel_reference(const KernelCaptureReference &a, const KernelCaptureReference &b) noexcept {
    return a.control_address == b.control_address && a.capture_generation == b.capture_generation &&
           a.context_generation == b.context_generation;
}

inline bool valid_kernel_identity(const KernelCallableIdentity &identity) noexcept {
    return identity.callable_id >= 0 && identity.callable_id < MAX_REGISTERED_CALLABLE_IDS &&
           identity.generation != 0 && identity.tensor_count <= CHIP_MAX_TENSOR_ARGS &&
           identity.scalar_count <= CHIP_MAX_SCALAR_ARGS &&
           identity.tensor_count + identity.scalar_count <= CHIP_MAX_TENSOR_ARGS;
}

inline size_t kernel_invocation_bytes(uint32_t tensors, uint32_t scalars) noexcept {
    return sizeof(SimplerKernelInvocationHeader) + sizeof(EagerKernelBinding) +
           static_cast<size_t>(tensors) * sizeof(ChipTensor) + static_cast<size_t>(scalars) * sizeof(uint64_t);
}

inline bool normalize_kernel_tensor(const ChipTensor &input, ChipTensor *output) noexcept {
    const uint64_t element_bytes = get_element_size(input.dtype);
    if (output == nullptr || input.address_space != AddressSpace::DEVICE || element_bytes == 0 || input.ndims == 0 ||
        input.ndims > MAX_TENSOR_DIMS)
        return false;
    bool empty = false;
    for (uint32_t i = 0; i < input.ndims; ++i)
        empty |= input.shapes[i] == 0;
    if (!empty) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        uint64_t elements = 1;
        uint64_t extent = 1;
        for (uint32_t i = 0; i < input.ndims; ++i) {
            if (input.strides[i] == 0 || elements > limit / input.shapes[i]) return false;
            elements *= input.shapes[i];
            const uint64_t term = static_cast<uint64_t>(input.shapes[i] - 1) * input.strides[i];
            if (term > limit - extent) return false;
            extent += term;
        }
        if (elements > limit / element_bytes || input.buffer.addr == 0 ||
            input.buffer.size > limit - input.buffer.addr || input.start_offset > limit - extent ||
            input.start_offset + extent > input.buffer.size / element_bytes)
            return false;
    }
    ChipTensor normalized{};
    normalized.buffer = input.buffer;
    normalized.start_offset = input.start_offset;
    normalized.ndims = input.ndims;
    normalized.dtype = input.dtype;
    normalized.address_space = input.address_space;
    for (uint32_t i = 0; i < input.ndims; ++i) {
        normalized.shapes[i] = input.shapes[i];
        normalized.strides[i] = input.strides[i];
    }
    *output = normalized;
    return true;
}

// A wholly empty reference is a cold staging snapshot. Device consumption
// requires a bound reference established by independent registration.
// The packet owns descriptors and scalar bits; caller-owned DEVICE storage is borrowed.
template <typename EntryArgs>
inline bool encode_kernel_invocation(
    const EntryArgs &args, const KernelCallableIdentity &identity, EagerKernelBinding binding, void *packet,
    size_t bytes
) {
    if (packet == nullptr || !valid_kernel_identity(identity) || binding.runtime_address == 0 ||
        binding.run_epoch == 0 ||
        (!valid_kernel_reference(binding.reference) && !unbound_kernel_reference(binding.reference)) ||
        args.tensor_count() != static_cast<int32_t>(identity.tensor_count) ||
        args.scalar_count() != static_cast<int32_t>(identity.scalar_count) ||
        bytes != kernel_invocation_bytes(identity.tensor_count, identity.scalar_count))
        return false;
    for (uint32_t i = 0; i < identity.tensor_count; ++i) {
        ChipTensor normalized{};
        if (!normalize_kernel_tensor(args.tensor(i).to_boundary(), &normalized)) return false;
    }
    SimplerKernelInvocationHeader header{};
    header.mode = SIMPLER_MODE_KERNEL;
    header.callable_id = identity.callable_id;
    header.generation = identity.generation;
    header.payload_bytes = bytes - sizeof(header);
    header.tensor_count = static_cast<int32_t>(identity.tensor_count);
    header.scalar_count = static_cast<int32_t>(identity.scalar_count);
    auto *dest = static_cast<unsigned char *>(packet);
    std::memcpy(dest, &header, sizeof(header));
    std::memcpy(dest + sizeof(header), &binding, sizeof(binding));
    size_t offset = sizeof(header) + sizeof(binding);
    for (uint32_t i = 0; i < identity.tensor_count; ++i) {
        ChipTensor normalized{};
        normalize_kernel_tensor(args.tensor(i).to_boundary(), &normalized);
        std::memcpy(dest + offset, &normalized, sizeof(normalized));
        offset += sizeof(normalized);
    }
    std::memcpy(dest + offset, args.scalars_, identity.scalar_count * sizeof(uint64_t));
    return true;
}

// Validation completes before any invocation storage is changed.
template <typename EntryArgs>
inline bool consume_kernel_invocation(
    const void *packet, size_t bytes, const KernelCallableIdentity &identity, EagerKernelBinding binding,
    EntryArgs *output
) {
    if (packet == nullptr || output == nullptr || !valid_kernel_identity(identity) || binding.runtime_address == 0 ||
        binding.run_epoch == 0 || !valid_kernel_reference(binding.reference) ||
        bytes != kernel_invocation_bytes(identity.tensor_count, identity.scalar_count))
        return false;
    SimplerKernelInvocationHeader header{};
    EagerKernelBinding encoded{};
    const auto *source = static_cast<const unsigned char *>(packet);
    std::memcpy(&header, source, sizeof(header));
    if (header.mode != SIMPLER_MODE_KERNEL || header.callable_id != identity.callable_id ||
        header.generation != identity.generation || header.payload_bytes != bytes - sizeof(header) ||
        header.tensor_count != static_cast<int32_t>(identity.tensor_count) ||
        header.scalar_count != static_cast<int32_t>(identity.scalar_count) || header.host_copy_tensor_count != 0 ||
        header.reserved_ != 0)
        return false;
    std::memcpy(&encoded, source + sizeof(header), sizeof(encoded));
    if (encoded.runtime_address != binding.runtime_address || encoded.run_epoch != binding.run_epoch ||
        !same_kernel_reference(encoded.reference, binding.reference))
        return false;
    const size_t tensors_offset = sizeof(header) + sizeof(encoded);
    for (uint32_t i = 0; i < identity.tensor_count; ++i) {
        ChipTensor tensor{};
        ChipTensor normalized{};
        std::memcpy(&tensor, source + tensors_offset + i * sizeof(tensor), sizeof(tensor));
        if (!normalize_kernel_tensor(tensor, &normalized)) return false;
    }
    for (uint32_t i = 0; i < identity.tensor_count; ++i) {
        ChipTensor tensor{};
        ChipTensor normalized{};
        std::memcpy(&tensor, source + tensors_offset + i * sizeof(tensor), sizeof(tensor));
        normalize_kernel_tensor(tensor, &normalized);
        using RuntimeTensor = std::remove_reference_t<decltype(output->tensors_[i])>;
        output->tensors_[i] = RuntimeTensor::from_boundary(normalized);
    }
    std::memcpy(
        output->scalars_, source + tensors_offset + identity.tensor_count * sizeof(ChipTensor),
        identity.scalar_count * sizeof(uint64_t)
    );
    output->tensor_count_ = static_cast<int32_t>(identity.tensor_count);
    output->scalar_count_ = static_cast<int32_t>(identity.scalar_count);
    return true;
}

}  // namespace simpler::tmr
