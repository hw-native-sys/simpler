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
#include <new>
#include <utility>
#include <vector>

#include "task_interface/task_args.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

namespace simpler::tmr {

class TmrEncodingCache;
class TmrEncodingCandidate;
inline InvocationStatus encode_tmr_invocation(
    const ChipStorageTaskArgs &, const PreparedInvocationView &, const TmrExecutionBindingView &,
    const TmrEncodingCache &, TmrEncodingCandidate *
);

// Each candidate owns a distinct packet through the native copy operation.
// A successfully encoded candidate can be committed only after the full
// native enqueue sequence succeeds; device failures are asynchronous.
class TmrEncodingCandidate {
public:
    TmrEncodingCandidate() = default;
    TmrEncodingCandidate(TmrEncodingCandidate &&) noexcept = default;
    TmrEncodingCandidate &operator=(TmrEncodingCandidate &&) noexcept = default;
    TmrEncodingCandidate(const TmrEncodingCandidate &) = delete;
    TmrEncodingCandidate &operator=(const TmrEncodingCandidate &) = delete;

    ByteSpan packet() const noexcept { return {packet_.data(), packet_.size()}; }
    bool structural_hit() const noexcept { return structural_hit_; }
    bool same_invocation(const TmrEncodingCandidate &other) const noexcept {
        return !packet_.empty() && packet_ == other.packet_;
    }

private:
    std::vector<uint8_t> packet_;
    std::vector<uint8_t> pending_template_;
    bool structural_hit_{false};
    friend class TmrEncodingCache;
    friend InvocationStatus encode_tmr_invocation(
        const ChipStorageTaskArgs &, const PreparedInvocationView &, const TmrExecutionBindingView &,
        const TmrEncodingCache &, TmrEncodingCandidate *
    );
};

// One entry per prepared callable. The owner serializes reads, commit and
// destruction with lookup/enqueue/close; this class has no internal lock.
// Submitted tasks never reference the cache's host storage.
class TmrEncodingCache {
public:
    TmrEncodingCache() = default;
    TmrEncodingCache(const TmrEncodingCache &) = delete;
    TmrEncodingCache &operator=(const TmrEncodingCache &) = delete;

    void commit(TmrEncodingCandidate &&candidate) noexcept {
        if (!candidate.pending_template_.empty()) {
            template_.swap(candidate.pending_template_);
            candidate.pending_template_.clear();
        }
    }

    size_t bytes() const noexcept { return template_.size(); }

private:
    // Canonical packet with tensor addresses and scalar bits zeroed. Scope,
    // generations, geometry, offset and backing size remain in the key.
    std::vector<uint8_t> template_;
    friend InvocationStatus encode_tmr_invocation(
        const ChipStorageTaskArgs &, const PreparedInvocationView &, const TmrExecutionBindingView &,
        const TmrEncodingCache &, TmrEncodingCandidate *
    );
};

inline InvocationStatus encode_tmr_invocation(
    const ChipStorageTaskArgs &args, const PreparedInvocationView &callable, const TmrExecutionBindingView &binding,
    const TmrEncodingCache &cache, TmrEncodingCandidate *out
) {
    if (out == nullptr) return InvocationStatus::InvalidArgument;
    size_t bytes = 0;
    auto status = tmr_invocation_size(callable, &bytes);
    if (status != InvocationStatus::Ok) return status;
    if (binding.device_binding_addr == 0 || binding.context_generation == 0) return InvocationStatus::InvalidBinding;
    if (args.tensor_count() != callable.tensor_count || args.scalar_count() != callable.scalar_count)
        return InvocationStatus::InvalidCounts;

    try {
        TmrEncodingCandidate candidate;
        SimplerKernelInvocationHeader header;
        std::memset(&header, 0, sizeof(header));
        header.mode = SIMPLER_MODE_KERNEL;
        header.callable_id = callable.callable_id;
        header.generation = callable.slot_generation;
        header.payload_bytes = bytes - sizeof(header);
        header.tensor_count = callable.tensor_count;
        header.scalar_count = callable.scalar_count;
        const TmrBindingRef reference{binding.device_binding_addr, binding.context_generation};
        const size_t tensors_offset = sizeof(header) + sizeof(reference);
        bool matches = cache.template_.size() == bytes &&
                       std::memcmp(cache.template_.data(), &header, sizeof(header)) == 0 &&
                       std::memcmp(cache.template_.data() + sizeof(header), &reference, sizeof(reference)) == 0;
        std::vector<ChipTensor> normalized(static_cast<size_t>(callable.tensor_count));
        for (int32_t i = 0; i < callable.tensor_count; ++i) {
            auto &tensor = normalized[static_cast<size_t>(i)];
            status = normalize_invocation_tensor(args.tensor(i), &tensor);
            if (status != InvocationStatus::Ok) return status;
            tensor.buffer.addr = 0;
            if (matches && std::memcmp(
                               cache.template_.data() + tensors_offset + static_cast<size_t>(i) * sizeof(tensor),
                               &tensor, sizeof(tensor)
                           ) != 0)
                matches = false;
        }
        candidate.structural_hit_ = matches;
        if (matches) {
            candidate.packet_ = cache.template_;
        } else {
            candidate.packet_.resize(bytes, 0);
            std::memcpy(candidate.packet_.data(), &header, sizeof(header));
            std::memcpy(candidate.packet_.data() + sizeof(header), &reference, sizeof(reference));
            if (!normalized.empty())
                std::memcpy(
                    candidate.packet_.data() + tensors_offset, normalized.data(), normalized.size() * sizeof(ChipTensor)
                );
            candidate.pending_template_ = candidate.packet_;
        }
        for (int32_t i = 0; i < callable.tensor_count; ++i) {
            const size_t addr_offset = tensors_offset + static_cast<size_t>(i) * sizeof(ChipTensor) +
                                       offsetof(ChipTensor, buffer) + offsetof(PTOBufferHandle, addr);
            std::memcpy(candidate.packet_.data() + addr_offset, &args.tensor(i).buffer.addr, sizeof(uint64_t));
        }
        const size_t scalar_offset = tensors_offset + static_cast<size_t>(callable.tensor_count) * sizeof(ChipTensor);
        if (callable.scalar_count != 0)
            std::memcpy(
                candidate.packet_.data() + scalar_offset, args.scalar_data(),
                static_cast<size_t>(callable.scalar_count) * sizeof(uint64_t)
            );
        *out = std::move(candidate);
        return InvocationStatus::Ok;
    } catch (const std::bad_alloc &) {
        return InvocationStatus::AllocationFailure;
    }
}

// The SDK stores argsSize in uint32_t. Actual packet length is taken from
// owned storage, never from an unvalidated payload_bytes field.
inline InvocationStatus validate_tmr_submission(
    const TmrEncodingCandidate &candidate, const PreparedInvocationView &callable,
    const TmrExecutionBindingView &binding
) noexcept {
    if (candidate.packet().size > std::numeric_limits<uint32_t>::max()) return InvocationStatus::InvalidSize;
    TmrInvocationView view;
    return decode_tmr_invocation(candidate.packet(), callable, binding, &view);
}

}  // namespace simpler::tmr
