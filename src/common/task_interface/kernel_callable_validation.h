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

#include "callable.h"
#include "kernel_invocation_validation.h"

inline constexpr size_t kKernelCallableByteLimit = 512ULL * 1024 * 1024;

namespace simpler::kernel {

// Check metadata before cache maintenance or reading the image. Readability
// and lifetime of the range still belong to the trusted resource provider.
inline bool valid_kernel_callable_span(const void *callable, size_t bytes) noexcept {
    const uintptr_t address = reinterpret_cast<uintptr_t>(callable);
    // Misaligned parents also misalign every inline CoreCallable.
    return address != 0 && address % alignof(ChipCallable) == 0 && bytes >= sizeof(ChipCallable) &&
           bytes <= kKernelCallableByteLimit && bytes <= UINTPTR_MAX - address;
}

// Shared by the C entry and the residency cache, before any hash or upload.
inline bool valid_kernel_callable_image(const void *callable, size_t callable_size) noexcept {
    if (!valid_kernel_callable_span(callable, callable_size)) return false;
    const auto *bytes = static_cast<const uint8_t *>(callable);
    int32_t sig_count = 0;
    int32_t cached_scalars = 0;
    std::memcpy(&sig_count, bytes + offsetof(ChipCallable, sig_count_), sizeof(sig_count));
    std::memcpy(&cached_scalars, bytes + offsetof(ChipCallable, scalar_count_), sizeof(cached_scalars));
    int32_t tensors = 0;
    int32_t scalars = 0;
    const auto *signature = reinterpret_cast<const ArgDirection *>(bytes + offsetof(ChipCallable, signature_));
    if (simpler::kernel::derive_invocation_counts(signature, sig_count, cached_scalars, &tensors, &scalars) !=
        simpler::kernel::InvocationStatus::Ok)
        return false;

    const auto *image = static_cast<const ChipCallable *>(callable);
    const auto valid_name = [](const char *name, uint32_t length) {
        return length < CALLABLE_FUNC_NAME_MAX && name[length] == '\0' && std::memchr(name, '\0', length) == nullptr;
    };
    if (!valid_name(image->func_name_, image->func_name_len_) ||
        !valid_name(image->config_name_, image->config_name_len_))
        return false;
    const size_t storage_size = callable_size - offsetof(ChipCallable, storage_);
    size_t used = image->binary_size_;
    constexpr size_t max_children = sizeof(image->child_offsets_) / sizeof(image->child_offsets_[0]);
    if (used > storage_size || image->child_count_ < 0 || static_cast<size_t>(image->child_count_) > max_children)
        return false;
    for (int32_t i = 0; i < image->child_count_; ++i) {
        const size_t offset = image->child_offsets_[i];
        // Canonical child packing starts at the next aligned byte after
        // the preceding binary; subtraction precedes every span read.
        const size_t padding = (CALLABLE_ALIGN - used % CALLABLE_ALIGN) % CALLABLE_ALIGN;
        if (padding > storage_size - used || offset != used + padding ||
            CoreCallable::binary_data_offset() > storage_size - offset)
            return false;
        const auto *child = reinterpret_cast<const CoreCallable *>(image->storage_ + offset);
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS) return false;
        const size_t binary_offset = offset + CoreCallable::binary_data_offset();
        if (child->binary_size_ > storage_size - binary_offset) return false;
        used = binary_offset + child->binary_size_;
    }
    if (used != storage_size) return false;
    return true;
}

}  // namespace simpler::kernel
