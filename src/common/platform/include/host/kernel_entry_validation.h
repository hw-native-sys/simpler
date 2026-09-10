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

#include <stddef.h>
#include <stdint.h>
#include <cstring>

#include "callable.h"
#include "callable_protocol.h"
#include "kernel_invocation_validation.h"
#include "runtime_c_api.h"

/**
 * Structural argument validation shared by every host-runtime component's
 * kernel-mode entries. Both platform c_api implementations (onboard and sim)
 * route through these checks, so a stub and a real implementation accept and
 * reject exactly the same arguments — the same parity rule the shared phase
 * machine follows. Each function returns 0 or a classified host error and
 * mutates nothing; logging stays with the callers.
 */

/* A binary buffer and its size describe one object, so they are valid only
   present-together or absent-together. */
inline bool kernel_binary_span_is_consistent(const void *binary, size_t size) {
    return (binary == nullptr) == (size == 0);
}

inline int validate_kernel_init_args(
    const void *ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size, const void *config,
    uint64_t context_generation
) {
    if (ctx == nullptr || config == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (device_id < 0 || context_generation == 0) return PTO_RUNTIME_ERR_INTERNAL;
    if (!kernel_binary_span_is_consistent(aicpu_binary, aicpu_size) ||
        !kernel_binary_span_is_consistent(aicore_binary, aicore_size) ||
        !kernel_binary_span_is_consistent(dispatcher_binary, dispatcher_size)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}

inline int validate_kernel_prepare_callable_args(
    const void *ctx, const void *callable, size_t callable_size, const SimplerCallableHandle *out_handle
) {
    if (ctx == nullptr || callable == nullptr || out_handle == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (callable_size < sizeof(ChipCallable)) return PTO_RUNTIME_ERR_INTERNAL;
    /* ChipCallable's storage_ is CALLABLE_CHILD_ALIGN-aligned relative to the
       header, so a misaligned image puts every child at a misaligned address. */
    if (reinterpret_cast<uintptr_t>(callable) % alignof(ChipCallable) != 0) return PTO_RUNTIME_ERR_INTERNAL;
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
        return PTO_RUNTIME_ERR_INTERNAL;

    const auto *image = static_cast<const ChipCallable *>(callable);
    const auto valid_name = [](const char *name, uint32_t length) {
        return length < CALLABLE_FUNC_NAME_MAX && name[length] == '\0' && std::memchr(name, '\0', length) == nullptr;
    };
    if (!valid_name(image->func_name_, image->func_name_len_) ||
        !valid_name(image->config_name_, image->config_name_len_))
        return PTO_RUNTIME_ERR_INTERNAL;
    const size_t storage_size = callable_size - offsetof(ChipCallable, storage_);
    size_t used = image->binary_size_;
    constexpr size_t max_children = sizeof(image->child_offsets_) / sizeof(image->child_offsets_[0]);
    if (used > storage_size || image->child_count_ < 0 || static_cast<size_t>(image->child_count_) > max_children)
        return PTO_RUNTIME_ERR_INTERNAL;
    for (int32_t i = 0; i < image->child_count_; ++i) {
        const size_t offset = image->child_offsets_[i];
        // Canonical child packing starts at the next aligned byte after
        // the preceding binary; subtraction precedes every span read.
        const size_t padding = (CALLABLE_ALIGN - used % CALLABLE_ALIGN) % CALLABLE_ALIGN;
        if (padding > storage_size - used || offset != used + padding ||
            CoreCallable::binary_data_offset() > storage_size - offset)
            return PTO_RUNTIME_ERR_INTERNAL;
        const auto *child = reinterpret_cast<const CoreCallable *>(image->storage_ + offset);
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS) return PTO_RUNTIME_ERR_INTERNAL;
        const size_t binary_offset = offset + CoreCallable::binary_data_offset();
        if (child->binary_size_ > storage_size - binary_offset) return PTO_RUNTIME_ERR_INTERNAL;
        used = binary_offset + child->binary_size_;
    }
    if (used != storage_size) return PTO_RUNTIME_ERR_INTERNAL;
    return 0;
}

inline int validate_kernel_launch_args(
    const void *ctx, SimplerCallableHandle handle, const void *args, const void *caller_stream
) {
    if (ctx == nullptr || args == nullptr || caller_stream == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (handle.callable_id < 0 || handle.generation == 0) return PTO_RUNTIME_ERR_INTERNAL;
    if (handle.callable_id >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED;
    return 0;
}
