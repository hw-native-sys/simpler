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

#include "callable.h"
#include "callable_protocol.h"
#include "runtime_c_api.h"

/**
 * Structural argument validation shared by every host-runtime component's
 * kernel-mode entries. Both platform c_api implementations (onboard and sim)
 * route through these checks, so a stub and a real implementation accept and
 * reject exactly the same arguments — the same parity rule the shared phase
 * machine follows. Each function returns 0 or PTO_RUNTIME_ERR_INTERNAL and
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
    const void *ctx, int32_t callable_id, const void *callable, size_t callable_size, const void *caller_stream
) {
    if (ctx == nullptr || callable == nullptr || caller_stream == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_INTERNAL;
    if (callable_size < sizeof(ChipCallable)) return PTO_RUNTIME_ERR_INTERNAL;
    /* ChipCallable's storage_ is CALLABLE_CHILD_ALIGN-aligned relative to the
       header, so a misaligned image puts every child at a misaligned address. */
    if (reinterpret_cast<uintptr_t>(callable) % alignof(ChipCallable) != 0) return PTO_RUNTIME_ERR_INTERNAL;
    return 0;
}

inline int
validate_kernel_launch_args(const void *ctx, int32_t callable_id, const void *args, const void *caller_stream) {
    if (ctx == nullptr || args == nullptr || caller_stream == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_INTERNAL;
    return 0;
}
