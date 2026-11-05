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
#include <memory>
#include <new>
#include "runtime_c_api.h"

struct AlignedImageStorage {
    std::unique_ptr<std::byte[]> storage;
    size_t capacity{0};

    int acquire(size_t bytes, size_t alignment, void **out) {
        if (out == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        *out = nullptr;
        if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 || bytes > SIZE_MAX - (alignment - 1))
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        const size_t needed = bytes + alignment - 1;
        if (capacity < needed) {
            std::unique_ptr<std::byte[]> next(new (std::nothrow) std::byte[needed]);
            if (!next) return PTO_RUNTIME_ERR_INTERNAL;
            storage = std::move(next);
            capacity = needed;
        }
        const auto raw = reinterpret_cast<uintptr_t>(storage.get());
        *out = reinterpret_cast<void *>((raw + alignment - 1) & ~static_cast<uintptr_t>(alignment - 1));
        return 0;
    }
};
