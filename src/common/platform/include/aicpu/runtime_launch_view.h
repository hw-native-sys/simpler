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

#include "runtime.h"

// RTS argument storage guarantees no Runtime cache-line alignment. Each AICPU
// invocation keeps its descriptor view until its own device entry returns.
class RuntimeLaunchView {
public:
    RuntimeLaunchView() = default;
    RuntimeLaunchView(const RuntimeLaunchView &) = delete;
    RuntimeLaunchView &operator=(const RuntimeLaunchView &) = delete;
    Runtime *bind(Runtime *source, uint32_t bytes) {
        if (source == nullptr) return nullptr;
        if (bytes == 0) return source;
        if (bytes != RUNTIME_LAUNCH_DESCRIPTOR_BYTES) return nullptr;
        constexpr uintptr_t alignment = alignof(DeviceRuntimeLaunchDesc);
        const uintptr_t address = (reinterpret_cast<uintptr_t>(storage_) + alignment - 1) & ~(alignment - 1);
        void *aligned = reinterpret_cast<void *>(address);
        std::memcpy(aligned, source, bytes);
        return static_cast<Runtime *>(aligned);
    }

private:
    unsigned char storage_[RUNTIME_LAUNCH_DESCRIPTOR_BYTES + alignof(DeviceRuntimeLaunchDesc) - 1];
};
