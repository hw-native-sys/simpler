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

#include <cstdint>
#include <type_traits>

constexpr uint32_t CORE_ASYNC_CAPABILITY_VERSION = 1;
constexpr uint32_t CORE_ASYNC_MAX_CONDITIONS = 64;
constexpr uint32_t CORE_ASYNC_CONDITION_COUNTER = 1;
constexpr uint32_t CORE_ASYNC_CONDITION_SDMA_POST_DONE = 2;
constexpr uint32_t CORE_ASYNC_CONDITION_TYPES = CORE_ASYNC_CONDITION_COUNTER | CORE_ASYNC_CONDITION_SDMA_POST_DONE;

enum class CoreAsyncMode : uint32_t { UNKNOWN = 0, SYNCHRONOUS = 1, DEFERRED = 2 };

// This descriptor is part of the callable image and its identity. UNKNOWN is
// an unqualified binary, never evidence that a kernel completes synchronously.
struct CoreAsyncCapability {
    uint32_t version{CORE_ASYNC_CAPABILITY_VERSION};
    CoreAsyncMode mode{CoreAsyncMode::UNKNOWN};
    uint32_t max_conditions{0};
    uint32_t condition_types{0};
};

inline constexpr bool core_async_capability_valid(const CoreAsyncCapability &capability) {
    if (capability.version != CORE_ASYNC_CAPABILITY_VERSION) return false;
    switch (capability.mode) {
    case CoreAsyncMode::UNKNOWN:
    case CoreAsyncMode::SYNCHRONOUS:
        return capability.max_conditions == 0 && capability.condition_types == 0;
    case CoreAsyncMode::DEFERRED:
        return capability.max_conditions >= 1 && capability.max_conditions <= CORE_ASYNC_MAX_CONDITIONS &&
               capability.condition_types != 0 && (capability.condition_types & ~CORE_ASYNC_CONDITION_TYPES) == 0;
    }
    return false;
}

static_assert(sizeof(CoreAsyncCapability) == 16, "CoreAsyncCapability wire ABI size changed");
static_assert(
    std::is_trivially_copyable_v<CoreAsyncCapability> && std::is_standard_layout_v<CoreAsyncCapability>,
    "CoreAsyncCapability must remain a memcpy-able wire descriptor"
);
