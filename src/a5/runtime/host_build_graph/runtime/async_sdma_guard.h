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

#include "common/platform_config.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

inline constexpr uint64_t ASYNC_SDMA_GUARD_MAGIC = UINT64_C(0x53444d4147554152);
inline constexpr uint64_t ASYNC_SDMA_GUARD_VERSION = 1;
inline constexpr uint32_t ASYNC_SDMA_CHANNELS = 48;

// This immutable header and the atomic-only guard rows occupy separate lines.
// One run owns the allocation. A row guards a physical SDMA queue; independent
// workspaces may conservatively share the same short submission critical section.
struct alignas(128) AsyncSdmaGuardContext {
    uint64_t magic{ASYNC_SDMA_GUARD_MAGIC};
    uint64_t version{ASYNC_SDMA_GUARD_VERSION};
    uint64_t run_epoch{0};
    uint64_t guard_count{0};
    uint64_t bytes{0};
    uint8_t reserved[88]{};
};

struct alignas(128) AsyncSdmaSubmissionGuard {
    volatile uint64_t word{0};
};

static_assert(sizeof(AsyncSdmaGuardContext) == 128);
static_assert(sizeof(AsyncSdmaSubmissionGuard) == 128);
static_assert(std::is_trivially_copyable_v<AsyncSdmaGuardContext>);
static_assert(std::is_standard_layout_v<AsyncSdmaGuardContext>);
static_assert(std::is_trivially_copyable_v<AsyncSdmaSubmissionGuard>);
static_assert(std::is_standard_layout_v<AsyncSdmaSubmissionGuard>);

inline __aicore__ bool async_sdma_guard_context_valid(__gm__ const AsyncSdmaGuardContext *context) {
    const uint64_t address = reinterpret_cast<uint64_t>(context);
    return context != nullptr && address % 128 == 0 && context->magic == ASYNC_SDMA_GUARD_MAGIC &&
           context->version == ASYNC_SDMA_GUARD_VERSION && context->run_epoch != 0 && context->guard_count != 0 &&
           context->guard_count <= ASYNC_SDMA_CHANNELS && context->bytes == (context->guard_count + 1) * 128 &&
           context->bytes <= UINT64_MAX - address;
}

inline __aicore__ bool async_sdma_guard_region(
    uint64_t base, uint64_t bytes, uint64_t epoch, uint64_t offset, uint64_t count, uint64_t *address
) {
    if (address == nullptr) return false;
    *address = 0;
    if (count == 0) return offset == 0;
    if (base == 0 || bytes > UINT64_MAX - base || offset == 0 || offset % 128 != 0 || count > ASYNC_SDMA_CHANNELS ||
        offset > bytes || (count + 1) * 128 > bytes - offset)
        return false;
    const auto *context = reinterpret_cast<__gm__ const AsyncSdmaGuardContext *>(base + offset);
    if (!async_sdma_guard_context_valid(context) || context->run_epoch != epoch || context->guard_count != count)
        return false;
    *address = base + offset;
    return true;
}
