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
#include <cstddef>
#include <cstdint>

#include "aicpu/cache_maintenance.h"

namespace aicpu_cache_maintenance {

#if defined(__aarch64__)

void invalidate_range_impl(const void *addr, size_t size) {
    if (size == 0) {
        return;
    }
    const size_t kCacheLineSize = 64;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kCacheLineSize - 1);
    uintptr_t end = (reinterpret_cast<uintptr_t>(addr) + size + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    for (uintptr_t p = start; p < end; p += kCacheLineSize) {
        __asm__ __volatile__("dc civac, %0" ::"r"(p) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}

void flush_range_impl(const void *addr, size_t size) {
    if (size == 0) {
        return;
    }
    const size_t kCacheLineSize = 64;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kCacheLineSize - 1);
    uintptr_t end = (reinterpret_cast<uintptr_t>(addr) + size + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
    for (uintptr_t p = start; p < end; p += kCacheLineSize) {
        __asm__ __volatile__("dc cvac, %0" ::"r"(p) : "memory");
    }
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}

#else

// host_build_graph runs orchestration on the host CPU, which reaches device
// memory only through driver H2D DMA (cache-coherent on x86). The manual
// AICPU-side cache maintenance the aarch64 path performs has no host-side
// referent here, so both operations are inert.
void invalidate_range_impl(const void * /* addr */, size_t /* size */) {}

void flush_range_impl(const void * /* addr */, size_t /* size */) {}

#endif

}  // namespace aicpu_cache_maintenance
