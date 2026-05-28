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

#include <cstdint>

#include <pto/pto-inst.hpp>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "pipe_sync.h"

static constexpr uint64_t kCacheLineBytes = 64;
static constexpr uint32_t kMaxPollIters = 1024U;

static inline __aicore__ void flush_range(volatile __gm__ void *addr, uint64_t size_bytes) {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(kCacheLineBytes) - 1u);
    uintptr_t end =
        (reinterpret_cast<uintptr_t>(addr) + size_bytes + kCacheLineBytes - 1u) & ~(uintptr_t(kCacheLineBytes) - 1u);
    for (uintptr_t p = start; p < end; p += kCacheLineBytes) {
        dcci((__gm__ int32_t *)p, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
#else
    (void)addr;
    (void)size_bytes;
    __asm__ __volatile__("" ::: "memory");
#endif
}

static inline __aicore__ void invalidate_range(volatile __gm__ void *addr, uint64_t size_bytes) {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(kCacheLineBytes) - 1u);
    uintptr_t end =
        (reinterpret_cast<uintptr_t>(addr) + size_bytes + kCacheLineBytes - 1u) & ~(uintptr_t(kCacheLineBytes) - 1u);
    for (uintptr_t p = start; p < end; p += kCacheLineBytes) {
        dcci((__gm__ int32_t *)p, SINGLE_CACHE_LINE);
    }
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
#else
    (void)addr;
    (void)size_bytes;
    __asm__ __volatile__("" ::: "memory");
#endif
}

static inline __aicore__ volatile __gm__ uint32_t *signal_slot(__gm__ uint8_t *signal_base, uint32_t signal_id) {
    return reinterpret_cast<volatile __gm__ uint32_t *>(signal_base + signal_id * kCacheLineBytes);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *data = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[0]));
    auto *signal_base = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[1]));
    auto *signal0 = signal_slot(signal_base, 0);
    auto *signal1 = signal_slot(signal_base, 1);
    uint32_t seq = static_cast<uint32_t>(args[2]);
    uint32_t nbytes = static_cast<uint32_t>(args[3]);

    bool observed = false;
    for (uint32_t i = 0; i < kMaxPollIters; ++i) {
        invalidate_range(signal0, kCacheLineBytes);
        if (*signal0 >= seq) {
            observed = true;
            break;
        }
    }
    if (!observed) {
        return;
    }

    invalidate_range(data, nbytes);
    for (uint32_t i = 0; i < nbytes; ++i) {
        uint8_t mask = static_cast<uint8_t>(seq + i * 3U);
        data[nbytes + i] = static_cast<uint8_t>(data[i] ^ mask);
    }
    flush_range(data + nbytes, nbytes);

    *signal1 = seq;
    flush_range(signal1, kCacheLineBytes);
}
