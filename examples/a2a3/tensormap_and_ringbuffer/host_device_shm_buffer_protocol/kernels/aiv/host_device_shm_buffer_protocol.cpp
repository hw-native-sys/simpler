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
static constexpr uint32_t kShmMagic = 0x48534442U;
static constexpr uint32_t kShmVersion = 1U;

struct alignas(64) ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t status;
    uint32_t reserved0;
    uint64_t seq;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t checksum;
    uint64_t reserved[2];
};

struct ShmBufferView {
    __gm__ ShmHeader *header;
    __gm__ uint8_t *input;
    __gm__ uint8_t *output;
};

static_assert(sizeof(ShmHeader) == 64);
static_assert(offsetof(ShmHeader, seq) == 16);
static_assert(offsetof(ShmHeader, checksum) == 40);

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

static inline __aicore__ bool wait_signal_l2(__gm__ uint8_t *signal_base, uint32_t signal_id, uint32_t seq) {
    auto *signal = signal_slot(signal_base, signal_id);
    for (uint32_t i = 0; i < kMaxPollIters; ++i) {
        invalidate_range(signal, kCacheLineBytes);
        if (*signal >= seq) {
            return true;
        }
    }
    return false;
}

static inline __aicore__ void notify_signal_l2(__gm__ uint8_t *signal_base, uint32_t signal_id, uint32_t seq) {
    auto *signal = signal_slot(signal_base, signal_id);
    *signal = seq;
    flush_range(signal, kCacheLineBytes);
}

static inline __aicore__ uint64_t checksum64_update(uint64_t acc, uint8_t value, uint64_t index) {
    acc ^= (static_cast<uint64_t>(value) + ((index + 1U) * 0x100000001B3ULL));
    acc = ((acc << 7U) | (acc >> 57U));
    return acc * 0xD6E8FEB86659FD93ULL;
}

static inline __aicore__ bool shm_buffer_recv_l2(
    __gm__ uint8_t *data, __gm__ uint8_t *signal_base, uint32_t seq, uint32_t payload_bytes, ShmBufferView *view
) {
    if (!wait_signal_l2(signal_base, 0, seq)) {
        return false;
    }
    view->header = reinterpret_cast<__gm__ ShmHeader *>(data);
    view->input = data + sizeof(ShmHeader);
    view->output = view->input + payload_bytes;
    invalidate_range(view->header, sizeof(ShmHeader));
    invalidate_range(view->input, payload_bytes);
    return true;
}

static inline __aicore__ void shm_buffer_send_l2(
    const ShmBufferView &view, __gm__ uint8_t *signal_base, uint32_t seq, uint32_t status, uint64_t output_bytes,
    uint64_t checksum
) {
    if (output_bytes > 0U) {
        flush_range(view.output, output_bytes);
    }
    view.header->status = status;
    view.header->output_bytes = output_bytes;
    view.header->checksum = checksum;
    flush_range(view.header, sizeof(ShmHeader));
    notify_signal_l2(signal_base, 1, seq);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *data = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[0]));
    auto *signal_base = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[1]));
    uint32_t seq = static_cast<uint32_t>(args[2]);
    uint32_t payload_bytes = static_cast<uint32_t>(args[3]);

    ShmBufferView buffer{};
    if (!shm_buffer_recv_l2(data, signal_base, seq, payload_bytes, &buffer)) {
        return;
    }

    bool valid = buffer.header->magic == kShmMagic && buffer.header->version == kShmVersion &&
                 buffer.header->seq == seq && buffer.header->input_bytes == payload_bytes;
    if (!valid) {
        shm_buffer_send_l2(buffer, signal_base, seq, 1U, 0U, 0U);
        return;
    }

    uint64_t checksum = 0x9E3779B185EBCA87ULL;
    for (uint32_t i = 0; i < payload_bytes; ++i) {
        uint8_t mask = static_cast<uint8_t>(seq + i * 13U);
        uint8_t value = static_cast<uint8_t>(buffer.input[i] ^ mask);
        buffer.output[i] = value;
        checksum = checksum64_update(checksum, value, i);
    }

    shm_buffer_send_l2(buffer, signal_base, seq, 0U, payload_bytes, checksum);
}
