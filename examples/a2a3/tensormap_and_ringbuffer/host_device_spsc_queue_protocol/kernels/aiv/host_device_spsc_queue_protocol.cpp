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
static constexpr uint32_t kLaneDepth = 16U;
static constexpr uint32_t kLaneMask = kLaneDepth - 1U;
static constexpr uint32_t kMaxInlineBytes = 256U;
static constexpr uint32_t kCpuToL2 = 0U;
static constexpr uint32_t kL2ToCpu = 1U;

struct alignas(64) HostDeviceChannelHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t lane_count_cpu_to_l2;
    uint32_t lane_count_l2_to_cpu;
    uint32_t lane_depth;
    uint32_t max_message_bytes;
    uint32_t reserved0;
    uint64_t control_bytes;
    uint64_t fatal_status;
    uint64_t reserved[2];
};

struct alignas(64) HostDeviceLaneHeader {
    volatile uint32_t head;
    volatile uint32_t tail;
    uint32_t depth;
    uint32_t depth_mask;
    uint64_t dropped_count;
    uint64_t blocked_count;
    uint64_t reserved[4];
};

struct alignas(64) HostDeviceDesc {
    uint32_t flags;
    uint32_t payload_bytes;
    uint64_t seq;
    uint64_t correlation_id;
    uint32_t route;
    uint32_t reserved0;
    uint8_t reserved1[32];
    uint8_t inline_data[kMaxInlineBytes];
};

static_assert(sizeof(HostDeviceChannelHeader) == 64);
static_assert(sizeof(HostDeviceLaneHeader) == 64);
static_assert(sizeof(HostDeviceDesc) == 320);
static_assert(offsetof(HostDeviceDesc, inline_data) == 64);

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

static inline __aicore__ __gm__ HostDeviceLaneHeader *lane(__gm__ uint8_t *data, uint32_t direction) {
    auto *channel = reinterpret_cast<__gm__ HostDeviceChannelHeader *>(data);
    auto *cpu_lane = reinterpret_cast<__gm__ HostDeviceLaneHeader *>(channel + 1);
    if (direction == kCpuToL2) {
        return cpu_lane;
    }
    return reinterpret_cast<__gm__ HostDeviceLaneHeader *>(
        reinterpret_cast<__gm__ uint8_t *>(cpu_lane) + sizeof(HostDeviceLaneHeader) +
        kLaneDepth * sizeof(HostDeviceDesc)
    );
}

static inline __aicore__ __gm__ HostDeviceDesc *desc_base(__gm__ uint8_t *data, uint32_t direction) {
    return reinterpret_cast<__gm__ HostDeviceDesc *>(
        reinterpret_cast<__gm__ uint8_t *>(lane(data, direction)) + sizeof(HostDeviceLaneHeader)
    );
}

static inline __aicore__ void read_desc_l2(__gm__ HostDeviceDesc *slot, HostDeviceDesc *out) {
    out->flags = slot->flags;
    out->payload_bytes = slot->payload_bytes;
    out->seq = slot->seq;
    out->correlation_id = slot->correlation_id;
    out->route = slot->route;
    out->reserved0 = slot->reserved0;
    for (uint32_t i = 0; i < kMaxInlineBytes; ++i) {
        out->inline_data[i] = slot->inline_data[i];
    }
}

static inline __aicore__ void write_desc_l2(__gm__ HostDeviceDesc *slot, const HostDeviceDesc &msg) {
    slot->flags = msg.flags;
    slot->payload_bytes = msg.payload_bytes;
    slot->seq = msg.seq;
    slot->correlation_id = msg.correlation_id;
    slot->route = msg.route;
    slot->reserved0 = msg.reserved0;
    for (uint32_t i = 0; i < kMaxInlineBytes; ++i) {
        slot->inline_data[i] = msg.inline_data[i];
    }
}

static inline __aicore__ bool
channel_recv_l2(__gm__ uint8_t *data, __gm__ uint8_t *signal_base, uint32_t seq, HostDeviceDesc *out) {
    if (!wait_signal_l2(signal_base, 0, seq)) {
        return false;
    }
    auto *input = lane(data, kCpuToL2);
    invalidate_range(input, sizeof(HostDeviceLaneHeader));
    uint32_t head = input->head;
    uint32_t tail = input->tail;
    if (head == tail) {
        return false;
    }
    auto *slot = desc_base(data, kCpuToL2) + (head & kLaneMask);
    invalidate_range(slot, sizeof(HostDeviceDesc));
    read_desc_l2(slot, out);
    input->head = head + 1U;
    flush_range(input, sizeof(uint32_t));
    return true;
}

static inline __aicore__ bool
channel_send_l2(__gm__ uint8_t *data, __gm__ uint8_t *signal_base, const HostDeviceDesc &msg) {
    auto *output = lane(data, kL2ToCpu);
    invalidate_range(output, sizeof(HostDeviceLaneHeader));
    uint32_t head = output->head;
    uint32_t tail = output->tail;
    if (tail - head >= kLaneDepth) {
        output->blocked_count += 1U;
        flush_range(output, sizeof(HostDeviceLaneHeader));
        return false;
    }
    auto *slot = desc_base(data, kL2ToCpu) + (tail & kLaneMask);
    write_desc_l2(slot, msg);
    flush_range(slot, sizeof(HostDeviceDesc));
    output->tail = tail + 1U;
    flush_range(
        reinterpret_cast<volatile __gm__ uint8_t *>(output) + offsetof(HostDeviceLaneHeader, tail), sizeof(uint32_t)
    );
    notify_signal_l2(signal_base, 1, static_cast<uint32_t>(msg.seq));
    return true;
}

static inline __aicore__ void transform_message(const HostDeviceDesc &in, HostDeviceDesc *out) {
    out->flags = in.flags;
    out->seq = in.seq;
    out->correlation_id = in.correlation_id;
    out->route = in.route ^ 0x80000000U;
    out->reserved0 = 0U;
    uint32_t nbytes = in.payload_bytes > kMaxInlineBytes ? kMaxInlineBytes : in.payload_bytes;
    out->payload_bytes = nbytes;
    for (uint32_t i = 0; i < sizeof(out->reserved1); ++i) {
        out->reserved1[i] = 0U;
    }
    for (uint32_t i = 0; i < nbytes; ++i) {
        uint8_t mask = static_cast<uint8_t>(in.seq + i * 7U);
        out->inline_data[i] = static_cast<uint8_t>(in.inline_data[i] ^ mask);
    }
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *data = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[0]));
    auto *signal_base = reinterpret_cast<__gm__ uint8_t *>(static_cast<uint64_t>(args[1]));
    uint32_t seq = static_cast<uint32_t>(args[2]);
    uint32_t max_messages_to_drain = static_cast<uint32_t>(args[3]);

    for (uint32_t i = 0; i < max_messages_to_drain; ++i) {
        HostDeviceDesc msg{};
        if (!channel_recv_l2(data, signal_base, seq, &msg)) {
            break;
        }
        HostDeviceDesc response{};
        transform_message(msg, &response);
        if (!channel_send_l2(data, signal_base, response)) {
            break;
        }
    }
}
