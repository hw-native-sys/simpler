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

#include "host_device_channel.h"

#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <new>
#include <thread>

namespace {

bool is_power_of_two(uint32_t v) { return v != 0 && (v & (v - 1U)) == 0; }

size_t align_up(size_t v, size_t alignment) { return (v + alignment - 1U) & ~(alignment - 1U); }

bool valid_cfg(const HostDeviceChannelConfig *cfg) {
    return cfg != nullptr && cfg->lane_count_cpu_to_l2 > 0 && cfg->lane_count_l2_to_cpu > 0 &&
           is_power_of_two(cfg->lane_depth) && cfg->max_message_bytes > 0 &&
           cfg->max_message_bytes <= HDCH_MAX_INLINE_BYTES;
}

HostDeviceLaneHeader *first_lane(HostDeviceChannelHeader *hdr) {
    return reinterpret_cast<HostDeviceLaneHeader *>(reinterpret_cast<uint8_t *>(hdr) + sizeof(HostDeviceChannelHeader));
}

HostDeviceLaneHeader *lane_at(HostDeviceLaneHeader *base, uint32_t lane_depth, uint32_t index) {
    size_t stride = sizeof(HostDeviceLaneHeader) + static_cast<size_t>(lane_depth) * sizeof(HostDeviceDesc);
    return reinterpret_cast<HostDeviceLaneHeader *>(reinterpret_cast<uint8_t *>(base) + stride * index);
}

HostDeviceDesc *lane_slots(HostDeviceLaneHeader *lane) {
    return reinterpret_cast<HostDeviceDesc *>(reinterpret_cast<uint8_t *>(lane) + sizeof(HostDeviceLaneHeader));
}

HostDeviceLaneHeader *cpu_to_l2_lanes(HostDeviceChannel *ch) {
    return first_lane(reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base));
}

HostDeviceLaneHeader *l2_to_cpu_lanes(HostDeviceChannel *ch) {
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base);
    return lane_at(cpu_to_l2_lanes(ch), hdr->lane_depth, hdr->lane_count_cpu_to_l2);
}

uint32_t load_u32(const volatile uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

void store_u32(volatile uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

int send_one(
    HostDeviceLaneHeader *lanes, uint32_t lane_count, uint32_t lane_depth, uint32_t max_message_bytes, uint32_t *cursor,
    uint32_t route, const void *data, size_t nbytes, uint64_t correlation_id, uint32_t timeout_us
) {
    if (lanes == nullptr || cursor == nullptr || (data == nullptr && nbytes != 0)) return HDCH_ERR_INVALID;
    if (nbytes > max_message_bytes) return HDCH_ERR_MSG_TOO_LARGE;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        for (uint32_t i = 0; i < lane_count; ++i) {
            uint32_t lane_index = (*cursor + i) % lane_count;
            HostDeviceLaneHeader *lane = lane_at(lanes, lane_depth, lane_index);
            uint32_t head = load_u32(&lane->head);
            uint32_t tail = load_u32(&lane->tail);
            if ((tail - head) >= lane_depth) {
                continue;
            }

            HostDeviceDesc *slot = lane_slots(lane) + (tail & lane->depth_mask);
            slot->flags = 0;
            slot->payload_bytes = static_cast<uint32_t>(nbytes);
            slot->seq = tail;
            slot->correlation_id = correlation_id;
            slot->route = route;
            slot->reserved0 = 0;
            if (nbytes != 0) {
                memcpy(slot->inline_data, data, nbytes);
            }
            store_u32(&lane->tail, tail + 1U);
            *cursor = (lane_index + 1U) % lane_count;
            return HDCH_OK;
        }

        if (timeout_us == 0) return HDCH_ERR_WOULD_BLOCK;
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_us) return HDCH_ERR_WOULD_BLOCK;
        std::this_thread::yield();
    }
}

int recv_one(
    HostDeviceLaneHeader *lanes, uint32_t lane_count, uint32_t lane_depth, uint32_t *cursor, void *dst,
    size_t dst_capacity, size_t *out_nbytes, uint64_t *out_correlation_id, uint32_t *out_route, uint32_t timeout_us
) {
    if (lanes == nullptr || cursor == nullptr || dst == nullptr || out_nbytes == nullptr ||
        out_correlation_id == nullptr || out_route == nullptr) {
        return HDCH_ERR_INVALID;
    }

    auto start = std::chrono::steady_clock::now();
    while (true) {
        for (uint32_t i = 0; i < lane_count; ++i) {
            uint32_t lane_index = (*cursor + i) % lane_count;
            HostDeviceLaneHeader *lane = lane_at(lanes, lane_depth, lane_index);
            uint32_t head = load_u32(&lane->head);
            uint32_t tail = load_u32(&lane->tail);
            if (head == tail) {
                continue;
            }

            HostDeviceDesc *slot = lane_slots(lane) + (head & lane->depth_mask);
            if (slot->payload_bytes > dst_capacity) return HDCH_ERR_MSG_TOO_LARGE;
            if (slot->payload_bytes != 0) {
                memcpy(dst, slot->inline_data, slot->payload_bytes);
            }
            *out_nbytes = slot->payload_bytes;
            *out_correlation_id = slot->correlation_id;
            *out_route = slot->route;
            store_u32(&lane->head, head + 1U);
            *cursor = (lane_index + 1U) % lane_count;
            return HDCH_OK;
        }

        if (timeout_us == 0) return HDCH_ERR_WOULD_BLOCK;
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_us) return HDCH_ERR_WOULD_BLOCK;
        std::this_thread::yield();
    }
}

}  // namespace

size_t host_device_channel_required_bytes(const HostDeviceChannelConfig *cfg) {
    if (!valid_cfg(cfg)) return 0;
    size_t lane_bytes = sizeof(HostDeviceLaneHeader) + static_cast<size_t>(cfg->lane_depth) * sizeof(HostDeviceDesc);
    size_t total_lanes = static_cast<size_t>(cfg->lane_count_cpu_to_l2) + cfg->lane_count_l2_to_cpu;
    return align_up(sizeof(HostDeviceChannelHeader) + lane_bytes * total_lanes, 64);
}

int host_device_channel_init_region(void *host_base, size_t bytes, const HostDeviceChannelConfig *cfg) {
    if (host_base == nullptr || !valid_cfg(cfg)) return HDCH_ERR_INVALID;
    size_t required = host_device_channel_required_bytes(cfg);
    if (required == 0 || bytes < required) return HDCH_ERR_INVALID;

    memset(host_base, 0, required);
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(host_base);
    hdr->magic = HDCH_MAGIC;
    hdr->version = HDCH_VERSION;
    hdr->flags = cfg->flags;
    hdr->lane_count_cpu_to_l2 = cfg->lane_count_cpu_to_l2;
    hdr->lane_count_l2_to_cpu = cfg->lane_count_l2_to_cpu;
    hdr->lane_depth = cfg->lane_depth;
    hdr->max_message_bytes = cfg->max_message_bytes;
    hdr->control_bytes = required;

    HostDeviceLaneHeader *lane = first_lane(hdr);
    uint32_t lane_count = cfg->lane_count_cpu_to_l2 + cfg->lane_count_l2_to_cpu;
    for (uint32_t i = 0; i < lane_count; ++i) {
        HostDeviceLaneHeader *cur = lane_at(lane, cfg->lane_depth, i);
        cur->depth = cfg->lane_depth;
        cur->depth_mask = cfg->lane_depth - 1U;
    }
    return HDCH_OK;
}

HostDeviceChannel *host_device_channel_wrap(
    void *device_base, void *host_base, size_t bytes, const HostDeviceChannelConfig *cfg, uint32_t owns_host_allocation,
    void (*free_host_allocation)(void *)
) {
    if (device_base == nullptr || host_base == nullptr || !valid_cfg(cfg)) return nullptr;
    int rc = host_device_channel_init_region(host_base, bytes, cfg);
    if (rc != HDCH_OK) return nullptr;
    HostDeviceChannel *ch = new (std::nothrow) HostDeviceChannel();
    if (ch == nullptr) return nullptr;
    ch->device_base = device_base;
    ch->host_base = host_base;
    ch->bytes = host_device_channel_required_bytes(cfg);
    ch->cpu_tx_cursor = 0;
    ch->l2_tx_cursor = 0;
    ch->owns_host_allocation = owns_host_allocation;
    ch->free_host_allocation = free_host_allocation;
    return ch;
}

void host_device_channel_destroy(HostDeviceChannel *ch) {
    if (ch == nullptr) return;
    if (ch->owns_host_allocation && ch->free_host_allocation != nullptr && ch->host_base != nullptr) {
        ch->free_host_allocation(ch->host_base);
    }
    delete ch;
}

int host_device_channel_send_cpu(
    HostDeviceChannel *ch, uint32_t route, const void *data, size_t nbytes, uint64_t correlation_id,
    uint32_t timeout_us
) {
    if (ch == nullptr || ch->host_base == nullptr) return HDCH_ERR_INVALID;
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base);
    if (hdr->magic != HDCH_MAGIC || hdr->version != HDCH_VERSION) return HDCH_ERR_INVALID;
    return send_one(
        cpu_to_l2_lanes(ch), hdr->lane_count_cpu_to_l2, hdr->lane_depth, hdr->max_message_bytes, &ch->cpu_tx_cursor,
        route, data, nbytes, correlation_id, timeout_us
    );
}

int host_device_channel_recv_cpu(
    HostDeviceChannel *ch, void *dst, size_t dst_capacity, size_t *out_nbytes, uint64_t *out_correlation_id,
    uint32_t *out_route, uint32_t timeout_us
) {
    if (ch == nullptr || ch->host_base == nullptr) return HDCH_ERR_INVALID;
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base);
    if (hdr->magic != HDCH_MAGIC || hdr->version != HDCH_VERSION) return HDCH_ERR_INVALID;
    return recv_one(
        l2_to_cpu_lanes(ch), hdr->lane_count_l2_to_cpu, hdr->lane_depth, &ch->l2_tx_cursor, dst, dst_capacity,
        out_nbytes, out_correlation_id, out_route, timeout_us
    );
}

int host_device_channel_send_l2_for_test(
    HostDeviceChannel *ch, uint32_t route, const void *data, size_t nbytes, uint64_t correlation_id,
    uint32_t timeout_us
) {
    if (ch == nullptr || ch->host_base == nullptr) return HDCH_ERR_INVALID;
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base);
    if (hdr->magic != HDCH_MAGIC || hdr->version != HDCH_VERSION) return HDCH_ERR_INVALID;
    return send_one(
        l2_to_cpu_lanes(ch), hdr->lane_count_l2_to_cpu, hdr->lane_depth, hdr->max_message_bytes, &ch->l2_tx_cursor,
        route, data, nbytes, correlation_id, timeout_us
    );
}

int host_device_channel_recv_l2_for_test(
    HostDeviceChannel *ch, void *dst, size_t dst_capacity, size_t *out_nbytes, uint64_t *out_correlation_id,
    uint32_t *out_route, uint32_t timeout_us
) {
    if (ch == nullptr || ch->host_base == nullptr) return HDCH_ERR_INVALID;
    auto *hdr = reinterpret_cast<HostDeviceChannelHeader *>(ch->host_base);
    if (hdr->magic != HDCH_MAGIC || hdr->version != HDCH_VERSION) return HDCH_ERR_INVALID;
    return recv_one(
        cpu_to_l2_lanes(ch), hdr->lane_count_cpu_to_l2, hdr->lane_depth, &ch->cpu_tx_cursor, dst, dst_capacity,
        out_nbytes, out_correlation_id, out_route, timeout_us
    );
}
