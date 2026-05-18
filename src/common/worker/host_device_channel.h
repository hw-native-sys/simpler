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

#ifndef SRC_COMMON_WORKER_HOST_DEVICE_CHANNEL_H_
#define SRC_COMMON_WORKER_HOST_DEVICE_CHANNEL_H_

#include <stddef.h>
#include <stdint.h>

static constexpr uint32_t HDCH_MAGIC = 0x48444348U;  // "HDCH"
static constexpr uint32_t HDCH_VERSION = 1;
static constexpr uint32_t HDCH_MAX_INLINE_BYTES = 256;

static constexpr int HDCH_OK = 0;
static constexpr int HDCH_ERR_WOULD_BLOCK = -11;
static constexpr int HDCH_ERR_INVALID = -22;
static constexpr int HDCH_ERR_MSG_TOO_LARGE = -75;
static constexpr int HDCH_ERR_BACKEND = -5;

#include "pto_runtime_c_api.h"

struct alignas(64) HostDeviceDesc {
    uint32_t flags;
    uint32_t payload_bytes;
    uint64_t seq;
    uint64_t correlation_id;
    uint32_t route;
    uint32_t reserved0;
    uint8_t inline_data[HDCH_MAX_INLINE_BYTES];
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
    uint64_t reserved[4];
};

struct HostDeviceChannel {
    void *device_base;
    void *host_base;
    size_t bytes;
    uint32_t cpu_tx_cursor;
    uint32_t l2_tx_cursor;
    uint32_t owns_host_allocation;
    void (*free_host_allocation)(void *);
};

size_t host_device_channel_required_bytes(const HostDeviceChannelConfig *cfg);
int host_device_channel_init_region(void *host_base, size_t bytes, const HostDeviceChannelConfig *cfg);
HostDeviceChannel *host_device_channel_wrap(
    void *device_base, void *host_base, size_t bytes, const HostDeviceChannelConfig *cfg, uint32_t owns_host_allocation,
    void (*free_host_allocation)(void *)
);
void host_device_channel_destroy(HostDeviceChannel *ch);

int host_device_channel_send_cpu(
    HostDeviceChannel *ch, uint32_t route, const void *data, size_t nbytes, uint64_t correlation_id,
    uint32_t timeout_us
);
int host_device_channel_recv_cpu(
    HostDeviceChannel *ch, void *dst, size_t dst_capacity, size_t *out_nbytes, uint64_t *out_correlation_id,
    uint32_t *out_route, uint32_t timeout_us
);

// Test/sim endpoint for the L2 side. V2 AICPU broker should use the same POD
// layout and publish/consume protocol from device code.
int host_device_channel_send_l2_for_test(
    HostDeviceChannel *ch, uint32_t route, const void *data, size_t nbytes, uint64_t correlation_id,
    uint32_t timeout_us
);
int host_device_channel_recv_l2_for_test(
    HostDeviceChannel *ch, void *dst, size_t dst_capacity, size_t *out_nbytes, uint64_t *out_correlation_id,
    uint32_t *out_route, uint32_t timeout_us
);

#endif  // SRC_COMMON_WORKER_HOST_DEVICE_CHANNEL_H_
