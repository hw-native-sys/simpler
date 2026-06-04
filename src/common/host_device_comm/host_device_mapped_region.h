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

#ifndef SRC_COMMON_WORKER_HOST_DEVICE_MAPPED_REGION_H_
#define SRC_COMMON_WORKER_HOST_DEVICE_MAPPED_REGION_H_

#include <stddef.h>
#include <stdint.h>

#include "worker/pto_runtime_c_api.h"

static constexpr uint32_t HDMR_MAGIC = 0x48444D52U;
static constexpr uint32_t HDMR_VERSION = 1;

struct alignas(64) HostDeviceMappedRegionHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t signal_count;
    uint64_t signal_offset;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t total_bytes;
    uint64_t reserved[2];
};

struct alignas(64) HostDeviceMappedRegionSignalSlot {
    volatile uint32_t value;
    uint32_t reserved0;
    uint64_t reserved[7];
};

struct HostDeviceMappedRegionPlatform {
    void *resource;
    uint64_t device_id;
    void *cache_ops_cookie;
    int (*flush_host_range)(HostDeviceMappedRegionPlatform *platform, void *host_ptr, uint64_t bytes);
    int (*invalidate_host_range)(HostDeviceMappedRegionPlatform *platform, void *host_ptr, uint64_t bytes);
    void (*release)(HostDeviceMappedRegionPlatform *platform);
};

using HostDeviceMappedRegionAllocateFn = int (*)(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
);

int host_device_mapped_region_compute_total_bytes(
    const HostDeviceMappedRegionConfig *cfg, uint64_t *signal_offset, uint64_t *data_offset, uint64_t *total_bytes
);

int host_device_mapped_region_open_common(
    DeviceContextHandle ctx, const HostDeviceMappedRegionConfig *cfg, HostDeviceMappedRegionHandle *out_region,
    HostDeviceMappedRegionAllocateFn allocate
);

int host_device_mapped_region_close_common(DeviceContextHandle ctx, HostDeviceMappedRegionHandle region);

void host_device_mapped_region_close_all_common(DeviceContextHandle ctx);

int host_device_mapped_region_info_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, HostDeviceMappedRegionInfo *info
);

int host_device_mapped_region_datacopy_h2region_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, const void *src, size_t nbytes
);

int host_device_mapped_region_datacopy_region2h_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint64_t offset, void *dst, size_t nbytes
);

int host_device_mapped_region_notify_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t value
);

int host_device_mapped_region_wait_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle region, uint32_t signal_id, uint32_t target,
    uint32_t timeout_us
);

#endif  // SRC_COMMON_WORKER_HOST_DEVICE_MAPPED_REGION_H_
