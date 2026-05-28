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

#ifndef SRC_A2A3_PLATFORM_ONBOARD_HOST_HOST_DEVICE_MAPPED_REGION_ONBOARD_H_
#define SRC_A2A3_PLATFORM_ONBOARD_HOST_HOST_DEVICE_MAPPED_REGION_ONBOARD_H_

#include "host_device_comm/host_device_mapped_region.h"

/**
 * Allocate a host/device mapped region on A2/A3 onboard platforms.
 *
 * @param ctx          DeviceRunner context used for device allocation and host registration.
 * @param total_bytes  Total allocation size in bytes. Must be non-zero.
 * @param platform     Output platform callbacks and release state.
 * @param host_base    Output host-visible mapped base pointer.
 * @param device_base  Output device-visible base pointer.
 * @return 0 on success, negative error code on invalid context, allocation, or registration failure.
 *
 * The returned device allocation and host registration are owned by
 * platform->release. Cache callbacks in platform provide host/device
 * visibility maintenance for the mapped range.
 */
int a2a3_onboard_host_device_mapped_region_allocate(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
);

#endif  // SRC_A2A3_PLATFORM_ONBOARD_HOST_HOST_DEVICE_MAPPED_REGION_ONBOARD_H_
