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

int a2a3_onboard_host_device_mapped_region_allocate(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
);

#endif  // SRC_A2A3_PLATFORM_ONBOARD_HOST_HOST_DEVICE_MAPPED_REGION_ONBOARD_H_
