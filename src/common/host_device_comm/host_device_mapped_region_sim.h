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

#ifndef SRC_COMMON_HOST_DEVICE_COMM_HOST_DEVICE_MAPPED_REGION_SIM_H_
#define SRC_COMMON_HOST_DEVICE_COMM_HOST_DEVICE_MAPPED_REGION_SIM_H_

#include "host_device_comm/host_device_mapped_region.h"

int host_device_mapped_region_allocate_sim(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
);

#endif  // SRC_COMMON_HOST_DEVICE_COMM_HOST_DEVICE_MAPPED_REGION_SIM_H_
