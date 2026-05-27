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

#include "host_device_comm/host_device_mapped_region_sim.h"

#include <errno.h>

#include <cstdlib>

int host_device_mapped_region_allocate_sim(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    (void)ctx;
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, static_cast<size_t>(total_bytes)) != 0) {
        return -ENOMEM;
    }
    platform->resource = ptr;
    platform->release = [](HostDeviceMappedRegionPlatform *p) {
        std::free(p->resource);
        p->resource = nullptr;
    };
    *host_base = ptr;
    *device_base = ptr;
    return 0;
}
