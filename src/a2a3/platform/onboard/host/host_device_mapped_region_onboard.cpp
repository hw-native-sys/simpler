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

#include "host_device_mapped_region_onboard.h"

#include <errno.h>

#include <new>

#include "device_runner.h"

namespace {

struct OnboardMappedRegionResource {
    DeviceRunner *runner = nullptr;
    void *dev_ptr = nullptr;
    void *host_ptr = nullptr;
};

}  // namespace

int a2a3_onboard_host_device_mapped_region_allocate(
    DeviceContextHandle ctx, uint64_t total_bytes, HostDeviceMappedRegionPlatform *platform, void **host_base,
    void **device_base
) {
    if (ctx == NULL || platform == NULL || host_base == NULL || device_base == NULL) {
        return -EINVAL;
    }
    *host_base = nullptr;
    *device_base = nullptr;
    auto *runner = static_cast<DeviceRunner *>(ctx);
    if (runner->device_id() < 0) {
        return -EIO;
    }

    auto *resource = new (std::nothrow) OnboardMappedRegionResource;
    if (resource == nullptr) {
        return -ENOMEM;
    }
    resource->runner = runner;
    resource->dev_ptr = runner->allocate_tensor(static_cast<size_t>(total_bytes));
    if (resource->dev_ptr == nullptr) {
        delete resource;
        return -ENOMEM;
    }

    int rc =
        runner->host_register_device_memory(resource->dev_ptr, static_cast<size_t>(total_bytes), &resource->host_ptr);
    if (rc != 0 || resource->host_ptr == nullptr) {
        runner->free_tensor(resource->dev_ptr);
        delete resource;
        return -EIO;
    }

    platform->resource = resource;
    platform->device_id = static_cast<uint64_t>(runner->device_id());
    platform->cache_ops_cookie = runner;
    platform->flush_host_range = [](HostDeviceMappedRegionPlatform *p, void *host_ptr, uint64_t bytes) {
        auto *mapped_runner = static_cast<DeviceRunner *>(p->cache_ops_cookie);
        if (mapped_runner == nullptr) {
            return -1;
        }
        return mapped_runner->flush_host_cache_range(host_ptr, static_cast<size_t>(bytes));
    };
    platform->invalidate_host_range = [](HostDeviceMappedRegionPlatform *p, void *host_ptr, uint64_t bytes) {
        auto *mapped_runner = static_cast<DeviceRunner *>(p->cache_ops_cookie);
        if (mapped_runner == nullptr) {
            return -1;
        }
        return mapped_runner->invalidate_host_cache_range(host_ptr, static_cast<size_t>(bytes));
    };
    platform->release = [](HostDeviceMappedRegionPlatform *p) {
        auto *r = static_cast<OnboardMappedRegionResource *>(p->resource);
        if (r == nullptr) {
            return;
        }
        if (r->runner != nullptr) {
            if (r->host_ptr != nullptr) {
                (void)r->runner->host_unregister_device_memory(r->host_ptr);
            }
            if (r->dev_ptr != nullptr) {
                r->runner->free_tensor(r->dev_ptr);
            }
        }
        delete r;
        p->resource = nullptr;
    };
    *host_base = resource->host_ptr;
    *device_base = resource->dev_ptr;
    return 0;
}
