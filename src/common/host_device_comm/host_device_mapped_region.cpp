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

#include "host_device_comm/host_device_mapped_region.h"

#include <errno.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

static_assert(sizeof(HostDeviceMappedRegionHeader) == 64);
static_assert(alignof(HostDeviceMappedRegionHeader) == 64);
static_assert(sizeof(HostDeviceMappedRegionSignalSlot) == 64);
static_assert(alignof(HostDeviceMappedRegionSignalSlot) == 64);
static_assert(offsetof(HostDeviceMappedRegionSignalSlot, value) == 0);

static_assert(offsetof(HostDeviceMappedRegionInfo, host_data_ptr) == 0);
static_assert(offsetof(HostDeviceMappedRegionInfo, device_data_ptr) == 8);
static_assert(offsetof(HostDeviceMappedRegionInfo, data_bytes) == 16);
static_assert(offsetof(HostDeviceMappedRegionInfo, host_signal_ptr) == 24);
static_assert(offsetof(HostDeviceMappedRegionInfo, device_signal_ptr) == 32);
static_assert(offsetof(HostDeviceMappedRegionInfo, signal_count) == 40);
static_assert(offsetof(HostDeviceMappedRegionInfo, reserved0) == 44);
static_assert(offsetof(HostDeviceMappedRegionInfo, total_bytes) == 48);
static_assert(offsetof(HostDeviceMappedRegionInfo, flags) == 56);
static_assert(offsetof(HostDeviceMappedRegionInfo, reserved1) == 60);
static_assert(sizeof(HostDeviceMappedRegionInfo) == 64);

namespace {

constexpr uint64_t kAlignment = 64;

struct HostDeviceMappedRegion {
    DeviceContextHandle owner_ctx = nullptr;
    void *host_base = nullptr;
    void *device_base = nullptr;
    uint64_t total_bytes = 0;
    uint64_t data_offset = 0;
    uint64_t data_bytes = 0;
    uint64_t signal_offset = 0;
    uint32_t signal_count = 0;
    uint32_t flags = 0;
    std::mutex signal_mu;
    std::mutex op_mu;
    std::condition_variable op_cv;
    uint32_t active_ops = 0;
    HostDeviceMappedRegionPlatform platform{};
};

std::mutex &registry_mutex() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<DeviceContextHandle, std::vector<HostDeviceMappedRegion *>> &registry_by_ctx() {
    static std::unordered_map<DeviceContextHandle, std::vector<HostDeviceMappedRegion *>> registry;
    return registry;
}

std::unordered_map<HostDeviceMappedRegionHandle, HostDeviceMappedRegion *> &registry_by_handle() {
    static std::unordered_map<HostDeviceMappedRegionHandle, HostDeviceMappedRegion *> registry;
    return registry;
}

bool add_overflow(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return true;
    }
    *out = a + b;
    return false;
}

bool mul_overflow(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    *out = a * b;
    return false;
}

bool align64(uint64_t value, uint64_t *out) {
    uint64_t padded = 0;
    if (add_overflow(value, kAlignment - 1, &padded)) {
        return false;
    }
    *out = padded & ~(kAlignment - 1);
    return true;
}

uint8_t *byte_ptr(void *base, uint64_t offset) { return static_cast<uint8_t *>(base) + offset; }

const uint8_t *byte_ptr(const void *base, uint64_t offset) { return static_cast<const uint8_t *>(base) + offset; }

HostDeviceMappedRegion *lookup_region_locked(DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle) {
    if (ctx == nullptr || handle == nullptr) {
        return nullptr;
    }
    auto it = registry_by_handle().find(handle);
    if (it == registry_by_handle().end()) {
        return nullptr;
    }
    HostDeviceMappedRegion *region = it->second;
    if (region == nullptr || region->owner_ctx != ctx) {
        return nullptr;
    }
    return region;
}

HostDeviceMappedRegion *acquire_region(DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    HostDeviceMappedRegion *region = lookup_region_locked(ctx, handle);
    if (region == nullptr) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> op_lock(region->op_mu);
        ++region->active_ops;
    }
    return region;
}

void release_region_op(HostDeviceMappedRegion *region) {
    std::lock_guard<std::mutex> op_lock(region->op_mu);
    --region->active_ops;
    if (region->active_ops == 0) {
        region->op_cv.notify_all();
    }
}

void wait_for_region_ops(HostDeviceMappedRegion *region) {
    std::unique_lock<std::mutex> op_lock(region->op_mu);
    region->op_cv.wait(op_lock, [region] {
        return region->active_ops == 0;
    });
}

void release_region(HostDeviceMappedRegion *region) {
    if (region == nullptr) {
        return;
    }
    if (region->platform.release != nullptr) {
        region->platform.release(&region->platform);
    }
    delete region;
}

int validate_range(HostDeviceMappedRegion *region, uint64_t offset, size_t nbytes) {
    if (offset > region->data_bytes) {
        return -EINVAL;
    }
    if (static_cast<uint64_t>(nbytes) > region->data_bytes - offset) {
        return -EINVAL;
    }
    return 0;
}

HostDeviceMappedRegionSignalSlot *signal_slot(HostDeviceMappedRegion *region, uint32_t signal_id) {
    auto *slots =
        reinterpret_cast<HostDeviceMappedRegionSignalSlot *>(byte_ptr(region->host_base, region->signal_offset));
    return &slots[signal_id];
}

int flush_host_range(HostDeviceMappedRegion *region, void *host_ptr, uint64_t bytes) {
    if (bytes == 0 || region->platform.flush_host_range == nullptr) {
        return 0;
    }
    return region->platform.flush_host_range(&region->platform, host_ptr, bytes);
}

int invalidate_host_range(HostDeviceMappedRegion *region, void *host_ptr, uint64_t bytes) {
    if (bytes == 0 || region->platform.invalidate_host_range == nullptr) {
        return 0;
    }
    return region->platform.invalidate_host_range(&region->platform, host_ptr, bytes);
}

}  // namespace

int host_device_mapped_region_compute_total_bytes(
    const HostDeviceMappedRegionConfig *cfg, uint64_t *signal_offset, uint64_t *data_offset, uint64_t *total_bytes
) {
    if (cfg == nullptr || signal_offset == nullptr || data_offset == nullptr || total_bytes == nullptr) {
        return -EINVAL;
    }
    if (cfg->data_bytes == 0 || cfg->signal_count == 0 || cfg->flags != 0) {
        return -EINVAL;
    }

    uint64_t signal_bytes = 0;
    if (mul_overflow(cfg->signal_count, sizeof(HostDeviceMappedRegionSignalSlot), &signal_bytes)) {
        return -EINVAL;
    }

    *signal_offset = sizeof(HostDeviceMappedRegionHeader);
    uint64_t signals_end = 0;
    if (add_overflow(*signal_offset, signal_bytes, &signals_end)) {
        return -EINVAL;
    }
    if (!align64(signals_end, data_offset)) {
        return -EINVAL;
    }

    uint64_t data_end = 0;
    if (add_overflow(*data_offset, cfg->data_bytes, &data_end)) {
        return -EINVAL;
    }
    if (!align64(data_end, total_bytes)) {
        return -EINVAL;
    }
    return 0;
}

int host_device_mapped_region_open_common(
    DeviceContextHandle ctx, const HostDeviceMappedRegionConfig *cfg, HostDeviceMappedRegionHandle *out_region,
    HostDeviceMappedRegionAllocateFn allocate
) {
    if (out_region != nullptr) {
        *out_region = nullptr;
    }
    if (ctx == nullptr || cfg == nullptr || out_region == nullptr || allocate == nullptr) {
        return -EINVAL;
    }

    uint64_t signal_offset = 0;
    uint64_t data_offset = 0;
    uint64_t total_bytes = 0;
    int rc = host_device_mapped_region_compute_total_bytes(cfg, &signal_offset, &data_offset, &total_bytes);
    if (rc != 0) {
        return rc;
    }

    auto *region = new (std::nothrow) HostDeviceMappedRegion;
    if (region == nullptr) {
        return -ENOMEM;
    }
    region->owner_ctx = ctx;
    region->total_bytes = total_bytes;
    region->data_offset = data_offset;
    region->data_bytes = cfg->data_bytes;
    region->signal_offset = signal_offset;
    region->signal_count = cfg->signal_count;
    region->flags = cfg->flags;

    rc = allocate(ctx, total_bytes, &region->platform, &region->host_base, &region->device_base);
    if (rc != 0) {
        delete region;
        return rc;
    }
    if (region->host_base == nullptr || region->device_base == nullptr) {
        release_region(region);
        return -EIO;
    }

    std::memset(region->host_base, 0, static_cast<size_t>(total_bytes));
    auto *header = reinterpret_cast<HostDeviceMappedRegionHeader *>(region->host_base);
    header->magic = HDMR_MAGIC;
    header->version = HDMR_VERSION;
    header->flags = cfg->flags;
    header->signal_count = cfg->signal_count;
    header->signal_offset = signal_offset;
    header->data_offset = data_offset;
    header->data_bytes = cfg->data_bytes;
    header->total_bytes = total_bytes;
    rc = flush_host_range(region, region->host_base, total_bytes);
    if (rc != 0) {
        release_region(region);
        return -EIO;
    }

    HostDeviceMappedRegionHandle handle = static_cast<HostDeviceMappedRegionHandle>(region);
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        registry_by_handle()[handle] = region;
        registry_by_ctx()[ctx].push_back(region);
    }
    *out_region = handle;
    return 0;
}

int host_device_mapped_region_close_common(DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle) {
    HostDeviceMappedRegion *region = nullptr;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        region = lookup_region_locked(ctx, handle);
        if (region == nullptr) {
            return -EINVAL;
        }
        registry_by_handle().erase(handle);
        auto &regions = registry_by_ctx()[ctx];
        for (auto it = regions.begin(); it != regions.end(); ++it) {
            if (*it == region) {
                regions.erase(it);
                break;
            }
        }
        if (regions.empty()) {
            registry_by_ctx().erase(ctx);
        }
    }
    wait_for_region_ops(region);
    release_region(region);
    return 0;
}

void host_device_mapped_region_close_all_common(DeviceContextHandle ctx) {
    if (ctx == nullptr) {
        return;
    }
    std::vector<HostDeviceMappedRegion *> regions;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        auto it = registry_by_ctx().find(ctx);
        if (it == registry_by_ctx().end()) {
            return;
        }
        regions.swap(it->second);
        registry_by_ctx().erase(it);
        for (HostDeviceMappedRegion *region : regions) {
            registry_by_handle().erase(static_cast<HostDeviceMappedRegionHandle>(region));
        }
    }
    for (HostDeviceMappedRegion *region : regions) {
        wait_for_region_ops(region);
        release_region(region);
    }
}

int host_device_mapped_region_info_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle, HostDeviceMappedRegionInfo *info
) {
    if (info == nullptr) {
        return -EINVAL;
    }
    HostDeviceMappedRegion *region = acquire_region(ctx, handle);
    if (region == nullptr) {
        return -EINVAL;
    }
    std::memset(info, 0, sizeof(*info));
    info->host_data_ptr = reinterpret_cast<uint64_t>(byte_ptr(region->host_base, region->data_offset));
    info->device_data_ptr = reinterpret_cast<uint64_t>(byte_ptr(region->device_base, region->data_offset));
    info->data_bytes = region->data_bytes;
    info->host_signal_ptr = reinterpret_cast<uint64_t>(byte_ptr(region->host_base, region->signal_offset));
    info->device_signal_ptr = reinterpret_cast<uint64_t>(byte_ptr(region->device_base, region->signal_offset));
    info->signal_count = region->signal_count;
    info->total_bytes = region->total_bytes;
    info->flags = region->flags;
    release_region_op(region);
    return 0;
}

int host_device_mapped_region_datacopy_h2region_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle, uint64_t offset, const void *src, size_t nbytes
) {
    if (src == nullptr && nbytes != 0) {
        return -EINVAL;
    }
    HostDeviceMappedRegion *region = acquire_region(ctx, handle);
    if (region == nullptr) {
        return -EINVAL;
    }
    int rc = validate_range(region, offset, nbytes);
    if (rc != 0) {
        release_region_op(region);
        return rc;
    }
    if (nbytes != 0) {
        uint8_t *dst = byte_ptr(region->host_base, region->data_offset + offset);
        std::memcpy(dst, src, nbytes);
        rc = flush_host_range(region, dst, static_cast<uint64_t>(nbytes));
        if (rc != 0) {
            release_region_op(region);
            return -EIO;
        }
    }
    release_region_op(region);
    return 0;
}

int host_device_mapped_region_datacopy_region2h_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle, uint64_t offset, void *dst, size_t nbytes
) {
    if (dst == nullptr && nbytes != 0) {
        return -EINVAL;
    }
    HostDeviceMappedRegion *region = acquire_region(ctx, handle);
    if (region == nullptr) {
        return -EINVAL;
    }
    int rc = validate_range(region, offset, nbytes);
    if (rc != 0) {
        release_region_op(region);
        return rc;
    }
    if (nbytes != 0) {
        uint8_t *src = byte_ptr(region->host_base, region->data_offset + offset);
        rc = invalidate_host_range(region, src, static_cast<uint64_t>(nbytes));
        if (rc != 0) {
            release_region_op(region);
            return -EIO;
        }
        std::memcpy(dst, src, nbytes);
    }
    release_region_op(region);
    return 0;
}

int host_device_mapped_region_notify_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle, uint32_t signal_id, uint32_t value
) {
    HostDeviceMappedRegion *region = acquire_region(ctx, handle);
    if (region == nullptr || signal_id >= region->signal_count) {
        if (region != nullptr) {
            release_region_op(region);
        }
        return -EINVAL;
    }
    auto *slot = signal_slot(region, signal_id);
    int rc = 0;
    {
        std::lock_guard<std::mutex> lock(region->signal_mu);
        uint32_t current = slot->value;
        if (value < current) {
            rc = -EINVAL;
        } else {
            if (value > current) {
                slot->value = value;
            }
            rc = flush_host_range(region, slot, sizeof(*slot));
            if (rc != 0) {
                rc = -EIO;
            }
        }
    }
    release_region_op(region);
    return rc;
}

int host_device_mapped_region_wait_common(
    DeviceContextHandle ctx, HostDeviceMappedRegionHandle handle, uint32_t signal_id, uint32_t target,
    uint32_t timeout_us
) {
    HostDeviceMappedRegion *region = acquire_region(ctx, handle);
    if (region == nullptr || signal_id >= region->signal_count) {
        if (region != nullptr) {
            release_region_op(region);
        }
        return -EINVAL;
    }
    HostDeviceMappedRegionSignalSlot *slot = signal_slot(region, signal_id);

    auto check_signal = [&]() -> int {
        std::lock_guard<std::mutex> lock(region->signal_mu);
        int rc = invalidate_host_range(region, slot, sizeof(*slot));
        if (rc != 0) {
            return -EIO;
        }
        return slot->value >= target ? 1 : 0;
    };

    int rc = check_signal();
    if (rc < 0) {
        release_region_op(region);
        return rc;
    }
    if (rc > 0) {
        release_region_op(region);
        return 0;
    }
    if (timeout_us == 0) {
        release_region_op(region);
        return -EAGAIN;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    do {
        rc = check_signal();
        if (rc < 0) {
            release_region_op(region);
            return rc;
        }
        if (rc > 0) {
            release_region_op(region);
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    release_region_op(region);
    return -EAGAIN;
}
