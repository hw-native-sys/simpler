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

/**
 * Host-side tensor access for the host orchestrator. See
 * host_build_graph/host_tensor_access.h for the contract.
 */

#include "host_build_graph/host_tensor_access.h"

#include <string.h>

#include <vector>

#include "common/host_api.h"

// How a region's bytes are reached.
enum class AccessMeans : uint8_t {
    // Not yet decided. A child-memory region starts here and resolves on the
    // first access that lands in it.
    Unresolved,
    // The caller's host buffer the bind copied in, or a mapping this accessor installed.
    // `needs_push_back` decides whether a write must also reach the device.
    HostView,
    // No host mapping was available for this allocation, so every access is a
    // device copy. Holds no bytes, so nothing here can go stale.
    DeviceCopy,
};

struct HostTensorRegion {
    uint64_t dev_base;
    uint64_t size;
    unsigned char *host_view;
    // A caller-buffer fallback needs a push-back. A platform mapping writes
    // the device allocation directly, even when its host VA differs.
    bool needs_push_back;
    AccessMeans means;
};

// One entry per caller tensor of the run being orchestrated. A run has a
// handful of tensors and orchestration reads are cold-path, so a linear scan
// costs less than the map that would replace it.
struct HostTensorAccessor::Impl {
    const HostApi *api;
    std::vector<HostTensorRegion> regions;
    std::vector<void *> mappings;
    // Bytes covered by `mappings`, i.e. excluding regions serving a fallback view.
    uint64_t mapped_bytes;
    uint64_t device_copy_count;
};

// The region serving the whole of [dev_addr, dev_addr + bytes), or nullptr.
// `*offset` is the span's distance from that region's base.
HostTensorRegion *
find_region(std::vector<HostTensorRegion> &regions, uint64_t dev_addr, uint64_t bytes, uint64_t *offset) {
    for (HostTensorRegion &region : regions) {
        if (dev_addr < region.dev_base) {
            continue;
        }
        uint64_t off = dev_addr - region.dev_base;
        if (off > region.size || bytes > region.size - off) {
            continue;
        }
        *offset = off;
        return &region;
    }
    return nullptr;
}

HostTensorAccessor::HostTensorAccessor(const HostApi *api) :
    impl_(new Impl{api, {}, {}, 0, 0}) {}

HostTensorAccessor::~HostTensorAccessor() {
    close();
    delete impl_;
}

bool HostTensorAccessor::add(uint64_t dev_base, uint64_t size, void *fallback_host_view) {
    if (impl_->api == nullptr || dev_base == 0 || size == 0) {
        return false;
    }
    impl_->regions.reserve(impl_->regions.size() + 1);
    const bool needs_push_back = fallback_host_view != nullptr;
    void *host_view = fallback_host_view;
    if (host_view == nullptr) {
        // Reserve before registration: a later reallocation that threw would
        // leak the mapping.
        impl_->mappings.reserve(impl_->mappings.size() + 1);
        host_view = impl_->api->register_device_memory_to_host(reinterpret_cast<void *>(dev_base), size);
        if (host_view != nullptr) {
            impl_->mappings.push_back(reinterpret_cast<void *>(dev_base));
            impl_->mapped_bytes += size;
        }
    }
    if (host_view == nullptr) {
        return false;
    }
    impl_->regions.push_back(
        {dev_base, size, static_cast<unsigned char *>(host_view), needs_push_back, AccessMeans::HostView}
    );
    return true;
}

bool HostTensorAccessor::add_child_memory(uint64_t dev_base, uint64_t size) {
    if (impl_->api == nullptr || dev_base == 0 || size == 0) {
        return false;
    }
    impl_->regions.push_back({dev_base, size, nullptr, false, AccessMeans::Unresolved});
    return true;
}

// Pick the means for a child-memory region. The platform owns any mapping it
// hands back for the allocation's lifetime, so this accessor records the
// address without taking responsibility for releasing it.
void resolve_means(const HostApi *api, HostTensorRegion *region) {
    void *host_view = api->acquire_child_memory_host_view(reinterpret_cast<void *>(region->dev_base), region->size);
    if (host_view != nullptr) {
        region->host_view = static_cast<unsigned char *>(host_view);
        region->needs_push_back = false;
        region->means = AccessMeans::HostView;
        return;
    }
    region->means = AccessMeans::DeviceCopy;
}

bool HostTensorAccessor::read(uint64_t dev_addr, void *dst, uint64_t bytes) {
    uint64_t offset = 0;
    HostTensorRegion *region = find_region(impl_->regions, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    if (region->means == AccessMeans::Unresolved) {
        resolve_means(impl_->api, region);
    }
    if (region->means == AccessMeans::DeviceCopy) {
        ++impl_->device_copy_count;
        return impl_->api->copy_from_device(dst, reinterpret_cast<void *>(dev_addr), static_cast<size_t>(bytes)) == 0;
    }
    memcpy(dst, region->host_view + offset, bytes);
    return true;
}

bool HostTensorAccessor::write(uint64_t dev_addr, const void *src, uint64_t bytes) {
    uint64_t offset = 0;
    HostTensorRegion *region = find_region(impl_->regions, dev_addr, bytes, &offset);
    if (region == nullptr) {
        return false;
    }
    if (region->means == AccessMeans::Unresolved) {
        resolve_means(impl_->api, region);
    }
    if (region->means == AccessMeans::DeviceCopy) {
        ++impl_->device_copy_count;
        return impl_->api->copy_to_device(reinterpret_cast<void *>(dev_addr), src, static_cast<size_t>(bytes)) == 0;
    }
    unsigned char *dst = region->host_view + offset;
    memcpy(dst, src, bytes);
    if (!region->needs_push_back) {
        return true;
    }
    return impl_->api->copy_to_device(reinterpret_cast<void *>(dev_addr), dst, static_cast<size_t>(bytes)) == 0;
}

size_t HostTensorAccessor::mapping_count() const noexcept { return impl_->mappings.size(); }

uint64_t HostTensorAccessor::mapped_bytes() const noexcept { return impl_->mapped_bytes; }

uint64_t HostTensorAccessor::device_copy_count() const noexcept { return impl_->device_copy_count; }

void HostTensorAccessor::close() noexcept {
    for (void *dev_ptr : impl_->mappings) {
        impl_->api->unregister_device_memory_from_host(dev_ptr);
    }
    impl_->mappings.clear();
    impl_->regions.clear();
    impl_->mapped_bytes = 0;
    impl_->device_copy_count = 0;
}

bool host_tensor_read(HostTensorAccessor *accessor, uint64_t dev_addr, void *dst, uint64_t bytes) {
    return accessor != nullptr && accessor->read(dev_addr, dst, bytes);
}

bool host_tensor_write(HostTensorAccessor *accessor, uint64_t dev_addr, const void *src, uint64_t bytes) {
    return accessor != nullptr && accessor->write(dev_addr, src, bytes);
}
