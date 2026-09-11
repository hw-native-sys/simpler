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

#include "host/kernel_device_resources.h"

#include <algorithm>
#include <limits>

#include "utils/device_arena.h"

namespace {
constexpr uint64_t kAlignment = DeviceArena::kDefaultBaseAlign;
bool is_pipeline_arena_kind(uint32_t kind) {
    return kind >= PTO_PIPELINE_GM_HEAP && kind <= PTO_PIPELINE_RUNTIME_IMAGE;
}

int validate_layout(const KernelResourceLayout &layout) {
    const auto &contract = layout.contract;
    if (!is_valid_pipeline_contract(&contract, SIMPLER_MODE_KERNEL) || contract.pipeline_depth != 1 ||
        !has_serviceable_arena_topology(contract) || !has_serviceable_stream_topology(contract) ||
        layout.regions.empty() || layout.schema == 0)
        return PTO_RUNTIME_ERR_INTERNAL;
    uint64_t total = 0;
    for (uint32_t i = 0; i < contract.resource_count; ++i) {
        const auto &resource = contract.resources[i];
        if (!is_pipeline_arena_kind(resource.kind)) continue;
        if (resource.resource_class == PTO_PIPELINE_EXEC_HANDLE) return PTO_RUNTIME_ERR_INTERNAL;
        if (resource.bytes_per_copy > SIZE_MAX - (kAlignment - 1) ||
            resource.bytes_per_copy + kAlignment - 1 > UINT64_MAX - total)
            return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        total += resource.bytes_per_copy + kAlignment - 1;
    }
    for (size_t i = 0; i < layout.regions.size(); ++i) {
        const auto &region = layout.regions[i];
        const auto *arena = find_pipeline_resource(contract, region.arena_kind);
        if (!is_pipeline_arena_kind(region.arena_kind) || arena == nullptr || region.offset > arena->bytes_per_copy ||
            region.bytes > arena->bytes_per_copy - region.offset || (region.bytes == 0 && region.offset != 0) ||
            (region.bytes != 0 && region.offset % kAlignment != 0))
            return PTO_RUNTIME_ERR_INTERNAL;
        for (size_t j = 0; j < i; ++j) {
            const auto &other = layout.regions[j];
            if (region.arena_kind == other.arena_kind && region.bytes != 0 && other.bytes != 0 &&
                region.offset < other.offset + other.bytes && other.offset < region.offset + region.bytes)
                return PTO_RUNTIME_ERR_INTERNAL;
        }
    }
    return 0;
}
}  // namespace

int KernelDeviceResources::prepare(const KernelResourceLayout &layout, const KernelResourceOps &ops) {
    if (closing_) return PTO_RUNTIME_ERR_INVALID_STATE;
    const int valid = validate_layout(layout);
    if (valid != 0) return valid;
    if (!ops.valid()) return PTO_RUNTIME_ERR_INTERNAL;
    if (prepared_) {
        if (layout.schema != layout_.schema || ops.context != ops_.context || ops.allocate != ops_.allocate ||
            ops.release != ops_.release || layout.regions.size() != layout_.regions.size() ||
            layout.contract.resource_count != layout_.contract.resource_count)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        for (uint32_t i = 0; i < layout.contract.resource_count; ++i) {
            const auto &requested = layout.contract.resources[i];
            const auto *capacity = find_pipeline_resource(layout_.contract, requested.kind);
            if (capacity == nullptr || requested.resource_class != capacity->resource_class)
                return PTO_RUNTIME_ERR_INVALID_STATE;
            if (requested.bytes_per_copy > capacity->bytes_per_copy) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        }
        for (size_t i = 0; i < layout.regions.size(); ++i) {
            if (layout.regions[i].arena_kind != layout_.regions[i].arena_kind) return PTO_RUNTIME_ERR_INVALID_STATE;
            if (layout.regions[i].bytes > layout_.regions[i].bytes) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        }
        // Reuse the original offsets as well as the original allocations.
        return 0;
    }
    // All Host allocations happen before the first device allocation. Moving
    // these containers below does not throw after ownership has been acquired.
    auto next_layout = layout;
    std::vector<Allocation> next_allocations;
    std::vector<KernelResourceView> next_views(layout.regions.size());
    for (uint32_t i = 0; i < layout.contract.resource_count; ++i) {
        const auto &resource = layout.contract.resources[i];
        if (is_pipeline_arena_kind(resource.kind)) next_allocations.push_back({resource.kind, resource.bytes_per_copy});
    }
    layout_ = std::move(next_layout);
    allocations_ = std::move(next_allocations);
    views_ = std::move(next_views);
    ops_ = ops;
    try {
        for (auto &allocation : allocations_) {
            allocation.allocation = ops_.allocate(ops_.context, static_cast<size_t>(allocation.bytes + kAlignment - 1));
            const auto raw = reinterpret_cast<uintptr_t>(allocation.allocation);
            if (allocation.allocation == nullptr || raw > UINTPTR_MAX - (kAlignment - 1) ||
                ((raw + kAlignment - 1) & ~(kAlignment - 1)) > UINTPTR_MAX - allocation.bytes) {
                close();
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            allocation.base = (raw + kAlignment - 1) & ~(kAlignment - 1);
        }
    } catch (...) {
        close();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    for (size_t i = 0; i < layout_.regions.size(); ++i) {
        const auto &region = layout_.regions[i];
        if (region.bytes == 0) continue;
        const auto allocation = std::find_if(allocations_.begin(), allocations_.end(), [&](const auto &entry) {
            return entry.kind == region.arena_kind;
        });
        views_[i] = {allocation->base + region.offset, region.bytes};
    }
    prepared_ = true;
    return 0;
}

int KernelDeviceResources::freeze() {
    if (!prepared_ || closing_ || frozen_) return PTO_RUNTIME_ERR_INVALID_STATE;
    frozen_ = true;
    return 0;
}

int KernelDeviceResources::bind(
    uint64_t schema, const uint64_t *required, size_t count, KernelResourceBinding &out
) const {
    if (!prepared_ || !frozen_ || closing_) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (schema != layout_.schema || required == nullptr || count != views_.size()) return PTO_RUNTIME_ERR_INTERNAL;
    for (size_t i = 0; i < count; ++i) {
        if (required[i] > views_[i].capacity) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    }
    out = {views_.data(), views_.size()};
    return 0;
}

int KernelDeviceResources::close() {
    closing_ = true;
    int error = 0;
    for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) {
        if (it->allocation == nullptr) continue;
        int rc = PTO_RUNTIME_ERR_INTERNAL;
        try {
            rc = ops_.release(ops_.context, it->allocation);
        } catch (...) {
            // Preserve the pointer for explicit cleanup retry.
        }
        if (rc != 0) {
            if (error == 0) error = rc;
            continue;
        }
        it->allocation = nullptr;
        it->base = 0;
    }
    if (error != 0) {
        if (cleanup_error_ == 0) cleanup_error_ = error;
        return error;
    }
    allocations_.clear();
    views_.clear();
    layout_ = {};
    ops_ = {};
    prepared_ = frozen_ = closing_ = false;
    return 0;
}

bool KernelDeviceResources::has_live_resources() const {
    return std::any_of(allocations_.begin(), allocations_.end(), [](const auto &entry) {
        return entry.allocation != nullptr;
    });
}
