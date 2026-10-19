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
#pragma once

#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include "common/host_api.h"
#include "runtime_c_api.h"

// One invocation's metadata values. DMA sources belong to the slot; publication
// copies into that pinned storage before any asynchronous reader is queued.
class RuntimeLaunchMetadata {
public:
    RuntimeLaunchMetadata() = default;
    RuntimeLaunchMetadata(const RuntimeLaunchMetadata &) = delete;
    RuntimeLaunchMetadata &operator=(const RuntimeLaunchMetadata &) = delete;
    RuntimeLaunchMetadata(RuntimeLaunchMetadata &&other) noexcept :
        regions_(std::move(other.regions_)),
        bytes_(std::exchange(other.bytes_, 0)) {
        other.regions_.clear();
    }
    RuntimeLaunchMetadata &operator=(RuntimeLaunchMetadata &&other) noexcept {
        if (this != &other) {
            regions_ = std::move(other.regions_);
            bytes_ = std::exchange(other.bytes_, 0);
            other.regions_.clear();
        }
        return *this;
    }

    int prepare(const HostMetadataRegion *regions, size_t count) {
        if (count != 0 && regions == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto &region = regions[i];
            if (region.device_target == nullptr || region.host_source == nullptr || region.bytes == 0 ||
                region.bytes > std::numeric_limits<size_t>::max() - total)
                return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
            total += region.bytes;
        }
        std::vector<Region> captured;
        captured.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto *source = static_cast<const std::byte *>(regions[i].host_source);
            captured.push_back({regions[i].device_target, {source, source + regions[i].bytes}});
        }
        regions_ = std::move(captured);
        bytes_ = total;
        return 0;
    }

    size_t size() const { return bytes_; }
    void clear() {
        regions_.clear();
        bytes_ = 0;
    }

    template <typename Copy>
    int publish(void *pinned, size_t capacity, Copy &&copy) {
        if (bytes_ > capacity || (bytes_ != 0 && pinned == nullptr)) return PTO_RUNTIME_ERR_INVALID_STATE;
        auto captured = std::move(regions_);
        clear();
        auto *source = static_cast<std::byte *>(pinned);
        size_t offset = 0;
        for (const auto &region : captured) {
            std::memcpy(source + offset, region.bytes.data(), region.bytes.size());
            offset += region.bytes.size();
        }
        offset = 0;
        for (const auto &region : captured) {
            const int rc = copy(region.target, source + offset, region.bytes.size());
            if (rc != 0) return rc;
            offset += region.bytes.size();
        }
        return 0;
    }

private:
    struct Region {
        void *target;
        std::vector<std::byte> bytes;
    };
    std::vector<Region> regions_;
    size_t bytes_{0};
};
