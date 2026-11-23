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
        execution_image_(std::move(other.execution_image_)),
        bytes_(std::exchange(other.bytes_, 0)) {
        other.clear();
    }
    RuntimeLaunchMetadata &operator=(RuntimeLaunchMetadata &&other) noexcept {
        if (this != &other) {
            regions_ = std::move(other.regions_);
            execution_image_ = std::move(other.execution_image_);
            bytes_ = std::exchange(other.bytes_, 0);
            other.clear();
        }
        return *this;
    }

    int prepare(
        const HostMetadataRegion *regions, size_t count, bool deliver_execution_image = false,
        std::vector<HbgImageReference> *owner = nullptr
    ) {
        if (count != 0 && regions == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        size_t total = 0;
        bool image_seen = false;
        for (size_t i = 0; i < count; ++i) {
            const auto &region = regions[i];
            if (region.device_target == nullptr || region.host_source == nullptr || region.bytes == 0)
                return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
            if (deliver_execution_image && region.execution_image) {
                if (image_seen || region.bytes > UINT32_MAX ||
                    (region.mutable_prefix_bytes != UINT64_MAX && region.mutable_prefix_bytes > region.bytes) ||
                    region.reference_count > (UINT32_MAX - region.bytes) / sizeof(HbgImageReference) ||
                    (region.reference_count != 0 && region.image_references == nullptr))
                    return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
                for (size_t j = 0; j < region.reference_count; ++j) {
                    const auto &ref = region.image_references[j];
                    if (ref.device_field_address == 0 || ref.device_field_address % alignof(uint64_t) != 0 ||
                        ref.image_offset >= region.bytes)
                        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
                }
                if (owner && (owner->data() != region.image_references || owner->size() != region.reference_count))
                    return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
                image_seen = true;
                continue;
            }
            if (region.bytes > std::numeric_limits<size_t>::max() - total) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
            total += region.bytes;
        }
        if (owner && !image_seen) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        std::vector<Region> captured;
        captured.reserve(count);
        Region image{};
        for (size_t i = 0; i < count; ++i) {
            const auto *source = static_cast<const std::byte *>(regions[i].host_source);
            Region region{regions[i].device_target, {source, source + regions[i].bytes}};
            if (deliver_execution_image && regions[i].execution_image) {
                region.mutable_prefix_bytes =
                    regions[i].mutable_prefix_bytes == UINT64_MAX ? regions[i].bytes : regions[i].mutable_prefix_bytes;
                if (regions[i].reference_count != 0 && !owner)
                    region.references.assign(
                        regions[i].image_references, regions[i].image_references + regions[i].reference_count
                    );
                image = std::move(region);
            } else captured.push_back(std::move(region));
        }
        if (owner) image.references = std::move(*owner);
        regions_ = std::move(captured);
        execution_image_ = std::move(image);
        bytes_ = total;
        return 0;
    }

    int append(const HostMetadataRegion &region) {
        if (region.device_target == nullptr || region.host_source == nullptr || region.bytes == 0 ||
            region.bytes > std::numeric_limits<size_t>::max() - bytes_)
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        const auto *source = static_cast<const std::byte *>(region.host_source);
        regions_.push_back({region.device_target, {source, source + region.bytes}});
        bytes_ += region.bytes;
        return 0;
    }

    size_t size() const { return bytes_; }
    bool has_execution_image() const { return !execution_image_.bytes.empty(); }
    HostMetadataRegion execution_image() const {
        return {
            execution_image_.target,
            execution_image_.bytes.data(),
            execution_image_.bytes.size(),
            true,
            execution_image_.mutable_prefix_bytes,
            execution_image_.references.data(),
            execution_image_.references.size()
        };
    }
    void clear() {
        regions_.clear();
        execution_image_ = {};
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
        uint64_t mutable_prefix_bytes{0};
        std::vector<HbgImageReference> references{};
    };
    std::vector<Region> regions_;
    Region execution_image_{};
    size_t bytes_{0};
};
