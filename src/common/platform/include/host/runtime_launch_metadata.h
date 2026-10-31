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
    struct Region {
        void *target;
        std::vector<std::byte> bytes;
    };

public:
    // Owns the region records, but borrows the pinned source and device targets.
    // Their owner must retain both allocations and immutable source contents
    // through all queued readers (and through graph retirement when captured).
    class Publication {
    public:
        Publication() = default;
        Publication(const Publication &) = delete;
        Publication &operator=(const Publication &) = delete;
        Publication(Publication &&other) noexcept :
            regions_(std::move(other.regions_)),
            source_(std::exchange(other.source_, nullptr)),
            ready_(std::exchange(other.ready_, false)) {}
        Publication &operator=(Publication &&) = delete;

        // Enqueue does not rewrite sources or consume the record. A failure may
        // leave accepted readers; retirement belongs to the submission owner,
        // and this operation provides no automatic retry or completion proof.
        template <typename Copy>
        int enqueue(Copy &&copy) const {
            if (!ready_) return PTO_RUNTIME_ERR_INVALID_STATE;
            size_t offset = 0;
            for (const auto &region : regions_) {
                const int rc = copy(region.target, source_ + offset, region.bytes.size());
                if (rc != 0) return rc;
                offset += region.bytes.size();
            }
            return 0;
        }

    private:
        friend class RuntimeLaunchMetadata;
        std::vector<Region> regions_;
        const std::byte *source_{nullptr};
        bool ready_{false};
    };

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
    void clear() {
        regions_.clear();
        bytes_ = 0;
    }

    // All source bytes are materialized before any enqueue callback is allowed.
    // Invalid capacity or an occupied output leaves the pending snapshot intact.
    int materialize(void *pinned, size_t capacity, Publication &publication) {
        if (publication.ready_ || bytes_ > capacity || (bytes_ != 0 && pinned == nullptr))
            return PTO_RUNTIME_ERR_INVALID_STATE;
        auto *source = static_cast<std::byte *>(pinned);
        size_t offset = 0;
        for (const auto &region : regions_) {
            std::memcpy(source + offset, region.bytes.data(), region.bytes.size());
            offset += region.bytes.size();
        }
        publication.regions_ = std::move(regions_);
        publication.source_ = source;
        publication.ready_ = true;
        clear();
        return 0;
    }

    template <typename Copy>
    int publish(void *pinned, size_t capacity, Copy &&copy) {
        Publication publication;
        const int rc = materialize(pinned, capacity, publication);
        if (rc != 0) return rc;
        return publication.enqueue(std::forward<Copy>(copy));
    }

private:
    std::vector<Region> regions_;
    size_t bytes_{0};
};
