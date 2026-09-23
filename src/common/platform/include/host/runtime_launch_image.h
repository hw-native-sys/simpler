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

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "runtime.h"
#include "runtime_c_api.h"

// A host-owned snapshot of the uploaded prefix of the device descriptor for one
// invocation. The destination is owned separately by the run's persistent slot
// and is allocated to the full device extent, which is never shorter: a
// descriptor may end in storage no host copy reaches — device-initialized, or
// never host-initialized at all — and no bytes for that range are snapshotted
// or copied. The caller supplies the length, because which prefix a publication
// carries depends on whether its destination block has been initialized yet.
// Publication is synchronous; consuming a snapshot neither frees nor resets the
// destination.
//
// The snapshot is the only source a publication reads, so a caller may capture
// it and publish later without re-reading the `Runtime` it came from — which by
// then may hold a successor's values. What a caller may still change is
// `patch`: a control word whose value is a property of the publication rather
// than of the captured state, decided once the run knows which route it takes.
// `publish` takes its own length so the same snapshot can serve either route,
// and refuses one longer than what was captured.
class RuntimeLaunchImage {
public:
    RuntimeLaunchImage() = default;
    RuntimeLaunchImage(const RuntimeLaunchImage &) = delete;
    RuntimeLaunchImage &operator=(const RuntimeLaunchImage &) = delete;
    RuntimeLaunchImage(RuntimeLaunchImage &&) noexcept = default;
    RuntimeLaunchImage &operator=(RuntimeLaunchImage &&) noexcept = default;

    void prepare(const Runtime &runtime, size_t bytes) {
        bytes_.resize(bytes);
        std::memcpy(bytes_.data(), &runtime, bytes_.size());
    }

    size_t size() const { return bytes_.size(); }

    // Overwrite a window of the captured bytes. Byte-wise, and refused unless
    // the whole window is inside the snapshot, so a patch can neither reach
    // past what will be published nor assume the snapshot's alignment.
    bool patch(size_t offset, const void *source, size_t bytes) {
        if (source == nullptr) return false;
        if (offset > bytes_.size() || bytes > bytes_.size() - offset) return false;
        std::memcpy(bytes_.data() + offset, source, bytes);
        return true;
    }

    // Read a window of the captured bytes into `dest`. Same containment rule:
    // the launch package copies its entry values out of here, not out of the
    // caller's `Runtime`.
    bool read(size_t offset, void *dest, size_t bytes) const {
        if (dest == nullptr) return false;
        if (offset > bytes_.size() || bytes > bytes_.size() - offset) return false;
        std::memcpy(dest, bytes_.data() + offset, bytes);
        return true;
    }

    template <typename Copy>
    int publish(Copy &&copy, size_t bytes) {
        if (bytes_.empty() || bytes == 0 || bytes > bytes_.size()) return PTO_RUNTIME_ERR_INTERNAL;
        auto captured = std::move(bytes_);
        // The consumed source stays empty unless prepared again; a repeated or
        // reentrant publish is rejected instead of copying the descriptor twice.
        bytes_.clear();
        return copy(captured.data(), bytes);
    }

    void clear() { bytes_.clear(); }

private:
    std::vector<std::byte> bytes_;
};
