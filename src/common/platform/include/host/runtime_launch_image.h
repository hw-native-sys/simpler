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

// A per-invocation descriptor snapshot. Publication reads only captured bytes;
// its callback may enqueue DMA or construct an RTS launch package. Consuming
// the snapshot does not release the destination or any external device address.
// Bounds-checked patch/read operations preserve the chosen publication length.
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
