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

    template <typename Copy>
    int publish(Copy &&copy) {
        if (bytes_.empty()) return PTO_RUNTIME_ERR_INTERNAL;
        auto bytes = std::move(bytes_);
        // The consumed source stays empty unless prepared again; a repeated or
        // reentrant publish is rejected instead of copying the descriptor twice.
        bytes_.clear();
        return copy(bytes.data(), bytes.size());
    }

    void clear() { bytes_.clear(); }

private:
    std::vector<std::byte> bytes_;
};
