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

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ccu_native_abi.h"

namespace simpler::ccu {

using BuildIdentity = std::array<uint8_t, SIMPLER_CCU_BUILD_ID_SIZE>;

struct ExpectedBuildIdentity {
    BuildIdentity sdk;
    BuildIdentity pto_isa;
};

struct NativeKernelDefinition {
    SimplerCcuKernelRecord record{};
    std::string entry_symbol;
    const void *entry = nullptr;
};

class NativeModule final {
public:
    ~NativeModule();
    NativeModule(const NativeModule &) = delete;
    NativeModule &operator=(const NativeModule &) = delete;

    static std::shared_ptr<NativeModule>
    load(std::vector<uint8_t> module_bytes, const ExpectedBuildIdentity &expected_identity);

    const std::vector<uint8_t> &module_bytes() const { return module_bytes_; }
    const std::vector<uint8_t> &manifest_bytes() const { return manifest_bytes_; }
    const std::vector<NativeKernelDefinition> &kernels() const { return kernels_; }
    const std::vector<uint8_t> &cache_identity() const { return cache_identity_; }

private:
    NativeModule() = default;

    void *handle_ = nullptr;
    int backing_fd_ = -1;
    std::vector<uint8_t> module_bytes_;
    std::vector<uint8_t> manifest_bytes_;
    std::vector<NativeKernelDefinition> kernels_;
    std::vector<uint8_t> cache_identity_;
};

class NativeModuleCache final {
public:
    std::shared_ptr<NativeModule>
    load(std::vector<uint8_t> module_bytes, const ExpectedBuildIdentity &expected_identity);
    size_t cached_module_count() const;

private:
    void prune_expired();

    mutable std::mutex mutex_;
    std::map<std::vector<uint8_t>, std::weak_ptr<NativeModule>> modules_;
};

}  // namespace simpler::ccu
