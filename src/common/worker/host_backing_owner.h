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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "host/host_backing.h"
#include "runtime_module.h"

namespace simpler::host {

int query_host_backing_identity(const RuntimeModule &module);

class HostBackingOwner {
public:
    explicit HostBackingOwner(const std::string &runtime_path);
    ~HostBackingOwner();
    HostBackingOwner(const HostBackingOwner &) = delete;
    HostBackingOwner &operator=(const HostBackingOwner &) = delete;

    std::pair<uint64_t, std::vector<uint8_t>> allocate(int device, uint32_t numa, uint64_t bytes);
    void authorize(const std::vector<int32_t> &identities);
    void require_mapping_support() const;
    uint64_t base() const;
    void release();
    int process_identity() const;

private:
    std::shared_ptr<RuntimeModule> module_;
    void *owner_{nullptr};
    decltype(&host_backing_allocate) allocate_;
    decltype(&host_backing_authorize) authorize_;
    decltype(&host_backing_get_base) base_;
    decltype(&host_backing_release) release_;
};
}  // namespace simpler::host
