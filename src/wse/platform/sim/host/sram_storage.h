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
#include "common/hardware_spec.h"
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace wse {
// One device-owned backing: contiguous application windows followed by private runtime windows.
class SramStorage {
public:
    SramStorage();
    void freeze();
    uint32_t core_count() const;
    std::vector<uint64_t> configuration() const;
    std::vector<uint64_t> info(uint32_t core) const;
    void *claim(size_t bytes);
    bool owns(const void *ptr) const;
    void release(void *ptr);
    void check_application(uint64_t address, uint64_t bytes) const;
    std::vector<SramAllocation> allocations() const;
    std::vector<SramRegion> regions() const;
    void set_runtime_used(const std::vector<uint64_t> &used);
    void reset_runtime_used() noexcept;
    void clear();
    size_t committed_bytes() const;

private:
    struct Free {
        void operator()(std::byte *p) const noexcept;
    };
    mutable std::mutex mutex_;
    std::unique_ptr<std::byte, Free> backing_;
    std::vector<uint64_t> capacities_, reserves_, runtime_used_;
    uint64_t application_bytes_{0};
    uint64_t total_bytes_{0};
    bool claimed_{false};
    void initialize_locked();
};
}  // namespace wse
