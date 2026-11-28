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
#include "sram_storage.h"
#include "hardware_defaults.h"
#include <sys/mman.h>
#include <algorithm>
#include <stdexcept>

namespace wse {
void SramStorage::Unmap::operator()(std::byte *p) const noexcept { munmap(p, bytes); }

SramStorage::Mapping SramStorage::map(size_t bytes) {
    if (!bytes) return {nullptr, Unmap{0}};
    void *base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) throw std::bad_alloc();
    return {static_cast<std::byte *>(base), Unmap{bytes}};
}

SramStorage::SramStorage() :
    capacities_(DEFAULT_CAPACITIES, DEFAULT_CAPACITIES + DEFAULT_CORES),
    reserves_(DEFAULT_RESERVES, DEFAULT_RESERVES + DEFAULT_CORES),
    runtime_used_(DEFAULT_CORES, 0) {
    if (!DEFAULT_CORES || DEFAULT_CORES > MAX_CORE_COUNT || DEFAULT_PES != PE_PER_CORE)
        throw std::invalid_argument("unsupported WSE Core/PE configuration");
    for (size_t i = 0; i < capacities_.size(); ++i) {
        if (!reserves_[i] || reserves_[i] >= capacities_[i] || capacities_[i] % SRAM_ALIGNMENT ||
            reserves_[i] % SRAM_ALIGNMENT || capacities_[i] > SIZE_MAX - total_bytes_)
            throw std::invalid_argument("invalid WSE SRAM capacity/reserve/alignment");
        starts_.push_back(total_bytes_);
        total_bytes_ += capacities_[i];
    }
    if (DEFAULT_GLOBAL_MEMORY_BYTES > SIZE_MAX - total_bytes_)
        throw std::invalid_argument("invalid WSE global memory capacity");
}
void SramStorage::initialize() {
    std::scoped_lock lock(mutex_);
    if (local_) return;
    auto local = map(total_bytes_);
    auto global = map(DEFAULT_GLOBAL_MEMORY_BYTES);
    local_ = std::move(local);
    global_ = std::move(global);
}
MemoryWindow SramStorage::local_window_locked(uint32_t core) const {
    return {reinterpret_cast<uint64_t>(local_.get()) + starts_[core], capacities_[core] - reserves_[core]};
}
MemoryWindow SramStorage::window(int32_t core) const {
    std::scoped_lock lock(mutex_);
    if (!local_) throw std::logic_error("WSE fixed memory is not initialized");
    if (core == -1) return {reinterpret_cast<uint64_t>(global_.get()), DEFAULT_GLOBAL_MEMORY_BYTES};
    if (core < 0 || static_cast<size_t>(core) >= capacities_.size()) throw std::invalid_argument("invalid SRAM Core");
    return local_window_locked(core);
}
uint32_t SramStorage::core_count() const { return capacities_.size(); }
std::vector<uint64_t> SramStorage::configuration() const {
    std::scoped_lock lock(mutex_);
    std::vector<uint64_t> values{capacities_.size(), PE_PER_CORE, MAX_CORE_COUNT, 1};
    for (size_t core = 0; core < capacities_.size(); ++core) {
        values.push_back(capacities_[core]);
        values.push_back(reserves_[core]);
    }
    return values;
}
std::vector<uint64_t> SramStorage::info(uint32_t core) const {
    std::scoped_lock lock(mutex_);
    if (core >= capacities_.size()) throw std::invalid_argument("invalid SRAM Core");
    const uint64_t base = local_ ? local_window_locked(core).base : 0;
    const uint64_t available = capacities_[core] - reserves_[core];
    return {
        base,
        capacities_[core],
        reserves_[core],
        local_ ? available : 0,
        local_ ? 0 : available,
        local_ ? 0 : available,
        runtime_used_[core]
    };
}
void SramStorage::check_application(uint64_t address, uint64_t bytes) const {
    std::scoped_lock lock(mutex_);
    const auto contains = [&](MemoryWindow window) {
        return window.base && address >= window.base && address - window.base <= window.bytes &&
               bytes <= window.bytes - (address - window.base);
    };
    if (local_) {
        if (contains({reinterpret_cast<uint64_t>(global_.get()), DEFAULT_GLOBAL_MEMORY_BYTES})) return;
        const auto base = reinterpret_cast<uint64_t>(local_.get());
        if (address >= base && address - base < total_bytes_) {
            const auto next = std::upper_bound(starts_.begin(), starts_.end(), address - base);
            const auto core = static_cast<uint32_t>(next - starts_.begin() - 1);
            if (contains(local_window_locked(core))) return;
        }
    }
    throw std::invalid_argument("WSE copy range exceeds a fixed application memory window");
}
std::vector<SramAllocation> SramStorage::allocations() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramAllocation> result;
    if (!local_) return result;
    for (uint32_t core = 0; core < capacities_.size(); ++core) {
        const auto application = local_window_locked(core);
        result.push_back({application.base, application.bytes, core, 0});
    }
    return result;
}
std::vector<SramRegion> SramStorage::regions() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramRegion> result;
    if (!local_) return result;
    for (uint32_t core = 0; core < capacities_.size(); ++core) {
        const auto application = local_window_locked(core);
        result.push_back({application.base, capacities_[core], reserves_[core], application.base + application.bytes});
    }
    return result;
}
void SramStorage::set_runtime_used(const std::vector<uint64_t> &used) {
    std::scoped_lock lock(mutex_);
    if (used.size() != capacities_.size()) throw std::invalid_argument("SRAM runtime accounting shape");
    for (size_t i = 0; i < used.size(); ++i)
        if (used[i] > reserves_[i]) throw std::invalid_argument("SRAM runtime accounting overflow");
    runtime_used_ = used;
}
void SramStorage::reset_runtime_used() noexcept {
    std::scoped_lock lock(mutex_);
    for (auto &used : runtime_used_)
        used = 0;
}
size_t SramStorage::committed_bytes() const {
    std::scoped_lock lock(mutex_);
    return local_ ? total_bytes_ + DEFAULT_GLOBAL_MEMORY_BYTES : 0;
}
void SramStorage::clear() {
    std::scoped_lock lock(mutex_);
    global_.reset();
    local_.reset();
    for (auto &used : runtime_used_)
        used = 0;
}
}  // namespace wse
