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
#include "wse_hardware_defaults.h"
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace wse {
void SramStorage::Free::operator()(std::byte *p) const noexcept { std::free(p); }
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
        total_bytes_ += capacities_[i];
        application_bytes_ += capacities_[i] - reserves_[i];
    }
}
void SramStorage::initialize_locked() {
    if (backing_) return;
    void *ptr = nullptr;
    if (posix_memalign(&ptr, SRAM_ALIGNMENT, total_bytes_) != 0) throw std::bad_alloc();
    backing_.reset(static_cast<std::byte *>(ptr));
}
void SramStorage::freeze() {
    std::scoped_lock lock(mutex_);
    initialize_locked();
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
    uint64_t base = reinterpret_cast<uint64_t>(backing_.get());
    if (base)
        for (uint32_t i = 0; i < core; ++i)
            base += capacities_[i] - reserves_[i];
    const uint64_t available = capacities_[core] - reserves_[core];
    return {
        base,
        capacities_[core],
        reserves_[core],
        claimed_ ? available : 0,
        claimed_ ? 0 : available,
        claimed_ ? 0 : available,
        runtime_used_[core]
    };
}
void *SramStorage::claim(size_t bytes) {
    std::scoped_lock lock(mutex_);
    if (bytes != application_bytes_)
        throw std::invalid_argument("WSE malloc requires the complete application SRAM size");
    if (claimed_) throw std::invalid_argument("WSE application SRAM already has a live Buffer");
    initialize_locked();
    claimed_ = true;
    return backing_.get();
}
bool SramStorage::owns(const void *ptr) const {
    std::scoped_lock lock(mutex_);
    return ptr && ptr == backing_.get();
}
void SramStorage::release(void *ptr) {
    std::scoped_lock lock(mutex_);
    if (!claimed_ || ptr != backing_.get())
        throw std::invalid_argument("WSE free requires its live application Buffer");
    claimed_ = false;
}
void SramStorage::check_application(uint64_t address, uint64_t bytes) const {
    std::scoped_lock lock(mutex_);
    const auto base = reinterpret_cast<uint64_t>(backing_.get());
    if (!claimed_ || address < base || address - base > application_bytes_ ||
        bytes > application_bytes_ - (address - base))
        throw std::invalid_argument("SRAM access is outside the live application Buffer");
}
std::vector<SramAllocation> SramStorage::allocations() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramAllocation> result;
    if (!claimed_) return result;
    uint64_t base = reinterpret_cast<uint64_t>(backing_.get());
    for (uint32_t core = 0; core < capacities_.size(); ++core) {
        const auto bytes = capacities_[core] - reserves_[core];
        result.push_back({base, bytes, core, 0});
        base += bytes;
    }
    return result;
}
std::vector<SramRegion> SramStorage::regions() const {
    std::scoped_lock lock(mutex_);
    std::vector<SramRegion> result;
    if (!backing_) return result;
    uint64_t base = reinterpret_cast<uint64_t>(backing_.get());
    uint64_t runtime = base + application_bytes_;
    for (size_t core = 0; core < capacities_.size(); ++core) {
        result.push_back({base, capacities_[core], reserves_[core], runtime});
        base += capacities_[core] - reserves_[core];
        runtime += reserves_[core];
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
    return backing_ ? total_bytes_ : 0;
}
void SramStorage::clear() {
    std::scoped_lock lock(mutex_);
    claimed_ = false;
    backing_.reset();
    for (auto &used : runtime_used_)
        used = 0;
}
}  // namespace wse
