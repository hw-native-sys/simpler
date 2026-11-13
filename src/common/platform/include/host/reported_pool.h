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
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "host/memory_allocator.h"
#include "host/raii_scope_guard.h"
#include "runtime_c_api.h"

/**
 * Device regions owned by run epoch rather than pipeline slot.
 *
 * An epoch keeps its region until the caller proves last use, so a successor
 * reusing the slot never overwrites an unobserved result. A proved release does
 * not free immediately: the pool keeps one idle region for the next epoch, which
 * preserves both the allocation and any replay-evidence bytes already in it.
 * That single idle slot bounds the cache; every other released region is freed.
 */
class ReportedPool {
public:
    struct Lease {
        uint64_t run_epoch{0};
        uint32_t pipeline_slot{UINT32_MAX};
        void *device_ptr{nullptr};
        size_t bytes{0};
        bool initialized{false};
        // Non-zero on the nonnull output of a failed allocation, which stays
        // owned for the caller's checked retry.
        int allocation_rc{0};
    };

    explicit ReportedPool(MemoryAllocator &alloc) :
        alloc_(alloc) {}
    ReportedPool(const ReportedPool &) = delete;
    ReportedPool &operator=(const ReportedPool &) = delete;

    /**
     * Acquire this epoch's region. A repeated acquire for the same epoch
     * returns the same region; a new epoch reuses the idle region when it is
     * large enough and otherwise allocates one. The returned region carries the
     * cold-initialization state of the backing allocation, which a reused idle
     * region preserves across epochs.
     */
    int acquire(uint64_t run_epoch, uint32_t pipeline_slot, size_t bytes, Lease *out, bool *free_failure = nullptr);

    /** Record that the returned region's cold initialization completed. */
    void mark_initialized(const Lease &lease);

    /**
     * Release this epoch's ownership. Without `last_use_proved` the epoch keeps
     * its region. With proof, `keep_idle` moves the region into the idle cache
     * (or frees it when the cache already holds one); without `keep_idle` the
     * region is freed outright.
     */
    int release(const Lease &lease, bool last_use_proved, bool keep_idle = true);

    /** Free every region the pool still owns; a failed free stays for retry. */
    int finalize();

    /** Forget every region without freeing them (a device reset invalidated them). */
    void abandon();

    size_t live_count() const;
    size_t live_bytes() const;
    /** The idle region's address, or null when the cache is empty. */
    void *idle_device_ptr() const;
    bool idle_region_initialized() const;
    /** The first released region whose checked free still has to run. */
    void *pending_failed_device_ptr() const;

private:
    struct Region {
        uint32_t pipeline_slot{UINT32_MAX};
        void *device_ptr{nullptr};
        size_t bytes{0};
        bool initialized{false};
        int allocation_rc{0};
    };

    static bool matches(const Region &region, const Lease &lease) {
        return region.device_ptr == lease.device_ptr && region.bytes == lease.bytes;
    }

    static Lease lease_for(uint64_t run_epoch, const Region &region) {
        return Lease{run_epoch,    region.pipeline_slot, region.device_ptr,
                     region.bytes, region.initialized,   region.allocation_rc};
    }

    int free_region(Region region);

    MemoryAllocator &alloc_;
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Region> active_;
    std::optional<Region> idle_;
    std::vector<Region> failed_frees_;
};

inline int
ReportedPool::acquire(uint64_t run_epoch, uint32_t pipeline_slot, size_t bytes, Lease *out, bool *free_failure) {
    if (free_failure != nullptr) *free_failure = false;
    if (out == nullptr || run_epoch == 0 || bytes == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_.find(run_epoch);
    if (it != active_.end()) {
        if (it->second.bytes < bytes) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (it->second.allocation_rc == 0) {
            *out = lease_for(run_epoch, it->second);
            return 0;
        }
        // A failed allocation from an earlier attempt is freed before the
        // retry; a failed free keeps it owned so the next call retries.
        const int free_rc = free_region(it->second);
        if (free_rc != 0) {
            if (free_failure != nullptr) *free_failure = true;
            return free_rc;
        }
        active_.erase(it);
    }
    // Released runs whose failed allocation could not be freed are retried
    // before a new region is handed out. A pending region belongs to the slot
    // that allocated it, so its failed free does not block another slot.
    for (auto failed = failed_frees_.begin(); failed != failed_frees_.end();) {
        if (failed->pipeline_slot != pipeline_slot) {
            ++failed;
            continue;
        }
        const int free_rc = free_region(*failed);
        if (free_rc != 0) {
            if (free_failure != nullptr) *free_failure = true;
            return free_rc;
        }
        failed = failed_frees_.erase(failed);
    }
    // Reserve ownership bookkeeping before moving an idle region or allocating
    // device memory. Nothing after either transfer may allocate a map node.
    const auto entry = active_.try_emplace(run_epoch).first;
    auto rollback = RAIIScopeGuard([&] {
        active_.erase(entry);
    });
    Region &region = entry->second;
    if (idle_.has_value()) {
        if (idle_->bytes >= bytes) {
            region = *idle_;
            region.pipeline_slot = pipeline_slot;
            idle_.reset();
        } else {
            const int rc = free_region(*idle_);
            if (rc != 0) return rc;
            idle_.reset();
        }
    }
    if (region.device_ptr == nullptr) {
        int status = 0;
        void *ptr = alloc_.alloc(bytes, &status);
        if (ptr == nullptr) return status != 0 ? status : PTO_RUNTIME_ERR_INTERNAL;
        region = Region{pipeline_slot, ptr, bytes, false, status};
        rollback.dismiss();
        *out = lease_for(run_epoch, region);
        return status;
    }
    rollback.dismiss();
    *out = lease_for(run_epoch, region);
    return 0;
}

inline void ReportedPool::mark_initialized(const Lease &lease) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_.find(lease.run_epoch);
    if (it != active_.end() && matches(it->second, lease)) {
        it->second.initialized = true;
        return;
    }
    if (idle_.has_value() && matches(*idle_, lease)) idle_->initialized = true;
}

inline int ReportedPool::release(const Lease &lease, bool last_use_proved, bool keep_idle) {
    if (!last_use_proved) return 0;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_.find(lease.run_epoch);
    if (it == active_.end() || !matches(it->second, lease)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (it->second.allocation_rc != 0) {
        // A failed allocation is not a valid result; keep it pending so the
        // next acquire performs the checked free.
        failed_frees_.push_back(it->second);
        active_.erase(it);
        return 0;
    }
    if (!keep_idle) {
        const int rc = free_region(it->second);
        if (rc != 0) return rc;
        active_.erase(it);
        return 0;
    }
    if (!idle_.has_value()) {
        idle_ = it->second;
        active_.erase(it);
        return 0;
    }
    // A failed free leaves the epoch owning its region, so the caller's retry
    // releases the same lease instead of losing it.
    const int rc = free_region(it->second);
    if (rc != 0) return rc;
    active_.erase(it);
    return 0;
}

inline int ReportedPool::free_region(Region region) {
    if (region.device_ptr == nullptr) return 0;
    return alloc_.free(region.device_ptr);
}

inline int ReportedPool::finalize() {
    std::lock_guard<std::mutex> lock(mu_);
    int first_error = 0;
    auto release_one = [&](Region &region) {
        const int rc = free_region(region);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
            return false;
        }
        region.device_ptr = nullptr;
        region.bytes = 0;
        return true;
    };
    for (auto it = active_.begin(); it != active_.end();) {
        if (release_one(it->second)) it = active_.erase(it);
        else ++it;
    }
    if (idle_.has_value() && release_one(*idle_)) idle_.reset();
    for (auto it = failed_frees_.begin(); it != failed_frees_.end();) {
        if (release_one(*it)) it = failed_frees_.erase(it);
        else ++it;
    }
    return first_error;
}

inline void ReportedPool::abandon() {
    std::lock_guard<std::mutex> lock(mu_);
    active_.clear();
    idle_.reset();
    failed_frees_.clear();
}

inline void *ReportedPool::idle_device_ptr() const {
    std::lock_guard<std::mutex> lock(mu_);
    return idle_.has_value() ? idle_->device_ptr : nullptr;
}

inline bool ReportedPool::idle_region_initialized() const {
    std::lock_guard<std::mutex> lock(mu_);
    return idle_.has_value() && idle_->initialized;
}

inline void *ReportedPool::pending_failed_device_ptr() const {
    std::lock_guard<std::mutex> lock(mu_);
    return failed_frees_.empty() ? nullptr : failed_frees_.front().device_ptr;
}

inline size_t ReportedPool::live_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return active_.size() + (idle_.has_value() ? 1u : 0u) + failed_frees_.size();
}

inline size_t ReportedPool::live_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t bytes = 0;
    for (const auto &entry : active_)
        bytes += entry.second.bytes;
    if (idle_.has_value()) bytes += idle_->bytes;
    for (const auto &entry : failed_frees_)
        bytes += entry.bytes;
    return bytes;
}
