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
/**
 * @file memory_allocator.h
 * @brief Memory Allocator - Centralized Memory Management
 *
 * This module provides centralized management of memory allocations with
 * automatic tracking and cleanup to prevent memory leaks.
 *
 * Platform Support (same shape for both arches):
 * - onboard: Device memory management using CANN runtime API (rtMalloc/rtFree)
 * - sim: Host memory management using 64-byte-aligned allocation and free
 *
 * Key Features:
 * - Automatic tracking of all allocated memory
 * - Safe deallocation with existence checking
 * - Automatic cleanup via destructor (RAII pattern)
 * - Idempotent finalize() for explicit cleanup with error checking
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

/**
 * MemoryAllocator class for managing memory allocations
 *
 * Platform Behavior:
 * - onboard: Wraps CANN runtime memory allocation APIs (rtMalloc/rtFree)
 * - sim: Wraps 64-byte-aligned host allocation and free
 *
 * Both implementations provide automatic tracking of allocations to
 * prevent memory leaks. Uses RAII pattern for automatic cleanup.
 */
class MemoryAllocator {
public:
    MemoryAllocator() = default;
    ~MemoryAllocator();

    // Prevent copying
    MemoryAllocator(const MemoryAllocator &) = delete;
    MemoryAllocator &operator=(const MemoryAllocator &) = delete;

    /**
     * Allocate memory and track the pointer
     *
     * Platform-specific behavior:
     * - onboard: Allocates device memory using rtMalloc
     * - sim: Allocates 64-byte-aligned host memory
     *
     * @param size  Size in bytes to allocate
     * @param status Optional raw platform status. When supplied, return any
     *               nonnull output even on error so the caller can retain it.
     * @return The allocation on success. Without status, failure returns null.
     *         Every nonnull platform output is tracked for checked cleanup;
     *         an error does not establish that its contents are usable.
     */
    void *alloc(size_t size, int *status = nullptr);

    /**
     * Free memory if tracked
     *
     * Checks if the pointer exists in the tracking set. If found, frees the
     * memory and removes it from the set. Safe to call with nullptr or
     * untracked pointers.
     *
     * Platform-specific behavior:
     * - onboard: Frees device memory using rtFree
     * - sim: Frees host memory using free
     *
     * @param ptr  Memory pointer to free
     * @return 0 on success, error code on failure, 0 if ptr not tracked
     */
    int free(void *ptr);

    /** Release adopted storage and its metadata; a nonzero status retains allocator ownership. */
    using StorageReleaseFn = int (*)(void *storage, void *base);

    /**
     * Take ownership only on success. Logical bytes bound views; physical bytes include VMM padding.
     * The release callback runs under the allocator lock and must not reenter this allocator.
     */
    bool
    adopt(void *base, size_t logical_bytes, size_t physical_bytes, StorageReleaseFn release_storage, void *storage) {
        std::scoped_lock lock(mu_);
        if (base == nullptr || logical_bytes == 0 || physical_bytes < logical_bytes || release_storage == nullptr ||
            ptr_size_map_.count(base)) {
            return false;
        }
        ptr_size_map_.emplace(base, logical_bytes);
        try {
            adopted_.emplace(base, AdoptedStorage{physical_bytes, release_storage, storage});
        } catch (...) {
            ptr_size_map_.erase(base);
            throw;
        }
        committed_bytes_ += physical_bytes;
        return true;
    }

    /** Which action a sweep took on one tracked allocation. */
    enum class SweepAction : uint32_t { FreeIt = 0, KeepIt = 1 };

    /**
     * Whether the sweep may release this allocation. Called under the
     * allocator's lock, once per tracked allocation, before anything is freed.
     * An allocation the callback does not recognise is freed, which is what
     * every allocation outside the workspace ledger needs.
     */
    using SweepClassifyFn = SweepAction (*)(void *base, size_t bytes, void *ctx);

    /**
     * Where one allocation's sweep outcome is recorded. Called under the same
     * lock, immediately after the action and therefore before the tracking map
     * is cleared: the map carries no result field, so a caller that needs the
     * outcome has to receive it here, into storage it already owns.
     */
    using SweepRecordFn = void (*)(void *base, int rc, SweepAction acted, void *ctx);

    /**
     * A tracking slot taken before the platform allocation it will describe.
     *
     * Holds the allocator's lock and a pre-created map node for its whole
     * lifetime, so the capacity it reserved cannot be consumed by another
     * writer between the reservation and the insertion, and the bookkeeping
     * commit that follows a successful platform allocation needs no further
     * host allocation. Destroying it without committing cancels: the node is
     * returned and nothing was changed.
     *
     * Move-only, and the platform allocation happens inside the held lock —
     * which serializes this allocator's own bookkeeping, and is why the caller
     * must not hold a reservation across unrelated work.
     */
    class Reservation {
    public:
        Reservation() = default;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        Reservation(Reservation &&) noexcept = default;
        Reservation &operator=(Reservation &&) noexcept = default;

        bool valid() const noexcept { return owner_ != nullptr && !node_.empty(); }

        /**
         * Allocate `size` bytes and commit this slot's bookkeeping.
         *
         * @param status Optional raw platform status, with the same error-output
         *               ownership contract as alloc().
         * @return the allocation on success; null on error when status is absent.
         *         A nonnull error output is tracked. An invalid reservation or
         *         null platform output commits no allocation.
         */
        void *commit_alloc(size_t size, int *status = nullptr);

    private:
        friend class MemoryAllocator;
        MemoryAllocator *owner_{nullptr};
        std::unique_lock<std::mutex> lock_;
        std::unordered_map<void *, size_t>::node_type node_;
    };

    /**
     * Take a tracking slot for one upcoming allocation.
     *
     * May throw (it reserves bucket capacity and creates a map node); performs
     * no platform allocation, so a throw here leaves no allocation behind.
     */
    Reservation begin_reservation() {
        Reservation res;
        res.lock_ = std::unique_lock<std::mutex>(mu_);
        ptr_size_map_.reserve(ptr_size_map_.size() + 1);
        // The sentinel is created and extracted under the lock this reservation
        // keeps, so it is never observable in the map. Its value is not a
        // possible allocation: no platform allocator returns address 1.
        void *sentinel = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        auto [it, inserted] = ptr_size_map_.try_emplace(sentinel, 0);
        if (!inserted) return res;  // invalid: commit_alloc refuses rather than allocating
        res.node_ = ptr_size_map_.extract(it);
        res.owner_ = this;
        return res;
    }

    /**
     * Release adopted storage first, then sweep ordinary allocations through the classifier.
     * Failed or kept adopted storage retains its handles/accounting and stops this sweep.
     *
     * The ordinary-allocation sweep is terminal: its map is cleared even if a platform free
     * failed, so no later free path — a second close, the
     * destructor — can reach an address this call has already accounted for.
     * A kept allocation is not freed and its bytes move to the relinquished
     * total, which is what keeps a caller that could not prove its last
     * consumer finished from handing those bytes back to the platform.
     *
     * Records reach the caller through `record` before the clear. Diagnostics
     * are emitted after it, so a logging failure cannot leave the map holding
     * addresses this call has already decided about.
     *
     * @return the last platform free error, or 0
     */
    int finalize_except(SweepClassifyFn classify, SweepRecordFn record, void *ctx);

    /** Bytes forgotten without being freed by `finalize_except`. */
    uint64_t relinquished_bytes() const {
        std::scoped_lock lk(mu_);
        return relinquished_bytes_;
    }

    /**
     * Free all remaining tracked allocations
     *
     * Iterates through all tracked pointers, frees them, and clears the
     * tracking set. Can be called explicitly for error checking, or
     * automatically via destructor. Idempotent - safe to call multiple times.
     *
     * @return 0 on success, error code if any frees failed
     */
    int finalize();

    /**
     * Forget tracked device pointers without calling the platform free API.
     *
     * Use only after a device reset has already invalidated every allocation,
     * or when a fatal device state makes further runtime calls unsafe. This
     * makes the RAII destructor a no-op while leaving healthy teardown on the
     * normal finalize() path.
     */
    void abandon_after_device_failure() {
        std::scoped_lock lk(mu_);
        adopted_.clear();
        ptr_size_map_.clear();
        committed_bytes_ = 0;
    }

    /**
     * Resolve an address to the tracked allocation that contains it.
     *
     * An address inside an allocation is not itself a key: a tensor may sit at
     * an offset into a larger buffer. Callers that must act on the allocation
     * as a whole — a host mapping, which is established and released per
     * allocation — recover its base and size here.
     *
     * @param ptr   Any address, allocation base or interior
     * @param base  Receives the containing allocation's base
     * @param size  Receives its requested size
     * @return true when a tracked allocation covers `ptr`
     */
    bool owning_allocation(const void *ptr, void **base, size_t *size) const {
        if (ptr == nullptr) return false;
        const auto *addr = static_cast<const unsigned char *>(ptr);
        std::scoped_lock lk(mu_);
        for (const auto &[alloc_ptr, alloc_size] : ptr_size_map_) {
            const auto *alloc_addr = static_cast<const unsigned char *>(alloc_ptr);
            if (addr < alloc_addr || addr >= alloc_addr + alloc_size) continue;
            if (base != nullptr) *base = alloc_ptr;
            if (size != nullptr) *size = alloc_size;
            return true;
        }
        return false;
    }

    /**
     * Get number of tracked allocations
     *
     * @return Number of currently tracked pointers
     */
    size_t get_allocation_count() const {
        std::scoped_lock lk(mu_);
        return ptr_size_map_.size();
    }

    /**
     * Get total bytes currently committed by this allocator
     *
     * Sums committed physical bytes, including VMM padding. Used by
     * downstream runtimes to account for device HBM this process has reserved
     * (which may be invisible to aclrtGetMemInfo) when sizing their own caches.
     *
     * @return Physical bytes committed by currently tracked allocations
     */
    size_t committed_bytes() const {
        std::scoped_lock lk(mu_);
        return committed_bytes_;
    }

private:
    struct AdoptedStorage {
        size_t physical_bytes;
        StorageReleaseFn release_storage;
        void *storage;
    };

    int release_adopted_locked(void *base) {
        const auto &record = adopted_.at(base);
        const int status = record.release_storage(record.storage, base);
        if (status != 0) return status;
        committed_bytes_ -= record.physical_bytes;
        adopted_.erase(base);
        ptr_size_map_.erase(base);
        return 0;
    }

    int
    sweep_adopted_locked(SweepClassifyFn classify = nullptr, SweepRecordFn record = nullptr, void *context = nullptr) {
        while (!adopted_.empty()) {
            void *base = adopted_.begin()->first;
            if (classify != nullptr && classify(base, ptr_size_map_.at(base), context) == SweepAction::KeepIt) {
                return -1;
            }
            const int status = release_adopted_locked(base);
            if (record != nullptr) record(base, status, SweepAction::FreeIt, context);
            if (status != 0) return status;
        }
        return 0;
    }

    std::unordered_map<void *, AdoptedStorage> adopted_;
    mutable std::mutex mu_;
    std::unordered_map<void *, size_t> ptr_size_map_;
    size_t committed_bytes_ = 0;
    uint64_t relinquished_bytes_ = 0;
};
