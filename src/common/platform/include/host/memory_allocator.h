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
 * - sim: Host memory management using standard malloc/free
 *
 * Key Features:
 * - Automatic tracking of all allocated memory
 * - Safe deallocation with existence checking
 * - Automatic cleanup via destructor (RAII pattern)
 * - Idempotent finalize() for explicit cleanup with error checking
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_MEMORY_ALLOCATOR_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <mutex>
#include <unordered_map>

/**
 * MemoryAllocator class for managing memory allocations
 *
 * Platform Behavior:
 * - onboard: Wraps CANN runtime memory allocation APIs (rtMalloc/rtFree)
 * - sim: Wraps standard malloc/free
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
     * - sim: Allocates host memory using malloc
     *
     * @param size  Size in bytes to allocate
     * @return Memory pointer on success, nullptr on failure
     */
    void *alloc(size_t size);

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
     * Sums the requested sizes of all currently-tracked allocations. Used by
     * downstream runtimes to account for device HBM this process has reserved
     * (which may be invisible to aclrtGetMemInfo) when sizing their own caches.
     *
     * @return Sum of the requested sizes of all currently-tracked allocations
     */
    size_t committed_bytes() const {
        std::scoped_lock lk(mu_);
        return committed_bytes_;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<void *, size_t> ptr_size_map_;
    size_t committed_bytes_ = 0;
};

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_MEMORY_ALLOCATOR_H_
