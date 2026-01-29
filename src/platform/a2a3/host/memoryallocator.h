/**
 * Memory Allocator - Centralized Device Memory Management
 *
 * This module provides centralized management of device memory allocations
 * using the Ascend CANN runtime API. It tracks all allocated pointers and
 * ensures proper cleanup, preventing memory leaks.
 *
 * Key Features:
 * - Automatic tracking of all allocated device memory
 * - Safe deallocation with existence checking
 * - Automatic cleanup via destructor (RAII pattern)
 * - Idempotent finalize() for explicit cleanup with error checking
 */

#ifndef RUNTIME_MEMORYALLOCATOR_H
#define RUNTIME_MEMORYALLOCATOR_H

#include <cstddef>
#include <set>

/**
 * MemoryAllocator class for managing device memory
 *
 * This class wraps the CANN runtime memory allocation APIs (rtMalloc/rtFree)
 * and provides automatic tracking of allocations to prevent memory leaks.
 * Uses RAII pattern for automatic cleanup.
 */
class MemoryAllocator {
public:
    MemoryAllocator() = default;
    ~MemoryAllocator();

    // Prevent copying
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;

    /**
     * Allocate device memory and track the pointer
     *
     * Allocates device memory using rtMalloc and stores the pointer in the
     * tracking set for automatic cleanup.
     *
     * @param size  Size in bytes to allocate
     * @return Device pointer on success, nullptr on failure
     */
    void* alloc(size_t size);

    /**
     * Free device memory if tracked
     *
     * Checks if the pointer exists in the tracking set. If found, frees the
     * memory using rtFree and removes it from the set. Safe to call with
     * nullptr or untracked pointers.
     *
     * @param ptr  Device pointer to free
     * @return 0 on success, error code on failure, 0 if ptr not tracked
     */
    int free(void* ptr);

    /**
     * Free all remaining tracked allocations
     *
     * Iterates through all tracked pointers, frees them using rtFree, and
     * clears the tracking set. Can be called explicitly for error checking,
     * or automatically via destructor. Idempotent - safe to call multiple
     * times.
     *
     * @return 0 on success, error code if any frees failed
     */
    int finalize();

    /**
     * Get number of tracked allocations
     *
     * @return Number of currently tracked pointers
     */
    size_t get_allocation_count() const { return ptr_set_.size(); }

private:
    std::set<void*> ptr_set_;
    bool finalized_{false};
};

#endif  // RUNTIME_MEMORYALLOCATOR_H
