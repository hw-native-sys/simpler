/**
 * @file memory_allocator.h
 * @brief Memory Allocator - Centralized Memory Management
 *
 * This module provides centralized management of memory allocations with
 * automatic tracking and cleanup to prevent memory leaks.
 *
 * Platform Support:
 * - a2a3: Device memory management using CANN runtime API (rtMalloc/rtFree)
 * - a2a3sim: Host memory management using standard malloc/free
 *
 * Key Features:
 * - Automatic tracking of all allocated memory
 * - Safe deallocation with existence checking
 * - Automatic cleanup via destructor (RAII pattern)
 * - Idempotent finalize() for explicit cleanup with error checking
 */

#ifndef PLATFORM_MEMORY_ALLOCATOR_H_
#define PLATFORM_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <set>

/**
 * MemoryAllocator class for managing memory allocations
 *
 * Platform Behavior:
 * - a2a3: Wraps CANN runtime memory allocation APIs (rtMalloc/rtFree)
 * - a2a3sim: Wraps standard malloc/free
 *
 * Both implementations provide automatic tracking of allocations to
 * prevent memory leaks. Uses RAII pattern for automatic cleanup.
 */
class MemoryAllocator {
public:
    MemoryAllocator() = default;
    ~MemoryAllocator();

    // Prevent copying
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;

    /**
     * Allocate memory and track the pointer
     *
     * Platform-specific behavior:
     * - a2a3: Allocates device memory using rtMalloc
     * - a2a3sim: Allocates host memory using malloc
     *
     * @param size  Size in bytes to allocate
     * @return Memory pointer on success, nullptr on failure
     */
    void* alloc(size_t size);

    /**
     * Free memory if tracked
     *
     * Checks if the pointer exists in the tracking set. If found, frees the
     * memory and removes it from the set. Safe to call with nullptr or
     * untracked pointers.
     *
     * Platform-specific behavior:
     * - a2a3: Frees device memory using rtFree
     * - a2a3sim: Frees host memory using free
     *
     * @param ptr  Memory pointer to free
     * @return 0 on success, error code on failure, 0 if ptr not tracked
     */
    int free(void* ptr);

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
     * Get number of tracked allocations
     *
     * @return Number of currently tracked pointers
     */
    size_t get_allocation_count() const { return ptr_set_.size(); }

private:
    std::set<void*> ptr_set_;
    bool finalized_{false};
};

#endif  // PLATFORM_MEMORY_ALLOCATOR_H_
