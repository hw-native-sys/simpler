/**
 * Memory Allocator - Host Memory Management (Simulation)
 *
 * This module provides host memory management that simulates device memory.
 * Instead of using CANN runtime APIs (rtMalloc/rtFree), we use standard
 * malloc/free with tracking for cleanup.
 */

#ifndef RUNTIME_MEMORYALLOCATOR_H
#define RUNTIME_MEMORYALLOCATOR_H

#include <cstddef>
#include <set>

/**
 * MemoryAllocator class for managing host memory (simulating device memory)
 *
 * This class wraps standard malloc/free and provides automatic tracking
 * of allocations to prevent memory leaks. Uses RAII pattern for cleanup.
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
     * @param size  Size in bytes to allocate
     * @return Pointer on success, nullptr on failure
     */
    void* alloc(size_t size);

    /**
     * Free memory if tracked
     *
     * @param ptr  Pointer to free
     * @return 0 on success
     */
    int free(void* ptr);

    /**
     * Free all remaining tracked allocations
     *
     * @return 0 on success
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
