/**
 * Memory Allocator Implementation
 *
 * This file implements centralized device memory management using the
 * Ascend CANN runtime API with RAII pattern.
 */

#include "host/memory_allocator.h"

#include <runtime/rt.h>

#include "common/unified_log.h"

MemoryAllocator::~MemoryAllocator() { finalize(); }

void* MemoryAllocator::alloc(size_t size) {
    void* ptr = nullptr;
    int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        LOG_ERROR("rtMalloc failed: %d (size=%zu)", rc, size);
        return nullptr;
    }

    // Track the pointer
    ptr_set_.insert(ptr);
    return ptr;
}

int MemoryAllocator::free(void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }

    // Check if we're tracking this pointer
    auto it = ptr_set_.find(ptr);
    if (it == ptr_set_.end()) {
        // Not tracked by us, don't free
        return 0;
    }

    // Free the memory
    int rc = rtFree(ptr);
    if (rc != 0) {
        LOG_ERROR("rtFree failed: %d", rc);
        return rc;
    }

    // Remove from tracking set
    ptr_set_.erase(it);
    return 0;
}

int MemoryAllocator::finalize() {
    // Idempotent - safe to call multiple times
    if (finalized_) {
        return 0;
    }

    int last_error = 0;

    // Free all remaining tracked pointers
    for (void* ptr : ptr_set_) {
        int rc = rtFree(ptr);
        if (rc != 0) {
            LOG_ERROR("rtFree failed during Finalize: %d", rc);
            last_error = rc;
        }
    }

    // Clear the set
    ptr_set_.clear();
    finalized_ = true;

    return last_error;
}
