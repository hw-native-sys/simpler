/**
 * Memory Allocator Implementation (Simulation)
 *
 * Uses standard malloc/free to simulate device memory operations.
 */

#include "memory_allocator.h"

#include <cstdlib>
#include <iostream>

MemoryAllocator::~MemoryAllocator() {
    finalize();
}

void* MemoryAllocator::alloc(size_t size) {
    void* ptr = std::malloc(size);
    if (ptr == nullptr) {
        std::cerr << "Error: malloc failed (size=" << size << ")\n";
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
    std::free(ptr);

    // Remove from tracking set
    ptr_set_.erase(it);
    return 0;
}

int MemoryAllocator::finalize() {
    // Idempotent - safe to call multiple times
    if (finalized_) {
        return 0;
    }

    // Free all remaining tracked pointers
    for (void* ptr : ptr_set_) {
        std::free(ptr);
    }

    // Clear the set
    ptr_set_.clear();
    finalized_ = true;

    return 0;
}
