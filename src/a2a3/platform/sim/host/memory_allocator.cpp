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
 * Memory Allocator Implementation (Simulation)
 *
 * Uses standard malloc/free to simulate device memory operations.
 */

#include "host/memory_allocator.h"

#include <cstdlib>
#include "common/unified_log.h"

MemoryAllocator::~MemoryAllocator() { finalize(); }

void *MemoryAllocator::alloc(size_t size) {
    void *ptr = std::malloc(size);
    if (ptr == nullptr) {
        LOG_ERROR("malloc failed (size=%zu)", size);
        return nullptr;
    }

    // Track the pointer
    ptr_set_.insert(ptr);
    return ptr;
}

int MemoryAllocator::free(void *ptr) {
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
    for (void *ptr : ptr_set_) {
        std::free(ptr);
    }

    // Clear the set
    ptr_set_.clear();
    finalized_ = true;

    return 0;
}
