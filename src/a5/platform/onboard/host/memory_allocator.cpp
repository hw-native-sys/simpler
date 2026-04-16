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
 * Memory Allocator Implementation
 *
 * This file implements centralized device memory management using the
 * Ascend CANN runtime API with RAII pattern.
 */

#include "host/memory_allocator.h"

#include <runtime/rt.h>

#include "common/unified_log.h"

MemoryAllocator::~MemoryAllocator() { finalize(); }

void *MemoryAllocator::alloc(size_t size) {
    void *ptr = nullptr;
    int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        LOG_ERROR("rtMalloc failed: %d (size=%zu)", rc, size);
        return nullptr;
    }

    std::lock_guard<std::mutex> lk(mu_);
    ptr_set_.insert(ptr);
    return ptr;
}

int MemoryAllocator::free(void *ptr) {
    if (ptr == nullptr) {
        return 0;
    }

    std::lock_guard<std::mutex> lk(mu_);
    auto it = ptr_set_.find(ptr);
    if (it == ptr_set_.end()) {
        return 0;
    }

    int rc = rtFree(ptr);
    if (rc != 0) {
        LOG_ERROR("rtFree failed: %d", rc);
        return rc;
    }

    ptr_set_.erase(it);
    return 0;
}

int MemoryAllocator::finalize() {
    std::lock_guard<std::mutex> lk(mu_);
    int last_error = 0;
    for (void *ptr : ptr_set_) {
        int rc = rtFree(ptr);
        if (rc != 0) {
            LOG_ERROR("rtFree failed during Finalize: %d", rc);
            last_error = rc;
        }
    }
    ptr_set_.clear();
    return last_error;
}
