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

#include "dist_ring.h"

#include <stdexcept>

void DistRing::init(int32_t window_size) {
    if (window_size <= 0 || (window_size & (window_size - 1)) != 0)
        throw std::invalid_argument("DistRing window_size must be a positive power of 2");
    window_size_ = window_size;
    window_mask_ = window_size - 1;
    next_task_id_ = 0;
    released_count_.store(0, std::memory_order_relaxed);
    shutdown_ = false;
}

DistTaskSlot DistRing::alloc() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
        if (shutdown_) return true;
        // Active = allocated - released.  Allow alloc when active < window_size.
        return (next_task_id_ - released_count_.load(std::memory_order_acquire)) < window_size_;
    });
    if (shutdown_) return DIST_INVALID_SLOT;
    int32_t task_id = next_task_id_++;
    return task_id & window_mask_;
}

void DistRing::release(DistTaskSlot /*slot*/) {
    // Simply count released slots.  Out-of-order release is safe: alloc() only
    // checks total active count, not which specific slots are free.
    released_count_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
}

int32_t DistRing::active_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return next_task_id_ - released_count_.load(std::memory_order_acquire);
}

void DistRing::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}
