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
 * DistRing — task slot allocator with back-pressure.
 *
 * Maintains a circular window of slots.  The Orchestrator calls alloc() to
 * claim the next slot.  The Scheduler calls release() when a task reaches
 * CONSUMED.
 *
 * Back-pressure: alloc() blocks when all slots are occupied.
 * Out-of-order release: tracked via released_count_ (total released so far).
 * alloc() checks (next_task_id_ - released_count_) < window_size_.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include "dist_types.h"

class DistRing {
public:
    void init(int32_t window_size = DIST_TASK_WINDOW_SIZE);

    // Allocate next slot.  Blocks until space is available.
    // Returns the slot index (task_id % window_size).
    DistTaskSlot alloc();

    // Release slot.  Called by Scheduler when task reaches CONSUMED.
    // Safe for out-of-order release.
    void release(DistTaskSlot slot);

    int32_t window_size() const { return window_size_; }
    int32_t active_count() const;

    void shutdown();

private:
    int32_t window_size_{DIST_TASK_WINDOW_SIZE};
    int32_t window_mask_{DIST_TASK_WINDOW_SIZE - 1};
    int32_t next_task_id_{0};                 // orch-only, no atomic needed
    std::atomic<int32_t> released_count_{0};  // total slots released (any order)

    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
};
