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

#include "types.h"

// =============================================================================
// TaskSlotState
// =============================================================================

void TaskSlotState::reset() {
    state.store(TaskState::FREE, std::memory_order_relaxed);
    fanin_count = 0;
    fanin_released.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(fanout_mu);
        fanout_consumers.clear();
        fanout_total = 0;
    }
    fanout_released.store(0, std::memory_order_relaxed);
    output_keys.clear();
    fanin_producers.clear();
    worker_type = WorkerType::NEXT_LEVEL;
    callable = 0;
    callable_id = -1;
    config = ChipCallConfig{};
    task_args.clear();
    task_args_list.clear();
    is_group_ = false;
    // ring_idx / ring_slot_idx are deliberately NOT cleared here: Ring
    // stamps them at alloc() before the Orchestrator ever calls reset(),
    // and Ring::release() needs to read them for the FIFO advance. The
    // fields are rewritten on every alloc, so stale values never escape.
    sub_complete_count.store(0, std::memory_order_relaxed);
}

// =============================================================================
// ReadyQueue
// =============================================================================

void ReadyQueue::push(TaskSlot slot) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(slot);
    }
    cv_.notify_one();
}

bool ReadyQueue::try_pop(TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = q_.front();
    q_.pop();
    return true;
}

bool ReadyQueue::wait_pop(TaskSlot &out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
        return !q_.empty() || shutdown_;
    });
    if (q_.empty()) return false;
    out = q_.front();
    q_.pop();
    return true;
}

void ReadyQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}
