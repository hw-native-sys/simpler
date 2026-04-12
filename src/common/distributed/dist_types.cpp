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

#include "dist_types.h"

#include "dist_orchestrator.h"

// =============================================================================
// DistTaskSlotState
// =============================================================================

void DistTaskSlotState::reset() {
    state.store(TaskState::FREE, std::memory_order_relaxed);
    fanin_count = 0;
    fanin_released.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(fanout_mu);
        fanout_consumers.clear();
        fanout_total = 0;
    }
    fanout_released.store(0, std::memory_order_relaxed);
    for (size_t i = 0; i < output_bufs.size(); ++i) {
        if (output_ownerships[i] == static_cast<int32_t>(DistOutputOwnership::ALLOCATED)) {
            ::operator delete(output_bufs[i]);
        }
    }
    output_bufs.clear();
    output_sizes.clear();
    output_ownerships.clear();
    output_keys.clear();
    fanin_producers.clear();
    payload = WorkerPayload{};
    args_list.clear();
    sub_complete_count.store(0, std::memory_order_relaxed);
}

// =============================================================================
// DistReadyQueue
// =============================================================================

void DistReadyQueue::push(DistTaskSlot slot) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push(slot);
    }
    cv_.notify_one();
}

bool DistReadyQueue::try_pop(DistTaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = q_.front();
    q_.pop();
    return true;
}

bool DistReadyQueue::wait_pop(DistTaskSlot &out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
        return !q_.empty() || shutdown_;
    });
    if (q_.empty()) return false;
    out = q_.front();
    q_.pop();
    return true;
}

void DistReadyQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}
