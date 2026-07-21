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
    eligible_worker_ids.clear();
    fanin_producers.clear();
    failure_message.clear();
    worker_type = WorkerType::NEXT_LEVEL;
    callable = CallableIdentity{};
    config = CallConfig{};
    task_args.clear();
    task_args_list.clear();
    is_group_ = false;
    remote_sidecar.clear();
    remote_sidecars.clear();
    affinities.clear();
    // ring_idx / ring_slot_idx are deliberately NOT cleared here: Ring
    // stamps them at alloc() before the Orchestrator ever calls reset(),
    // and Ring::release() needs to read them for the FIFO advance. The
    // fields are rewritten on every alloc, so stale values never escape.
    sub_complete_count.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(group_mu);
        group_member_states.clear();
        group_member_outcomes.clear();
        group_failed = false;
        group_first_failure_index = -1;
        group_first_failure_message.clear();
    }
    group_terminal_count.store(0, std::memory_order_relaxed);
    group_dispatched_count.store(0, std::memory_order_relaxed);
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

bool ReadyQueue::empty() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.empty();
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
