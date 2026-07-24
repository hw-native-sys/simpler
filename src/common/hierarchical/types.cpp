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

#include <stdexcept>

// =============================================================================
// TaskSlotState
// =============================================================================

void TaskSlotState::reset() {
    state.store(TaskState::FREE, std::memory_order_relaxed);
    run_id = INVALID_RUN_ID;
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
    failure_message.clear();
    worker_type = WorkerType::NEXT_LEVEL;
    callable = CallableIdentity{};
    config = CallConfig{};
    task_args.clear();
    task_args_list.clear();
    is_group_ = false;
    remote_sidecar.clear();
    remote_sidecars.clear();
    target_worker_ids.clear();
    // ring_idx / ring_slot_idx are deliberately NOT cleared here: Ring
    // stamps them at alloc() before the Orchestrator ever calls reset(),
    // and Ring::release() needs to read them for the FIFO advance. The
    // fields are rewritten on every alloc, so stale values never escape.
    {
        std::lock_guard<std::mutex> lk(group_mu);
        group_member_states.clear();
        group_member_outcomes.clear();
        group_failed = false;
        group_first_failure_index = -1;
        group_first_failure_message.clear();
    }
    group_terminal_count.store(0, std::memory_order_relaxed);
}

// =============================================================================
// ReadyQueue
// =============================================================================

void ReadyQueue::push(TaskSlot slot) {
    std::lock_guard<std::mutex> lk(mu_);
    q_.push(slot);
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

bool ReadyQueue::try_front(TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return false;
    out = q_.front();
    return true;
}

// =============================================================================
// NextLevelReadyQueues
// =============================================================================

void NextLevelReadyQueues::reset(const std::vector<int32_t> &worker_ids) {
    worker_ids_.clear();
    queues_.clear();
    worker_ids_.reserve(worker_ids.size());
    queues_.reserve(worker_ids.size());
    for (int32_t worker_id : worker_ids) {
        if (worker_id < 0) throw std::invalid_argument("NextLevelReadyQueues::reset: negative worker id");
        for (int32_t existing : worker_ids_) {
            if (existing == worker_id) {
                throw std::invalid_argument("NextLevelReadyQueues::reset: duplicate worker id");
            }
        }
        worker_ids_.push_back(worker_id);
        queues_.push_back(std::make_unique<ReadyQueue>());
    }
}

size_t NextLevelReadyQueues::index_for(int32_t worker_id) const {
    for (size_t i = 0; i < worker_ids_.size(); ++i) {
        if (worker_ids_[i] == worker_id) return i;
    }
    throw std::out_of_range("NextLevelReadyQueues: unknown worker id " + std::to_string(worker_id));
}

void NextLevelReadyQueues::push_single(int32_t worker_id, TaskSlot slot) { queues_[index_for(worker_id)]->push(slot); }

bool NextLevelReadyQueues::try_pop_single(int32_t worker_id, TaskSlot &out) {
    return queues_[index_for(worker_id)]->try_pop(out);
}

void NextLevelReadyQueues::push_group(TaskSlot slot) { group_queue_.push(slot); }

bool NextLevelReadyQueues::try_front_group(TaskSlot &out) { return group_queue_.try_front(out); }

bool NextLevelReadyQueues::try_pop_group(TaskSlot &out) { return group_queue_.try_pop(out); }

bool NextLevelReadyQueues::empty() const {
    if (!group_queue_.empty()) return false;
    for (const auto &queue : queues_) {
        if (!queue->empty()) return false;
    }
    return true;
}
