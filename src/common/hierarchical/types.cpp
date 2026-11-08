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

#include <algorithm>
#include <stdexcept>

// =============================================================================
// TaskSlotState
// =============================================================================

void TaskSlotState::reset() {
    state.store(TaskState::FREE, std::memory_order_relaxed);
    run_id = INVALID_RUN_ID;
    pipeline_lease = PipelineSlotLease{};
    fanin_count.store(0, std::memory_order_relaxed);
    fanin_released.store(0, std::memory_order_relaxed);
    failure_propagation_pending.store(false, std::memory_order_relaxed);
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

bool try_mark_ready(TaskSlotState &s) {
    std::lock_guard<std::mutex> lk(s.fanout_mu);
    if (s.state.load(std::memory_order_acquire) != TaskState::PENDING) return false;
    if (s.fanin_released.load(std::memory_order_acquire) < s.fanin_count.load(std::memory_order_acquire)) return false;
    s.state.store(TaskState::READY, std::memory_order_release);
    return true;
}

std::optional<TaskState> claim_task_failure(TaskSlotState &s, const std::string &message) {
    std::lock_guard<std::mutex> lk(s.fanout_mu);
    TaskState current = s.state.load(std::memory_order_acquire);
    // Retry rather than attempt once: a producer completing concurrently moves
    // its consumer PENDING -> READY without holding fanout_mu, and a single
    // failed exchange would leave that slot dispatchable under a run that is
    // already failing. The loop ends as soon as the slot leaves the claimable
    // set, which is either our own FAILED or a dispatch that beat us to it.
    while (current == TaskState::PENDING || current == TaskState::READY || current == TaskState::BUILDING) {
        // Copy before the state transition. A failed string allocation must
        // leave the slot claimable rather than publish FAILED without a
        // durable reason or propagation owner.
        std::string durable_message(message);
        if (s.state.compare_exchange_strong(
                current, TaskState::FAILED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            s.failure_message.swap(durable_message);
            // The debt remains set while propagation takes every fallible
            // snapshot for a BUILDING slot. PENDING and READY claims remain
            // exclusively owned by their claimant, so cancellation cannot
            // prepare them concurrently.
            if (current == TaskState::BUILDING) {
                s.failure_propagation_pending.store(true, std::memory_order_release);
            }
            return current;
        }
    }
    return std::nullopt;
}

bool commit_failure_propagation(TaskSlotState &s) noexcept {
    return s.failure_propagation_pending.exchange(false, std::memory_order_acq_rel);
}

void mark_group_members_skipped(TaskSlotState &s, const std::string &message) {
    if (!s.is_group()) return;
    std::lock_guard<std::mutex> lk(s.group_mu);
    const int32_t group_size = s.group_size();
    const size_t size = static_cast<size_t>(group_size);
    const bool repair_bookkeeping = s.group_member_states.size() != size || s.group_member_outcomes.size() != size;

    std::vector<GroupMemberState> member_states;
    std::vector<EndpointOutcome> member_outcomes;
    if (repair_bookkeeping) {
        member_states.assign(size, GroupMemberState::NOT_DISPATCHED);
        member_outcomes.assign(size, EndpointOutcome::SKIPPED);
        std::copy_n(s.group_member_states.begin(), std::min(size, s.group_member_states.size()), member_states.begin());
        std::copy_n(
            s.group_member_outcomes.begin(), std::min(size, s.group_member_outcomes.size()), member_outcomes.begin()
        );
    }

    std::string first_failure_message;
    if (!s.group_failed) {
        first_failure_message = message;
    }

    // Every allocation above completes before either bookkeeping vector is
    // replaced, so an exception cannot expose mismatched sizes to a retry.
    if (repair_bookkeeping) {
        s.group_member_states.swap(member_states);
        s.group_member_outcomes.swap(member_outcomes);
    }
    if (!s.group_failed) {
        s.group_failed = true;
        s.group_first_failure_index = -1;
        s.group_first_failure_message.swap(first_failure_message);
    }
    for (int32_t i = 0; i < group_size; ++i) {
        GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
        if (member_state == GroupMemberState::SUCCESS || member_state == GroupMemberState::FAILED ||
            member_state == GroupMemberState::SKIPPED) {
            continue;
        }
        member_state = GroupMemberState::SKIPPED;
        s.group_member_outcomes[static_cast<size_t>(i)] = EndpointOutcome::SKIPPED;
        s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
    }
}

// =============================================================================
// ReadyQueue
// =============================================================================

void ReadyQueue::push(TaskSlot slot) { push(INVALID_RUN_ID, slot); }

void ReadyQueue::push(RunId run_id, TaskSlot slot) {
    std::lock_guard<std::mutex> lk(mu_);
    auto [it, inserted] = queues_.try_emplace(run_id);
    if (inserted) run_order_.push_back(run_id);
    it->second.push(slot);
}

bool ReadyQueue::try_pop(TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (run_order_.empty()) return false;
    auto it = queues_.find(run_order_.front());
    if (it == queues_.end() || it->second.empty()) throw std::logic_error("ReadyQueue: corrupt run order");
    out = it->second.front();
    it->second.pop();
    if (it->second.empty()) {
        queues_.erase(it);
        run_order_.pop_front();
    }
    return true;
}

bool ReadyQueue::try_pop(RunId run_id, TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = queues_.find(run_id);
    if (it == queues_.end() || it->second.empty()) return false;
    out = it->second.front();
    it->second.pop();
    if (it->second.empty()) {
        queues_.erase(it);
        auto order_it = std::find(run_order_.begin(), run_order_.end(), run_id);
        if (order_it != run_order_.end()) run_order_.erase(order_it);
    }
    return true;
}

bool ReadyQueue::empty() const {
    std::lock_guard<std::mutex> lk(mu_);
    return run_order_.empty();
}

bool ReadyQueue::empty(RunId run_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = queues_.find(run_id);
    return it == queues_.end() || it->second.empty();
}

bool ReadyQueue::try_front(TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (run_order_.empty()) return false;
    auto it = queues_.find(run_order_.front());
    if (it == queues_.end() || it->second.empty()) throw std::logic_error("ReadyQueue: corrupt run order");
    out = it->second.front();
    return true;
}

bool ReadyQueue::try_front(RunId run_id, TaskSlot &out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = queues_.find(run_id);
    if (it == queues_.end() || it->second.empty()) return false;
    out = it->second.front();
    return true;
}

void ReadyQueue::erase_run(RunId run_id) {
    std::lock_guard<std::mutex> lk(mu_);
    queues_.erase(run_id);
    auto it = std::find(run_order_.begin(), run_order_.end(), run_id);
    if (it != run_order_.end()) run_order_.erase(it);
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

void NextLevelReadyQueues::push_single(int32_t worker_id, RunId run_id, TaskSlot slot) {
    queues_[index_for(worker_id)]->push(run_id, slot);
}

bool NextLevelReadyQueues::try_pop_single(int32_t worker_id, TaskSlot &out) {
    return queues_[index_for(worker_id)]->try_pop(out);
}

bool NextLevelReadyQueues::try_front_single(int32_t worker_id, TaskSlot &out) {
    return queues_[index_for(worker_id)]->try_front(out);
}

bool NextLevelReadyQueues::try_front_single(int32_t worker_id, RunId run_id, TaskSlot &out) {
    return queues_[index_for(worker_id)]->try_front(run_id, out);
}

bool NextLevelReadyQueues::try_pop_single(int32_t worker_id, RunId run_id, TaskSlot &out) {
    return queues_[index_for(worker_id)]->try_pop(run_id, out);
}

void NextLevelReadyQueues::push_group(TaskSlot slot) { group_queue_.push(slot); }
void NextLevelReadyQueues::push_group(RunId run_id, TaskSlot slot) { group_queue_.push(run_id, slot); }

bool NextLevelReadyQueues::try_front_group(TaskSlot &out) { return group_queue_.try_front(out); }
bool NextLevelReadyQueues::try_front_group(RunId run_id, TaskSlot &out) { return group_queue_.try_front(run_id, out); }

bool NextLevelReadyQueues::try_pop_group(TaskSlot &out) { return group_queue_.try_pop(out); }
bool NextLevelReadyQueues::try_pop_group(RunId run_id, TaskSlot &out) { return group_queue_.try_pop(run_id, out); }

bool NextLevelReadyQueues::groups_empty() const { return group_queue_.empty(); }
bool NextLevelReadyQueues::groups_empty(RunId run_id) const { return group_queue_.empty(run_id); }

bool NextLevelReadyQueues::single_empty(int32_t worker_id, RunId run_id) const {
    return queues_[index_for(worker_id)]->empty(run_id);
}

bool NextLevelReadyQueues::singles_empty(RunId run_id) const {
    for (const auto &queue : queues_) {
        if (!queue->empty(run_id)) return false;
    }
    return true;
}

void NextLevelReadyQueues::erase_run(RunId run_id) {
    group_queue_.erase_run(run_id);
    for (const auto &queue : queues_)
        queue->erase_run(run_id);
}
