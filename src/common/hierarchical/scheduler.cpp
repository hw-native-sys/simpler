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

#include "scheduler.h"

#include <stdexcept>
#include <utility>

#include "ring.h"
#include "types.h"
#include "worker_manager.h"

namespace {

bool is_failure(EndpointOutcome outcome) {
    return outcome == EndpointOutcome::TASK_FAILURE || outcome == EndpointOutcome::ENDPOINT_FAILURE;
}

bool is_terminal_group_state(GroupMemberState state) {
    return state == GroupMemberState::SUCCESS || state == GroupMemberState::FAILED ||
           state == GroupMemberState::SKIPPED;
}

}  // namespace

// =============================================================================
// Scheduler
// =============================================================================

// =============================================================================
// Scheduler
// =============================================================================

void Scheduler::start(const Config &cfg) {
    if (cfg.ring == nullptr || cfg.ready_next_level_queue == nullptr || cfg.ready_sub_queue == nullptr ||
        cfg.manager == nullptr)
        throw std::invalid_argument("Scheduler::start: null config fields");
    cfg_ = cfg;

    stop_requested_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    sched_thread_ = std::thread(&Scheduler::run, this);
}

void Scheduler::stop() {
    stop_requested_.store(true, std::memory_order_release);
    completion_cv_.notify_all();
    // Shut down both per-type ready queues so any wait_pop waiters unblock.
    cfg_.ready_next_level_queue->shutdown();
    cfg_.ready_sub_queue->shutdown();

    if (sched_thread_.joinable()) sched_thread_.join();

    running_.store(false, std::memory_order_release);
}

// =============================================================================
// WorkerThread completion callback (called from WorkerThread via Manager)
// =============================================================================

void Scheduler::worker_done(WorkerCompletion completion) {
    TaskSlotState &s = *cfg_.ring->slot_state(completion.task_slot);

    // Group aggregation: only push to completion queue when ALL workers done
    if (s.is_group()) {
        WorkerCompletion terminal = completion;
        bool group_terminal = false;
        {
            std::lock_guard<std::mutex> lk(s.group_mu);
            const int32_t group_size = s.group_size();
            if (s.group_member_states.size() != static_cast<size_t>(group_size)) {
                s.group_member_states.assign(static_cast<size_t>(group_size), GroupMemberState::NOT_DISPATCHED);
                s.group_member_outcomes.assign(static_cast<size_t>(group_size), EndpointOutcome::SKIPPED);
            }
            bool invalid_group_index = completion.group_index < 0 || completion.group_index >= group_size;
            if (invalid_group_index) {
                terminal.outcome = EndpointOutcome::ENDPOINT_FAILURE;
                terminal.error_message = "Scheduler::worker_done: group_index " +
                                         std::to_string(completion.group_index) + " out of range for group_size " +
                                         std::to_string(group_size);
            }

            int32_t index = invalid_group_index ? -1 : terminal.group_index;
            if (index >= 0 && index < group_size) {
                GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(index)];
                if (is_terminal_group_state(member_state)) return;

                if (terminal.outcome == EndpointOutcome::SUCCESS) {
                    member_state = GroupMemberState::SUCCESS;
                } else {
                    member_state = GroupMemberState::FAILED;
                    if (!s.group_failed) {
                        s.group_first_failure_index = index;
                        s.group_first_failure_message = terminal.error_message;
                    }
                    s.group_failed = true;
                }
                s.group_member_outcomes[static_cast<size_t>(index)] = terminal.outcome;
                s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
            } else {
                if (!s.group_failed) {
                    s.group_first_failure_index = -1;
                    s.group_first_failure_message = terminal.error_message;
                }
                s.group_failed = true;
            }

            if (s.group_failed) {
                for (int32_t i = 0; i < group_size; ++i) {
                    GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
                    if (is_terminal_group_state(member_state)) continue;
                    if (!invalid_group_index && member_state != GroupMemberState::NOT_DISPATCHED) continue;
                    member_state = GroupMemberState::SKIPPED;
                    s.group_member_outcomes[static_cast<size_t>(i)] = EndpointOutcome::SKIPPED;
                    s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
                }
            }

            if (s.group_terminal_count.load(std::memory_order_acquire) < group_size) return;

            group_terminal = true;
            if (s.group_failed) {
                int32_t failure_index = s.group_first_failure_index;
                terminal.group_index = failure_index;
                terminal.outcome = EndpointOutcome::TASK_FAILURE;
                if (failure_index >= 0 && failure_index < group_size) {
                    EndpointOutcome member_outcome = s.group_member_outcomes[static_cast<size_t>(failure_index)];
                    if (is_failure(member_outcome)) terminal.outcome = member_outcome;
                }
                terminal.error_message = s.group_first_failure_message;
            } else {
                terminal.outcome = EndpointOutcome::SUCCESS;
                terminal.error_message.clear();
            }
        }
        if (!group_terminal) return;
        completion = std::move(terminal);
    }

    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        completion_queue_.push(std::move(completion));
    }
    completion_cv_.notify_one();
}

void Scheduler::notify_ready() {
    std::lock_guard<std::mutex> lk(completion_mu_);
    completion_cv_.notify_one();
}

// =============================================================================
// Scheduler loop
// =============================================================================

void Scheduler::run() {
    while (true) {
        // Wait until there's something to process
        {
            std::unique_lock<std::mutex> lk(completion_mu_);
            completion_cv_.wait(lk, [this] {
                return !completion_queue_.empty() || !cfg_.ready_next_level_queue->empty() ||
                       !cfg_.ready_sub_queue->empty() || stop_requested_.load(std::memory_order_acquire);
            });
        }

        // Hold loop_mu_ across the entire slot-touching body so drain()'s
        // reset_to_empty() cannot free TaskSlotStates while on_task_complete /
        // dispatch_ready are still reading them (heap-use-after-free).
        std::lock_guard<std::mutex> loop_lk(loop_mu_);

        // Phase 1: drain completions
        while (true) {
            WorkerCompletion completion;
            {
                std::lock_guard<std::mutex> lk(completion_mu_);
                if (completion_queue_.empty()) break;
                completion = std::move(completion_queue_.front());
                completion_queue_.pop();
            }
            on_task_complete(completion);
        }

        // Phase 2: dispatch ready tasks
        dispatch_ready();

        // Exit when stop requested and all workers idle
        if (stop_requested_.load(std::memory_order_acquire)) {
            if (!cfg_.manager->any_busy()) {
                // Final drain
                while (true) {
                    WorkerCompletion completion;
                    {
                        std::lock_guard<std::mutex> lk(completion_mu_);
                        if (completion_queue_.empty()) break;
                        completion = std::move(completion_queue_.front());
                        completion_queue_.pop();
                    }
                    on_task_complete(completion);
                }
                dispatch_ready();
                break;  // loop_lk released on scope exit before exiting run()
            }
        }
    }
}

// =============================================================================
// on_task_complete / try_consume
// =============================================================================

void Scheduler::on_task_complete(const WorkerCompletion &completion) {
    TaskSlot slot = completion.task_slot;
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    bool failed = is_failure(completion.outcome);
    if (failed) {
        s.failure_message = completion.error_message;
        s.state.store(TaskState::FAILED, std::memory_order_release);
    } else {
        s.state.store(TaskState::COMPLETED, std::memory_order_release);
    }

    // Release fanin on downstream consumers
    std::vector<TaskSlot> consumers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
    }
    for (TaskSlot consumer : consumers) {
        if (failed) {
            poison_task(consumer, completion.error_message);
            continue;
        }
        TaskSlotState &cs = *cfg_.ring->slot_state(consumer);
        int32_t released = cs.fanin_released.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (released >= cs.fanin_count) {
            TaskState expected = TaskState::PENDING;
            if (cs.state.compare_exchange_strong(expected, TaskState::READY, std::memory_order_acq_rel)) {
                // Strict-4: route the freshly-ready consumer to the queue
                // matching its own worker type.
                auto *q =
                    (cs.worker_type == WorkerType::NEXT_LEVEL) ? cfg_.ready_next_level_queue : cfg_.ready_sub_queue;
                q->push(consumer);
                completion_cv_.notify_one();
            }
        }
    }

    try_consume(slot);

    // Deferred release: release one fanout ref on each producer this task consumed.
    std::vector<TaskSlot> producers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        producers = s.fanin_producers;
    }
    for (TaskSlot prod : producers) {
        try_consume(prod);
    }
}

void Scheduler::poison_task(TaskSlot slot, const std::string &root_message) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    TaskState state = s.state.load(std::memory_order_acquire);
    if (state == TaskState::FAILED || state == TaskState::CONSUMED || state == TaskState::FREE) return;
    if (state == TaskState::RUNNING || state == TaskState::COMPLETED) return;

    s.failure_message = root_message;
    s.state.store(TaskState::FAILED, std::memory_order_release);

    if (s.is_group()) {
        std::lock_guard<std::mutex> lk(s.group_mu);
        const int32_t group_size = s.group_size();
        if (s.group_member_states.size() != static_cast<size_t>(group_size)) {
            s.group_member_states.assign(static_cast<size_t>(group_size), GroupMemberState::NOT_DISPATCHED);
            s.group_member_outcomes.assign(static_cast<size_t>(group_size), EndpointOutcome::SKIPPED);
        }
        if (!s.group_failed) {
            s.group_failed = true;
            s.group_first_failure_index = -1;
            s.group_first_failure_message = root_message;
        }
        for (int32_t i = 0; i < group_size; ++i) {
            GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
            if (is_terminal_group_state(member_state)) continue;
            member_state = GroupMemberState::SKIPPED;
            s.group_member_outcomes[static_cast<size_t>(i)] = EndpointOutcome::SKIPPED;
            s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::vector<TaskSlot> consumers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
    }
    for (TaskSlot consumer : consumers) {
        poison_task(consumer, root_message);
    }

    try_consume(slot);

    std::vector<TaskSlot> producers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        producers = s.fanin_producers;
    }
    for (TaskSlot prod : producers) {
        try_consume(prod);
    }
}

void Scheduler::try_consume(TaskSlot slot) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    int32_t released = s.fanout_released.fetch_add(1, std::memory_order_acq_rel) + 1;
    int32_t total;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        total = s.fanout_total;
    }
    if (released >= total + 1) {
        TaskState state = s.state.load(std::memory_order_acquire);
        if (state == TaskState::COMPLETED || state == TaskState::FAILED) {
            if (cfg_.on_consumed_cb) cfg_.on_consumed_cb(slot);
        }
    }
}

// =============================================================================
// Dispatch — delegates to WorkerManager
// =============================================================================

void Scheduler::dispatch_ready() {
    // Strict-4: drain each per-type queue with its OWN head-of-line break.
    // A saturated pool of one type only stalls its own queue; the other
    // type continues to dispatch from its pool of idle workers.
    auto drain_one = [this](ReadyQueue *q) {
        TaskSlot slot;
        while (q->try_pop(slot)) {
            TaskSlotState &s = *cfg_.ring->slot_state(slot);
            if (s.state.load(std::memory_order_acquire) != TaskState::READY) {
                continue;
            }
            int N = s.group_size();  // 1 for normal tasks

            // Affinity-aware dispatch: pin args[i] to affinities[i] when set,
            // fill remaining slots from the idle pool. NEXT_LEVEL affinity
            // values are stable worker ids; SUB keeps index semantics for
            // internal callers.
            std::vector<WorkerThread *> workers(static_cast<size_t>(N), nullptr);
            bool ok = true;

            // Pass 1: satisfy affinity constraints
            for (int i = 0; i < N; i++) {
                int32_t aff = s.get_affinity(i);
                if (aff >= 0) {
                    auto *wt = s.worker_type == WorkerType::NEXT_LEVEL ?
                                   cfg_.manager->get_worker_by_id(s.worker_type, aff) :
                                   cfg_.manager->get_worker_by_index(s.worker_type, aff);
                    if (!wt || !wt->idle() || !s.worker_allowed(i, wt->worker_id())) {
                        ok = false;
                        break;
                    }
                    workers[static_cast<size_t>(i)] = wt;
                }
            }

            // Pass 2: fill unconstrained slots from idle pool
            if (ok) {
                for (int i = 0; i < N; i++) {
                    if (workers[static_cast<size_t>(i)] != nullptr) continue;
                    auto *wt =
                        cfg_.manager->pick_idle_excluding_eligible(s.worker_type, workers, s.eligible_workers_for(i));
                    if (!wt) {
                        ok = false;
                        break;
                    }
                    workers[static_cast<size_t>(i)] = wt;
                }
            }

            if (!ok) {
                q->push(slot);
                break;
            }

            s.state.store(TaskState::RUNNING, std::memory_order_release);
            if (s.is_group()) {
                std::lock_guard<std::mutex> lk(s.group_mu);
                s.group_member_states.assign(static_cast<size_t>(N), GroupMemberState::NOT_DISPATCHED);
                s.group_member_outcomes.assign(static_cast<size_t>(N), EndpointOutcome::SKIPPED);
                s.group_terminal_count.store(0, std::memory_order_relaxed);
                s.group_dispatched_count.store(0, std::memory_order_relaxed);
                s.group_failed = false;
                s.group_first_failure_index = -1;
                s.group_first_failure_message.clear();
            }
            for (int i = 0; i < N; i++) {
                if (s.is_group()) {
                    std::lock_guard<std::mutex> lk(s.group_mu);
                    GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
                    if (member_state != GroupMemberState::NOT_DISPATCHED || s.group_failed) {
                        continue;
                    }
                    member_state = GroupMemberState::RUNNING;
                    s.group_dispatched_count.fetch_add(1, std::memory_order_relaxed);
                }
                WorkerDispatch d;
                d.task_slot = slot;
                d.group_index = i;
                workers[static_cast<size_t>(i)]->dispatch(d);
            }
        }
    };

    drain_one(cfg_.ready_next_level_queue);
    drain_one(cfg_.ready_sub_queue);
}
