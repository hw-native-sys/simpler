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

#include "ring.h"
#include "types.h"
#include "worker_manager.h"

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

void Scheduler::worker_done(TaskSlot slot) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);

    // Group aggregation: only push to completion queue when ALL workers done
    if (s.is_group()) {
        int32_t done = s.sub_complete_count.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (done < s.group_size()) return;
    }

    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        completion_queue_.push(slot);
    }
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
            completion_cv_.wait_for(lk, std::chrono::milliseconds(1), [this] {
                return !completion_queue_.empty() || stop_requested_.load(std::memory_order_acquire);
            });
        }

        // Phase 1: drain completions
        while (true) {
            TaskSlot slot;
            {
                std::lock_guard<std::mutex> lk(completion_mu_);
                if (completion_queue_.empty()) break;
                slot = completion_queue_.front();
                completion_queue_.pop();
            }
            on_task_complete(slot);
        }

        // Phase 2: dispatch ready tasks
        dispatch_ready();

        // Exit when stop requested and all workers idle
        if (stop_requested_.load(std::memory_order_acquire)) {
            if (!cfg_.manager->any_busy()) {
                // Final drain
                while (true) {
                    TaskSlot slot;
                    {
                        std::lock_guard<std::mutex> lk(completion_mu_);
                        if (completion_queue_.empty()) break;
                        slot = completion_queue_.front();
                        completion_queue_.pop();
                    }
                    on_task_complete(slot);
                }
                dispatch_ready();
                break;
            }
        }
    }
}

// =============================================================================
// on_task_complete / try_consume
// =============================================================================

void Scheduler::on_task_complete(TaskSlot slot) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    s.state.store(TaskState::COMPLETED, std::memory_order_release);

    // Release fanin on downstream consumers
    std::vector<TaskSlot> consumers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
    }
    for (TaskSlot consumer : consumers) {
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

void Scheduler::try_consume(TaskSlot slot) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    int32_t released = s.fanout_released.fetch_add(1, std::memory_order_acq_rel) + 1;
    int32_t total;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        total = s.fanout_total;
    }
    if (released >= total + 1) {
        if (s.state.load(std::memory_order_acquire) == TaskState::COMPLETED) {
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
            int N = s.group_size();  // 1 for normal tasks

            auto workers = cfg_.manager->pick_n_idle(s.worker_type, N);
            if (static_cast<int>(workers.size()) < N) {
                q->push(slot);
                break;
            }

            s.state.store(TaskState::RUNNING, std::memory_order_release);
            for (int i = 0; i < N; i++) {
                WorkerDispatch d;
                d.task_slot = slot;
                d.group_index = i;
                workers[i]->dispatch(d);
            }
        }
    };

    drain_one(cfg_.ready_next_level_queue);
    drain_one(cfg_.ready_sub_queue);
}
