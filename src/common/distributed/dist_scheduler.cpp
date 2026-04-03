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

#include "dist_scheduler.h"

#include <stdexcept>

// =============================================================================
// WorkerThread
// =============================================================================

void WorkerThread::start(IWorker *worker, const std::function<void(DistTaskSlot)> &on_complete) {
    worker_ = worker;
    on_complete_ = on_complete;
    shutdown_ = false;
    idle_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&WorkerThread::loop, this);
}

void WorkerThread::dispatch(const WorkerPayload &payload) {
    idle_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push(payload);
    cv_.notify_one();
}

void WorkerThread::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void WorkerThread::loop() {
    while (true) {
        WorkerPayload payload;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] {
                return !queue_.empty() || shutdown_;
            });
            if (queue_.empty()) break;  // shutdown
            payload = queue_.front();
            queue_.pop();
        }

        worker_->run(payload);  // blocking in this thread
        idle_.store(true, std::memory_order_release);
        on_complete_(payload.task_slot);  // notify Scheduler
    }
}

// =============================================================================
// DistScheduler
// =============================================================================

void DistScheduler::start(const Config &cfg) {
    if (cfg.slots == nullptr || cfg.ready_queue == nullptr)
        throw std::invalid_argument("DistScheduler::start: null config fields");
    cfg_ = cfg;

    // Create a WorkerThread per IWorker
    auto make_threads = [&](const std::vector<IWorker *> &workers,
                            std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (IWorker *w : workers) {
            auto wt = std::make_unique<WorkerThread>();
            wt->start(w, [this](DistTaskSlot slot) {
                worker_done(slot);
            });
            threads.push_back(std::move(wt));
        }
    };
    make_threads(cfg_.chip_workers, chip_threads_);
    make_threads(cfg_.sub_workers, sub_threads_);

    stop_requested_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    sched_thread_ = std::thread(&DistScheduler::run, this);
}

void DistScheduler::stop() {
    stop_requested_.store(true, std::memory_order_release);
    completion_cv_.notify_all();
    cfg_.ready_queue->shutdown();

    if (sched_thread_.joinable()) sched_thread_.join();

    for (auto &wt : chip_threads_)
        wt->stop();
    for (auto &wt : sub_threads_)
        wt->stop();
    chip_threads_.clear();
    sub_threads_.clear();

    running_.store(false, std::memory_order_release);
}

// =============================================================================
// WorkerThread completion callback (called from WorkerThread)
// =============================================================================

void DistScheduler::worker_done(DistTaskSlot slot) {
    DistTaskSlotState &s = cfg_.slots[slot];

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

void DistScheduler::run() {
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
            DistTaskSlot slot;
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
            bool any_busy = false;
            for (auto &wt : chip_threads_)
                if (!wt->idle()) {
                    any_busy = true;
                    break;
                }
            if (!any_busy)
                for (auto &wt : sub_threads_)
                    if (!wt->idle()) {
                        any_busy = true;
                        break;
                    }
            if (!any_busy) {
                // Final drain
                while (true) {
                    DistTaskSlot slot;
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

void DistScheduler::on_task_complete(DistTaskSlot slot) {
    DistTaskSlotState &s = cfg_.slots[slot];
    s.state.store(TaskState::COMPLETED, std::memory_order_release);

    // Release fanin on downstream consumers
    std::vector<DistTaskSlot> consumers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
    }
    for (DistTaskSlot consumer : consumers) {
        DistTaskSlotState &cs = cfg_.slots[consumer];
        int32_t released = cs.fanin_released.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (released >= cs.fanin_count) {
            TaskState expected = TaskState::PENDING;
            if (cs.state.compare_exchange_strong(expected, TaskState::READY, std::memory_order_acq_rel)) {
                cfg_.ready_queue->push(consumer);
                completion_cv_.notify_one();
            }
        }
    }

    try_consume(slot);

    // Deferred release: release one fanout ref on each producer this task consumed.
    // Mirrors L2 "deferred release: walk fanin → release producer".
    std::vector<DistTaskSlot> producers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        producers = s.fanin_producers;
    }
    for (DistTaskSlot prod : producers) {
        try_consume(prod);
    }
}

void DistScheduler::try_consume(DistTaskSlot slot) {
    DistTaskSlotState &s = cfg_.slots[slot];
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
// Dispatch
// =============================================================================

void DistScheduler::dispatch_ready() {
    DistTaskSlot slot;
    while (cfg_.ready_queue->try_pop(slot)) {
        DistTaskSlotState &s = cfg_.slots[slot];
        int N = s.group_size();  // 1 for normal tasks

        auto workers = pick_n_idle(s.payload.worker_type, N);
        if (static_cast<int>(workers.size()) < N) {
            cfg_.ready_queue->push(slot);
            break;
        }

        s.state.store(TaskState::RUNNING, std::memory_order_release);
        for (int i = 0; i < N; i++) {
            WorkerPayload p = s.payload;
            p.args = s.args_list[i];
            workers[i]->dispatch(p);
        }
    }
}

WorkerThread *DistScheduler::pick_idle(WorkerType type) {
    auto &threads = (type == WorkerType::CHIP) ? chip_threads_ : sub_threads_;
    for (auto &wt : threads) {
        if (wt->idle()) return wt.get();
    }
    return nullptr;
}

std::vector<WorkerThread *> DistScheduler::pick_n_idle(WorkerType type, int n) {
    auto &threads = (type == WorkerType::CHIP) ? chip_threads_ : sub_threads_;
    std::vector<WorkerThread *> result;
    result.reserve(n);
    for (auto &wt : threads) {
        if (wt->idle()) {
            result.push_back(wt.get());
            if (static_cast<int>(result.size()) >= n) break;
        }
    }
    return result;
}
