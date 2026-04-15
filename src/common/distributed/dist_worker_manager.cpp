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

#include "dist_worker_manager.h"

#include <stdexcept>

#include "dist_ring.h"

// =============================================================================
// WorkerThread
// =============================================================================

void WorkerThread::start(IWorker *worker, DistRing *ring, const std::function<void(DistTaskSlot)> &on_complete) {
    worker_ = worker;
    ring_ = ring;
    on_complete_ = on_complete;
    shutdown_ = false;
    idle_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&WorkerThread::loop, this);
}

void WorkerThread::dispatch(WorkerDispatch d) {
    idle_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lk(mu_);
    queue_.push(d);
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
        WorkerDispatch d;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] {
                return !queue_.empty() || shutdown_;
            });
            if (queue_.empty()) break;  // shutdown
            d = queue_.front();
            queue_.pop();
        }

        // Resolve callable / args / config from the slot. For NEXT_LEVEL
        // tasks `callable` is the ChipCallable buffer pointer; for SUB
        // tasks it encodes the registry id in the low 32 bits. The view
        // is a zero-copy handle over the slot's vector-backed TaskArgs.
        DistTaskSlotState &s = *ring_->slot_state(d.task_slot);
        uint64_t callable = (s.worker_type == WorkerType::SUB) ? static_cast<uint64_t>(s.callable_id) : s.callable;
        TaskArgsView view = s.args_view(d.group_index);

        worker_->run(callable, view, s.config);  // blocking in this thread
        idle_.store(true, std::memory_order_release);
        on_complete_(d.task_slot);  // notify Scheduler
    }
}

// =============================================================================
// DistWorkerManager
// =============================================================================

void DistWorkerManager::add_next_level(IWorker *worker) { next_level_workers_.push_back(worker); }

void DistWorkerManager::add_sub(IWorker *worker) { sub_workers_.push_back(worker); }

void DistWorkerManager::start(DistRing *ring, const OnCompleteFn &on_complete) {
    if (ring == nullptr) throw std::invalid_argument("DistWorkerManager::start: null ring");
    auto make_threads = [&](const std::vector<IWorker *> &workers,
                            std::vector<std::unique_ptr<WorkerThread>> &threads) {
        for (IWorker *w : workers) {
            auto wt = std::make_unique<WorkerThread>();
            wt->start(w, ring, on_complete);
            threads.push_back(std::move(wt));
        }
    };
    make_threads(next_level_workers_, next_level_threads_);
    make_threads(sub_workers_, sub_threads_);
}

void DistWorkerManager::stop() {
    for (auto &wt : next_level_threads_)
        wt->stop();
    for (auto &wt : sub_threads_)
        wt->stop();
    next_level_threads_.clear();
    sub_threads_.clear();
}

WorkerThread *DistWorkerManager::pick_idle(WorkerType type) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
    for (auto &wt : threads) {
        if (wt->idle()) return wt.get();
    }
    return nullptr;
}

std::vector<WorkerThread *> DistWorkerManager::pick_n_idle(WorkerType type, int n) const {
    auto &threads = (type == WorkerType::NEXT_LEVEL) ? next_level_threads_ : sub_threads_;
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

bool DistWorkerManager::any_busy() const {
    for (auto &wt : next_level_threads_)
        if (!wt->idle()) return true;
    for (auto &wt : sub_threads_)
        if (!wt->idle()) return true;
    return false;
}
