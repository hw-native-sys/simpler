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
 * DistWorkerManager — worker pool lifecycle and dispatch.
 *
 * Owns WorkerThread instances (one per registered IWorker).
 * Provides idle-worker selection and dispatch to the Scheduler.
 * The Scheduler drives the DAG; the Manager drives the workers.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "dist_types.h"

// =============================================================================
// WorkerThread — gives one IWorker its own execution thread
// =============================================================================

class WorkerThread {
public:
    WorkerThread() = default;
    ~WorkerThread() { stop(); }
    WorkerThread(const WorkerThread &) = delete;
    WorkerThread &operator=(const WorkerThread &) = delete;

    // Start the worker thread.
    // on_complete(slot) is called (in the WorkerThread) after each run().
    void start(IWorker *worker, const std::function<void(DistTaskSlot)> &on_complete);

    // Enqueue a task for the worker.  Non-blocking.
    void dispatch(const WorkerPayload &payload);

    // True if the worker has no active task.
    bool idle() const { return idle_.load(std::memory_order_acquire); }

    void stop();

private:
    IWorker *worker_{nullptr};
    std::function<void(DistTaskSlot)> on_complete_;

    std::thread thread_;
    std::queue<WorkerPayload> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
    std::atomic<bool> idle_{true};

    void loop();
};

// =============================================================================
// DistWorkerManager — worker pool lifecycle and dispatch
// =============================================================================

class DistWorkerManager {
public:
    using OnCompleteFn = std::function<void(DistTaskSlot)>;

    void add_next_level(IWorker *worker);
    void add_sub(IWorker *worker);

    /// Start all WorkerThreads. on_complete is called (from the WorkerThread)
    /// after each task finishes — the Scheduler hooks into this.
    void start(const OnCompleteFn &on_complete);

    /// Stop and join all WorkerThreads.
    void stop();

    /// Pick one idle WorkerThread of the given type. Returns nullptr if none idle.
    WorkerThread *pick_idle(WorkerType type) const;

    /// Pick up to n idle WorkerThreads of the given type.
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n) const;

    /// True if any WorkerThread (of any type) is currently busy.
    bool any_busy() const;

private:
    std::vector<IWorker *> next_level_workers_;
    std::vector<IWorker *> sub_workers_;

    std::vector<std::unique_ptr<WorkerThread>> next_level_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;
};
