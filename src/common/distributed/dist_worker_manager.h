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
 *
 * Each WorkerThread carries a `WorkerDispatch` queue (slot id + group
 * sub-index); on dispatch the thread reads `callable` / `task_args` /
 * `config` from the ring's slot pool and builds a `TaskArgsView` on
 * demand. The old `WorkerPayload` dispatch carrier is gone (PR-C).
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

class DistRing;  // forward decl — owns the slot state pool

// =============================================================================
// WorkerDispatch — per-dispatch handle handed to a WorkerThread.
// =============================================================================
//
// `task_slot` is the slot id; `group_index` is 0 for single tasks and
// 0..group_size-1 for group members. The thread resolves callable / args /
// config by reading `ring->slot_state(task_slot)`.

struct WorkerDispatch {
    DistTaskSlot task_slot{DIST_INVALID_SLOT};
    int32_t group_index{0};
};

// =============================================================================
// WorkerThread — gives one IWorker its own execution thread
// =============================================================================

class WorkerThread {
public:
    WorkerThread() = default;
    ~WorkerThread() { stop(); }
    WorkerThread(const WorkerThread &) = delete;
    WorkerThread &operator=(const WorkerThread &) = delete;

    // Start the worker thread. `ring` is a borrowed pointer to the engine's
    // slot-state pool — the thread reads callable/args/config from
    // `ring->slot_state(task_slot)` on each dispatch.
    // on_complete(slot) is called (in the WorkerThread) after each run().
    void start(IWorker *worker, DistRing *ring, const std::function<void(DistTaskSlot)> &on_complete);

    // Enqueue a dispatch for the worker. Non-blocking.
    void dispatch(WorkerDispatch d);

    // True if the worker has no active task.
    bool idle() const { return idle_.load(std::memory_order_acquire); }

    void stop();

private:
    IWorker *worker_{nullptr};
    DistRing *ring_{nullptr};
    std::function<void(DistTaskSlot)> on_complete_;

    std::thread thread_;
    std::queue<WorkerDispatch> queue_;
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

    /// Start all WorkerThreads. `ring` is the engine's slot-state pool;
    /// WorkerThreads read slot state from it at dispatch time.
    /// on_complete is called (from the WorkerThread) after each task finishes.
    void start(DistRing *ring, const OnCompleteFn &on_complete);

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
