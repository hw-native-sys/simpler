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
 * DistScheduler — Scheduler thread + per-worker WorkerThread model.
 *
 * Each registered IWorker gets a WorkerThread wrapper with its own thread
 * and task queue.  The Scheduler thread routes tasks from ready_queue to
 * idle WorkerThreads and waits on a shared completion CV instead of polling.
 *
 * Flow:
 *   Orch: submit() → ready_queue.push(slot) + cv.notify()
 *
 *   Scheduler thread:
 *     wait on cv (ready_queue OR completion_queue non-empty)
 *     drain completion_queue → on_task_complete → fanout release → ready_queue
 *     drain ready_queue → pick idle WorkerThread → worker_thread.dispatch(slot)
 *
 *   WorkerThread (one per IWorker):
 *     loop: task_queue.pop() (blocking) → worker.run(payload) →
 *           completion_queue.push(slot) + cv.notify()
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
// DistScheduler
// =============================================================================

class DistScheduler {
public:
    struct Config {
        DistTaskSlotState *slots;
        int32_t num_slots;
        DistReadyQueue *ready_queue;
        std::vector<IWorker *> chip_workers;  // WorkerType::CHIP
        std::vector<IWorker *> sub_workers;   // WorkerType::SUB
        // Called when a task reaches CONSUMED (TensorMap cleanup + ring release).
        std::function<void(DistTaskSlot)> on_consumed_cb;
    };

    void start(const Config &cfg);
    void stop();

    bool running() const { return running_.load(std::memory_order_acquire); }

private:
    Config cfg_;

    // Per-worker threads
    std::vector<std::unique_ptr<WorkerThread>> chip_threads_;
    std::vector<std::unique_ptr<WorkerThread>> sub_threads_;

    // Shared completion queue (WorkerThread → Scheduler)
    std::queue<DistTaskSlot> completion_queue_;
    std::mutex completion_mu_;
    std::condition_variable completion_cv_;

    std::thread sched_thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};

    void run();
    void on_task_complete(DistTaskSlot slot);
    void try_consume(DistTaskSlot slot);
    void dispatch_ready();
    WorkerThread *pick_idle(WorkerType type);
    std::vector<WorkerThread *> pick_n_idle(WorkerType type, int n);

    // Called by WorkerThread after run() completes
    void worker_done(DistTaskSlot slot);
};
