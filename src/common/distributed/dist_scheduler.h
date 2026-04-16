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
 * DistScheduler — DAG scheduling engine.
 *
 * The Scheduler thread routes tasks through the DAG lifecycle:
 *   ready_queue → dispatch (via WorkerManager) → completion → fanout release → new ready
 *
 * Worker pool management (WorkerThread creation, idle selection, dispatch) is
 * delegated to DistWorkerManager. The Scheduler only drives the DAG state machine.
 *
 * Flow:
 *   Orch: submit() → ready_queue.push(slot) + cv.notify()
 *
 *   Scheduler thread:
 *     wait on cv (ready_queue OR completion_queue non-empty)
 *     drain completion_queue → on_task_complete → fanout release → ready_queue
 *     drain ready_queue → manager.pick_n_idle → dispatch
 *
 *   WorkerThread (managed by DistWorkerManager):
 *     loop: task_queue.pop() → worker.run(payload) →
 *           completion callback → Scheduler.worker_done(slot)
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "dist_types.h"

class DistWorkerManager;  // forward decl
class DistRing;           // forward decl

// =============================================================================
// DistScheduler — DAG engine (no worker pool ownership)
// =============================================================================

class DistScheduler {
public:
    struct Config {
        DistRing *ring;  // owns slot state storage; Scheduler reads via ring->slot_state(id)
        // Strict-4 per-worker-type ready queues. `dispatch_ready` walks each
        // queue independently so a saturated pool of one worker type cannot
        // head-of-line-block dispatch for the other.
        DistReadyQueue *ready_next_level_queue;
        DistReadyQueue *ready_sub_queue;
        DistWorkerManager *manager;  // not owned — Scheduler calls manager for dispatch
        // Called when a task reaches CONSUMED (TensorMap cleanup + ring release).
        std::function<void(DistTaskSlot)> on_consumed_cb;
    };

    void start(const Config &cfg);
    void stop();

    bool running() const { return running_.load(std::memory_order_acquire); }

    // Called by WorkerManager (from WorkerThread) after run() completes.
    void worker_done(DistTaskSlot slot);

private:
    Config cfg_;

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
};
