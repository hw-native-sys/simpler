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
 * Scheduler — DAG scheduling engine.
 *
 * The Scheduler thread routes tasks through the DAG lifecycle:
 *   ready_queue → dispatch (via WorkerManager) → completion → fanout release → new ready
 *
 * Worker pool management (WorkerThread creation, idle selection, dispatch) is
 * delegated to WorkerManager. The Scheduler only drives the DAG state machine.
 *
 * Flow:
 *   Orch: submit() → ready_queue.push(slot) + Scheduler::notify_ready()
 *
 *   Scheduler thread:
 *     wait on cv (ready_queue OR completion_queue non-empty)
 *     drain completion_queue → on_task_complete → fanout release → ready_queue
 *     drain ready_queue → manager.pick_idle → dispatch
 *
 *   WorkerThread (managed by WorkerManager):
 *     loop: task_queue.pop() → endpoint.run(dispatch) →
 *           completion callback → Scheduler.worker_done(completion)
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "types.h"

class WorkerManager;  // forward decl
class Ring;           // forward decl

// =============================================================================
// Scheduler — DAG engine (no worker pool ownership)
// =============================================================================

class Scheduler {
public:
    struct Config {
        Ring *ring;  // owns slot state storage; Scheduler reads via ring->slot_state(id)
        // NEXT_LEVEL singles use one FIFO per stable worker id. NEXT_LEVEL
        // groups and SUB tasks use their shared queues.
        ReadyQueue *ready_next_level_queue;
        ReadyQueue *ready_sub_queue;
        PerWorkerReadyQueues *ready_next_level_single_queues;
        WorkerManager *manager;  // not owned — Scheduler calls manager for dispatch
        // Shared READY routing path owned by Orchestrator.
        std::function<void(TaskSlot)> enqueue_ready_cb;
        // Called when a task reaches CONSUMED (TensorMap cleanup + ring release).
        std::function<void(TaskSlot)> on_consumed_cb;
    };

    void start(const Config &cfg);
    void stop();

    bool running() const { return running_.load(std::memory_order_acquire); }

    // Called by WorkerManager (from WorkerThread) after endpoint run() reaches
    // a terminal outcome.
    void worker_done(WorkerCompletion completion);

    // Called by Orchestrator after it pushes a newly-ready root task. ReadyQueue
    // has its own condition variable, but the Scheduler waits on completion_cv_.
    void notify_ready();

    // Mutex held by run() across each loop iteration's slot-touching body
    // (completion processing + dispatch). Orchestrator::drain() acquires it
    // before Ring::reset_to_empty() so the ring can't be torn down while the
    // scheduler thread is mid-on_task_complete (heap-use-after-free).
    std::mutex &loop_mutex() { return loop_mu_; }

private:
    Config cfg_;
    std::mutex loop_mu_;

    // Shared completion queue (WorkerThread → Scheduler)
    std::queue<WorkerCompletion> completion_queue_;
    std::mutex completion_mu_;
    std::condition_variable completion_cv_;

    std::thread sched_thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};

    void run();
    void on_task_complete(const WorkerCompletion &completion);
    void poison_task(TaskSlot slot, const std::string &root_message);
    void try_consume(TaskSlot slot);
    void dispatch_ready();
    void dispatch_next_level_singles();
};
