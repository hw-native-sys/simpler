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
 * Worker endpoint ownership and dispatch are delegated to WorkerManager.
 * NEXT_LEVEL placement is fixed at submit; SUB remains free.
 *
 * Flow:
 *   Orch: submit() → directed NEXT_LEVEL queue or shared SUB queue + notify
 *
 *   Scheduler thread:
 *     wait on cv while no endpoint is active
 *     poll every active endpoint through WorkerManager
 *     drain completion_queue → on_task_complete → fanout release → ready_queue
 *     launch directed NEXT_LEVEL tasks, then freely scheduled SUB tasks
 *
 * The idle wait is edge-triggered, so any state change that turns already-
 * queued work into placeable work must advance the wake generation. Active
 * endpoints keep the loop running until all endpoint-owned work terminalizes.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>

#include "types.h"

class WorkerManager;  // forward decl
class WorkerThread;   // forward decl
class Ring;           // forward decl
struct WorkerDispatch;

// =============================================================================
// Scheduler — DAG engine (no worker pool ownership)
// =============================================================================

/**
 * Take ownership of a READY slot for dispatch, atomically.
 *
 * Reading READY and later storing RUNNING are not the same thing. A run whose
 * graph callback throws fails its own unstarted slots and consumes them, and
 * its fence can then release the run's pipeline lease — all while the scheduler
 * sits between those two points picking workers. A plain store would overwrite
 * that cancelled state and dispatch a task whose run is already terminal and
 * whose slot may have been reused. A failed claim means the slot is no longer
 * ours: whoever moved it out of READY owns its consume.
 *
 * Call this at the point of dispatch, after every other admission check — the
 * window this closes is exactly the code between the queue pop and the launch.
 */
bool claim_for_dispatch(TaskSlotState &s);

class Scheduler {
public:
    struct ReservationStallDiagnostic {
        TaskSlot group_slot{INVALID_SLOT};
        const int32_t *busy_target_worker_ids{nullptr};
        size_t busy_target_count{0};
        const int32_t *idle_queued_target_worker_ids{nullptr};
        const TaskSlot *idle_queued_single_head_slots{nullptr};
        size_t idle_queued_target_count{0};
    };

    using ReservationStallSink = void (*)(void *, const ReservationStallDiagnostic &) noexcept;

    struct Config {
        Ring *ring;  // owns slot state storage; Scheduler reads via ring->slot_state(id)
        ReadyQueue *ready_sub_queue;
        NextLevelReadyQueues *ready_next_level_queues;
        WorkerManager *manager;  // not owned — Scheduler calls manager for dispatch
        // Shared READY routing path owned by Orchestrator.
        std::function<void(TaskSlot)> enqueue_ready_cb;
        // Production workers expose exactly one whole-run FIFO head. Tests
        // that omit this callback retain the legacy unpartitioned queue path.
        std::function<RunId()> active_run_cb;
        std::function<RunId()> preparable_run_cb;
        // The staged successor authorized to launch its device work while the
        // active run is still executing, or INVALID_RUN_ID. Omitted by callers
        // configured at launch depth one, which authorize nothing.
        std::function<RunId()> early_launch_run_cb;
        // Called when a task reaches CONSUMED (TensorMap cleanup + ring release).
        std::function<void(TaskSlot)> on_consumed_cb;
        // Called as soon as an endpoint reports failure so the error is
        // attached to the task's run even when a group has other members live.
        std::function<void(TaskSlot, const std::string &)> on_task_failed_cb;
        // Diagnostic-only reservation stall reporting. The sink must not
        // block: it runs on the scheduler dispatch path.
        std::chrono::milliseconds reservation_stall_warn_after{std::chrono::seconds(5)};
        ReservationStallSink reservation_stall_sink{nullptr};
        void *reservation_stall_sink_context{nullptr};
        // Test seam. Invoked immediately before the dispatch claim, which is
        // the one instant a cancelling run can still take a slot away. The
        // window is unreachable from outside — every other observable point is
        // either before the pop or after the launch — so a test that wants to
        // exercise the losing side of the claim has to be let in here.
        // Unset in production.
        std::function<void(TaskSlot)> before_claim_cb;
        // Test seam for a group arriving after the group phase observed an
        // empty queue but before this pass starts claiming singles.
        std::function<void()> after_group_phase_cb;
    };

    void start(const Config &cfg);
    void request_stop();
    void stop();

    bool running() const { return running_.load(std::memory_order_acquire); }
    // Diagnostic only — counts dispatch passes so an observer can tell a parked
    // scheduler from one spinning on unplaceable work. Orders nothing.
    uint64_t dispatch_round_count() const { return dispatch_round_count_.load(std::memory_order_relaxed); }
    // Diagnostic only — tasks actually handed to a worker, which a dispatch pass
    // may do any number of times including none. Orders nothing.
    uint64_t dispatched_total() const { return dispatched_total_.load(std::memory_order_relaxed); }

    // Called by WorkerManager after endpoint progress reaches a terminal outcome.
    void worker_done(WorkerCompletion completion);

    // Called by Orchestrator after it pushes a newly-ready root task. ReadyQueue
    // has its own condition variable, but the Scheduler waits on completion_cv_.
    void notify_ready();

    // Mutex held by run() across each loop iteration's slot-touching body
    // (completion processing + dispatch). Orchestrator::release_run() acquires
    // it before optional Ring::reset_to_empty() compaction so the ring cannot
    // be torn down while the scheduler thread is mid-on_task_complete.
    std::mutex &loop_mutex() { return loop_mu_; }

private:
    Config cfg_;
    std::mutex loop_mu_;

    // Endpoint completion queue owned and drained by the Scheduler thread.
    std::queue<WorkerCompletion> completion_queue_;
    std::mutex completion_mu_;
    std::condition_variable completion_cv_;
    uint64_t wake_generation_{0};

    std::thread sched_thread_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    struct NextLevelGroupDispatchResult {
        std::unordered_set<int32_t> reserved_worker_ids;
        TaskSlot blocked_group_slot{INVALID_SLOT};
        std::vector<int32_t> busy_target_worker_ids;
        std::vector<int32_t> idle_queued_target_worker_ids;
        std::vector<TaskSlot> idle_queued_single_head_slots;
    };

    struct ReservationStallEpisode {
        TaskSlot group_slot{INVALID_SLOT};
        std::chrono::steady_clock::time_point started_at;
        bool reported{false};
    };

    std::atomic<uint64_t> dispatch_round_count_{0};
    // Tasks handed to a worker, counted at the one funnel every dispatch passes
    // through. Atomic for the same reason as the round count above: the loop
    // writes it on sched_thread_ while a reader is on another thread.
    std::atomic<uint64_t> dispatched_total_{0};
    // sched_thread_ owns this: update_reservation_stall() writes it under
    // loop_mu_ and reservation_stall_deadline() reads it under completion_mu_,
    // which is only race-free because both run on that one thread. start()
    // resets it before the thread exists. Any reader added off sched_thread_
    // needs a lock the two paths do not currently share.
    std::optional<ReservationStallEpisode> reservation_stall_episode_;

    void run();
    void on_task_complete(const WorkerCompletion &completion);
    void poison_task(TaskSlot slot, const std::string &root_message);

    void try_consume(TaskSlot slot);
    void dispatch_ready(bool scan_preparable);
    void dispatch_claimed(WorkerThread *worker, WorkerDispatch dispatch, bool prepared);
    void dispatch_preparable_next_level_singles();
    NextLevelGroupDispatchResult dispatch_next_level_group(const std::optional<RunId> &run_snapshot);
    bool dispatch_next_level_singles(
        const std::unordered_set<int32_t> &reserved_worker_ids, const std::optional<RunId> &run_snapshot,
        bool verify_group_barrier
    );
    void dispatch_sub_ready(const std::optional<RunId> &run_snapshot);
    void update_reservation_stall(const NextLevelGroupDispatchResult &dispatch_result);
    std::optional<std::chrono::steady_clock::time_point> reservation_stall_deadline() const;
};
