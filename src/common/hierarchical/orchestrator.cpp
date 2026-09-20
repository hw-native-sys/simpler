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

#include "orchestrator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "common/host_span_names.h"
#include "common/host_span_scope.h"
#include "worker_manager.h"

void Orchestrator::init(
    TensorMap *tensormap, Ring *allocator, Scope *scope, ReadyQueue *ready_sub_queue,
    NextLevelReadyQueues *ready_next_level_queues, WorkerManager *manager, std::function<void()> ready_notify_cb
) {
    tensormap_ = tensormap;
    allocator_ = allocator;
    scope_ = scope;
    ready_sub_queue_ = ready_sub_queue;
    ready_next_level_queues_ = ready_next_level_queues;
    manager_ = manager;
    ready_notify_cb_ = std::move(ready_notify_cb);
}

bool Orchestrator::is_terminal(RunPhase phase) { return phase == RunPhase::COMPLETED || phase == RunPhase::FAILED; }

bool Orchestrator::acceptance_ready(const std::shared_ptr<RunState> &run) {
    return run->submission_closed && (run->pending_accepts.load(std::memory_order_acquire) == 0 ||
                                      is_terminal(run->phase.load(std::memory_order_acquire)));
}

std::shared_ptr<RunState> Orchestrator::find_run(RunId run_id) const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    auto it = runs_.find(run_id);
    return it == runs_.end() ? nullptr : it->second;
}

std::shared_ptr<RunState> Orchestrator::get_run(RunId run_id) const {
    auto run = find_run(run_id);
    if (run == nullptr) throw std::invalid_argument("Orchestrator: unknown run id");
    return run;
}

std::shared_ptr<RunState> Orchestrator::current_building_run() const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    auto it = runs_.find(building_run_id_);
    if (building_run_id_ == INVALID_RUN_ID || it == runs_.end()) {
        throw std::logic_error("Orchestrator: task submission requires an active run");
    }
    return it->second;
}

RunId Orchestrator::begin_run() {
    RunId run_id = INVALID_RUN_ID;
    {
        std::unique_lock<std::mutex> lk(runs_mu_);
        if (building_run_id_ != INVALID_RUN_ID) {
            throw std::logic_error("Orchestrator::begin_run: another run is still building");
        }
        ++begin_run_waiters_;
        try {
            // Admission is bounded by the FIFO alone. A run takes no native
            // pipeline-slot lease here; `refresh_leases_locked` hands one out
            // when a run reaches a role that can use it.
            runs_cv_.wait(lk, [this] {
                return run_fifo_.size() < pending_run_limit_;
            });
        } catch (...) {
            --begin_run_waiters_;
            throw;
        }
        --begin_run_waiters_;
        // The pre-wait check was evaluated before the wait released the mutex,
        // so several callers can pass it and wake to find a builder published.
        if (building_run_id_ != INVALID_RUN_ID) {
            throw std::logic_error("Orchestrator::begin_run: another run is still building");
        }
        if (next_run_id_ == INVALID_RUN_ID || next_run_id_ == std::numeric_limits<RunId>::max()) {
            throw std::overflow_error("Orchestrator::begin_run: run id space exhausted");
        }
        run_id = next_run_id_++;
        bool map_published = false;
        bool fifo_published = false;
        try {
            // Generation zero is the reserved "no slot held" lease encoding.
            auto run = std::make_shared<RunState>(run_id, PipelineSlotLease{});
            bool inserted = runs_.emplace(run_id, run).second;
            if (!inserted) throw std::logic_error("Orchestrator::begin_run: duplicate run id");
            map_published = true;
            if (test_hook_) test_hook_(OrchestratorTestPoint::BEGIN_RUN_MAP_PUBLISHED);

            run_fifo_.push_back(run_id);
            fifo_published = true;
            if (test_hook_) test_hook_(OrchestratorTestPoint::BEGIN_RUN_FIFO_PUBLISHED);

            building_run_id_ = run_id;
            run->phase.store(RunPhase::BUILDING, std::memory_order_release);
        } catch (...) {
            if (fifo_published) run_fifo_.pop_back();
            if (map_published) runs_.erase(run_id);
            // No lease was taken, so none is returned. The notify still owes a
            // waiter the FIFO entry this rollback freed.
            runs_cv_.notify_all();
            throw;
        }
    }

    // The FIFO head may execute while its graph callback is still building.
    // Existing orchestration callbacks use this path to submit device work and
    // wait for L2 communication before returning. A successor remains gated by
    // active_run_id_ until the prior run reaches its terminal fence.
    activate_fifo_head();
    return run_id;
}

void Orchestrator::configure_pipeline_depth(uint32_t depth, uint32_t pending_depth) {
    if (depth == 0 || depth > PTO_PIPELINE_MAX_DEPTH) {
        throw std::invalid_argument("Orchestrator: pipeline depth is outside the supported range");
    }
    std::lock_guard<std::mutex> lk(runs_mu_);
    if (!runs_.empty() || building_run_id_ != INVALID_RUN_ID || active_run_id_ != INVALID_RUN_ID) {
        throw std::logic_error("Orchestrator: pipeline depth cannot change after admission starts");
    }
    admission_depth_ = depth;
    // Below `depth` is legal: it leaves the successor role unfillable, because
    // that role needs a second FIFO entry to exist, and admission serializes.
    // That is a caller's choice, not an error.
    pending_run_limit_ = pending_depth == 0 ? depth : pending_depth;
}

bool Orchestrator::acquire_lease_locked(const std::shared_ptr<RunState> &run, bool *assigned) {
    if (run->lease_released) return false;
    if (pipeline_slots_.owns(run->lease)) return true;
    std::optional<PipelineSlotLease> lease = pipeline_slots_.try_acquire(admission_depth_);
    if (!lease.has_value()) return false;

    // Publishing the lease and restamping the slots already registered under it
    // is one step under `completion_mu`, the same mutex `register_run_slot`
    // holds. A builder registering concurrently therefore either registers
    // before this and is restamped here, or registers after and reads the new
    // lease — no ordering leaves a slot on the invalid encoding while its run
    // owns a pipeline slot.
    {
        std::lock_guard<std::mutex> run_lk(run->completion_mu);
        run->lease = *lease;
        for (TaskSlot slot : run->task_slots) {
            TaskSlotState *state = allocator_->slot_state(slot);
            if (state != nullptr) state->pipeline_lease = run->lease;
        }
    }
    if (assigned != nullptr) *assigned = true;
    return true;
}

bool Orchestrator::refresh_leases_locked() {
    bool assigned = false;
    if (run_fifo_.empty()) return assigned;
    auto head = runs_.find(run_fifo_.front());
    if (head == runs_.end()) return assigned;
    if (is_terminal(head->second->phase.load(std::memory_order_acquire))) return assigned;
    if (!acquire_lease_locked(head->second, &assigned)) return assigned;
    if (run_fifo_.size() < 2) return assigned;
    auto successor = runs_.find(run_fifo_[1]);
    if (successor == runs_.end()) return assigned;
    // Only a closed successor can be prepared, the same condition
    // `preparable_run_id` reports on.
    if (successor->second->phase.load(std::memory_order_acquire) != RunPhase::PREPARED) return assigned;
    (void)acquire_lease_locked(successor->second, &assigned);
    return assigned;
}

void Orchestrator::finish_run_if_ready(const std::shared_ptr<RunState> &run) {
    bool notify = false;
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        RunPhase phase = run->phase.load(std::memory_order_acquire);
        if (!run->submission_closed || run->active_tasks.load(std::memory_order_acquire) != 0 || is_terminal(phase) ||
            (phase != RunPhase::EXECUTING && !run->submission_failed)) {
            return;
        }
        bool failed = run->submission_failed || static_cast<bool>(run->first_error);
        run->trace_terminal_ns = simpler::host_trace::enabled() ? simpler::host_trace::now_ns() : 0;
        run->phase.store(failed ? RunPhase::FAILED : RunPhase::COMPLETED, std::memory_order_release);
        notify = true;
    }
    if (!notify) return;
    run->completion_cv.notify_all();
    retire_terminal_run(run);
}

void Orchestrator::clear_run_ready_queues(RunId run_id) {
    if (ready_sub_queue_ != nullptr) ready_sub_queue_->erase_run(run_id);
    if (ready_next_level_queues_ != nullptr) ready_next_level_queues_->erase_run(run_id);
}

void Orchestrator::retire_terminal_run(const std::shared_ptr<RunState> &run) {
    bool leased_a_successor = false;
    {
        std::lock_guard<std::mutex> lk(runs_mu_);
        if (active_run_id_ == run->id) active_run_id_ = INVALID_RUN_ID;
        auto pos = std::find(run_fifo_.begin(), run_fifo_.end(), run->id);
        if (pos != run_fifo_.end()) run_fifo_.erase(pos);
        // The lease goes back under runs_mu_, not after it. The FIFO entry this
        // erase frees is what begin_run's wait predicate tests, so a release or
        // an erase that lands outside this mutex can fall between that
        // evaluation and the waiter registering on the condition variable — the
        // notify finds nobody and the capacity is stranded for good. The pool
        // takes its own mutex under this one, the order acquisition uses too.
        //
        // Only a lease this run holds is returned; a run that never reached an
        // active or preparable role has the invalid encoding.
        if (!run->lease_released) {
            run->lease_released = true;
            if (run->lease.generation != 0) pipeline_slots_.release(run->lease);
        }
        // `activate_fifo_head` below returns early while a run is active, which
        // is exactly the case where a successor handed its lease back.
        leased_a_successor = refresh_leases_locked();
    }
    clear_run_ready_queues(run->id);
    runs_cv_.notify_all();
    // A run that gained a lease here becomes preparable with no later event to
    // announce it.
    if (leased_a_successor && ready_notify_cb_) ready_notify_cb_();
    activate_fifo_head();
}

void Orchestrator::activate_fifo_head() {
    std::shared_ptr<RunState> run;
    {
        std::lock_guard<std::mutex> lk(runs_mu_);
        if (active_run_id_ != INVALID_RUN_ID || run_fifo_.empty()) return;
        auto it = runs_.find(run_fifo_.front());
        if (it == runs_.end()) return;
        run = it->second;
        RunPhase phase = run->phase.load(std::memory_order_acquire);
        if (phase != RunPhase::BUILDING && phase != RunPhase::PREPARED) return;
        // A head with no slot is not dispatchable, so promoting it to
        // EXECUTING would strand the scheduler on a run it cannot reach.
        if (!acquire_lease_locked(run)) {
            run.reset();
            return;
        }
        active_run_id_ = run->id;
        run->phase.store(RunPhase::EXECUTING, std::memory_order_release);
        refresh_leases_locked();
    }
    // Direct device control waits on this, not only admission does: a prepared
    // successor blocked in copy_to must wake when it becomes the active run.
    runs_cv_.notify_all();
    if (ready_notify_cb_) ready_notify_cb_();
    finish_run_if_ready(run);
}

void Orchestrator::close_run_submission(RunId run_id) {
    auto run = get_run(run_id);
    {
        std::lock_guard<std::mutex> runs_lk(runs_mu_);
        if (building_run_id_ != run_id) {
            throw std::logic_error("Orchestrator::close_run_submission: run is not building");
        }
        building_run_id_ = INVALID_RUN_ID;
    }
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        if (run->submission_closed) {
            throw std::logic_error("Orchestrator::close_run_submission: submission already closed");
        }
        run->submission_closed = true;
        if (run->phase.load(std::memory_order_acquire) == RunPhase::BUILDING) {
            run->phase.store(RunPhase::PREPARED, std::memory_order_release);
        }
    }
    run->completion_cv.notify_all();
    {
        // Before the wake, not after: the scheduler's wake is a generation
        // counter, so one consumed while the successor is still lease-less is
        // spent, and `activate_fifo_head` below issues none while a run is
        // active.
        std::lock_guard<std::mutex> runs_lk(runs_mu_);
        refresh_leases_locked();
    }
    if (ready_notify_cb_) ready_notify_cb_();
    activate_fifo_head();
    finish_run_if_ready(run);
}

void Orchestrator::fail_run_submission(RunId run_id, std::exception_ptr error) {
    auto run = get_run(run_id);
    std::string message = "graph construction failed";
    if (error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception &e) {
            message = e.what();
        } catch (...) {}
    }
    if (error) record_run_error(run_id, std::move(error));
    {
        std::lock_guard<std::mutex> runs_lk(runs_mu_);
        // Failing a run closes it out from whatever state submission reached,
        // so the fence always becomes reachable. Refusing a run that is no
        // longer building would leave its waiter blocked forever.
        if (building_run_id_ == run_id) building_run_id_ = INVALID_RUN_ID;
    }
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        run->submission_failed = true;
        run->submission_closed = true;
    }
    // The FIFO head may already be executing while graph construction is
    // open. Cancel only slots that have not started; concurrently running work
    // remains part of the failed run's terminal fence.
    cancel_unstarted_run(run, message);
    run->completion_cv.notify_all();
    finish_run_if_ready(run);
}

void Orchestrator::wait_run_accepted(RunId run_id) {
    // A released run is past acceptance by definition, so a waiter that raced
    // the run's completion returns instead of throwing.
    auto run = find_run(run_id);
    if (run == nullptr) return;
    std::unique_lock<std::mutex> lk(run->completion_mu);
    run->completion_cv.wait(lk, [&run] {
        return acceptance_ready(run);
    });
}

bool Orchestrator::run_accepted(RunId run_id) const {
    auto run = get_run(run_id);
    std::lock_guard<std::mutex> lk(run->completion_mu);
    return acceptance_ready(run);
}

void Orchestrator::wait_run(RunId run_id) {
    auto run = get_run(run_id);
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lk(run->completion_mu);
        run->completion_cv.wait(lk, [&run] {
            return is_terminal(run->phase.load(std::memory_order_acquire));
        });
        error = run->first_error;
    }
    if (error) std::rethrow_exception(error);
}

bool Orchestrator::wait_run_for(RunId run_id, double timeout_seconds) {
    auto run = get_run(run_id);
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lk(run->completion_mu);
        if (!run->completion_cv.wait_for(lk, std::chrono::duration<double>(timeout_seconds), [&run] {
                return is_terminal(run->phase.load(std::memory_order_acquire));
            })) {
            return false;
        }
        error = run->first_error;
    }
    if (error) std::rethrow_exception(error);
    return true;
}

bool Orchestrator::run_done(RunId run_id) const {
    return is_terminal(get_run(run_id)->phase.load(std::memory_order_acquire));
}

bool Orchestrator::run_failed(RunId run_id) const {
    auto run = get_run(run_id);
    std::lock_guard<std::mutex> lk(run->completion_mu);
    return run->submission_failed || static_cast<bool>(run->first_error);
}

bool Orchestrator::dispatchable_locked(RunId run_id) const {
    if (active_run_id_ != run_id) return false;
    auto it = runs_.find(run_id);
    return it != runs_.end() && it->second->phase.load(std::memory_order_acquire) == RunPhase::EXECUTING &&
           pipeline_slots_.owns(it->second->lease);
}

bool Orchestrator::can_dispatch_run(RunId run_id) const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    return dispatchable_locked(run_id);
}

RunId Orchestrator::dispatchable_run_id() const {
    // The scheduler asks for a run to dispatch, not merely which run is at the
    // FIFO head: a head whose lease has been released is not dispatchable, and
    // answering with its id would send the scheduler looking for its tasks.
    std::lock_guard<std::mutex> lk(runs_mu_);
    return dispatchable_locked(active_run_id_) ? active_run_id_ : INVALID_RUN_ID;
}

RunId Orchestrator::active_run_id() const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    return active_run_id_;
}

RunId Orchestrator::preparable_run_id() const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    if (!dispatchable_locked(active_run_id_) || run_fifo_.size() < 2 || run_fifo_.front() != active_run_id_) {
        return INVALID_RUN_ID;
    }
    auto it = runs_.find(run_fifo_[1]);
    if (it == runs_.end() || it->second->phase.load(std::memory_order_acquire) != RunPhase::PREPARED ||
        !pipeline_slots_.owns(it->second->lease)) {
        return INVALID_RUN_ID;
    }
    return it->first;
}

bool Orchestrator::quiescent_locked() const { return runs_.empty() && building_run_id_ == INVALID_RUN_ID; }

void Orchestrator::compact_if_quiescent() {
    // `begin_run` takes only runs_mu_, and no slot is allocated without a
    // building run, so holding runs_mu_ across both the test and the reset is
    // what stops a run registered mid-compaction from losing its first slot.
    // Lock order is the scheduler loop mutex then runs_mu_, matching the
    // scheduler's on_consumed path.
    std::unique_lock<std::mutex> sched_lk;
    if (sched_loop_mu_ != nullptr) sched_lk = std::unique_lock<std::mutex>(*sched_loop_mu_);
    std::lock_guard<std::mutex> runs_lk(runs_mu_);
    if (quiescent_locked() && allocator_->active_count() == 0) allocator_->reset_to_empty();
}

void Orchestrator::release_run(RunId run_id) {
    int64_t terminal_ns = 0;
    PipelineSlotLease lease{};
    {
        std::lock_guard<std::mutex> lk(runs_mu_);
        auto it = runs_.find(run_id);
        if (it == runs_.end()) throw std::invalid_argument("Orchestrator::release_run: unknown run id");
        if (!is_terminal(it->second->phase.load(std::memory_order_acquire))) {
            throw std::logic_error("Orchestrator::release_run: run is not terminal");
        }
        terminal_ns = it->second->trace_terminal_ns;
        lease = it->second->lease;
        runs_.erase(it);
    }

    compact_if_quiescent();
#if SIMPLER_HOST_STRACE
    if (terminal_ns != 0 && simpler::host_trace::enabled()) {
        std::ostringstream retire_attrs;
        retire_attrs << "run_id=" << run_id << " slot_id=" << lease.slot_id << " generation=" << lease.generation
                     << " role=facade";
        simpler::host_trace::emit(
            simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::PostFenceRetirement), run_id, 0, 0,
            terminal_ns, simpler::host_trace::now_ns() - terminal_ns, retire_attrs.str().c_str()
        );
    }
#endif
}

void Orchestrator::register_run_slot(const std::shared_ptr<RunState> &run, TaskSlot slot) {
    std::lock_guard<std::mutex> lk(run->completion_mu);
    // Registration and the identity stamp are one step, paired with the
    // publication in `acquire_lease_locked`: whichever of the two runs second
    // sees the other's effect.
    TaskSlotState *state = allocator_->slot_state(slot);
    if (state != nullptr) state->pipeline_lease = run->lease;
    run->task_slots.push_back(slot);
    run->active_tasks.fetch_add(1, std::memory_order_relaxed);
}

void Orchestrator::cancel_unstarted_run(const std::shared_ptr<RunState> &run, const std::string &message) {
    std::vector<TaskSlot> slots;
    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        slots.assign(run->task_slots.begin(), run->task_slots.end());
    }

    // Only the slots this call actually cancelled. A slot already RUNNING is
    // owned by the device: it keeps reading its producers' outputs until it
    // completes, and its normal completion path releases those references.
    //
    // A slot still BUILDING here is one whose submit threw part-way through
    // wiring: submission is closed before this runs, so no submitting thread
    // is left to publish it and this call owns its propagation. That is the
    // opposite of the scheduler's poison path, which can hit a slot whose
    // submit is still in flight and must leave it alone.
    struct FailurePropagation {
        TaskSlot slot{INVALID_SLOT};
        std::vector<TaskSlot> producers;
        bool owns_releases{false};
    };
    std::vector<FailurePropagation> cancelled;
    cancelled.reserve(slots.size());
    for (TaskSlot slot : slots) {
        TaskSlotState &state = slot_state(slot);
        std::optional<TaskState> claimed = claim_task_failure(state, message);
        if (claimed.has_value() && *claimed != TaskState::BUILDING) {
            // Cancellation is the sole propagation owner for a READY/PENDING
            // claim it won. Publish retryable debt before any preparation can
            // throw; scheduler-owned claims deliberately expose no such debt.
            state.failure_propagation_pending.store(true, std::memory_order_release);
        }
        // A slot a producer already claimed mid-wiring is FAILED but not
        // propagated: its submit was supposed to finish the job and threw
        // instead. Taking that debt over is what keeps its references from
        // stranding the run fence — skipping it for being FAILED would not.
        // The debt is only observed here, not settled: everything up to the
        // first reference release can throw, and a debt cleared before that
        // point leaves a FAILED slot every later attempt skips.
        bool owed = state.failure_propagation_pending.load(std::memory_order_acquire);
        if (!claimed.has_value() && !owed) continue;
        mark_group_members_skipped(state, message);
        if (test_hook_) test_hook_(OrchestratorTestPoint::FAILURE_FANIN_SNAPSHOT);
        FailurePropagation propagation;
        propagation.slot = slot;
        {
            std::lock_guard<std::mutex> lk(state.fanout_mu);
            propagation.producers = state.fanin_producers;
        }
        cancelled.push_back(std::move(propagation));
    }
    // Every step that could still fail recoverably is done, and each of them is
    // idempotent, so an exception above leaves the debt for a retry to resume
    // from. Cancellation owns every debt in this batch exclusively; the
    // exchange settles each debt immediately before its non-repeatable
    // releases.
    for (FailurePropagation &propagation : cancelled) {
        propagation.owns_releases = commit_failure_propagation(slot_state(propagation.slot));
    }
    for (const FailurePropagation &propagation : cancelled) {
        if (propagation.owns_releases) try_consume(propagation.slot);
    }
    // Releasing a producer reference held by a slot that is still RUNNING would
    // let the producer reach CONSUMED — and its HeapRing output be reclaimed —
    // while the device is still reading it, and its real completion would then
    // release the same reference a second time.
    for (const FailurePropagation &propagation : cancelled) {
        if (!propagation.owns_releases) continue;
        for (TaskSlot producer : propagation.producers)
            try_consume(producer);
    }
}

void Orchestrator::decrement_run_tasks(RunId run_id) {
    auto run = find_run(run_id);
    if (run == nullptr) return;
    int32_t remaining = run->active_tasks.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining < 0) {
        // Reaching the fence is what lets the caller observe the failure, so a
        // count mismatch is surfaced as the run's error rather than thrown:
        // this runs on the scheduler thread, where an escaping exception is
        // fatal to the process.
        run->active_tasks.store(0, std::memory_order_release);
        record_run_error(run, std::make_exception_ptr(std::logic_error("Orchestrator: run task count underflow")));
    }
    if (remaining <= 0) finish_run_if_ready(run);
}

void Orchestrator::decrement_run_accepts(RunId run_id) {
    auto run = find_run(run_id);
    if (run == nullptr) return;
    bool notify = false;
    bool underflow = false;
    {
        // acceptance_ready() is evaluated under this mutex, so the decrement
        // that can satisfy it happens under the same one — otherwise it lands
        // between the waiter's predicate check and its block, and the notify
        // below is lost.
        std::lock_guard<std::mutex> lk(run->completion_mu);
        int32_t remaining = run->pending_accepts.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (remaining < 0) {
            run->pending_accepts.store(0, std::memory_order_release);
            underflow = true;
            remaining = 0;
        }
        notify = remaining == 0;
    }
    // record_run_error takes completion_mu, so it runs outside the block above.
    if (underflow) {
        record_run_error(
            run, std::make_exception_ptr(std::logic_error("Orchestrator: run acceptance count underflow"))
        );
    }
    if (notify) run->completion_cv.notify_all();
}

void Orchestrator::record_run_error(const std::shared_ptr<RunState> &run, std::exception_ptr error) {
    if (run == nullptr || !error) return;
    std::lock_guard<std::mutex> lk(run->completion_mu);
    if (!run->first_error) run->first_error = std::move(error);
}

void Orchestrator::record_run_error(RunId run_id, std::exception_ptr error) {
    if (!error) return;
    record_run_error(find_run(run_id), std::move(error));
}

bool Orchestrator::current_building_run_failed_for_test() const {
    auto run = current_building_run();
    std::lock_guard<std::mutex> lk(run->completion_mu);
    return static_cast<bool>(run->first_error);
}

size_t Orchestrator::begin_run_waiter_count_for_test() const {
    std::lock_guard<std::mutex> lk(runs_mu_);
    return begin_run_waiters_;
}

void Orchestrator::report_task_error(TaskSlot slot, const std::string &message) {
    TaskSlotState &task = slot_state(slot);
    record_run_error(task.run_id, std::make_exception_ptr(std::runtime_error(message)));
}

void Orchestrator::mark_task_accepted(TaskSlot slot) {
    // Reached from a WorkerThread, where an escaping exception is fatal to the
    // process, so an unknown slot degrades to "no fence advance" instead of
    // throwing; the dispatch's completion still reports the failure.
    TaskSlotState *task = allocator_->slot_state(slot);
    if (task == nullptr) return;
    decrement_run_accepts(task->run_id);
}

void Orchestrator::await_run_admission(RunId run_id) {
    // Direct device control — malloc / free / copy_* / domain and region
    // creation — reaches a child without a TaskSlot, so the ready-queue FIFO
    // does not order it and the mailbox mutex only serialises one command at a
    // time. A prepared successor that freed or overwrote child memory the
    // active run is still reading would break the whole-run ordering the FIFO
    // exists to provide.
    //
    // The run is named by the caller rather than read from building_run_id_:
    // that field says a graph callback is open *somewhere*, not that this
    // thread is inside it, and a public Worker.copy_* on another thread must
    // not be charged to whichever run happens to be building.
    if (run_id == INVALID_RUN_ID) return;
    std::unique_lock<std::mutex> lk(runs_mu_);
    runs_cv_.wait(lk, [this, run_id] {
        if (active_run_id_ == run_id) return true;
        auto it = runs_.find(run_id);
        return it == runs_.end() || is_terminal(it->second->phase.load(std::memory_order_acquire));
    });
}

uint64_t Orchestrator::committed_device_memory(int worker_id) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::committed_device_memory: invalid worker_id");
    return wt->control_committed_device_memory();
}

DeviceMemoryInfo Orchestrator::device_memory_info(int worker_id) {
    auto *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
    if (!wt) throw std::runtime_error("Orchestrator::device_memory_info: invalid worker_id");
    return wt->control_device_memory_info();
}

TaskSlotState &Orchestrator::slot_state(TaskSlot s) {
    TaskSlotState *p = allocator_->slot_state(s);
    if (!p) throw std::runtime_error("Orchestrator::slot_state: invalid slot id");
    return *p;
}

// ---------------------------------------------------------------------------
// alloc(shape, dtype) — user-facing intermediate buffer from the HeapRing
// ---------------------------------------------------------------------------

uint64_t Orchestrator::output_alloc_bytes(const ChipTensor &t) { return align_up(t.nbytes(), HEAP_ALIGN); }

uint64_t Orchestrator::alloc(const std::vector<uint32_t> &shape, DataType dtype, const CanonicalIdentity &identity) {
    auto run = current_building_run();
    if (shape.empty()) {
        // Rank-0 tensors are not supported across the ABI, and a buffer no
        // consumer can name is not worth allocating or registering.
        throw std::invalid_argument("Orchestrator::alloc: shape must have at least one dimension");
    }
    if (shape.size() > MAX_TENSOR_DIMS) {
        throw std::invalid_argument("Orchestrator::alloc: shape exceeds MAX_TENSOR_DIMS");
    }

    uint64_t numel = 1;
    for (uint32_t d : shape)
        numel *= static_cast<uint64_t>(d);
    uint64_t bytes = numel * get_element_size(dtype);
    uint64_t aligned = align_up(bytes, HEAP_ALIGN);

    // Inherit the caller's scope depth so alloc buffers land in the same
    // ring as any tasks submitted inside that scope — an alloc inside a
    // nested `with orch.scope():` uses the nested ring and reclaims
    // independently of the outer ring (Strict-1).
    AllocResult ar = allocator_->alloc(aligned, scope_->current_depth());
    if (ar.slot == INVALID_SLOT) {
        throw std::runtime_error("Orchestrator::alloc: allocator shutdown");
    }

    TaskSlotState &s = slot_state(ar.slot);
    s.reset();
    s.run_id = run->id;
    // From the moment the run owns this slot until COMPLETED publication, a
    // failed alloc must remain claimable by run cancellation. FREE would make
    // cancellation skip the registered slot and strand active_tasks forever.
    s.state.store(TaskState::BUILDING, std::memory_order_release);
    try {
        if (test_hook_) test_hook_(OrchestratorTestPoint::ALLOC_RUN_SLOT_REGISTERING);
        register_run_slot(run, ar.slot);
    } catch (...) {
        // Registration itself has strong ownership semantics: push_back either
        // published the slot or threw without changing the run. On the latter
        // path no cancellation can discover it, so release the Ring claim here.
        s.state.store(TaskState::CONSUMED, std::memory_order_release);
        allocator_->release(ar.slot);
        throw;
    }

    // No fanin — alloc has no work to wait on.
    s.fanin_count.store(0, std::memory_order_relaxed);
    s.fanin_released.store(0, std::memory_order_relaxed);

    // Initial fanout_total = scope_ref. Tensor-derived consumers that retain
    // this slot increment fanout_total in infer_deps.
    const int32_t scope_ref = (scope_->depth() > 0) ? 1 : 0;
    if (scope_ref > 0) {
        // Register before charging fanout_total. If registration throws, scope
        // teardown cannot release this slot, so charging first would leave a
        // reference neither scope_end nor cancellation can satisfy.
        scope_->register_task(ar.slot);
        if (test_hook_) test_hook_(OrchestratorTestPoint::SCOPE_REGISTERED);
    }
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanout_total = scope_ref;
    }

    uint64_t ptr = reinterpret_cast<uint64_t>(ar.heap_ptr);
    // A 0-byte request has no buffer for anyone to depend on, so registering a
    // mapping would only wire a consumer to a backing that does not exist.
    if (ptr != 0) {
        // A ChipTensor over this VA carries the same identity, so keying on the
        // identity's canonical hash is what lets infer_deps resolve it here.
        TensorKey key = TensorKey::local_host(CanonicalIdentityHash{}(identity));
        // Prepare the cleanup journal before publishing the TensorMap entry.
        // A failed insert then leaves either no mapping or a mapping cancellation
        // can erase; the reverse order can leak an unjournaled producer entry.
        s.output_keys.push_back(key);
        if (test_hook_) test_hook_(OrchestratorTestPoint::ALLOC_OUTPUT_KEY_PREPARED);
        // The allocation covers its whole backing, which is what a default TensorFootprint
        // spans — so every view a consumer later takes of it resolves back to this slot.
        tensormap_->insert(run->id, key, TensorFootprint{}, ar.slot);
    }

    // Simulate the self try_consume that on_task_complete would normally
    // contribute for a slot that ran through the scheduler. Without this
    // bump, the fanout-release threshold (`>= total + 1`) would be one
    // short and the slot would never reach CONSUMED.
    // Publish it only after every fallible BUILDING step: a failed alloc gets
    // its terminal self release from cancellation instead.
    s.fanout_released.store(1, std::memory_order_relaxed);

    s.state.store(TaskState::COMPLETED, std::memory_order_release);
    return ptr;
}

// =============================================================================
// User-facing submit_* — thin wrappers around submit_impl
// =============================================================================

SubmitResult Orchestrator::submit_next_level(
    const CallableIdentity &callable, const TaskArgs &args, const CallConfig &config, int32_t worker_id,
    const std::vector<int32_t> &eligible_worker_ids, const RemoteTaskArgsSidecar &remote_sidecar
) {
    std::vector<int32_t> target_worker_ids{worker_id};
    std::vector<std::vector<int32_t>> worker_id_sets;
    if (!eligible_worker_ids.empty()) worker_id_sets = {eligible_worker_ids};
    std::vector<RemoteTaskArgsSidecar> sidecars;
    if (!remote_sidecar.tensors.empty() || !remote_sidecar.inline_payload.empty()) sidecars = {remote_sidecar};
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, {args}, std::move(target_worker_ids), std::move(worker_id_sets),
        std::move(sidecars)
    );
}

SubmitResult Orchestrator::submit_next_level_group(
    const CallableIdentity &callable, const std::vector<TaskArgs> &args_list, const CallConfig &config,
    const std::vector<int32_t> &worker_ids, const std::vector<std::vector<int32_t>> &eligible_worker_ids,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
) {
    return submit_impl(
        WorkerType::NEXT_LEVEL, callable, config, args_list, worker_ids, eligible_worker_ids, remote_sidecars
    );
}

SubmitResult Orchestrator::submit_sub(const CallableIdentity &callable, const TaskArgs &args) {
    return submit_impl(WorkerType::SUB, callable, CallConfig{}, {args});
}

SubmitResult Orchestrator::submit_sub_group(const CallableIdentity &callable, const std::vector<TaskArgs> &args_list) {
    return submit_impl(WorkerType::SUB, callable, CallConfig{}, args_list);
}

// =============================================================================
// submit_impl — shared 7-step submit machinery
// =============================================================================

SubmitResult Orchestrator::submit_impl(
    WorkerType worker_type, const CallableIdentity &callable, const CallConfig &config, std::vector<TaskArgs> args_list,
    std::vector<int32_t> target_worker_ids, std::vector<std::vector<int32_t>> eligible_worker_ids,
    std::vector<RemoteTaskArgsSidecar> remote_sidecars
) {
    auto run = current_building_run();
    if (args_list.empty()) throw std::invalid_argument("Orchestrator: args_list must not be empty");
    config.validate();
    validate_worker_eligibility(worker_type, args_list.size(), target_worker_ids, eligible_worker_ids);
    validate_remote_sidecars(args_list, remote_sidecars, eligible_worker_ids);
    validate_explicit_deps(run->id, args_list);
    validate_submit_args(args_list);

    {
        std::lock_guard<std::mutex> lk(run->completion_mu);
        if (run->first_error) std::rethrow_exception(run->first_error);
    }

    // --- Step 1: Atomically claim slot + auto-alloc any OUTPUT tensors that
    // arrived with a null data pointer. Both resources come from the same
    // merged allocator (Strict-2) so there is no partial-failure rollback
    // path.
    AllocResult ar = reserve_outputs_and_slot(args_list, remote_sidecars);
    if (ar.slot == INVALID_SLOT) {
        throw std::runtime_error("Orchestrator: allocator shutdown");
    }
    TaskSlot slot = ar.slot;

#if SIMPLER_HOST_STRACE
    std::optional<simpler::host_trace::SpanScope> submit_trace;
    if (worker_type == WorkerType::NEXT_LEVEL && simpler::host_trace::enabled()) {
        uint64_t trace_callable_hash = 0;
        std::memcpy(&trace_callable_hash, callable.digest.data(), sizeof(trace_callable_hash));
        std::ostringstream trace_attrs;
        trace_attrs << "run_id=" << run->id << " task_slot=" << slot
                    << " group_index=" << (args_list.size() == 1 ? 0 : -1) << " group_size=" << args_list.size();
        if (target_worker_ids.size() == 1) trace_attrs << " worker_id=" << target_worker_ids.front();
        trace_attrs << " role=facade";
        submit_trace.emplace(
            simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::Submit), run->id, trace_callable_hash, 0,
            trace_attrs.str()
        );
    }
#endif

    TaskSlotState &s = slot_state(slot);
    s.reset();
    s.run_id = run->id;
    // BUILDING until the publication at the end of this function. Step 2
    // registers this slot's outputs in the tensormap and Step 5 appends it to
    // its producers' fanout lists, so it is observable well before its
    // bookkeeping is final; BUILDING is what tells every other thread the
    // counters below cannot be acted on yet.
    s.state.store(TaskState::BUILDING, std::memory_order_release);
    try {
        if (test_hook_) test_hook_(OrchestratorTestPoint::SUBMIT_RUN_SLOT_REGISTERING);
        register_run_slot(run, slot);
    } catch (...) {
        // The run never acquired this slot, so cancellation cannot discover
        // it. Return the combined slot/HeapRing reservation directly.
        s.state.store(TaskState::CONSUMED, std::memory_order_release);
        allocator_->release(slot);
        throw;
    }
    // The scope reference is registered before it is charged, and both happen
    // before anything below can throw. Charging first would leave a slot owing
    // a release that no scope_end can make: cancellation contributes only the
    // terminal release, the threshold is never reached, the slot never reaches
    // CONSUMED, and the run's task count — and its fence — never resolve.
    // fanout_total is charged here rather than later for the opposite reason:
    // once Step 2 publishes this slot's outputs, a retaining consumer can wire
    // onto it and increment fanout_total, and a later assignment would drop that.
    int32_t scope_ref = (scope_->depth() > 0) ? 1 : 0;
    if (scope_ref > 0) {
        scope_->register_task(slot);
        if (test_hook_) test_hook_(OrchestratorTestPoint::SCOPE_REGISTERED);
    }
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanout_total = scope_ref;
    }

    s.worker_type = worker_type;
    s.callable = callable;
    s.config = config;
    // --- Step 2: Walk tags → tensormap.lookup (deps) + tensormap.insert
    // (outputs). Must happen before we move args_list into the slot because
    // infer_deps reads tensor data pointers and tags from it.
    std::vector<ProducerDependency> producers;
    infer_deps(slot, args_list, target_worker_ids, remote_sidecars, producers, s.output_keys);

    // --- Step 3: Store TaskArgs directly (no chip-storage pre-build) ---
    // Dispatch builds a TaskArgsView on demand via `slot.args_view(i)`
    // (THREAD mode) or write_blob → read_blob (PROCESS mode). The L2 ABI
    // ChipStorageTaskArgs conversion now runs inside ChipWorker::run
    // rather than at submit time.
    std::vector<GroupMemberState> group_member_states;
    std::vector<EndpointOutcome> group_member_outcomes;
    if (args_list.size() > 1) {
        // Scheduler dispatch must not allocate after it publishes RUNNING: a
        // failure there would leave neither dispatch nor cancellation owning
        // the slot. Prepare both vectors while BUILDING and commit them only
        // after both allocations succeed.
        group_member_states.assign(args_list.size(), GroupMemberState::NOT_DISPATCHED);
        if (test_hook_) test_hook_(OrchestratorTestPoint::GROUP_MEMBER_STATES_PREPARED);
        group_member_outcomes.assign(args_list.size(), EndpointOutcome::SKIPPED);
    }
    if (args_list.size() == 1) {
        s.is_group_ = false;
        s.task_args = std::move(args_list.front());
        if (!remote_sidecars.empty()) s.remote_sidecar = std::move(remote_sidecars.front());
    } else {
        std::lock_guard<std::mutex> lk(s.group_mu);
        s.is_group_ = true;
        s.task_args_list = std::move(args_list);
        s.remote_sidecars = std::move(remote_sidecars);
        s.group_member_states.swap(group_member_states);
        s.group_member_outcomes.swap(group_member_outcomes);
    }
    s.target_worker_ids = std::move(target_worker_ids);

    // --- Step 5: Finalize fanin — lock each producer's fanout_mu, attach ---
    //
    // For COMPLETED producers (notably alloc-created synthetic slots), we
    // still wire the fanout edge so the producer waits for this consumer
    // before being CONSUMED (and freeing any owned buffers). The consumer
    // itself doesn't gain a live fanin — it can run immediately because the
    // producer is already done. CONSUMED producers are gone (resources freed),
    // so we skip them entirely.
    int32_t live_fanins = 0;
    bool poisoned_by_failed_producer = false;
    std::string poison_message;
    for (const ProducerDependency &dependency : producers) {
        TaskSlot prod = dependency.slot;
        TaskSlotState &ps = slot_state(prod);
        std::lock_guard<std::mutex> lk(ps.fanout_mu);

        TaskState ps_state = ps.state.load(std::memory_order_acquire);
        if (ps_state == TaskState::CONSUMED) {
            continue;
        }
        if (!dependency.retain && (ps_state == TaskState::COMPLETED || ps_state == TaskState::FAILED)) {
            if (ps_state == TaskState::FAILED) {
                poisoned_by_failed_producer = true;
                if (poison_message.empty()) poison_message = ps.failure_message;
            }
            continue;
        }
        ps.fanout_consumers.push_back(slot);
        if (dependency.retain) {
            ps.fanout_total++;
            try {
                if (test_hook_) test_hook_(OrchestratorTestPoint::PRODUCER_FORWARD_EDGE_PUBLISHED);
                s.fanin_producers.push_back(prod);
            } catch (...) {
                ps.fanout_total--;
                ps.fanout_consumers.pop_back();
                throw;
            }
        }
        if (ps_state == TaskState::FAILED) {
            poisoned_by_failed_producer = true;
            if (poison_message.empty()) poison_message = ps.failure_message;
        } else if (ps_state != TaskState::COMPLETED) {
            live_fanins++;
        }
    }

    // --- Step 6: Publication — the one point where the slot leaves BUILDING.
    //
    // fanin_count and the transition out of BUILDING are published together,
    // under the lock every completing producer takes to decide readiness. A
    // producer that completed while this slot was BUILDING already advanced
    // fanin_released and found nothing it could do with it; the comparison
    // here is what picks that release up. A producer that arrives afterwards
    // reads a count that is already published. Neither can act on one half.
    bool ready_now = false;
    if (!poisoned_by_failed_producer) {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        s.fanin_count.store(live_fanins, std::memory_order_release);
        TaskState expected = TaskState::BUILDING;
        bool satisfied = s.fanin_released.load(std::memory_order_acquire) >= live_fanins;
        TaskState published = satisfied ? TaskState::READY : TaskState::PENDING;
        if (s.state.compare_exchange_strong(
                expected, published, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            ready_now = satisfied;
            // Counted here, under the same lock and in the same step that
            // publishes the slot, because this branch is the only moment at
            // which the slot is certain to reach an endpoint. mark_task_accepted
            // is the sole matching decrement, and a slot that fails instead
            // never reaches one, so counting a failed slot would leave
            // pending_accepts permanently above zero: the run would still
            // finish, since acceptance_ready() also accepts a terminal phase,
            // but the fence would open at completion rather than at acceptance
            // and serialise the next submission behind the whole run. Counting
            // after the lock is released would be equally wrong in the other
            // direction — a published slot is visible to Scheduler::poison_task,
            // which can claim and fail it before the count lands.
            //
            // pending_accepts is incremented directly here; the run pointer is
            // already resolved, so this avoids taking runs_mu_ while holding
            // fanout_mu.
            run->pending_accepts.fetch_add(s.group_size(), std::memory_order_relaxed);
        } else {
            // Only a failure claim moves a BUILDING slot, and it stops there
            // precisely so this thread — the one that knows the wiring is
            // final — runs the propagation.
            poisoned_by_failed_producer = true;
        }
    }

    if (poisoned_by_failed_producer) {
        // Publishing our own failure goes through the shared claim, so exactly
        // one party writes failure_message. Losing the claim to a producer that
        // failed while we were wiring is fine: its message stands, and the
        // propagation is ours either way.
        (void)claim_task_failure(s, poison_message.empty() ? "producer task failed" : poison_message);
        std::string message;
        {
            std::lock_guard<std::mutex> lk(s.fanout_mu);
            message = s.failure_message;
        }
        if (message.empty()) message = "producer task failed";
        // Ordered so that everything which can throw runs while the propagation
        // debt is still owed, and every one of those steps is idempotent: group
        // marking skips already-terminal members, the error is first-wins, and
        // the producer list is a copy. An exception here therefore leaves a
        // FAILED slot whose debt run cancellation still sees, and it resumes
        // from the top. Settling the debt first would leave that slot FAILED
        // with nothing recorded, and cancellation skips exactly those.
        mark_group_members_skipped(s, message);
        record_run_error(run->id, std::make_exception_ptr(std::runtime_error(message)));
        std::vector<TaskSlot> fanin_producers;
        {
            std::lock_guard<std::mutex> lk(s.fanout_mu);
            fanin_producers = s.fanin_producers;
        }
        // Past the resumable part: the releases below are the propagation, and
        // repeating one of them would release a reference twice. Submission is
        // still open here, so cancellation cannot take the BUILDING debt over
        // concurrently.
        if (commit_failure_propagation(s)) {
            try_consume(slot);
            for (TaskSlot prod : fanin_producers) {
                try_consume(prod);
            }
        }
        return SubmitResult{run->id, slot};
    }

    // Enqueued outside the publication lock: enqueue_ready takes runs_mu_, and
    // a producer completing between the two finds the slot already READY and
    // does nothing, so the queue receives it exactly once.
    if (ready_now) {
        enqueue_ready(slot);
        if (ready_notify_cb_) ready_notify_cb_();
    }

    return SubmitResult{run->id, slot};
}

void Orchestrator::enqueue_ready(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);
    if (s.worker_type == WorkerType::NEXT_LEVEL && ready_next_level_queues_ == nullptr) {
        throw std::runtime_error("Orchestrator::enqueue_ready: NEXT_LEVEL queues are not initialized");
    }
    // retire_terminal_run erases this run's partitions, and the scheduler only
    // ever reads the partition of the run holding the FIFO head — so a push
    // that lands after the erase builds one nothing will drain. Holding
    // runs_mu_ across both the test and the push is what makes them one
    // decision: releasing it in between lets retire erase the partition we
    // just decided was live. Every requeue route comes through here for the
    // same reason.
    std::lock_guard<std::mutex> lk(runs_mu_);
    auto it = runs_.find(s.run_id);
    if (it == runs_.end() || is_terminal(it->second->phase.load(std::memory_order_acquire))) return;
    if (s.worker_type == WorkerType::NEXT_LEVEL) {
        if (s.is_group()) {
            ready_next_level_queues_->push_group(s.run_id, slot);
        } else {
            ready_next_level_queues_->push_single(s.target_worker_id(0), s.run_id, slot);
        }
        return;
    }
    ready_sub_queue_->push(s.run_id, slot);
}

void Orchestrator::validate_worker_eligibility(
    WorkerType worker_type, size_t args_count, const std::vector<int32_t> &target_worker_ids,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids
) const {
    if (worker_type == WorkerType::SUB) {
        if (!target_worker_ids.empty() || !eligible_worker_ids.empty()) {
            throw std::invalid_argument("Orchestrator: SUB tasks do not accept worker-selection metadata");
        }
        return;
    }

    if (target_worker_ids.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: NEXT_LEVEL target count " + std::to_string(target_worker_ids.size()) +
            " does not match args length " + std::to_string(args_count)
        );
    }
    if (!eligible_worker_ids.empty() && eligible_worker_ids.size() != args_count) {
        throw std::invalid_argument(
            "Orchestrator: eligible worker-id set length " + std::to_string(eligible_worker_ids.size()) +
            " does not match args length " + std::to_string(args_count)
        );
    }

    std::unordered_set<int32_t> unique_targets;
    for (int32_t worker_id : target_worker_ids) {
        if (worker_id < 0) {
            throw std::invalid_argument("Orchestrator: NEXT_LEVEL worker id must be non-negative");
        }
        if (!unique_targets.insert(worker_id).second) {
            throw std::invalid_argument("Orchestrator: duplicate NEXT_LEVEL worker id " + std::to_string(worker_id));
        }
    }

    const std::vector<int32_t> empty_eligible;
    for (size_t i = 0; i < args_count; ++i) {
        const auto &eligible = eligible_worker_ids.empty() ? empty_eligible : eligible_worker_ids[i];
        if (!eligible_worker_ids.empty() && eligible.empty()) {
            throw std::invalid_argument(
                "Orchestrator: final eligible worker-id set is empty for member " + std::to_string(i)
            );
        }
        if (manager_ != nullptr) {
            for (int32_t worker_id : eligible) {
                if (manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id) == nullptr) {
                    throw std::invalid_argument(
                        "Orchestrator: eligible worker-id " + std::to_string(worker_id) + " is not a registered worker"
                    );
                }
            }
            if (manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, target_worker_ids[i]) == nullptr) {
                throw std::invalid_argument(
                    "Orchestrator: target worker " + std::to_string(target_worker_ids[i]) +
                    " is not a registered worker"
                );
            }
        }

        if (!eligible_worker_ids.empty()) {
            bool allowed = false;
            for (int32_t id : eligible) {
                if (id == target_worker_ids[i]) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                throw std::invalid_argument(
                    "Orchestrator: target worker " + std::to_string(target_worker_ids[i]) +
                    " is not in the slot's final eligible worker-id set"
                );
            }
        }
    }
}

void Orchestrator::validate_remote_sidecars(
    const std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars,
    const std::vector<std::vector<int32_t>> &eligible_worker_ids
) const {
    if (remote_sidecars.empty()) return;
    if (remote_sidecars.size() != args_list.size()) {
        throw std::invalid_argument(
            "Orchestrator: remote sidecar length " + std::to_string(remote_sidecars.size()) +
            " does not match args length " + std::to_string(args_list.size())
        );
    }
    if (eligible_worker_ids.empty()) {
        throw std::invalid_argument("Orchestrator: remote sidecars require an explicit eligible worker-id set");
    }
    for (size_t g = 0; g < args_list.size(); ++g) {
        const TaskArgs &args = args_list[g];
        const RemoteTaskArgsSidecar &sidecar = remote_sidecars[g];
        if (sidecar.empty() && args.tensor_count() == 0) continue;
        if (sidecar.tensors.size() != static_cast<size_t>(args.tensor_count())) {
            throw std::invalid_argument("Orchestrator: remote sidecar tensor count does not match TaskArgs");
        }
        for (int32_t worker_id : eligible_worker_ids[g]) {
            if (manager_ == nullptr) continue;
            WorkerThread *wt = manager_->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
            if (wt == nullptr) {
                throw std::invalid_argument(
                    "Orchestrator: remote sidecar names an unknown worker " + std::to_string(worker_id)
                );
            }
            if (!wt->caps().remote) {
                throw std::invalid_argument(
                    "Orchestrator: remote sidecar cannot be submitted to local worker " + std::to_string(worker_id)
                );
            }
        }
        for (int32_t i = 0; i < args.tensor_count(); ++i) {
            const Tensor &ref = args.tensor(i);
            const RemoteTensorSidecar &tensor_sidecar = sidecar.tensors[static_cast<size_t>(i)];
            // A remote arg carries no local backing: its Tensor is a placeholder (nbytes 0 or a
            // REMOTE_SIDECAR backend) and the real descriptor lives in the sidecar.
            bool has_local_backing =
                ref.buffer.nbytes != 0 && ref.buffer.backend_kind != static_cast<uint8_t>(BackendKind::REMOTE_SIDECAR);
            if (tensor_sidecar.present && has_local_backing) {
                throw std::invalid_argument("Orchestrator: remote tensor metadata must not carry a local backing");
            }
            if (!tensor_sidecar.present && has_local_backing) {
                throw std::invalid_argument("Orchestrator: remote tensor uses a local backing without sidecar");
            }
            if (args.tag(i) == TensorArgType::OUTPUT && !has_local_backing && !tensor_sidecar.present) {
                throw std::invalid_argument("Orchestrator: remote OUTPUT tensor requires a RemoteTensorRef sidecar");
            }
            if (ref.buffer.address_space == static_cast<uint8_t>(AddressSpace::DEVICE) && !tensor_sidecar.present) {
                throw std::invalid_argument("Orchestrator: remote device-memory tensor requires a sidecar");
            }
            if (tensor_sidecar.present && tensor_sidecar.desc.address_space != RemoteAddressSpace::HOST_INLINE) {
                if (tensor_sidecar.desc.owner_worker_id < 0) {
                    throw std::invalid_argument("Orchestrator: remote tensor sidecar has invalid owner worker");
                }
                bool has_allowed_worker = false;
                for (int32_t worker_id : eligible_worker_ids[g]) {
                    has_allowed_worker = true;
                    if (tensor_sidecar.desc.address_space == RemoteAddressSpace::REMOTE_DEVICE &&
                        worker_id != tensor_sidecar.desc.owner_worker_id) {
                        throw std::invalid_argument(
                            "Orchestrator: remote tensor sidecar requires IMPORT_BUFFER before submitting to worker " +
                            std::to_string(worker_id)
                        );
                    }
                }
                if (!has_allowed_worker) {
                    throw std::invalid_argument("Orchestrator: remote tensor has no final eligible worker");
                }
            }
        }
    }
}

void Orchestrator::validate_explicit_deps(RunId run_id, const std::vector<TaskArgs> &args_list) const {
    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.explicit_dep_count(); ++i) {
            const TaskHandle &dep = args.explicit_dep(i);
            if (dep.run_id != run_id) {
                throw std::invalid_argument("Orchestrator: explicit dependency belongs to another run");
            }
            TaskSlotState *producer = allocator_->slot_state(dep.task_slot);
            if (producer == nullptr || producer->run_id != run_id) {
                throw std::invalid_argument("Orchestrator: explicit dependency names an unknown task");
            }
        }
    }
}

// =============================================================================
// reserve_slot — claim this submit's task slot
// =============================================================================
//
// A slot-only allocation (0 heap bytes). Args are Tensors backed by buffers the
// caller already owns (create_buffer), so the orchestrator no
// longer auto-allocates OUTPUT memory — an OUTPUT is a handle-backed ref like any
// other, tracked by canonical identity in infer_deps.

AllocResult Orchestrator::reserve_outputs_and_slot(
    std::vector<TaskArgs> &args_list, const std::vector<RemoteTaskArgsSidecar> &remote_sidecars
) {
    (void)args_list;
    (void)remote_sidecars;
    return allocator_->alloc(0, scope_->current_depth());
}

// =============================================================================
// infer_deps — tag-driven dependency inference
// =============================================================================

void Orchestrator::infer_deps(
    TaskSlot slot, const std::vector<TaskArgs> &args_list, const std::vector<int32_t> &target_worker_ids,
    const std::vector<RemoteTaskArgsSidecar> &remote_sidecars, std::vector<ProducerDependency> &producers,
    std::vector<TensorKey> &output_keys
) {
    RunId run_id = slot_state(slot).run_id;
    std::unordered_map<TaskSlot, size_t> producer_indices;
    size_t tensor_count_hint = 0;
    size_t explicit_dep_count_hint = 0;
    for (const TaskArgs &args : args_list) {
        tensor_count_hint += static_cast<size_t>(args.tensor_count());
        explicit_dep_count_hint += static_cast<size_t>(args.explicit_dep_count());
    }
    producer_indices.reserve(tensor_count_hint + explicit_dep_count_hint);

    auto add_unique_producer = [&](TaskSlot p, bool retain) {
        // Group submits walk many TaskArgs under one slot: if two entries in
        // the same group tag the same buffer (e.g. both OUTPUT 0xCAFE), the
        // second-pass lookup would return the slot that the first pass just
        // inserted — a self-loop. Skip it.
        if (p == slot) return;
        auto [it, inserted] = producer_indices.emplace(p, producers.size());
        if (inserted) {
            producers.push_back(ProducerDependency{p, retain});
        } else if (retain) {
            producers[it->second].retain = true;
        }
    };

    for (const TaskArgs &args : args_list) {
        for (int32_t i = 0; i < args.explicit_dep_count(); ++i) {
            add_unique_producer(args.explicit_dep(i).task_slot, args.explicit_dep_retain(i));
        }
    }

    // One key can resolve to several producers, one per live view under it that this arg's own
    // view reaches. `overlapping` is scratch reused across args.
    std::vector<TaskSlot> overlapping;
    auto take_producers = [&](TensorKey key, const TensorFootprint &view) {
        overlapping.clear();
        tensormap_->lookup_overlapping(run_id, key, view, overlapping);
        for (TaskSlot prod : overlapping) {
            add_unique_producer(prod, true);
        }
    };

    // Tag-driven dependency inference — mirrors L2
    // (src/a2a3/runtime/tensormap_and_ringbuffer/runtime/orchestrator.cpp
    //  steps B and 4):
    //   INPUT            → lookup only (RaW)
    //   INOUT            → lookup + insert (RaW + WaW)
    //   OUTPUT_EXISTING  → insert only (user-provided buffer; any WaW dep on
    //                      the creator must be expressed via INOUT instead)
    //   OUTPUT           → insert only (pure overwrite; if auto-alloc is
    //                      needed, the data ptr is assigned in
    //                      reserve_outputs_and_slot before this step)
    //   NO_DEP           → skip
    for (size_t g = 0; g < args_list.size(); ++g) {
        int32_t worker_id = (g < target_worker_ids.size()) ? target_worker_ids[g] : -1;
        const TaskArgs &a = args_list[g];
        for (int32_t i = 0; i < a.tensor_count(); ++i) {
            const Tensor &r = a.tensor(i);
            TensorKey key{};
            // The bytes this arg touches inside the backing its key names. Two args under one
            // key conflict only where their views intersect, which is what keeps two disjoint
            // slices of one buffer — `x[0]` and `x[1]` of a rank-major tensor, or two column
            // blocks whose bounding boxes interleave — independent.
            TensorFootprint view{};
            bool has_key = false;
            if (!remote_sidecars.empty()) {
                const auto &sidecar = remote_sidecars[g];
                if (static_cast<size_t>(i) < sidecar.tensors.size() &&
                    sidecar.tensors[static_cast<size_t>(i)].present) {
                    const RemoteTensorDesc &desc = sidecar.tensors[static_cast<size_t>(i)].desc;
                    TensorAddressKind kind = desc.address_space == RemoteAddressSpace::HOST_INLINE ?
                                                 TensorAddressKind::HOST_INLINE :
                                                 TensorAddressKind::REMOTE_BUFFER;
                    // A remote key already carries the view's offset, so every entry under one
                    // of them denotes the same origin and a footprint adds nothing; the default
                    // whole-backing one keeps this path's behaviour exactly as it was.
                    key = TensorKey::remote_buffer(
                        kind, desc.owner_worker_id, desc.buffer_id, desc.generation, desc.offset
                    );
                    has_key = true;
                }
            }
            if (!has_key) {
                if (r.buffer.nbytes == 0) continue;  // placeholder / null ref — nothing to track
                // Key a local Tensor by its canonical identity alone, so every view of one backing
                // lands on one key and no real dependency can hide behind a differing offset. The
                // view geometry decides which of those candidates actually conflict.
                CanonicalIdentityHash idh;
                uint64_t k = idh(r.buffer.identity);
                key = r.buffer.address_space == static_cast<uint8_t>(AddressSpace::DEVICE) ?
                          TensorKey::local_child(k, worker_id) :
                          TensorKey::local_host(k);
                view = tensor_footprint(r);
                has_key = true;
            }
            TensorArgType tag = a.tag(i);
            switch (tag) {
            case TensorArgType::INPUT: {
                take_producers(key, view);
                break;
            }
            case TensorArgType::INOUT: {
                take_producers(key, view);
                output_keys.push_back(key);
                if (test_hook_) test_hook_(OrchestratorTestPoint::SUBMIT_OUTPUT_KEY_PREPARED);
                tensormap_->insert(run_id, key, view, slot);
                break;
            }
            case TensorArgType::OUTPUT:
            case TensorArgType::OUTPUT_EXISTING: {
                output_keys.push_back(key);
                if (test_hook_) test_hook_(OrchestratorTestPoint::SUBMIT_OUTPUT_KEY_PREPARED);
                tensormap_->insert(run_id, key, view, slot);
                break;
            }
            case TensorArgType::NO_DEP:
            default:
                break;
            }
        }
    }
}

// =============================================================================
// Scope
// =============================================================================

void Orchestrator::scope_begin() {
    (void)current_building_run();
    scope_->scope_begin();
}

void Orchestrator::scope_end() {
    scope_->scope_end([this](TaskSlot slot) {
        release_ref(slot);
    });
}

// =============================================================================
// Reference release helpers
// =============================================================================

void Orchestrator::release_ref(TaskSlot slot) { try_consume(slot); }

void Orchestrator::try_consume(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);
    int32_t released = s.fanout_released.fetch_add(1, std::memory_order_acq_rel) + 1;
    int32_t total;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        total = s.fanout_total;
    }
    // Threshold matches Scheduler::try_consume. fanout_total counts
    // scope_ref + N consumer refs; the extra +1 is the terminal self
    // release. These refs can be released from completion, poison, or
    // scope_end paths in different orders.
    TaskState state = s.state.load(std::memory_order_acquire);
    if (released >= total + 1 && (state == TaskState::COMPLETED || state == TaskState::FAILED)) {
        on_consumed(slot);
    }
}

bool Orchestrator::on_consumed(TaskSlot slot) {
    TaskSlotState &s = slot_state(slot);

    // Idempotent: the threshold can be hit by either release_ref (scope_end,
    // Orch thread) or try_consume (consumer's deferred release, scheduler
    // thread). Whichever fires last wins; subsequent callers see CONSUMED
    // and bail.
    TaskState expected = TaskState::COMPLETED;
    if (!s.state.compare_exchange_strong(
            expected, TaskState::CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        expected = TaskState::FAILED;
        if (!s.state.compare_exchange_strong(
                expected, TaskState::CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return false;
        }
    }

    RunId run_id = s.run_id;
    tensormap_->erase_task_outputs(run_id, slot, s.output_keys);

    // HeapRing-owned OUTPUT slabs are reclaimed implicitly when the allocator
    // advances last_alive past this slot — no per-slot munmap needed.
    allocator_->release(slot);

    decrement_run_tasks(run_id);
    return true;
}
