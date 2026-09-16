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

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/host_span_names.h"
#include "common/host_span_scope.h"
#include "ring.h"
#include "types.h"
#include "worker_manager.h"

namespace {

bool is_failure(EndpointOutcome outcome) {
    return outcome == EndpointOutcome::TASK_FAILURE || outcome == EndpointOutcome::ENDPOINT_FAILURE;
}

bool is_terminal_group_state(GroupMemberState state) {
    return state == GroupMemberState::SUCCESS || state == GroupMemberState::FAILED ||
           state == GroupMemberState::SKIPPED;
}

struct PreparedGroupVectors {
    std::vector<GroupMemberState> member_states;
    std::vector<EndpointOutcome> member_outcomes;
    int32_t terminal_count{0};

    bool replaces_existing() const noexcept { return !member_states.empty(); }
};

PreparedGroupVectors
prepare_group_vectors_locked(const TaskSlotState &state, int32_t group_size, GroupMemberState initial_member_state) {
    const size_t size = static_cast<size_t>(group_size);
    if (state.group_member_states.size() == size && state.group_member_outcomes.size() == size) return {};

    // Prepare both allocations before changing either shared vector. Dispatch
    // does this while the slot is still READY, so even a second-allocation
    // failure leaves cancellation free to claim and reclaim the slot. A
    // completion-side invariant repair also preserves live/terminal members;
    // resetting them could consume a failed group while a peer still runs.
    PreparedGroupVectors prepared;
    prepared.member_states.assign(size, initial_member_state);
    prepared.member_outcomes.assign(size, EndpointOutcome::SKIPPED);
    std::copy_n(
        state.group_member_states.begin(), std::min(size, state.group_member_states.size()),
        prepared.member_states.begin()
    );
    std::copy_n(
        state.group_member_outcomes.begin(), std::min(size, state.group_member_outcomes.size()),
        prepared.member_outcomes.begin()
    );
    prepared.terminal_count = static_cast<int32_t>(
        std::count_if(prepared.member_states.begin(), prepared.member_states.end(), is_terminal_group_state)
    );
    return prepared;
}

PreparedGroupVectors
prepare_group_vectors(TaskSlotState &state, int32_t group_size, GroupMemberState initial_member_state) {
    std::lock_guard<std::mutex> lk(state.group_mu);
    return prepare_group_vectors_locked(state, group_size, initial_member_state);
}

void commit_group_vectors_locked(TaskSlotState &state, PreparedGroupVectors &prepared) noexcept {
    if (!prepared.replaces_existing()) return;
    state.group_member_states.swap(prepared.member_states);
    state.group_member_outcomes.swap(prepared.member_outcomes);
    state.group_terminal_count.store(prepared.terminal_count, std::memory_order_relaxed);
}

void reset_group_state_locked(TaskSlotState &state, GroupMemberState initial_member_state) noexcept {
    // Exact-sized storage is prepared while BUILDING and defensively repaired
    // before the dispatch claim. Everything after READY -> RUNNING is now a
    // no-allocation commit.
    std::fill(state.group_member_states.begin(), state.group_member_states.end(), initial_member_state);
    std::fill(state.group_member_outcomes.begin(), state.group_member_outcomes.end(), EndpointOutcome::SKIPPED);
    state.group_terminal_count.store(0, std::memory_order_relaxed);
    state.group_failed = false;
    state.group_first_failure_index = -1;
    state.group_first_failure_message.clear();
}

}  // namespace

// =============================================================================
// Scheduler
// =============================================================================

void Scheduler::start(const Config &cfg) {
    if (cfg.ring == nullptr || cfg.ready_sub_queue == nullptr || cfg.ready_next_level_queues == nullptr ||
        cfg.manager == nullptr || !cfg.enqueue_ready_cb)
        throw std::invalid_argument("Scheduler::start: null config fields");
    if (cfg.reservation_stall_warn_after < std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("Scheduler::start: negative reservation stall warning interval");
    }
    cfg_ = cfg;

    {
        // run()'s observed generation restarts at zero, so any advance here
        // arms the first round.
        std::lock_guard<std::mutex> lk(completion_mu_);
        ++wake_generation_;
    }
    dispatch_round_count_.store(0, std::memory_order_relaxed);
    reservation_stall_episode_.reset();
    stop_requested_.store(false, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    sched_thread_ = std::thread(&Scheduler::run, this);
}

void Scheduler::request_stop() {
    stop_requested_.store(true, std::memory_order_release);
    if (running_.load(std::memory_order_acquire)) cfg_.manager->stop_workers();
    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        ++wake_generation_;
    }
    completion_cv_.notify_all();
}

void Scheduler::stop() {
    request_stop();

    if (sched_thread_.joinable()) sched_thread_.join();

    running_.store(false, std::memory_order_release);
}

// =============================================================================
// Endpoint completion callback
// =============================================================================

void Scheduler::worker_done(WorkerCompletion completion) {
    TaskSlotState &s = *cfg_.ring->slot_state(completion.task_slot);
    bool failure_reported = is_failure(completion.outcome);
    if (failure_reported && cfg_.on_task_failed_cb) {
        cfg_.on_task_failed_cb(completion.task_slot, completion.error_message);
    }

    // Group aggregation: only push to completion queue when ALL workers done
    if (s.is_group()) {
        WorkerCompletion terminal = completion;
        {
            std::unique_lock<std::mutex> lk(s.group_mu);
            const int32_t group_size = s.group_size();
            PreparedGroupVectors prepared =
                prepare_group_vectors_locked(s, group_size, GroupMemberState::NOT_DISPATCHED);
            commit_group_vectors_locked(s, prepared);
            bool invalid_group_index = completion.group_index < 0 || completion.group_index >= group_size;
            if (invalid_group_index) {
                terminal.outcome = EndpointOutcome::ENDPOINT_FAILURE;
                terminal.error_message = "Scheduler::worker_done: group_index " +
                                         std::to_string(completion.group_index) + " out of range for group_size " +
                                         std::to_string(group_size);
            }

            int32_t index = invalid_group_index ? -1 : terminal.group_index;
            if (index >= 0 && index < group_size) {
                GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(index)];
                if (is_terminal_group_state(member_state)) {
                    lk.unlock();
                    notify_ready();
                    return;
                }

                if (terminal.outcome == EndpointOutcome::SUCCESS) {
                    member_state = GroupMemberState::SUCCESS;
                } else {
                    member_state = GroupMemberState::FAILED;
                    if (!s.group_failed) {
                        s.group_first_failure_index = index;
                        s.group_first_failure_message = terminal.error_message;
                    }
                    s.group_failed = true;
                }
                s.group_member_outcomes[static_cast<size_t>(index)] = terminal.outcome;
                s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
            } else {
                if (!s.group_failed) {
                    s.group_first_failure_index = -1;
                    s.group_first_failure_message = terminal.error_message;
                }
                s.group_failed = true;
            }

            if (s.group_failed) {
                for (int32_t i = 0; i < group_size; ++i) {
                    GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
                    if (is_terminal_group_state(member_state)) continue;
                    if (!invalid_group_index && member_state != GroupMemberState::NOT_DISPATCHED) continue;
                    member_state = GroupMemberState::SKIPPED;
                    s.group_member_outcomes[static_cast<size_t>(i)] = EndpointOutcome::SKIPPED;
                    s.group_terminal_count.fetch_add(1, std::memory_order_acq_rel);
                }
            }

            if (s.group_terminal_count.load(std::memory_order_acquire) < group_size) {
                lk.unlock();
                notify_ready();
                return;
            }

            if (s.group_failed) {
                int32_t failure_index = s.group_first_failure_index;
                terminal.group_index = failure_index;
                terminal.outcome = EndpointOutcome::TASK_FAILURE;
                if (failure_index >= 0 && failure_index < group_size) {
                    EndpointOutcome member_outcome = s.group_member_outcomes[static_cast<size_t>(failure_index)];
                    if (is_failure(member_outcome)) terminal.outcome = member_outcome;
                }
                terminal.error_message = s.group_first_failure_message;
            } else {
                terminal.outcome = EndpointOutcome::SUCCESS;
                terminal.error_message.clear();
            }
        }
        completion = std::move(terminal);
    }

    if (!failure_reported && is_failure(completion.outcome) && cfg_.on_task_failed_cb) {
        cfg_.on_task_failed_cb(completion.task_slot, completion.error_message);
    }

    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        completion_queue_.push(std::move(completion));
    }
    completion_cv_.notify_one();
}

void Scheduler::notify_ready() {
    {
        std::lock_guard<std::mutex> lk(completion_mu_);
        ++wake_generation_;
    }
    completion_cv_.notify_one();
}

// =============================================================================
// Scheduler loop
// =============================================================================

void Scheduler::run() {
    uint64_t observed_wake_generation = 0;
#if SIMPLER_HOST_STRACE
    // Iterations that neither drained nor dispatched, since the last iteration
    // that did. The loop skips its wait whenever any worker is busy, so its
    // iteration count is bounded by CPU speed rather than by work — a span per
    // iteration would be unbounded. Carrying the count on the next span that
    // does describe work keeps the spin ratio observable at zero records.
    uint64_t spins = 0;
#endif
    while (true) {
        {
            std::unique_lock<std::mutex> lk(completion_mu_);
            auto ready = [this, &observed_wake_generation] {
                return !completion_queue_.empty() || wake_generation_ != observed_wake_generation;
            };
            if (!cfg_.manager->any_busy()) {
                const auto stall_deadline = reservation_stall_deadline();
                if (stall_deadline.has_value()) {
                    completion_cv_.wait_until(lk, *stall_deadline, ready);
                } else {
                    completion_cv_.wait(lk, ready);
                }
            }
            observed_wake_generation = wake_generation_;
        }

#if SIMPLER_HOST_STRACE
        // Outside the loop_mu_ acquisition below, so a span covers the wait for
        // that lock: the gap between two spans is then the condition-variable
        // wait alone, which is the "no work to do" reading the span cannot give.
        const bool trace_loop = simpler::host_trace::enabled();
        const int64_t loop_start_ns = trace_loop ? simpler::host_trace::now_ns() : 0;
        const uint64_t dispatched_before = dispatched_total_.load(std::memory_order_relaxed);
#endif

        // Hold loop_mu_ across the entire slot-touching body so quiescent
        // compaction cannot free TaskSlotStates while on_task_complete or
        // dispatch_ready is still reading them.
        std::lock_guard<std::mutex> loop_lk(loop_mu_);

        cfg_.manager->progress();

        // Phase 1: drain completions
        [[maybe_unused]] uint64_t drained = 0;
        while (true) {
            WorkerCompletion completion;
            {
                std::lock_guard<std::mutex> lk(completion_mu_);
                if (completion_queue_.empty()) break;
                completion = std::move(completion_queue_.front());
                completion_queue_.pop();
            }
            on_task_complete(completion);
            drained++;
        }

#if SIMPLER_HOST_STRACE
        const int64_t drain_end_ns = trace_loop ? simpler::host_trace::now_ns() : 0;
#endif

        // Phase 2: dispatch ready tasks. Once teardown publishes stop, the
        // existing endpoint-owned work drains but no new slot enters a worker.
        if (!stop_requested_.load(std::memory_order_acquire)) dispatch_ready();

#if SIMPLER_HOST_STRACE
        const uint64_t dispatched = dispatched_total_.load(std::memory_order_relaxed) - dispatched_before;
        if (drained == 0 && dispatched == 0) {
            spins++;
        } else {
            if (trace_loop) {
                const int64_t loop_end_ns = simpler::host_trace::now_ns();
                const std::string loop_attrs =
                    "role=scheduler drained=" + std::to_string(drained) + " dispatched=" + std::to_string(dispatched) +
                    " drain_ns=" + std::to_string(drain_end_ns - loop_start_ns) + " spins=" + std::to_string(spins);
                simpler::host_trace::emit(
                    simpler::host_trace::host_span_name(simpler::host_trace::HostSpan::SchedulerLoop), 0, 0, 0,
                    loop_start_ns, loop_end_ns - loop_start_ns, loop_attrs.c_str()
                );
            }
            spins = 0;
        }
#endif

        // Exit when stop requested and all workers idle
        if (stop_requested_.load(std::memory_order_acquire)) {
            if (!cfg_.manager->any_busy()) {
                // Final drain
                while (true) {
                    WorkerCompletion completion;
                    {
                        std::lock_guard<std::mutex> lk(completion_mu_);
                        if (completion_queue_.empty()) break;
                        completion = std::move(completion_queue_.front());
                        completion_queue_.pop();
                    }
                    on_task_complete(completion);
                }
                break;  // loop_lk released on scope exit before exiting run()
            }
        }
    }
}

// =============================================================================
// on_task_complete / try_consume
// =============================================================================

void Scheduler::on_task_complete(const WorkerCompletion &completion) {
    TaskSlot slot = completion.task_slot;
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    bool failed = is_failure(completion.outcome);

    std::string failure_message;
    if (failed) failure_message = completion.error_message;

    // The transition and the consumer snapshot are one decision, taken under
    // fanout_mu. A task wiring itself onto this slot holds the same lock while
    // it appends itself and reads the state, so it either lands in `consumers`
    // — and is released or poisoned below — or sees the terminal state and
    // handles it in submit. Splitting the two lets a consumer register as a
    // live fanin after the snapshot and never be released, or observe
    // COMPLETED, decline to count the fanin, and still be released here, which
    // makes it READY one producer early.
    std::vector<TaskSlot> consumers;
    std::vector<TaskSlot> producers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
        producers = s.fanin_producers;
        if (failed) s.failure_message.swap(failure_message);
        s.state.store(failed ? TaskState::FAILED : TaskState::COMPLETED, std::memory_order_release);
    }
    for (TaskSlot consumer : consumers) {
        if (failed) {
            poison_task(consumer, completion.error_message);
            continue;
        }
        TaskSlotState &cs = *cfg_.ring->slot_state(consumer);
        cs.fanin_released.fetch_add(1, std::memory_order_acq_rel);
        // A consumer still BUILDING is not readiable, and this release is not
        // lost: submit's publication compares the pair under the same lock.
        if (try_mark_ready(cs)) {
            cfg_.enqueue_ready_cb(consumer);
        }
    }

    try_consume(slot);

    // Deferred release: release one fanout ref on each producer this task consumed.
    for (TaskSlot prod : producers) {
        try_consume(prod);
    }
}

void Scheduler::poison_task(TaskSlot slot, const std::string &root_message) {
    TaskSlotState &s = *cfg_.ring->slot_state(slot);
    std::optional<TaskState> claimed = claim_task_failure(s, root_message);
    // Not claimable: RUNNING or COMPLETED is owned by the device until its own
    // completion arrives, and FAILED/CONSUMED/FREE is already someone else's.
    // Cancellation is the only contender allowed to take over an existing
    // debt, because it runs after graph submission has closed and therefore
    // knows a BUILDING slot no longer has a submitting owner.
    if (!claimed.has_value()) return;
    // Claimed while submit was still wiring it: submit finishes the
    // propagation, using the counters only it can know are final.
    if (*claimed == TaskState::BUILDING) return;

    mark_group_members_skipped(s, root_message);

    std::vector<TaskSlot> consumers;
    std::vector<TaskSlot> producers;
    {
        std::lock_guard<std::mutex> lk(s.fanout_mu);
        consumers = s.fanout_consumers;
        producers = s.fanin_producers;
    }
    for (TaskSlot consumer : consumers) {
        poison_task(consumer, root_message);
    }

    // All fallible preparation, including recursive poison, finishes before
    // any non-repeatable release. This claimant owns the propagation
    // exclusively; cancellation does not take over scheduler-owned debt.
    try_consume(slot);
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
        TaskState state = s.state.load(std::memory_order_acquire);
        if (state == TaskState::COMPLETED || state == TaskState::FAILED) {
            if (cfg_.on_consumed_cb) cfg_.on_consumed_cb(slot);
        }
    }
}

// =============================================================================
// Dispatch — delegates to WorkerManager
// =============================================================================

// The NEXT_LEVEL worker-id set is fixed by Worker::init() (NextLevelReadyQueues
// is reset once, before scheduling starts) and every queued/targeted worker id
// is validated in Orchestrator::submit_impl. The endpoint lane resolves expected
// lane/capacity/stopping rejections through exactly one complete_unpublished
// call, under the same non-throwing completion-callback contract as ordinary
// endpoint completion.
void Scheduler::dispatch_ready() {
    dispatch_round_count_.fetch_add(1, std::memory_order_relaxed);
    std::optional<RunId> run_snapshot;
    if (cfg_.active_run_cb) {
        RunId active_run = cfg_.active_run_cb();
        if (active_run == INVALID_RUN_ID) return;
        run_snapshot = active_run;
        cfg_.manager->activate_prepared_run(active_run);
    }

    dispatch_preparable_next_level_singles();

    // Group reservations and every queue pop in one pass belong to the same
    // whole-run FIFO head, even if a completion advances the head mid-pass.
    bool group_arrived_between_phases = false;
    do {
        const NextLevelGroupDispatchResult group_result = dispatch_next_level_group(run_snapshot);
        update_reservation_stall(group_result);
        if (cfg_.after_group_phase_cb) cfg_.after_group_phase_cb();
        group_arrived_between_phases = dispatch_next_level_singles(
            group_result.reserved_worker_ids, run_snapshot, group_result.blocked_group_slot == INVALID_SLOT
        );
    } while (group_arrived_between_phases);
    dispatch_sub_ready(run_snapshot);
}

bool claim_for_dispatch(TaskSlotState &s) {
    TaskState expected = TaskState::READY;
    return s.state.compare_exchange_strong(
        expected, TaskState::RUNNING, std::memory_order_acq_rel, std::memory_order_acquire
    );
}

void Scheduler::dispatch_claimed(WorkerThread *worker, WorkerDispatch dispatch, bool prepared) {
    dispatched_total_.fetch_add(1, std::memory_order_relaxed);
    // The endpoint lane owns publication failure through one terminal callback.
    // Retrying here after that callback starts can duplicate a partially
    // published completion. The callback contract is the same non-throwing
    // contract used by ordinary endpoint completions.
    if (prepared) {
        worker->dispatch_prepared(dispatch);
    } else {
        worker->dispatch(dispatch);
    }
}

void Scheduler::dispatch_preparable_next_level_singles() {
    if (!cfg_.preparable_run_cb) return;
    RunId run_id = cfg_.preparable_run_cb();
    if (run_id == INVALID_RUN_ID || cfg_.manager->has_staged_run(run_id) ||
        !cfg_.ready_next_level_queues->groups_empty(run_id)) {
        return;
    }

    for (int32_t worker_id : cfg_.ready_next_level_queues->worker_ids()) {
        WorkerThread *worker = cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
        if (worker == nullptr || !worker->can_stage()) continue;
        TaskSlot slot;
        if (!cfg_.ready_next_level_queues->try_pop_single(worker_id, run_id, slot)) continue;
        if (!cfg_.ready_next_level_queues->groups_empty(run_id)) {
            cfg_.enqueue_ready_cb(slot);
            return;
        }
        TaskSlotState &state = *cfg_.ring->slot_state(slot);
        if (state.state.load(std::memory_order_acquire) != TaskState::READY) continue;
        if (state.run_id != run_id || state.worker_type != WorkerType::NEXT_LEVEL || state.is_group() ||
            state.target_worker_id(0) != worker_id) {
            cfg_.enqueue_ready_cb(slot);
            continue;
        }
        if (cfg_.before_claim_cb) cfg_.before_claim_cb(slot);
        if (!claim_for_dispatch(state)) continue;
        dispatch_claimed(worker, WorkerDispatch{slot, 0}, /*prepared=*/true);
        return;
    }
}

void Scheduler::dispatch_sub_ready(const std::optional<RunId> &run_snapshot) {
    TaskSlot slot;
    while (run_snapshot ? cfg_.ready_sub_queue->try_pop(*run_snapshot, slot) : cfg_.ready_sub_queue->try_pop(slot)) {
        TaskSlotState &s = *cfg_.ring->slot_state(slot);
        if (s.state.load(std::memory_order_acquire) != TaskState::READY) continue;
        if (run_snapshot && s.run_id != *run_snapshot) {
            cfg_.enqueue_ready_cb(slot);
            return;
        }
        if (s.worker_type != WorkerType::SUB) {
            throw std::runtime_error("Scheduler::dispatch_sub_ready: misrouted task slot");
        }

        const int32_t group_size = s.group_size();
        std::vector<WorkerThread *> workers;
        workers.reserve(static_cast<size_t>(group_size));
        for (int32_t i = 0; i < group_size; ++i) {
            WorkerThread *worker = cfg_.manager->pick_idle_sub_excluding(workers);
            if (worker == nullptr) {
                // Through the same entry point as every other requeue: pushing
                // straight onto the queue skips the check that this run is
                // still live, and would rebuild a partition retire has erased.
                cfg_.enqueue_ready_cb(slot);
                return;
            }
            workers.push_back(worker);
        }

        if (s.is_group()) {
            // The expected path is allocation-free because submit prepared
            // exact-sized storage while BUILDING. Repair any violated invariant
            // into local storage while the slot is still cancellation-claimable.
            // Do not publish that storage unless this scheduler wins the claim:
            // cancellation may have marked the shared vectors terminal meanwhile.
            PreparedGroupVectors prepared = prepare_group_vectors(s, group_size, GroupMemberState::NOT_DISPATCHED);
            if (cfg_.before_claim_cb) cfg_.before_claim_cb(slot);
            {
                // Acquire the only fallible synchronization primitive before
                // publishing RUNNING; the commit after the claim only fills
                // existing storage and resets scalar metadata.
                std::lock_guard<std::mutex> lk(s.group_mu);
                if (!claim_for_dispatch(s)) continue;
                commit_group_vectors_locked(s, prepared);
                reset_group_state_locked(s, GroupMemberState::NOT_DISPATCHED);
            }
        } else {
            if (cfg_.before_claim_cb) cfg_.before_claim_cb(slot);
            if (!claim_for_dispatch(s)) continue;
        }
        for (int32_t i = 0; i < group_size; ++i) {
            if (s.is_group()) {
                std::lock_guard<std::mutex> lk(s.group_mu);
                GroupMemberState &member_state = s.group_member_states[static_cast<size_t>(i)];
                if (member_state != GroupMemberState::NOT_DISPATCHED || s.group_failed) continue;
                member_state = GroupMemberState::RUNNING;
            }
            dispatch_claimed(workers[static_cast<size_t>(i)], WorkerDispatch{slot, i}, /*prepared=*/false);
        }
    }
}

Scheduler::NextLevelGroupDispatchResult Scheduler::dispatch_next_level_group(const std::optional<RunId> &run_snapshot) {
    TaskSlot slot;
    while (run_snapshot ? cfg_.ready_next_level_queues->try_front_group(*run_snapshot, slot) :
                          cfg_.ready_next_level_queues->try_front_group(slot)) {
        TaskSlotState &s = *cfg_.ring->slot_state(slot);
        if (s.state.load(std::memory_order_acquire) != TaskState::READY) {
            TaskSlot stale;
            if (run_snapshot) {
                cfg_.ready_next_level_queues->try_pop_group(*run_snapshot, stale);
            } else {
                cfg_.ready_next_level_queues->try_pop_group(stale);
            }
            continue;
        }
        if (run_snapshot && s.run_id != *run_snapshot) {
            TaskSlot misplaced;
            cfg_.ready_next_level_queues->try_pop_group(*run_snapshot, misplaced);
            cfg_.enqueue_ready_cb(slot);
            return {};
        }
        if (s.worker_type != WorkerType::NEXT_LEVEL || !s.is_group()) {
            throw std::runtime_error("Scheduler::dispatch_next_level_group: misrouted task slot");
        }

        const int32_t group_size = s.group_size();
        NextLevelGroupDispatchResult result;
        result.blocked_group_slot = slot;
        std::vector<WorkerThread *> workers;
        workers.reserve(static_cast<size_t>(group_size));
        result.reserved_worker_ids.reserve(static_cast<size_t>(group_size));
        bool all_workers_idle = true;
        for (int32_t i = 0; i < group_size; ++i) {
            const int32_t worker_id = s.target_worker_id(i);
            WorkerThread *worker = cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
            if (worker == nullptr) {
                throw std::runtime_error("Scheduler::dispatch_next_level_group: invalid target worker");
            }
            if (!result.reserved_worker_ids.insert(worker_id).second) {
                throw std::runtime_error("Scheduler::dispatch_next_level_group: duplicate target worker");
            }
            const bool worker_idle = worker->idle();
            if (!worker_idle) {
                all_workers_idle = false;
                result.busy_target_worker_ids.push_back(worker_id);
            }
            workers.push_back(worker);
        }
        if (!all_workers_idle) {
            for (size_t i = 0; i < workers.size(); ++i) {
                TaskSlot single_head = INVALID_SLOT;
                const int32_t worker_id = s.target_worker_id(static_cast<int32_t>(i));
                if (std::find(result.busy_target_worker_ids.begin(), result.busy_target_worker_ids.end(), worker_id) !=
                    result.busy_target_worker_ids.end())
                    continue;
                const bool has_queued_single =
                    run_snapshot ?
                        cfg_.ready_next_level_queues->try_front_single(worker_id, *run_snapshot, single_head) :
                        cfg_.ready_next_level_queues->try_front_single(worker_id, single_head);
                if (!has_queued_single) continue;
                result.idle_queued_target_worker_ids.push_back(worker_id);
                result.idle_queued_single_head_slots.push_back(single_head);
            }
            return result;
        }

        // The head was observed before the worker checks above, so a run
        // cancelling in that window can consume the slot and erase its whole
        // partition. That is legal, and throwing here would end the process:
        // this runs on the scheduler thread, which has no handler.
        TaskSlot popped;
        bool popped_ok = run_snapshot ? cfg_.ready_next_level_queues->try_pop_group(*run_snapshot, popped) :
                                        cfg_.ready_next_level_queues->try_pop_group(popped);
        if (!popped_ok) return {};
        if (popped != slot) {
            // A different group reached the head meanwhile. Put back what we
            // took and re-decide from the top rather than dispatching a group
            // whose workers we never checked.
            cfg_.enqueue_ready_cb(popped);
            return {};
        }

        PreparedGroupVectors prepared = prepare_group_vectors(s, group_size, GroupMemberState::RUNNING);
        if (cfg_.before_claim_cb) cfg_.before_claim_cb(slot);
        {
            std::lock_guard<std::mutex> lk(s.group_mu);
            if (!claim_for_dispatch(s)) continue;
            commit_group_vectors_locked(s, prepared);
            reset_group_state_locked(s, GroupMemberState::RUNNING);
        }
        for (int32_t i = 0; i < group_size; ++i) {
            dispatch_claimed(workers[static_cast<size_t>(i)], WorkerDispatch{slot, i}, /*prepared=*/false);
        }
    }
    return {};
}

void Scheduler::update_reservation_stall(const NextLevelGroupDispatchResult &dispatch_result) {
    if (dispatch_result.blocked_group_slot == INVALID_SLOT || dispatch_result.idle_queued_target_worker_ids.empty()) {
        reservation_stall_episode_.reset();
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!reservation_stall_episode_.has_value() ||
        reservation_stall_episode_->group_slot != dispatch_result.blocked_group_slot) {
        reservation_stall_episode_ = ReservationStallEpisode{dispatch_result.blocked_group_slot, now, false};
    }

    ReservationStallEpisode &episode = *reservation_stall_episode_;
    if (episode.reported || now < episode.started_at + cfg_.reservation_stall_warn_after) return;

    if (cfg_.reservation_stall_sink != nullptr) {
        const ReservationStallDiagnostic diagnostic{
            dispatch_result.blocked_group_slot,
            dispatch_result.busy_target_worker_ids.data(),
            dispatch_result.busy_target_worker_ids.size(),
            dispatch_result.idle_queued_target_worker_ids.data(),
            dispatch_result.idle_queued_single_head_slots.data(),
            dispatch_result.idle_queued_target_worker_ids.size(),
        };
        cfg_.reservation_stall_sink(cfg_.reservation_stall_sink_context, diagnostic);
    }
    episode.reported = true;
}

std::optional<std::chrono::steady_clock::time_point> Scheduler::reservation_stall_deadline() const {
    if (!reservation_stall_episode_.has_value() || reservation_stall_episode_->reported) return std::nullopt;
    return reservation_stall_episode_->started_at + cfg_.reservation_stall_warn_after;
}

bool Scheduler::dispatch_next_level_singles(
    const std::unordered_set<int32_t> &reserved_worker_ids, const std::optional<RunId> &run_snapshot,
    bool verify_group_barrier
) {
    for (int32_t worker_id : cfg_.ready_next_level_queues->worker_ids()) {
        if (reserved_worker_ids.find(worker_id) != reserved_worker_ids.end()) continue;

        WorkerThread *worker = cfg_.manager->get_worker_by_id(WorkerType::NEXT_LEVEL, worker_id);
        if (worker == nullptr) {
            throw std::runtime_error(
                "Scheduler::dispatch_next_level_singles: unknown worker id " + std::to_string(worker_id)
            );
        }
        if (!worker->idle()) continue;

        TaskSlot slot;
        while (run_snapshot ? cfg_.ready_next_level_queues->try_pop_single(worker_id, *run_snapshot, slot) :
                              cfg_.ready_next_level_queues->try_pop_single(worker_id, slot)) {
            TaskSlotState &s = *cfg_.ring->slot_state(slot);
            if (s.state.load(std::memory_order_acquire) != TaskState::READY) continue;
            if (run_snapshot && s.run_id != *run_snapshot) {
                cfg_.enqueue_ready_cb(slot);
                break;
            }
            const bool group_waiting =
                verify_group_barrier && (run_snapshot ? !cfg_.ready_next_level_queues->groups_empty(*run_snapshot) :
                                                        !cfg_.ready_next_level_queues->groups_empty());
            if (group_waiting) {
                cfg_.enqueue_ready_cb(slot);
                return true;
            }
            if (s.worker_type != WorkerType::NEXT_LEVEL || s.is_group() || s.target_worker_id(0) != worker_id) {
                throw std::runtime_error("Scheduler::dispatch_next_level_singles: misrouted task slot");
            }
            if (cfg_.before_claim_cb) cfg_.before_claim_cb(slot);
            if (!claim_for_dispatch(s)) continue;
            dispatch_claimed(worker, WorkerDispatch{slot, 0}, /*prepared=*/false);
            break;
        }
    }
    return false;
}
