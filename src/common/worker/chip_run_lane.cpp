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

#include "chip_run_lane.h"

#include "chip_worker.h"
#include "credit_call_gate.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "host/chip_swimlane_runs.h"

struct ChipRunState {
    enum class Phase : uint8_t { QUEUED, PREPARED, LAUNCHED, TERMINAL };

    int32_t callable_id{0};
    ChipStorageTaskArgs args{};
    CallConfig config{};
    PipelineSlotLease lease{};
    uint64_t run_id{0};
    uint64_t dispatch_id{0};
    // Where the submission came from, for diagnostic attribution only. Stored
    // because a queued run prepares at the front, long after its submit.
    NativeRunProvenance provenance{};
    volatile int32_t *accepted_state{nullptr};
    int32_t accepted_value{0};
    ChipWorkerNativeRun native_run{};
    void *caller_stream{nullptr};
    Phase phase{Phase::QUEUED};
    ChipRunPreparationDisposition disposition{ChipRunPreparationDisposition::VALIDATED_ONLY};
    bool pipeline_leased{true};
    bool activated{false};
    bool crossed_launch_fence{false};
    bool depth_one_fallback{false};
    // Its prepare asked to wait: the device resources it needs are held by a
    // consumer this worker will finish, and its saved plan is what the next
    // attempt continues from. Unlike `depth_one_fallback` this is retried every
    // round, because what clears it is another run's retirement rather than
    // this run reaching the front. A run carrying it owns resources an
    // unprepared run does not, so every abandonment path gives them back.
    bool resource_wait{false};
    // The backend declined to order this run behind the run ahead of it. Kept
    // per run because the pair cannot change while both are live: the run ahead
    // stays ahead until it retires, and this one then reaches the front and
    // launches ordinarily. Without it the declined attempt would be re-issued
    // on every progress round for the rest of the predecessor's execution.
    bool joined_launch_declined{false};
    /**
     * This run was refused admission because some enabled diagnostic owner's
     * capacity is still held by a run ahead.
     *
     * `credit_blocked_generation` is what the refusal read *before* asking any
     * owner, so a waiter parking on it cannot miss a release that landed in
     * between. `credit_deadline_set` fixes one steady-clock deadline at the
     * first such refusal **while this run is at the front**: a successor's
     * decline starts nothing, polls do not renew it, and a later refusal of
     * the same run does not move it. On expiry only this un-launched run
     * fails; every run ahead keeps its own outcome.
     */
    bool credit_blocked{false};
    const char *credit_blocked_owner{nullptr};
    uint64_t credit_blocked_generation{0};
    bool credit_deadline_set{false};
    std::chrono::steady_clock::time_point credit_deadline{};
    // Every device span this run names is covered by a caller allocation of
    // this device context, and this run holds the borrow over each. Set once at
    // admission and cleared at release, so it answers "may this run join".
    // False can still mean partial lifetime references; caller_references_live
    // tracks the release obligation independently of full coverage.
    bool device_spans_borrowed{false};
    // This run's identity may still name something in the caller-buffer table: the borrow it took
    // at admission, or the declaration of what it produces that its own bind made later. Set for
    // every admitted run rather than only for one that borrowed, because the declaration is the
    // runtime's to make. Incomplete span coverage can also leave a partial borrow that still
    // needs releasing. Cleared when the release discharges both.
    bool caller_references_live{false};
    std::exception_ptr error;
    std::exception_ptr poison_error;
};

/**
 * The borrow identity for one run: its pipeline slot, offset so that it is
 * never zero.
 *
 * The slot rather than a counter of the lane's own, because this identity has
 * to be one the *runtime* can also name. A host-side orchestrator asks whether
 * a device span it is about to read is held by a run other than its own, and
 * the slot is what its HostApi binding already carries. A slot holds exactly
 * one run for that run's whole lifetime — the lane refuses an occupied one —
 * so the slot identifies the borrow holder for exactly as long as the borrow
 * exists.
 */
static uint64_t borrow_identity(const ChipRunState &run) { return static_cast<uint64_t>(run.lease.slot_id) + 1; }

struct ChipRunLaneState {
    explicit ChipRunLaneState(ChipWorker &worker) :
        worker(&worker),
        generations(worker.pipeline_depth(), 0) {
        // Captured once, here, while the worker is demonstrably alive. Every
        // later call reads them from the gate: this state holds the worker as
        // a raw pointer, and a parked waiter is exactly the thing that can
        // outlive it.
        credit_gate.entries = worker.credit_entries();
    }

    /** The right to be inside a runtime credit entry. Outlives the worker. */
    CreditCallGate credit_gate;

    /**
     * Read the credit generation, or 0 when there is nothing to read.
     *
     * Safe under this lane's mutex: it holds the gate lock only for the few
     * instructions that copy the targets, and the entry itself is
     * non-blocking. A 0 is "no progress information", never progress.
     */
    uint64_t credit_generation() {
        CreditCall call(credit_gate, Admit::Ordinary);
        if (!call || call.entries().generation == nullptr || call.entries().ctx == nullptr) return 0;
        return call.entries().generation(call.entries().ctx);
    }

    /**
     * Park until the credit generation leaves `seen`, or the budget elapses.
     *
     * Must be called with this lane's mutex released: the threads that advance
     * the generation are the collectors' writers, and a waiter holding the
     * lane mutex would block every other caller on this lane — including the
     * retirement that frees the capacity being waited for.
     *
     * Returns whether a wait was actually taken. False means there was
     * nothing to park on — the gate refused the call, or this runtime exports
     * no wait entry — which the caller must tell apart from a wait that
     * elapsed, because retrying the first is a spin and retrying the second
     * is progress.
     */
    bool wait_for_credit(uint64_t seen, int timeout_ms) {
        CreditCall call(credit_gate, Admit::Ordinary);
        if (!call || call.entries().wait == nullptr || call.entries().ctx == nullptr) return false;
        (void)call.entries().wait(call.entries().ctx, seen, timeout_ms);
        return true;
    }

    /** How the lane states its poisoned condition, for a caller reading one message. */
    static std::string poisoned_text(const std::exception_ptr &error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception &e) {
            return std::string("chip run lane is poisoned: ") + e.what();
        } catch (...) {
            return "chip run lane is poisoned by an unknown native failure";
        }
    }

    static std::string cause_text(const std::exception_ptr &error) {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception &e) {
            return e.what();
        } catch (...) {
            return "an unknown native failure";
        }
    }

    [[noreturn]] static void rethrow_as_poisoned(const std::exception_ptr &error) {
        throw std::runtime_error(poisoned_text(error));
    }

    void require_usable() const {
        if (closed) throw std::runtime_error("chip run lane is closed");
        if (admission_stopped) throw std::runtime_error("chip run lane has stopped admitting runs");
        if (poison != nullptr) rethrow_as_poisoned(poison);
        worker->validate_execution_context();
    }

    /**
     * Finish every run that has not crossed the launch boundary, without
     * launching it.
     *
     * The shape `ChipRun::abandon` uses for one run, applied to all of them: a
     * prepared run hands its native preparation back, a queued one holds none,
     * and neither reaches the device. Launched runs are left alone — they own
     * device work, events and resources that only their own drain can retire.
     *
     * Runs first rather than last, because an ordinary drain launches before it
     * waits: `drain_front` calls `launch_ready_prefix`, so draining a lane that
     * still held an activated prepared successor would put it on the device on
     * the way out.
     */
    void abandon_unlaunched() noexcept {
        for (auto it = fifo.begin(); it != fifo.end();) {
            const auto run = *it;
            if (run->phase == ChipRunState::Phase::LAUNCHED) {
                ++it;
                continue;
            }
            if (run->phase == ChipRunState::Phase::PREPARED) {
                bool finalized = true;
                try {
                    worker->finalize_native_run(run->native_run);
                } catch (...) {
                    finalized = false;
                    const std::exception_ptr finalize_error = std::current_exception();
                    if (run->error == nullptr) run->error = finalize_error;
                    poison_with(run, finalize_error);
                }
                release_device_spans(run, finalized);
            } else {
                // Queued: it never crossed the launch boundary, so no device
                // work of this run's can name a borrowed address and the
                // reference is simply given back.
                const bool plan_released = release_run_plan(run);
                release_device_spans(run, plan_released);
            }
            run->phase = ChipRunState::Phase::TERMINAL;
            it = fifo.erase(it);
        }
    }

    void rethrow_run_error(const std::shared_ptr<ChipRunState> &run) const {
        if (run->error == nullptr) return;
        // A run whose failure *is* the poison carries the same exception in
        // both, and reports as poisoned alone.
        if (run->poison_error != nullptr && run->poison_error == run->error) {
            rethrow_as_poisoned(run->poison_error);
        }
        // Two independent facts when the exceptions differ, and the caller is
        // owed both: why *this* run did not run, and that the lane is poisoned
        // by something else — a cleanup that failed after the run had already
        // failed does not correct the first, and leaving it out would hide
        // that no further run can be admitted. The run's own cause leads,
        // because it is the one the caller asked about.
        if (run->poison_error != nullptr) {
            throw std::runtime_error(cause_text(run->error) + "; " + poisoned_text(run->poison_error));
        }
        std::rethrow_exception(run->error);
    }

    /**
     * Whether the predecessor at the FIFO head can carry a prepared successor.
     *
     * A diagnostics config is no longer disqualifying on its own: the collector
     * pools and per-run state a preparation would otherwise have armed are built
     * and reset under the execution claim.
     *
     * The successor's own configuration does not enter, at any level: a bind's
     * host-orchestration phase state is held per pipeline slot, and everything
     * it hands to the resident collector is published under this claim.
     *
     * The predecessor's shape does. A successor's preparation runs its own host
     * graph build, which may read the bytes of a device argument, and what keeps
     * that read off bytes this run has not produced is this run's declaration of
     * what it produces — which covers only spans whose owner it could prove. A
     * predecessor holding no borrow over every device span it names has no such
     * proof, so it carries no preparation either: its successor prepares at the
     * front, after it has retired. That is the same condition `joinable_shape`
     * puts on the launch, applied to the earlier boundary.
     */
    bool permits_native_successor(const ChipRunState &predecessor) const {
        if (!worker->supports_concurrent_native_prepare()) return false;
        if (predecessor.phase != ChipRunState::Phase::LAUNCHED) return false;
        return joinable_shape(predecessor);
    }

    void prepare(const std::shared_ptr<ChipRunState> &run) {
        run->native_run = worker->prepare_native_run_for_lane(
            run->callable_id, &run->args, run->config, run->lease, run->run_id, run->dispatch_id, run->accepted_state,
            run->accepted_value, run->pipeline_leased, run->provenance
        );
        run->phase = ChipRunState::Phase::PREPARED;
        run->disposition = ChipRunPreparationDisposition::NATIVE_PREPARED;
    }

    void poison_with(const std::shared_ptr<ChipRunState> &run, std::exception_ptr error) {
        if (run->poison_error == nullptr) run->poison_error = error;
        if (poison == nullptr) poison = error;
    }

    /**
     * Take this run's borrow over every caller allocation its device tensors name.
     *
     * Allocates, so it can throw, and both callers run it inside their admission unwind: a throw
     * here has to remove the queued entry rather than leave it behind, and `release_device_spans`
     * discharges whatever this identity did establish, which after a throw here is nothing.
     *
     * No direction is recorded — `ChipStorageTaskArgs` carries no tag — so the borrow says only
     * that this run may reach those bytes, which is what a release has to respect either way. A
     * run whose spans do not all resolve still holds the known allocations and stays on the
     * serial path. Unknown external storage retains the caller's lifetime obligation.
     *
     * The span is the tensor's `nbytes()`, while the runtime's bind names `buffer.size` for the
     * same tensor. Both resolve to the containing allocation, so they agree on every question
     * either asks; they differ only for a tensor whose shape is empty while its buffer is not,
     * which takes no borrow here and is still declared there — over-declaration, and the safe
     * direction, since such a tensor produces no bytes.
     */
    void borrow_device_spans(const std::shared_ptr<ChipRunState> &run) {
        std::vector<ChipWorker::CallerDeviceSpan> spans;
        for (int32_t i = 0; i < run->args.tensor_count(); ++i) {
            const auto tensor = run->args.tensor(i);
            if (!tensor.is_device_memory()) continue;
            const uint64_t bytes = static_cast<uint64_t>(tensor.nbytes());
            if (bytes == 0) continue;
            spans.push_back(ChipWorker::CallerDeviceSpan{tensor.buffer.addr, bytes});
        }
        if (!spans.empty()) {
            run->device_spans_borrowed =
                worker->borrow_caller_device_spans(borrow_identity(*run), spans.data(), spans.size());
        }
        // Last, and whether or not a borrow was taken: reaching here is what admits the run, and
        // an admitted run's identity may name something in the table from here on — the borrow
        // above where it was taken, and the declaration its own bind makes later either way. A
        // throw above acquired nothing, so the identity stays absent and there is nothing to
        // discharge.
        run->caller_references_live = true;
    }

    /**
     * Give back what a run that never reached the device still holds.
     *
     * Only a waiting run holds anything here: its prepare saved a plan, kept a
     * reference on its slot's input staging and may still hold this context's
     * growth permission. Called before the span release on every path that
     * abandons a queued run, and it refuses when the run could still have
     * reached the device — which the caller turns into the run's failure rather
     * than handing its borrows back.
     */
    bool release_run_plan(const std::shared_ptr<ChipRunState> &run) noexcept {
        if (!run->resource_wait) return true;
        run->resource_wait = false;
        if (worker == nullptr) return true;
        int rc = 0;
        try {
            // Zero epoch: a run whose prepare asked to wait never received a
            // native-run identity, and the slot holds exactly one plan, so the
            // slot alone names what has to be given back.
            rc = worker->abort_run_plan(run->lease.slot_id, 0);
        } catch (...) {
            rc = -1;
        }
        if (rc != 0) {
            const auto failure = std::make_exception_ptr(
                std::runtime_error("chip run lane could not give back a waiting run's device resources")
            );
            if (run->error == nullptr) run->error = failure;
            poison_with(run, failure);
            return false;
        }
        return true;
    }

    /**
     * Discharge this run's caller references, or keep them when its last consumer is unproven.
     *
     * Both of them: the borrow over the allocations its arguments name, and the declaration of
     * which of them it produces that its own bind made. One call because they share this run's
     * identity and its lifetime — and it runs for every admitted run, not only one that borrowed,
     * since incomplete coverage can leave partial borrows as well as declarations.
     *
     * `proven_done` means successful finalize or a clean preparation rollback. Finalize
     * drains device work, copies outputs back and releases bindings; rollback releases
     * bindings without submitting an operator kernel. Either outcome is what
     * establishes that nothing on the device can still reach these bytes or still be writing
     * them. Failure establishes the opposite, and both facts are kept: the caller's release stays
     * refused rather than handing back pages the device may name, and the bytes this run declared
     * stay unreadable rather than being served mid-write.
     */
    void release_device_spans(const std::shared_ptr<ChipRunState> &run, bool proven_done) noexcept {
        if (!run->caller_references_live) return;
        worker->release_caller_device_borrow(borrow_identity(*run), !proven_done);
        run->caller_references_live = false;
        run->device_spans_borrowed = false;
    }

    void finish(const std::shared_ptr<ChipRunState> &run) noexcept {
        bool finalized = true;
        try {
            worker->finalize_native_run(run->native_run);
        } catch (...) {
            finalized = false;
            const std::exception_ptr finalize_error = std::current_exception();
            if (run->error == nullptr) run->error = finalize_error;
            poison_with(run, finalize_error);
        }
        // After the finalize, never before: that call drains the device work,
        // copies this run's outputs back and releases its bindings, so it is
        // the point at which the last consumer of a borrowed address is done.
        release_device_spans(run, finalized);
        run->phase = ChipRunState::Phase::TERMINAL;
        // By identity rather than from the front. Ordinarily the front is what
        // finishes first — the device executes the launched runs in order — but
        // a run that fails its own launch or is abandoned finishes wherever it
        // sits, and leaving it in the queue would make it the front of a FIFO
        // whose front is supposed to be the oldest live run.
        auto it = std::find(fifo.begin(), fifo.end(), run);
        if (it != fifo.end()) fifo.erase(it);
    }

    void fail_prepare(const std::shared_ptr<ChipRunState> &run) noexcept {
        run->error = std::current_exception();
        bool inputs_released = false;
        try {
            std::rethrow_exception(run->error);
        } catch (const RunPreparationRefused &) {
            inputs_released = true;
        } catch (...) {}
        const bool plan_released = release_run_plan(run);
        release_device_spans(run, inputs_released && plan_released);
        // No launch does not prove host bindings were released. An unknown
        // rollback keeps its borrows and prevents a new occupant of this slot.
        if (!inputs_released) poison_with(run, run->error);
        run->phase = ChipRunState::Phase::TERMINAL;
        const auto position = std::find(fifo.begin(), fifo.end(), run);
        if (position != fifo.end()) fifo.erase(position);
    }

    void fail_launch(const std::shared_ptr<ChipRunState> &run) noexcept {
        run->error = std::current_exception();
        finish(run);
    }

    void launch_front() noexcept {
        if (admission_stopped || fifo.empty()) return;
        const auto run = fifo.front();
        if (poison != nullptr) {
            if (run->phase == ChipRunState::Phase::PREPARED) {
                run->error = poison;
                run->poison_error = poison;
                finish(run);
            } else if (run->phase == ChipRunState::Phase::QUEUED) {
                run->error = poison;
                run->poison_error = poison;
                const bool plan_released = release_run_plan(run);
                release_device_spans(run, plan_released);
                run->phase = ChipRunState::Phase::TERMINAL;
                fifo.pop_front();
            }
            return;
        }
        if (!run->activated || run->phase == ChipRunState::Phase::LAUNCHED ||
            run->phase == ChipRunState::Phase::TERMINAL) {
            return;
        }
        if (run->phase == ChipRunState::Phase::QUEUED) {
            try {
                prepare(run);
            } catch (const ChipWorker::PreparedRunResourceWait &) {
                // Its resources are held by a consumer this worker will finish,
                // and its saved plan is what the next attempt continues from.
                // It keeps its slot, its lease and its borrow, and this returns
                // without launching: the run ahead of it is what frees those
                // resources, and the progress that retires it is the same
                // progress that brings this run back here.
                run->resource_wait = true;
                return;
            } catch (...) {
                fail_prepare(run);
                return;
            }
            run->resource_wait = false;
        }
        // Read before the attempt, so a release landing between the two cannot
        // be missed by a waiter that parks on it.
        const uint64_t generation_before = credit_generation();
        try {
            if (worker->launch_native_run(run->native_run, run->caller_stream) ==
                ChipWorker::NativeLaunch::DiagnosticCreditFull) {
                note_credit_blocked(run, generation_before, /*at_front=*/true);
                return;
            }
            run->credit_blocked = false;
            run->phase = ChipRunState::Phase::LAUNCHED;
            run->crossed_launch_fence = true;
        } catch (...) {
            fail_launch(run);
        }
    }

    /**
     * Record that this run could not be admitted for want of diagnostic
     * credit, and start its deadline the first time that happens at the front.
     *
     * The deadline is fixed once and never renewed: a later refusal of the
     * same run, another poll, or a successor's decline all leave it where the
     * first front refusal put it. Only a run that reaches the front gets one,
     * because only there is nothing else it could be waiting for.
     */
    void note_credit_blocked(const std::shared_ptr<ChipRunState> &run, uint64_t generation, bool at_front) noexcept {
        run->credit_blocked = true;
        run->credit_blocked_generation = generation;
        if (!at_front || run->credit_deadline_set) return;
        run->credit_deadline_set = true;
        run->credit_deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(simpler::dfx::runs::kCreditWaitBudgetMs);
    }

    /**
     * Fail a front run whose credit never came back inside its budget.
     *
     * Only this run, and only while it is still un-launched: every run ahead
     * keeps its own outcome and its own drain, and nothing device-side has
     * been submitted for this one, so there is no device failure to invent and
     * no reason to poison the lane. Returns true when the run was failed.
     */
    bool fail_if_credit_deadline_passed(const std::shared_ptr<ChipRunState> &run) noexcept {
        if (!run->credit_blocked || !run->credit_deadline_set) return false;
        if (run->phase == ChipRunState::Phase::LAUNCHED || run->phase == ChipRunState::Phase::TERMINAL) return false;
        if (std::chrono::steady_clock::now() < run->credit_deadline) return false;
        try {
            const std::string owner = run->credit_blocked_owner != nullptr ? run->credit_blocked_owner : "a collector";
            run->error = std::make_exception_ptr(
                std::runtime_error(
                    "chip run was not admitted within " + std::to_string(simpler::dfx::runs::kCreditWaitBudgetMs) +
                    " ms: " + owner + " had no diagnostic capacity free"
                )
            );
        } catch (...) {
            run->error = std::current_exception();
        }
        // The carrier is retired before anything is called released. A run
        // that reached PREPARED owns a native run on the worker — the
        // runtime's own prepared state, its bindings and its entry in
        // `native_runs_` — and only `finalize_native_run` destroys it.
        // Marking the run terminal without that leaves the carrier live for
        // the worker's whole life and makes the slot's reuse a claim rather
        // than a fact. A QUEUED run never crossed that boundary and owns no
        // carrier, so it has nothing to retire.
        bool finalized = true;
        if (run->phase == ChipRunState::Phase::PREPARED) {
            try {
                worker->finalize_native_run(run->native_run);
            } catch (...) {
                // The admission failure is this run's first cause and is kept:
                // a cleanup that then failed is a second fact about the lane,
                // not a replacement for why the run never launched. It
                // poisons the lane on the existing terms, because an
                // un-retired carrier is exactly what a later run must not
                // prepare over.
                finalized = false;
                poison_with(run, std::current_exception());
            }
        }
        // After the finalize, never before, for the reason `finish` gives: it
        // is the call that releases this run's bindings, so it is the point at
        // which the last consumer of a borrowed address is done. A finalize
        // that failed cannot prove that, and the reference is kept.
        release_device_spans(run, finalized);
        run->phase = ChipRunState::Phase::TERMINAL;
        auto it = std::find(fifo.begin(), fifo.end(), run);
        if (it != fifo.end()) fifo.erase(it);
        return true;
    }

    /** How many runs in the queue have crossed the launch boundary. */
    size_t launched_count() const {
        size_t count = 0;
        for (const auto &candidate : fifo) {
            if (candidate->phase == ChipRunState::Phase::LAUNCHED) ++count;
        }
        return count;
    }

    /**
     * Whether a run's own shape keeps it inside the joined-launch scope.
     *
     * A host tensor is copied into the runner-owned staging the run retains for
     * its whole lifetime, which is what makes a run that fails while another is
     * queued behind it safe to abandon without freeing anything the device may
     * still read.
     *
     * A device tensor is the caller's address, and its lifetime is the
     * caller's. What brings it inside the scope is the borrow: a run holds one
     * over the caller allocation covering every device span it names, so the
     * caller's release is refused for as long as the device may still reach
     * those bytes. Joined launch requires full coverage: unresolved spans keep
     * the run on the serial path, with their lifetime guaranteed by the caller.
     * Known allocations remain borrowed even when coverage is incomplete.
     */
    static bool joinable_shape(const ChipRunState &run) {
        for (int32_t i = 0; i < run.args.tensor_count(); ++i) {
            if (!run.args.tensor(i).is_device_memory()) continue;
            if (!run.device_spans_borrowed) return false;
        }
        return true;
    }

    /**
     * Whether `successor` may reach the device while `predecessor` executes.
     *
     * The backend answers the part about itself, and it answers per moment
     * rather than once: a code publication can leave it unable to order two
     * runs even where it normally can. A successor its own preparation pushed
     * back to depth one is excluded here too, since it holds no native
     * preparation to launch.
     *
     * Both runs are checked for shape, not just the joining one: the ordering
     * makes them share a device generation that either one's failure can end.
     * A live communication resource disqualifies every run on the worker, for
     * the same reason — the child releases those before its reset.
     *
     * Every device diagnostic owner can hold two runs open: each stamps the
     * run that produced a record onto it, and each run's accounting is
     * published into a terminal bank its own identity reserved. What a pair
     * must still agree on is the *settings* — one shared word per owner that
     * the device latches once — and whether every enabled owner has capacity
     * for one more. `permits_joined_diagnostic_launch` decides the first here,
     * and asks the backend the second.
     */
    bool permits_joined_launch(const ChipRunState &predecessor, const ChipRunState &successor) const {
        if (!worker->supports_joined_native_launch()) return false;
        if (predecessor.phase != ChipRunState::Phase::LAUNCHED) return false;
        if (successor.phase != ChipRunState::Phase::PREPARED) return false;
        if (!successor.activated || successor.depth_one_fallback || successor.error != nullptr) return false;
        if (successor.joined_launch_declined) return false;
        if (!joinable_shape(predecessor) || !joinable_shape(successor)) return false;
        if (worker->holds_live_comm_resources()) return false;
        if (predecessor.config.diagnostics_any() || successor.config.diagnostics_any()) {
            return permits_joined_diagnostic_launch(predecessor, successor);
        }
        return true;
    }

    /**
     * Whether two diagnostic runs may be ordered on the device together.
     *
     * The five device owners' settings must be *equal* on both runs: each
     * owner's producers stamp their own run's identity onto what they publish,
     * so two open runs are told apart by that, but a level or an event mode is
     * one shared word the device latches once. Everything else about an owner
     * — whether its retention is on, whether its collector can hold two runs
     * open, whether a bank is free — is the backend's to answer, because it is
     * the backend that owns the collector. What is decided here is only what
     * the lane can see for itself.
     *
     * Re-asked every round and never remembered. A decline here is a statement
     * about right now — capacity comes back, and a run declined this round can
     * join the next — which is exactly what `joined_launch_declined` must not
     * be set for: that flag means the backend refused an attempt.
     */
    bool permits_joined_diagnostic_launch(const ChipRunState &predecessor, const ChipRunState &successor) const {
        const CallConfig &pred = predecessor.config;
        const CallConfig &succ = successor.config;
        if (pred.enable_chip_swimlane != succ.enable_chip_swimlane) return false;
        if (pred.enable_dump_args != succ.enable_dump_args) return false;
        if (pred.enable_pmu != succ.enable_pmu) return false;
        if (pred.enable_dep_gen != succ.enable_dep_gen) return false;
        if (pred.enable_scope_stats != succ.enable_scope_stats) return false;
        return worker->supports_joined_diagnostic_launch(successor.native_run, predecessor.native_run);
    }

    /**
     * Launch the front, then every activated run behind it that may follow it
     * onto the device, up to the configured launch depth.
     *
     * The launched runs are a prefix of the queue: only the front is launched
     * on its own, and each further one is ordered behind the newest launched
     * run, so the front is the *oldest* launched run rather than the only one.
     *
     * A backend that declines a join leaves that run prepared, to launch
     * ordinarily once it reaches the front. Whether the decline is remembered
     * depends on what could change it. For a pair with no diagnostics the
     * reasons cannot change while the run ahead executes, so re-asking would
     * issue a device call per progress round and get the same answer — that
     * one is latched. For a diagnostic pair the reason may be a bank or an
     * admission taken between the capacity pre-check and the attempt, and the
     * next round may well find one free, so it is re-asked. Whether the *lane*
     * would permit a join is re-tested every round either way.
     *
     * A join that fails rather than declining is the successor's own failure,
     * and it stops the lane. By the launch transaction's own grading such a
     * failure may already have reached the device, and whatever it holds sits in
     * the stream pair the run ahead is still executing on; nothing here can tell
     * that apart from a clean rollback.
     */
    void launch_ready_prefix() noexcept {
        if (admission_stopped) return;
        launch_front();
        if (poison != nullptr) return;
        size_t launched = launched_count();
        while (launched != 0 && launched < static_cast<size_t>(worker->launch_depth()) && launched < fifo.size()) {
            const auto predecessor = fifo[launched - 1];
            const auto successor = fifo[launched];
            if (!permits_joined_launch(*predecessor, *successor)) return;
            bool joined = false;
            const uint64_t generation_before = credit_generation();
            try {
                joined = worker->launch_native_run_joined(
                    successor->native_run, predecessor->native_run, successor->caller_stream
                );
            } catch (...) {
                successor->error = std::current_exception();
                // Before `finish`, so that a finalize failure on top of this
                // cannot become the reason the lane stopped: the launch is.
                poison_with(successor, successor->error);
                finish(successor);
                return;
            }
            if (!joined) {
                // A non-diagnostic pair's reasons cannot change while the run
                // ahead executes, so the decline is remembered and the backend
                // is not asked again. A diagnostic pair's can: the pre-check
                // above already passed its capacity terms, so a refusal here is
                // a bank or an admission taken between the two, and the next
                // round may well find one. Remembering it would deny this run
                // the join for the rest of its predecessor's execution over a
                // lost race.
                if (!predecessor->config.diagnostics_any() && !successor->config.diagnostics_any()) {
                    successor->joined_launch_declined = true;
                } else {
                    // No deadline: a successor is not at the front, and what
                    // it waits for is also the run ahead of it retiring. It
                    // gets its own deadline if it is still blocked when it
                    // reaches the front.
                    note_credit_blocked(successor, generation_before, /*at_front=*/false);
                }
                return;
            }
            successor->phase = ChipRunState::Phase::LAUNCHED;
            successor->crossed_launch_fence = true;
            ++launched;
        }
    }

    /**
     * Prepare a queued run while the runs ahead of it execute, if it may be.
     *
     * A backend that reports this run cannot be prepared beside the active one leaves it queued:
     * it keeps its slot, its borrow and whatever its aborted bind declared, nothing of the run
     * ahead is touched, and it prepares again once it reaches the front. That is also the answer
     * when its graph build needs bytes the run ahead has not finished producing — the value it
     * waits for exists by the time it is at the front, which is the behaviour the same program had
     * before a device argument could be enqueued early. One attempt only: the reasons cannot
     * change while the run ahead is still executing.
     *
     * The predecessor is the run immediately ahead in the FIFO, whatever the lane's capacity: each
     * run is prepared beside the one it will be ordered behind, so a third run is judged against
     * the second rather than against the front.
     */
    void prepare_successor_if_eligible(const std::shared_ptr<ChipRunState> &run) {
        const auto position = std::find(fifo.begin(), fifo.end(), run);
        if (position == fifo.end() || position == fifo.begin()) return;
        if (run->phase != ChipRunState::Phase::QUEUED || run->depth_one_fallback) return;
        if (!permits_native_successor(**std::prev(position))) return;
        try {
            prepare(run);
        } catch (const ChipWorker::PreparedRunIncompatible &) {
            run->depth_one_fallback = true;
            return;
        } catch (const ChipWorker::PreparedRunResourceWait &) {
            // Same treatment as at the front: it keeps everything it holds and
            // is offered again, because what it waits for is another run's
            // retirement rather than its own position in the queue.
            run->resource_wait = true;
            return;
        } catch (...) {
            fail_prepare(run);
        }
    }

    /**
     * Offer every queued run behind the launched prefix its one preparation attempt.
     *
     * Front to back, because a run's eligibility is decided by the run ahead of it: preparing the
     * second is what can make the third eligible in the same round. A run that declines keeps its
     * place, and the walk stops at the first one that cannot be prepared — nothing behind it can
     * be judged against a predecessor that is still queued.
     */
    void prepare_queued_successors() {
        for (size_t index = 1; index < fifo.size(); ++index) {
            const auto candidate = fifo[index];
            if (candidate->phase != ChipRunState::Phase::QUEUED) continue;
            prepare_successor_if_eligible(candidate);
            if (candidate->phase == ChipRunState::Phase::QUEUED) return;
        }
    }

    bool progress(const std::shared_ptr<ChipRunState> &target) {
        if (target->phase == ChipRunState::Phase::TERMINAL) return true;
        worker->validate_execution_context();
        if (fifo.empty()) throw std::runtime_error("chip run lane lost a nonterminal run");
        if (fifo.front() != target) {
            (void)progress(fifo.front());
            if (target->phase == ChipRunState::Phase::TERMINAL) return true;
            if (fifo.empty() || fifo.front() != target) return false;
        }

        launch_ready_prefix();
        // Enforced here rather than around a wait, so it applies to every
        // caller: the local L3 child progresses with `done()` and takes no
        // credit wait at all, and a credit-starved run on that path would
        // otherwise never fail. It is a timestamp comparison — no blocking,
        // no signal — and it fails only the run whose own deadline passed.
        if (fail_if_credit_deadline_passed(target)) {
            launch_ready_prefix();
            return true;
        }
        // Whatever the target is: a launch just now may have made the run behind the newly
        // launched one eligible, and that run is judged against its own predecessor rather than
        // against the front.
        prepare_queued_successors();
        if (target->phase == ChipRunState::Phase::TERMINAL) {
            launch_ready_prefix();
            return true;
        }
        if (target->phase != ChipRunState::Phase::LAUNCHED) return false;

        try {
            if (!worker->poll_native_run(target->native_run)) return false;
        } catch (...) {
            const std::exception_ptr poll_error = std::current_exception();
            finish(target);
            if (target->error == nullptr) target->error = poll_error;
            poison_with(target, target->error);
            launch_ready_prefix();
            return true;
        }
        finish(target);
        launch_ready_prefix();
        return true;
    }

    /**
     * What one drain attempt did, for a caller that loops on it.
     *
     * `CreditBlocked` is the one outcome the caller must not retry under this
     * lane's mutex: the front is activated and prepared but was refused
     * admission, and the capacity that would release it is freed by a
     * collector's writer, which cannot make progress a caller observes while
     * that caller holds the lane. Looping on it is exactly the spin this
     * reports instead of performing.
     */
    enum class DrainOutcome : int { Progressed, CreditBlocked };

    DrainOutcome drain_front() noexcept {
        if (fifo.empty()) return DrainOutcome::Progressed;
        const auto run = fifo.front();
        if (poison != nullptr && run->phase == ChipRunState::Phase::QUEUED) {
            run->error = poison;
            run->poison_error = poison;
            const bool plan_released = release_run_plan(run);
            release_device_spans(run, plan_released);
            run->phase = ChipRunState::Phase::TERMINAL;
            fifo.pop_front();
            return DrainOutcome::Progressed;
        }
        if (run->phase == ChipRunState::Phase::QUEUED && !run->activated) {
            const bool plan_released = release_run_plan(run);
            release_device_spans(run, plan_released);
            run->phase = ChipRunState::Phase::TERMINAL;
            fifo.pop_front();
            return DrainOutcome::Progressed;
        }
        launch_ready_prefix();
        if (run->phase == ChipRunState::Phase::TERMINAL) return DrainOutcome::Progressed;
        // Its own deadline may already have passed, in which case the run
        // fails here rather than being reported as blocked forever.
        if (fail_if_credit_deadline_passed(run)) return DrainOutcome::Progressed;
        if (run->phase == ChipRunState::Phase::PREPARED && (!run->activated || poison != nullptr)) {
            if (poison != nullptr && run->error == nullptr) {
                run->error = poison;
                run->poison_error = poison;
            }
            finish(run);
            return DrainOutcome::Progressed;
        }
        if (run->phase == ChipRunState::Phase::LAUNCHED) {
            try {
                worker->wait_native_run(run->native_run);
            } catch (...) {
                run->error = std::current_exception();
                poison_with(run, run->error);
            }
            finish(run);
            return DrainOutcome::Progressed;
        }
        // Activated, prepared, and still not launched: the launch above
        // refused it for credit. Nothing this call can do moves it.
        if (run->credit_blocked && run->activated && run->phase == ChipRunState::Phase::PREPARED) {
            return DrainOutcome::CreditBlocked;
        }
        return DrainOutcome::Progressed;
    }

    /**
     * Bring this run's admission deadline to now.
     *
     * Used when nothing further can admit it — the lane is closing and the
     * credit gate is already detached, so no release can reach a waiter. The
     * next drain fails the run with its own credit reason, which is the same
     * outcome the deadline would have produced, reached without spending the
     * budget waiting for capacity that cannot arrive.
     */
    void expire_credit_deadline(const std::shared_ptr<ChipRunState> &run) noexcept {
        run->credit_deadline_set = true;
        run->credit_deadline = std::chrono::steady_clock::now();
    }

    /**
     * How long this run may still wait before its own deadline fails it.
     *
     * Zero when the deadline has passed, and the whole budget when the run has
     * not been refused at the front yet — a successor blocked behind another
     * run has no deadline of its own to bound the wait by.
     */
    /**
     * Park until diagnostic credit moves, with this lane's mutex released.
     *
     * `lk` owns this lane's mutex on entry and owns it again on return. In
     * between it is released, because the threads that free capacity are the
     * collectors' writers and a waiter holding the lane would block the very
     * retirement it waits for — and, on the serial route, the interpreter.
     *
     * Nothing is carried across the unlock but values: the run by
     * `shared_ptr`, the generation and the budget by copy. No FIFO iterator
     * and no native carrier survives it, and the runtime entry is reached
     * through the gate rather than through `worker`, which this state holds
     * as a raw pointer a parked waiter can outlive.
     *
     * The budget is the smaller of what this run has left before its own
     * admission deadline and what the caller has left before its own: the
     * first ends the run, the second ends only the wait, and conflating them
     * would let a patient caller outlive a run's deadline or an impatient one
     * shorten it.
     *
     * A runtime that exports no credit entry, or a gate that has refused this
     * call, leaves nothing to park on. Retrying then would be a spin against
     * a signal nobody will raise, so the run's deadline is brought forward
     * and the next attempt fails it — which is what a backend without these
     * entries did before them: a capacity refusal ended the run.
     *
     * @return true when the caller should re-attempt under the re-acquired
     *         lock, false when it must not — the lane closed, was poisoned,
     *         stopped admitting, lost its worker, or this run is no longer the
     *         front it was waiting as.
     */
    bool
    park_for_credit(std::unique_lock<std::mutex> &lk, const std::shared_ptr<ChipRunState> &run, int caller_budget_ms) {
        const uint64_t seen = run->credit_blocked_generation;
        int budget = credit_wait_budget_ms(run);
        if (caller_budget_ms >= 0 && caller_budget_ms < budget) budget = caller_budget_ms;
        if (budget < 0) budget = 0;

        lk.unlock();
        const bool waited = wait_for_credit(seen, budget);
        lk.lock();

        if (!waited) expire_credit_deadline(run);

        // Re-read every decision input: the lane could have been closed,
        // poisoned or drained while this thread held nothing.
        if (closed || admission_stopped || poison != nullptr || worker == nullptr) return false;
        if (run->phase == ChipRunState::Phase::TERMINAL) return false;
        if (fifo.empty() || fifo.front() != run) return false;
        return true;
    }

    int credit_wait_budget_ms(const std::shared_ptr<ChipRunState> &run) const noexcept {
        if (!run->credit_deadline_set) return simpler::dfx::runs::kCreditWaitBudgetMs;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                              run->credit_deadline - std::chrono::steady_clock::now()
        )
                              .count();
        if (left <= 0) return 0;
        return static_cast<int>(left);
    }

    // Block on the device for the launched front, for waiters with no deadline
    // to bound them. The front is the oldest launched run, and the device
    // executes the launched runs in queue order, so its completion is the next
    // event that lets any waiter in the FIFO advance. Re-polling instead would
    // hold a core for the whole run — the case codestyle rule 5 sends to a
    // wakeup primitive rather than a busy loop. Reports whether it actually
    // blocked, so a caller that cannot be unblocked this way does not spin on
    // it.
    bool block_on_front() noexcept {
        if (fifo.empty()) return false;
        const auto front = fifo.front();
        if (front->phase != ChipRunState::Phase::LAUNCHED) return false;
        try {
            worker->wait_native_run(front->native_run);
        } catch (...) {
            const std::exception_ptr wait_error = std::current_exception();
            if (front->error == nullptr) front->error = wait_error;
            poison_with(front, front->error);
        }
        finish(front);
        launch_ready_prefix();
        return true;
    }

    ChipWorker *worker;
    mutable std::mutex mu;
    std::deque<std::shared_ptr<ChipRunState>> fifo;
    std::vector<uint64_t> generations;
    uint64_t direct_generation{0};
    std::exception_ptr poison;
    bool closed{false};
    // Set when a caller has established that nothing further may reach the
    // device. Distinct from `poison`, which says a run failed: this says the
    // lane must not launch, while the runs already launched keep their own
    // outcomes and their own drains.
    bool admission_stopped{false};
};

ChipRun::ChipRun(std::shared_ptr<ChipRunLaneState> lane, std::shared_ptr<ChipRunState> run) :
    lane_(std::move(lane)),
    run_(std::move(run)) {}

bool ChipRun::done() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return lane_->progress(run_);
}

bool ChipRun::wait_until(Deadline deadline) {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    const bool unbounded = deadline == Deadline::max();
    while (true) {
        std::shared_ptr<ChipRunState> blocked_front;
        {
            std::unique_lock<std::mutex> lk(lane_->mu);
            if (lane_->progress(run_)) {
                lane_->rethrow_run_error(run_);
                return true;
            }
            // An unbounded waiter has no deadline to end its loop, so polling
            // here would spin for the whole run. Block on the device instead;
            // if nothing is blockable yet the poll loop below still applies.
            if (unbounded && lane_->block_on_front()) continue;
            // Nothing to block on at the device, because the front was never
            // launched: it is waiting for a collector's capacity. That is the
            // other thing this loop would otherwise spin on, and the only
            // place it can be waited for is outside this mutex.
            if (!lane_->fifo.empty()) {
                const auto &front = lane_->fifo.front();
                if (front->credit_blocked && front->phase == ChipRunState::Phase::PREPARED) blocked_front = front;
            }
        }
        if (blocked_front != nullptr) {
            // Bounded by whichever of the two deadlines comes first. They mean
            // different things and are never conflated: the caller's ends this
            // wait and leaves the run live, the run's ends the run.
            int caller_budget_ms = -1;
            if (!unbounded) {
                const auto left =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
                caller_budget_ms = left <= 0 ? 0 : static_cast<int>(left);
            }
            std::unique_lock<std::mutex> lk(lane_->mu);
            (void)lane_->park_for_credit(lk, blocked_front, caller_budget_ms);
        }
        if (Clock::now() >= deadline) return false;
    }
}

ChipRun::Deadline ChipRun::credit_deadline_for_test() const {
    if (lane_ == nullptr || run_ == nullptr) return Deadline{};
    std::lock_guard<std::mutex> lk(lane_->mu);
    return run_->credit_deadline_set ? run_->credit_deadline : Deadline{};
}

void ChipRun::expire_credit_deadline_for_test() {
    if (lane_ == nullptr || run_ == nullptr) return;
    std::lock_guard<std::mutex> lk(lane_->mu);
    run_->credit_deadline_set = true;
    run_->credit_deadline = Clock::now() - std::chrono::milliseconds(1);
}

void ChipRun::activate() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    if (run_->phase == ChipRunState::Phase::TERMINAL) {
        lane_->rethrow_run_error(run_);
        return;
    }
    lane_->worker->validate_execution_context();
    run_->activated = true;
    lane_->launch_ready_prefix();
    // Activating a queued run is what launches it, and its launch is what lets the run behind it
    // be prepared — so this follows every activation, not only the front's.
    lane_->prepare_queued_successors();
}

void ChipRun::abandon() {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    if (run_->phase == ChipRunState::Phase::TERMINAL) return;
    if (run_->phase == ChipRunState::Phase::LAUNCHED) {
        throw std::runtime_error("cannot abandon a launched ChipRun");
    }
    lane_->worker->validate_execution_context();
    auto it = std::find(lane_->fifo.begin(), lane_->fifo.end(), run_);
    if (it == lane_->fifo.end()) throw std::runtime_error("chip run lane lost an unlaunched run");
    if (run_->phase == ChipRunState::Phase::PREPARED) {
        bool finalized = true;
        try {
            lane_->worker->finalize_native_run(run_->native_run);
        } catch (...) {
            finalized = false;
            run_->error = std::current_exception();
            lane_->poison_with(run_, run_->error);
        }
        lane_->release_device_spans(run_, finalized);
    } else {
        const bool plan_released = lane_->release_run_plan(run_);
        lane_->release_device_spans(run_, plan_released);
    }
    run_->phase = ChipRunState::Phase::TERMINAL;
    lane_->fifo.erase(it);
    if (run_->error != nullptr) std::rethrow_exception(run_->error);
}

bool ChipRun::launched() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return run_->crossed_launch_fence;
}

bool ChipRun::lane_poisoned() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return lane_->poison != nullptr;
}

ChipRunPreparationDisposition ChipRun::preparation_disposition() const {
    if (lane_ == nullptr || run_ == nullptr) throw std::runtime_error("empty ChipRun handle");
    std::lock_guard<std::mutex> lk(lane_->mu);
    return run_->disposition;
}

ChipRunLane::ChipRunLane(ChipWorker &worker) :
    state_(std::make_shared<ChipRunLaneState>(worker)) {}

ChipRunLane::~ChipRunLane() {
    try {
        close();
    } catch (...) {}
}

ChipRun ChipRunLane::submit(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
    uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value, bool activated,
    NativeRunProvenance provenance
) {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->require_usable();
    if (lease.reserved != 0 || lease.generation == 0 || lease.slot_id >= state_->generations.size()) {
        throw std::runtime_error("chip run lane received an invalid pipeline lease");
    }
    if (lease.generation < state_->generations[lease.slot_id]) {
        throw std::runtime_error("chip run lane pipeline lease generation is stale");
    }
    for (const auto &candidate : state_->fifo) {
        if ((run_id != 0 && candidate->run_id == run_id) ||
            (dispatch_id != 0 && candidate->dispatch_id == dispatch_id)) {
            throw std::runtime_error("chip run lane received a duplicate run or dispatch identity");
        }
        if (candidate->lease.slot_id == lease.slot_id) {
            throw std::runtime_error("chip run lane pipeline slot is already occupied");
        }
    }
    if (state_->fifo.size() >= state_->generations.size()) {
        throw std::runtime_error("chip run lane capacity exceeded before native preparation");
    }
    if (!state_->fifo.empty() && state_->fifo.front()->phase == ChipRunState::Phase::LAUNCHED &&
        dispatch_id < state_->fifo.front()->dispatch_id) {
        throw std::runtime_error("chip run lane cannot reorder before an active dispatch");
    }
    auto run = std::make_shared<ChipRunState>();
    run->callable_id = callable_id;
    run->args = args;
    run->config = config;
    run->lease = lease;
    run->run_id = run_id;
    run->dispatch_id = dispatch_id;
    run->provenance = provenance;
    run->accepted_state = accepted_state;
    run->accepted_value = accepted_value;
    run->pipeline_leased = true;
    run->activated = activated;
    state_->generations[lease.slot_id] = lease.generation;
    auto position = std::upper_bound(
        state_->fifo.begin(), state_->fifo.end(), dispatch_id,
        [](uint64_t id, const std::shared_ptr<ChipRunState> &candidate) {
            return id < candidate->dispatch_id;
        }
    );
    state_->fifo.insert(position, run);

    try {
        // Inside the unwind, and before anything can prepare or launch: from admission on, the
        // caller's release of an address this run names is refused, and preparation — which is
        // what first hands those addresses to the device — is the next statement. Borrowing
        // allocates, so a failure here has to leave no queued entry behind.
        state_->borrow_device_spans(run);
        if (state_->fifo.front() == run) {
            state_->launch_ready_prefix();
            state_->prepare_queued_successors();
        } else {
            state_->prepare_successor_if_eligible(run);
        }
    } catch (...) {
        run->error = std::current_exception();
        // Only what was acquired: this run threw while building its span list or taking its
        // borrow, so it established nothing and gives nothing back. The run ahead is untouched.
        //
        // `proven_done` is read from the launch fence rather than asserted: admission also runs
        // `launch_ready_prefix`, so a run that has already crossed it must not have its reference
        // dropped as if nothing could reach those bytes. Today nothing throws after that point —
        // `launch_ready_prefix` is noexcept — and this keeps the statement true by construction
        // rather than by that coincidence.
        const bool plan_released = state_->release_run_plan(run);
        state_->release_device_spans(run, plan_released && !run->crossed_launch_fence);
        run->phase = ChipRunState::Phase::TERMINAL;
        auto it = std::find(state_->fifo.begin(), state_->fifo.end(), run);
        if (it != state_->fifo.end()) state_->fifo.erase(it);
    }
    return ChipRun(state_, std::move(run));
}

ChipRun ChipRunLane::submit(
    int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, volatile int32_t *accepted_state,
    int32_t accepted_value, void *caller_stream
) {
    std::unique_lock<std::mutex> lk(state_->mu);
    state_->require_usable();
    if (caller_stream != nullptr) state_->worker->validate_caller_stream(caller_stream);
    if (state_->generations.empty()) throw std::runtime_error("chip run lane has no runtime slots");

    // Direct admission follows the runtime contract but keeps the lane as its
    // only authority. A run whose predecessor permits a native successor may queue behind it up to
    // the lane's capacity; otherwise admission drains the front and retains depth-one behavior.
    // In particular, the submit past that capacity waits here before a slot generation is
    // minted or native preparation begins.
    while (!state_->fifo.empty()) {
        const bool has_successor_capacity =
            state_->fifo.size() < state_->generations.size() && state_->permits_native_successor(*state_->fifo.back());
        if (has_successor_capacity) break;
        const auto front = state_->fifo.front();
        if (state_->drain_front() == ChipRunLaneState::DrainOutcome::CreditBlocked) {
            // Retrying under this lock would spin: the capacity comes back on
            // a collector's writer, which this thread is holding the lane
            // against. Park outside it instead, bounded by that run's own
            // admission deadline, and re-attempt once credit moves.
            if (!state_->park_for_credit(lk, front, /*caller_budget_ms=*/-1)) {
                state_->require_usable();
                continue;
            }
        }
        state_->require_usable();
        state_->launch_ready_prefix();
    }

    uint32_t slot_id = 0;
    for (; slot_id < state_->generations.size(); ++slot_id) {
        const bool occupied = std::any_of(
            state_->fifo.begin(), state_->fifo.end(), [slot_id](const std::shared_ptr<ChipRunState> &candidate) {
                return candidate->lease.slot_id == slot_id;
            }
        );
        if (!occupied) break;
    }
    if (slot_id == state_->generations.size()) {
        throw std::runtime_error("chip run lane has no free direct runtime slot after admission");
    }
    if (state_->direct_generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("chip run lane exhausted its direct generation space");
    }
    ++state_->direct_generation;

    auto run = std::make_shared<ChipRunState>();
    run->callable_id = callable_id;
    run->args = args;
    run->config = config;
    run->lease = PipelineSlotLease{slot_id, 0, state_->direct_generation};
    run->accepted_state = accepted_state;
    run->accepted_value = accepted_value;
    run->caller_stream = caller_stream;
    run->pipeline_leased = false;
    run->activated = true;
    state_->fifo.push_back(run);
    try {
        state_->borrow_device_spans(run);
    } catch (...) {
        // This overload reports an admission failure by throwing, so the queued entry goes back
        // out and the caller sees its own error. The run ahead keeps its own borrow, and this
        // run's reference is discharged as proven-done only while it has not crossed the launch
        // fence — the same rule as the other overload's unwind.
        state_->release_device_spans(run, !run->crossed_launch_fence);
        run->phase = ChipRunState::Phase::TERMINAL;
        state_->fifo.pop_back();
        throw;
    }
    if (state_->fifo.front() == run) {
        state_->launch_ready_prefix();
    } else {
        state_->prepare_successor_if_eligible(run);
    }
    if (caller_stream != nullptr) {
        // Returning a prepared-only handle would let the caller enqueue its
        // consumer ahead of this run's exit wait. A declined join waits for
        // the predecessor here, before the caller can append any such work.
        state_->launch_ready_prefix();
        while (run->phase != ChipRunState::Phase::LAUNCHED && run->phase != ChipRunState::Phase::TERMINAL) {
            const auto front = state_->fifo.empty() ? nullptr : state_->fifo.front();
            if (state_->drain_front() == ChipRunLaneState::DrainOutcome::CreditBlocked && front != nullptr) {
                // Same reason as the capacity loop above: nothing this thread
                // can do under the lane mutex releases a collector's slot.
                (void)state_->park_for_credit(lk, front, /*caller_budget_ms=*/-1);
            }
            state_->launch_ready_prefix();
        }
        state_->rethrow_run_error(run);
        if (!run->crossed_launch_fence) throw std::runtime_error("caller-stream run did not reach launch");
    }
    return ChipRun(state_, std::move(run));
}

void ChipRunLane::drain() {
    std::unique_lock<std::mutex> lk(state_->mu);
    if (state_->worker != nullptr) state_->worker->validate_execution_context();
    while (!state_->fifo.empty()) {
        const auto front = state_->fifo.front();
        if (state_->drain_front() == ChipRunLaneState::DrainOutcome::CreditBlocked) {
            // The run's own admission deadline bounds this: when it passes,
            // the next drain fails that run and the queue advances, so a drain
            // terminates even if credit never comes back. A park that declines
            // because the queue moved under it is progress and is retried;
            // one that declines with this run still at the front is the lane
            // going away, and there is nothing further to drain.
            if (!state_->park_for_credit(lk, front, /*caller_budget_ms=*/-1)) {
                if (!state_->fifo.empty() && state_->fifo.front() == front) break;
            }
        }
    }
    if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
}

void ChipRunLane::close() {
    // Refusal before mutation. A close on the wrong device context is not a
    // close: it changes nothing and throws, and the worker it was called on is
    // still open afterwards. Cancelling first would have detached that open
    // worker's credit entries on the way to throwing, leaving a lane that can
    // never wait for capacity again and no record of why.
    //
    // Taken under the lane mutex so the worker pointer cannot be cleared
    // between the read and the call, and released again before the gate is
    // touched — `close_credit_gate` drains callers who need that mutex.
    {
        std::lock_guard<std::mutex> validate(state_->mu);
        if (state_->worker != nullptr) state_->worker->validate_execution_context();
    }

    // Now that the context is this one's: cancelling makes every parked
    // waiter's predicate true, and the drain below needs those waiters to
    // return in order to finish. The other order is the wait-owns-the-lease /
    // close-needs-the-lease cycle — a waiter parked for its whole budget would
    // hold a lease this close waits on while holding nothing it could take.
    // `close_credit_gate` is also what proves every credit call has left
    // before the targets are detached, so a later unload has no reader.
    //
    // Exactly one thread owns this; every other close joins it and returns
    // only once detaching is a fact.
    (void)close_credit_gate(state_->credit_gate);

    std::unique_lock<std::mutex> lk(state_->mu);
    if (state_->closed) {
        if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
        return;
    }
    while (!state_->fifo.empty()) {
        const auto front = state_->fifo.front();
        if (state_->drain_front() == ChipRunLaneState::DrainOutcome::CreditBlocked) {
            // The gate was detached above, so no release can reach a waiter
            // any more and no amount of waiting will admit this run. Waiting
            // out its budget would be a spin against a signal nobody will
            // raise, so its deadline is brought to now and the next drain
            // fails it with its own credit reason — the same outcome, reached
            // without the wait.
            state_->expire_credit_deadline(front);
        }
    }
    state_->closed = true;
    if (state_->poison != nullptr) std::rethrow_exception(state_->poison);
}

void ChipRunLane::detach_after_failed_teardown(std::exception_ptr error) noexcept {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->closed = true;
    state_->poison = error;
    for (const auto &run : state_->fifo) {
        run->error = error;
        run->poison_error = error;
        run->phase = ChipRunState::Phase::TERMINAL;
    }
    state_->worker = nullptr;
}

void ChipRunLane::stop_admission() noexcept {
    std::lock_guard<std::mutex> lk(state_->mu);
    state_->admission_stopped = true;
    try {
        if (state_->worker != nullptr) state_->worker->validate_execution_context();
    } catch (...) {
        return;
    }
    state_->abandon_unlaunched();
}

bool ChipRunLane::poisoned() const {
    std::lock_guard<std::mutex> lk(state_->mu);
    return state_->poison != nullptr;
}
