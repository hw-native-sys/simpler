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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

#include "native_run_execution.h"
#include "runtime_c_api.h"

/**
 * One launched run's completion boundary on the two streams it submits to.
 *
 * A stream query answers "is this queue drained", which is a fact about the
 * queue rather than about a run. Once a successor is queued behind a
 * predecessor, that answer covers the successor too, so a run's own completion
 * needs a boundary of its own: an event recorded on each stream immediately
 * after that stream's kernel. Stream order then makes the event complete only
 * after the kernel it follows has exited, and a run is device-complete when
 * both of its recorded boundaries complete. Neither one alone establishes it —
 * the two kernels handshake, so either may still be running while the other has
 * exited — and a device-side handshake flag does not establish it either, since
 * it is published before the kernel returns.
 *
 * Event handles are runner-owned and live for one pipeline slot, like the
 * slot's `SlotPersistentArgs` device blocks: an event created with the
 * reuse-capable flag re-records without a reset, so creating and destroying a
 * pair per run would add device calls to every dispatch for no gain. The facts
 * — which kernels were submitted, which boundaries were recorded, which have
 * completed — belong to one run, so every read and mutation is keyed on that
 * run's `NativeRunIdentity`. A stale poller and a later run reusing the slot
 * therefore both fail the identity check rather than observing a boundary that
 * is not theirs.
 *
 * Submission and completion are separate facts because a record can fail after
 * its kernel is already submitted. Such a run is `Unfenced`: it holds device
 * work its own events cannot decide, and appending another event proves
 * nothing, because AICore may already be waiting on an AICPU kernel whose
 * submission failed. The caller owes it an independent bounded quiescence
 * proof.
 *
 * Threading: drain retires while a progress thread may be polling, so the state
 * is mutex-guarded, and `poll` try-locks so that a query never waits behind the
 * bounded wait drain holds. Device operations are injected so the whole state
 * machine is exercisable without a device.
 */
class RunCompletionFence {
public:
    /** Which of a run's two streams a boundary belongs to. */
    enum class StreamRole : size_t { Aicore = 0, Aicpu = 1 };
    static constexpr size_t kRoleCount = 2;

    enum class Completion {
        Pending,
        Complete,
        Error,
        /** Submitted device work that no recorded boundary covers. */
        Unfenced,
    };

    /**
     * Predecessor-boundary reference held by a successor that queued a stream
     * wait on it.
     *
     * `boundary` names the predecessor event waited on; `waiter` names the
     * successor stream the wait was queued into, which is the stream whose own
     * completion later covers it. The two are distinct on purpose: the join
     * crosses the streams, so a proof about one stream cannot retire a wait
     * sitting in the other.
     *
     * The token *is* the count it holds, so it moves and never copies: two
     * tokens naming one reference would each be releasable, and the second
     * release would consume some other live wait's count. A move empties the
     * source, and move assignment is deleted rather than allowed to overwrite —
     * and so drop — a live destination token.
     *
     * It also remembers which fence and which run it was minted against.
     * Neither is decoration: the counters are per boundary role, so a token
     * offered to the wrong fence, or to the right fence after it re-armed for a
     * later run, would otherwise decrement whichever counter happened to match
     * its two roles.
     */
    class WaitReference {
    public:
        WaitReference() = default;
        WaitReference(const WaitReference &) = delete;
        WaitReference &operator=(const WaitReference &) = delete;
        WaitReference &operator=(WaitReference &&) = delete;

        WaitReference(WaitReference &&other) noexcept :
            fence_(other.fence_),
            identity_(other.identity_),
            boundary_(other.boundary_),
            waiter_(other.waiter_),
            state_(other.state_) {
            other.clear();
        }

        bool valid() const { return state_ != State::Empty; }
        bool committed() const { return state_ == State::Committed; }
        StreamRole boundary() const { return boundary_; }
        StreamRole waiter() const { return waiter_; }

    private:
        friend class RunCompletionFence;
        enum class State : uint8_t { Empty, Reserved, Committed };

        void clear() {
            fence_ = nullptr;
            identity_ = NativeRunIdentity{};
            boundary_ = StreamRole::Aicore;
            waiter_ = StreamRole::Aicore;
            state_ = State::Empty;
        }

        const RunCompletionFence *fence_{nullptr};
        NativeRunIdentity identity_{};
        StreamRole boundary_{StreamRole::Aicore};
        StreamRole waiter_{StreamRole::Aicore};
        State state_{State::Empty};
    };

    using CreateEventFn = std::function<int(void **out_event)>;
    using RecordEventFn = std::function<int(void *event, void *stream)>;
    using QueryEventFn = std::function<int(void *event, bool *complete)>;
    using WaitEventFn = std::function<int(void *event, int timeout_ms)>;
    using DestroyEventFn = std::function<int(void *event)>;

    struct DeviceEventOps {
        CreateEventFn create;
        RecordEventFn record;
        QueryEventFn query;
        WaitEventFn wait;
        DestroyEventFn destroy;
    };

    explicit RunCompletionFence(DeviceEventOps ops) :
        ops_(std::move(ops)) {}
    RunCompletionFence(const RunCompletionFence &) = delete;
    RunCompletionFence &operator=(const RunCompletionFence &) = delete;

    /**
     * Take the fence for one run, committing both events on first use.
     *
     * Creation happens here, ahead of the caller's first device-visible
     * submission, so a run that cannot be fenced fails while it is still
     * rollback-able. A previous run's facts are dropped; outstanding wait
     * references are not droppable, so an armed-over fence that still holds
     * one is refused and its slot stops admitting runs.
     */
    int arm(const NativeRunIdentity &identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (references_outstanding()) return PTO_RUNTIME_ERR_INTERNAL;
        for (Boundary &boundary : boundaries_) {
            if (boundary.event != nullptr) continue;
            void *event = nullptr;
            int rc = ops_.create(&event);
            if (rc != 0) return rc;
            boundary.event = event;
        }
        for (Boundary &boundary : boundaries_) {
            boundary.reset_run_facts();
        }
        identity_ = identity;
        armed_ = true;
        return 0;
    }

    /**
     * Record that one stream's kernel reached its device queue.
     *
     * Separate from `record` so the caller states it the instant the
     * submission is accepted, before anything that can still fail.
     */
    void note_kernel_submitted(const NativeRunIdentity &identity, StreamRole role) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return;
        boundary(role).kernel_submitted = true;
    }

    /** Record this run's boundary on one stream, after that stream's kernel. */
    int record(const NativeRunIdentity &identity, StreamRole role, void *stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity) || stream == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        Boundary &target = boundary(role);
        if (target.event == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        // A boundary another run's join still waits on cannot be re-recorded:
        // its wait would then be satisfied by this run's kernel instead.
        if (target.committed_waits_total() > 0 || target.pending_reservations > 0) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        int rc = ops_.record(target.event, stream);
        if (rc != 0) {
            if (target.error == 0) target.error = rc;
            return rc;
        }
        target.recorded = true;
        target.complete = false;
        return 0;
    }

    /**
     * Whether this run's completion is decidable from its own boundaries: both
     * kernels submitted and both boundaries recorded.
     */
    bool fenced(const NativeRunIdentity &identity) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owns(identity) && fully_fenced();
    }

    /** Whether any device operation still names this run's resources. */
    bool has_device_references(const NativeRunIdentity &identity) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return false;
        for (const Boundary &target : boundaries_) {
            if (target.kernel_submitted || target.committed_waits_total() > 0) return true;
        }
        return false;
    }

    /**
     * Query both boundaries without waiting.
     *
     * Reports `Pending` rather than waiting behind the run's owner: drain holds
     * the state for the whole of its bounded wait, and a progress thread that
     * blocked there would stall for that timeout — on a2a3 while also holding
     * the stream pair, whose retirement would then block too. "Ask again" is
     * the correct answer for a non-blocking query, and it is what
     * `RunStreamPair::poll` answers in the same situation.
     */
    Completion poll(const NativeRunIdentity &identity) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return Completion::Pending;
        if (!owns(identity)) return Completion::Error;
        if (first_error() != 0) return Completion::Error;
        if (!fully_fenced()) return Completion::Unfenced;
        for (Boundary &target : boundaries_) {
            if (target.complete) continue;
            bool complete = false;
            int rc = ops_.query(target.event, &complete);
            if (rc != 0) {
                if (target.error == 0) target.error = rc;
                return Completion::Error;
            }
            target.complete = complete;
        }
        return fully_complete() ? Completion::Complete : Completion::Pending;
    }

    /**
     * Wait for both boundaries with a bounded per-boundary timeout.
     *
     * Reports the first device error, so the caller's timeout and recovery
     * policy reads the same rc it reads from a stream wait. A fence that cannot
     * decide the run returns `PTO_RUNTIME_ERR_INTERNAL` rather than a false
     * completion — the caller owes an `Unfenced` run its own bounded proof.
     */
    int wait(const NativeRunIdentity &identity, int timeout_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity) || !fully_fenced()) return PTO_RUNTIME_ERR_INTERNAL;
        // AICPU first, mirroring the whole-stream wait this replaces, so an
        // op-timeout keeps surfacing on the same side.
        for (StreamRole role : {StreamRole::Aicpu, StreamRole::Aicore}) {
            Boundary &target = boundary(role);
            if (target.complete) continue;
            int rc = ops_.wait(target.event, timeout_ms);
            if (rc != 0) {
                if (target.error == 0) target.error = rc;
                return rc;
            }
            target.complete = true;
        }
        return 0;
    }

    /**
     * Query one recorded boundary of one run, without waiting and without
     * touching that run's own completion facts.
     *
     * Separate from `poll` because this answers a question *about* a run asked
     * by someone else — a successor deciding whether there is anything left to
     * be ordered behind — rather than deciding the run. It caches nothing for
     * the same reason: the observation belongs to the asker, and the owner's
     * `complete` flags stay the record of what the owner itself observed.
     *
     * Refuses a run this fence no longer owns and a boundary that was never
     * recorded, which is what makes a true answer attributable to that exact
     * run's boundary rather than to a slot that has moved on.
     */
    int query_boundary(const NativeRunIdentity &identity, StreamRole role, bool *complete) const {
        if (complete == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        *complete = false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return PTO_RUNTIME_ERR_INTERNAL;
        const Boundary &target = boundary(role);
        if (!target.recorded || target.event == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        if (target.complete) {
            *complete = true;
            return 0;
        }
        return ops_.query(target.event, complete);
    }

    /**
     * Give up this run's arming so the slot can serve its next run.
     *
     * Idempotent, and a no-op for a caller that does not own the arming, so
     * every path that can end a run may call it. A wait reference cannot be
     * abandoned by retiring, so a fence still holding one stays armed and its
     * slot stops admitting runs until the reference is released.
     */
    int retire(const NativeRunIdentity &identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return 0;
        if (references_outstanding()) return PTO_RUNTIME_ERR_INTERNAL;
        armed_ = false;
        identity_ = NativeRunIdentity{};
        return 0;
    }

    /**
     * Destroy both events, keeping a handle whose destroy failed.
     *
     * Refuses while any reservation or committed reference is outstanding, for
     * the reason `arm`, `record` and `retire` do: destroying an event a queued
     * wait still names is the hazard the whole reference model exists to
     * prevent, and it is the last place that could still happen. The counters
     * and handles are left untouched so a caller that has since released can
     * retry. Teardown after a verified reset takes `abandon()` instead, which
     * invalidates the whole generation in one step rather than per handle.
     */
    int release() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (references_outstanding()) return PTO_RUNTIME_ERR_INTERNAL;
        int first_rc = 0;
        for (Boundary &target : boundaries_) {
            if (target.event == nullptr) continue;
            int rc = ops_.destroy(target.event);
            if (rc != 0) {
                if (first_rc == 0) first_rc = rc;
                continue;
            }
            target = Boundary{};
        }
        armed_ = false;
        identity_ = NativeRunIdentity{};
        return first_rc;
    }

    /**
     * Forget both handles after a device reset without invoking destroy.
     *
     * Unlike `release()` this drops outstanding references too, because a
     * verified reset invalidates every device reference to this generation at
     * once — which is the one proof that covers a queued wait without the
     * stream it sits in ever completing. Tokens still held against that
     * generation now name counters that are gone;
     * `discard_stale_wait_reference` is how their holders drop them.
     */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (Boundary &target : boundaries_) {
            target = Boundary{};
        }
        armed_ = false;
        identity_ = NativeRunIdentity{};
    }

    // ---- Cross-run join contract -----------------------------------------
    //
    // A successor orders itself after this run by queueing, on each of its own
    // streams, a wait for the opposite predecessor boundary:
    //
    //     successor AICore stream waits predecessor cpu_done
    //     successor AICPU  stream waits predecessor core_done
    //
    // Together with each stream's own order, those two edges put both
    // predecessor kernels ahead of either successor kernel. The reference
    // protocol below is what keeps the predecessor's events alive for exactly
    // as long as those queued waits can name them. Production insertion of the
    // waits, and admitting a second launched run, belong to the change that
    // opens admission.
    //
    // A reference is a count, and every guard below exists because there is
    // exactly one way to get a count wrong: decrement one that is not yours.
    // The token therefore moves rather than copies (no second releaser), and
    // carries the fence and the arming it was minted against, which every
    // mutation re-checks (no other fence's counter, no later run's). The only
    // thing that may drop a count without a per-stream proof is a verified
    // reset, and it does so wholesale through `abandon()`.

    /**
     * Reserve a reference to one recorded boundary before queueing a wait on
     * it. Reserving ahead of the wait is what makes the failure path safe: a
     * wait that was never queued is revoked, and one that was is committed.
     */
    int reserve_wait_reference(
        const NativeRunIdentity &identity, StreamRole boundary_role, StreamRole waiter_role, WaitReference *out
    ) {
        if (out == nullptr || out->valid()) return PTO_RUNTIME_ERR_INTERNAL;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return PTO_RUNTIME_ERR_INTERNAL;
        Boundary &target = boundary(boundary_role);
        if (!target.recorded) return PTO_RUNTIME_ERR_INTERNAL;
        ++target.pending_reservations;
        out->fence_ = this;
        out->identity_ = identity;
        out->boundary_ = boundary_role;
        out->waiter_ = waiter_role;
        out->state_ = WaitReference::State::Reserved;
        return 0;
    }

    /**
     * The device event a reserved reference names, for the caller to queue its
     * stream wait on. Null unless this run owns a recorded boundary there.
     */
    void *boundary_event(const NativeRunIdentity &identity, StreamRole role) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(identity)) return nullptr;
        const Boundary &target = boundary(role);
        return target.recorded ? target.event : nullptr;
    }

    /** Promote a reservation whose stream wait was queued successfully. */
    int commit_wait_reference(WaitReference &ref) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref.state_ != WaitReference::State::Reserved || !recognises(ref)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        Boundary &target = boundary(ref.boundary_);
        --target.pending_reservations;
        ++target.committed_waits[static_cast<size_t>(ref.waiter_)];
        ref.state_ = WaitReference::State::Committed;
        return 0;
    }

    /** Drop a reservation whose stream wait was never queued. */
    int revoke_wait_reference(WaitReference &ref) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref.state_ != WaitReference::State::Reserved || !recognises(ref)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        --boundary(ref.boundary_).pending_reservations;
        ref.clear();
        return 0;
    }

    /**
     * Release a committed reference against the completion of the successor
     * stream that holds its wait.
     *
     * `proven_waiter` must be that stream: a proof about the other one leaves
     * the wait uncovered, so it is refused rather than silently accepted.
     */
    int release_wait_reference(WaitReference &ref, StreamRole proven_waiter) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref.state_ != WaitReference::State::Committed || !recognises(ref)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (proven_waiter != ref.waiter_) return PTO_RUNTIME_ERR_INTERNAL;
        return drop_committed_reference(ref);
    }

    /**
     * Release a committed reference against a proof that covers the device as
     * a whole — a verified reset or quarantine that invalidated every
     * reference to this generation's events.
     *
     * Still bound to the arming that minted the token: quiescence licenses
     * releasing a reference without a per-stream proof, not charging one run's
     * release to a later run's counters. A token whose generation is gone is
     * `discard_stale_wait_reference`'s business.
     */
    int release_wait_reference_on_quiescence(WaitReference &ref) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ref.state_ != WaitReference::State::Committed || !recognises(ref)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        return drop_committed_reference(ref);
    }

    /**
     * Empty a token whose generation on *this* fence is gone, holding every
     * counter still.
     *
     * `abandon()` drops the counts of an invalidated device generation in one
     * step, which leaves whoever held tokens against it holding names for
     * counters that are gone. This is how those are dropped.
     *
     * Only the fence that minted a token may judge it stale, and only two
     * answers are refusals rather than one. A token this fence still recognises
     * is current, so emptying it would leak the count it holds — release it
     * instead. And a token *another* fence minted is not stale here either: it
     * is simply not this fence's to judge, and emptying it would strand the
     * count it still holds over there, leaving that fence blocked from retiring
     * with no token left to release it. "Not mine" and "no longer live" are
     * different questions; conflating them is how a foreign fence deletes a
     * live reference.
     */
    int discard_stale_wait_reference(WaitReference &ref) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ref.valid()) return PTO_RUNTIME_ERR_INTERNAL;
        if (ref.fence_ != this) return PTO_RUNTIME_ERR_INTERNAL;
        if (owns(ref.identity_)) return PTO_RUNTIME_ERR_INTERNAL;
        ref.clear();
        return 0;
    }

    /** Committed references naming one boundary, summed over waiter streams. */
    unsigned committed_wait_count(StreamRole boundary_role) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return boundary(boundary_role).committed_waits_total();
    }

    bool armed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return armed_;
    }
    bool kernel_submitted(const NativeRunIdentity &identity, StreamRole role) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owns(identity) && boundary(role).kernel_submitted;
    }
    bool boundary_recorded(const NativeRunIdentity &identity, StreamRole role) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owns(identity) && boundary(role).recorded;
    }
    int boundary_error(const NativeRunIdentity &identity, StreamRole role) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owns(identity) ? boundary(role).error : 0;
    }

private:
    struct Boundary {
        void *event{nullptr};
        bool kernel_submitted{false};
        bool recorded{false};
        bool complete{false};
        // First query / wait / record failure seen on this boundary.
        int error{0};
        unsigned pending_reservations{0};
        // Committed references, indexed by the successor stream holding the wait.
        std::array<unsigned, kRoleCount> committed_waits{};

        /** Drop the previous run's facts, keeping the event and its references. */
        void reset_run_facts() {
            kernel_submitted = false;
            recorded = false;
            complete = false;
            error = 0;
        }

        unsigned committed_waits_total() const {
            unsigned total = 0;
            for (unsigned count : committed_waits)
                total += count;
            return total;
        }
    };

    bool owns(const NativeRunIdentity &identity) const { return armed_ && identity_ == identity; }

    /**
     * Whether this token names a reference on *this* fence's current arming.
     *
     * Both halves matter. The counters are per boundary role, so a token minted
     * on another fence would decrement whichever of this fence's counters
     * happened to share its roles; and a token minted before this fence
     * re-armed names a generation whose counts are gone, so charging its
     * release here would consume the current run's.
     */
    bool recognises(const WaitReference &ref) const { return ref.fence_ == this && owns(ref.identity_); }

    Boundary &boundary(StreamRole role) { return boundaries_[static_cast<size_t>(role)]; }
    const Boundary &boundary(StreamRole role) const { return boundaries_[static_cast<size_t>(role)]; }

    bool fully_fenced() const {
        for (const Boundary &target : boundaries_) {
            if (!target.kernel_submitted || !target.recorded) return false;
        }
        return true;
    }

    bool fully_complete() const {
        for (const Boundary &target : boundaries_) {
            if (!target.complete) return false;
        }
        return true;
    }

    int first_error() const {
        for (const Boundary &target : boundaries_) {
            if (target.error != 0) return target.error;
        }
        return 0;
    }

    bool references_outstanding() const {
        for (const Boundary &target : boundaries_) {
            if (target.pending_reservations > 0 || target.committed_waits_total() > 0) return true;
        }
        return false;
    }

    /**
     * Consume one committed reference. Callers establish that this fence
     * recognises the token (`recognises`) before reaching here — that check is
     * what makes the counter this decrements the token's own.
     */
    int drop_committed_reference(WaitReference &ref) {
        unsigned &count = boundary(ref.boundary_).committed_waits[static_cast<size_t>(ref.waiter_)];
        if (count == 0) return PTO_RUNTIME_ERR_INTERNAL;
        --count;
        ref.clear();
        return 0;
    }

    mutable std::mutex mutex_;
    std::array<Boundary, kRoleCount> boundaries_{};
    NativeRunIdentity identity_{};
    bool armed_{false};
    DeviceEventOps ops_;
};
