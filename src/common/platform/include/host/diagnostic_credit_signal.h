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
 * @file diagnostic_credit_signal.h
 * @brief The one place a caller learns that diagnostic credit moved.
 *
 * A run that cannot be admitted because every output slot or terminal bank of
 * some owner is taken has nothing to poll: the capacity comes back on a
 * collector's own writer thread, at a moment no caller can predict. This is
 * what that writer raises and what the caller blocks on.
 *
 * The value is deliberately a bare generation rather than a per-owner state.
 * A waiter does not need to know which owner released: it re-attempts the
 * whole all-or-nothing admission, which is the only thing that can tell it
 * whether it may now launch. That keeps the signal free of any collector type
 * and lets it cross the C boundary as an integer.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace simpler::dfx::credit {

/** What a bounded credit wait observed. Mirrors the C entry's three results. */
enum class CreditWait : int {
    Cancelled = -1,
    TimedOut = 0,
    Advanced = 1,
};

/**
 * One device context's credit progress counter, its waiters, and its
 * cancellation latch.
 *
 * **Lock order.** A release path takes this mutex *after* dropping its own
 * collector lock, so a notify never runs under one. Nothing here ever takes a
 * collector lock or a caller's lane mutex, so this is a leaf.
 */
class DiagnosticCreditSignal {
public:
    /**
     * Read the current generation.
     *
     * Taken before an admission attempt and handed back to `wait_for`, which
     * is what closes the lost-wakeup window: the attempt takes only collector
     * locks, so a release that lands between the read and the wait has already
     * advanced the value the waiter will compare against.
     */
    uint64_t generation() const {
        std::lock_guard<std::mutex> lk(mu_);
        return generation_;
    }

    /**
     * Suppress this thread's own notifications for as long as it is held.
     *
     * One case needs it, and only one: an all-or-nothing admission that takes
     * some owners' capacity and gives it back when a later owner reports full.
     * That rollback restores exactly the state every waiter already saw before
     * the attempt, so it is progress for nobody — but it runs through the same
     * release paths as a real publication, and notifying there would wake the
     * attempting thread onto its own generation bump. Repeating the attempt
     * then manufactures its own wake every round, spinning through the shard
     * acknowledgement work until the deadline while the blocking owner has
     * freed nothing.
     *
     * Thread-local because the thread undoing an attempt is the thread that
     * made it. A genuine release on any other thread still advances past the
     * attempt's pre-attempt snapshot, so the lost-wakeup argument is untouched:
     * what is suppressed is not a release, it is the absence of one.
     *
     * **It covers a restoration and nothing else.** A withdrawal can fail to
     * restore — an owner that could not prove its references released
     * quarantines the run, and a fatal one defers its storage to teardown —
     * and then the capacity every waiter last observed is gone for good, which
     * is an event and not the absence of one. So the only caller is
     * `undo_admission_attempt`, which holds this across the withdrawal and
     * publishes an advance itself when the withdrawal reports that some
     * owner's slot did not come back. Holding it across any wider span would
     * drop that owner's fatal, and the waiter would park to its deadline
     * instead of re-attempting and being told.
     *
     * Deliberately *not* used by the post-launch withdrawal of a run that
     * submitted nothing. That run held its admission while other waiters
     * observed it taken, so giving it back is real progress and must be seen.
     */
    class SuppressSelfNotify {
    public:
        SuppressSelfNotify() { depth()++; }
        ~SuppressSelfNotify() { depth()--; }
        SuppressSelfNotify(const SuppressSelfNotify &) = delete;
        SuppressSelfNotify &operator=(const SuppressSelfNotify &) = delete;

        static bool active() { return depth() != 0; }

    private:
        static unsigned &depth() {
            static thread_local unsigned value = 0;
            return value;
        }
    };

    /**
     * Record that credit moved, and wake every waiter.
     *
     * Called on release of an output slot or a terminal bank, on an owner
     * becoming fatal, and at teardown. Over-notifying is free — a woken
     * waiter re-attempts the admission and parks again — while under-notifying
     * costs a run its whole deadline, so every path that could free capacity
     * calls this rather than reasoning about whether it had to. The one
     * exception is `SuppressSelfNotify` above, which covers a release that did
     * not happen rather than one that did, and covers only that.
     */
    void advance() {
        if (SuppressSelfNotify::active()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            generation_++;
        }
        cv_.notify_all();
    }

    /**
     * Block until the generation differs from `seen`, the budget elapses, or
     * waits are cancelled.
     *
     * A non-positive budget is a poll: the predicate is evaluated once. The
     * cancelled latch is sticky, so a wait entered after cancellation returns
     * immediately rather than parking against a signal nobody will raise
     * again.
     */
    CreditWait wait_for(uint64_t seen, int32_t timeout_ms) {
        std::unique_lock<std::mutex> lk(mu_);
        if (cancelled_) return CreditWait::Cancelled;
        if (generation_ != seen) return CreditWait::Advanced;
        if (timeout_ms <= 0) return CreditWait::TimedOut;
        const bool woke = cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this, seen] {
            return cancelled_ || generation_ != seen;
        });
        if (!woke) return CreditWait::TimedOut;
        return cancelled_ ? CreditWait::Cancelled : CreditWait::Advanced;
    }

    /**
     * Make every present and future wait return `Cancelled`.
     *
     * Non-blocking and idempotent, and it proves nothing about a waiter having
     * left — that is the caller's to establish. Being non-blocking is what
     * lets a thread that is itself inside a guarded call invoke it without
     * waiting on its own lease.
     */
    void cancel() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            cancelled_ = true;
            generation_++;
        }
        cv_.notify_all();
    }

    /**
     * True once `cancel()` has run. Sticky for this signal's whole life.
     *
     * Which is one device context's: `ChipWorker::finalize` destroys the
     * context and nulls its handle, so a later `initialize` builds a new
     * runner and therefore a new signal. A reinitialized worker does not
     * inherit a cancellation from the lifetime before it.
     */
    bool cancelled() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cancelled_;
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t generation_{1};
    bool cancelled_{false};
};

}  // namespace simpler::dfx::credit
