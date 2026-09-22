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

#include <atomic>
#include <cstdint>

/**
 * What the device-fault notification channel says about one device.
 *
 * About a **device**, never about a run. Measured on a2a3 in #2303: a notice
 * carries `task_id == 0` for every error class, a pipelined pair shares one
 * stream pair so `stream_id` names the faulting side but not the run, and the
 * longest arrival lag measured is 16 s — wider than the interval between two
 * runs, and a sample rather than a bound the SDK promises. So a notice can
 * arrive while a later, healthy run is finalizing. That run's verdict is
 * untouched; the device is still suspect, because it did fault.
 *
 * Notices are process-wide — the ring reserves no share per device — while a
 * quarantine is per device. The notice's own `device_id` is what resolves that,
 * and it is in the **logical** space, the same one the runner names its device
 * in. Measured: card 5 bound as logical 0 through `ASCEND_RT_VISIBLE_DEVICES`
 * reports `device_id=0`, so the comparison is direct and must not translate
 * through `acl_to_hal_device_id`.
 *
 * Generations exist because a quarantine has to be escapable. Every notice
 * reported before a confirmed device reset described a device that no longer
 * exists, so the ring's history is not evidence about the generation that
 * replaced it — without that fence a single recovered fault would quarantine the
 * card for the life of the process.
 *
 * **Two lifetimes live here, and the names say which.** The generation-scoped
 * state — the suspicion and the code that named it — is what a confirmed reset
 * retires, because it is a claim about a device generation that no longer
 * exists. The `_total` counters are runner-lifetime: they answer "how much has
 * this channel reported on this card", which a count that reset on every
 * recovery could not. Admission reads the generation-scoped suspicion through
 * `device_admits_new_run`; nothing reads the totals.
 */
class DeviceHealthState {
public:
    /**
     * A notice naming this device. The device is suspect from here until a
     * confirmed reset retires the generation, and admission refuses new runs
     * while it is — see `device_admits_new_run`.
     *
     * Returns whether this call is what made it suspect, so a caller can log
     * and recover once per fault rather than once per notice.
     */
    bool note_own_device_fault(uint32_t error_code) {
        own_faults_.fetch_add(1, std::memory_order_acq_rel);
        // First, not last: a fault cascade's later codes are consequences of its
        // first, so the first is the one worth keeping. Same rule as the
        // per-run terminal record's.
        uint32_t expected = 0;
        first_error_code_.compare_exchange_strong(
            expected, error_code == 0 ? kUnspecifiedFault : error_code, std::memory_order_acq_rel,
            std::memory_order_relaxed
        );
        return !suspect_.exchange(true, std::memory_order_acq_rel);
    }

    /**
     * A notice this runner cannot attribute to its own runs — it names another
     * device in this process, or a stream none of this runner's runs submits on.
     *
     * Counted, and it refuses nothing. Measured on a2a3: a `507018` on
     * `stream_id=45/46` with `task_id=10` arrives while every run on the card
     * succeeds. The drain's synchronize never observed those streams either, so a
     * channel replacing it must not either. This is not a claim that such a fault
     * is harmless — it is that nothing here says which resources it touched.
     */
    void note_unattributed_fault() { unattributed_faults_.fetch_add(1, std::memory_order_acq_rel); }

    /**
     * Notices the channel could not deliver — overwritten in the ring, or
     * abandoned by the reporter.
     *
     * Counted, and it refuses nothing, which is a deliberate asymmetry with
     * `note_own_device_fault`: an undelivered notice names nothing, so it cannot
     * be told from one that would have named another device or a stream no run of
     * this runner's uses. Refusing work on it would refuse on evidence that
     * identifies no resource. What it does mean is that the channel is lossy right
     * now — so a suspicion this class does not hold is not evidence of health.
     */
    void note_undelivered_notices(uint64_t lost, uint64_t dropped) {
        if (lost == 0 && dropped == 0) return;
        undelivered_.fetch_add(lost + dropped, std::memory_order_acq_rel);
    }

    /**
     * A confirmed device reset. Retires the generation every notice so far
     * described, and reports whether it cleared a suspicion — which is what
     * distinguishes a recovery from a reset that had nothing to recover.
     *
     * Retires the **generation-scoped** state only: the suspicion and the code
     * that named it. The `_total` counters below are runner-lifetime on purpose
     * and survive — see the two-lifetime note on this class.
     *
     * Only a *confirmed* reset may call this. A reset that failed leaves the
     * generation in place, so the quarantine stands.
     */
    bool retire_generation() {
        generation_.fetch_add(1, std::memory_order_acq_rel);
        first_error_code_.store(0, std::memory_order_release);
        return suspect_.exchange(false, std::memory_order_acq_rel);
    }

    // ===== Generation-scoped: retired by a confirmed reset =====

    bool suspect() const { return suspect_.load(std::memory_order_acquire); }
    /** The first fault's code in the current generation, or 0 while healthy. */
    uint32_t first_error_code() const { return first_error_code_.load(std::memory_order_acquire); }
    /** How many generations this device has been through. Never retired. */
    uint64_t generation() const { return generation_.load(std::memory_order_acquire); }

    // ===== Runner-lifetime: diagnostics that deliberately outlive a reset =====
    //
    // Named `_total` so a reader cannot mistake them for the generation-scoped
    // state above. Nothing in the product reads them; they exist so a log or a
    // test can say how much this channel has reported over a runner's whole life,
    // and a count that reset on every recovery could not answer that.

    uint64_t own_faults_total() const { return own_faults_.load(std::memory_order_acquire); }
    uint64_t unattributed_faults_total() const { return unattributed_faults_.load(std::memory_order_acquire); }
    uint64_t undelivered_notices_total() const { return undelivered_.load(std::memory_order_acquire); }

    /** Stands in for a fault whose notice carried no code, so 0 keeps meaning "healthy". */
    static constexpr uint32_t kUnspecifiedFault = 0xFFFFFFFFu;

private:
    std::atomic<bool> suspect_{false};
    std::atomic<uint32_t> first_error_code_{0};
    std::atomic<uint64_t> generation_{0};
    std::atomic<uint64_t> own_faults_{0};
    std::atomic<uint64_t> unattributed_faults_{0};
    std::atomic<uint64_t> undelivered_{0};
};

/**
 * Whether a runner may take a new run.
 *
 * Two independent refusals, and they are not reducible to each other:
 *
 * - `arch_quarantined` is the arch's own post-failure flag, set when a launch or
 *   a sync failed and cleared by its finalize after a confirmed reset.
 * - a suspicion in `health` is a fault the channel matched to a stream this
 *   runner's runs were recorded on. It is generation-scoped, so the confirmed
 *   reset that retires the generation restores admission with no separate clear,
 *   and a reset that failed leaves the refusal standing.
 *
 * Neither is derived from any run's verdict: a successful run does not clear a
 * suspicion and a failed one does not create one. Refusal applies to *future*
 * admission only; the run whose finalize recorded the notice keeps its own
 * outcome, and refusing reclaims nothing and resets nothing.
 *
 * Only a notice this runner matched refuses. Unattributed, undecided, lost and
 * dropped notices leave admission open, and so does the absence of a notice —
 * which is not evidence of health, only of silence.
 */
inline bool device_admits_new_run(bool arch_quarantined, const DeviceHealthState &health) {
    return !arch_quarantined && !health.suspect();
}

/**
 * The driver ids of the streams that have carried this device's runs, for the
 * live generation.
 *
 * Recorded rather than queried, for two reasons that are both defects the other
 * shape has.
 *
 * **A handle is only safe to ask while a run is using it.** By the time notices
 * are consumed at teardown the device may already have been force-reset, and the
 * runner's handles are dead but not yet cleared — asking one for its id there is
 * a query against RTS resources the fatal path has deliberately isolated. An id
 * captured at boundary-record time is a number, and a number survives the reset
 * that invalidates the handle.
 *
 * **Streams are replaced, and their ids go with them.** Publishing AICore code
 * marks a2a3's AICore stream stale, and the next launch destroys and recreates
 * it. A notice naming the old stream arrives after that — the channel has lagged
 * by as much as the 16 s longest measured — so a filter that knows only the
 * *current* pair would read a run's own fault as someone else's and step the
 * cursor past it for good. Keeping the ids a run actually submitted on is what
 * makes that notice still attributable.
 *
 * History can be incomplete two ways — capacity refuses a further id, or a query
 * never yielded one. Both are reported rather than hidden: an id that matches
 * nothing is `NotMine` only while the history is whole, and `Undecided`
 * otherwise, because a missing id is exactly what a stale notice would carry.
 *
 * **Threading: one writer, many readers, serialized retirement.** `note` and
 * `note_unidentified_stream` run on the launch path, which holds the execution
 * claim, so exactly one thread inserts at a time. `attribute` and the observers
 * run on whichever thread finalizes a run, concurrently with that writer.
 * `retire_generation` runs on the confirmed-reset path, which no launch overlaps.
 *
 * The atomics are what make a concurrent *reader* safe; they are **not** a
 * multi-writer insert algorithm. Two threads inserting distinct ids could read
 * the same count and claim the same slot. Nothing in the product does that, so
 * the hot path takes no lock — and a future caller that needs concurrent inserts
 * owes a real algorithm rather than an assumption that this already is one.
 */
class RunStreamIdentities {
public:
    /** Enough for many stream replacements in one generation; a reset clears it. */
    static constexpr size_t kCapacity = 32;

    enum class Attribution {
        /** The notice names a stream this device's runs submitted on. */
        Mine,
        /** It does not, and the history that would say so is complete. */
        NotMine,
        /** It does not, but an id is missing — refused by capacity, or never obtained. */
        Undecided,
    };

    /**
     * Record `id` as one this device's runs submit on. Call where the handle is
     * known live — the boundary record at launch — never at teardown.
     *
     * Ignores a repeat. Once full it keeps the ids it has and marks the history
     * incomplete, rather than dropping an id silently. A negative id is treated as
     * unidentified, since that is what a failed query yields.
     */
    void note(int32_t id) {
        if (id < 0) {
            note_unidentified_stream();
            return;
        }
        const size_t live = count_.load(std::memory_order_acquire);
        for (size_t i = 0; i < live; ++i) {
            if (ids_[i].load(std::memory_order_acquire) == id) return;
        }
        if (live >= kCapacity) {
            complete_.store(false, std::memory_order_release);
            return;
        }
        ids_[live].store(id, std::memory_order_release);
        count_.store(live + 1, std::memory_order_release);
    }

    /**
     * A stream this device's runs submit on whose id could not be obtained.
     *
     * The history is incomplete from here, for the same reason a refused id makes
     * it incomplete: an id that is missing is exactly the one a later notice
     * might carry, so a non-match can no longer be read as "not mine". Silently
     * skipping a failed query would leave the history *claiming* to be whole
     * while missing the very entry that would decide the next notice.
     */
    void note_unidentified_stream() { complete_.store(false, std::memory_order_release); }

    Attribution attribute(uint32_t stream_id) const {
        const size_t live = count_.load(std::memory_order_acquire);
        for (size_t i = 0; i < live; ++i) {
            const int32_t id = ids_[i].load(std::memory_order_acquire);
            if (id >= 0 && static_cast<uint32_t>(id) == stream_id) return Attribution::Mine;
        }
        return complete_.load(std::memory_order_acquire) ? Attribution::NotMine : Attribution::Undecided;
    }

    /**
     * A confirmed reset. The ids described streams on a device generation that no
     * longer exists, and the next generation's streams get new ones.
     */
    void retire_generation() {
        count_.store(0, std::memory_order_release);
        complete_.store(true, std::memory_order_release);
    }

    /** Whether every id this generation used is still held. */
    bool complete() const { return complete_.load(std::memory_order_acquire); }
    size_t size() const { return count_.load(std::memory_order_acquire); }

private:
    std::atomic<int32_t> ids_[kCapacity]{};
    std::atomic<size_t> count_{0};
    std::atomic<bool> complete_{true};
};

/** What a confirmed device reset did to one runner's evidence. */
struct DeviceGenerationRetirement {
    /** The monitor's re-install rc, or 0 when no monitor is held. */
    int monitor_reinstall_rc{0};
    /** Whether a monitor was there to re-install at all. */
    bool monitor_reinstalled{false};
    /** Whether the retirement cleared a live suspicion, rather than finding none. */
    bool cleared_suspicion{false};
};

/**
 * Apply a **confirmed** device reset to one runner's local evidence, and to the
 * process fault monitor if this runner holds one.
 *
 * Two orderings are the contract, and the second is the one that was wrong.
 *
 * **The local generation retires unconditionally, before anything that depends
 * on a monitor.** Whether the driver callback was ever installed has nothing to
 * do with whether the device was reset: `ensure_device_initialized` does not
 * fail a runner whose `acquire` failed, and a run records its stream ids without
 * consulting the monitor — so a runner with no monitor still accumulates a
 * generation's worth of evidence, and a reset still invalidates it. Retiring
 * first also means no guard added above can silently skip it.
 *
 * **The monitor's own fence is conditional, and only on the monitor.** Its
 * re-install and the cursor skip both dereference it, so they run only when one
 * is held.
 *
 * Only a confirmed reset may call this. A reset that failed leaves the
 * generation — and so any suspicion in it — standing, which is what keeps this
 * from clearing evidence it did not earn.
 *
 * Templated on the monitor and cursor so the decision above is exercisable with
 * the real types and no device, which is the same reason `DeviceFaultMonitor`
 * injects its platform ops.
 */
template <typename Monitor, typename Cursor>
inline DeviceGenerationRetirement retire_after_confirmed_device_reset(
    DeviceHealthState &health, RunStreamIdentities &ids, Monitor *monitor, Cursor &notices
) {
    DeviceGenerationRetirement retirement;
    // Unconditional, and first.
    ids.retire_generation();
    retirement.cleared_suspicion = health.retire_generation();
    if (monitor == nullptr) return retirement;

    retirement.monitor_reinstalled = true;
    retirement.monitor_reinstall_rc = monitor->reinstall_after_device_reset();
    // Notices already in the ring described the retired generation, so they are
    // not this one's to read.
    notices.skip_to_current(*monitor);
    return retirement;
}
