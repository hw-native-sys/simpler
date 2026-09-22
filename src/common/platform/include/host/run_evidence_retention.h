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
 * @file run_evidence_retention.h
 * @brief What each run's own drain, poll and record read observed, kept for a
 *        reader that arrives after the run's resources are gone.
 *
 * Both channels are observed once and asked about later, so both are retained
 * at the moment of observation rather than re-derived:
 *
 *  - The fence cannot be asked after the fact. A run's drain retires the
 *    fence's arming during cleanup (`retire_run_fence`), and from then on
 *    `poll` no longer owns that identity and answers `Error` without asking
 *    the device anything — a statement about the fence, not about the run.
 *  - The record region is read once per run, and the bytes it leaves behind
 *    cannot say whether a copy was attempted. A run whose slot holds no region
 *    attempted nothing; a run whose D2H failed attempted and lost. Both leave
 *    the region empty, so only the read's own bookkeeping separates them.
 *
 * The observation and its retention live in the same call — `fence.wait` and
 * the ledger write are not two steps a caller can perform independently — so a
 * caller cannot obtain one of these observations without keeping it.
 *
 * Header-only and device-free: every device operation is the caller's, injected
 * the same way `RunCompletionFence` injects its event operations, so the
 * retention rules are exercisable without hardware.
 *
 * Threading: a run's observations are written and read by whoever drives that
 * run's phase functions, and `runtime_c_api.h` requires the caller to serialize
 * those for a given context/storage pair — poll, wait and finalize are not
 * concurrent operations on the same run. So these hold no synchronization of
 * their own.
 */

#pragma once

#include <array>
#include <cstdint>

#include "host/run_completion_fence.h"
#include "host/run_outcome_decision.h"
#include "runtime_c_api.h"
#include "worker/native_run_execution.h"

/** One retained boundary observation per pipeline slot. */
template <size_t Slots>
class RunBoundaryLedgerT {
public:
    /** Retain what this run's drain or poll just observed. */
    void note(const NativeRunIdentity &identity, RunCompletionFence::Completion completion) {
        if (identity.pipeline_slot >= Slots) return;
        Entry &entry = entries_[identity.pipeline_slot];
        entry.identity = identity;
        entry.completion = completion;
        entry.observed = true;
    }

    /**
     * What this run observed, or `Pending` when this run observed nothing.
     *
     * Pending rather than a distinct "never observed": a run with no terminal
     * boundary observation has, as far as any reader can tell, not reached one
     * — which is what Pending already means to `decide_run_execution`.
     *
     * Keying on the full `NativeRunIdentity` is what stops a successor reusing
     * the slot from inheriting its predecessor's answer: a slot holds one
     * observation, and only the run that made it can read it back.
     */
    RunCompletionFence::Completion observed(const NativeRunIdentity &identity) const {
        if (identity.pipeline_slot >= Slots) return RunCompletionFence::Completion::Pending;
        const Entry &entry = entries_[identity.pipeline_slot];
        if (!entry.observed || entry.identity != identity) return RunCompletionFence::Completion::Pending;
        return entry.completion;
    }

private:
    struct Entry {
        NativeRunIdentity identity{};
        RunCompletionFence::Completion completion{RunCompletionFence::Completion::Pending};
        bool observed{false};
    };
    std::array<Entry, Slots> entries_{};
};

/** What a drain's bounded wait observed, and the rc the caller still owes. */
struct RunBoundaryWait {
    RunCompletionFence::Completion completion{RunCompletionFence::Completion::Pending};
    /** The fence's own rc. Zero for an `Unfenced` run, which the fence does not decide. */
    int rc{0};
};

/**
 * Drain this run's boundaries and retain what that observed.
 *
 * A run whose boundaries are not both recorded is `Unfenced`: the fence cannot
 * decide it and produces no rc, and the caller owes it an independent bounded
 * quiescence proof. The observation is retained before that fallback runs, so
 * it survives whatever the fallback then does to the streams.
 *
 * A fenced run's completion is the wait's own rc: zero is both boundaries
 * complete, and any other value is a boundary the fence could not decide.
 */
template <size_t Slots>
RunBoundaryWait wait_and_retain_run_boundaries(
    RunCompletionFence &fence, RunBoundaryLedgerT<Slots> &ledger, const NativeRunIdentity &identity, int timeout_ms
) {
    if (!fence.fenced(identity)) {
        ledger.note(identity, RunCompletionFence::Completion::Unfenced);
        return {RunCompletionFence::Completion::Unfenced, 0};
    }
    const int rc = fence.wait(identity, timeout_ms);
    const RunCompletionFence::Completion completion =
        rc == 0 ? RunCompletionFence::Completion::Complete : RunCompletionFence::Completion::Error;
    ledger.note(identity, completion);
    return {completion, rc};
}

/** Query this run's boundaries without waiting, and retain what that observed. */
template <size_t Slots>
RunCompletionFence::Completion poll_and_retain_run_boundaries(
    RunCompletionFence &fence, RunBoundaryLedgerT<Slots> &ledger, const NativeRunIdentity &identity
) {
    const RunCompletionFence::Completion completion = fence.poll(identity);
    ledger.note(identity, completion);
    return completion;
}

/**
 * One read of the result region per run, which of the three read states that
 * run's slot is in, and the status the transfer itself reported.
 */
template <size_t Slots>
class RunRecordReadLedgerT {
public:
    /**
     * Take this run's one read of `region` into `out`, retaining the
     * transfer's own status.
     *
     * `copy_in(void *dst, const void *src)` performs the copy and returns the
     * transfer's status code: zero when the bytes landed, otherwise the code
     * the transfer itself reported. It is invoked exactly when this run has
     * not read yet and a region exists, so an absent region costs no copy and
     * is not reported as one that failed: a run that never launched has no
     * region and did not lose a read. `out` is emptied whenever a read is
     * taken, so a run never reads a predecessor's bytes, and emptied again
     * when the copy fails, so it never reads a partial one.
     *
     * Returns the status retained for this run, which is the copy's code on
     * the read this run takes and the same code on every later call for it.
     * A caller that acts on the status therefore reads the same value however
     * many times it asks, and a deduplicated call is not mistaken for a
     * transfer that succeeded.
     *
     * One read per run: later consumers share the first read's bytes. A retry
     * would either cost a second D2H for the same answer or, after a device
     * recovery, sample a generation this run never wrote. A run with no epoch
     * is not a run that can own a read, so it is never deduplicated against.
     */
    template <typename Region, typename CopyIn>
    int read_with_status(uint32_t slot, uint64_t run_epoch, Region &out, const void *region, const CopyIn &copy_in) {
        if (slot >= Slots) return 0;
        if (run_epoch != 0 && epochs_[slot] == run_epoch) return transfer_statuses_[slot];
        epochs_[slot] = run_epoch;
        states_[slot] = RunRecordRead::NotAttempted;
        transfer_statuses_[slot] = 0;
        out = Region{};
        if (region == nullptr) return 0;
        const int status = copy_in(static_cast<void *>(&out), region);
        if (status != 0) {
            out = Region{};
            states_[slot] = RunRecordRead::Failed;
            transfer_statuses_[slot] = status;
            return status;
        }
        states_[slot] = RunRecordRead::Ok;
        return 0;
    }

    /**
     * Take this run's one read from a copy that reports only success.
     *
     * A caller with no status code to retain reports a failed copy as
     * `PTO_RUNTIME_ERR_INTERNAL`, which is the same read state as the
     * status-carrying form and carries no claim about what the transport said.
     */
    template <typename Region, typename CopyIn>
    void read(uint32_t slot, uint64_t run_epoch, Region &out, const void *region, const CopyIn &copy_in) {
        (void)read_with_status(slot, run_epoch, out, region, [&copy_in](void *dst, const void *src) {
            return copy_in(dst, src) ? 0 : PTO_RUNTIME_ERR_INTERNAL;
        });
    }

    /**
     * Which read state this slot holds for `run_epoch`.
     *
     * A read taken for another run is not a read of this one, and a run with no
     * epoch owns no read at all — both are `NotAttempted` rather than any
     * verdict about the bytes.
     */
    RunRecordRead state(uint32_t slot, uint64_t run_epoch) const {
        if (slot >= Slots || run_epoch == 0 || epochs_[slot] != run_epoch) return RunRecordRead::NotAttempted;
        return states_[slot];
    }

    /**
     * The status this slot's transfer reported for `run_epoch`.
     *
     * Zero for a run whose copy landed and for a run that owns no read here,
     * so a caller must pair a non-zero answer with `state()` rather than read
     * zero as evidence that a transfer happened.
     */
    int transfer_status(uint32_t slot, uint64_t run_epoch) const {
        if (slot >= Slots || run_epoch == 0 || epochs_[slot] != run_epoch) return 0;
        return transfer_statuses_[slot];
    }

    /** Forget every slot's read, for a runner starting a fresh device generation. */
    void reset() {
        epochs_.fill(0);
        states_.fill(RunRecordRead::NotAttempted);
        transfer_statuses_.fill(0);
    }

private:
    std::array<uint64_t, Slots> epochs_{};
    std::array<RunRecordRead, Slots> states_{};
    std::array<int, Slots> transfer_statuses_{};
};
