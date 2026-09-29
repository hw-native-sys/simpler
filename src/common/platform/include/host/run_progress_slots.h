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
 * Where each run-resource set is in its run, and which run that is.
 *
 * One record per set, not one per runner: with two runs launched each needs its
 * own answer, and a runner-wide sticky state would let one run's drain report
 * the other complete without any boundary having been polled.
 *
 * **The precondition this relies on, stated because the code does not enforce
 * it.** A chip lane has one progress owner, so `claim`, `publish` and `state_of`
 * for one set are driven from one thread; and a set is handed to another run
 * only after the run holding it has finalized, which is what makes the previous
 * owner's calls finished rather than merely unlikely. The epoch is therefore an
 * *identity check against a retired owner*, not a lock: it rejects a call that
 * names a run the set no longer describes, and it does not — and cannot —
 * serialize an arbitrary writer racing a claim. `publish` is an epoch test and
 * a separate store; two concurrent publishers for different runs on one set
 * would still be a defect in the caller, not something this excludes.
 *
 * What the bracketing in `state_of` does buy, under that precondition, is that
 * a *reader* asking about one run is never told another run's progress: a set
 * being claimed for a new run answers `Idle` rather than the old run's terminal
 * state or the new run's later one.
 */

#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include "native_run_execution.h"
#include "runtime_c_api.h"

/** How far the set's current run has gone. `Idle` also means "not this run". */
enum class RunProgressState : uint8_t {
    Idle,
    Enqueuing,
    Submitted,
    DeviceComplete,
    Drained,
};

/**
 * One state per run-resource set, keyed on the run epoch the set describes.
 *
 * `run_epoch` is the whole identity: it is minted from a process-wide counter
 * and documented as unique for the process lifetime (`native_run_execution.h`),
 * where a pipeline slot is reused.
 */
class RunProgressSlots {
public:
    /**
     * Take a set for one run, before anything of that run is submitted.
     *
     * The state is cleared before the identity is published, so a reader asking
     * about the incoming run during the claim sees `Idle` rather than the
     * outgoing run's terminal state.
     */
    void claim(const NativeRunIdentity &identity) {
        if (identity.pipeline_slot >= slots_.size()) return;
        Slot &slot = slots_[identity.pipeline_slot];
        slot.state.store(RunProgressState::Idle, std::memory_order_release);
        slot.run_epoch.store(identity.run_epoch, std::memory_order_release);
    }

    /**
     * Move one run's own state.
     *
     * Refused when the set no longer describes that run, which is what keeps a
     * call from a run that has given the set up — a late drain, a rollback —
     * from moving the state of the run holding it now. Returns whether it was
     * published, so a caller can report a refusal rather than assume it.
     */
    bool publish(const NativeRunIdentity &identity, RunProgressState state) {
        if (identity.pipeline_slot >= slots_.size()) return false;
        Slot &slot = slots_[identity.pipeline_slot];
        if (slot.run_epoch.load(std::memory_order_acquire) != identity.run_epoch) return false;
        slot.state.store(state, std::memory_order_release);
        return true;
    }

    /**
     * This run's state, or `Idle` when the set does not describe it.
     *
     * The identity brackets the state read on both sides. Seeing the run's
     * epoch first means its claim completed, so the state read cannot precede
     * it; reading the epoch again afterwards excludes a later claim's run
     * having published its own state in between. Either mismatch answers
     * `Idle`, which refuses rather than reporting another run's progress.
     */
    RunProgressState state_of(uint32_t pipeline_slot, uint64_t run_epoch) const {
        if (pipeline_slot >= slots_.size() || run_epoch == 0) return RunProgressState::Idle;
        const Slot &slot = slots_[pipeline_slot];
        if (slot.run_epoch.load(std::memory_order_acquire) != run_epoch) return RunProgressState::Idle;
        const RunProgressState state = slot.state.load(std::memory_order_acquire);
        if (slot.run_epoch.load(std::memory_order_acquire) != run_epoch) return RunProgressState::Idle;
        return state;
    }

    RunProgressState state_of(const NativeRunIdentity &identity) const {
        return state_of(identity.pipeline_slot, identity.run_epoch);
    }

private:
    struct Slot {
        std::atomic<RunProgressState> state{RunProgressState::Idle};
        std::atomic<uint64_t> run_epoch{0};
    };
    std::array<Slot, PTO_PIPELINE_MAX_DEPTH> slots_{};
};
