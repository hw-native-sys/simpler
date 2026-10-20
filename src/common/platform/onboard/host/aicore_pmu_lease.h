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

#include <cstdint>
#include <mutex>
#include <set>
#include <string>

/**
 * @file aicore_pmu_lease.h
 * @brief Holds the driver's FFTS task-profiling channel open for the duration
 *        of a PMU collection, so the AICore PMU control register this process
 *        programmed out-of-band is not reset underneath it.
 *
 * simpler programs the AICore PMU window by writing MMIO directly from the
 * AICPU. TS also writes those registers, and nothing arbitrates between the
 * two: an AICore-utilization query (`npu-smi info -t usages`) or a background
 * periodic source resets `PMU_CTRL_0` to a resting value mid-collection, and
 * every counter reads zero from then on. While prof channel 53
 * (`CHANNEL_FFTS_PROFILE_BUFFER_TASK`) is started on a device, that reset does
 * not happen.
 *
 * What the measurements support, and what they do not:
 * - Starting the channel is what protects. The configuration's content does
 *   not: `cfgMode` 0 and 3 protect alike, and so does clearing the config's
 *   sub-task bit. 9/9 rounds lost every counter with no channel held; 0/15
 *   lost any with one held, across 30780 rows.
 * - `cfgMode = 0` ("none") is nevertheless required, because this lease is
 *   taken in the process that submits the tasks: at `cfgMode = 3` the device
 *   emits 128 bytes per task into a host ring that no reader drains, and that
 *   ring overwrites its oldest bytes without accounting for the loss. At
 *   `cfgMode = 0` a 4000-kernel in-process workload produced nothing.
 * - *Why* holding the channel protects is not established. It may be an
 *   implementation detail of the device firmware, which is not open and which
 *   a CANN upgrade may change without any signal here. The collector's
 *   zero-counter reporting is what detects that, and is not optional.
 *
 * The channel is exclusive across processes, enforced below user space: a
 * second start returns -10 while one is held. So a held lease denies the
 * channel to msprof on that device for as long as PMU collection runs, and a
 * profiler already holding it denies the lease to us (reported as
 * `State::kOccupied`; collection proceeds unprotected).
 *
 * A lease never fails a run. PMU collection is a diagnostic channel, so every
 * refusal is a logged state, not an error return.
 */
class AicorePmuLease {
public:
    /**
     * @brief Outcome of the most recent acquire attempt.
     *
     * `kAbsent` covers both a driver without the symbols and a build with no
     * driver at all; neither is a fault of the caller.
     */
    enum class State { kAbsent, kHeld, kOccupied, kRefused };

    AicorePmuLease() = default;
    ~AicorePmuLease();

    AicorePmuLease(const AicorePmuLease &) = delete;
    AicorePmuLease &operator=(const AicorePmuLease &) = delete;

    /**
     * @brief Takes a reference on the channel, starting it on the first one.
     *
     * `device_id` is an ACL/rt *logical* id, the same one `aclrtSetDevice`
     * takes. It binds the lease on the first acquire; a later acquire naming a
     * different device is a caller defect and is refused rather than silently
     * retargeting the channel.
     *
     * A reference is recorded against `run_epoch`, and only a release naming an
     * epoch that holds one has any effect. That is what keeps the reference
     * count honest while runs overlap: a run whose acquire was refused owns
     * nothing, so its own boundary cannot drop a successor's reference, and a
     * path that releases twice drops one.
     *
     * @return true when the channel is held once this returns. A false return
     *         is a diagnosis to log, not a reason to abandon collection.
     */
    bool acquire(int device_id, std::uint64_t run_epoch);

    /**
     * @brief Drops a reference, stopping the channel when the last one goes.
     *
     * A no-op unless `run_epoch` holds a reference, so every drop path can
     * call it unconditionally: the rollback that fires when an earlier
     * collector refused the run, the refusal path inside the PMU block, and
     * the run's close boundary.
     *
     * Stopping drains the channel once and reports a non-zero byte count at
     * WARN: at `cfgMode = 0` there is nothing to drain, so bytes here mean
     * that invariant no longer holds.
     */
    void release(std::uint64_t run_epoch);

    State state() const;

    /// Stable lowercase token for logs and for the collector's terminal line.
    const char *state_name() const;

    /// Empty unless the state is `kOccupied` or `kRefused`.
    std::string why() const;

    bool held() const;

private:
    mutable std::mutex mutex_;
    /// Logical id the caller named; what an owner-binding check compares.
    int device_id_ = -1;
    /**
     * Driver-visible id every HAL call here passes, resolved once at acquire.
     *
     * The two spaces differ whenever `ASCEND_RT_VISIBLE_DEVICES` remaps, and
     * holding the channel for the wrong one protects nothing while reporting
     * success. Resolved once so start, flush, read and stop — including the
     * destructor's — cannot disagree about the target.
     */
    int hal_device_id_ = -1;
    /// Epochs holding a reference; `size()` is the count.
    std::set<std::uint64_t> owners_;
    State state_ = State::kAbsent;
    std::string why_;
};
