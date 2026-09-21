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

#include <memory>

#include "device_runner_base.h"
#include "runtime_c_api.h"

/**
 * #2267's late-read retention fixture: a predecessor that has completed and
 * been read but not finalized, while its successor executes on the same
 * streams.
 *
 * That state is what removing the drain's whole-stream synchronize depends on,
 * and production refuses to produce it — measured on a2a3, in three separate
 * places. Preparing both runs before launching either is refused by the lane
 * ("requires an active predecessor before staging a successor"); the
 * successor's launch is then refused because the execution claim, taken at
 * launch and released only by `simpler_finalize_run`, is still the
 * predecessor's; and a2a3's `RunStreamPair::ensure()` independently refuses
 * while an unretired run still owns the pair.
 *
 * So the fixture reaches it by replacing exactly three decisions, each of which
 * is one node E changes:
 *
 *   1. the predecessor's wait becomes its own two boundary events, with no
 *      `sync_stream_pair` on any shape — `wait_run_fence` reaches that call on
 *      every shape but a normal success;
 *   2. the predecessor's stream-pair ownership is retired on that boundary
 *      evidence alone, without the drain that normally accompanies it;
 *   3. the successor's launch permit is minted rather than claimed.
 *
 * Everything else — preparation, the predecessor's launch, both drains,
 * finalize — is the production path unchanged. Nothing in the product calls
 * any of this, and no production decision is altered by its presence.
 */
class RunRetentionProbePeer {
public:
    using PreparedExecution = DeviceRunnerBase::PreparedExecution;
    using ActiveExecution = DeviceRunnerBase::ActiveExecution;

    /**
     * A permit for a run that holds no execution claim.
     *
     * This is the fixture's whole admission difference, isolated to one call so
     * a reader can see that the launch transaction itself is untouched.
     */
    static LaunchPermit mint_permit(const NativeRunIdentity &identity) { return LaunchPermit(identity); }

    /** This slot's completion fence, which is per slot and keyed on run identity. */
    static RunCompletionFence &fence(DeviceRunnerBase &runner, uint32_t pipeline_slot) {
        return runner.run_fence(pipeline_slot);
    }

    /** The bounded whole-pair synchronize `wait_run_fence` falls back to. */
    static int sync_streams(DeviceRunnerBase &runner, rtStream_t aicpu, rtStream_t aicore) {
        return runner.sync_stream_pair(aicpu, aicore);
    }

    static int stream_sync_timeout_ms(const DeviceRunnerBase &runner) {
        return runner.timeout_config_.stream_sync_timeout_ms;
    }

    /**
     * Whether this slot's cached result region holds a successful read of
     * `run_epoch`'s record.
     *
     * The read entry point returns void, so nothing else tells the fixture
     * whether the copy succeeded. Reporting it rather than assuming it is what
     * lets the decision rule reach its own read-failure path, which is the path
     * node E will depend on.
     */
    static bool result_read_ok(const DeviceRunnerBase &runner, uint32_t pipeline_slot, uint64_t run_epoch) {
        return runner.device_run_result_reads_.state(pipeline_slot, run_epoch) == RunRecordRead::Ok;
    }

    /**
     * The streams this runner's runs submit on.
     *
     * Per arch because the two do not agree on what owns them: a2a3 hands each
     * run a `RunStreamPair`, a5 submits every run on the persistent bootstrap
     * streams created at device bring-up.
     */
    static void run_streams(DeviceRunnerBase &runner, rtStream_t *aicpu, rtStream_t *aicore);

    /**
     * Release whatever this arch makes a successor's launch wait for, using
     * `prepared`'s already-proven boundary completion.
     *
     * a2a3 retires the run's stream-pair ownership as Complete, which keeps the
     * AICore stream and lets `ensure()` accept the successor. a5 has no pair
     * and nothing to retire, so this is where the two arches genuinely differ
     * rather than where one is a special case of the other.
     *
     * Retires ownership only. The run keeps its pipeline slot, its result
     * region, its fence arming, its runtime and its argument views — which is
     * what leaves its result readable afterwards.
     */
    static int retire_predecessor_ownership(DeviceRunnerBase &runner, PreparedExecution &prepared);

    /**
     * Make `prepared` the run this arch's ordinary drain will accept.
     *
     * a5 gates poll and drain on one runner-wide slot id that the successor's
     * launch overwrites and no path restores, so its predecessor becomes
     * undrainable through the ordinary entry. Restoring it is fixture cleanup
     * and nothing else: it runs only after the successor has fully drained, so
     * it never stands in for two concurrently drainable runs — which is the
     * misuse that would make the whole measurement invalid. a2a3 keys the same
     * decisions on the submitting run's own pointer and needs nothing here.
     */
    static void adopt_drain_ownership(DeviceRunnerBase &runner, const PreparedExecution &prepared);
};

/**
 * Run the sequence over a launched predecessor and a prepared successor.
 *
 * `active` is the predecessor, already launched through the ordinary path and
 * still owning device work. `prepared_successor` is prepared on a different
 * pipeline slot and not launched; it is taken by reference because the fixture
 * consumes it only when it actually launches, and a successor it does not
 * launch must go back to the caller still owned.
 *
 * On return a launched successor has drained and `active_successor_out` holds
 * its execution state, so the caller can hand both runs to
 * `simpler_finalize_run` exactly as an ordinary wait would.
 *
 * Every step's status lands in `report`; a device failure part-way through
 * stops the sequence and still returns 0, because the probe's job is to measure
 * rather than to decide. A negative return means the fixture refused its
 * arguments.
 */
int run_retention_probe(
    DeviceRunnerBase &runner, DeviceRunnerBase::ActiveExecution &active,
    std::unique_ptr<DeviceRunnerBase::PreparedExecution> &prepared_successor, const RunRetentionProbeConfig &config,
    RunRetentionProbeReport *report, std::unique_ptr<DeviceRunnerBase::ActiveExecution> *active_successor_out
);
