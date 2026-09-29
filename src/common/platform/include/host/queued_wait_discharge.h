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
 * The ladder a run climbs to prove the stream waits naming its boundaries were
 * consumed, before anything it owns is released.
 *
 * Separate from the runner so the rungs and their return-code precedence are
 * exercisable on their own: the two device-facing steps arrive as callables,
 * and the wait table arrives with its own injectable event ops. Both runners
 * pass the real ones, and each reports its own step — this carries no logging
 * of its own so that it depends on nothing but the table.
 */

#pragma once

#include <functional>

#include "queued_stream_waits.h"
#include "runtime_c_api.h"

/** The two device-facing steps the ladder needs, supplied by the runner. */
struct QueuedWaitDischargeOps {
    /**
     * A bounded synchronize over the pair of streams a wait could be sitting
     * in. Covers the boundary and everything queued behind it, which is what an
     * unproven queued wait needs.
     */
    std::function<int()> synchronize;
    /**
     * Report that a reference no proof covers is still unresolved: the runner
     * stops admitting runs and quarantines what this run holds. Called with the
     * failure being reported, and only on the path where a wait may still name
     * one of this run's events. It does not establish that the reference is
     * gone — only an independent proof of consumption, or a confirmed teardown
     * or device reset elsewhere, can do that.
     */
    std::function<void(int)> poison;
};

/**
 * Retire every queued wait naming a boundary of `identity`, and report whether
 * that could be proved.
 *
 * Three rungs, cheapest first, and the first that proves consumption wins:
 *
 *   1. the table's own evidence — the run's completed boundaries for its own
 *      wait, the proof event recorded behind a cross-run wait for that one;
 *   2. a bounded whole-pair synchronize followed by a quiescence retirement,
 *      which is a failure-path cost only;
 *   3. nothing. A wait may still name an event of this run, so the caller
 *      releases nothing against it: the reference stays retained and the runner
 *      is asked to stop admitting runs and quarantine what this one holds.
 *
 * **Return-code precedence: the first failure reported wins.** A later rung's
 * error never replaces the error that sent the run down the ladder, because
 * that first one is the caller's original failure and the rest are consequences
 * of it. A rung that succeeds after an earlier failure still does not clear it.
 *
 * Returning zero means the references are retired *and* nothing reported an
 * error. A non-zero return with the references retired is an error to report
 * with nothing quarantined — the caller's run fails, the runner keeps admitting.
 * A non-zero return with a reference still held is the third rung: this ladder
 * has proved nothing about that reference, and nothing here proves it safe to
 * release later either. That proof belongs to whoever next establishes
 * consumption, quiescence, or a confirmed teardown or reset.
 */
inline int discharge_queued_waits(
    QueuedStreamWaits &waits, const NativeRunIdentity &identity, bool boundaries_complete, int synchronize_timeout_ms,
    const QueuedWaitDischargeOps &ops
) {
    if (!waits.holds_reference_to(identity)) return 0;

    int first_rc = waits.discharge(identity, boundaries_complete, synchronize_timeout_ms);
    if (!waits.holds_reference_to(identity)) return first_rc;

    const int sync_rc = ops.synchronize ? ops.synchronize() : PTO_RUNTIME_ERR_INTERNAL;
    if (sync_rc == 0) {
        const int quiesce_rc = waits.discharge_on_quiescence(identity);
        if (first_rc == 0) first_rc = quiesce_rc;
        if (!waits.holds_reference_to(identity)) return first_rc;
    } else if (first_rc == 0) {
        first_rc = sync_rc;
    }

    if (first_rc == 0) first_rc = PTO_RUNTIME_ERR_INTERNAL;
    if (ops.poison) ops.poison(first_rc);
    return first_rc;
}
