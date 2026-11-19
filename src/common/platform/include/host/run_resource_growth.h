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

#include <cstddef>
#include <cstdint>

#include "host/auto_admission.h"
#include "host/run_resource_trim.h"
#include "host/workspace_manager.h"

/**
 * What a growth under pressure needs to know about its endpoint.
 *
 * The device runner owns every one of these and decides none of them: each is a
 * fact it can report — which sets a live run holds, where a slot's staging is,
 * what a bank's regions are — while the decisions built on them live in
 * `RunResourceGrowth`. That split is the point: the runner's own translation
 * unit needs the device SDK and is therefore not reachable by a unit test, so
 * the decisions sit on this side of the boundary where they can be driven
 * directly, and the runner keeps only the answers nothing else can give.
 */
class RunResourceEndpoint {
public:
    virtual ~RunResourceEndpoint() = default;

    /** Run-resource sets this endpoint was granted. */
    virtual std::size_t run_resource_sets() const = 0;

    /** Whether a live run holds this set's slot. */
    virtual bool slot_reserved(uint32_t slot) const = 0;

    /**
     * Whether a live run holds this set's arena bank.
     *
     * Asked separately from the slot because a runtime whose arena is shared
     * resolves every slot to one bank: a run in another slot is then what makes
     * this bank unavailable, and the slot's own occupancy says nothing about it.
     */
    virtual bool bank_reserved(uint32_t slot) const = 0;

    /** Whether a saved plan would resume into this set. */
    virtual bool plan_live(uint32_t slot) const = 0;

    /** The slot's current input-staging backing, or null when it has none. */
    virtual void *staging_backing(uint32_t slot) const = 0;

    /** Stop naming the slot's staging backing. Called before any release. */
    virtual void release_staging_backing(uint32_t slot) = 0;

    /**
     * Fill this set's arena region requests, each asking for zero bytes.
     *
     * @return regions written, or 0 when this set has no arena of its own
     */
    virtual std::size_t arena_release_requests(uint32_t slot, ArenaRegionRequest *out, std::size_t capacity) = 0;

    /** Whether this set's bank owns the prebuilt runtime-image entry. */
    virtual bool owns_prebuilt_cache(uint32_t slot) const = 0;

    /** The prebuilt runtime-image entry, or null when this set owns none. */
    virtual PrebuiltRuntimeArenaCache *prebuilt_cache(uint32_t slot) = 0;

    /** Withdraw the proof that a run's resources can be held in this set. */
    virtual void withdraw_run_resource_proof(uint32_t slot) = 0;

    /** Release every obsolete generation whose consumers are finished. */
    virtual int reclaim_obsolete() = 0;
};

/** What one growth attempt produced. */
enum class GrowthAttempt : uint8_t {
    /** The request was served. */
    Served,
    /** Refused, and returning storage could change the answer. */
    Refused,
    /**
     * Refused by a ledger that will publish nothing further. Neither a retry
     * nor a wait can differ, so this is the caller's own failure to report.
     */
    Degraded,
    /** Invalid request or ownership failure; returning bytes cannot repair it. */
    Failed,
};

inline GrowthAttempt classify_growth_refusal(WorkspaceManager::AcquireRefusal why) {
    using Refusal = WorkspaceManager::AcquireRefusal;
    if (why == Refusal::OverQuota || why == Refusal::BackendFailed) return GrowthAttempt::Refused;
    if (why == Refusal::Degraded) return GrowthAttempt::Degraded;
    return GrowthAttempt::Failed;
}

/**
 * The decisions a growth under allocator pressure makes, in one place.
 *
 * Three of them, and they are separate because they answer to different owners:
 * who may grow (one permission per endpoint), what a refused plan may wait for
 * (an immutable set of launched consumer identities), and what storage is safe
 * to take back before asking again (every set nothing can reach).
 *
 * Holds references and no state of its own, so a caller constructs one around a
 * call rather than keeping one: the state is the ledger's, the admission
 * state's and the endpoint's.
 */
class RunResourceGrowth {
public:
    RunResourceGrowth(WorkspaceManager &ledger, AutoAdmissionState &admission, RunResourceEndpoint &endpoint) :
        ledger_(ledger),
        admission_(admission),
        endpoint_(endpoint) {}

    /**
     * Open or re-enter this slot's plan and take the endpoint's growth
     * permission, or report that it must wait.
     *
     * @return 0 when this plan may grow now, else a typed resource-wait code
     */
    int claim_permission(uint32_t slot, std::uint64_t run_epoch, AutoAdmissionState::Phase phase);

    /** Whether a waiting plan may not yet try again. */
    bool still_blocked(uint32_t slot, AutoAdmissionState::Phase phase);

    /** Give back every reference this thread's abandoned transaction took. */
    void rollback_claims();

    /**
     * Return the storage of every run-resource set nothing can reach.
     *
     * @return sets whose storage was detached
     */
    std::size_t trim_idle_sets(uint32_t requesting_slot);

    /**
     * One growth attempt, then the trim and the reclaim, then exactly one more.
     *
     * The whole retry policy: a refusal is answered by returning storage the
     * endpoint provably does not need and asking again once — never by a loop,
     * and never at all when the ledger is degraded, because nothing it
     * publishes can change.
     *
     * `rollback_before_retry` is for a caller whose attempt stages storage it
     * must hand back before the trim can see it; a single-allocation attempt
     * that already left nothing behind passes false.
     */
    GrowthAttempt attempt_with_one_retry(
        uint32_t requesting_slot, bool rollback_before_retry, GrowthAttempt (*attempt)(void *ctx), void *ctx
    );

    /**
     * Turn a refused growth into a bounded wait on named consumers, into a
     * failure when there is nothing left to wait for, or into nothing when the
     * caller must report its own failure.
     *
     * A degraded ledger is the last case and is decided here rather than at
     * each growth point: it will publish nothing further, so no holder's
     * retirement can change its answer and parking on one would be a wait that
     * never ends. An endpoint that does not measure itself is the same case for
     * a different reason — it has no plan to park.
     *
     * @return a typed resource-wait or resource-unavailable code, or 0 when the
     *         caller owns reporting the failure
     */
    int report_refusal(uint32_t slot, AutoAdmissionState::Phase phase, std::size_t shortfall, GrowthAttempt outcome);

private:
    WorkspaceManager &ledger_;
    AutoAdmissionState &admission_;
    RunResourceEndpoint &endpoint_;
};
