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

#include "host/run_resource_growth.h"

#include <vector>

#include "common/unified_log.h"
#include "runtime_c_api.h"

namespace {

/** Regions one arena bank publishes. */
constexpr std::size_t kBankRegionCount = static_cast<std::size_t>(WorkspaceManager::ArenaRegion::kCount);

}  // namespace

int RunResourceGrowth::claim_permission(uint32_t slot, std::uint64_t run_epoch, AutoAdmissionState::Phase phase) {
    if (!admission_.enabled()) return 0;
    admission_.begin_plan(slot, run_epoch, phase);
    if (still_blocked(slot, phase)) return SIMPLER_RUNTIME_ERR_RESOURCE_WAIT;
    if (admission_.acquire_growth_token(slot, run_epoch) == AutoAdmissionState::Growth::WaitForToken) {
        // One grower per endpoint: two plans replacing regions at once would
        // each hold their old blocks beside their new ones, so neither would be
        // asking the allocator the question it answers, and a refusal could not
        // be attributed to either shortfall.
        admission_.note_parked(slot, AutoAdmissionState::Park::Token);
        return SIMPLER_RUNTIME_ERR_RESOURCE_WAIT;
    }
    return 0;
}

bool RunResourceGrowth::still_blocked(uint32_t slot, AutoAdmissionState::Phase phase) {
    const AutoAdmissionState::Park parked = admission_.parked(slot);
    if (parked == AutoAdmissionState::Park::None) return false;
    if (parked == AutoAdmissionState::Park::Token) {
        if (admission_.token_held_by_other(slot)) return true;
        admission_.note_parked(slot, AutoAdmissionState::Park::None);
        return false;
    }
    // Waiting on bytes: only the finish of an identity this plan captured when
    // it first could not proceed releases it, and each identity is spent once.
    // An unrelated run finishing changes nothing here, and a run that starts
    // afterwards is not added — so the waits are bounded by the set's own size.
    return !admission_.consume_finished_holder(slot, phase, [this](std::uint64_t holder) {
        return ledger_.holds_references(holder);
    });
}

void RunResourceGrowth::rollback_claims() {
    ::rollback_workspace_claims(ledger_, [](void *base) {
        LOG_ERROR("workspace rollback: the ledger no longer holds %p, which this transaction claimed", base);
    });
}

std::size_t RunResourceGrowth::trim_idle_sets(uint32_t requesting_slot) {
    const std::size_t granted = endpoint_.run_resource_sets();
    if (granted == 0) return 0;

    // A parked plan has given its attempt's references back and released its
    // reservation, yet its saved plan still describes the bank it was
    // committing into — and which bank that is cannot be read from a slot that
    // holds no reservation. One such plan therefore withdraws the arena trim
    // for every bank. The staging trim stays per slot, where the plan's own
    // slot is what excludes it.
    bool arena_trim_available = true;
    for (std::size_t slot = 0; slot < granted; ++slot) {
        const uint32_t id = static_cast<uint32_t>(slot);
        if (endpoint_.plan_live(id) && !endpoint_.slot_reserved(id)) arena_trim_available = false;
    }

    std::vector<TrimmableRunResourceSet> rows(granted);
    std::vector<std::array<ArenaRegionRequest, kBankRegionCount>> requests(granted);
    for (std::size_t slot = 0; slot < granted; ++slot) {
        const uint32_t id = static_cast<uint32_t>(slot);
        TrimmableRunResourceSet &row = rows[slot];
        row.slot = id;
        // Busy is decided by what holds a set, never by what its blocks look
        // like: a live run's set is unreferenced in the window between its
        // facts, and the asking set must keep everything it already had.
        row.busy = id == requesting_slot || endpoint_.slot_reserved(id) || endpoint_.plan_live(id);
        if (row.busy) continue;
        row.staging = endpoint_.staging_backing(id);
        if (!arena_trim_available || endpoint_.bank_reserved(id)) continue;
        const std::size_t region_count = endpoint_.arena_release_requests(id, requests[slot].data(), kBankRegionCount);
        if (region_count == 0) continue;
        row.arena = requests[slot].data();
        row.arena_count = region_count;
        row.owns_prebuilt_cache = endpoint_.owns_prebuilt_cache(id);
        row.prebuilt_cache = endpoint_.prebuilt_cache(id);
    }

    RunResourceEndpoint &endpoint = endpoint_;
    const RunResourceTrimOutcome outcome = trim_idle_run_resource_sets(
        ledger_, rows.data(), rows.size(),
        [&endpoint](uint32_t slot) {
            // The slot stops naming the block before anything frees it, so no
            // released address stays published. Its contents are not needed: a
            // staging block is written by the run that grows it, and the next
            // run on this slot grows its own from nothing.
            endpoint.release_staging_backing(slot);
        },
        [&endpoint](uint32_t slot) {
            // The proof was that a run's resources could be held here, and they
            // no longer are. A set offered on the strength of storage that has
            // been given back would admit a run with nothing behind it.
            endpoint.withdraw_run_resource_proof(slot);
        }
    );
    if (outcome.sets_trimmed != 0) {
        LOG_INFO(
            "run resources: slot %u took back %zu idle set(s) — %zu staging block(s) and %zu arena region(s), %llu "
            "bytes — before asking again",
            requesting_slot, outcome.sets_trimmed, outcome.staging_blocks, outcome.arena_regions,
            static_cast<unsigned long long>(outcome.bytes_detached)
        );
    }
    return outcome.sets_trimmed;
}

GrowthAttempt RunResourceGrowth::attempt_with_one_retry(
    uint32_t requesting_slot, bool rollback_before_retry, GrowthAttempt (*attempt)(void *ctx), void *ctx
) {
    if (attempt == nullptr) return GrowthAttempt::Degraded;
    GrowthAttempt outcome = attempt(ctx);
    if (outcome != GrowthAttempt::Refused) return outcome;
    // An endpoint that does not size itself keeps the single attempt it always
    // had: the trim exists to serve a capacity this endpoint derives, and a
    // caller who configured its own has not asked for storage to be moved
    // underneath it.
    if (!admission_.enabled()) return outcome;
    // Give back what this attempt claimed, take back the sets nothing can
    // reach, reclaim what this endpoint has provably finished with — which
    // includes both the blocks the rollback just made recoverable and the sets
    // the trim detached — and ask exactly once more.
    if (rollback_before_retry) rollback_claims();
    (void)trim_idle_sets(requesting_slot);
    (void)endpoint_.reclaim_obsolete();
    return attempt(ctx);
}

int RunResourceGrowth::report_refusal(
    uint32_t slot, AutoAdmissionState::Phase phase, std::size_t shortfall, GrowthAttempt outcome
) {
    rollback_claims();
    if (!admission_.enabled() || outcome != GrowthAttempt::Refused) return 0;
    // Every holder on this endpoint, not only the holders of the regions this
    // plan is short of: the blocks are charged against one device allocator, so
    // a predecessor occupying another set holds bytes this request can receive.
    // Its region keys are that set's and never the candidate's, and a set being
    // offered for the first time has no prior holder of its own keys at all.
    const std::uint64_t self = admission_.plan(slot).epoch;
    std::vector<std::uint64_t> holders;
    ledger_.eligible_holders(self, &holders);
    if (admission_.capture_wait_set(slot, phase, holders)) {
        const std::vector<std::uint64_t> pending = admission_.pending_wait_set(slot, phase);
        LOG_INFO(
            "run resources: slot %u needs %zu more bytes than this endpoint can take now; waiting for %zu launched "
            "consumer(s) on this endpoint to finish",
            slot, shortfall, pending.size()
        );
        admission_.note_parked(slot, AutoAdmissionState::Park::Bytes);
        return SIMPLER_RUNTIME_ERR_RESOURCE_WAIT;
    }
    LOG_ERROR(
        "run resources: slot %u cannot take %zu bytes, and no launched consumer remains on this endpoint whose "
        "finish could supply them",
        slot, shortfall
    );
    return SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE;
}
