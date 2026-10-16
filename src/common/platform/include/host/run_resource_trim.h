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
#include <vector>

#include "host/arena_replacement_transaction.h"
#include "host/auto_admission.h"
#include "host/workspace_manager.h"

/**
 * Give back every reference the abandoned transaction on this thread took.
 *
 * A staged block a transaction gives up keeps both its charge and the plan's
 * reference: the arena's abort frees through the allocation callback, which
 * does nothing for a block the ledger owns, and relinquishing the
 * published-owner claim drops no reference. So the record of what was claimed
 * is the only thing that can recover those bytes before the whole run retires
 * — which a plan that is going to resume never does.
 *
 * Only this plan's own reference goes back. Dropping every reference of the
 * epoch instead would also return the input staging a waiting plan legitimately
 * holds and still needs.
 *
 * `on_missing(base)` reports a claimed address the ledger no longer holds,
 * which is a bookkeeping contradiction rather than a recoverable state.
 */
template <typename OnMissing>
inline void rollback_workspace_claims(WorkspaceManager &ledger, OnMissing on_missing) {
    std::vector<AutoAdmissionState::Claim> &claims = AutoAdmissionState::claims();
    // Newest first: a later claim can only be a block an earlier one does not
    // own, so undoing in reverse keeps each step independent of the rest.
    for (auto it = claims.rbegin(); it != claims.rend(); ++it) {
        if (!ledger.release_ref(it->base, it->epoch)) on_missing(it->base);
        // A block the allocator handed back because it is already this region's
        // backing is not a generation to relinquish: the region's ownership of
        // it survives this transaction, and clearing it would leave the region
        // publishing an address the next growth may reclaim underneath it.
        if (!it->was_region_backing) (void)ledger.note_unpublished(it->base);
    }
    claims.clear();
}

/**
 * Returning an idle run-resource set's storage when a growth cannot be served.
 *
 * `reclaim_obsolete` releases only what a region has *replaced*: a set nothing
 * is using keeps its blocks, because idle is not abandoned and keeping them
 * warm is what lets the next same-sized run reuse them. Under pressure that
 * same rule is what makes a request fail beside storage it could have had — a
 * set cached from an earlier run holds bytes the asking set needs, and no
 * reclamation will ever offer them.
 *
 * Trimming is the one step that converts them, and the ledger cannot take it
 * alone: a published block is still named by the arena or the slot that holds
 * it, so the owner has to stop naming it *before* anything frees it. That
 * ordering is the whole content of this header — detach, let the owner clear
 * what it published and withdraw the proof that rested on it, and only then
 * let `reclaim_obsolete` perform the device release.
 *
 * Nothing here decides what is busy. A set reaches this only once its owner has
 * established that no live run and no saved plan can reach its storage; what
 * this adds is the ledger's own answer about each block — unreferenced,
 * provable, and its region's current generation.
 */
struct TrimmableRunResourceSet {
    /** The set's own index, reported back so its owner can withdraw its proof. */
    uint32_t slot{0};
    /**
     * Whether anything can still reach this set's storage.
     *
     * Defaults to busy so a row its owner did not fill is never trimmed: the
     * safe answer to "can something reach this" is yes.
     */
    bool busy{true};
    /** The slot's current input-staging backing, or null when it has none. */
    void *staging{nullptr};
    /**
     * This set's pooled arena regions, each requesting zero bytes so the
     * transaction's own release path detaches them. Null when the set has no
     * arena storage of its own — a runtime whose arena is shared leaves this
     * empty, because a shared bank is never one set's to give back.
     */
    ArenaRegionRequest *arena{nullptr};
    std::size_t arena_count{0};
    /** Whether this set's bank owns the prebuilt runtime-image entry. */
    bool owns_prebuilt_cache{false};
    PrebuiltRuntimeArenaCache *prebuilt_cache{nullptr};
};

/** What one trim detached, and from where. */
struct RunResourceTrimOutcome {
    std::size_t sets_trimmed{0};
    std::size_t staging_blocks{0};
    std::size_t arena_regions{0};
    std::uint64_t bytes_detached{0};
};

/**
 * Whether every committed region of this set's bank is idle managed backing.
 *
 * All or nothing per bank, because the transaction publishes its regions
 * together: a bank one live consumer still reaches is not an idle set, and
 * detaching the rest of it would hand back part of a set in use. A region
 * holding nothing is neither an obstacle nor a contribution — it has nothing
 * to give back and nothing to keep — so a bank made only of those yields
 * false and is skipped rather than walked.
 */
inline bool arena_set_is_trimmable(const WorkspaceManager &ledger, const TrimmableRunResourceSet &set) {
    if (set.arena == nullptr) return false;
    bool any = false;
    for (std::size_t i = 0; i < set.arena_count; ++i) {
        if (set.arena[i].arena == nullptr) return false;
        void *backing = set.arena[i].arena->raw_backing();
        if (backing == nullptr) continue;
        if (!ledger.idle_backing(backing)) return false;
        any = true;
    }
    return any;
}

/**
 * Detach every eligible idle set's managed backing so a reclaim can free it.
 *
 * `staging_detached(slot)` must make the slot stop naming its staging block,
 * and `set_trimmed(slot)` must withdraw whatever that set's storage was
 * evidence for — its proof that a run can be held there. Both run *before*
 * this returns and therefore before the caller's reclamation, so no address
 * this hands to the platform is still published by its owner.
 *
 * The caller performs the reclamation itself, because the device release needs
 * a thread it has proved attached and this decides ownership only.
 */
template <typename StagingDetached, typename SetTrimmed>
inline RunResourceTrimOutcome trim_idle_run_resource_sets(
    WorkspaceManager &ledger, TrimmableRunResourceSet *sets, std::size_t count, StagingDetached staging_detached,
    SetTrimmed set_trimmed
) {
    RunResourceTrimOutcome outcome;
    if (sets == nullptr) return outcome;
    for (std::size_t i = 0; i < count; ++i) {
        TrimmableRunResourceSet &set = sets[i];
        if (set.busy) continue;
        bool trimmed_any = false;
        if (set.staging != nullptr) {
            // Decided and detached in one ledger call, so a block another
            // request took as its region's backing meanwhile is refused here
            // rather than detached out from under it.
            const std::uint64_t bytes = ledger.detach_idle_backing(set.staging);
            if (bytes != 0) {
                staging_detached(set.slot);
                outcome.bytes_detached += bytes;
                ++outcome.staging_blocks;
                trimmed_any = true;
            }
        }
        if (arena_set_is_trimmable(ledger, set)) {
            for (std::size_t r = 0; r < set.arena_count; ++r) {
                void *backing = set.arena[r].arena->raw_backing();
                if (backing == nullptr) continue;
                outcome.bytes_detached += ledger.block_bytes(backing);
                ++outcome.arena_regions;
            }
            // Every request asks for zero, so the transaction takes its own
            // release path for each committed region: the arena gives up its
            // base, the disposition tells the ledger the region now publishes
            // nothing, and the free that follows is a no-op for a block the
            // ledger owns. Nothing is staged, so there is no attempt to fail
            // and no state to roll back.
            const BankArenaSetupOutcome arena_outcome =
                run_bank_arena_setup(set.arena, set.arena_count, set.owns_prebuilt_cache, set.prebuilt_cache);
            if (arena_outcome.rc == 0 && arena_outcome.transaction.changed_regions != 0) trimmed_any = true;
        }
        if (!trimmed_any) continue;
        set_trimmed(set.slot);
        ++outcome.sets_trimmed;
    }
    return outcome;
}
