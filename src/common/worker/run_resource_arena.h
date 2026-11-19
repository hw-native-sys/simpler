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
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "runtime_c_api.h"

/**
 * How many run-resource sets a context holds when nothing configured it.
 *
 * Two is what every context held before the count could be requested, so a
 * context that wants more says so and an unconfigured one moves neither its
 * capacity nor its footprint.
 */
static constexpr uint32_t kDefaultRunResourceSets = 2;
static_assert(kDefaultRunResourceSets >= 1, "a context must be able to hold at least one run");

/**
 * Owner of one context's run-resource sets and of the leases that name them.
 *
 * A set is the group of per-run resources a run borrows for its whole lifetime;
 * this class owns which of them exist and who holds each one, and nothing about
 * what is in one. The count is the finite grant selected for this context, and a
 * lease carries the generation that makes a released set unusable to its former
 * holder.
 *
 * Three counts, which together always equal the grant's worth of sets that have
 * ever been handed out:
 *
 * - **free** — constructed or constructible, available to the next run.
 * - **live** — held by a run right now.
 * - **quarantined** — held by a run whose teardown could not be proved, never
 *   reused, and still counted against the grant so a context cannot grow its
 *   concurrency by failing.
 *
 * This is capability only: callers decide when admission is allowed. A released
 * lease remains idempotently releasable until the set is acquired again; after
 * reuse, the old generation can neither access nor release it.
 */
class RunResourceArena {
public:
    explicit RunResourceArena(uint32_t granted) :
        granted_(granted),
        sets_(granted) {
        if (granted == 0) {
            throw std::invalid_argument("a context must be granted at least one run-resource set");
        }
    }

    std::optional<PipelineSlotLease> try_acquire() { return try_acquire(granted_); }

    /**
     * Size the arena to a grant settled after construction.
     *
     * Refused once any set has been handed out: the sets a run holds are named
     * by index, and an index whose meaning changed underneath a live run is not
     * a capacity change but a different resource.
     *
     * The new table is built before anything is published, so a failed
     * allocation leaves the previous grant whole rather than a count that no
     * longer matches the table behind it.
     */
    void regrant(uint32_t granted) {
        if (granted == 0) {
            throw std::invalid_argument("a context must be granted at least one run-resource set");
        }
        std::vector<SetState> built(granted);
        std::lock_guard<std::mutex> lock(mu_);
        for (const SetState &state : sets_) {
            if (state.in_use || state.quarantined || state.generation != 0) {
                throw std::logic_error("the run-resource grant cannot change after a set has been handed out");
            }
        }
        // Both steps below are non-throwing, so the count and the table it
        // describes move together.
        sets_.swap(built);
        granted_ = granted;
    }

    std::optional<PipelineSlotLease> try_acquire(uint32_t admission_depth) {
        if (admission_depth == 0 || admission_depth > granted_) {
            throw std::invalid_argument("run-resource admission depth is outside the arena's grant");
        }
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t slot = 0; slot < admission_depth; ++slot) {
            SetState &state = sets_[slot];
            if (state.in_use || state.quarantined) continue;
            if (state.generation == std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error("run-resource set generation space exhausted");
            }
            ++state.generation;
            state.in_use = true;
            return PipelineSlotLease{slot, 0, state.generation};
        }
        return std::nullopt;
    }

    bool owns(const PipelineSlotLease &lease) const {
        std::lock_guard<std::mutex> lock(mu_);
        return owns_locked(lease);
    }

    bool release(const PipelineSlotLease &lease) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!matches_generation_locked(lease)) return false;
        if (sets_[lease.slot_id].quarantined) return false;
        // Releasing the current generation twice is deliberately idempotent.
        sets_[lease.slot_id].in_use = false;
        return true;
    }

    /**
     * Withdraw one set from service for the rest of this context's life.
     *
     * The run that held it could not prove what it left behind, so the set is
     * never handed to another run and never released. It keeps counting against
     * the grant, which is what stops a context from acquiring more concurrency
     * than it was granted by failing repeatedly.
     */
    bool quarantine(const PipelineSlotLease &lease) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!matches_generation_locked(lease)) return false;
        sets_[lease.slot_id].quarantined = true;
        sets_[lease.slot_id].in_use = false;
        return true;
    }

    bool quarantined(uint32_t slot_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        return slot_id < sets_.size() && sets_[slot_id].quarantined;
    }

    uint32_t granted() const { return granted_; }

    /** How many sets a run could take right now. */
    uint32_t free_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        uint32_t free_sets = 0;
        for (const SetState &state : sets_) {
            if (!state.in_use && !state.quarantined) ++free_sets;
        }
        return free_sets;
    }

    /** How many sets runs hold right now. */
    uint32_t live_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        uint32_t live = 0;
        for (const SetState &state : sets_) {
            if (state.in_use && !state.quarantined) ++live;
        }
        return live;
    }

    uint32_t quarantined_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        uint32_t withdrawn = 0;
        for (const SetState &state : sets_) {
            if (state.quarantined) ++withdrawn;
        }
        return withdrawn;
    }

private:
    struct SetState {
        uint64_t generation{0};
        bool in_use{false};
        bool quarantined{false};
    };

    bool matches_generation_locked(const PipelineSlotLease &lease) const {
        return lease.reserved == 0 && lease.generation != 0 && lease.slot_id < sets_.size() &&
               sets_[lease.slot_id].generation == lease.generation;
    }

    bool owns_locked(const PipelineSlotLease &lease) const {
        return matches_generation_locked(lease) && sets_[lease.slot_id].in_use;
    }

    uint32_t granted_;
    mutable std::mutex mu_;
    std::vector<SetState> sets_;
};

/**
 * Replay filter for a consumer that executes leases minted elsewhere.
 *
 * `RunResourceArena` is the sole mint and the sole authority on who holds a
 * set. A consumer downstream of it never sees an acquire or a release, so it
 * cannot re-derive ownership; it records the newest generation each set has
 * presented and refuses anything older. Repeating the current generation stays
 * admissible, because one run may dispatch several times under one lease.
 *
 * This is strictly weaker than an ownership check, and deliberately so. It
 * rejects a *superseded* lease — one whose successor has already presented
 * itself — but a lease that was released while its successor has not yet
 * reached this consumer still passes, because nothing has raised the mark.
 * Closing that window requires gating dispatch on `RunResourceArena::owns`
 * before work is handed down, which belongs to whoever admits runs.
 *
 * The filter mints nothing. A consumer that also served unleased work must not
 * invent generations for it, or a later real lease is rejected as stale.
 *
 * Sized by `resize` once the grant is known; before that it holds nothing and
 * admits nothing, which is the same refusal an out-of-range set gets.
 */
class PipelineSlotGenerationFilter {
public:
    void resize(uint32_t granted) {
        std::lock_guard<std::mutex> lock(mu_);
        newest_.assign(granted, 0);
    }

    // Preview half of the preview/commit pair: reports the same verdict admit()
    // would give, without advancing the set. Const because committing nothing
    // is the whole reason it exists — preparation can fail after this check, and
    // a generation spent on a run that then falls back is not recoverable.
    bool is_admissible(const PipelineSlotLease &lease) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (lease.slot_id >= newest_.size()) return false;
        return lease.generation >= newest_[lease.slot_id];
    }

    bool admit(const PipelineSlotLease &lease) {
        std::lock_guard<std::mutex> lock(mu_);
        if (lease.slot_id >= newest_.size()) return false;
        uint64_t &newest = newest_[lease.slot_id];
        if (lease.generation < newest) return false;
        newest = lease.generation;
        return true;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        std::fill(newest_.begin(), newest_.end(), 0);
    }

private:
    mutable std::mutex mu_;
    std::vector<uint64_t> newest_;
};
