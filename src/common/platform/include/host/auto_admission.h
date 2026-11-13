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

#include "common/module_thread_value.h"
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "host/run_resource_cost.h"

/**
 * Admission state for a device context that sizes its own concurrency.
 *
 * Three questions, kept apart because they have different owners and different
 * lifetimes:
 *
 *   - **how many run-resource sets may be admitted** — one slot is offered
 *     beyond the proven prefix, and only while no growth is in flight;
 *   - **who may grow** — one exclusive token per context, so two plans never
 *     superpose their replacement peaks and an allocator refusal belongs to
 *     exactly one shortfall;
 *   - **what a refused plan may wait for** — an immutable set of consumer
 *     identities captured at its first wait, consumed one retirement at a time.
 *
 * A slot is **proven** when a run on it completed its resource commit, not when
 * it holds bytes: a run whose private demand is legitimately zero — every
 * argument already on the device, every region satisfied by an existing
 * allocation — proves its slot exactly as a run that allocated does. Inferring
 * the proof from held storage instead would leave such a context at one run for
 * ever, and inferring it from an untouched slot's absent storage would admit
 * every slot before anything had been demonstrated.
 *
 * Capacity is a count of identities, and stays one: the proven prefix is a
 * set of facts, so no byte figure withdraws a slot that a run has already
 * committed on. What a byte figure does decide is the **exploratory** set — the
 * one offered beyond the proven prefix. When a finite budget is configured, the
 * exploratory set is withheld unless the bytes the ledger already charges plus
 * the cost of the largest proven set stay within the budget, so the probe is
 * not offered into a shortfall that one more set of the known size would
 * certainly create. Without a budget the probe is offered on proof alone,
 * exactly as before; a context that configured no byte policy has none to
 * measure against.
 */
class AutoAdmissionState {
public:
    /** Which prepare stage a slot's saved plan has reached. */
    enum class Phase : uint8_t {
        /** No live plan on this slot. */
        None = 0,
        /** Input staging: the plan's own host-to-device arguments. */
        Staging = 1,
        /** The orchestration callback has run and its product is saved. */
        Planned = 2,
        /** Committing the pooled device regions. */
        Committing = 3,
    };

    /** Why a plan is waiting, or that it is not. */
    enum class Park : uint8_t { None = 0, Token = 1, Bytes = 2 };

    /** Whether a plan may grow now, or must wait for the current grower. */
    enum class Growth : uint8_t { Granted = 0, WaitForToken = 1 };

    /**
     * One acquisition this plan's in-flight transaction caused.
     *
     * Recorded because a staged block the transaction abandons keeps both its
     * charge and this plan's reference: the arena's abort frees through the
     * allocation callback, which is a no-op for a block the ledger owns, and
     * relinquishing the published-owner claim drops no reference. Without the
     * record there is nothing to roll back by, and the bytes are recoverable
     * only when the whole run retires — which a plan that is going to resume
     * never does.
     *
     * `was_region_backing` marks the one case where the allocator handed back
     * the address its own region is already published at: that claim's
     * reference is this plan's to drop, but the region's ownership is not.
     */
    struct Claim {
        void *base{nullptr};
        uint32_t region_domain{0};
        uint32_t region_index{0};
        uint64_t epoch{0};
        bool was_region_backing{false};
    };

    /** What a plan still owns, and what it is waiting for. */
    struct SlotPlan {
        Phase phase{Phase::None};
        uint64_t epoch{0};
        bool holds_token{false};
        bool plan_stamped{false};
        bool holds_input_staging{false};
        Park parked{Park::None};
        /**
         * The consumer identities this plan may wait for, captured once per
         * stage and never recomputed. Separate per stage because the consumer
         * population genuinely differs between staging inputs and committing
         * regions, and a stage that never waited holds none.
         */
        std::vector<uint64_t> staging_snapshot;
        std::vector<uint64_t> commit_snapshot;
        /**
         * Identities whose retirement this plan has already spent, across both
         * stages. Monotone, so a plan cannot wait twice on one retirement.
         */
        std::vector<uint64_t> consumed;
    };

    AutoAdmissionState() = default;
    AutoAdmissionState(const AutoAdmissionState &) = delete;
    AutoAdmissionState &operator=(const AutoAdmissionState &) = delete;

    /**
     * Turn automatic admission on for `granted` slots, once.
     *
     * Off until a context asks: an explicitly configured context, a borrowed
     * one and a simulated one all leave this holding nothing, and every query
     * below answers as it did before.
     *
     * `budget_bytes` is the byte policy the context's ledger enforces on its
     * arena regions, or zero when it enforces none. It gates the exploratory
     * set only; see the class comment.
     */
    void configure(size_t granted, uint64_t budget_bytes = 0) {
        std::scoped_lock lk(mu_);
        if (enabled_ || granted == 0) return;
        plans_.resize(granted);
        proven_.assign(granted, false);
        proven_cost_.assign(granted, RunResourceCost{});
        budget_bytes_ = budget_bytes;
        enabled_ = true;
    }

    bool enabled() const {
        std::scoped_lock lk(mu_);
        return enabled_;
    }

    size_t slots() const {
        std::scoped_lock lk(mu_);
        return plans_.size();
    }

    // ---------------------------------------------------------------- capacity

    /**
     * Record that a run on `slot` completed its resource commit.
     *
     * The commit is the proof, so this is called on the success path whether or
     * not it allocated anything.
     *
     * `cost` is what the set's commit committed. It is recorded for the
     * exploratory-set test and for reporting, and it never changes a proof.
     */
    void note_proven(uint32_t slot, const RunResourceCost &cost = {}) {
        std::scoped_lock lk(mu_);
        if (slot >= proven_.size()) return;
        proven_[slot] = true;
        proven_cost_[slot] = cost;
    }

    /**
     * Withdraw a slot's proof.
     *
     * Called when the resources that commit established are gone or can no
     * longer be relied on: their storage was reclaimed, their ownership became
     * unprovable, or the first launch on the slot could not take the
     * per-set resources that live outside this accounting. A slot that cannot
     * carry a run must stop being offered.
     */
    void clear_proven(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot >= proven_.size()) return;
        proven_[slot] = false;
        proven_cost_[slot] = RunResourceCost{};
    }

    bool proven(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return slot < proven_.size() && proven_[slot];
    }

    /**
     * Proven slots counted from zero, stopping at the first that is not.
     *
     * A prefix rather than a population count because the lease pool hands out
     * the lowest free slot below the admitted depth: a count of two with slots
     * 0 and 2 proven would offer slot 1, which nothing has demonstrated. A
     * middle slot losing its proof therefore caps what is offered, which is the
     * conservative answer.
     */
    uint32_t proven_prefix() const {
        std::scoped_lock lk(mu_);
        uint32_t prefix = 0;
        while (prefix < proven_.size() && proven_[prefix])
            ++prefix;
        return prefix;
    }

    /**
     * Sets this context can report it is able to hold.
     *
     * The proven prefix plus one exploratory slot, and the exploratory slot is
     * offered only while no growth is in flight — the same exclusivity the
     * token enforces, so the count cannot invite a second simultaneous
     * replacement peak. For enabled contexts with a positive directory limit,
     * never zero and never above that limit. Disabled or empty directories
     * report the supplied limit unchanged.
     *
     * `committed_budgeted_bytes` is what the context's ledger charges right
     * now. With a finite budget, the exploratory slot is additionally withheld
     * when one more set of the largest proven size cannot fit beside that
     * charge. The proven prefix is never withdrawn by the budget: those slots
     * are already committed, and reporting fewer would not return a byte.
     */
    uint32_t affordable_sets(uint32_t directory_limit, uint64_t committed_budgeted_bytes = 0) const {
        std::scoped_lock lk(mu_);
        if (!enabled_ || directory_limit == 0) return directory_limit;
        const uint32_t prefix = proven_prefix_locked();
        uint32_t probe = token_owner_ == kNoOwner ? 1u : 0u;
        if (probe != 0 && budget_bytes_ != 0 && committed_budgeted_bytes + max_proven_budgeted_locked() > budget_bytes_)
            probe = 0;
        const uint32_t offered = prefix > directory_limit - probe ? directory_limit : prefix + probe;
        return std::max<uint32_t>(1u, std::min(offered, directory_limit));
    }

    /** The byte policy the exploratory set is measured against; zero when none. */
    uint64_t budget_bytes() const {
        std::scoped_lock lk(mu_);
        return budget_bytes_;
    }

    bool budget_enforced() const {
        std::scoped_lock lk(mu_);
        return enabled_ && budget_bytes_ != 0;
    }

    /** What the set on `slot` committed when it proved the slot. */
    RunResourceCost proven_cost(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return slot < proven_.size() && proven_[slot] ? proven_cost_[slot] : RunResourceCost{};
    }

    /**
     * The largest footprint among the proven sets.
     *
     * This is the exploratory set's reserve: a new set is estimated at the
     * largest one this context has demonstrated, because a run smaller than
     * that is the case the budget must not refuse and the caller has no
     * declaration of the next run's size to consult. Only its budgeted part is
     * compared with the budget.
     */
    RunResourceCost max_proven_cost() const {
        std::scoped_lock lk(mu_);
        return max_proven_cost_locked();
    }

    // ------------------------------------------------------------------- token

    /**
     * Take the context's growth permission for one plan, or report who holds it.
     *
     * Idempotent for the holder: a plan that grew its inputs and then grows its
     * regions is one growth attempt, so it keeps the token across both rather
     * than releasing and re-taking it — which would let another plan interleave
     * a second peak between the two halves of this one.
     */
    Growth acquire_growth_token(uint32_t slot, uint64_t epoch) {
        std::scoped_lock lk(mu_);
        if (!enabled_ || slot >= plans_.size()) return Growth::Granted;
        SlotPlan &plan = plans_[slot];
        if (token_owner_ == slot && plan.holds_token) return Growth::Granted;
        if (token_owner_ != kNoOwner) return Growth::WaitForToken;
        token_owner_ = slot;
        token_epoch_ = epoch;
        plan.holds_token = true;
        return Growth::Granted;
    }

    void release_growth_token(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size()) return;
        plans_[slot].holds_token = false;
        if (token_owner_ == slot) {
            token_owner_ = kNoOwner;
            token_epoch_ = 0;
        }
    }

    bool token_free() const {
        std::scoped_lock lk(mu_);
        return token_owner_ == kNoOwner;
    }

    bool holds_token(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return slot < plans_.size() && plans_[slot].holds_token;
    }

    // -------------------------------------------------------------------- plan

    /**
     * Open or re-enter a plan on `slot`.
     *
     * A retry is the *same* submission continuing, so its wait set, the
     * identities it has spent and its growth permission all survive — only the
     * attempt's own identity and stage move. That is what bounds the waits: a
     * retry that re-captured a fresh wait set each time would wait once per
     * live consumer for ever, which is the retry loop this design refuses.
     *
     * A plan is ended by `end_plan` — at a completed commit or an abandonment —
     * and only then does the next run on the slot start clean.
     */
    void begin_plan(uint32_t slot, uint64_t epoch, Phase phase) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size()) return;
        SlotPlan &plan = plans_[slot];
        plan.epoch = epoch;
        plan.phase = phase;
    }

    /**
     * The caller has exclusively reserved this slot for a retry. Adopt its
     * existing plan before any fallible preparation, so even an attach failure
     * can abort by the new exact epoch. A fresh slot must stay plan-free.
     */
    void adopt_plan_attempt(uint32_t slot, uint64_t epoch) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size() || plans_[slot].phase == Phase::None) return;
        plans_[slot].epoch = epoch;
        if (token_owner_ == slot) token_epoch_ = epoch;
    }

    void set_phase(uint32_t slot, Phase phase) {
        std::scoped_lock lk(mu_);
        if (slot < plans_.size()) plans_[slot].phase = phase;
    }

    void note_plan_stamped(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot < plans_.size()) plans_[slot].plan_stamped = true;
    }

    void note_input_staging_held(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot < plans_.size()) plans_[slot].holds_input_staging = true;
    }

    /**
     * Record that this attempt's epoch-scoped references have been given back.
     *
     * A wait hands back what belongs to the attempt that could not proceed —
     * its claims and its input-staging reference — because the retry takes its
     * own under its own identity. Keeping them would leave one reference per
     * attempt on a block that is only ever used by one, and a block's
     * reference table is finite. What the wait keeps is the slot's saved plan,
     * which is the slot's rather than any attempt's.
     */
    void note_epoch_refs_released(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot < plans_.size()) plans_[slot].holds_input_staging = false;
    }

    SlotPlan plan(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return slot < plans_.size() ? plans_[slot] : SlotPlan{};
    }

    /** Forget a plan entirely. The caller owns releasing whatever it held. */
    void end_plan(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size()) return;
        if (token_owner_ == slot) {
            token_owner_ = kNoOwner;
            token_epoch_ = 0;
        }
        plans_[slot] = SlotPlan{};
    }

    // --------------------------------------------------------------- waiting

    /**
     * Capture this stage's wait set, once.
     *
     * `holders` is whatever the caller has established can still be returned:
     * consumers that are running now and whose retirement the progress owner
     * will reach. A later call for the same stage is ignored — the set a plan
     * waits on is the one it had when it first could not proceed, so a consumer
     * that launches afterwards neither extends the wait nor rescues it.
     *
     * @return false when the stage already has a snapshot, or when `holders`
     *         offers nothing this plan has not already spent
     */
    bool capture_wait_set(uint32_t slot, Phase phase, const std::vector<uint64_t> &holders) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size()) return false;
        SlotPlan &plan = plans_[slot];
        std::vector<uint64_t> &snapshot = phase == Phase::Staging ? plan.staging_snapshot : plan.commit_snapshot;
        if (!snapshot.empty()) return has_pending_locked(plan, snapshot);
        for (const uint64_t holder : holders) {
            if (holder == 0 || holder == plan.epoch) continue;
            if (contains(plan.consumed, holder)) continue;
            if (contains(snapshot, holder)) continue;
            snapshot.push_back(holder);
        }
        return !snapshot.empty();
    }

    /** Identities of `phase`'s wait set this plan has not yet spent. */
    std::vector<uint64_t> pending_wait_set(uint32_t slot, Phase phase) const {
        std::scoped_lock lk(mu_);
        std::vector<uint64_t> pending;
        if (slot >= plans_.size()) return pending;
        const SlotPlan &plan = plans_[slot];
        const std::vector<uint64_t> &snapshot = phase == Phase::Staging ? plan.staging_snapshot : plan.commit_snapshot;
        for (const uint64_t holder : snapshot) {
            if (!contains(plan.consumed, holder)) pending.push_back(holder);
        }
        return pending;
    }

    void note_parked(uint32_t slot, Park reason) {
        std::scoped_lock lk(mu_);
        if (slot < plans_.size()) plans_[slot].parked = reason;
    }

    Park parked(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return slot < plans_.size() ? plans_[slot].parked : Park::None;
    }

    /**
     * Spend one identity of this stage's wait set whose consumer has finished.
     *
     * `still_holding` answers, for one identity, whether that consumer still
     * holds resources. Only an identity from the captured set clears the wait,
     * so an unrelated run finishing does not wake a plan, and each identity is
     * spent once — which is what bounds the waits by the set's own size rather
     * than by a capacity that can change while the plan waits.
     *
     * @return true when an identity was spent, so the caller may try again
     */
    template <typename StillHolding>
    bool consume_finished_holder(uint32_t slot, Phase phase, StillHolding still_holding) {
        std::scoped_lock lk(mu_);
        if (slot >= plans_.size()) return false;
        SlotPlan &plan = plans_[slot];
        std::vector<uint64_t> &snapshot = phase == Phase::Staging ? plan.staging_snapshot : plan.commit_snapshot;
        for (const uint64_t holder : snapshot) {
            if (contains(plan.consumed, holder)) continue;
            if (still_holding(holder)) continue;
            plan.consumed.push_back(holder);
            plan.parked = Park::None;
            return true;
        }
        return false;
    }

    /** Whether the plan holding the token is one this plan may wait for. */
    bool token_held_by_other(uint32_t slot) const {
        std::scoped_lock lk(mu_);
        return token_owner_ != kNoOwner && token_owner_ != slot;
    }

    // ------------------------------------------------------------------ claims

    // The prepare transaction owns its rollback ledger through exception cleanup.
    // TLS only selects the live scope; it never owns a vector or its capacity.
    class ClaimScope {
    public:
        ClaimScope() :
            previous_(active_claims_.get()) {
            active_claims_.set(&claims_);
        }
        ~ClaimScope() { active_claims_.restore(previous_); }
        ClaimScope(const ClaimScope &) = delete;
        ClaimScope &operator=(const ClaimScope &) = delete;

    private:
        std::vector<Claim> claims_;
        std::vector<Claim> *previous_;
    };
    static bool has_claim_scope() noexcept { return active_claims_.get() != nullptr; }
    static void record_claim(const Claim &claim) { claims().push_back(claim); }
    static std::vector<Claim> &claims() {
        if (active_claims_.get() == nullptr) throw std::logic_error("workspace claim without prepare scope");
        return *active_claims_.get();
    }
    // Ownership moves to the run; clearing the ledger does not release its blocks.
    static void commit_claims() {
        if (auto *claims = active_claims_.get()) claims->clear();
    }

private:
    inline static __attribute__((visibility("hidden"))) simpler::ModuleThreadValue<std::vector<Claim> *> active_claims_;
    static constexpr uint32_t kNoOwner = 0xFFFFFFFFu;

    static bool contains(const std::vector<uint64_t> &values, uint64_t value) {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    bool has_pending_locked(const SlotPlan &plan, const std::vector<uint64_t> &snapshot) const {
        for (const uint64_t holder : snapshot) {
            if (!contains(plan.consumed, holder)) return true;
        }
        return false;
    }

    uint32_t proven_prefix_locked() const {
        uint32_t prefix = 0;
        while (prefix < proven_.size() && proven_[prefix])
            ++prefix;
        return prefix;
    }

    RunResourceCost max_proven_cost_locked() const {
        RunResourceCost largest{};
        for (size_t slot = 0; slot < proven_.size(); ++slot) {
            if (!proven_[slot]) continue;
            const RunResourceCost &cost = proven_cost_[slot];
            if (cost.total_bytes() > largest.total_bytes()) largest = cost;
        }
        return largest;
    }

    uint64_t max_proven_budgeted_locked() const {
        uint64_t largest = 0;
        for (size_t slot = 0; slot < proven_.size(); ++slot) {
            if (!proven_[slot]) continue;
            largest = std::max(largest, proven_cost_[slot].budgeted_bytes());
        }
        return largest;
    }

    mutable std::mutex mu_;
    bool enabled_{false};
    uint64_t budget_bytes_{0};
    uint32_t token_owner_{kNoOwner};
    uint64_t token_epoch_{0};
    std::vector<SlotPlan> plans_;
    std::vector<bool> proven_;
    std::vector<RunResourceCost> proven_cost_;
};
