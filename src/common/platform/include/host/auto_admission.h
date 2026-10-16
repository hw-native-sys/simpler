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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

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
 * Capacity is a count of identities, never a byte budget. This class holds no
 * byte figure and makes no allocation decision; whether a request fits is the
 * allocator's answer, reported back here only as proof or as a refusal.
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
     */
    void configure(size_t granted) {
        std::scoped_lock lk(mu_);
        if (enabled_ || granted == 0) return;
        plans_.resize(granted);
        proven_.assign(granted, false);
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
     */
    void note_proven(uint32_t slot) {
        std::scoped_lock lk(mu_);
        if (slot < proven_.size()) proven_[slot] = true;
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
        if (slot < proven_.size()) proven_[slot] = false;
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
     * replacement peak. Never zero: a context always holds the run it is
     * executing. Never above the identity directory's own limit.
     */
    uint32_t affordable_sets(uint32_t directory_limit) const {
        std::scoped_lock lk(mu_);
        if (!enabled_ || directory_limit == 0) return directory_limit;
        const uint32_t prefix = proven_prefix_locked();
        const uint32_t probe = token_owner_ == kNoOwner ? 1u : 0u;
        const uint32_t offered = prefix > directory_limit - probe ? directory_limit : prefix + probe;
        return std::max<uint32_t>(1u, std::min(offered, directory_limit));
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

    /**
     * Record one acquisition of the transaction in flight on this thread.
     *
     * Thread-local rather than per slot: a successor prepared on another thread
     * runs its own transaction, and charging its claims to this one would roll
     * back a block the other plan is still staging.
     */
    static void record_claim(const Claim &claim) { claims().push_back(claim); }

    static std::vector<Claim> &claims() {
        static thread_local std::vector<Claim> live;
        return live;
    }

    /** Adopt this transaction's claims as live references and stop tracking them. */
    static void commit_claims() { claims().clear(); }

private:
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

    mutable std::mutex mu_;
    bool enabled_{false};
    uint32_t token_owner_{kNoOwner};
    uint64_t token_epoch_{0};
    std::vector<SlotPlan> plans_;
    std::vector<bool> proven_;
};
