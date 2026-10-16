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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "host/run_resource_growth.h"
#include "runtime_c_api.h"

namespace {

using Phase = AutoAdmissionState::Phase;

constexpr std::size_t kRegionCount = static_cast<std::size_t>(WorkspaceManager::ArenaRegion::kCount);
constexpr std::size_t kHeap = static_cast<std::size_t>(WorkspaceManager::ArenaRegion::GmHeap);

/**
 * A device allocator with a finite extent, so a refusal is reproducible.
 *
 * The pressure this feature answers is the allocator's own: it returns nothing
 * and names no figure, which is why nothing above it queries free capacity. A
 * unit test cannot exhaust real HBM, so the extent stands in for it — and no
 * ledger byte limit is set, because the default managed context enforces none
 * and the ledger's own quota path would otherwise reclaim and hide the point.
 */
class BoundedBackend {
public:
    explicit BoundedBackend(std::uint64_t capacity) :
        capacity_(capacity) {}

    WorkspaceManager::Backend ops() {
        WorkspaceManager::Backend b;
        b.ctx = this;
        b.acquire = [](void *ctx, std::size_t bytes) -> void * {
            return static_cast<BoundedBackend *>(ctx)->acquire(bytes);
        };
        b.release = [](void *ctx, void *base, int *platform_rc) {
            return static_cast<BoundedBackend *>(ctx)->release(base, platform_rc);
        };
        return b;
    }

    void *acquire(std::size_t bytes) {
        ++acquire_calls;
        if (used_ + bytes > capacity_) {
            ++refusals;
            return nullptr;
        }
        next_ += 0x1000;
        void *base = reinterpret_cast<void *>(next_);
        live[base] = bytes;
        used_ += bytes;
        return base;
    }

    WorkspaceManager::ReleaseOutcome release(void *base, int *platform_rc) {
        released.push_back(base);
        if (before_release != nullptr) before_release(base);
        if (fail_release_of == base) {
            if (platform_rc != nullptr) *platform_rc = -7;
            return WorkspaceManager::ReleaseOutcome::FreeFailed;
        }
        if (keep_mapped_of == base) {
            if (platform_rc != nullptr) *platform_rc = -9;
            return WorkspaceManager::ReleaseOutcome::StillMapped;
        }
        auto it = live.find(base);
        if (it != live.end()) {
            used_ -= it->second;
            live.erase(it);
        }
        return WorkspaceManager::ReleaseOutcome::Freed;
    }

    std::uint64_t free_bytes() const { return capacity_ - used_; }

    void *fail_release_of{nullptr};
    void *keep_mapped_of{nullptr};
    // Observed at the instant of the platform free, which is where the
    // detach-before-free ordering is either true or not.
    void (*before_release)(void *base){nullptr};
    int acquire_calls{0};
    int refusals{0};
    std::vector<void *> released;
    std::map<void *, std::size_t> live;

private:
    std::uint64_t capacity_;
    std::uint64_t used_{0};
    std::uintptr_t next_{0x200000};
};

/**
 * One endpoint's run-resource sets, driven through the production coordinator.
 *
 * `RunResourceGrowth` is the same object the onboard runner constructs, from
 * the same translation unit the onboard host runtime links
 * (`shared/host/run_resource_growth.cpp`), and this class implements the same
 * `RunResourceEndpoint` the runner's own view implements. So every decision
 * exercised below — which sets are trimmable, in what order they are detached,
 * how many attempts a refusal is allowed, whether a plan parks or fails, and
 * which holders it may wait for — is production code, not a copy of it.
 *
 * What stays on this side is what the runner alone can answer: which slots and
 * banks live runs hold, and the two allocation attempts themselves. The ledger,
 * the arenas, the replacement transaction, the claim rollback and the admission
 * state are all the production implementations.
 *
 * Not covered here: the runner's own member lookups, whose translation unit
 * needs the device SDK and is not linkable in this suite.
 */
class Endpoint final : public RunResourceEndpoint {
public:
    Endpoint(std::size_t sets, std::uint64_t capacity, bool sizes_itself = true, std::size_t base_align = 1) :
        backend(capacity),
        sets_(sets),
        base_align_(base_align) {
        AutoAdmissionState::claims().clear();
        ledger.configure(backend.ops());
        ledger.commit(WorkspaceManager::build(sets));
        // An endpoint a caller configured never asks, which is what leaves the
        // admission state holding nothing and every decision below inert.
        if (sizes_itself) admission.configure(sets);
        staging_.assign(sets, nullptr);
        reserved_slot_.assign(sets, false);
        reserved_bank_.assign(sets, false);
        announces_.resize(sets);
        banks_.reserve(sets);
        for (std::size_t i = 0; i < sets; ++i) {
            banks_.push_back(std::make_unique<Bank>(this));
            announces_[i] = {this, i};
        }
    }

    // ------------------------------------------------- RunResourceEndpoint

    std::size_t run_resource_sets() const override { return sets_; }
    bool slot_reserved(uint32_t slot) const override { return reserved_slot_[slot]; }
    bool bank_reserved(uint32_t slot) const override { return reserved_bank_[slot]; }
    bool plan_live(uint32_t slot) const override { return admission.plan(slot).phase != Phase::None; }
    void *staging_backing(uint32_t slot) const override { return staging_[slot]; }
    void release_staging_backing(uint32_t slot) override { staging_[slot] = nullptr; }
    bool owns_prebuilt_cache(uint32_t /*slot*/) const override { return false; }
    PrebuiltRuntimeArenaCache *prebuilt_cache(uint32_t /*slot*/) override { return nullptr; }
    void withdraw_run_resource_proof(uint32_t slot) override { admission.clear_proven(slot); }
    int reclaim_obsolete() override { return ledger.reclaim_obsolete(); }

    std::size_t arena_release_requests(uint32_t slot, ArenaRegionRequest *out, std::size_t capacity) override {
        if (out == nullptr || capacity < kRegionCount) return 0;
        const std::size_t zero[kRegionCount] = {0, 0, 0};
        const std::array<ArenaRegionRequest, kRegionCount> built = bank_requests(slot, zero);
        for (std::size_t i = 0; i < kRegionCount; ++i) {
            out[i] = built[i];
        }
        return kRegionCount;
    }

    // ---------------------------------------------------------- bookkeeping

    void *staging(std::size_t slot) const { return staging_[slot]; }

    void reserve(std::size_t slot, std::size_t bank) {
        reserved_slot_[slot] = true;
        reserved_bank_[bank] = true;
    }
    void release_reservation(std::size_t slot, std::size_t bank) {
        reserved_slot_[slot] = false;
        reserved_bank_[bank] = false;
    }

    /** The one fact a holder needs before it is worth waiting for. */
    void report_launched(std::size_t slot, std::uint64_t epoch) {
        ledger.note_run_fact(static_cast<uint32_t>(slot), epoch, WorkspaceManager::RunFact::Launched);
    }

    /** Every fact a run reports as its phases produce them. */
    void retire(std::size_t slot, std::uint64_t epoch) {
        const uint32_t s = static_cast<uint32_t>(slot);
        ledger.note_run_fact(s, epoch, WorkspaceManager::RunFact::Launched);
        ledger.note_run_fact(s, epoch, WorkspaceManager::RunFact::DrainAttempted);
        ledger.note_run_fact(s, epoch, WorkspaceManager::RunFact::DrainProvedComplete);
        ledger.note_run_fact(s, epoch, WorkspaceManager::RunFact::CopybackReturned);
        ledger.note_run_fact(s, epoch, WorkspaceManager::RunFact::BindingsReleased);
    }

    // ------------------------------------------------------------ the paths

    /** `DeviceRunnerBase::acquire_retained_temp`'s managed branch. */
    int grow_staging(std::size_t slot, std::uint64_t epoch, std::size_t bytes) {
        const uint32_t s = static_cast<uint32_t>(slot);
        RunResourceGrowth growth(ledger, admission, *this);
        const int permission = growth.claim_permission(s, epoch, Phase::Staging);
        if (permission != 0) return permission;
        plan_epoch_ = epoch;
        plan_region_ = WorkspaceManager::staging_region(s);
        StagingAttempt attempt{this, plan_region_, bytes};
        const GrowthAttempt outcome = growth.attempt_with_one_retry(
            s, /*rollback_before_retry=*/false,
            [](void *ctx) {
                auto *a = static_cast<StagingAttempt *>(ctx);
                WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
                a->grown = a->owner->ledger.acquire(a->region, a->owner->plan_epoch_, a->bytes, &why);
                if (a->grown != nullptr) return GrowthAttempt::Served;
                return classify_growth_refusal(why);
            },
            &attempt
        );
        if (attempt.grown == nullptr) {
            const int refusal = growth.report_refusal(s, Phase::Staging, bytes, outcome);
            return refusal != 0 ? refusal : PTO_RUNTIME_ERR_INTERNAL;
        }
        staging_[slot] = attempt.grown;
        ledger.note_published(plan_region_, attempt.grown);
        admission.note_input_staging_held(s);
        return 0;
    }

    // The toy byte budgets describe payloads. Alignment one makes raw ledger
    // charges equal those payloads; production retains its device alignment.
    /** `DeviceRunnerBase::setup_static_arena`: the whole bank staged together. */
    int
    commit_arena(std::size_t slot, std::size_t bank, std::uint64_t epoch, const std::size_t (&sizes)[kRegionCount]) {
        const uint32_t s = static_cast<uint32_t>(slot);
        RunResourceGrowth growth(ledger, admission, *this);
        std::array<ArenaRegionRequest, kRegionCount> requests = bank_requests(bank, sizes);
        bool needs_growth = false;
        for (std::size_t i = 0; i < kRegionCount; ++i) {
            if (arena_region_action(requests[i]) == ArenaRegionAction::Replace) needs_growth = true;
        }
        if (needs_growth) {
            const int permission = growth.claim_permission(s, epoch, Phase::Committing);
            if (permission != 0) return permission;
        }
        plan_epoch_ = epoch;
        ArenaAttempt attempt{this, requests.data(), bank};
        const GrowthAttempt outcome = growth.attempt_with_one_retry(
            s, /*rollback_before_retry=*/true,
            [](void *ctx) {
                auto *a = static_cast<ArenaAttempt *>(ctx);
                a->owner->last_refusal_ = WorkspaceManager::AcquireRefusal::None;
                a->outcome = run_bank_arena_setup(a->requests, kRegionCount, false, nullptr, a->owner->base_align_);
                if (a->outcome.rc == 0) return GrowthAttempt::Served;
                return classify_growth_refusal(a->owner->last_refusal_);
            },
            &attempt
        );
        if (attempt.outcome.rc != 0) {
            const int refusal = growth.report_refusal(s, Phase::Committing, sizes[kHeap], outcome);
            return refusal != 0 ? refusal : PTO_RUNTIME_ERR_INTERNAL;
        }
        // The transaction already published its replacements. A later failure
        // while attaching consumers cannot roll them back as staged claims.
        AutoAdmissionState::commit_claims();
        if (refuse_reference) return PTO_RUNTIME_ERR_INTERNAL;
        for (std::size_t i = 0; i < kRegionCount; ++i) {
            if (!requests[i].arena->is_committed()) continue;
            void *owned = requests[i].arena->raw_backing();
            EXPECT_TRUE(ledger.reference(owned, epoch));
            ledger.note_published(
                WorkspaceManager::arena_region(
                    static_cast<uint32_t>(bank), static_cast<WorkspaceManager::ArenaRegion>(i)
                ),
                owned
            );
        }
        AutoAdmissionState::commit_claims();
        return 0;
    }

    /**
     * The prewarm path: a bank committed outside any run plan, so every request
     * belongs to the context rather than to a run.
     */
    int prewarm_arena(std::size_t bank, const std::size_t (&sizes)[kRegionCount]) {
        plan_epoch_ = WorkspaceManager::kContextEpoch;
        std::array<ArenaRegionRequest, kRegionCount> requests = bank_requests(bank, sizes);
        const BankArenaSetupOutcome outcome =
            run_bank_arena_setup(requests.data(), kRegionCount, false, nullptr, base_align_);
        if (outcome.rc != 0) return outcome.rc;
        for (std::size_t i = 0; i < kRegionCount; ++i) {
            if (!requests[i].arena->is_committed()) continue;
            ledger.note_published(
                WorkspaceManager::arena_region(
                    static_cast<uint32_t>(bank), static_cast<WorkspaceManager::ArenaRegion>(i)
                ),
                requests[i].arena->raw_backing()
            );
        }
        AutoAdmissionState::commit_claims();
        return 0;
    }

    /** `DeviceRunnerBase::note_run_resources_proven`. */
    void note_proven(std::size_t slot, std::uint64_t epoch) {
        const uint32_t s = static_cast<uint32_t>(slot);
        if (admission.plan(s).epoch == epoch) admission.release_growth_token(s);
        admission.note_proven(s);
        AutoAdmissionState::commit_claims();
        admission.end_plan(s);
    }

    std::size_t trim(uint32_t requesting_slot) {
        return RunResourceGrowth(ledger, admission, *this).trim_idle_sets(requesting_slot);
    }

    bool still_blocked(std::size_t slot, Phase phase) {
        return RunResourceGrowth(ledger, admission, *this).still_blocked(static_cast<uint32_t>(slot), phase);
    }

    /** The committed backing of one region, as the ledger knows it. */
    void *region_backing(std::size_t bank, std::size_t region) const {
        return banks_[bank]->region(region).raw_backing();
    }

    WorkspaceManager ledger;
    AutoAdmissionState admission;
    BoundedBackend backend;
    bool refuse_reference{false};

private:
    /** One bank's three arenas, allocating through the ledger. */
    struct Bank {
        explicit Bank(Endpoint *owner) :
            gm_heap(&alloc, &free_backing, owner),
            gm_sm(&alloc, &free_backing, owner),
            runtime_pool(&alloc, &free_backing, owner) {}

        DeviceArena &region(std::size_t index) { return index == 0 ? gm_heap : (index == 1 ? gm_sm : runtime_pool); }
        const DeviceArena &region(std::size_t index) const {
            return index == 0 ? gm_heap : (index == 1 ? gm_sm : runtime_pool);
        }

        DeviceArena gm_heap;
        DeviceArena gm_sm;
        DeviceArena runtime_pool;
        std::size_t cached[kRegionCount]{0, 0, 0};
    };

    struct StagingAttempt {
        Endpoint *owner;
        WorkspaceManager::RegionKey region;
        std::size_t bytes;
        void *grown{nullptr};
    };

    struct ArenaAttempt {
        Endpoint *owner;
        ArenaRegionRequest *requests;
        std::size_t bank;
        BankArenaSetupOutcome outcome{};
    };

    // `DeviceRunnerBase::acquire_arena_backing`: the region is named by the
    // announce, because the callback's own arguments carry only a byte count.
    static void *alloc(void *ctx, std::size_t size) {
        auto *self = static_cast<Endpoint *>(ctx);
        WorkspaceManager::AcquireRefusal why = WorkspaceManager::AcquireRefusal::None;
        void *base = self->ledger.acquire(self->plan_region_, self->plan_epoch_, size, &why);
        if (base == nullptr) {
            self->last_refusal_ = why;
            return nullptr;
        }
        AutoAdmissionState::Claim claim;
        claim.base = base;
        claim.epoch = self->plan_epoch_;
        claim.was_region_backing = base == self->plan_region_backing_;
        AutoAdmissionState::record_claim(claim);
        return base;
    }

    // `DeviceRunnerBase::release_arena_backing`: a managed block belongs to the
    // ledger, so the arena dropping its base is not permission to free.
    static void free_backing(void *ctx, void *p) {
        auto *self = static_cast<Endpoint *>(ctx);
        EXPECT_TRUE(self->ledger.owns(p)) << "an unmanaged block reached the managed path";
    }

    static void announce_region(void *ctx, std::size_t region_index) {
        const auto &announce = *static_cast<BankAnnounce *>(ctx);
        auto *self = announce.endpoint;
        self->plan_region_ = WorkspaceManager::arena_region(
            static_cast<uint32_t>(announce.bank), static_cast<WorkspaceManager::ArenaRegion>(region_index)
        );
        self->plan_region_backing_ = self->banks_[announce.bank]->region(region_index).raw_backing();
    }

    // `note_arena_region_disposition`: the region stops publishing the block,
    // and the ledger — not the arena — owns whether its bytes may go back.
    static void region_gave_up(void *ctx, std::size_t region_index, ArenaRegionDisposition what, void *base) {
        auto *self = static_cast<Endpoint *>(ctx);
        (void)region_index;
        (void)what;
        (void)self->ledger.note_unpublished(base);
    }

    std::array<ArenaRegionRequest, kRegionCount>
    bank_requests(std::size_t bank, const std::size_t (&sizes)[kRegionCount]) {
        Bank &b = *banks_[bank];
        static constexpr const char *kNames[kRegionCount] = {"gm_heap", "gm_sm", "runtime_pool"};
        std::array<ArenaRegionRequest, kRegionCount> requests{};
        for (std::size_t i = 0; i < kRegionCount; ++i) {
            requests[i].arena = &b.region(i);
            requests[i].cached_size = &b.cached[i];
            requests[i].requested_size = sizes[i];
            requests[i].name = kNames[i];
            requests[i].announce = &announce_region;
            requests[i].announce_ctx = &announces_[bank];
            requests[i].disposition = &region_gave_up;
            requests[i].disposition_ctx = this;
        }
        return requests;
    }

    std::size_t sets_;
    std::size_t base_align_;
    std::vector<void *> staging_;
    std::vector<bool> reserved_slot_;
    std::vector<bool> reserved_bank_;
    std::vector<std::unique_ptr<Bank>> banks_;
    struct BankAnnounce {
        Endpoint *endpoint;
        std::size_t bank;
    };
    std::vector<BankAnnounce> announces_;
    WorkspaceManager::RegionKey plan_region_{};
    void *plan_region_backing_{nullptr};
    std::uint64_t plan_epoch_{0};
    WorkspaceManager::AcquireRefusal last_refusal_{WorkspaceManager::AcquireRefusal::None};
};

// A set whose arena nothing can reach is detached, the proof that rested on it
// is withdrawn, and only then is the storage freed. Without this the request
// below fails beside bytes it could have had: `reclaim_obsolete` keeps a set's
// idle current backing on purpose, so no reclamation would ever offer them.
TEST(RunResourceGrowthTrim, AnIdleSetsArenaIsGivenBackSoTheNextRequestFits) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    const std::size_t twenty[kRegionCount] = {20, 0, 0};

    // Slot 2 ran once and finished: 30 bytes of cached arena, no consumer left.
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);

    // Slot 0 is executing and holds 20.
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);

    ASSERT_EQ(e.backend.free_bytes(), 10u);
    void *idle = e.region_backing(2, kHeap);
    ASSERT_NE(idle, nullptr);
    EXPECT_TRUE(e.ledger.idle_backing(idle));
    EXPECT_EQ(e.ledger.reclaim_obsolete(), 0);
    EXPECT_TRUE(e.backend.released.empty()) << "a reclaim alone never takes a set's current backing";

    // Slot 1 asks for 40, which neither fits in the 10 that is free nor can be
    // reclaimed. The trim converts slot 2's 30, and the one retry succeeds.
    e.reserve(1, 1);
    const std::size_t forty[kRegionCount] = {40, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 90, forty), 0);
    EXPECT_EQ(e.backend.refusals, 1) << "one refusal, then one retry that fitted";
    EXPECT_EQ(e.backend.free_bytes(), 0u);
    EXPECT_FALSE(e.ledger.owns(idle));
    EXPECT_FALSE(e.admission.proven(2)) << "the set was admitted on resources it no longer holds";
    void *candidate = e.region_backing(1, kHeap);
    e.retire(1, 90);
    e.release_reservation(1, 1);
    EXPECT_EQ(e.ledger.reclaim_obsolete(), 0);
    EXPECT_TRUE(e.ledger.owns(candidate)) << "published generation remains current after its consumer retires";
    const auto region = WorkspaceManager::arena_region(1, WorkspaceManager::ArenaRegion::GmHeap);
    EXPECT_EQ(e.ledger.acquire(region, 91, 40), candidate)
        << "retry belongs to bank 1, not the last bank inspected by trim";
}

// The 40-needed / 30-idle / 20-held sequence, where the trim alone is not
// enough: the plan parks on a named launched consumer, that consumer retires,
// and the re-activation succeeds. The `20 < 45` comparison never happens,
// because free capacity is never queried.
TEST(RunResourceGrowthWait, ARefusedPlanWaitsForALaunchedHolderAndThenFits) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    const std::size_t twenty[kRegionCount] = {20, 0, 0};

    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);

    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);
    ASSERT_EQ(e.backend.free_bytes(), 10u);

    // 45 does not fit in 10, nor in the 40 the trim of slot 2 produces.
    e.reserve(1, 1);
    const std::size_t forty_five[kRegionCount] = {45, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 90, forty_five), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(e.admission.pending_wait_set(1, Phase::Committing).size(), 1u)
        << "the launched holder on slot 0 is the one thing to wait for";
    EXPECT_EQ(e.backend.free_bytes(), 40u) << "the idle set came back even though it was not enough";
    EXPECT_TRUE(e.still_blocked(1, Phase::Committing)) << "slot 0 is still holding";

    // Slot 0 finishes. Its references drop, which is what the wait was for.
    e.retire(0, 70);
    e.release_reservation(0, 0);
    EXPECT_FALSE(e.still_blocked(1, Phase::Committing));

    // One re-activation, paid for by one real retirement: the holder's set is
    // now idle, so the trim takes it and the retry fits.
    EXPECT_EQ(e.commit_arena(1, 1, 91, forty_five), 0);
    EXPECT_FALSE(e.admission.proven(0)) << "slot 0's set was trimmed to serve this one";
    EXPECT_EQ(e.admission.affordable_sets(4), 1u) << "nothing is proven but the set being committed";
}

// The staging half of the same two steps, and the ordering asserted where it
// matters: at the instant of the platform free, the slot must already have
// stopped naming the address.
TEST(RunResourceGrowthTrim, AnIdleStagingBlockIsDetachedBeforeItIsFreed) {
    static Endpoint *observed = nullptr;
    static void *detached = nullptr;
    Endpoint e(4, /*capacity=*/50);
    observed = &e;
    detached = nullptr;
    e.backend.before_release = [](void *base) {
        if (base != detached) return;
        EXPECT_EQ(observed->staging(2), nullptr) << "the slot still named the block the platform is taking back";
    };

    e.reserve(2, 2);
    ASSERT_EQ(e.grow_staging(2, 72, 30), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);
    detached = e.staging(2);
    ASSERT_NE(detached, nullptr);

    e.reserve(1, 1);
    ASSERT_EQ(e.grow_staging(1, 90, 40), 0);
    EXPECT_EQ(e.staging(2), nullptr);
    EXPECT_FALSE(e.ledger.owns(detached));
    EXPECT_FALSE(e.admission.proven(2));
    EXPECT_EQ(e.backend.refusals, 1);
    observed = nullptr;
}

// Staging under pressure with nothing to trim: the plan parks on the launched
// holder and succeeds once that holder's set has been given back.
TEST(RunResourceGrowthWait, AStagingRefusalWaitsForALaunchedHolderAndThenFits) {
    Endpoint e(4, /*capacity=*/40);
    e.reserve(0, 0);
    ASSERT_EQ(e.grow_staging(0, 70, 25), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);

    e.reserve(1, 1);
    EXPECT_EQ(e.grow_staging(1, 90, 30), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(e.admission.pending_wait_set(1, Phase::Staging).size(), 1u);

    e.retire(0, 70);
    e.release_reservation(0, 0);
    ASSERT_FALSE(e.still_blocked(1, Phase::Staging));
    EXPECT_EQ(e.grow_staging(1, 91, 30), 0);
    EXPECT_FALSE(e.admission.proven(0));
}

// F2's shape on its own: the holder occupies another set, so its region keys
// are never the candidate's. A candidate whose own keys decide the wait set
// finds nothing — a set being offered for the first time has no prior holder of
// its own keys at all — and fails where it should have waited.
TEST(RunResourceGrowthWait, AHolderOnAnotherSetIsFoundThoughItsRegionKeysDiffer) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);

    // The holder's blocks are keyed to bank 0; the candidate asks about bank 1,
    // and has never held anything of its own.
    std::vector<std::uint64_t> holders;
    e.ledger.eligible_holders(/*requesting_epoch=*/90, &holders);
    ASSERT_EQ(holders.size(), 1u);
    EXPECT_EQ(holders[0], 70u);

    e.reserve(1, 1);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 90, thirty), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT)
        << "the predecessor is the named releaser";
    EXPECT_EQ(e.admission.pending_wait_set(1, Phase::Committing).size(), 1u);
}

// Launched is what makes a holder worth waiting for. A run that has only
// prepared is queued behind the same grower as the caller, so it cannot retire
// while the caller waits and naming it would be a wait nothing can end.
TEST(RunResourceGrowthWait, APreparedButUnlaunchedHolderIsNotSomethingToWaitFor) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    // No Launched fact: this run has committed its resources and nothing else.

    std::vector<std::uint64_t> holders;
    e.ledger.eligible_holders(90, &holders);
    EXPECT_TRUE(holders.empty());

    e.reserve(1, 1);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 90, thirty), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
}

TEST(RunResourceGrowthWait, TheAskingRunIsNeverItsOwnHolder) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.report_launched(0, 70);

    std::vector<std::uint64_t> holders;
    e.ledger.eligible_holders(/*requesting_epoch=*/70, &holders);
    EXPECT_TRUE(holders.empty()) << "a plan that waited for itself would never be released";
}

TEST(RunResourceGrowthWait, ARetiredHolderIsAlreadyDischarged) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.retire(0, 70);

    std::vector<std::uint64_t> holders;
    e.ledger.eligible_holders(90, &holders);
    EXPECT_TRUE(holders.empty());
}

// Once the captured set is spent and the request still cannot be served, the
// plan takes the typed failure rather than waiting on a consumer that launched
// since — the set is immutable, so it is exhausted rather than extended.
TEST(RunResourceGrowthWait, AnExhaustedWaitSetFailsRatherThanWaitingAgain) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);

    e.reserve(1, 1);
    const std::size_t too_much[kRegionCount] = {40, 0, 0};
    ASSERT_EQ(e.commit_arena(1, 1, 90, too_much), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    ASSERT_EQ(e.admission.pending_wait_set(1, Phase::Committing).size(), 1u);

    e.retire(0, 70);
    e.release_reservation(0, 0);
    ASSERT_FALSE(e.still_blocked(1, Phase::Committing));
    // One byte more than the whole device: the retry refuses and the captured
    // set has nothing left to spend.
    const std::size_t impossible[kRegionCount] = {41, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 91, impossible), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
    EXPECT_TRUE(e.admission.pending_wait_set(1, Phase::Committing).empty());
}

// A quarantined block's storage never comes back, so its holder is not a
// releaser and its backing is not trimmable. This is the stop-unknown case the
// ledger can see: a context destroyed before its run's facts completed.
TEST(RunResourceGrowthTrim, QuarantinedBackingIsNeitherAHolderNorTrimmable) {
    Endpoint e(4, /*capacity=*/40);
    const std::size_t twenty[kRegionCount] = {20, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, twenty), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);
    void *held = e.region_backing(0, kHeap);
    e.release_reservation(0, 0);
    e.ledger.note_run_fact(0, 70, WorkspaceManager::RunFact::ContextDestroyed);

    EXPECT_EQ(e.ledger.block_state(held), WorkspaceManager::BlockState::Quarantined);
    EXPECT_FALSE(e.ledger.idle_backing(held));
    EXPECT_EQ(e.ledger.detach_idle_backing(held), 0u);
    std::vector<std::uint64_t> holders;
    e.ledger.eligible_holders(90, &holders);
    EXPECT_TRUE(holders.empty());

    EXPECT_EQ(e.trim(1), 0u);
    EXPECT_TRUE(e.backend.released.empty());
}

// A release that failed leaves ownership of the allocation unknown, so the
// block is not offered again by any path.
TEST(RunResourceGrowthTrim, ReleaseUnconfirmedBackingIsNotTrimmedAgain) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);

    void *idle = e.region_backing(2, kHeap);
    e.backend.fail_release_of = idle;
    ASSERT_EQ(e.trim(1), 1u);
    EXPECT_NE(e.ledger.reclaim_obsolete(), 0);
    EXPECT_EQ(e.ledger.block_state(idle), WorkspaceManager::BlockState::ReleaseUnconfirmed);
    EXPECT_FALSE(e.ledger.idle_backing(idle));
    EXPECT_EQ(e.ledger.detach_idle_backing(idle), 0u);
}

// A failed free degrades the ledger, and a degraded ledger publishes nothing
// further — so the growth that follows is a failure to report rather than a
// wait, however many launched holders exist.
TEST(RunResourceGrowthWait, ADegradedLedgerFailsInsteadOfWaiting) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    const std::size_t ten[kRegionCount] = {10, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);

    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, ten), 0);
    e.note_proven(0, 70);
    e.report_launched(0, 70);

    // The one block the trim can take is the one whose free will fail.
    e.backend.fail_release_of = e.region_backing(2, kHeap);
    e.reserve(1, 1);
    const std::size_t too_much[kRegionCount] = {40, 0, 0};
    EXPECT_EQ(e.commit_arena(1, 1, 90, too_much), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(e.admission.pending_wait_set(1, Phase::Committing).empty())
        << "a degraded ledger offers nothing to wait for";
}

// A block whose host mapping could not be dropped cannot be handed to another
// allocation at all, so it is excluded from the trim and from the wait set.
TEST(RunResourceGrowthTrim, StillMappedBackingIsExcluded) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);
    void *idle = e.region_backing(2, kHeap);
    ASSERT_TRUE(e.ledger.note_mapping_unregister_failed(idle));

    EXPECT_FALSE(e.ledger.idle_backing(idle));
    EXPECT_EQ(e.trim(1), 0u);
}

// Busy is decided by what holds a set, never by what its blocks look like: a
// live run's set is unreferenced only in the instant between its facts, and
// trimming it would take storage out from under executing work.
TEST(RunResourceGrowthTrim, AReservedSetIsNeverTrimmedHoweverIdleItsBlocksLook) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);  // references dropped, but the reservation is still held

    void *idle = e.region_backing(2, kHeap);
    EXPECT_TRUE(e.ledger.idle_backing(idle)) << "the ledger sees no consumer";
    EXPECT_EQ(e.trim(1), 0u) << "its owner still holds it";
    EXPECT_TRUE(e.ledger.owns(idle));
}

// The requesting set is excluded by name, so a refusal still leaves it naming
// everything it had.
TEST(RunResourceGrowthTrim, TheRequestingSetIsExcludedFromItsOwnTrim) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    ASSERT_EQ(e.commit_arena(1, 1, 90, thirty), 0);
    e.note_proven(1, 90);
    e.retire(1, 90);

    void *own = e.region_backing(1, kHeap);
    EXPECT_EQ(e.trim(1), 0u);
    EXPECT_TRUE(e.ledger.owns(own));
    EXPECT_TRUE(e.admission.proven(1));
}

// A set a saved plan would resume into keeps its storage, and — because the
// bank that plan was committing into cannot be named from a slot holding no
// reservation — so does every other bank. Its own slot's staging is still the
// plan's and is excluded on its own terms.
TEST(RunResourceGrowthTrim, AParkedPlanWithdrawsTheArenaTrimForEveryBank) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.note_proven(2, 72);
    e.retire(2, 72);
    e.release_reservation(2, 2);

    // Slot 3 holds a plan and no reservation: a parked one.
    e.admission.begin_plan(3, 99, Phase::Committing);
    ASSERT_TRUE(e.admission.plan(3).phase != Phase::None);

    void *idle = e.region_backing(2, kHeap);
    EXPECT_TRUE(e.ledger.idle_backing(idle));
    EXPECT_EQ(e.trim(1), 0u) << "the parked plan's bank cannot be identified, so no bank is taken";
    EXPECT_TRUE(e.ledger.owns(idle));
}

// A bank another run reserved is excluded even when this set's own slot is
// free, which is what keeps a shared arena — every slot resolving to one bank —
// out of the trim entirely while any run is live.
TEST(RunResourceGrowthTrim, ABankAnotherRunReservedIsExcludedFromThisSetsTrim) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    // Bank 0 commits under slot 0, then slot 1's run takes bank 0 too — the
    // shape a runtime whose arena is shared produces.
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, thirty), 0);
    e.note_proven(0, 70);
    e.retire(0, 70);
    e.release_reservation(0, 0);
    e.reserve(1, 0);

    void *shared = e.region_backing(0, kHeap);
    EXPECT_TRUE(e.ledger.idle_backing(shared));
    EXPECT_EQ(e.trim(2), 0u) << "slot 0 is free, but its bank is not";
    EXPECT_TRUE(e.ledger.owns(shared));
}

// All or nothing per bank: the transaction publishes its regions together, so a
// bank one live consumer still reaches is not an idle set and no part of it is
// given back.
TEST(RunResourceGrowthTrim, ABankWithOneLiveRegionKeepsEveryRegion) {
    Endpoint e(4, /*capacity=*/90);
    const std::size_t two_regions[kRegionCount] = {20, 20, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, two_regions), 0);
    e.note_proven(2, 72);
    e.release_reservation(2, 2);
    // Only the heap's consumer went away.
    void *heap = e.region_backing(2, kHeap);
    void *sm = e.region_backing(2, 1);
    ASSERT_TRUE(e.ledger.release_ref(heap, 72));

    EXPECT_TRUE(e.ledger.idle_backing(heap));
    EXPECT_FALSE(e.ledger.idle_backing(sm));
    EXPECT_EQ(e.trim(1), 0u);
    EXPECT_TRUE(e.ledger.owns(heap));
    EXPECT_TRUE(e.ledger.owns(sm));
}

// A runtime that commits its pooled arena at prewarm holds backing that was
// never any run's, with its device image already uploaded into it. It is
// unreferenced from the moment it is published, so idleness alone would offer
// it — and freeing it would take the context's own storage.
TEST(RunResourceGrowthTrim, APrewarmedBankIsNotAnIdleSet) {
    Endpoint e(4, /*capacity=*/60);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    ASSERT_EQ(e.prewarm_arena(2, thirty), 0);
    void *prewarmed = e.region_backing(2, kHeap);
    ASSERT_NE(prewarmed, nullptr);

    EXPECT_FALSE(e.ledger.idle_backing(prewarmed));
    EXPECT_EQ(e.trim(1), 0u);
    EXPECT_TRUE(e.ledger.owns(prewarmed));
    EXPECT_TRUE(e.backend.released.empty());
}

// A refused transaction gives back every reference it took, which is what makes
// the storage it staged recoverable. Keeping them would leave the bytes charged
// and referenced until the whole run retired — which a plan that resumes never
// does.
TEST(RunResourceGrowthTrim, ARefusedTransactionLeavesNoClaimCharged) {
    Endpoint e(4, /*capacity=*/50);
    const std::size_t two_regions[kRegionCount] = {20, 40, 0};
    e.reserve(1, 1);
    // The heap stages 20, then the shared-memory region's 40 cannot be served:
    // the transaction holds both peaks at once and publishes neither.
    EXPECT_NE(e.commit_arena(1, 1, 90, two_regions), 0);
    EXPECT_EQ(e.backend.acquire_calls, 4) << "both attempts stage heap then refuse the second region";
    EXPECT_EQ(e.backend.refusals, 2);
    EXPECT_EQ(e.backend.released.size(), 1u) << "first staged allocation was rolled back before retry";
    EXPECT_EQ(e.ledger.reclaim_obsolete(), 0);
    EXPECT_EQ(e.backend.free_bytes(), 50u) << "the staged block went back";
    EXPECT_TRUE(AutoAdmissionState::claims().empty());
    EXPECT_EQ(e.region_backing(1, kHeap), nullptr) << "a refusal publishes nothing";
}

// An endpoint a caller configured keeps the single attempt it always had. The
// trim moves storage to serve a capacity this endpoint derives, and a caller
// who set its own has not asked for that.
TEST(RunResourceGrowthTrim, AnEndpointThatDoesNotSizeItselfIsNeitherTrimmedNorRetried) {
    Endpoint e(4, /*capacity=*/60, /*sizes_itself=*/false);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(2, 2);
    ASSERT_EQ(e.commit_arena(2, 2, 72, thirty), 0);
    e.retire(2, 72);
    e.release_reservation(2, 2);
    void *idle = e.region_backing(2, kHeap);
    ASSERT_TRUE(e.ledger.idle_backing(idle));

    e.reserve(1, 1);
    const std::size_t forty[kRegionCount] = {40, 0, 0};
    const int before = e.backend.acquire_calls;
    EXPECT_NE(e.commit_arena(1, 1, 90, forty), 0);
    EXPECT_EQ(e.backend.acquire_calls - before, 1) << "one attempt, as it had before";
    EXPECT_TRUE(e.ledger.owns(idle)) << "the idle set was not moved underneath its caller";
}

// Exactly two attempts per activation: one before the trim and one after it.
// More would be a retry loop, and the design has no timer to bound one.
TEST(RunResourceGrowthTrim, ARefusalCostsTwoAttemptsAndNoMore) {
    Endpoint e(4, /*capacity=*/10);
    e.reserve(0, 0);
    const int before = e.backend.acquire_calls;
    EXPECT_EQ(e.grow_staging(0, 70, 40), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
    EXPECT_EQ(e.backend.acquire_calls - before, 2);
    EXPECT_EQ(e.backend.refusals, 2);
}

}  // namespace

TEST(RunResourceGrowthTrim, ProductionAlignmentIsChargedAsRawBackingBytes) {
    constexpr size_t alignment = DeviceArena::kDefaultBaseAlign;
    constexpr size_t heap_charge = 30 + alignment - 1;
    Endpoint e(4, heap_charge + alignment - 1, true, alignment);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(0, 0);
    ASSERT_EQ(e.commit_arena(0, 0, 70, thirty), 0);
    e.note_proven(0, 70);
    EXPECT_EQ(e.ledger.reserved_bytes(), heap_charge);
    const std::size_t one[kRegionCount] = {1, 0, 0};
    e.reserve(1, 1);
    EXPECT_EQ(e.commit_arena(1, 1, 71, one), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
    EXPECT_EQ(e.backend.free_bytes(), alignment - 1);
    EXPECT_EQ(e.backend.refusals, 2) << "one payload byte still requires a full aligned backing request";
}

TEST(RunResourceGrowthWait, OnlyCapacityRefusalsMayRetryOrWaitForAConsumer) {
    using Refusal = WorkspaceManager::AcquireRefusal;
    EXPECT_EQ(classify_growth_refusal(Refusal::OverQuota), GrowthAttempt::Refused);
    EXPECT_EQ(classify_growth_refusal(Refusal::BackendFailed), GrowthAttempt::Refused);
    Endpoint e(4, 100);
    e.reserve(1, 1);
    ASSERT_EQ(e.grow_staging(1, 71, 30), 0);
    e.note_proven(1, 71);
    e.report_launched(1, 71);
    e.admission.begin_plan(0, 72, Phase::Committing);
    RunResourceGrowth growth(e.ledger, e.admission, e);
    for (Refusal why : {Refusal::None, Refusal::NotServing, Refusal::Overflow, Refusal::Degraded}) {
        struct Attempt {
            GrowthAttempt outcome;
            int calls{0};
        } attempt{classify_growth_refusal(why)};
        auto result = growth.attempt_with_one_retry(
            0, true,
            [](void *ctx) {
                auto &a = *static_cast<Attempt *>(ctx);
                ++a.calls;
                return a.outcome;
            },
            &attempt
        );
        EXPECT_EQ(attempt.calls, 1);
        EXPECT_EQ(growth.report_refusal(0, Phase::Committing, 10, result), 0);
    }
}

TEST(RunResourceGrowthTrim, AReferenceFailureCannotRollBackAlreadyPublishedBacking) {
    Endpoint e(4, 100);
    const std::size_t thirty[kRegionCount] = {30, 0, 0};
    e.reserve(0, 0);
    e.refuse_reference = true;
    EXPECT_EQ(e.commit_arena(0, 0, 70, thirty), PTO_RUNTIME_ERR_INTERNAL);
    void *const published = e.region_backing(0, kHeap);
    ASSERT_NE(published, nullptr);
    EXPECT_TRUE(AutoAdmissionState::claims().empty());
    RunResourceGrowth(e.ledger, e.admission, e).rollback_claims();
    e.retire(0, 70);
    ASSERT_EQ(e.ledger.reclaim_obsolete(), 0);
    EXPECT_TRUE(e.ledger.owns(published));
    EXPECT_TRUE(e.backend.released.empty());
    EXPECT_EQ(
        e.ledger.acquire(WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::GmHeap), 71, 30), published
    );
}
