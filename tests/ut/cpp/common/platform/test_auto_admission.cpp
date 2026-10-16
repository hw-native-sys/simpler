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

#include <set>
#include <vector>

#include "host/auto_admission.h"

namespace {

using Phase = AutoAdmissionState::Phase;
using Park = AutoAdmissionState::Park;
using Growth = AutoAdmissionState::Growth;

constexpr uint32_t kDirectory = 8;

/**
 * A configured state, constructed in place.
 *
 * The class owns a mutex and is deliberately non-copyable, so each case holds
 * its own rather than taking one by value.
 */
struct ConfiguredState {
    explicit ConfiguredState(size_t slots = kDirectory) { state.configure(slots); }
    AutoAdmissionState *operator->() { return &state; }
    AutoAdmissionState state;
};

// A context that never asks keeps reporting the grant it was given, so every
// route this feature excludes is excluded by the state holding nothing.
TEST(AutoAdmissionCapacity, AnUnconfiguredStateReportsTheGrantUnchanged) {
    AutoAdmissionState state;
    EXPECT_FALSE(state.enabled());
    EXPECT_EQ(state.affordable_sets(kDirectory), kDirectory);
    EXPECT_EQ(state.proven_prefix(), 0u);
    // Asking for growth permission is a no-op rather than a refusal, so an
    // unmanaged context's prepare path is the one it had before.
    EXPECT_EQ(state.acquire_growth_token(0, 11), Growth::Granted);
}

// E1: nothing has been demonstrated, so one slot is offered — not the whole
// directory, which is what inferring the proof from an untouched slot's absent
// storage would have produced.
TEST(AutoAdmissionCapacity, ColdStartOffersExactlyOneSet) {
    ConfiguredState state;
    EXPECT_EQ(state->proven_prefix(), 0u);
    EXPECT_EQ(state->affordable_sets(kDirectory), 1u);
    for (uint32_t slot = 0; slot < kDirectory; ++slot) {
        EXPECT_FALSE(state->proven(slot)) << "slot " << slot << " was never prepared";
    }
}

// E2a: the commit is the proof, so a set whose private byte demand was
// legitimately zero proves its slot exactly as one that allocated. This is the
// case a storage-derived predicate left at one set for ever.
TEST(AutoAdmissionCapacity, ACompletedCommitProvesItsSlotWithoutAnyBytes) {
    ConfiguredState state;
    state->note_proven(0);
    EXPECT_EQ(state->proven_prefix(), 1u);
    EXPECT_EQ(state->affordable_sets(kDirectory), 2u);
    state->note_proven(1);
    EXPECT_EQ(state->affordable_sets(kDirectory), 3u);
}

// The prefix is what the lease pool scans, so a proven slot above an unproven
// one must not be counted: a population count of two would offer slot 1, which
// nothing has demonstrated.
TEST(AutoAdmissionCapacity, SparseProofsAreCountedAsAPrefixNotAPopulation) {
    ConfiguredState state;
    state->note_proven(0);
    state->note_proven(2);
    EXPECT_EQ(state->proven_prefix(), 1u);
    EXPECT_EQ(state->affordable_sets(kDirectory), 2u);
}

// E2: withdrawing a middle slot's proof caps what is offered at that slot,
// whatever is proven above it.
TEST(AutoAdmissionCapacity, WithdrawingAMiddleProofCapsTheOffer) {
    ConfiguredState state;
    state->note_proven(0);
    state->note_proven(1);
    state->note_proven(2);
    EXPECT_EQ(state->affordable_sets(kDirectory), 4u);
    state->clear_proven(1);
    EXPECT_EQ(state->proven_prefix(), 1u);
    EXPECT_EQ(state->affordable_sets(kDirectory), 2u);
    EXPECT_TRUE(state->proven(2)) << "a slot above the cap keeps its own proof";
}

TEST(AutoAdmissionCapacity, TheOfferNeverExceedsTheDirectory) {
    ConfiguredState state(2);
    state->note_proven(0);
    state->note_proven(1);
    EXPECT_EQ(state->affordable_sets(2), 2u);
}

// The exploratory set is offered only while no growth is in flight, which is
// the same exclusivity the token enforces: the count cannot invite a second
// simultaneous replacement peak.
TEST(AutoAdmissionCapacity, TheExploratorySetIsWithheldWhileAGrowerHoldsTheToken) {
    ConfiguredState state;
    state->note_proven(0);
    EXPECT_EQ(state->affordable_sets(kDirectory), 2u);
    ASSERT_EQ(state->acquire_growth_token(1, 7), Growth::Granted);
    EXPECT_EQ(state->affordable_sets(kDirectory), 1u);
    state->release_growth_token(1);
    EXPECT_EQ(state->affordable_sets(kDirectory), 2u);
}

// One grower per context, and the holder keeps it across its own stages: a plan
// that grows its inputs and then its regions is one growth attempt.
TEST(AutoAdmissionToken, OneGrowerAtATimeAndTheHolderIsIdempotent) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Staging);
    ASSERT_EQ(state->acquire_growth_token(0, 100), Growth::Granted);
    EXPECT_EQ(state->acquire_growth_token(0, 100), Growth::Granted) << "the holder re-asking is not a wait";
    EXPECT_EQ(state->acquire_growth_token(1, 200), Growth::WaitForToken);
    EXPECT_TRUE(state->token_held_by_other(1));
    EXPECT_FALSE(state->token_held_by_other(0));
    state->release_growth_token(0);
    EXPECT_TRUE(state->token_free());
    EXPECT_EQ(state->acquire_growth_token(1, 200), Growth::Granted);
}

TEST(AutoAdmissionToken, EndingAPlanReleasesItsToken) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Staging);
    ASSERT_EQ(state->acquire_growth_token(0, 100), Growth::Granted);
    state->end_plan(0);
    EXPECT_TRUE(state->token_free());
    EXPECT_FALSE(state->holds_token(0));
}

// The wait set is captured once. A consumer that launches after the plan first
// could not proceed neither extends the wait nor rescues it, which is what
// bounds the waits by the captured set's own size.
TEST(AutoAdmissionWait, TheWaitSetIsCapturedOnceAndNeverGrows) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12}));
    EXPECT_EQ(state->pending_wait_set(0, Phase::Committing).size(), 2u);
    // A later report naming a third consumer does not extend it.
    EXPECT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12, 13}));
    const std::vector<uint64_t> pending = state->pending_wait_set(0, Phase::Committing);
    ASSERT_EQ(pending.size(), 2u);
    const std::set<uint64_t> as_set(pending.begin(), pending.end());
    EXPECT_EQ(as_set, (std::set<uint64_t>{11, 12}));
}

// A plan never waits for itself.
TEST(AutoAdmissionWait, ThePlansOwnIdentityIsNotAConsumerItCanWaitFor) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    EXPECT_FALSE(state->capture_wait_set(0, Phase::Committing, {100}));
    EXPECT_TRUE(state->pending_wait_set(0, Phase::Committing).empty());
}

TEST(AutoAdmissionWait, AnEmptyHolderSetIsNotAdmissible) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    EXPECT_FALSE(state->capture_wait_set(0, Phase::Committing, {}));
}

// E34: two named consumers finishing separately cost one wait each, in the
// order they finish, and the set is exhausted afterwards.
TEST(AutoAdmissionWait, TwoNamedConsumersAreSpentOneAtATime) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12}));
    std::set<uint64_t> still_holding{11, 12};
    const auto holding = [&still_holding](uint64_t epoch) {
        return still_holding.count(epoch) != 0;
    };

    EXPECT_FALSE(state->consume_finished_holder(0, Phase::Committing, holding))
        << "nothing has finished, so there is nothing to spend";

    still_holding.erase(12);
    EXPECT_TRUE(state->consume_finished_holder(0, Phase::Committing, holding));
    EXPECT_EQ(state->pending_wait_set(0, Phase::Committing), (std::vector<uint64_t>{11}));
    EXPECT_FALSE(state->consume_finished_holder(0, Phase::Committing, holding))
        << "12 is already spent and 11 is still holding";

    still_holding.erase(11);
    EXPECT_TRUE(state->consume_finished_holder(0, Phase::Committing, holding));
    EXPECT_TRUE(state->pending_wait_set(0, Phase::Committing).empty());
    EXPECT_FALSE(state->consume_finished_holder(0, Phase::Committing, holding));
}

// E35: a consumer outside the captured set finishing is not this plan's event.
TEST(AutoAdmissionWait, AnUnrelatedFinishSpendsNothingAndClearsNoWait) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11}));
    state->note_parked(0, Park::Bytes);
    const auto holding = [](uint64_t epoch) {
        return epoch == 11;  // 11 still holds; 99 never did
    };
    EXPECT_FALSE(state->consume_finished_holder(0, Phase::Committing, holding));
    EXPECT_EQ(state->parked(0), Park::Bytes) << "the plan is still waiting for its own consumer";
}

// Consuming an identity is what clears the wait, so the caller may try again.
TEST(AutoAdmissionWait, SpendingAnIdentityClearsTheParkedMark) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11}));
    state->note_parked(0, Park::Bytes);
    const auto nothing_holds = [](uint64_t) {
        return false;
    };
    EXPECT_TRUE(state->consume_finished_holder(0, Phase::Committing, nothing_holds));
    EXPECT_EQ(state->parked(0), Park::None);
}

// E37: once the captured set is spent, a consumer that launched later is not
// appended — the plan fails rather than waiting on a newcomer.
TEST(AutoAdmissionWait, AnExhaustedSetIsNotRefilledByANewConsumer) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11}));
    const auto nothing_holds = [](uint64_t) {
        return false;
    };
    ASSERT_TRUE(state->consume_finished_holder(0, Phase::Committing, nothing_holds));
    // A later report, naming a consumer that started after the first wait.
    EXPECT_FALSE(state->capture_wait_set(0, Phase::Committing, {77}))
        << "the set is immutable, so it is exhausted rather than extended";
    EXPECT_TRUE(state->pending_wait_set(0, Phase::Committing).empty());
}

// The two stages hold separate sets, because the consumer population genuinely
// differs between staging inputs and committing regions — but the spent
// identities are the plan's and span both, so no retirement is used twice.
TEST(AutoAdmissionWait, StagesHoldSeparateSetsAndShareTheSpentIdentities) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Staging);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Staging, {11, 12}));
    const auto only_11_holds = [](uint64_t epoch) {
        return epoch == 11;
    };
    ASSERT_TRUE(state->consume_finished_holder(0, Phase::Staging, only_11_holds));  // spends 12

    state->set_phase(0, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12, 13}));
    const std::vector<uint64_t> pending = state->pending_wait_set(0, Phase::Committing);
    const std::set<uint64_t> as_set(pending.begin(), pending.end());
    EXPECT_EQ(as_set, (std::set<uint64_t>{11, 13})) << "12's retirement was already spent in staging";
}

// E36: a capacity decrease while a plan waits leaves the captured set — and so
// the number of waits it may still take — untouched. The bound is the identity
// directory, not a figure that can shrink underneath the plan.
TEST(AutoAdmissionWait, ACapacityDecreaseDoesNotShrinkACapturedWaitSet) {
    ConfiguredState state;
    state->note_proven(0);
    state->begin_plan(2, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(2, Phase::Committing, {11, 12, 13}));
    ASSERT_EQ(state->pending_wait_set(2, Phase::Committing).size(), 3u);

    // Every other slot loses its proof, so the offer falls back to one set.
    state->clear_proven(0);
    ASSERT_EQ(state->affordable_sets(kDirectory), 1u);

    EXPECT_EQ(state->pending_wait_set(2, Phase::Committing).size(), 3u)
        << "the plan may still spend every identity it captured";
}

// A retry is the same submission continuing, so re-entering the plan under the
// attempt's new identity must not re-open the wait. Without this a plan would
// capture a fresh set on every attempt and wait once per live consumer for
// ever, which is the retry loop this design exists to avoid.
TEST(AutoAdmissionPlan, ARetryUnderANewIdentityKeepsTheWaitItAlreadySpent) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12}));
    const auto only_11_holds = [](uint64_t epoch) {
        return epoch == 11;
    };
    ASSERT_TRUE(state->consume_finished_holder(0, Phase::Committing, only_11_holds));  // spends 12

    // The next attempt mints its own run identity; the plan is the same one.
    state->begin_plan(0, 101, Phase::Committing);
    EXPECT_EQ(state->plan(0).epoch, 101u);
    EXPECT_EQ(state->pending_wait_set(0, Phase::Committing), (std::vector<uint64_t>{11}))
        << "12's retirement stays spent across the attempt boundary";
    // Still something to wait for — 11 has not finished — but the set it is
    // waiting on is the one captured at the first wait, so a consumer that
    // appeared since is not added to it.
    EXPECT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12, 77}));
    EXPECT_EQ(state->pending_wait_set(0, Phase::Committing), (std::vector<uint64_t>{11}))
        << "77 appeared after the first wait, so it neither extends nor rescues it";
}

// Only an ended plan starts clean, and that is what a completed commit or an
// abandonment does. Otherwise the slot's next occupant would inherit a wait
// set naming consumers it never named.
TEST(AutoAdmissionPlan, ANewRunOnASlotStartsWithNoWaitState) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11}));
    state->note_parked(0, Park::Bytes);
    state->note_plan_stamped(0);
    state->note_input_staging_held(0);

    state->end_plan(0);
    state->begin_plan(0, 101, Phase::Staging);
    const AutoAdmissionState::SlotPlan plan = state->plan(0);
    EXPECT_EQ(plan.epoch, 101u);
    EXPECT_EQ(plan.parked, Park::None);
    EXPECT_FALSE(plan.plan_stamped);
    EXPECT_FALSE(plan.holds_input_staging);
    EXPECT_TRUE(state->pending_wait_set(0, Phase::Committing).empty());
}

// The wait hands back the attempt's own references, because the retry takes
// its own. Keeping them would leave one reference per attempt on a block only
// one attempt will ever use, and a block's reference table is finite.
TEST(AutoAdmissionPlan, AWaitGivesBackTheAttemptsOwnReferencesAndKeepsThePlan) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Staging);
    state->note_input_staging_held(0);
    state->note_plan_stamped(0);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Staging, {11}));
    ASSERT_EQ(state->acquire_growth_token(0, 100), Growth::Granted);

    state->note_epoch_refs_released(0);

    const AutoAdmissionState::SlotPlan plan = state->plan(0);
    EXPECT_FALSE(plan.holds_input_staging) << "the attempt's reference is back";
    EXPECT_TRUE(plan.plan_stamped) << "the slot's saved plan is not the attempt's to discard";
    EXPECT_TRUE(state->holds_token(0)) << "it is still this context's one outstanding grower";
    EXPECT_EQ(state->pending_wait_set(0, Phase::Staging), (std::vector<uint64_t>{11}));
}

// The growth permission is the context's, not the plan's, so it survives one
// plan replacing another on the same slot — the holder is still the one
// outstanding grower until it releases.
TEST(AutoAdmissionPlan, ARetryKeepsTheGrowthPermissionItAlreadyHeld) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Staging);
    ASSERT_EQ(state->acquire_growth_token(0, 100), Growth::Granted);
    state->begin_plan(0, 101, Phase::Staging);
    EXPECT_TRUE(state->holds_token(0)) << "the same submission is still the one outstanding grower";
    EXPECT_EQ(state->acquire_growth_token(0, 101), Growth::Granted);
}

TEST(AutoAdmissionPlan, PlanStateRecordsWhatAnAbandonmentHasToGiveBack) {
    ConfiguredState state;
    state->begin_plan(1, 55, Phase::Staging);
    state->note_input_staging_held(1);
    state->set_phase(1, Phase::Planned);
    state->note_plan_stamped(1);

    const AutoAdmissionState::SlotPlan plan = state->plan(1);
    EXPECT_EQ(plan.phase, Phase::Planned);
    EXPECT_EQ(plan.epoch, 55u);
    EXPECT_TRUE(plan.holds_input_staging);
    EXPECT_TRUE(plan.plan_stamped);

    state->end_plan(1);
    EXPECT_EQ(state->plan(1).phase, Phase::None);
    EXPECT_FALSE(state->plan(1).holds_input_staging);
}

// A plan parked before its inputs were staged owns neither a stamp nor a
// staging reference, which is what makes an abandonment at that stage safe
// without any of the later releases.
TEST(AutoAdmissionPlan, AStagingStageParkOwnsNeitherStampNorStaging) {
    ConfiguredState state;
    state->begin_plan(0, 70, Phase::Staging);
    state->note_parked(0, Park::Bytes);
    const AutoAdmissionState::SlotPlan plan = state->plan(0);
    EXPECT_FALSE(plan.plan_stamped);
    EXPECT_FALSE(plan.holds_input_staging);
    EXPECT_EQ(plan.phase, Phase::Staging);
}

// Claims are this thread's in-flight transaction, so a successor prepared on
// another thread cannot have its staged block rolled back by this one.
TEST(AutoAdmissionClaims, ClaimsAreThreadLocalAndClearedOnCommit) {
    AutoAdmissionState::claims().clear();
    int marker = 0;
    AutoAdmissionState::Claim claim;
    claim.base = &marker;
    claim.epoch = 42;
    claim.region_index = 3;
    AutoAdmissionState::record_claim(claim);
    ASSERT_EQ(AutoAdmissionState::claims().size(), 1u);
    EXPECT_EQ(AutoAdmissionState::claims()[0].base, &marker);
    EXPECT_EQ(AutoAdmissionState::claims()[0].epoch, 42u);
    AutoAdmissionState::commit_claims();
    EXPECT_TRUE(AutoAdmissionState::claims().empty());
}

TEST(AutoAdmissionClaims, ARegionsOwnBackingIsMarkedSoARollbackLeavesItPublished) {
    AutoAdmissionState::claims().clear();
    int first = 0;
    int second = 0;
    AutoAdmissionState::Claim fresh;
    fresh.base = &first;
    fresh.was_region_backing = false;
    AutoAdmissionState::Claim reused;
    reused.base = &second;
    reused.was_region_backing = true;
    AutoAdmissionState::record_claim(fresh);
    AutoAdmissionState::record_claim(reused);
    ASSERT_EQ(AutoAdmissionState::claims().size(), 2u);
    EXPECT_FALSE(AutoAdmissionState::claims()[0].was_region_backing);
    EXPECT_TRUE(AutoAdmissionState::claims()[1].was_region_backing);
    AutoAdmissionState::commit_claims();
}

}  // namespace

TEST(AutoAdmissionPlan, AReservedRetryOwnsItsPlanBeforeAnyFalliblePreparation) {
    ConfiguredState state;
    state->begin_plan(0, 100, Phase::Committing);
    state->note_plan_stamped(0);
    ASSERT_EQ(state->acquire_growth_token(0, 100), Growth::Granted);
    ASSERT_TRUE(state->capture_wait_set(0, Phase::Committing, {11, 12}));
    ASSERT_TRUE(state->consume_finished_holder(0, Phase::Committing, [](uint64_t epoch) {
        return epoch == 11;
    }));
    state->note_parked(0, Park::Bytes);
    state->note_epoch_refs_released(0);

    // Native reservation succeeds, then attach fails before begin_plan or
    // set_prepare_plan. Cleanup must already recognize the new exact epoch.
    state->adopt_plan_attempt(0, 101);
    const auto adopted = state->plan(0);
    EXPECT_EQ(adopted.epoch, 101u);
    EXPECT_NE(adopted.epoch, 100u) << "a stale attempt cannot match the new plan";
    EXPECT_EQ(adopted.phase, Phase::Committing);
    EXPECT_EQ(adopted.parked, Park::Bytes);
    EXPECT_TRUE(adopted.plan_stamped);
    EXPECT_TRUE(adopted.holds_token);
    EXPECT_FALSE(adopted.holds_input_staging);
    EXPECT_EQ(state->pending_wait_set(0, Phase::Committing), (std::vector<uint64_t>{11}));
    EXPECT_EQ(adopted.consumed, (std::vector<uint64_t>{12}));
    // The matching cleanup ends both the host plan and its growth permission.
    state->end_plan(0);
    EXPECT_TRUE(state->token_free());
    EXPECT_EQ(state->plan(0).phase, Phase::None);

    state->adopt_plan_attempt(1, 102);
    EXPECT_EQ(state->plan(1).phase, Phase::None) << "reservation alone creates no plan for a fresh run";
}
