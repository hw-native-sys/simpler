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

#include "scheduler/scheduler_async_discovery.h"

namespace {
using Result = SchedulerAsyncDiscoverResult;
using Phase = SchedulerAsyncPhase;
using Kind = SchedulerAsyncWorkKind;
using Admission = SchedulerAsyncAdmission;

struct Directory {
    std::vector<SchedulerAsyncDirectoryWord> words;
    std::vector<SchedulerAsyncLease> leases;
    explicit Directory(uint64_t records) :
        words(scheduler_async_directory_word_count(records)),
        leases(records) {}
    SchedulerAsyncDirectoryView view() { return {words.data(), leases.data(), leases.size()}; }
    void activate(uint64_t index) {
        ASSERT_EQ(scheduler_async_lease_publish(leases[index]).result, SchedulerAsyncLeaseResult::SUCCESS);
        ASSERT_EQ(
            scheduler_async_directory_publish_once(words[index / 64], index % 64),
            SchedulerAsyncDirectoryResult::SUCCESS
        );
    }
    void handoff(const SchedulerAsyncClaim &claim) {
        ASSERT_EQ(
            scheduler_async_lease_handoff(leases[claim.index], claim.owned_word, Phase::RUNNABLE).result,
            SchedulerAsyncLeaseResult::SUCCESS
        );
    }
};

TEST(HbgAsyncDiscovery, EmptyDirectoriesConsumeNoAtomicsAndRotatePriority) {
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    EXPECT_EQ(scheduler_async_discover({}, {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(state.next_kind, Kind::FINISH);
    EXPECT_EQ(budget.directory_words, 4u);
    EXPECT_EQ(budget.lease_probes, 2u);
    EXPECT_EQ(budget.claim_attempts, 2u);
    EXPECT_EQ(budget.work_units, 8u);
    EXPECT_EQ(scheduler_async_discover({}, {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(state.next_kind, Kind::CHECK);
    EXPECT_EQ(scheduler_async_directory_word_count(UINT64_MAX), (UINT64_MAX / 64) + 1);
}

TEST(HbgAsyncDiscovery, SparseDirectoryProgressesAcrossWordBudgetAndRevisitsEmptyWords) {
    Directory checks(65 * 64);
    checks.activate(64 * 64 + 7);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncClaim claim;
    for (uint32_t step = 0; step < 16; ++step) {
        SchedulerAsyncSliceBudget budget;
        EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::NONE);
        EXPECT_EQ(budget.directory_words, 0u);
        EXPECT_EQ(budget.lease_probes, 2u);
        EXPECT_EQ(state.check.word, (step + 1) * 4);
    }
    SchedulerAsyncSliceBudget budget;
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 64 * 64 + 7);
    checks.handoff(claim);
    checks.activate(0);
    bool found_new = false;
    for (uint32_t step = 0; step < 18; ++step) {
        budget = {};
        const auto result = scheduler_async_discover(checks.view(), {}, &state, &budget, &claim);
        if (result == Result::CLAIMED) {
            found_new = claim.index == 0;
            checks.handoff(claim);
            if (found_new) break;
        }
    }
    EXPECT_TRUE(found_new);
}

TEST(HbgAsyncDiscovery, ClaimsAtMostOneAndAlternatesCheckAndFinish) {
    Directory checks(3), finishes(1);
    for (uint64_t i = 0; i < 3; ++i)
        checks.activate(i);
    finishes.activate(0);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncClaim claim;
    for (uint32_t step = 0; step < 6; ++step) {
        SchedulerAsyncSliceBudget budget;
        ASSERT_EQ(scheduler_async_discover(checks.view(), finishes.view(), &state, &budget, &claim), Result::CLAIMED);
        EXPECT_EQ(claim.kind, step % 2 == 0 ? Kind::CHECK : Kind::FINISH);
        EXPECT_EQ(budget.lease_probes, 1u);
        EXPECT_EQ(budget.claim_attempts, 1u);
        EXPECT_EQ(budget.work_units, 8u);
        if (claim.kind == Kind::CHECK) {
            EXPECT_EQ(claim.index, step / 2);
            checks.handoff(claim);
        } else {
            finishes.handoff(claim);
        }
    }
}

TEST(HbgAsyncDiscovery, OwnedAndTerminalCandidatesConsumeProbeBudgetWithoutCas) {
    Directory checks(4);
    for (uint64_t i = 0; i < 4; ++i)
        checks.activate(i);
    auto first = scheduler_async_lease_try_claim(checks.leases[0], scheduler_async_lease_load(checks.leases[0]));
    ASSERT_EQ(first.result, SchedulerAsyncLeaseResult::SUCCESS);
    auto second = scheduler_async_lease_try_claim(checks.leases[1], scheduler_async_lease_load(checks.leases[1]));
    ASSERT_EQ(second.result, SchedulerAsyncLeaseResult::SUCCESS);
    ASSERT_EQ(
        scheduler_async_lease_handoff(checks.leases[1], second.word, Phase::TERMINAL).result,
        SchedulerAsyncLeaseResult::SUCCESS
    );
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(budget.lease_probes, 0u);
    EXPECT_EQ(budget.claim_attempts, 2u);
    EXPECT_EQ(state.check.bit, 2u);
    budget = {};
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 2u);
    EXPECT_EQ(scheduler_async_directory_load(checks.words[0]), 15u);
}

struct LosingClaim : SchedulerAsyncDiscoveryAtomics {
    static SchedulerAsyncLeaseOutcome claim(SchedulerAsyncLease &lease, uint64_t observed) {
        // Another worker claims and hands off before this worker's single CAS.
        const auto competitor = scheduler_async_lease_try_claim(lease, observed);
        if (competitor.result == SchedulerAsyncLeaseResult::SUCCESS)
            scheduler_async_lease_handoff(lease, competitor.word, Phase::RUNNABLE);
        return scheduler_async_lease_try_claim(lease, observed);
    }
};

TEST(HbgAsyncDiscovery, LostClaimsSpendBothBudgetsAndAdvanceBeyondTheHotRecord) {
    Directory checks(3);
    for (uint64_t i = 0; i < 3; ++i)
        checks.activate(i);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    EXPECT_EQ(scheduler_async_discover<LosingClaim>(checks.view(), {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(budget.directory_words, 3u);
    EXPECT_EQ(budget.lease_probes, 0u);
    EXPECT_EQ(budget.claim_attempts, 0u);
    EXPECT_EQ(state.check.bit, 2u);
    budget = {};
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 2u);
}

TEST(HbgAsyncDiscovery, Bit63WrapAndPartialTailNeverProbeOutOfRange) {
    Directory checks(65);
    checks.activate(63);
    checks.activate(64);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncClaim claim;
    SchedulerAsyncSliceBudget budget;
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 63u);
    EXPECT_EQ(state.check.word, 1u);
    EXPECT_EQ(state.check.bit, 0u);
    budget = {};
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 64u);
    budget = {};
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(budget.claim_attempts, 2u);
    EXPECT_LT(state.check.word, 2u);
    EXPECT_LT(state.check.bit, 64u);
}

TEST(HbgAsyncDiscovery, MissingPreferredWorkUsesRemainingBudgetForTheOtherKind) {
    Directory checks(64), finishes(1);
    finishes.activate(0);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    ASSERT_EQ(scheduler_async_discover(checks.view(), finishes.view(), &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.kind, Kind::FINISH);
    EXPECT_EQ(budget.directory_words, 2u);
}

TEST(HbgAsyncDiscovery, LargeEmptyPreferredDirectoryCannotStarveTheOtherKind) {
    Directory checks(16 * 64), finishes(1);
    finishes.activate(0);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncClaim claim;
    for (uint32_t i = 0; i < 4; ++i) {
        SchedulerAsyncSliceBudget budget;
        EXPECT_EQ(scheduler_async_discover(checks.view(), finishes.view(), &state, &budget, &claim), Result::NONE);
        EXPECT_EQ(budget.directory_words, 0u);
        budget = {};
        ASSERT_EQ(scheduler_async_discover(checks.view(), finishes.view(), &state, &budget, &claim), Result::CLAIMED);
        EXPECT_EQ(claim.kind, Kind::FINISH);
        finishes.handoff(claim);
    }
    EXPECT_EQ(state.check.word, 0u);
}

TEST(HbgAsyncDiscovery, ContentionAtWordAndTailBoundaryStillReachesTheNextRecord) {
    Directory checks(65);
    for (uint64_t index : {0, 63, 64})
        checks.activate(index);
    SchedulerAsyncDiscoveryState state;
    state.check.bit = 63;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    EXPECT_EQ(scheduler_async_discover<LosingClaim>(checks.view(), {}, &state, &budget, &claim), Result::NONE);
    EXPECT_EQ(budget.lease_probes, 0u);
    EXPECT_EQ(budget.claim_attempts, 0u);
    EXPECT_EQ(state.check.word, 0u);
    EXPECT_EQ(state.check.bit, 0u);
    budget = {};
    ASSERT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::CLAIMED);
    EXPECT_EQ(claim.index, 0u);
}

TEST(HbgAsyncDiscovery, MalformedDirectoryCursorAndLeaseFailWithoutProgressAccess) {
    Directory checks(1);
    SchedulerAsyncDiscoveryState state;
    SchedulerAsyncSliceBudget budget;
    SchedulerAsyncClaim claim;
    checks.words[0].bits = 2;
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::INVALID_STATE);
    checks.words[0].bits = 1;
    budget = {};
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::INVALID_STATE);
    state = {};
    checks.leases[0].word = 0;
    budget = {};
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::INVALID_STATE);
    state.check.word = 1;
    budget = {};
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::INVALID_STATE);
    state = {};
    state.check.bit = 64;
    EXPECT_EQ(scheduler_async_discover(checks.view(), {}, &state, &budget, &claim), Result::INVALID_STATE);
    state = {};
    EXPECT_EQ(scheduler_async_discover({nullptr, nullptr, 1}, {}, &state, &budget, &claim), Result::INVALID_STATE);
}

TEST(HbgAsyncDiscovery, DistributedStartsRemainValidAndDoNotAlwaysSelectBitZero) {
    std::set<uint32_t> starts;
    for (uint64_t worker = 0; worker < 64; ++worker) {
        auto cursor = scheduler_async_directory_start(64, worker);
        EXPECT_EQ(cursor.word, 0u);
        starts.insert(cursor.bit);
        cursor = scheduler_async_directory_start(65, worker);
        EXPECT_LT(cursor.word, 2u);
        EXPECT_LT(cursor.word * 64 + cursor.bit, 65u);
    }
    EXPECT_EQ(starts.size(), 64u);
}

TEST(HbgAsyncDiscovery, GateServesAfterFourReturnsAndNeverAfterPermit) {
    SchedulerAsyncServiceGate gate;
    for (uint32_t i = 0; i < 4; ++i) {
        gate.begin_iteration();
        EXPECT_FALSE(gate.take(true, Admission::BEFORE_ADMISSION, true));
        gate.kernel_returned();
    }
    gate.begin_iteration();
    EXPECT_TRUE(gate.take(true, Admission::BEFORE_ADMISSION, true));
    EXPECT_EQ(gate.business_returns, 0u);
    EXPECT_FALSE(gate.take(true, Admission::ADMISSION_DENIED, false));
    EXPECT_FALSE(gate.take(true, Admission::BEFORE_ADMISSION, false));
    gate.begin_iteration();
    EXPECT_FALSE(gate.take(true, Admission::PERMIT_GRANTED, false));
    EXPECT_FALSE(gate.spent);
}

TEST(HbgAsyncDiscovery, IdleAndDeniedPathsShareOneSliceAndSynchronousRunsSkipIt) {
    SchedulerAsyncServiceGate gate;
    EXPECT_FALSE(gate.take(false, Admission::BEFORE_ADMISSION, false));
    EXPECT_FALSE(gate.spent);
    EXPECT_TRUE(gate.take(true, Admission::BEFORE_ADMISSION, false));
    EXPECT_FALSE(gate.take(true, Admission::ADMISSION_DENIED, false));
    gate.begin_iteration();
    EXPECT_FALSE(gate.take(true, Admission::BEFORE_ADMISSION, true));
    EXPECT_TRUE(gate.take(true, Admission::ADMISSION_DENIED, true));
    EXPECT_FALSE(gate.take(true, Admission::ADMISSION_DENIED, true));
    for (uint32_t i = 0; i < 12; ++i)
        gate.kernel_returned();
    EXPECT_EQ(gate.business_returns, 4u);
    gate.begin_iteration();
    EXPECT_TRUE(gate.take(true, Admission::BEFORE_ADMISSION, true));
    EXPECT_EQ(gate.business_returns, 0u);
}
}  // namespace
