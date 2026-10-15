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
 * The two capacities a diagnostic run holds, and what its own terminal
 * accounting proves.
 *
 * Both sides are production code: the reservation pool, the token and the
 * reconciliation are the ones the collectors use, and the terminal entries are
 * written through the shared wire helpers rather than by hand.
 *
 * A bank is sized by the platform maximum and a run's participants are the
 * prefix the collector brought up, so every case states both: `stride` is how
 * many entries exist, `expected` is how many must have published. Publishing
 * fewer than `expected` is a missing participant, never an idle one.
 *
 * What these cannot cover, stated rather than simulated: device cache
 * visibility — every store here is plain host memory — and a real producer's
 * ordering, which only the device's own barrier establishes.
 */

#include "host/diagnostic_run_banks.h"
#include "host/diagnostic_run_credit.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/chip_swimlane_profiling.h"

namespace credit = simpler::dfx::credit;
namespace runs = simpler::dfx::runs;

// Banks per collector are mirrored in `common/diagnostic_run_terminal.h` rather
// than included from the swimlane's device grid. This is the one place both
// headers are visible, so it is where the mirror is held to its original.
static_assert(
    credit::RunCredits::kTerminalBanks == static_cast<size_t>(PLATFORM_RUN_TERMINAL_BANKS),
    "the queue-pair terminal bank count must track the swimlane's"
);

namespace {

// A bank wide enough to hold more entries than any case's run brings up, so
// "stride" and "expected" are never the same number by accident.
constexpr int kStride = 4;

/** One run's bank, in the state a reset leaves it. */
std::vector<DiagnosticRunTerminal> make_bank(int stride) {
    std::vector<DiagnosticRunTerminal> bank(static_cast<size_t>(stride));
    for (auto &entry : bank)
        diagnostic_terminal_reset(&entry);
    return bank;
}

void publish(DiagnosticRunTerminal &entry, uint64_t epoch, const DiagnosticRunTerminalCounts &counts) {
    diagnostic_terminal_store_counts(&entry, &counts);
    diagnostic_terminal_commit(&entry, epoch);
}

/** What a producer leaves at its run init, before it has collected anything. */
void claim(DiagnosticRunTerminal &entry, uint64_t epoch) { diagnostic_terminal_claim(&entry, epoch); }

/** Counters of a producer that delivered everything it took on. */
DiagnosticRunTerminalCounts clean_counts(uint32_t records) {
    DiagnosticRunTerminalCounts counts{};
    counts.total = records;
    counts.published_records = records;
    counts.published_buffers = records == 0 ? 0 : 1;
    return counts;
}

}  // namespace

TEST(DiagnosticRunCredit, ReservesUpToCapacityThenRefusesWithoutBlocking) {
    credit::RunCredits pool;
    std::vector<credit::Reservation> held;
    for (size_t i = 0; i < credit::RunCredits::kTerminalBanks; ++i) {
        credit::Reservation got = pool.reserve_terminal_bank(100 + i);
        ASSERT_TRUE(got.held());
        held.push_back(std::move(got));
    }
    EXPECT_EQ(pool.banks_in_use(), credit::RunCredits::kTerminalBanks);

    const credit::Reservation refused = pool.reserve_terminal_bank(999);
    EXPECT_FALSE(refused.held());
    EXPECT_EQ(pool.bank_refusals(), 1u);

    // Every index is distinct, because a bank addresses storage.
    std::vector<size_t> indices;
    for (const auto &reservation : held)
        indices.push_back(reservation.index());
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(std::unique(indices.begin(), indices.end()), indices.end());
}

TEST(DiagnosticRunCredit, ScopeExitRollsBackAPartialAdmission) {
    credit::RunCredits pool;
    {
        const credit::Reservation bank = pool.reserve_terminal_bank(7);
        const credit::Reservation slot = pool.reserve_output_slot(7);
        ASSERT_TRUE(bank.held());
        ASSERT_TRUE(slot.held());
        EXPECT_EQ(pool.banks_in_use(), 1u);
        EXPECT_EQ(pool.slots_in_use(), 1u);
    }
    EXPECT_EQ(pool.banks_in_use(), 0u);
    EXPECT_EQ(pool.slots_in_use(), 0u);
}

TEST(DiagnosticRunCredit, BankAndSlotAreIndependentCapacities) {
    credit::RunCredits pool;
    std::vector<credit::Reservation> slots;
    for (size_t i = 0; i < credit::RunCredits::kOutputSlots; ++i) {
        slots.push_back(pool.reserve_output_slot(200 + i));
        ASSERT_TRUE(slots.back().held());
    }
    EXPECT_FALSE(pool.reserve_output_slot(300).held());

    // A writer still holding every output slot does not deny a bank: the bank
    // is freed by the host's copy, the slot by publication.
    const credit::Reservation bank = pool.reserve_terminal_bank(300);
    EXPECT_TRUE(bank.held());
}

TEST(DiagnosticRunCredit, ReleaseIsIdempotentAndMoveLeavesNoSecondOwner) {
    credit::RunCredits pool;
    credit::Reservation first = pool.reserve_terminal_bank(11);
    ASSERT_TRUE(first.held());

    credit::Reservation second = std::move(first);
    EXPECT_FALSE(first.held());
    EXPECT_TRUE(second.held());
    EXPECT_EQ(pool.banks_in_use(), 1u);

    second.release();
    EXPECT_EQ(pool.banks_in_use(), 0u);
    second.release();  // a second release must not hand the same index out twice
    EXPECT_EQ(pool.banks_in_use(), 0u);
}

TEST(DiagnosticRunCredit, AStaleTokenCannotFreeTheIndexItsSuccessorHolds) {
    credit::RunCredits pool;
    credit::Reservation stale = pool.reserve_terminal_bank(5);
    ASSERT_TRUE(stale.held());
    const size_t index = stale.index();
    const uint64_t before = pool.generation();

    pool.reset();
    EXPECT_GT(pool.generation(), before);
    EXPECT_EQ(pool.banks_in_use(), 0u);

    // The rebuilt pool hands the same index to someone else. The stale token
    // names it, passes nothing else, and must not release it: the generation is
    // read before the write and cannot by itself protect the new holder, so the
    // per-index holder ticket is what refuses this.
    credit::Reservation fresh = pool.reserve_terminal_bank(6);
    ASSERT_TRUE(fresh.held());
    ASSERT_EQ(fresh.index(), index);
    stale.release();
    EXPECT_EQ(pool.banks_in_use(), 1u) << "a stale token released a live holder's bank";
    EXPECT_TRUE(fresh.held());
}

TEST(DiagnosticRunCredit, ATokenIsBoundToItsPoolKindAndGeneration) {
    credit::RunCredits mine;
    credit::RunCredits theirs;

    const credit::Reservation bank = mine.reserve_terminal_bank(12);
    const credit::Reservation slot = mine.reserve_output_slot(12);
    const credit::Reservation foreign = theirs.reserve_terminal_bank(12);
    ASSERT_TRUE(bank.held());

    EXPECT_TRUE(bank.issued_by(&mine, credit::CreditKind::TerminalBank));
    EXPECT_FALSE(bank.issued_by(&theirs, credit::CreditKind::TerminalBank)) << "another pool's token was accepted";
    EXPECT_FALSE(bank.issued_by(&mine, credit::CreditKind::OutputSlot)) << "the two capacities are interchangeable";
    EXPECT_FALSE(slot.issued_by(&mine, credit::CreditKind::TerminalBank));
    EXPECT_FALSE(foreign.issued_by(&mine, credit::CreditKind::TerminalBank));
    EXPECT_FALSE(credit::Reservation{}.issued_by(&mine, credit::CreditKind::TerminalBank));
}

TEST(DiagnosticRunCredit, AGenerationBehindIsNoLongerThisPoolsToken) {
    credit::RunCredits pool;
    const credit::Reservation stale = pool.reserve_terminal_bank(13);
    ASSERT_TRUE(stale.issued_by(&pool, credit::CreditKind::TerminalBank));
    pool.reset();
    EXPECT_FALSE(stale.issued_by(&pool, credit::CreditKind::TerminalBank));
}

// ---------------------------------------------------------------------------
// The admission/reset boundary
// ---------------------------------------------------------------------------
//
// A grant takes an index and stamps a generation. Those were two decisions, and
// a reset between them could invalidate the index while the stamp still read as
// current — a token for capacity the pool had already handed to someone else.
// The cases below drive a reset into the middle of a grant, which the pool's
// own interleave hook makes deterministic rather than hoping for a schedule.

namespace {

/** Runs one reset the first time a grant reaches the hook. */
struct ResetOnce {
    credit::RunCredits *pool{nullptr};
    int fired{0};

    static void run(void *ctx) {
        auto *self = static_cast<ResetOnce *>(ctx);
        if (self->fired++ != 0) return;
        self->pool->reset();
    }
};

}  // namespace

TEST(DiagnosticRunCredit, AResetAcrossAGrantRefusesThatGrant) {
    credit::RunCredits pool;
    ResetOnce hook{&pool, 0};
    pool.set_grant_interleave_for_test(&ResetOnce::run, &hook);

    // The review's first order: the claim lands, the reset then advances the
    // generation and sweeps the index, and the grant must not come back holding
    // it. A token here would name capacity the pool no longer owns.
    const credit::Reservation grant = pool.reserve_terminal_bank(101);
    ASSERT_EQ(hook.fired, 1) << "the interleave did not run inside the grant";
    EXPECT_FALSE(grant.held()) << "a grant the reset swept was returned as valid";
    EXPECT_EQ(pool.banks_in_use(), 0u) << "the swept index was left occupied";
    EXPECT_EQ(pool.bank_refusals(), 1u);

    // And the pool is usable afterwards: the refusal is this grant's, not a
    // permanent state.
    pool.set_grant_interleave_for_test(nullptr, nullptr);
    const credit::Reservation after = pool.reserve_terminal_bank(102);
    EXPECT_TRUE(after.held());
    EXPECT_TRUE(after.issued_by(&pool, credit::CreditKind::TerminalBank));
}

TEST(DiagnosticRunCredit, AResetAcrossAGrantCannotDuplicateTheIndexItHandsOn) {
    credit::RunCredits pool;

    // The review's second order: a successor takes the same index while the
    // first grant is still inside its own window. Exactly one of them may come
    // back holding it.
    //
    // The successor's token is kept in this struct, whose lifetime ends inside
    // the test and before the pool's — a token must never outlive the pool it
    // names, because its destructor releases through that pointer.
    struct ResetThenTake {
        credit::RunCredits *pool{nullptr};
        int fired{0};
        bool successor_held{false};
        size_t successor_index{0};
        credit::Reservation successor{};

        static void run(void *ctx) {
            auto *self = static_cast<ResetThenTake *>(ctx);
            self->fired++;
            // Disarmed before the nested grant, so this runs exactly once and
            // the inner reservation is an ordinary uninterleaved one.
            self->pool->set_grant_interleave_for_test(nullptr, nullptr);
            self->pool->reset();
            // The successor the sweep has made room for, granted while the
            // outer grant is suspended inside its own window.
            self->successor = self->pool->reserve_terminal_bank(202);
            self->successor_held = self->successor.held();
            if (self->successor.held()) self->successor_index = self->successor.index();
        }
    } hook{&pool, 0, false, 0, {}};

    pool.set_grant_interleave_for_test(&ResetThenTake::run, &hook);
    {
        const credit::Reservation first = pool.reserve_terminal_bank(201);
        pool.set_grant_interleave_for_test(nullptr, nullptr);

        ASSERT_EQ(hook.fired, 1) << "the interleave ran more than once";
        ASSERT_TRUE(hook.successor_held) << "the successor could not be granted after the sweep";
        EXPECT_FALSE(first.held()) << "two tokens were issued for the same index across a reset";
        // The successor still holds its index, which the refused grant did not
        // give away on its way out.
        EXPECT_EQ(pool.banks_in_use(), 1u);
        EXPECT_EQ(hook.successor_index, 0u);
    }
    // Released here, while the pool it names is still alive.
    hook.successor.release();
    EXPECT_EQ(pool.banks_in_use(), 0u);
}

TEST(DiagnosticRunCredit, ASweepLeavesTheGateEvenAndAdvancesOneGeneration) {
    credit::RunCredits pool;
    const uint64_t before = pool.generation();
    EXPECT_FALSE(pool.resetting());

    pool.reset();

    // Two bumps of the sweep bit bracket the sweep, so a grant on either side
    // of it sees a different gate — and they leave the bit clear and the
    // generation exactly one step on, which is what `issued_by` compares.
    EXPECT_FALSE(pool.resetting());
    EXPECT_EQ(pool.generation(), before + 1);
    const credit::Reservation after = pool.reserve_terminal_bank(301);
    ASSERT_TRUE(after.held());
    EXPECT_EQ(after.generation(), before + 1);
    EXPECT_TRUE(after.issued_by(&pool, credit::CreditKind::TerminalBank));
}

TEST(DiagnosticRunCredit, ConcurrentReservationNeverOverCommits) {
    credit::RunCredits pool;
    std::atomic<size_t> granted{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&pool, &granted] {
            credit::Reservation got = pool.reserve_terminal_bank(42);
            if (got.held()) {
                granted.fetch_add(1, std::memory_order_relaxed);
                got.release();
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    EXPECT_GE(granted.load(), 1u);
    EXPECT_EQ(pool.banks_in_use(), 0u);
}

// ---------------------------------------------------------------------------
// Reconciliation: every expected participant, conserving per producer
// ---------------------------------------------------------------------------

TEST(DiagnosticRunReconcile, EveryExpectedParticipantPublishingIsWhatPublishes) {
    auto bank = make_bank(kStride);
    publish(bank[0], 9, clean_counts(12));
    publish(bank[1], 9, clean_counts(4));

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 9, 16);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_EQ(out.missing_producers, 0);
    EXPECT_EQ(out.expected_producers, 2);
    EXPECT_EQ(out.produced, 16u);
    EXPECT_EQ(out.received, 16u);
    EXPECT_EQ(out.unaccounted, 0u);
}

TEST(DiagnosticRunReconcile, AParticipantWithNothingToReportStillPublishes) {
    auto bank = make_bank(kStride);
    publish(bank[0], 9, clean_counts(6));
    publish(bank[1], 9, clean_counts(0));  // brought up, recorded nothing

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 9, 6);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_EQ(out.missing_producers, 0);
    EXPECT_EQ(out.produced, 6u);
}

TEST(DiagnosticRunReconcile, AClaimedParticipantThatNeverSettledIsNotACompleteRun) {
    auto bank = make_bank(kStride);
    publish(bank[0], 7, clean_counts(4));
    // bank[1] claimed this run at its init and never reached its close — a
    // producer killed mid-run. Its own entry says so, which is what stops the
    // survivors being read as the whole run.
    claim(bank[1], 7);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 7, 4);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown) << "a partial producer set reported a complete run";
    EXPECT_TRUE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, 2) << "a claimed participant was not counted as one";
}

TEST(DiagnosticRunReconcile, AProducerThisRunNeverBroughtUpIsNotAParticipant) {
    // The shape every real run has: a collector is dimensioned by the producers
    // the platform could have, and the device brings up fewer — a scheduler
    // that keeps one AICPU thread for orchestration runs one fewer dump
    // producer than the configured thread count. The lane it never started
    // claimed nothing, so it is not this run's participant and its silence is
    // not a loss. Reading it as one made every such run incomplete.
    auto bank = make_bank(kStride);
    publish(bank[0], 9, clean_counts(5));

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 9, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::Published) << "a producer this run never started was reported missing";
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, 1);
    EXPECT_EQ(out.expected_producers, 2) << "the dimension is still reported, it just decides nothing";
    EXPECT_EQ(out.produced, 5u);
}

TEST(DiagnosticRunReconcile, AnotherRunsOccupantIsFilteredByIdentityNotByPosition) {
    auto bank = make_bank(kStride);
    publish(bank[0], 9, clean_counts(3));
    // A slot still holding whatever its last user left. Identity is what keeps
    // it out — reading it would import another run's figures into this one —
    // and that holds wherever in the array it sits.
    publish(bank[2], 8, clean_counts(99));

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 3);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_EQ(out.reported_producers, 1);
    EXPECT_EQ(out.produced, 3u);
}

TEST(DiagnosticRunReconcile, APredecessorsOccupantIsNotThisRunsEvidence) {
    auto bank = make_bank(kStride);
    publish(bank[0], 8, clean_counts(12));  // the run that held this bank before

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
    EXPECT_FALSE(out.terminal_seen);
    EXPECT_EQ(out.missing_producers, 1);
    EXPECT_EQ(out.produced, 0u);
}

TEST(DiagnosticRunReconcile, AnUnpublishedTerminalIsUnknownNotEmpty) {
    const auto bank = make_bank(kStride);
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, AnExpectedCountOutsideTheBankIsRefused) {
    const auto bank = make_bank(kStride);
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, kStride + 1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, ConservationIsCheckedPerProducerNotAcrossTheRun) {
    auto bank = make_bank(kStride);
    // One producer claims a record it never accounts for; another claims one
    // more published than it produced. The two cancel in the aggregate and
    // must not cancel in the verdict.
    DiagnosticRunTerminalCounts short_by_one{};
    short_by_one.total = 5;
    short_by_one.published_records = 4;
    short_by_one.published_buffers = 1;
    DiagnosticRunTerminalCounts over_by_one{};
    over_by_one.total = 4;
    over_by_one.published_records = 5;
    over_by_one.published_buffers = 1;
    publish(bank[0], 9, short_by_one);
    publish(bank[1], 9, over_by_one);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 9, 9);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown) << "per-producer contradictions cancelled in an aggregate";
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, AProducerThatDoesNotConserveIsUnknown) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts{};
    counts.total = 5;
    counts.published_records = 4;  // and nothing dropped, nothing left live
    counts.published_buffers = 1;
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 4);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown) << "a record that went nowhere was reported as a clean run";
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, ContradictoryCountersDescribeNoRun) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts{};
    counts.total = 4;
    counts.published_records = 5;  // more delivered than appended
    counts.published_buffers = 1;
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, ReceiptAboveWhatTheDeviceCommittedIsUnknown) {
    auto bank = make_bank(kStride);
    publish(bank[0], 9, clean_counts(6));

    // A receipt count covering another identity's records is not this run's.
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 7);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunReconcile, LossIsAttributedAndNeverReconstructed) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts{};
    counts.total = 10;
    counts.dropped = 2;
    counts.published_records = 8;
    counts.published_buffers = 2;
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_EQ(out.dropped, 2u);
    EXPECT_EQ(out.unaccounted, 3u);  // committed 8, host took 5
    EXPECT_EQ(out.received, 5u);
}

TEST(DiagnosticRunReconcile, AWrittenThenLostRecordIsCountedOnce) {
    // The shape an ArgsDump producer publishes when a commit fails: the record
    // was written, so it is in `total` once, and losing it with its buffer puts
    // it in `dropped` — not in `total` a second time. Counting it twice would
    // make a reported loss read as a producer that cannot account for itself.
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts{};
    counts.total = 1;    // one record written
    counts.dropped = 1;  // and lost with the buffer it was written into
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_FALSE(out.counts_unknown) << "a written-then-lost record failed conservation";
    EXPECT_EQ(out.dropped, 1u);
    EXPECT_EQ(out.published, 0u);

    // The same record counted twice is what the defect looked like, and it is
    // refused rather than reported as a clean partial.
    auto double_counted = make_bank(kStride);
    DiagnosticRunTerminalCounts inflated = counts;
    inflated.total = 2;
    publish(double_counted[0], 9, inflated);
    const credit::RunReconciliation bad = credit::reconcile_run(double_counted.data(), kStride, 1, 9, 0);
    EXPECT_EQ(bad.verdict, runs::Verdict::CutUnknown);
}

TEST(DiagnosticRunReconcile, AHeldBufferAtCloseIsNotAClosedSet) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts{};
    counts.total = 7;
    counts.published_records = 4;
    counts.published_buffers = 1;
    counts.live_at_close = 3;
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 4);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_EQ(out.unsettled, 3u);
}

TEST(DiagnosticRunReconcile, AFatalTravelsBesideCountsThatStillBalance) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts = clean_counts(5);
    counts.fatal_latched = 1;
    publish(bank[0], 9, counts);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::PartialSafe);
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_TRUE(out.fatal_latched);
}

TEST(DiagnosticRunReconcile, PayloadBytesTravelWithTheRunThatProducedThem) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts counts = clean_counts(2);
    counts.published_payload_bytes = 4096;
    publish(bank[0], 9, counts);

    ASSERT_EQ(bank[0].published_payload_bytes, 4096u);
    ASSERT_EQ(bank[0].run_epoch, 9u);
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/1, 9, 2);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
}

// ---------------------------------------------------------------------------
// Saturation: a figure at its width is unknown, not a balance
// ---------------------------------------------------------------------------

TEST(DiagnosticRunSaturation, ACounterAtItsWidthIsStickyUnknownWhenStored) {
    DiagnosticRunTerminalCounts counts = clean_counts(0);
    counts.total = UINT32_MAX;
    counts.published_records = UINT32_MAX;
    EXPECT_EQ(diagnostic_terminal_counts_saturated(&counts), 1);

    auto bank = make_bank(1);
    publish(bank[0], 9, counts);
    EXPECT_NE(bank[0].counts_unknown, 0u) << "the producer's own saturation did not travel with the entry";

    // It conserves arithmetically — UINT32_MAX appended, UINT32_MAX published —
    // and is still unusable, which is the whole point of the flag.
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), 1, 1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunSaturation, ASumFormedPastItsWidthIsUnknownNotWrapped) {
    // Two terms each below the sentinel whose sum is not: the clamp is what
    // keeps the combined figure from reading as a small number.
    const uint32_t half = UINT32_MAX / 2 + 2;
    EXPECT_EQ(diagnostic_terminal_add_u32(half, half), UINT32_MAX);
    EXPECT_EQ(diagnostic_terminal_add_u32(UINT32_MAX, 1), UINT32_MAX);
    EXPECT_EQ(diagnostic_terminal_add_u32(5, 7), 12u);

    EXPECT_EQ(diagnostic_terminal_add_u64(UINT64_MAX - 1, 4), UINT64_MAX);
    EXPECT_EQ(diagnostic_terminal_add_u64(1024, 1024), 2048u);
}

TEST(DiagnosticRunSaturation, PayloadBytesAtTheirWidthAreUnknown) {
    DiagnosticRunTerminalCounts counts = clean_counts(2);
    counts.published_payload_bytes = UINT64_MAX;
    EXPECT_EQ(diagnostic_terminal_counts_saturated(&counts), 1);

    auto bank = make_bank(1);
    publish(bank[0], 9, counts);
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), 1, 1, 9, 2);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
}

TEST(DiagnosticRunSaturation, ACommitCountAtItsWidthDoesNotCondemnTheRun) {
    // published_buffers is a count of commits, not a term in the conservation
    // equation, so saturating it costs a reader nothing it uses.
    DiagnosticRunTerminalCounts counts = clean_counts(4);
    counts.published_buffers = UINT32_MAX;
    EXPECT_EQ(diagnostic_terminal_counts_saturated(&counts), 0);

    auto bank = make_bank(1);
    publish(bank[0], 9, counts);
    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), 1, 1, 9, 4);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
}

TEST(DiagnosticRunSaturation, OneSaturatedProducerCondemnsTheWholeRun) {
    auto bank = make_bank(kStride);
    DiagnosticRunTerminalCounts saturated = clean_counts(0);
    saturated.total = UINT32_MAX;
    saturated.published_records = UINT32_MAX;
    publish(bank[0], 9, clean_counts(3));
    publish(bank[1], 9, saturated);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), kStride, /*expected=*/2, 9, 3);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

// ---------------------------------------------------------------------------
// Wire layout
// ---------------------------------------------------------------------------

TEST(DiagnosticRunTerminalLayout, BankMajorKeepsOneRunsProducersContiguous) {
    std::vector<DiagnosticRunTerminal> banks(
        static_cast<size_t>(kStride) * static_cast<size_t>(PLATFORM_DIAGNOSTIC_TERMINAL_BANKS)
    );
    volatile DiagnosticRunTerminal *base = banks.data();

    EXPECT_EQ(diagnostic_terminal_entry(base, kStride, 0, 0), base);
    EXPECT_EQ(diagnostic_terminal_entry(base, kStride, 0, kStride - 1), base + (kStride - 1));
    EXPECT_EQ(diagnostic_terminal_entry(base, kStride, 1, 0), base + kStride);
    EXPECT_EQ(
        calc_diagnostic_terminal_bank_bytes(kStride),
        static_cast<size_t>(kStride) * PLATFORM_DIAGNOSTIC_TERMINAL_BANKS * sizeof(DiagnosticRunTerminal)
    );
}

TEST(DiagnosticRunTerminalLayout, ResetLeavesNothingAttributable) {
    auto bank = make_bank(1);
    publish(bank[0], 9, clean_counts(3));
    ASSERT_EQ(bank[0].run_epoch, 9u);

    diagnostic_terminal_reset(&bank[0]);
    EXPECT_EQ(bank[0].run_epoch, 0u);
    EXPECT_EQ(bank[0].total, 0u);
    EXPECT_EQ(bank[0].published_records, 0u);
    EXPECT_EQ(bank[0].counts_unknown, 0u);

    const credit::RunReconciliation out = credit::reconcile_run(bank.data(), 1, 1, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
}

TEST(DiagnosticRunTerminalLayout, StoringCountsLeavesTheIdentityUnpublished) {
    auto bank = make_bank(1);
    const DiagnosticRunTerminalCounts counts = clean_counts(3);
    diagnostic_terminal_store_counts(&bank[0], &counts);

    EXPECT_EQ(bank[0].run_epoch, 0u);
    EXPECT_EQ(bank[0].total, 3u);
    const credit::RunReconciliation before = credit::reconcile_run(bank.data(), 1, 1, 9, 0);
    EXPECT_FALSE(before.terminal_seen);

    diagnostic_terminal_commit(&bank[0], 9);
    const credit::RunReconciliation after = credit::reconcile_run(bank.data(), 1, 1, 9, 3);
    EXPECT_TRUE(after.terminal_seen);
    EXPECT_EQ(after.verdict, runs::Verdict::Published);
}

// ---------------------------------------------------------------------------
// Host-side bank ownership
// ---------------------------------------------------------------------------

namespace {

/** A device the host allocates from, copies out of, and can be made to fail. */
/**
 * A device allocation whose address is deliberately not its host address.
 *
 * The distinction is the point. An allocator hands back a device address, and
 * whether the host may dereference it is a separate fact established by a
 * mapping step — so a fixture whose device pointer is just a `malloc` result
 * cannot tell a reader that reads the mapping from one that reads the raw
 * device address, and both pass. Here the device address is the storage offset
 * by `kDeviceOffset` and is never dereferenced by anything: a read that uses it
 * has to go through the copy, which is what a real device-to-host operation is.
 */
struct FakeDevice {
    std::vector<DiagnosticRunTerminal> storage;
    bool dirty_allocation{false};
    bool seed_phantom_epochs{false};
    bool initialize_fails{false};
    bool initialize_throws{false};
    bool h2d_fails{false};
    bool omit_h2d{false};
    int initializations{0};
    int h2d_copies{0};
    credit::RunTerminalBanks *opening_banks{nullptr};
    std::vector<char> cleanup_order;
    // Whether this allocation was mapped for host access. False models the raw
    // device allocation an unregistered buffer is.
    bool host_mapped{true};
    bool copy_fails{false};
    bool alloc_fails{false};
    int free_rc{0};
    int unmap_rc{0};
    int allocations{0};
    int releases{0};
    int unmaps{0};
    // Which address the last read used, so a case can assert that rather than
    // only its result.
    bool read_through_host_view{false};

    static constexpr uintptr_t kDeviceOffset = 0x4000'0000ull;

    void *device_address() {
        return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(storage.data()) + kDeviceOffset);
    }
    static const void *resolve(const void *device_ptr) {
        return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(device_ptr) - kDeviceOffset);
    }

    static void *alloc(void *owner, size_t bytes, void **host_out) {
        auto *self = static_cast<FakeDevice *>(owner);
        *host_out = nullptr;
        if (self->alloc_fails) return nullptr;
        self->storage.resize(bytes / sizeof(DiagnosticRunTerminal));
        std::memset(self->storage.data(), self->dirty_allocation ? 0x5a : 0, bytes);
        if (self->seed_phantom_epochs) {
            constexpr int stride = 72;
            for (int bank = 0; bank < PLATFORM_DIAGNOSTIC_TERMINAL_BANKS; ++bank) {
                for (int producer : {28, 53}) {
                    auto *entry = self->entry_at(stride, bank, producer);
                    diagnostic_terminal_reset(entry);
                    DiagnosticRunTerminalCounts counts{};
                    counts.total = producer == 28 ? 2025 : 12;
                    counts.dropped = producer == 28 ? 917506 : 0;
                    diagnostic_terminal_store_counts(entry, &counts);
                    diagnostic_terminal_commit(entry, 1);
                }
            }
        }
        self->allocations++;
        if (self->host_mapped) *host_out = self->storage.data();
        return self->device_address();
    }
    static int unmap(void *owner, void *) {
        auto *self = static_cast<FakeDevice *>(owner);
        self->unmaps++;
        self->cleanup_order.push_back('U');
        return self->unmap_rc;
    }
    static int release(void *owner, void *) {
        auto *self = static_cast<FakeDevice *>(owner);
        self->releases++;
        self->cleanup_order.push_back('F');
        return self->free_rc;
    }
    static int copy(void *owner, void *host_dst, const void *device_src, const void *host_src, size_t bytes) {
        auto *self = static_cast<FakeDevice *>(owner);
        if (self->copy_fails) return -1;
        self->read_through_host_view = host_src != nullptr;
        // A mapped allocation is read where it lies; an unmapped one only
        // through this call, which is what stands in for the platform's own
        // device-to-host operation.
        const void *src = host_src != nullptr ? host_src : resolve(device_src);
        std::memcpy(host_dst, src, bytes);
        return 0;
    }

    static int initialize(void *owner, void *device_dst, void *host_dst, const void *initial, size_t bytes) {
        auto *self = static_cast<FakeDevice *>(owner);
        self->initializations++;
        if (self->opening_banks != nullptr) {
            EXPECT_FALSE(self->opening_banks->opened());
            EXPECT_FALSE(self->opening_banks->can_reserve());
            EXPECT_FALSE(self->opening_banks->reserve(1).held());
            EXPECT_EQ(self->opening_banks->device_address(), 0u);
            EXPECT_EQ(self->opening_banks->bank_address(0), 0u);
        }
        if (self->initialize_throws) throw std::runtime_error("initialization callback failed");
        if (self->initialize_fails) return -1;
        std::function<int(void *, const void *, size_t)> h2d;
        if (!self->omit_h2d) {
            h2d = [self](void *dst, const void *src, size_t count) {
                self->h2d_copies++;
                auto *actual = const_cast<void *>(resolve(dst));
                std::memcpy(actual, src, self->h2d_fails ? sizeof(DiagnosticRunTerminal) : count);
                return self->h2d_fails ? -1 : 0;
            };
        }
        return credit::initialize_run_terminal_bank(device_dst, host_dst, initial, bytes, h2d);
    }

    /** Publish as the producer would, into the storage behind the allocation. */
    void
    publish_into(int stride, size_t bank, int producer, uint64_t epoch, const DiagnosticRunTerminalCounts &counts) {
        diagnostic_terminal_store_counts(entry_at(stride, bank, producer), &counts);
        diagnostic_terminal_commit(entry_at(stride, bank, producer), epoch);
    }

    /** Claim as the producer would at its run init, before it has collected. */
    void claim_into(int stride, size_t bank, int producer, uint64_t epoch) {
        diagnostic_terminal_claim(entry_at(stride, bank, producer), epoch);
    }

    /** Write one producer's observation, as its own init does after the claim. */
    void note_into(int stride, size_t bank, int producer, uint32_t note) {
        entry_at(stride, bank, producer)->producer_note = note;
    }

    volatile DiagnosticRunTerminal *entry_at(int stride, size_t bank, int producer) {
        auto *base = reinterpret_cast<volatile DiagnosticRunTerminal *>(storage.data());
        return diagnostic_terminal_entry(base, stride, static_cast<int>(bank), producer);
    }
};

void open_banks(credit::RunTerminalBanks &banks, FakeDevice &device, int stride, int expected) {
    ASSERT_TRUE(banks.open(
        stride, expected, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
}

}  // namespace

TEST(DiagnosticRunBanks, DefaultPathMapsNothingAndAdmitsNoRetainedRun) {
    credit::RunTerminalBanks banks;
    EXPECT_FALSE(banks.opened());
    EXPECT_EQ(banks.device_address(), 0u);
    // A run that cannot reserve a bank stays off the retained path instead of
    // publishing into an address nobody mapped.
    EXPECT_FALSE(banks.reserve(5).held());
}

TEST(DiagnosticRunBanks, OpenAllocatesOneArrayAndAProvedStopReleasesIt) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 2);
    EXPECT_TRUE(banks.opened());
    EXPECT_NE(banks.device_address(), 0u);
    EXPECT_EQ(banks.bytes(), calc_diagnostic_terminal_bank_bytes(kStride));
    EXPECT_EQ(banks.stride(), kStride);
    EXPECT_EQ(banks.expected_producers(), 2);
    EXPECT_EQ(device.allocations, 1);

    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::Released);
    EXPECT_EQ(device.releases, 1);
    EXPECT_FALSE(banks.backing_retained());
    // Idempotent, so a teardown path may call it twice without freeing twice.
    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::NotOpen);
    EXPECT_EQ(device.releases, 1);
}

TEST(DiagnosticRunBanks, AnUnprovedDeviceStopRetainsTheBackingRatherThanFreeingIt) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    // The one thing that must not happen on an unproved path: handing this
    // memory back while a producer may still be writing it.
    EXPECT_EQ(banks.release_backing(/*device_stopped=*/false), credit::RunTerminalBanks::Release::DeviceUnproved);
    EXPECT_EQ(device.releases, 0);
    EXPECT_TRUE(banks.backing_retained());
    EXPECT_NE(banks.retained_backing_address(), 0u);
    // Admission is closed all the same: nothing may be settled against it.
    EXPECT_FALSE(banks.opened());
    EXPECT_FALSE(banks.reserve(1).held());
}

TEST(DiagnosticRunBanks, AFailedFreeIsReportedAndRetainsTheBacking) {
    FakeDevice device;
    device.free_rc = -1;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::FreeFailed);
    EXPECT_EQ(device.releases, 1) << "the allocator was asked";
    EXPECT_TRUE(banks.backing_retained()) << "an allocator that refused the pointer may still own it";
}

TEST(DiagnosticRunBanks, RetainedBackingIsNotFreeSpaceToReopenOver) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);
    const uint64_t first = banks.device_address();
    ASSERT_EQ(banks.release_backing(/*device_stopped=*/false), credit::RunTerminalBanks::Release::DeviceUnproved);
    ASSERT_TRUE(banks.backing_retained());
    ASSERT_FALSE(banks.opened());

    // `opened()` is false, and reopening would still overwrite the handle to
    // memory a producer may be writing and lose the only record of it.
    EXPECT_FALSE(banks.open(
        kStride, 1, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(banks.retained_backing_address(), first);
}

// --- Which address a bank is read through ------------------------------

TEST(DiagnosticRunBanks, AMappedArrayIsReadThroughItsHostViewAndNotItsDeviceAddress) {
    // The distinction this whole path turns on. The allocation has a host view,
    // so the read uses it; the device address is the producer's and is never
    // dereferenced here. A reader that took the device address instead would
    // be reading memory this process never mapped — which on a platform whose
    // allocator does not register is a fault, not a wrong number.
    FakeDevice device;
    device.host_mapped = true;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);
    ASSERT_NE(banks.device_address(), reinterpret_cast<uint64_t>(device.storage.data()))
        << "the fixture must model a device address distinct from its host view";

    DiagnosticRunTerminalCounts counts{};
    counts.total = 4;
    counts.published_records = 4;
    device.publish_into(kStride, /*bank=*/0, /*producer=*/0, /*epoch=*/7001, counts);

    credit::Reservation bank = banks.reserve(7001);
    ASSERT_TRUE(bank.held());
    const credit::RunReconciliation settled = banks.settle(bank, 7001, /*received=*/4);

    EXPECT_TRUE(device.read_through_host_view) << "a mapped array was read through its device address";
    EXPECT_TRUE(settled.terminal_seen);
    EXPECT_FALSE(settled.counts_unknown);
    EXPECT_EQ(settled.produced, 4u);
}

TEST(DiagnosticRunBanks, AnUnmappedArrayIsReachedOnlyByTheDeviceCopy) {
    // The same allocation with no host view: there is nothing to dereference,
    // so the platform's own device-to-host operation is the only way to the
    // figures. The result must be the same accounting, by the other route.
    FakeDevice device;
    device.host_mapped = false;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    DiagnosticRunTerminalCounts counts{};
    counts.total = 2;
    counts.published_records = 2;
    device.publish_into(kStride, /*bank=*/0, /*producer=*/0, /*epoch=*/7101, counts);

    credit::Reservation bank = banks.reserve(7101);
    ASSERT_TRUE(bank.held());
    const credit::RunReconciliation settled = banks.settle(bank, 7101, /*received=*/2);

    EXPECT_FALSE(device.read_through_host_view) << "an unmapped array was read as though it had a host view";
    EXPECT_TRUE(settled.terminal_seen);
    EXPECT_EQ(settled.produced, 2u);
}

TEST(DiagnosticRunBanks, AReadRuleWithNeitherAddressFailsRatherThanMirroringZeros) {
    // The shared rule on its own, because its failure mode is the dangerous
    // one: a read that quietly succeeds while copying nothing leaves the mirror
    // zeroed, and a zeroed mirror is indistinguishable from producers that
    // never published. Refusing is what turns that into a reported unknown.
    DiagnosticRunTerminal mirror{};
    diagnostic_terminal_reset(&mirror);
    EXPECT_NE(
        simpler::dfx::credit::read_run_terminal_bank(
            &mirror, /*device_src=*/nullptr, /*host_src=*/nullptr, sizeof(mirror), /*copy_from_device=*/{}
        ),
        0
    );

    // And with a host view it is that view the bytes come from, whatever the
    // device address says.
    DiagnosticRunTerminal mapped{};
    diagnostic_terminal_reset(&mapped);
    mapped.run_epoch = 4242;
    EXPECT_EQ(
        simpler::dfx::credit::read_run_terminal_bank(
            &mirror, /*device_src=*/nullptr, &mapped, sizeof(mapped), /*copy_from_device=*/{}
        ),
        0
    );
    EXPECT_EQ(mirror.run_epoch, 4242u);
}

TEST(DiagnosticRunBanks, AMappingTakenDownOnceIsNotTakenDownAgainByARetry) {
    // Release is two steps with two outcomes. An unmap that succeeded and a
    // free that then failed leave the array held with no host view, and the
    // retry that follows must free what is left rather than unmap a mapping
    // that is already gone.
    FakeDevice device;
    device.host_mapped = true;
    device.free_rc = -1;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::FreeFailed);
    EXPECT_EQ(device.unmaps, 1);
    EXPECT_EQ(device.releases, 1);
    EXPECT_TRUE(banks.backing_retained());

    device.free_rc = 0;
    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::Released);
    EXPECT_EQ(device.unmaps, 1) << "the retry took a mapping down that was already gone";
    EXPECT_EQ(device.releases, 2);
    EXPECT_FALSE(banks.backing_retained());
}

TEST(DiagnosticRunBanks, AFailedUnmapKeepsTheArrayAndDoesNotFreeIt) {
    // The other order: the mapping could not be taken down, so the memory is
    // still reachable from this process and must not be handed back.
    FakeDevice device;
    device.host_mapped = true;
    device.unmap_rc = -1;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    EXPECT_EQ(banks.release_backing(/*device_stopped=*/true), credit::RunTerminalBanks::Release::FreeFailed);
    EXPECT_EQ(device.unmaps, 1);
    EXPECT_EQ(device.releases, 0) << "memory still mapped into this process was given back";
    EXPECT_TRUE(banks.backing_retained());
}

TEST(DiagnosticRunBanks, DestructionLeavesAnUnprovedAllocationAlone) {
    FakeDevice device;
    {
        credit::RunTerminalBanks banks;
        open_banks(banks, device, kStride, 1);
    }
    // A destructor can prove no device stop and has nowhere to report a failed
    // free, so it frees nothing. The allocation outliving the object is the
    // reported outcome, not a silent one.
    EXPECT_EQ(device.releases, 0);
}

TEST(DiagnosticRunBanks, AnAlreadyOpenInstanceRefusesASecondOpen) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);
    const uint64_t first = banks.device_address();

    // Reopening would leak the first allocation and leave live tokens naming
    // banks inside it.
    EXPECT_FALSE(banks.open(
        kStride, 1, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
    EXPECT_EQ(banks.device_address(), first);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(device.releases, 0);
}

TEST(DiagnosticRunBanks, ARefusedDeviceAllocationLeavesNothingBehind) {
    FakeDevice device;
    device.alloc_fails = true;
    credit::RunTerminalBanks banks;
    EXPECT_FALSE(banks.open(
        kStride, 1, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
    EXPECT_FALSE(banks.opened());
    EXPECT_EQ(device.releases, 0);
    EXPECT_FALSE(banks.reserve(1).held());
}

TEST(DiagnosticRunBanks, AnInvalidParticipantCountIsRefusedBeforeAnyAllocation) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    EXPECT_FALSE(banks.open(
        kStride, kStride + 1, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
    EXPECT_FALSE(banks.open(
        kStride, 0, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
        &FakeDevice::initialize
    ));
    EXPECT_EQ(device.allocations, 0);
    EXPECT_FALSE(banks.opened());
}

TEST(DiagnosticRunBanks, SettleCopiesTheRunOutAndFreesTheBankForTheNextRun) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation reservation = banks.reserve(21);
    ASSERT_TRUE(reservation.held());
    DiagnosticRunTerminalCounts counts{};
    counts.total = 9;
    counts.published_records = 9;
    counts.published_buffers = 3;
    device.publish_into(kStride, reservation.index(), 0, 21, counts);

    const credit::RunReconciliation out = banks.settle(reservation, 21, 9);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_EQ(out.produced, 9u);
    EXPECT_FALSE(reservation.held()) << "settle consumed the token it released";
    // The bank is the next run's even though no artifact has been written.
    EXPECT_EQ(banks.credits().banks_in_use(), 0u);
    EXPECT_TRUE(banks.reserve(22).held());
}

TEST(DiagnosticRunBanks, SettleRequiresEveryParticipantThatClaimedTheRunItNames) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/2);

    credit::Reservation reservation = banks.reserve(23);
    ASSERT_TRUE(reservation.held());
    device.publish_into(kStride, reservation.index(), 0, 23, clean_counts(5));
    // Producer 1 claimed this run at its init and never settled, which its own
    // entry reports.
    device.claim_into(kStride, reservation.index(), 1, 23);

    const credit::RunReconciliation out = banks.settle(reservation, 23, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, 2);
}

TEST(DiagnosticRunBanks, SettleReadsOnlyTheProducersThisRunBroughtUp) {
    // A bank dimensioned for two whose run started one. The silent entry is a
    // producer that never ran, not one that lost its accounting, and the run
    // settles on what did report.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/2);

    credit::Reservation reservation = banks.reserve(24);
    ASSERT_TRUE(reservation.held());
    device.publish_into(kStride, reservation.index(), 0, 24, clean_counts(5));

    const credit::RunReconciliation out = banks.settle(reservation, 24, 5);
    EXPECT_EQ(out.verdict, runs::Verdict::Published);
    EXPECT_FALSE(out.counts_unknown);
    EXPECT_EQ(out.reported_producers, 1);
}

TEST(DiagnosticRunBanks, AQuarantinedBankIsReclaimedByItsOwnIdentityAndNoOther) {
    // A bank parked because nothing proved its producers stopped. The identity
    // it is reclaimed by is the token's own, so a proof naming another run
    // reaches none of it — and the capacity stays out of the pool until the
    // right proof arrives, which is what "withheld, not abandoned" means.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation first = banks.reserve(41);
    credit::Reservation second = banks.reserve(42);
    ASSERT_TRUE(first.held());
    ASSERT_TRUE(second.held());
    EXPECT_TRUE(banks.quarantine(std::move(first)));
    EXPECT_TRUE(banks.quarantine(std::move(second)));
    EXPECT_EQ(banks.quarantined_banks(), 2u);
    EXPECT_EQ(banks.credits().banks_in_use(), 2u) << "a parked bank was handed back to the pool";

    EXPECT_FALSE(banks.reclaim(99)) << "a run with nothing parked reclaimed something";
    EXPECT_EQ(banks.quarantined_banks(), 2u);

    EXPECT_TRUE(banks.reclaim(41));
    EXPECT_EQ(banks.quarantined_banks(), 1u);
    EXPECT_EQ(banks.credits().banks_in_use(), 1u) << "the reclaimed bank did not return to the pool";
    EXPECT_FALSE(banks.reclaim(41)) << "the same proof returned the same bank twice";

    EXPECT_TRUE(banks.reclaim(42));
    EXPECT_EQ(banks.credits().banks_in_use(), 0u);
}

TEST(DiagnosticRunBanks, QuarantiningNothingIsNotAQuarantine) {
    // An empty token names no run and holds no capacity, so there is nothing
    // to park and nothing to report as parked.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation empty;
    EXPECT_FALSE(banks.quarantine(std::move(empty)));
    EXPECT_EQ(banks.quarantined_banks(), 0u);
    EXPECT_EQ(banks.credits().banks_in_use(), 0u);
}

TEST(DiagnosticRunBanks, SettleRefusesAForeignOrMismatchedToken) {
    FakeDevice device;
    FakeDevice other_device;
    credit::RunTerminalBanks banks;
    credit::RunTerminalBanks other;
    open_banks(banks, device, kStride, 1);
    open_banks(other, other_device, kStride, 1);

    credit::Reservation foreign = other.reserve(31);
    ASSERT_TRUE(foreign.held());
    const credit::RunReconciliation by_foreign = banks.settle(foreign, 31, 0);
    EXPECT_EQ(by_foreign.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(foreign.held()) << "another pool's token was consumed";
    EXPECT_EQ(other.credits().banks_in_use(), 1u) << "another pool's capacity was released";

    credit::Reservation slot = banks.credits().reserve_output_slot(32);
    ASSERT_TRUE(slot.held());
    const credit::RunReconciliation by_slot = banks.settle(slot, 32, 0);
    EXPECT_EQ(by_slot.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(slot.held()) << "an output slot was accepted as a terminal bank";

    credit::Reservation bank = banks.reserve(33);
    ASSERT_TRUE(bank.held());
    const credit::RunReconciliation by_other_run = banks.settle(bank, 34, 0);
    EXPECT_EQ(by_other_run.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(bank.held()) << "a token was settled against a run it does not name";
}

TEST(DiagnosticRunBanks, SettleRefusesATokenFromARebuiltPool) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation stale = banks.reserve(41);
    ASSERT_TRUE(stale.held());
    banks.credits().reset();

    const credit::RunReconciliation out = banks.settle(stale, 41, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunBanks, AFailedCopyReportsUnknownRatherThanAHalfReadBank) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation reservation = banks.reserve(51);
    ASSERT_TRUE(reservation.held());
    device.publish_into(kStride, reservation.index(), 0, 51, clean_counts(4));
    device.copy_fails = true;

    const credit::RunReconciliation out = banks.settle(reservation, 51, 4);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
    EXPECT_EQ(out.received, 4u);
    // The bank is still returned: holding it would deny capacity for a failure
    // that has already been reported.
    EXPECT_EQ(banks.credits().banks_in_use(), 0u);
}

TEST(DiagnosticRunBanks, SettlingWithoutAReservationIsUnknownNotEmpty) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);
    credit::Reservation none;
    const credit::RunReconciliation out = banks.settle(none, 9, 0);
    EXPECT_EQ(out.verdict, runs::Verdict::CutUnknown);
    EXPECT_TRUE(out.counts_unknown);
}

TEST(DiagnosticRunBanks, TwoRunsHoldDistinctBanksAndNeitherReadsTheOthers) {
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, 1);

    credit::Reservation first = banks.reserve(61);
    credit::Reservation second = banks.reserve(62);
    ASSERT_TRUE(first.held());
    ASSERT_TRUE(second.held());
    ASSERT_NE(first.index(), second.index());

    device.publish_into(kStride, first.index(), 0, 61, clean_counts(2));
    device.publish_into(kStride, second.index(), 0, 62, clean_counts(5));

    const credit::RunReconciliation older = banks.settle(first, 61, 2);
    const credit::RunReconciliation newer = banks.settle(second, 62, 5);
    EXPECT_EQ(older.produced, 2u);
    EXPECT_EQ(newer.produced, 5u);
    EXPECT_EQ(older.verdict, runs::Verdict::Published);
    EXPECT_EQ(newer.verdict, runs::Verdict::Published);
}

TEST(DiagnosticRunBanks, AProducerNoteIsCountedByItsCodeAndNeverSummed) {
    // Three producers of one run, two reporting the same code and one another.
    // What a reader may say is how many producers said which thing: the payloads
    // differ, so a sum of the words would be a number nothing in the system
    // means.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/3);

    credit::Reservation bank = banks.reserve(810);
    ASSERT_TRUE(bank.held());
    for (int producer = 0; producer < 3; producer++) {
        device.publish_into(kStride, bank.index(), producer, 810, clean_counts(1));
    }
    device.note_into(kStride, bank.index(), 0, diagnostic_terminal_make_note(2, 0x7));
    device.note_into(kStride, bank.index(), 1, diagnostic_terminal_make_note(2, 0x5));
    device.note_into(kStride, bank.index(), 2, diagnostic_terminal_make_note(3, 0x0));

    const credit::RunReconciliation out = banks.settle(bank, 810, /*received=*/3);
    EXPECT_EQ(out.reported_producers, 3);
    EXPECT_EQ(out.notes_by_code[2], 2);
    EXPECT_EQ(out.notes_by_code[3], 1);
    EXPECT_EQ(out.notes_by_code[0], 0) << "a producer that reported a note is not also counted as silent";
    EXPECT_EQ(diagnostic_terminal_note_payload(out.first_note), 0x7u)
        << "the quoted payload is the lowest-numbered reporting producer's own word";
}

TEST(DiagnosticRunBanks, ANoteBelongsToTheRunWhoseEpochItsEntryCarries) {
    // The same storage, two runs. A note written under one identity must not be
    // readable as the other's, which is the whole reason it lives in the entry
    // the epoch filter already selects.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/1);

    credit::Reservation first = banks.reserve(901);
    ASSERT_TRUE(first.held());
    device.publish_into(kStride, first.index(), 0, 901, clean_counts(1));
    device.note_into(kStride, first.index(), 0, diagnostic_terminal_make_note(3, 0x11));
    const credit::RunReconciliation older = banks.settle(first, 901, /*received=*/1);
    ASSERT_EQ(older.notes_by_code[3], 1);

    // The successor claims the same entry: the claim is what makes it this
    // run's, so the predecessor's observation is gone rather than inherited.
    credit::Reservation second = banks.reserve(902);
    ASSERT_TRUE(second.held());
    device.claim_into(kStride, second.index(), 0, 902);
    device.publish_into(kStride, second.index(), 0, 902, clean_counts(1));
    const credit::RunReconciliation newer = banks.settle(second, 902, /*received=*/1);
    EXPECT_EQ(newer.reported_producers, 1);
    EXPECT_EQ(newer.notes_by_code[3], 0) << "the successor read its predecessor's note";
    EXPECT_EQ(newer.first_note, 0u);
}

TEST(DiagnosticRunBanks, AnUnreadableAccountingStillCarriesItsProducersNotes) {
    // A producer that claimed and never settled leaves `counts_unknown`, and the
    // note is not arithmetic: the observation it made at its init is still that
    // producer's statement about that run.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/1);

    credit::Reservation bank = banks.reserve(777);
    ASSERT_TRUE(bank.held());
    device.claim_into(kStride, bank.index(), 0, 777);
    device.note_into(kStride, bank.index(), 0, diagnostic_terminal_make_note(1, 0x0));

    const credit::RunReconciliation out = banks.settle(bank, 777, /*received=*/0);
    EXPECT_TRUE(out.counts_unknown) << "a claim with no close is unknown, as it was before";
    EXPECT_EQ(out.notes_by_code[1], 1);
}

TEST(DiagnosticRunBanks, ProducerNotesNameOnlyTheProducersThatRecorded) {
    // A run that claimed four producers and used two of them. A reader lining a
    // row up with the observation its own core made needs the ones that
    // recorded; a producer that claimed the run and executed nothing has no row
    // to line up and is left out.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/4);

    credit::Reservation bank = banks.reserve(5100);
    ASSERT_TRUE(bank.held());
    device.publish_into(kStride, bank.index(), 0, 5100, clean_counts(0));
    device.publish_into(kStride, bank.index(), 1, 5100, clean_counts(7));
    device.publish_into(kStride, bank.index(), 2, 5100, clean_counts(0));
    device.publish_into(kStride, bank.index(), 3, 5100, clean_counts(3));
    device.note_into(kStride, bank.index(), 1, diagnostic_terminal_make_note(2, 0x18));
    device.note_into(kStride, bank.index(), 3, diagnostic_terminal_make_note(2, 0x1A));

    credit::ProducerNote notes[8];
    int filled = 0;
    const credit::RunReconciliation out = banks.settle(bank, 5100, /*received=*/10, notes, 8, &filled);
    EXPECT_EQ(out.reported_producers, 4);
    ASSERT_EQ(filled, 2);
    EXPECT_EQ(notes[0].producer, 1);
    EXPECT_EQ(notes[0].records, 7u);
    EXPECT_EQ(diagnostic_terminal_note_payload(notes[0].note), 0x18u);
    EXPECT_EQ(notes[1].producer, 3);
    EXPECT_EQ(notes[1].records, 3u);
    EXPECT_EQ(diagnostic_terminal_note_payload(notes[1].note), 0x1Au);
}

TEST(DiagnosticRunBanks, ProducerNotesStopAtTheCallersBound) {
    // The bound is the caller's, because a reader quoting every producer of a
    // fully occupied device is not a bounded line.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/3);

    credit::Reservation bank = banks.reserve(5200);
    ASSERT_TRUE(bank.held());
    for (int producer = 0; producer < 3; producer++) {
        device.publish_into(kStride, bank.index(), producer, 5200, clean_counts(1));
        device.note_into(kStride, bank.index(), producer, diagnostic_terminal_make_note(3, 0x1u));
    }

    credit::ProducerNote notes[2];
    int filled = 0;
    (void)banks.settle(bank, 5200, /*received=*/3, notes, 2, &filled);
    EXPECT_EQ(filled, 2) << "the collector wrote past the capacity it was given";
    EXPECT_EQ(notes[0].producer, 0);
    EXPECT_EQ(notes[1].producer, 1);
}

TEST(DiagnosticRunBanks, ProducerNotesIgnoreAnotherRunsEntries) {
    // The same storage, a predecessor's figures still in it. The note collector
    // applies the identity filter itself rather than inheriting one, so an entry
    // carrying another epoch is not read.
    FakeDevice device;
    credit::RunTerminalBanks banks;
    open_banks(banks, device, kStride, /*expected=*/2);

    credit::Reservation first = banks.reserve(6100);
    ASSERT_TRUE(first.held());
    device.publish_into(kStride, first.index(), 0, 6100, clean_counts(4));
    device.note_into(kStride, first.index(), 0, diagnostic_terminal_make_note(2, 0x9));
    // Producer 1 of the same bank still carries an older run's snapshot.
    device.publish_into(kStride, first.index(), 1, 6099, clean_counts(4));
    device.note_into(kStride, first.index(), 1, diagnostic_terminal_make_note(3, 0x7));

    credit::ProducerNote notes[8];
    int filled = 0;
    const credit::RunReconciliation out = banks.settle(first, 6100, /*received=*/4, notes, 8, &filled);
    EXPECT_EQ(out.reported_producers, 1);
    ASSERT_EQ(filled, 1);
    EXPECT_EQ(notes[0].producer, 0);
    EXPECT_EQ(diagnostic_terminal_note_payload(notes[0].note), 0x9u);
}

TEST(DiagnosticRunBanks, InitializesDirtyMappedBackingBeforeFirstReservation) {
    FakeDevice device;
    device.dirty_allocation = true;
    credit::RunTerminalBanks banks;
    device.opening_banks = &banks;
    open_banks(banks, device, 72, 24);
    ASSERT_TRUE(banks.opened());
    EXPECT_EQ(device.initializations, 1);
    EXPECT_EQ(device.h2d_copies, 0);
    const auto *bytes = reinterpret_cast<const unsigned char *>(device.storage.data());
    EXPECT_TRUE(std::all_of(bytes, bytes + banks.bytes(), [](unsigned char value) {
        return value == 0;
    }));
    for (int bank = 0; bank < PLATFORM_DIAGNOSTIC_TERMINAL_BANKS; ++bank) {
        EXPECT_EQ(banks.bank_address(bank), banks.device_address() + bank * 72 * sizeof(DiagnosticRunTerminal));
    }
    EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
}

TEST(DiagnosticRunBanks, InitializesDirtyHostlessBackingBeforeFirstReservation) {
    FakeDevice device;
    device.host_mapped = false;
    device.dirty_allocation = true;
    credit::RunTerminalBanks banks;
    device.opening_banks = &banks;
    open_banks(banks, device, 72, 24);
    ASSERT_TRUE(banks.opened());
    EXPECT_EQ(device.initializations, 1);
    EXPECT_EQ(device.h2d_copies, 1);
    const auto *bytes = reinterpret_cast<const unsigned char *>(device.storage.data());
    EXPECT_TRUE(std::all_of(bytes, bytes + banks.bytes(), [](unsigned char value) {
        return value == 0;
    }));
    for (int bank = 0; bank < PLATFORM_DIAGNOSTIC_TERMINAL_BANKS; ++bank) {
        EXPECT_EQ(banks.bank_address(bank), banks.device_address() + bank * 72 * sizeof(DiagnosticRunTerminal));
    }
    EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
}

TEST(DiagnosticRunBanks, DirtyInactiveSlotsCannotForgeCurrentEpoch) {
    for (bool host_mapped : {true, false}) {
        SCOPED_TRACE(host_mapped);
        FakeDevice device;
        device.host_mapped = host_mapped;
        device.dirty_allocation = true;
        device.seed_phantom_epochs = true;
        credit::RunTerminalBanks banks;
        open_banks(banks, device, 72, 24);
        std::vector<credit::Reservation> reservations;
        for (int bank = 0; bank < PLATFORM_DIAGNOSTIC_TERMINAL_BANKS; ++bank) {
            reservations.push_back(banks.reserve(bank + 1));
            ASSERT_TRUE(reservations.back().held());
            ASSERT_EQ(reservations.back().index(), static_cast<size_t>(bank));
            for (int producer = 0; producer < 24; ++producer) {
                const uint32_t records = producer == 10 ? 40 : (producer == 12 ? 24 : 0);
                device.publish_into(72, bank, producer, bank + 1, clean_counts(records));
            }
        }
        for (int bank = 0; bank < PLATFORM_DIAGNOSTIC_TERMINAL_BANKS; ++bank) {
            credit::ProducerNote notes[72];
            int filled = 0;
            const auto settled = banks.settle(reservations[bank], bank + 1, 64, notes, 72, &filled);
            EXPECT_FALSE(settled.counts_unknown);
            EXPECT_EQ(settled.reported_producers, 24);
            EXPECT_EQ(settled.produced, 64u);
            EXPECT_EQ(settled.dropped, 0u);
            EXPECT_EQ(settled.unsettled, 0u);
            ASSERT_EQ(filled, 2);
            EXPECT_EQ(notes[0].producer, 10);
            EXPECT_EQ(notes[1].producer, 12);
        }
        EXPECT_EQ(device.initializations, 1);
        EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
    }
}

TEST(DiagnosticRunBanks, InitializationFailureReleasesUnpublishedBacking) {
    // Failure before the write, a partial H2D write, no H2D path, and a
    // throwing callback must all leave no address or credit available.
    for (int failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        FakeDevice device;
        device.host_mapped = failure == 0;
        device.dirty_allocation = true;
        device.initialize_fails = failure == 0;
        device.h2d_fails = failure == 1;
        device.omit_h2d = failure == 2;
        device.initialize_throws = failure == 3;
        credit::RunTerminalBanks banks;
        device.opening_banks = &banks;
        EXPECT_FALSE(banks.open(
            72, 24, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
            &FakeDevice::initialize
        ));
        EXPECT_FALSE(banks.opened());
        EXPECT_FALSE(banks.can_reserve());
        EXPECT_FALSE(banks.reserve(1).held());
        EXPECT_EQ(banks.device_address(), 0u);
        EXPECT_EQ(banks.bank_address(0), 0u);
        EXPECT_EQ(banks.credits().banks_in_use(), 0u);
        EXPECT_FALSE(banks.backing_retained());
        EXPECT_EQ(device.unmaps, device.host_mapped ? 1 : 0);
        EXPECT_EQ(device.releases, 1);
        EXPECT_EQ(device.cleanup_order, device.host_mapped ? std::vector<char>({'U', 'F'}) : std::vector<char>({'F'}));
        EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::NotOpen);
    }
}

TEST(DiagnosticRunBanks, InitializationFailureRetainsFailedUnmapOrFree) {
    for (int failure = 0; failure < 3; ++failure) {
        SCOPED_TRACE(failure);
        const bool fail_unmap = failure == 0;
        FakeDevice device;
        device.host_mapped = failure != 2;
        device.dirty_allocation = true;
        device.initialize_fails = true;
        device.unmap_rc = fail_unmap ? -1 : 0;
        device.free_rc = fail_unmap ? 0 : -1;
        credit::RunTerminalBanks banks;
        EXPECT_FALSE(banks.open(
            72, 24, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
            &FakeDevice::initialize
        ));
        ASSERT_TRUE(banks.backing_retained());
        EXPECT_EQ(banks.retained_backing_address(), reinterpret_cast<uint64_t>(device.device_address()));
        EXPECT_EQ(banks.device_address(), 0u);
        EXPECT_EQ(banks.bank_address(0), 0u);
        EXPECT_FALSE(banks.reserve(1).held());
        EXPECT_EQ(device.unmaps, device.host_mapped ? 1 : 0);
        EXPECT_EQ(device.releases, fail_unmap ? 0 : 1);
        EXPECT_FALSE(banks.open(
            72, 24, &device, &FakeDevice::alloc, &FakeDevice::unmap, &FakeDevice::release, &FakeDevice::copy,
            &FakeDevice::initialize
        ));
        EXPECT_EQ(device.allocations, 1);
        device.unmap_rc = 0;
        device.free_rc = 0;
        EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
        EXPECT_EQ(device.unmaps, fail_unmap ? 2 : (device.host_mapped ? 1 : 0));
        EXPECT_EQ(device.releases, fail_unmap ? 1 : 2);
        EXPECT_FALSE(banks.backing_retained());
        EXPECT_EQ(banks.retained_backing_address(), 0u);
        device.initialize_fails = false;
        open_banks(banks, device, 72, 24);
        EXPECT_TRUE(banks.opened());
        EXPECT_EQ(device.allocations, 2);
        EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
    }
}

TEST(DiagnosticRunBanks, InitializedBackingPreservesSparseClaimsAndUnclosedUnknown) {
    for (bool host_mapped : {true, false}) {
        SCOPED_TRACE(host_mapped);
        FakeDevice device;
        device.host_mapped = host_mapped;
        device.dirty_allocation = true;
        credit::RunTerminalBanks banks;
        open_banks(banks, device, 72, 24);
        auto first = banks.reserve(1);
        ASSERT_TRUE(first.held());
        const size_t bank = first.index();
        device.publish_into(72, bank, 53, 1, clean_counts(9));
        auto settled = banks.settle(first, 1, 9);
        EXPECT_FALSE(settled.counts_unknown);
        EXPECT_EQ(settled.reported_producers, 1);
        EXPECT_EQ(settled.produced, 9u);

        auto next = banks.reserve(2);
        ASSERT_TRUE(next.held());
        ASSERT_EQ(next.index(), bank);
        // Reservation cannot wipe a previously copied snapshot. Only the
        // producer's new epoch claim may replace its own entry.
        EXPECT_EQ(device.entry_at(72, bank, 53)->run_epoch, 1u);
        EXPECT_EQ(device.entry_at(72, bank, 53)->total, 9u);
        device.claim_into(72, bank, 28, 2);
        settled = banks.settle(next, 2, 0);
        EXPECT_TRUE(settled.counts_unknown);
        EXPECT_EQ(settled.reported_producers, 1);
        EXPECT_EQ(device.initializations, 1);
        EXPECT_EQ(banks.release_backing(true), credit::RunTerminalBanks::Release::Released);
    }
}
