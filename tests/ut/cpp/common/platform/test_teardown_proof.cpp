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
 * Which same-handle initializations the program entry admits, and what a
 * teardown has to have done to earn one.
 *
 * These drive the production rule itself: `simpler_init` calls
 * `may_initialize_context` for its refusal and `consume_teardown_proof` for
 * its one-shot spend, `finalize_device` calls `resolve_teardown_proof` on the
 * value its teardown returned, and the ordinary managed cleanup is the only
 * writer of `SweptClean`. There is no second copy of the transitions here —
 * the cases below name the states the way the entry sees them and assert on
 * the same functions the entry runs.
 *
 * What they cannot reach is the wiring: that `finalize_device` marks
 * `Unresolved` before its outstanding check and before `finish_retained_runs`,
 * that the cleanup writes `SweptClean` only after the sweep's lock is gone,
 * and that the entry consumes before it installs. The onboard runner and the C
 * entry are not compiled into this test tree (only the simulation runner is),
 * so those orderings are established by review of the call sites, not here.
 */

#include <gtest/gtest.h>

#include "host/teardown_proof.h"

namespace {

// The two states a handle can be admitted from, and the two it cannot.
constexpr bool kRunsOutstanding = true;
constexpr bool kNoRunsOutstanding = false;
constexpr bool kDeviceBound = true;
constexpr bool kNoDeviceBound = false;

TEST(TeardownProofGate, AFreshHandleIsAdmitted) {
    // Never initialized: nothing owns anything, so there is nothing to lose.
    EXPECT_TRUE(may_initialize_context(TeardownProof::NotAttempted, kNoRunsOutstanding, kNoDeviceBound));
}

TEST(TeardownProofGate, ALiveHandleIsRefusedEvenWithNothingOutstanding) {
    // A device is bound and no teardown has run: this context is live or
    // half-built, and initializing it again would run the bring-up over
    // resources it still owns. An idle moment is not a close.
    EXPECT_FALSE(may_initialize_context(TeardownProof::NotAttempted, kNoRunsOutstanding, kDeviceBound));
}

TEST(TeardownProofGate, AnOutstandingRunRefusesEveryState) {
    // Ahead of everything else the entry would walk into: a prepared or
    // launched run still holds slots, bindings and blocks. Not even a proof
    // from a previous close admits one.
    EXPECT_FALSE(may_initialize_context(TeardownProof::Proven, kRunsOutstanding, kNoDeviceBound));
    EXPECT_FALSE(may_initialize_context(TeardownProof::NotAttempted, kRunsOutstanding, kNoDeviceBound));
    EXPECT_FALSE(may_initialize_context(TeardownProof::SweptClean, kRunsOutstanding, kNoDeviceBound));
    EXPECT_FALSE(may_initialize_context(TeardownProof::Unresolved, kRunsOutstanding, kNoDeviceBound));
}

TEST(TeardownProofGate, ACleanlyClosedHandleIsAdmittedWhateverTheDeviceIdSays) {
    // The proof is what admits it, so a runner that has not yet cleared its
    // device id does not turn a clean close into a refusal.
    EXPECT_TRUE(may_initialize_context(TeardownProof::Proven, kNoRunsOutstanding, kNoDeviceBound));
    EXPECT_TRUE(may_initialize_context(TeardownProof::Proven, kNoRunsOutstanding, kDeviceBound));
}

TEST(TeardownProofGate, AnUnresolvedOrHalfProvedHandleIsRefused) {
    // `Unresolved` is every teardown that did not prove it completed: a
    // refused close, a throwing one, a failed attach, a failed sweep, a failed
    // reset, the fatal path, and an initialization that failed after the
    // consume. `SweptClean` alone is a cleanup whose outer teardown never
    // reported, which is no more admissible.
    EXPECT_FALSE(may_initialize_context(TeardownProof::Unresolved, kNoRunsOutstanding, kNoDeviceBound));
    EXPECT_FALSE(may_initialize_context(TeardownProof::Unresolved, kNoRunsOutstanding, kDeviceBound));
    EXPECT_FALSE(may_initialize_context(TeardownProof::SweptClean, kNoRunsOutstanding, kNoDeviceBound));
}

TEST(TeardownProofResolution, OnlyASweptCleanupWithAZeroReturnIsProven) {
    EXPECT_EQ(resolve_teardown_proof(TeardownProof::SweptClean, 0), TeardownProof::Proven);
}

TEST(TeardownProofResolution, AFatalTeardownReturningZeroEarnsNothing) {
    // The case the whole shape exists for: the fatal path abandons resources
    // instead of sweeping, so it never writes `SweptClean`, and on a5 it can
    // still return zero after a probe-confirmed reset. Promotion is only ever
    // from the cleanup's own fact, so that zero proves nothing.
    EXPECT_EQ(resolve_teardown_proof(TeardownProof::Unresolved, 0), TeardownProof::Unresolved);
    EXPECT_EQ(resolve_teardown_proof(TeardownProof::NotAttempted, 0), TeardownProof::Unresolved);
}

TEST(TeardownProofResolution, ACleanSweepWhoseTeardownFailedIsUnresolved) {
    // A clean sweep followed by a failed device reset is not a clean close:
    // the reset code reaches the outer return value and nothing else records
    // it, so the proof has to be withdrawn here.
    EXPECT_EQ(resolve_teardown_proof(TeardownProof::SweptClean, -1), TeardownProof::Unresolved);
    EXPECT_EQ(resolve_teardown_proof(TeardownProof::SweptClean, 507018), TeardownProof::Unresolved);
}

TEST(TeardownProofConsumption, AProofAuthorizesExactlyOneInitialization) {
    // Spent before anything in that initialization can change a resource, so a
    // failure later in it cannot fall back on the previous close.
    const TeardownProof after_first = consume_teardown_proof(TeardownProof::Proven);
    EXPECT_EQ(after_first, TeardownProof::NotAttempted);
    // And the second entry has to answer on its own: with a device still
    // bound, the spent proof admits nothing.
    EXPECT_FALSE(may_initialize_context(after_first, kNoRunsOutstanding, kDeviceBound));
}

TEST(TeardownProofConsumption, ARefusedEntryLeavesTheRecordAlone) {
    // A refusal touches no resource, so it must not spend anything either: a
    // caller that then closes properly still earns the proof that close makes.
    EXPECT_EQ(consume_teardown_proof(TeardownProof::Unresolved), TeardownProof::Unresolved);
    EXPECT_EQ(consume_teardown_proof(TeardownProof::NotAttempted), TeardownProof::NotAttempted);
    EXPECT_EQ(consume_teardown_proof(TeardownProof::SweptClean), TeardownProof::SweptClean);
}

TEST(TeardownProofConsumption, AFailedInitializationAfterTheConsumeRefusesTheRetry) {
    // The sequence a direct C caller can drive: close cleanly, initialize
    // again, have that initialization fail past the consume. Every such exit
    // records `Unresolved`, and the next attempt is refused until a close runs.
    TeardownProof proof = TeardownProof::Proven;
    proof = consume_teardown_proof(proof);
    proof = TeardownProof::Unresolved;  // what every post-consume failure exit writes
    EXPECT_FALSE(may_initialize_context(proof, kNoRunsOutstanding, kNoDeviceBound));
    EXPECT_FALSE(may_initialize_context(proof, kNoRunsOutstanding, kDeviceBound));
}

}  // namespace
