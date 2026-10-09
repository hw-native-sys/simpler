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
 * The two decisions the backend makes about a diagnostic pair, driven directly.
 *
 * `DeviceRunnerBase::can_join_diagnostic_run` and the C entry's
 * `owned_prepared_execution` both call the functions under test here; neither
 * restates the rule, so a case that passes here describes the production
 * answer. What stays with the runner is the pair's two resource terms -- the
 * latched collector shape and retained capacity -- which need a live collector
 * and are covered in `test_chip_swimlane_joined_diagnostics.cpp`.
 *
 * The lane cases in `test_chip_run_lane_joined_launch.cpp` are a different
 * layer: they prove the lane refuses before the backend is asked. These prove
 * what the backend answers once it is.
 */

#include <gtest/gtest.h>

#include "host/joined_diagnostic_eligibility.h"

namespace {

using simpler::dfx::classify_joined_diagnostic_pair;
using simpler::dfx::JoinedDiagnosticRun;
using simpler::dfx::JoinedDiagnosticVerdict;
using simpler::dfx::NativeRunHandleFacts;
using simpler::dfx::prepared_handle_is_owned;

/** A run that is eligible on every term, so each case changes exactly one. */
JoinedDiagnosticRun eligible_run(uint32_t slot, ChipSwimlaneLevel level = ChipSwimlaneLevel::TASK_TIMING) {
    JoinedDiagnosticRun run;
    run.chip_swimlane_level = level;
    run.other_channel_enabled = false;
    run.pipeline_slot = slot;
    return run;
}

TEST(JoinedDiagnosticPairTest, TwoRunsAtOneAdmittedLevelAgree) {
    for (const ChipSwimlaneLevel level : {ChipSwimlaneLevel::TASK_TIMING, ChipSwimlaneLevel::SCHEDULE_TIMING}) {
        EXPECT_EQ(
            classify_joined_diagnostic_pair(
                /*kernel_execution=*/false, eligible_run(0, level), eligible_run(1, level)
            ),
            JoinedDiagnosticVerdict::kConfigurationAgrees
        ) << "level "
          << static_cast<uint32_t>(level);
    }
}

TEST(JoinedDiagnosticPairTest, KernelExecutionDeclinesBeforeAnythingElseIsRead) {
    // Every other term agrees, so the kernel latch is the only reason left --
    // which is what makes this the answer rather than a coincidence.
    EXPECT_EQ(
        classify_joined_diagnostic_pair(/*kernel_execution=*/true, eligible_run(0), eligible_run(1)),
        JoinedDiagnosticVerdict::kKernelExecution
    );
}

TEST(JoinedDiagnosticPairTest, LevelsThatDisagreeAreRefusedEvenThoughBothAreAdmitted) {
    // 1 and 2 are each admitted on their own. The pair is still refused: the
    // device latches one level word, and a queued predecessor may not have
    // reached that latch when its successor arms.
    EXPECT_EQ(
        classify_joined_diagnostic_pair(
            false, eligible_run(0, ChipSwimlaneLevel::TASK_TIMING), eligible_run(1, ChipSwimlaneLevel::SCHEDULE_TIMING)
        ),
        JoinedDiagnosticVerdict::kLevelDisagrees
    );
    EXPECT_EQ(
        classify_joined_diagnostic_pair(
            false, eligible_run(0, ChipSwimlaneLevel::SCHEDULE_TIMING), eligible_run(1, ChipSwimlaneLevel::TASK_TIMING)
        ),
        JoinedDiagnosticVerdict::kLevelDisagrees
    );
}

TEST(JoinedDiagnosticPairTest, AgreeingLevelsOutsideOneAndTwoAreNotAdmitted) {
    // Agreement is not enough. Levels 3 and 4 arm phase pools this has not
    // established an ordering for, and 0 is not a diagnostic run at all.
    for (const ChipSwimlaneLevel level :
         {ChipSwimlaneLevel::DISABLED, ChipSwimlaneLevel::SCHED_PHASES, ChipSwimlaneLevel::ORCH_PHASES}) {
        EXPECT_EQ(
            classify_joined_diagnostic_pair(false, eligible_run(0, level), eligible_run(1, level)),
            JoinedDiagnosticVerdict::kLevelNotAdmitted
        ) << "level "
          << static_cast<uint32_t>(level);
    }
}

TEST(JoinedDiagnosticPairTest, AnotherChannelOnEitherSideRefusesThePair) {
    JoinedDiagnosticRun noisy = eligible_run(0);
    noisy.other_channel_enabled = true;
    EXPECT_EQ(
        classify_joined_diagnostic_pair(false, noisy, eligible_run(1)), JoinedDiagnosticVerdict::kOtherChannelEnabled
    ) << "the successor's own channel";
    EXPECT_EQ(
        classify_joined_diagnostic_pair(false, eligible_run(1), noisy), JoinedDiagnosticVerdict::kOtherChannelEnabled
    ) << "and the predecessor's, which the successor cannot see from its own config";
}

TEST(JoinedDiagnosticPairTest, OnePipelineSlotIsOneRunsStateNotAPair) {
    EXPECT_EQ(
        classify_joined_diagnostic_pair(false, eligible_run(2), eligible_run(2)),
        JoinedDiagnosticVerdict::kSamePipelineSlot
    );
}

// ---------------------------------------------------------------------------
// Which execution a handle owns, at the phase the caller asked for
// ---------------------------------------------------------------------------

/** A handle whose every ownership term holds at `phase`. */
NativeRunHandleFacts owned_facts(NativeRunPhase phase) {
    NativeRunHandleFacts facts;
    facts.phase = phase;
    facts.runner_claimed = phase == NativeRunPhase::Running;
    facts.runner_reserved = phase != NativeRunPhase::Running;
    facts.has_execution = true;
    facts.runtime_matches = true;
    facts.identity_matches = true;
    facts.slot_matches = true;
    return facts;
}

TEST(JoinedDiagnosticOwnershipTest, EachPhaseAcceptsItsOwnOwner) {
    EXPECT_TRUE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Prepared), NativeRunPhase::Prepared));
    EXPECT_TRUE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Running), NativeRunPhase::Running));
}

TEST(JoinedDiagnosticOwnershipTest, AHandleInTheWrongPhaseOwnsNothingTheGateMayRead) {
    // The pair gate asks for a prepared successor and a running predecessor by
    // name. A completed run, or the two swapped, is a different run's state.
    EXPECT_FALSE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Running), NativeRunPhase::Prepared));
    EXPECT_FALSE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Prepared), NativeRunPhase::Running));
    EXPECT_FALSE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Complete), NativeRunPhase::Running));
    EXPECT_FALSE(prepared_handle_is_owned(owned_facts(NativeRunPhase::Complete), NativeRunPhase::Prepared));
}

TEST(JoinedDiagnosticOwnershipTest, ThePhasesClaimIsTheOneThatCounts) {
    // Running reads the runner's claim and Prepared reads its reservation, so
    // a context carrying the other one is not an owner. The two are separate
    // words, and a run between them holds neither.
    NativeRunHandleFacts running = owned_facts(NativeRunPhase::Running);
    running.runner_claimed = false;
    running.runner_reserved = true;
    EXPECT_FALSE(prepared_handle_is_owned(running, NativeRunPhase::Running));

    NativeRunHandleFacts prepared = owned_facts(NativeRunPhase::Prepared);
    prepared.runner_reserved = false;
    prepared.runner_claimed = true;
    EXPECT_FALSE(prepared_handle_is_owned(prepared, NativeRunPhase::Prepared));
}

TEST(JoinedDiagnosticOwnershipTest, APhaseWithNoExecutionCarriesNoConfiguration) {
    // A running context whose active execution is gone, and a prepared one
    // whose execution was already moved out by a launch. Both are the hollow
    // object the gate must not read through.
    for (const NativeRunPhase phase : {NativeRunPhase::Prepared, NativeRunPhase::Running}) {
        NativeRunHandleFacts facts = owned_facts(phase);
        facts.has_execution = false;
        EXPECT_FALSE(prepared_handle_is_owned(facts, phase)) << "phase " << static_cast<int>(phase);
    }
}

TEST(JoinedDiagnosticOwnershipTest, AnExecutionFromAnotherRunOnTheSameContextIsRefused) {
    // Each of the three cross-checks on its own. A handle resolves to a
    // context, and a context can still be holding what an earlier run on the
    // same slot left, so any one of these disagreeing means this execution is
    // not the one the caller named.
    for (const NativeRunPhase phase : {NativeRunPhase::Prepared, NativeRunPhase::Running}) {
        NativeRunHandleFacts foreign_runtime = owned_facts(phase);
        foreign_runtime.runtime_matches = false;
        EXPECT_FALSE(prepared_handle_is_owned(foreign_runtime, phase))
            << "foreign runtime at " << static_cast<int>(phase);

        NativeRunHandleFacts stale_identity = owned_facts(phase);
        stale_identity.identity_matches = false;
        EXPECT_FALSE(prepared_handle_is_owned(stale_identity, phase))
            << "stale identity at " << static_cast<int>(phase);

        NativeRunHandleFacts wrong_slot = owned_facts(phase);
        wrong_slot.slot_matches = false;
        EXPECT_FALSE(prepared_handle_is_owned(wrong_slot, phase)) << "wrong slot at " << static_cast<int>(phase);
    }
}

}  // namespace
