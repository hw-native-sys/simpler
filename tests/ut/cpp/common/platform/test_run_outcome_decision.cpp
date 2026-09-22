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

#include <cstring>
#include <string>

#include "host/run_outcome_decision.h"

namespace {

using Completion = RunCompletionFence::Completion;

constexpr int32_t kRuntimeCode = -9;

RunOutcomeEvidence succeeded_record(Completion boundaries) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = boundaries;
    evidence.record_read = RunRecordRead::Ok;
    evidence.terminal.state = DeviceRunTerminalState::Succeeded;
    evidence.terminal.reason = nullptr;
    return evidence;
}

RunOutcomeEvidence failed_record(Completion boundaries, DeviceRunCodeSource source = DeviceRunCodeSource::ThreadRc) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = boundaries;
    evidence.record_read = RunRecordRead::Ok;
    evidence.terminal.state = DeviceRunTerminalState::Failed;
    evidence.terminal.code = kRuntimeCode;
    evidence.terminal.source = source;
    evidence.terminal.reason = nullptr;
    return evidence;
}

RunOutcomeEvidence undecided_record(Completion boundaries, const char *reason) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = boundaries;
    evidence.record_read = RunRecordRead::Ok;
    evidence.terminal.state = DeviceRunTerminalState::Undecided;
    evidence.terminal.reason = reason;
    return evidence;
}

}  // namespace

// ===== The one shape that is a success =====

TEST(RunOutcomeDecision, SuccessNeedsBothAValidOkRecordAndBothBoundaries) {
    const RunExecutionOutcome outcome = decide_run_execution(succeeded_record(Completion::Complete));
    EXPECT_EQ(outcome.state, RunExecutionState::Succeeded);
    EXPECT_EQ(outcome.code, 0);
    EXPECT_EQ(outcome.source, DeviceRunCodeSource::None);
    EXPECT_EQ(outcome.reason, nullptr);
}

// The record is written before the run's kernel returns, so a valid Ok read
// proves the orchestration finished — not that the submitted work ended.
// Retiring on it would free resources the device still holds.
TEST(RunOutcomeDecision, AnOkRecordBeforeTheBoundariesCompleteIsNotYetSuccess) {
    const RunExecutionOutcome outcome = decide_run_execution(succeeded_record(Completion::Pending));
    EXPECT_EQ(outcome.state, RunExecutionState::Pending);
    EXPECT_NE(outcome.state, RunExecutionState::Succeeded);
}

// ===== A published failure decides the run on its own =====

// A failing run is precisely the one whose boundaries may never complete, and
// its code is attributable because its device side wrote it before the kernel
// returned. So the failure is not held hostage to the boundary channel.
TEST(RunOutcomeDecision, APublishedFailureDecidesTheRunWhateverTheBoundariesSay) {
    for (const Completion boundaries :
         {Completion::Pending, Completion::Complete, Completion::Error, Completion::Unfenced}) {
        const RunExecutionOutcome outcome = decide_run_execution(failed_record(boundaries));
        EXPECT_EQ(outcome.state, RunExecutionState::Failed) << "boundaries=" << static_cast<int>(boundaries);
        EXPECT_EQ(outcome.code, kRuntimeCode);
        EXPECT_EQ(outcome.source, DeviceRunCodeSource::ThreadRc);
        EXPECT_EQ(outcome.reason, nullptr);
    }
}

TEST(RunOutcomeDecision, TheFailureCodeAndItsSourceAreCarriedThroughUnchanged) {
    for (const DeviceRunCodeSource source :
         {DeviceRunCodeSource::Header, DeviceRunCodeSource::ThreadRc, DeviceRunCodeSource::ShutdownRc}) {
        const RunExecutionOutcome outcome = decide_run_execution(failed_record(Completion::Complete, source));
        EXPECT_EQ(outcome.state, RunExecutionState::Failed);
        EXPECT_EQ(outcome.source, source);
    }
}

// ===== Uncovered work is undecided, never success =====

// An Unfenced completion means the run's own events cannot speak for work it
// submitted. A success record alongside it is not the missing proof: the record
// describes orchestration, and the uncovered kernel is what has no evidence.
TEST(RunOutcomeDecision, SubmittedWorkNoBoundaryCoversIsUndecidedEvenWithAnOkRecord) {
    const RunExecutionOutcome outcome = decide_run_execution(succeeded_record(Completion::Unfenced));
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    ASSERT_NE(outcome.reason, nullptr);
    EXPECT_NE(std::string(outcome.reason).find("no recorded boundary"), std::string::npos);
}

TEST(RunOutcomeDecision, AFailedBoundaryQueryIsUndecidedEvenWithAnOkRecord) {
    const RunExecutionOutcome outcome = decide_run_execution(succeeded_record(Completion::Error));
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    ASSERT_NE(outcome.reason, nullptr);
    EXPECT_NE(std::string(outcome.reason).find("boundary"), std::string::npos);
}

// ===== The record channel's three silences stay distinguishable =====

TEST(RunOutcomeDecision, CompleteBoundariesWithNoRecordReadAreUndecidedNotSuccess) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = Completion::Complete;
    evidence.record_read = RunRecordRead::NotAttempted;

    const RunExecutionOutcome outcome = decide_run_execution(evidence);
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    ASSERT_NE(outcome.reason, nullptr);
    EXPECT_NE(std::string(outcome.reason).find("no result record"), std::string::npos);
}

TEST(RunOutcomeDecision, AFailedReadIsUndecidedAndSaysSoRatherThanReadingAsAbsent) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = Completion::Complete;
    evidence.record_read = RunRecordRead::Failed;

    const RunExecutionOutcome outcome = decide_run_execution(evidence);
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    ASSERT_NE(outcome.reason, nullptr);
    EXPECT_NE(std::string(outcome.reason).find("read failed"), std::string::npos);
}

// The reason a record could not decide the run is the producer-defect report
// node B's reader produces, so it reaches the caller rather than being replaced
// by a generic one.
TEST(RunOutcomeDecision, AnUndecidedRecordKeepsTheReasonTheReaderGave) {
    const RunExecutionOutcome outcome =
        decide_run_execution(undecided_record(Completion::Complete, "no record published under this run's epoch"));
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    ASSERT_NE(outcome.reason, nullptr);
    EXPECT_STREQ(outcome.reason, "no record published under this run's epoch");
}

TEST(RunOutcomeDecision, AnUndecidedRecordWithNoReasonStillReportsOne) {
    const RunExecutionOutcome outcome = decide_run_execution(undecided_record(Completion::Complete, nullptr));
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
    EXPECT_NE(outcome.reason, nullptr);
}

// ===== Pending is in progress, and is not an undecided terminal observation =====

TEST(RunOutcomeDecision, ARunStillExecutingWithNothingReadYetIsPending) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = Completion::Pending;
    evidence.record_read = RunRecordRead::NotAttempted;

    const RunExecutionOutcome outcome = decide_run_execution(evidence);
    EXPECT_EQ(outcome.state, RunExecutionState::Pending);
    EXPECT_EQ(outcome.reason, nullptr);
}

// A read that was attempted and failed is a terminal observation failure, so it
// is Undecided even while the boundaries are still pending — unlike a read that
// was never attempted, which leaves the run merely in progress.
TEST(RunOutcomeDecision, AFailedReadIsUndecidedRatherThanPendingWhileBoundariesRun) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = Completion::Pending;
    evidence.record_read = RunRecordRead::Failed;

    const RunExecutionOutcome outcome = decide_run_execution(evidence);
    EXPECT_EQ(outcome.state, RunExecutionState::Undecided);
}

// ===== Defaults =====

// Default-constructed evidence is "nothing observed", which must read as an
// in-progress run rather than as a decided one.
TEST(RunOutcomeDecision, DefaultEvidenceIsPending) {
    const RunExecutionOutcome outcome = decide_run_execution(RunOutcomeEvidence{});
    EXPECT_EQ(outcome.state, RunExecutionState::Pending);
}

TEST(RunOutcomeDecision, EveryStateHasAName) {
    EXPECT_STREQ(run_execution_state_name(RunExecutionState::Pending), "pending");
    EXPECT_STREQ(run_execution_state_name(RunExecutionState::Succeeded), "succeeded");
    EXPECT_STREQ(run_execution_state_name(RunExecutionState::Failed), "failed");
    EXPECT_STREQ(run_execution_state_name(RunExecutionState::Undecided), "undecided");
}
