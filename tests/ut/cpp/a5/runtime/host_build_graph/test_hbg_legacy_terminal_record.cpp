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
// What a5 host_build_graph's legacy executor may call a success, and what it
// publishes for each shape the run can take.
//
// `aicore_legacy_run_completed_audited_path` is the decision
// `LegacyAicpuExecutor::snapshot_run_terminal` passes to `run_terminal_select`;
// these cases call that function, not a restatement of it, so inverting either
// of its conditions in production fails cases here. The runtime they drive is a
// real `Runtime` carrying the same `scheduler_bootstrap` selection the host
// publishes.
//
// Logic only. Nothing here observes a cache or runs the executor's threads, so
// no case states anything about device visibility or the rendezvous.
#include <gtest/gtest.h>

#include <cstdint>

#include "aicore_scheduler_state.h"
#include "aicpu/device_run_result_aicpu.h"
#include "common/device_run_result.h"
#include "common/run_terminal_accumulator.h"
#include "runtime.h"
#include "scheduler/scheduler_types.h"

namespace {

constexpr uint64_t kEpoch = 0x5100'0000'0000'0001ULL;
constexpr int32_t kHeaderCode = -507;
constexpr uint64_t kContextBase = 0x7f0000000000ULL;

uint64_t region_base(DeviceRunResultRegion &region) { return reinterpret_cast<uint64_t>(&region); }

/** A runtime whose host-published selection says legacy was chosen on purpose. */
class LegacyTerminalRun : public ::testing::Test {
protected:
    void SetUp() override { runtime.set_worker_count(8); }

    void select_explicit_legacy() {
        runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH, kContextBase);
    }

    /** No selection published: the shape `aicpu_execute` rejects. */
    void select_unmarked_fallback() {}

    /** What the executor's finalizer decides, through the production predicate. */
    RunTerminalSelection select(int32_t claims, int32_t participants, int32_t header_status = 0) {
        return run_terminal_select(
            aicore_legacy_run_completed_audited_path(&runtime, claims, participants), header_status, participants_acc
        );
    }

    Runtime runtime;
    RunTerminalAccumulator participants_acc;
};

}  // namespace

TEST_F(LegacyTerminalRun, AnAuthorizedRunWithEveryParticipantClaimingSucceeds) {
    select_explicit_legacy();
    participants_acc.record_participant(0, 0);
    participants_acc.record_participant(0, 0);

    const RunTerminalSelection selection = select(/*claims=*/2, /*participants=*/2);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Ok);
    EXPECT_EQ(selection.code, 0);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::None);
}

// The run the outer wrapper rejects. Every thread returned 0 and the header is
// clean, so only the published selection distinguishes this from the success
// above — and Ok here would contradict the -1 `aicpu_execute` returns for it.
TEST_F(LegacyTerminalRun, AnUnmarkedFallbackPublishesNothingRatherThanOk) {
    select_unmarked_fallback();
    participants_acc.record_participant(0, 0);

    const RunTerminalSelection selection = select(/*claims=*/1, /*participants=*/1);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::None) << "an unauthorized fallback is not an audited path";
}

// A resident selection reaching this executor is the same defect by another
// route: the run did not take the path its selection claims.
TEST_F(LegacyTerminalRun, AResidentSelectionReachingLegacyPublishesNothing) {
    runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_READY, kContextBase);
    participants_acc.record_participant(0, 0);

    EXPECT_EQ(select(/*claims=*/1, /*participants=*/1).verdict, DeviceRunVerdict::None);
}

// The mode gates the success, never the diagnosis.
TEST_F(LegacyTerminalRun, AnUnmarkedFallbackStillReportsItsFailure) {
    select_unmarked_fallback();
    participants_acc.record_participant(-31, 0);

    const RunTerminalSelection selection = select(/*claims=*/1, /*participants=*/1);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, -31);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::ThreadRc);
}

// The legacy boot leaves `rt` null on a failed attach; that thread retires its
// cores but never dispatched, so it does not claim.
TEST_F(LegacyTerminalRun, AParticipantThatNeverDispatchedWithholdsItsClaim) {
    select_explicit_legacy();
    participants_acc.record_participant(0, 0);
    participants_acc.record_participant(0, 0);

    EXPECT_EQ(select(/*claims=*/2, /*participants=*/3).verdict, DeviceRunVerdict::None);
}

// The boot leader that left `rt` null reports the failure itself, so the run is
// decided rather than merely unclaimed.
TEST_F(LegacyTerminalRun, ABootFailureIsReportedNotJustUnclaimed) {
    select_explicit_legacy();
    participants_acc.record_participant(-1, 0);
    participants_acc.record_participant(0, 0);

    const RunTerminalSelection selection = select(/*claims=*/1, /*participants=*/2);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, -1);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::ThreadRc);
}

// A participant's execution and its teardown are separate contributions.
TEST_F(LegacyTerminalRun, AShutdownFailureIsAttributedToTeardownNotExecution) {
    select_explicit_legacy();
    participants_acc.record_participant(0, -44);

    const RunTerminalSelection selection = select(/*claims=*/1, /*participants=*/1);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, -44);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::ShutdownRc);
}

// The supervisor latches the AICore scheduler's error in the shared header, and
// a peer that only waited on the shutdown barrier returns zero for that run.
TEST_F(LegacyTerminalRun, TheSharedHeaderDecidesARunWhoseThreadsAllReturnedZero) {
    select_explicit_legacy();
    participants_acc.record_participant(0, 0);
    participants_acc.record_participant(0, 0);

    const RunTerminalSelection selection = select(/*claims=*/2, /*participants=*/2, kHeaderCode);

    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, kHeaderCode);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::Header);
}

// Carrying the decision to the region: the producer publishes under the epoch
// the platform hands it, and a run that selected nothing leaves the region as
// it found it. Both matter for reuse — a stale epoch reads as undecided, never
// as this run's verdict.
TEST_F(LegacyTerminalRun, APublishedRecordIsReadableOnlyUnderItsOwnEpoch) {
    select_explicit_legacy();
    DeviceRunResultRegion region{};
    RunTerminalPublisher publisher;

    publisher.take(select(/*claims=*/1, /*participants=*/1), nullptr, 0);
    ASSERT_TRUE(publisher.publish(region_base(region), kEpoch));

    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(device_run_result_terminal(region, kEpoch + 1).state, DeviceRunTerminalState::Undecided)
        << "a successor must not read its predecessor's verdict";
}

// A run that selected nothing leaves the region exactly as it found it — with
// its predecessor's record still there. Seeded rather than started empty: an
// empty region reads undecided whether publication was skipped or wrongly
// cleared something, so only a seeded one tells those apart.
TEST_F(LegacyTerminalRun, ARunThatSelectedNothingLeavesThePredecessorsRecord) {
    DeviceRunResultRegion region{};
    const uint64_t predecessor_epoch = kEpoch - 1;
    ASSERT_TRUE(aicpu_publish_run_terminal(
        region_base(region), predecessor_epoch, DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None, nullptr, 0
    ));

    select_unmarked_fallback();
    RunTerminalPublisher publisher;
    publisher.take(select(/*claims=*/1, /*participants=*/1), nullptr, 0);

    EXPECT_FALSE(publisher.publish(region_base(region), kEpoch));
    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Undecided)
        << "this run published nothing, so its own epoch finds no record";
    EXPECT_EQ(device_run_result_terminal(region, predecessor_epoch).state, DeviceRunTerminalState::Succeeded)
        << "a run that publishes nothing must not disturb what the region already held";
}
