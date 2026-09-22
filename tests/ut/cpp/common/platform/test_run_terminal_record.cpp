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

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "aicpu/device_run_result_aicpu.h"
#include "common/device_run_result.h"
#include "common/run_terminal_accumulator.h"

namespace {

constexpr uint64_t kEpoch = 0x1234'5678'9abc'def0ULL;
constexpr int32_t kRuntimeCode = -507;

uint64_t region_base(DeviceRunResultRegion &region) { return reinterpret_cast<uint64_t>(&region); }

}  // namespace

// ===== Wire layout =====

TEST(DeviceRunResultLayout, HeaderWordsPrecedeThePayload) {
    EXPECT_EQ(offsetof(DeviceRunResultRegion, published), 0u);
    EXPECT_EQ(offsetof(DeviceRunResultRegion, payload_bytes), 8u);
    EXPECT_EQ(offsetof(DeviceRunResultRegion, verdict), 12u);
    EXPECT_EQ(offsetof(DeviceRunResultRegion, completion_code), 16u);
    EXPECT_EQ(offsetof(DeviceRunResultRegion, code_source), 20u);
    EXPECT_EQ(offsetof(DeviceRunResultRegion, payload), 24u);
    EXPECT_EQ(sizeof(DeviceRunResultRegion), device_run_result_bytes());
}

// ===== Reading a terminal record =====

TEST(DeviceRunResultTerminal, AZeroEpochNeverMatches) {
    DeviceRunResultRegion region{};
    const DeviceRunTerminal terminal = device_run_result_terminal(region, 0);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Undecided);
    EXPECT_NE(terminal.reason, nullptr);
}

TEST(DeviceRunResultTerminal, APredecessorsRecordIsUndecidedNotItsVerdict) {
    DeviceRunResultRegion region{};
    ASSERT_TRUE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None, nullptr, 0
    ));

    const DeviceRunTerminal terminal = device_run_result_terminal(region, kEpoch + 1);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Undecided);
    EXPECT_NE(terminal.reason, nullptr);
}

TEST(DeviceRunResultTerminal, AnUnpublishedRegionIsUndecidedNotSuccess) {
    DeviceRunResultRegion region{};
    const DeviceRunTerminal terminal = device_run_result_terminal(region, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Undecided);
}

TEST(DeviceRunResultTerminal, SuccessCarriesNoCodeSourceOrPayload) {
    DeviceRunResultRegion region{};
    ASSERT_TRUE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None, nullptr, 0
    ));

    const DeviceRunTerminal terminal = device_run_result_terminal(region, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(terminal.code, 0);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::None);
    EXPECT_EQ(terminal.reason, nullptr);
    // A success is still "no diagnostic payload" to the payload accessor.
    EXPECT_FALSE(device_run_result_published(region, kEpoch));
}

TEST(DeviceRunResultTerminal, FailureKeepsTheRuntimeCodeAndItsSource) {
    DeviceRunResultRegion region{};
    ASSERT_TRUE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Error, kRuntimeCode, DeviceRunCodeSource::Header, nullptr, 0
    ));

    const DeviceRunTerminal terminal = device_run_result_terminal(region, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, kRuntimeCode);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
}

TEST(DeviceRunResultTerminal, ASelfContradictoryRecordIsUndecidedRatherThanPartlyTrusted) {
    // Only a defective producer writes these, so they are built by hand.
    DeviceRunResultRegion success_with_code{};
    success_with_code.published = kEpoch;
    success_with_code.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    success_with_code.completion_code = kRuntimeCode;
    EXPECT_EQ(device_run_result_terminal(success_with_code, kEpoch).state, DeviceRunTerminalState::Undecided);

    DeviceRunResultRegion failure_without_code{};
    failure_without_code.published = kEpoch;
    failure_without_code.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    failure_without_code.code_source = static_cast<uint32_t>(DeviceRunCodeSource::ThreadRc);
    EXPECT_EQ(device_run_result_terminal(failure_without_code, kEpoch).state, DeviceRunTerminalState::Undecided);

    DeviceRunResultRegion failure_without_source{};
    failure_without_source.published = kEpoch;
    failure_without_source.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    failure_without_source.completion_code = kRuntimeCode;
    EXPECT_EQ(device_run_result_terminal(failure_without_source, kEpoch).state, DeviceRunTerminalState::Undecided);

    DeviceRunResultRegion unknown_verdict{};
    unknown_verdict.published = kEpoch;
    unknown_verdict.verdict = 99;
    EXPECT_EQ(device_run_result_terminal(unknown_verdict, kEpoch).state, DeviceRunTerminalState::Undecided);
}

TEST(DeviceRunResultTerminal, AnOversizedPayloadRefusesTheDiagnosticNotTheVerdict) {
    DeviceRunResultRegion region{};
    region.published = kEpoch;
    region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    region.completion_code = kRuntimeCode;
    region.code_source = static_cast<uint32_t>(DeviceRunCodeSource::ThreadRc);
    region.payload_bytes = DEVICE_RUN_RESULT_PAYLOAD_BYTES + 1;

    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Failed);
    EXPECT_FALSE(device_run_result_published(region, kEpoch));
}

// ===== Publishing =====

TEST(AicpuPublishRunTerminal, EpochIsCommittedAfterTheRecordItVouchesFor) {
    DeviceRunResultRegion region{};
    const uint8_t scene[] = {1, 2, 3, 4};
    ASSERT_TRUE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Error, kRuntimeCode, DeviceRunCodeSource::ShutdownRc, scene,
        sizeof(scene)
    ));

    EXPECT_EQ(region.published, kEpoch);
    EXPECT_EQ(region.payload_bytes, sizeof(scene));
    ASSERT_TRUE(device_run_result_published(region, kEpoch));
    EXPECT_EQ(std::memcmp(region.payload, scene, sizeof(scene)), 0);

    const DeviceRunTerminal terminal = device_run_result_terminal(region, kEpoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ShutdownRc);
}

TEST(AicpuPublishRunTerminal, ARefusedRecordLeavesThePredecessorsEpochInPlace) {
    DeviceRunResultRegion region{};
    const uint64_t predecessor = kEpoch - 1;
    region.published = predecessor;

    const uint8_t scene[DEVICE_RUN_RESULT_PAYLOAD_BYTES + 1]{};
    // Oversized payload, success carrying a code, failure carrying none, an
    // unknown verdict, no region and no epoch: each refuses rather than
    // publishing a repaired record.
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Error, kRuntimeCode, DeviceRunCodeSource::ThreadRc, scene,
        sizeof(scene)
    ));
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Ok, kRuntimeCode, DeviceRunCodeSource::None, nullptr, 0
    ));
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::Error, 0, DeviceRunCodeSource::ThreadRc, nullptr, 0
    ));
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(region), kEpoch, DeviceRunVerdict::None, 0, DeviceRunCodeSource::None, nullptr, 0
    ));
    EXPECT_FALSE(aicpu_publish_run_terminal(0, kEpoch, DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None, nullptr, 0));
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(region), 0, DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None, nullptr, 0
    ));

    EXPECT_EQ(region.published, predecessor);
    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Undecided);
}

TEST(AicpuPublishRunTerminal, WhatItCommitsIsAlwaysSomethingTheHostCanDecide) {
    // Publisher and reader ask `device_run_code_source_is_failure` the same
    // question, so no record that commits reads back as undecided. A source
    // outside the enum is refused rather than committed.
    DeviceRunResultRegion refused{};
    const auto unknown = static_cast<DeviceRunCodeSource>(9);
    EXPECT_FALSE(device_run_code_source_is_failure(unknown));
    EXPECT_FALSE(aicpu_publish_run_terminal(
        region_base(refused), kEpoch, DeviceRunVerdict::Error, kRuntimeCode, unknown, nullptr, 0
    ));
    EXPECT_EQ(refused.published, 0u);

    for (const DeviceRunCodeSource source :
         {DeviceRunCodeSource::Header, DeviceRunCodeSource::ThreadRc, DeviceRunCodeSource::ShutdownRc}) {
        DeviceRunResultRegion accepted{};
        ASSERT_TRUE(aicpu_publish_run_terminal(
            region_base(accepted), kEpoch, DeviceRunVerdict::Error, kRuntimeCode, source, nullptr, 0
        ));
        const DeviceRunTerminal terminal = device_run_result_terminal(accepted, kEpoch);
        EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
        EXPECT_EQ(terminal.source, source);
    }
}

TEST(RunTerminalPublisherTest, PublishingClearsTheSnapshotSoItCannotBeReused) {
    RunTerminalPublisher publisher;
    publisher.take({DeviceRunVerdict::Error, kRuntimeCode, DeviceRunCodeSource::ThreadRc}, nullptr, 0);

    DeviceRunResultRegion first{};
    ASSERT_TRUE(publisher.publish(region_base(first), kEpoch));
    EXPECT_EQ(device_run_result_terminal(first, kEpoch).code, kRuntimeCode);

    DeviceRunResultRegion second{};
    EXPECT_FALSE(publisher.publish(region_base(second), kEpoch + 1));
    EXPECT_EQ(device_run_result_terminal(second, kEpoch + 1).state, DeviceRunTerminalState::Undecided);
}

TEST(RunTerminalPublisherTest, ASuccessSnapshotDropsAnyOfferedPayload) {
    RunTerminalPublisher publisher;
    const uint8_t scene[] = {7, 7, 7};
    publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, scene, sizeof(scene));

    DeviceRunResultRegion region{};
    ASSERT_TRUE(publisher.publish(region_base(region), kEpoch));
    EXPECT_EQ(region.payload_bytes, 0u);
    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Succeeded);
}

// ===== Folding participants into one code =====

TEST(RunTerminalAccumulatorTest, AZeroReturnRecordsNothing) {
    RunTerminalAccumulator acc;
    acc.record_participant(0, 0);
    EXPECT_FALSE(acc.has_failure());
}

TEST(RunTerminalAccumulatorTest, ExecutionOutranksTeardownForTheSameParticipant) {
    RunTerminalAccumulator acc;
    acc.record_participant(-11, -22);
    EXPECT_EQ(acc.code(), -11);
    EXPECT_EQ(acc.source(), DeviceRunCodeSource::ThreadRc);

    RunTerminalAccumulator teardown_only;
    teardown_only.record_participant(0, -22);
    EXPECT_EQ(teardown_only.code(), -22);
    EXPECT_EQ(teardown_only.source(), DeviceRunCodeSource::ShutdownRc);
}

TEST(RunTerminalAccumulatorTest, ASucceedingParticipantCannotClearAFailure) {
    RunTerminalAccumulator acc;
    acc.record_participant(-11, 0);
    acc.record_participant(0, 0);
    acc.record_participant(-33, 0);
    EXPECT_TRUE(acc.has_failure());
    EXPECT_EQ(acc.code(), -11);
}

TEST(RunTerminalAccumulatorTest, ResetClearsTheRunsFailure) {
    RunTerminalAccumulator acc;
    acc.record_participant(-11, 0);
    acc.reset();
    EXPECT_FALSE(acc.has_failure());
    acc.record_participant(0, -22);
    EXPECT_EQ(acc.source(), DeviceRunCodeSource::ShutdownRc);
}

TEST(RunTerminalAccumulatorTest, ConcurrentParticipantsNeverPairACodeWithAnothersSource) {
    // Each participant contributes a code whose source is decided by its own
    // parity, so any (code, source) pair that disagrees is a torn selection.
    constexpr int kThreads = 8;
    for (int attempt = 0; attempt < 200; ++attempt) {
        RunTerminalAccumulator acc;
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&acc, &go, i] {
                while (!go.load(std::memory_order_acquire)) {}
                if (i % 2 == 0) {
                    acc.record_participant(-(i + 1), 0);
                } else {
                    acc.record_participant(0, -(i + 1));
                }
            });
        }
        go.store(true, std::memory_order_release);
        for (std::thread &t : threads)
            t.join();

        ASSERT_TRUE(acc.has_failure());
        const int index = -acc.code() - 1;
        ASSERT_GE(index, 0);
        ASSERT_LT(index, kThreads);
        EXPECT_EQ(acc.source(), (index % 2 == 0) ? DeviceRunCodeSource::ThreadRc : DeviceRunCodeSource::ShutdownRc);
    }
}

TEST(RunTerminalAccumulatorTest, AReaderRacingTheFirstRecordSeesBothHalvesOrNeither) {
    // The property the packed selection buys, which joining first cannot see:
    // a reader that catches the accumulator mid-record must never observe the
    // code without the source that chose it.
    //
    // What this does and does not catch: a two-store implementation's window is
    // a few instructions wide, and this loop does not reliably land inside it —
    // measured passing against a split-store accumulator. Widening that gap by
    // anything at all (a log, a branch, a second derived field) makes the tear
    // land on the first attempt. So read this as a barrier against the gap
    // growing, not as the proof that one store is required; that proof is the
    // type, which has no gap to grow.
    constexpr int kAttempts = 4000;
    int observations = 0;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        RunTerminalAccumulator acc;
        std::atomic<bool> go{false};
        std::atomic<bool> torn{false};
        std::atomic<bool> seen{false};

        std::thread writer([&acc, &go] {
            while (!go.load(std::memory_order_acquire)) {}
            acc.record_participant(0, -22);
        });
        std::thread reader([&acc, &go, &torn, &seen] {
            while (!go.load(std::memory_order_acquire)) {}
            for (int spin = 0; spin < 100000; ++spin) {
                if (!acc.has_failure()) continue;
                seen.store(true, std::memory_order_relaxed);
                if (acc.source() != DeviceRunCodeSource::ShutdownRc || acc.code() != -22) {
                    torn.store(true, std::memory_order_relaxed);
                }
                return;
            }
        });
        go.store(true, std::memory_order_release);
        writer.join();
        reader.join();

        if (seen.load(std::memory_order_relaxed)) ++observations;
        ASSERT_FALSE(torn.load(std::memory_order_relaxed)) << "attempt " << attempt;
    }
    // Without at least one mid-record observation the run proves nothing, so
    // fail rather than pass vacuously.
    EXPECT_GT(observations, 0);
}

// ===== Selecting the run's verdict =====

TEST(RunTerminalSelect, SharedErrorStateOutranksAParticipantsReturn) {
    RunTerminalAccumulator acc;
    acc.record_participant(-11, 0);
    const RunTerminalSelection selection = run_terminal_select(true, kRuntimeCode, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, kRuntimeCode);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::Header);
}

TEST(RunTerminalSelect, SharedErrorStateAloneFailsARunWhoseThreadsAllReturnedZero) {
    // The a2a3 host_build_graph shape: the resolution thread latches a
    // scheduler timeout in the header and still returns a non-negative
    // completed count, so every participant's return is zero.
    RunTerminalAccumulator acc;
    acc.record_participant(0, 0);
    const RunTerminalSelection selection = run_terminal_select(true, kRuntimeCode, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::Header);
}

TEST(RunTerminalSelect, AParticipantsFailureIsUsedWhenSharedStateIsClean) {
    RunTerminalAccumulator acc;
    acc.record_participant(0, -22);
    const RunTerminalSelection selection = run_terminal_select(true, 0, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, -22);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::ShutdownRc);
}

TEST(RunTerminalSelect, AnUncompletedNormalPathPublishesNothingRatherThanSuccess) {
    RunTerminalAccumulator acc;
    const RunTerminalSelection selection = run_terminal_select(false, 0, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::None);
}

TEST(RunTerminalSelect, AnUncompletedNormalPathStillReportsAKnownFailure) {
    RunTerminalAccumulator acc;
    acc.record_participant(-11, 0);
    const RunTerminalSelection selection = run_terminal_select(false, 0, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(selection.code, -11);
}

TEST(RunTerminalSelect, ACompletedNormalPathWithNothingToReportSucceeds) {
    RunTerminalAccumulator acc;
    const RunTerminalSelection selection = run_terminal_select(true, 0, acc);
    EXPECT_EQ(selection.verdict, DeviceRunVerdict::Ok);
    EXPECT_EQ(selection.code, 0);
    EXPECT_EQ(selection.source, DeviceRunCodeSource::None);
}
