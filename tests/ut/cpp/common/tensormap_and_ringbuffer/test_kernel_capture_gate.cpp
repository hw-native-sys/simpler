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
#include <limits>
#include <thread>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_capture_gate.h"

namespace {
using namespace simpler::tmr;

KernelCaptureControl prepared_control(uint32_t threads = 3) {
    KernelCaptureControl control{};
    control.context_generation = 5;
    control.capture_generation = 9;
    control.runtime_address = 0x1000;
    control.binding_epoch = 31;
    control.launched_threads = threads;
    return control;
}

TEST(KernelCaptureGate, NextRoundWaitsForEveryCpuReturn) {
    auto control = prepared_control();
    ASSERT_TRUE(capture_admit(control, true, -7));
    EXPECT_EQ(control.replay_epoch, 1U);
    EXPECT_TRUE(capture_finish(control, 1, 0));
    EXPECT_TRUE(capture_finish(control, 1, 0));
    const auto before = control;
    EXPECT_FALSE(capture_admit(control, true, -7));
    EXPECT_EQ(std::memcmp(&control, &before, sizeof(control)), 0);
    EXPECT_TRUE(capture_finish(control, 1, 0));
    EXPECT_EQ(control.completed_epoch, 1U);
    EXPECT_TRUE(capture_admit(control, true, -7));
    EXPECT_EQ(control.replay_epoch, 2U);
}

TEST(KernelCaptureGate, FailureKeepsItsOwnLateParticipantsAndRejectsQueuedRounds) {
    auto control = prepared_control();
    ASSERT_TRUE(capture_admit(control, true, -7));
    ASSERT_TRUE(capture_finish(control, 1, -19));
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Admitted));
    ASSERT_TRUE(capture_finish(control, 1, 0));
    ASSERT_TRUE(capture_finish(control, 1, -23));
    EXPECT_EQ(control.first_error, -19);
    for (uint64_t epoch : {2U, 3U}) {
        ASSERT_TRUE(capture_admit(control, true, -7));
        EXPECT_EQ(control.replay_epoch, epoch);
        EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
        for (int i = 0; i < 3; ++i)
            ASSERT_TRUE(capture_finish(control, epoch, 0));
        EXPECT_EQ(control.first_error, -19);
        EXPECT_EQ(control.completed_epoch, epoch);
    }
}

TEST(KernelCaptureGate, InvalidSnapshotPoisonsOnlyTheCaptureControl) {
    auto control = prepared_control();
    ASSERT_TRUE(capture_admit(control, false, -41));
    EXPECT_EQ(control.first_error, -41);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    EXPECT_EQ(control.completed_epoch, 0U);
}

TEST(KernelCaptureGate, StaleOrDuplicateReturnCannotCompleteAnotherGroup) {
    auto control = prepared_control(1);
    ASSERT_TRUE(capture_admit(control, true, -7));
    auto before = control;
    EXPECT_FALSE(capture_finish(control, 2, -99));
    EXPECT_EQ(std::memcmp(&control, &before, sizeof(control)), 0);
    ASSERT_TRUE(capture_finish(control, 1, 0));
    before = control;
    EXPECT_FALSE(capture_finish(control, 1, -99));
    EXPECT_EQ(std::memcmp(&control, &before, sizeof(control)), 0);
}

TEST(KernelCaptureGate, ConcurrentErrorsArePublishedBeforeCompletedEpoch) {
    auto control = prepared_control(16);
    ASSERT_TRUE(capture_admit(control, true, -7));
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i)
        threads.emplace_back([&control, i] {
            EXPECT_TRUE(capture_finish(control, 1, i == 7 ? -51 : 0));
        });
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(control.finished_threads, 16U);
    EXPECT_EQ(control.completed_epoch, 1U);
    EXPECT_EQ(control.first_error, -51);
}

TEST(KernelCaptureGate, MissingIdentityAndEpochExhaustionRefuseBeforeMutation) {
    auto control = prepared_control();
    control.capture_generation = 0;
    auto before = control;
    EXPECT_FALSE(capture_admit(control, true, -7));
    EXPECT_EQ(std::memcmp(&control, &before, sizeof(control)), 0);
    control = prepared_control();
    control.replay_epoch = std::numeric_limits<uint64_t>::max();
    control.completed_epoch = control.replay_epoch;
    before = control;
    EXPECT_FALSE(capture_admit(control, true, -7));
    EXPECT_EQ(std::memcmp(&control, &before, sizeof(control)), 0);
}
}  // namespace
