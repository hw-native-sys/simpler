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

#include "host/ccu_request_state.h"

namespace {
void submit(CcuRequestStateV1 &s, uint64_t seq) {
    ASSERT_EQ(s.Prepare(seq), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginArm(), CcuStateResultV1::OK);
    ASSERT_EQ(s.FinishArm(CcuSubmissionV1::SUBMITTED, true), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginTrigger(), CcuStateResultV1::OK);
    ASSERT_EQ(s.FinishTrigger(CcuSubmissionV1::SUBMITTED), CcuStateResultV1::OK);
}
}  // namespace

TEST(CcuRequestState, EnforcesSingleFlightMonotonicSequenceAndOneTrigger) {
    CcuRequestStateV1 s;
    EXPECT_EQ(s.Prepare(0), CcuStateResultV1::STALE_SEQUENCE);
    ASSERT_EQ(s.Prepare(8), CcuStateResultV1::OK);
    EXPECT_EQ(s.Prepare(9), CcuStateResultV1::BUSY);
    ASSERT_EQ(s.BeginArm(), CcuStateResultV1::OK);
    ASSERT_EQ(s.FinishArm(CcuSubmissionV1::SUBMITTED, true), CcuStateResultV1::OK);
    EXPECT_EQ(s.BeginTrigger(), CcuStateResultV1::OK);
    EXPECT_EQ(s.BeginTrigger(), CcuStateResultV1::INVALID_STATE);
    EXPECT_EQ(s.FinishTrigger(CcuSubmissionV1::SUBMITTED), CcuStateResultV1::OK);
}

TEST(CcuRequestState, UnknownSubmissionAndPeerDisagreementRetainOwners) {
    CcuRequestStateV1 s;
    ASSERT_EQ(s.Prepare(1), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginArm(), CcuStateResultV1::OK);
    EXPECT_EQ(s.FinishArm(CcuSubmissionV1::UNKNOWN, false), CcuStateResultV1::RETAINED);
    EXPECT_TRUE(s.Poisoned());
    EXPECT_EQ(s.Close(false), CcuStateResultV1::RETAINED);
}

TEST(CcuRequestState, DataReadyNeedsBothDrainsAndPeerRetirement) {
    CcuRequestStateV1 s;
    submit(s, 3);
    EXPECT_EQ(s.ObserveCompletion(2), CcuStateResultV1::PENDING);
    EXPECT_EQ(s.ObserveCompletion(3), CcuStateResultV1::OK);
    EXPECT_EQ(s.MarkDrained(true, false), CcuStateResultV1::PENDING);
    EXPECT_EQ(s.MarkDrained(false, true), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginRetire(), CcuStateResultV1::OK);
    EXPECT_EQ(s.FinishRetire(false), CcuStateResultV1::RETAINED);
}

TEST(CcuRequestState, SuccessfulRetirementAllowsNextRoundWithoutSequenceReuse) {
    CcuRequestStateV1 s;
    submit(s, 4);
    ASSERT_EQ(s.ObserveCompletion(4), CcuStateResultV1::OK);
    ASSERT_EQ(s.MarkDrained(true, true), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginRetire(), CcuStateResultV1::OK);
    ASSERT_EQ(s.FinishRetire(true), CcuStateResultV1::OK);
    EXPECT_EQ(s.Prepare(4), CcuStateResultV1::STALE_SEQUENCE);
    EXPECT_EQ(s.Prepare(5), CcuStateResultV1::OK);
}

TEST(CcuRequestState, TimeoutDoesNotPoisonButRequiresCompletionAndDrainEvidence) {
    CcuRequestStateV1 s;
    submit(s, 7);
    EXPECT_EQ(s.RetainPending(), CcuStateResultV1::RETAINED);
    EXPECT_FALSE(s.Poisoned());
    EXPECT_EQ(s.RecoverDrained(7, true, true), CcuStateResultV1::OK);
    ASSERT_EQ(s.BeginRetire(), CcuStateResultV1::OK);
    EXPECT_EQ(s.FinishRetire(true), CcuStateResultV1::OK);
}
