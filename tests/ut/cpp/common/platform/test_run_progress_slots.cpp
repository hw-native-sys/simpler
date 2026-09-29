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
 * Per-resource-set run progress: which run a set describes, and what a query
 * about another run is told.
 *
 * The cases drive the production record through the sequence a runner drives it
 * through, under that runner's single-progress-owner precondition. What they
 * establish is that one set's answer never becomes another's — not that the
 * record serializes arbitrary concurrent writers, which it does not claim to.
 */

#include <gtest/gtest.h>

#include "host/run_progress_slots.h"

namespace {

constexpr NativeRunIdentity kFirst{41, 1, 100, 0};
constexpr NativeRunIdentity kSecond{42, 1, 101, 1};
//: The same set as `kFirst`, a later run: a pipeline slot is reused and only the
//: epoch is unique for the process lifetime.
constexpr NativeRunIdentity kFirstReused{43, 1, 102, 0};

TEST(RunProgressSlotsTest, AnUnclaimedSetDescribesNoRun) {
    RunProgressSlots slots;
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Idle);
}

TEST(RunProgressSlotsTest, AClaimedSetCarriesItsOwnRunsStates) {
    RunProgressSlots slots;
    slots.claim(kFirst);
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Idle);
    EXPECT_TRUE(slots.publish(kFirst, RunProgressState::Enqueuing));
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Enqueuing);
    EXPECT_TRUE(slots.publish(kFirst, RunProgressState::Submitted));
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Submitted);
}

TEST(RunProgressSlotsTest, APredecessorReachingTerminalLeavesItsSuccessorSubmitted) {
    RunProgressSlots slots;
    slots.claim(kFirst);
    ASSERT_TRUE(slots.publish(kFirst, RunProgressState::Submitted));
    slots.claim(kSecond);
    ASSERT_TRUE(slots.publish(kSecond, RunProgressState::Submitted));

    // The predecessor completes and is drained. With one record per runner this
    // is where the successor would read as complete without any boundary being
    // polled, and its set would be recycled under it.
    ASSERT_TRUE(slots.publish(kFirst, RunProgressState::DeviceComplete));
    ASSERT_TRUE(slots.publish(kFirst, RunProgressState::Drained));

    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Drained);
    EXPECT_EQ(slots.state_of(kSecond), RunProgressState::Submitted);
}

TEST(RunProgressSlotsTest, AQueryFromAReplacedRunIsToldNothing) {
    RunProgressSlots slots;
    slots.claim(kFirst);
    ASSERT_TRUE(slots.publish(kFirst, RunProgressState::Drained));

    // The set is handed to a later run only after the run holding it finalized,
    // so a query naming the previous run is a query from a retired owner.
    slots.claim(kFirstReused);
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Idle);
    EXPECT_EQ(slots.state_of(kFirstReused), RunProgressState::Idle);

    ASSERT_TRUE(slots.publish(kFirstReused, RunProgressState::Submitted));
    // The new run's progress is its own: the previous epoch reads nothing
    // rather than reading the state of whichever run holds the set now.
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Idle);
    EXPECT_EQ(slots.state_of(kFirstReused), RunProgressState::Submitted);
}

TEST(RunProgressSlotsTest, APublishFromAReplacedRunIsRefusedRatherThanApplied) {
    RunProgressSlots slots;
    slots.claim(kFirst);
    slots.claim(kFirstReused);
    ASSERT_TRUE(slots.publish(kFirstReused, RunProgressState::Submitted));

    // A late call from the previous owner — a rollback, or a drain that lost its
    // race with finalize — must not move the state of the run holding the set.
    EXPECT_FALSE(slots.publish(kFirst, RunProgressState::Drained));
    EXPECT_EQ(slots.state_of(kFirstReused), RunProgressState::Submitted);
}

TEST(RunProgressSlotsTest, AClaimClearsTheOutgoingRunsTerminalState) {
    RunProgressSlots slots;
    slots.claim(kFirst);
    ASSERT_TRUE(slots.publish(kFirst, RunProgressState::Drained));
    slots.claim(kFirstReused);

    // Not `Drained`: a set taken for a new run has published nothing about it,
    // and a sticky terminal state inherited from the previous run would be read
    // as this run having completed.
    EXPECT_EQ(slots.state_of(kFirstReused), RunProgressState::Idle);
}

TEST(RunProgressSlotsTest, ASetOutsideTheLayoutAndAnEpochlessRunAreRefused) {
    RunProgressSlots slots;
    constexpr NativeRunIdentity kOutOfRange{44, 1, 103, PTO_PIPELINE_MAX_DEPTH};
    slots.claim(kOutOfRange);
    EXPECT_FALSE(slots.publish(kOutOfRange, RunProgressState::Submitted));
    EXPECT_EQ(slots.state_of(kOutOfRange), RunProgressState::Idle);

    // Epoch zero is what a record that never carried a run reads as, so it can
    // never match a set and can never be published for one.
    constexpr NativeRunIdentity kNoEpoch{0, 1, 104, 0};
    slots.claim(kFirst);
    EXPECT_FALSE(slots.publish(kNoEpoch, RunProgressState::Submitted));
    EXPECT_EQ(slots.state_of(kNoEpoch), RunProgressState::Idle);
    EXPECT_EQ(slots.state_of(kFirst), RunProgressState::Idle);
}

}  // namespace
