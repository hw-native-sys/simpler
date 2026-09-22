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

#include "host_build_graph/async_poll_phase_accumulator.h"

using simpler::hbg::AsyncPollPhaseAccumulator;

TEST(AsyncPollPhaseAccumulatorTest, PendingPollsAndResolvedPollFormOneCompactSummary) {
    AsyncPollPhaseAccumulator accumulator;

    accumulator.begin();
    EXPECT_FALSE(accumulator.add_poll(100, 110, 0));
    EXPECT_FALSE(accumulator.add_poll(200, 220, 0));
    EXPECT_TRUE(accumulator.add_poll(300, 340, 2));

    ASSERT_TRUE(accumulator.active());
    const auto summary = accumulator.flush(400);

    EXPECT_EQ(summary.start_time, 330u);
    EXPECT_EQ(summary.end_time, 400u);
    EXPECT_EQ(summary.resolved, 2u);
    EXPECT_FALSE(accumulator.active());
}

TEST(AsyncPollPhaseAccumulatorTest, FlushWithoutResolutionResetsTheNextBatch) {
    AsyncPollPhaseAccumulator accumulator;

    accumulator.add_poll(10, 15, 0);
    const auto pending = accumulator.flush(20);
    EXPECT_EQ(pending.start_time, 15u);
    EXPECT_EQ(pending.end_time, 20u);
    EXPECT_EQ(pending.resolved, 0u);

    accumulator.add_poll(30, 37, 1);
    const auto resolved = accumulator.flush(40);
    EXPECT_EQ(resolved.start_time, 33u);
    EXPECT_EQ(resolved.end_time, 40u);
    EXPECT_EQ(resolved.resolved, 1u);
}

TEST(AsyncPollPhaseAccumulatorTest, FailedPollRequestsAFlushWithoutResolution) {
    AsyncPollPhaseAccumulator accumulator;

    EXPECT_TRUE(accumulator.add_poll(50, 60, 0, true));
    const auto failed = accumulator.flush(60);

    EXPECT_EQ(failed.start_time, 50u);
    EXPECT_EQ(failed.end_time, 60u);
    EXPECT_EQ(failed.resolved, 0u);
}
