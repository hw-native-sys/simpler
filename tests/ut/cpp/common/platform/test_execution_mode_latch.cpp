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

// Write-once execution identity. Both platform runner bases hold one and every
// kernel-mode guard reads it, so these cases pin the whole contract the guards
// depend on: the unlatched start state, idempotence for the held mode, mutual
// exclusion in both directions, and re-latching across a finalized lifetime.

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "host/execution_mode_latch.h"

namespace {

TEST(ExecutionModeLatch, StartsUnlatched) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latched_mode(), std::nullopt);
    EXPECT_FALSE(latch.is_latched());
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, LatchingIsIdempotentForTheHeldMode) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_TRUE(latch.is_latched());
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, ModesAreMutuallyExclusiveInBothDirections) {
    ExecutionModeLatch program;
    EXPECT_EQ(program.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(program.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(program.is_kernel());

    ExecutionModeLatch kernel;
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_PROGRAM), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(kernel.is_kernel());
}

// finalize resets a runner's device state but not its identity, so the
// init -> finalize -> init sequence the program path already supports must
// still latch cleanly the second time.
TEST(ExecutionModeLatch, SameModeRelatchesAfterAFinalizedLifetime) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(ExecutionModeLatch, ConcurrentOppositeClaimsChooseExactlyOneIdentity) {
    ExecutionModeLatch latch;
    std::atomic<int> ready{0};
    int program_rc = PTO_RUNTIME_ERR_INTERNAL;
    int kernel_rc = PTO_RUNTIME_ERR_INTERNAL;
    auto claim = [&](SimplerExecutionMode mode, int &result) {
        ready.fetch_add(1);
        while (ready.load() != 2) {}
        result = latch.latch(mode);
    };
    std::thread program([&] {
        claim(SIMPLER_MODE_PROGRAM, program_rc);
    });
    std::thread kernel([&] {
        claim(SIMPLER_MODE_KERNEL, kernel_rc);
    });
    program.join();
    kernel.join();
    ASSERT_TRUE(latch.latched_mode().has_value());
    if (program_rc == 0) {
        EXPECT_EQ(kernel_rc, PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    } else {
        EXPECT_EQ(program_rc, PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_EQ(kernel_rc, 0);
        EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_KERNEL);
    }
}

}  // namespace
