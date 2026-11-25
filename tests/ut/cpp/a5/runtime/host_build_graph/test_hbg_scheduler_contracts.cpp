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
#include <cstdint>

#include "aicore_scheduler_error.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"

namespace {
TEST(AicoreSchedulerError, MapsInternalFailureToExistingHostStatus) {
    EXPECT_EQ(aicore_scheduler_runtime_error_code(0), SIMPLER_ERROR_NONE);
    EXPECT_EQ(aicore_scheduler_runtime_error_code(1), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)),
        SIMPLER_ERROR_SCHEDULER_TIMEOUT
    );
    EXPECT_EQ(aicore_scheduler_runtime_error_code(UINT64_MAX), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW)),
        SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW
    );
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_COMPLETION_INVALID)),
        SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
    );
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_REGISTRATION_FAILED)),
        SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
    );
}

TEST(AicoreSchedulerError, LatchesFirstHostVisibleFailure) {
    std::atomic<int32_t> status{SIMPLER_ERROR_NONE};
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 0));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_NONE);

    EXPECT_TRUE(latch_aicore_scheduler_runtime_error(&status, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_SCHEDULER_TIMEOUT);

    status.store(SIMPLER_ERROR_READY_QUEUE_OVERFLOW, std::memory_order_relaxed);
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 2));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(AicoreSchedulerWatchdog, UsesElapsedWallClockBudget) {
    EXPECT_FALSE(scheduler_watchdog_expired(100, UINT64_MAX, 0));
    EXPECT_FALSE(scheduler_watchdog_expired(100, 199, 100));
    EXPECT_TRUE(scheduler_watchdog_expired(100, 200, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(UINT64_MAX - 10, 3, 15));
    EXPECT_TRUE(scheduler_watchdog_expired(UINT64_MAX - 10, 4, 15));
}

TEST(AicoreSchedulerWatchdog, RejectsBackwardAndHalfRangeSamples) {
    constexpr uint64_t half_range = uint64_t{1} << 63;
    EXPECT_TRUE(scheduler_watchdog_expired(0, half_range - 1, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(0, half_range, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(0, UINT64_MAX, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(1000, 800, 0));
}

TEST(SchedulerClusterTopology, UsesMixedKernelLaunchCoordinate) {
    constexpr int32_t kClusters = 4;
    SchedulerClusterCoordinate coordinate{};
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(2, true, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 0);
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(9, false, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 2);
    EXPECT_FALSE(scheduler_cluster_coordinate_from_worker(12, false, kClusters, 2, &coordinate));
}
}  // namespace
