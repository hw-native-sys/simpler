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
#include <thread>
#include <vector>

#include "aicpu/device_run_result_aicpu.h"
#include "aicpu/platform_entry_args.h"
#include "common/kernel_args.h"
#include "common/native_execution_control.h"

namespace {
constexpr uint64_t kNativeEpoch = 73;

class NativeCaptureIdentity : public ::testing::Test {
protected:
    int prepare(uint32_t callbacks = 2) {
        return native_execution_prepare(
            control, NativeExecutionPrepareArgs{
                         reinterpret_cast<uint64_t>(&control), 1, reinterpret_cast<uint64_t>(&result), kNativeEpoch,
                         reinterpret_cast<uint64_t>(&first_error), callbacks, 0
                     }
        );
    }
    bool publish(int32_t status = 0) {
        return aicpu_publish_run_terminal(
            reinterpret_cast<uint64_t>(&result), kNativeEpoch,
            status == 0 ? DeviceRunVerdict::Ok : DeviceRunVerdict::Error, status,
            status == 0 ? DeviceRunCodeSource::None : DeviceRunCodeSource::ShutdownRc, nullptr, 0
        );
    }
    NativeExecutionControl control{};
    NativeExecutionFirstError first_error{};
    DeviceRunResultRegion result{};
};
}  // namespace

TEST_F(NativeCaptureIdentity, ReplaySequenceNeverBecomesTheNativeResultEpoch) {
    for (uint64_t sequence = 1; sequence <= kNativeEpoch + 2; ++sequence) {
        ASSERT_EQ(prepare(), 0);
        EXPECT_EQ(control.sequence, sequence);
        EXPECT_EQ(result.published, 0u);
        ASSERT_TRUE(publish());
        EXPECT_EQ(native_execution_finish(control, 0, [] {}, [](int32_t) {}), 0);
        EXPECT_EQ(control.remaining_callbacks, 1u);
        EXPECT_EQ(native_execution_finish(control, 0, [] {}, [](int32_t) {}), 0);
        EXPECT_EQ(control.remaining_callbacks, 0u);
        EXPECT_EQ(result.published, kNativeEpoch);
        EXPECT_EQ(device_run_result_terminal(result, kNativeEpoch).state, DeviceRunTerminalState::Succeeded);
    }
}

TEST_F(NativeCaptureIdentity, FrozenLaunchUsesUnstampedCoreReportsAndKeepsResultIdentity) {
    KernelArgs launch{};
    launch.run_result_epoch = kNativeEpoch;
    NativeExecutionControl control{};
    DeviceRunResultRegion result{};
    launch.execution_control = reinterpret_cast<uint64_t>(&control);
    launch.run_result_data_base = reinterpret_cast<uint64_t>(&result);
    AicoreLaunchArgs core{};
    fill_shared_launch_args(core, launch);
    EXPECT_EQ(sizeof(core), 40u);
    EXPECT_EQ(core.report_epoch, 0u);
    EXPECT_EQ(launch.run_result_epoch, kNativeEpoch);

    PlatformEntryArgs entry{};
    entry.unstamped_reports = true;
    set_platform_entry_args(0, entry);
    EXPECT_TRUE(get_platform_entry_args(0).unstamped_reports);
    // An ordinary invocation on the same thread must not inherit capture mode.
    set_platform_entry_args(0, PlatformEntryArgs{});
    EXPECT_FALSE(get_platform_entry_args(0).unstamped_reports);
    launch.execution_control = 0;
    fill_shared_launch_args(core, launch);
    EXPECT_EQ(core.report_epoch, kNativeEpoch);
}

TEST_F(NativeCaptureIdentity, UnfinishedAttemptCannotBeReset) {
    ASSERT_EQ(prepare(), 0);
    ASSERT_TRUE(publish());
    EXPECT_EQ(native_execution_finish(control, 0, [] {}, [](int32_t) {}), 0);
    EXPECT_NE(prepare(), 0);
    EXPECT_EQ(control.remaining_callbacks, 1u);
    EXPECT_EQ(control.sequence, 1u);
}

TEST_F(NativeCaptureIdentity, ConcurrentCallbacksSelectTheAggregateExactlyOnce) {
    constexpr unsigned participants = 16;
    ASSERT_EQ(prepare(participants), 0);
    ASSERT_TRUE(publish(-81));
    std::atomic<unsigned> refreshes{0};
    std::atomic<unsigned> publications{0};
    std::atomic<unsigned> failures{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < participants; ++i) {
        threads.emplace_back([&] {
            const int rc = native_execution_finish(
                control, 0,
                [&] {
                    ++refreshes;
                },
                [&](int32_t code) {
                    EXPECT_EQ(code, -81);
                    ++publications;
                }
            );
            if (rc != 0) ++failures;
        });
    }
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(refreshes.load(), 1u);
    EXPECT_EQ(publications.load(), 0u);
    EXPECT_EQ(failures.load(), 1u);
    EXPECT_EQ(control.remaining_callbacks, 0u);
    EXPECT_EQ(first_error.code, -81);
}
