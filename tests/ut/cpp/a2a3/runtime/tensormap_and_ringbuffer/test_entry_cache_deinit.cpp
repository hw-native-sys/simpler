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
#include <utility>
#include <vector>
#include "host/raii_scope_guard.h"

// Include the implementation to exercise the private executor, without a
// production test hook or an alternative teardown implementation.
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "aicpu/aicpu_executor.cpp"

namespace {
std::vector<std::pair<const void *, size_t>> invalidations;
}
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) { invalidations.emplace_back(address, bytes); }
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance

TEST(EntryCacheDeinit, FailedSnapshotFallsBackToFullThroughExecutor) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    runtime.dev.entry_tensor_count_ = 4;
    PlatformEntryArgs view{
        &runtime, LAUNCH_ENVELOPE_HEADER_BYTES, 4, 0, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)
    };
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::Live);
    // Direct executor tests provide the affinity cell normally bound by the wrapper.
    int32_t affinity_index = 0;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&affinity_index), 0);
    auto unbind = RAIIScopeGuard([] {
        platform_aicpu_affinity_bind_thread(nullptr);
    });
    set_platform_entry_args(0, view);
    AicpuExecutor executor;
    for (auto verdict : {DeviceRunVerdict::Ok, DeviceRunVerdict::Error, DeviceRunVerdict::None, DeviceRunVerdict::Ok}) {
        const bool success = verdict == DeviceRunVerdict::Ok;
        executor.terminal_publisher_.take(
            RunTerminalSelection{
                verdict, success ? 0 : -7, success ? DeviceRunCodeSource::None : DeviceRunCodeSource::ThreadRc
            },
            nullptr, 0
        );
        invalidations.clear();
        executor.deinit(&runtime);
        if (success) {
            ASSERT_EQ(invalidations.size(), 2u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, simpler::tmr::entry_tensor_offset + 4 * sizeof(Tensor));
            EXPECT_EQ(
                invalidations[1].first,
                reinterpret_cast<const char *>(&runtime) + offsetof(DeviceRuntimeLaunchDesc, workers)
            );
        } else {
            ASSERT_EQ(invalidations.size(), 1u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, sizeof(runtime.dev));
        }
        EXPECT_EQ(executor.terminal_publisher_.snapshot_is_ok(), success);
    }
    set_platform_entry_args(0, {});
}
