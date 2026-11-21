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
#include "aicpu/entry_cache_maintenance.h"
#include "aicpu/device_run_result_aicpu.h"

namespace {
using namespace simpler::tmr;
using Range = std::pair<size_t, size_t>;
TEST(EntryCacheMaintenance, TerminalSnapshotAndOwnMetadataGateEveryActualRange) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    PlatformEntryArgs view{
        &runtime, LAUNCH_ENVELOPE_HEADER_BYTES, 0, 0, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)
    };
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::Live);
    auto run = [&](bool ok) {
        std::vector<Range> ranges;
        clean_entry_runtime(runtime, view, ok, [&](const void *address, size_t bytes) {
            ranges.emplace_back(
                static_cast<const unsigned char *>(address) - reinterpret_cast<const unsigned char *>(&runtime), bytes
            );
        });
        return ranges;
    };
    const size_t shared_at = reinterpret_cast<uintptr_t>(runtime.get_workers()) - reinterpret_cast<uintptr_t>(&runtime);
    for (uint32_t count :
         {0u, 1u, static_cast<uint32_t>(CHIP_MAX_TENSOR_ARGS - 1), static_cast<uint32_t>(CHIP_MAX_TENSOR_ARGS)}) {
        runtime.dev.entry_tensor_count_ = view.tensor_count = count;
        const auto ranges = run(true);
        if (count == CHIP_MAX_TENSOR_ARGS) {
            EXPECT_EQ(ranges, (std::vector<Range>{{0, sizeof(runtime.dev)}, {shared_at, sizeof(RuntimeSharedState)}}));
        } else {
            EXPECT_EQ(
                ranges, (std::vector<Range>{
                            {0, entry_tensor_offset + count * sizeof(Tensor)}, {shared_at, sizeof(RuntimeSharedState)}
                        })
            );
        }
        EXPECT_EQ(run(false), (std::vector<Range>{{0, sizeof(runtime.dev)}, {shared_at, sizeof(RuntimeSharedState)}}));
    }
    ++view.tensor_count;
    EXPECT_EQ(run(true), (std::vector<Range>{{0, sizeof(runtime.dev)}, {shared_at, sizeof(RuntimeSharedState)}}));
    view.tensor_count = 0;
    runtime.dev.entry_tensor_count_ = 0;
    for (uint32_t policy : {0u, 1u, 99u}) {
        view.cache_maintenance = policy;
        EXPECT_EQ(run(true), (std::vector<Range>{{0, sizeof(runtime.dev)}, {shared_at, sizeof(RuntimeSharedState)}}));
    }
    RunTerminalPublisher publisher;
    EXPECT_FALSE(publisher.snapshot_is_ok());
    publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
    EXPECT_TRUE(publisher.snapshot_is_ok());
    publisher.take({}, nullptr, 0);
    EXPECT_FALSE(publisher.snapshot_is_ok());
}

TEST(EntryCacheMaintenance, KernelInvocationCleanupNeverTouchesAbsentEntryStorage) {
    Runtime runtime;
    RuntimeSharedState shared;
    runtime.dev.shared_state_addr = reinterpret_cast<uint64_t>(&shared);
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    PlatformEntryArgs view{};
    view.source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    for (bool success : {false, true}) {
        std::vector<std::pair<const void *, size_t>> ranges;
        clean_entry_runtime(runtime, view, success, [&](const void *address, size_t bytes) {
            ranges.emplace_back(address, bytes);
        });
        ASSERT_EQ(ranges.size(), 2u);
        EXPECT_EQ(ranges[0].first, &runtime);
        EXPECT_EQ(ranges[0].second, offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_));
        EXPECT_EQ(ranges[1].first, shared.workers);
        EXPECT_EQ(ranges[1].second, sizeof(shared));
    }
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::FullFreshen);
    bool touched = false;
    EXPECT_FALSE(freshen_entry_descriptor(runtime, view, [&](const void *, size_t) {
        touched = true;
    }));
    EXPECT_FALSE(touched);
}

TEST(EntryCacheMaintenance, AlignedLocalLaunchSnapshotOnlyCleansSharedDeviceState) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    PlatformEntryArgs view{};
    view.source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    view.runtime_descriptor_bytes = RUNTIME_LAUNCH_DESCRIPTOR_BYTES;
    for (bool success : {false, true}) {
        std::vector<std::pair<const void *, size_t>> ranges;
        clean_entry_runtime(runtime, view, success, [&](const void *address, size_t bytes) {
            ranges.emplace_back(address, bytes);
        });
        ASSERT_EQ(ranges.size(), 1u);
        EXPECT_EQ(ranges[0].first, runtime.get_workers());
        EXPECT_EQ(ranges[0].second, sizeof(RuntimeSharedState));
    }
}

TEST(EntryCacheMaintenance, DescriptorRefreshReadsCountsOnlyAfterFixedHeadAndBoundsActiveRange) {
    Runtime runtime;
    PlatformEntryArgs view;
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::FullFreshen);
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::Descriptor);
    runtime.dev.entry_tensor_count_ = 7;
    runtime.dev.entry_scalar_count_ = 2;
    runtime.dev.orch_args_storage_.tensor_count_ = -1;
    runtime.dev.orch_args_storage_.scalar_count_ = -1;
    std::vector<Range> ranges;
    auto clean = [&](const void *address, size_t bytes) {
        ranges.emplace_back(
            static_cast<const unsigned char *>(address) - reinterpret_cast<const unsigned char *>(&runtime), bytes
        );
        if (ranges.size() == 1) {
            runtime.dev.orch_args_storage_.tensor_count_ = 7;
            runtime.dev.orch_args_storage_.scalar_count_ = 2;
        }
    };
    ASSERT_TRUE(freshen_entry_descriptor(runtime, view, clean));
    EXPECT_EQ(
        ranges, (std::vector<Range>{
                    {offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_), offsetof(EntryArgsStorage, tensors_)},
                    {entry_tensor_offset, 7 * sizeof(Tensor)}
                })
    );
    for (uint32_t policy : {2u, 99u}) {
        view.cache_maintenance = policy;
        EXPECT_FALSE(valid_entry_cache_route(view));
    }
    view.cache_maintenance = 1;
    runtime.dev.entry_tensor_count_ = CHIP_MAX_TENSOR_ARGS + 1;
    ranges.clear();
    EXPECT_FALSE(freshen_entry_descriptor(runtime, view, clean));
    EXPECT_EQ(ranges.size(), 1u);
}
}  // namespace
