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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "runtime.h"

TEST(RuntimeWorkersBoundary, DescriptorLengthsExcludeSharedState) {
    const Runtime runtime;
    EXPECT_LE(runtime_device_launch_size(runtime), runtime_device_extent_size(runtime));
    EXPECT_LE(runtime_device_copy_size(runtime), runtime_device_extent_size(runtime));
    EXPECT_EQ(runtime_device_initialized_prefix_size(runtime), sizeof(DeviceRuntimeLaunchDesc));
    EXPECT_EQ(runtime_device_extent_size(runtime), sizeof(DeviceRuntimeLaunchDesc));
    const auto descriptor_begin = reinterpret_cast<uintptr_t>(&runtime.dev);
    const auto descriptor_end = descriptor_begin + sizeof(runtime.dev);
    const auto workers = reinterpret_cast<uintptr_t>(runtime.get_workers());
    EXPECT_TRUE(workers < descriptor_begin || workers >= descriptor_end);
}

TEST(RuntimeWorkersBoundary, DistinctDescriptorsAddressTheSameSharedState) {
    Runtime first;
    Runtime second;
    ASSERT_NE(first.get_workers(), second.get_workers());
    second.dev.shared_state_addr = first.dev.shared_state_addr;
    ASSERT_EQ(first.get_workers(), second.get_workers());
    ASSERT_EQ(first.get_teardown_gates(), second.get_teardown_gates());
    first.get_workers()[2].task = 0x12345678;
    EXPECT_EQ(second.get_workers()[2].task, 0x12345678u);
    second.get_workers()[2].aicore_done = 3;
    EXPECT_EQ(first.get_workers()[2].aicore_done, 3);
}

TEST(RuntimeWorkersBoundary, EveryHandshakeAndGateOccupiesItsOwnCacheLine) {
    Runtime runtime;
    EXPECT_EQ(sizeof(Handshake), 64U);
    EXPECT_EQ(alignof(Handshake), 64U);
    EXPECT_EQ(offsetof(RuntimeSharedState, teardown_gates) % 64U, 0U);
    for (int i = 0; i < RUNTIME_MAX_WORKER; ++i) {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&runtime.get_workers()[i]) % 64U, 0U);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&runtime.get_teardown_gates()[i]) % 64U, 0U);
    }
}

TEST(RuntimeWorkersBoundary, FreshHostStateIsZeroedAndCannotBeAccidentallyAliasedByCopy) {
    const Runtime runtime;
    const auto *bytes = reinterpret_cast<const unsigned char *>(runtime.get_workers());
    for (size_t i = 0; i < sizeof(RuntimeSharedState::workers); ++i)
        ASSERT_EQ(bytes[i], 0u);
    EXPECT_FALSE(std::is_copy_constructible_v<Runtime>);
    EXPECT_FALSE(std::is_move_constructible_v<Runtime>);
}

TEST(RuntimeWorkersBoundary, ShapeUpdatesDoNotWriteSharedWorkerState) {
    Runtime runtime;
    constexpr unsigned char sentinel = 0x3c;
    auto *region = reinterpret_cast<unsigned char *>(runtime.get_workers());
    std::memset(region, sentinel, sizeof(RuntimeSharedState::workers));
    runtime.set_core_type_rule(9, 3);
    EXPECT_EQ(runtime.core_type_rule_count(), 9U);
    EXPECT_EQ(runtime.core_type_rule(2), CoreType::AIC);
    EXPECT_EQ(runtime.core_type_rule(3), CoreType::AIV);
    for (size_t i = 0; i < sizeof(RuntimeSharedState::workers); ++i)
        ASSERT_EQ(region[i], sentinel);
}
