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

#include <cstring>

#include <gtest/gtest.h>

#include "host_build_graph/runtime.h"

namespace {

// None of these device addresses is host-readable. Capability lookups must use
// only the registration's retained host metadata, never a CoreCallable in GM.
constexpr uint64_t kTableDevice = 0x710000000000ull;
constexpr uint64_t kObjects[] = {0x720000001000ull, 0, 0x720000002000ull};
constexpr CoreAsyncCapability kDeferred{1, CoreAsyncMode::DEFERRED, 64, CORE_ASYNC_CONDITION_COUNTER};
constexpr CoreAsyncCapability kSynchronous{1, CoreAsyncMode::SYNCHRONOUS, 0, 0};

void expect_missing(const Runtime &runtime, int id) {
    CoreAsyncCapability out = kDeferred;
    EXPECT_FALSE(runtime.get_function_async_capability(id, &out));
    EXPECT_EQ(out.mode, CoreAsyncMode::UNKNOWN);
    EXPECT_EQ(out.version, CORE_ASYNC_CAPABILITY_VERSION);
    EXPECT_EQ(out.max_conditions, 0u);
    EXPECT_EQ(out.condition_types, 0u);
}

TEST(HbgCallableCapability, DistinguishesRegisteredUnknownFromHolesAndUnboundState) {
    Runtime runtime;
    expect_missing(runtime, 0);
    const CoreAsyncCapability capabilities[] = {{}, kDeferred, kSynchronous};
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 3, capabilities);
    CoreAsyncCapability out;
    ASSERT_TRUE(runtime.get_function_async_capability(0, &out));
    EXPECT_EQ(out.mode, CoreAsyncMode::UNKNOWN);
    expect_missing(runtime, 1);
    ASSERT_TRUE(runtime.get_function_async_capability(2, &out));
    EXPECT_EQ(out.mode, CoreAsyncMode::SYNCHRONOUS);
    expect_missing(runtime, -1);
    expect_missing(runtime, 3);
    expect_missing(runtime, CHIP_MAX_FUNC_ID);
    EXPECT_FALSE(runtime.get_function_async_capability(0, nullptr));
}

TEST(HbgCallableCapability, NarrowerOptionalAndEmptyRebindsDropOldCapabilities) {
    Runtime runtime;
    const CoreAsyncCapability first[] = {kDeferred, {}, kDeferred};
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 3, first);
    CoreAsyncCapability out;
    ASSERT_TRUE(runtime.get_function_async_capability(2, &out));
    const CoreAsyncCapability second[] = {kSynchronous};
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 1, second);
    ASSERT_TRUE(runtime.get_function_async_capability(0, &out));
    EXPECT_EQ(out.mode, CoreAsyncMode::SYNCHRONOUS);
    expect_missing(runtime, 2);

    // Existing callers can omit capability metadata, but cannot inherit it.
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 1);
    expect_missing(runtime, 0);
    EXPECT_EQ(runtime.get_function_bin_addr(0), kObjects[0]);
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 1, second);
    runtime.clear_callable_tables();
    expect_missing(runtime, 0);
    EXPECT_EQ(runtime.callable_table_len(), 0u);
}

TEST(HbgCallableCapability, InvalidTableInputsClearEveryReference) {
    Runtime runtime;
    const CoreAsyncCapability capabilities[] = {kDeferred};
    const auto check_cleared = [&]() {
        expect_missing(runtime, 0);
        EXPECT_EQ(runtime.callable_table_len(), 0u);
        EXPECT_EQ(runtime.dev.callable_table_addr_, 0u);
        EXPECT_EQ(runtime.callable_entry_table_addr(), 0u);
    };
    for (int invalid = 0; invalid < 4; ++invalid) {
        runtime.set_callable_tables(kObjects, kTableDevice, kTableDevice + 64, 1, capabilities);
        runtime.set_callable_tables(
            invalid == 0 ? nullptr : kObjects, invalid == 1 ? 0 : kTableDevice, kTableDevice + 64,
            invalid == 2 ? 0 : (invalid == 3 ? CHIP_MAX_FUNC_ID + 1 : 1), capabilities
        );
        check_cleared();
    }
}

TEST(HbgCallableCapability, BorrowsHostMetadataWithoutChangingTheDeviceImage) {
    Runtime runtime;
    // Initialize padding as well so the whole-image comparison is meaningful.
    std::memset(&runtime.dev, 0, sizeof(runtime.dev));
    CoreAsyncCapability capabilities[] = {kDeferred};
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 1);
    const auto device_before = runtime.dev;
    runtime.set_callable_tables(kObjects, kTableDevice, 0, 1, capabilities);
    EXPECT_EQ(std::memcmp(&runtime.dev, &device_before, sizeof(runtime.dev)), 0);
    // Mutation is only a test probe: production keeps registration tables
    // immutable. Seeing this value proves binding did not copy the table.
    capabilities[0].max_conditions = 2;
    CoreAsyncCapability out;
    ASSERT_TRUE(runtime.get_function_async_capability(0, &out));
    EXPECT_EQ(out.max_conditions, 2u);
}

}  // namespace
