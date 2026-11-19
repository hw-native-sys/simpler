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
#include "runtime.h"

namespace {
constexpr size_t kWorkersBytes = sizeof(DeviceRuntimeLaunchDesc::workers);
}

// This runtime publishes a fixed steady prefix: it ends exactly where the
// handshake region begins, for every run.
TEST(RuntimeWorkersBoundary, TheSteadyStatePrefixStopsWhereTheHandshakeRegionBegins) {
    const Runtime runtime;
    EXPECT_EQ(runtime_device_copy_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers));
}

// What the move is for, stated as the difference between the two prefixes.
TEST(RuntimeWorkersBoundary, ASteadyStateRunNoLongerRepublishesTheHandshakeRegion) {
    const Runtime runtime;
    EXPECT_EQ(runtime_device_initialized_prefix_size(runtime) - runtime_device_copy_size(runtime), kWorkersBytes)
        << "the two prefixes must differ by exactly the handshake region and nothing else";
}
