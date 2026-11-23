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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/l0_profiling.h"
#include "host/l0_capture_collector.h"
#include "common/platform/support/profiling_copy_fault.h"

extern "C" int rtMemset(void *destination, size_t capacity, int32_t value, size_t bytes) {
    if (bytes > capacity) return -1;
    std::memset(destination, value, bytes);
    return 0;
}

class L0CaptureSessionIo : public testing::TestWithParam<size_t> {
protected:
    L0CaptureSession session;
    std::vector<void *> allocations;

    void SetUp() override {
        copy_fault::reset();
        ASSERT_EQ(
            session.arm(
                0, 1, 1, 42,
                [this](size_t bytes) {
                    void *pointer = std::malloc(bytes);
                    allocations.push_back(pointer);
                    return pointer;
                },
                [this](void *pointer) {
                    allocations.erase(std::find(allocations.begin(), allocations.end(), pointer));
                    std::free(pointer);
                    return 0;
                }
            ),
            0
        );
    }

    void TearDown() override {
        session.reset(true);
        for (void *pointer : allocations)
            std::free(pointer);
        copy_fault::reset();
    }
};

TEST_P(L0CaptureSessionIo, SnapshotReadFailureRequiresDeviceReset) {
    copy_fault::arm({GetParam(), -7});

    EXPECT_EQ(session.read_terminal_snapshot(), -7);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1);
    EXPECT_TRUE(session.device_io_failed());
    EXPECT_NE(session.read_terminal_snapshot(), 0);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1);

    session.reset(true);
    EXPECT_TRUE(session.device_io_failed());
    session.confirm_device_reset();
    EXPECT_FALSE(session.device_io_failed());
}

INSTANTIATE_TEST_SUITE_P(
    SnapshotStages, L0CaptureSessionIo, testing::Values(sizeof(L0CoreRegSnapshot), sizeof(L0CoreEndSnapshot))
);
