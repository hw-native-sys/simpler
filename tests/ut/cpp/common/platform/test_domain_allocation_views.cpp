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
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "platform_comm/comm.h"
#include "platform_comm/comm_context.h"

namespace {
class DomainAllocationViews : public testing::Test {
protected:
    void SetUp() override {
        root = "/tmp/simpler_domain_views_" + std::to_string(getpid()) + "_" +
               testing::UnitTest::GetInstance()->current_test_info()->name();
        handle = comm_init(0, 1, nullptr, root.c_str());
        ASSERT_NE(handle, nullptr);
        const uint32_t rank = 0;
        ASSERT_EQ(comm_alloc_domain_windows(handle, 41, &rank, 1, 0, 4096, 0, &context, &window, &window_bytes), 0);
        allocated = true;
    }
    void TearDown() override {
        if (allocated) EXPECT_EQ(comm_release_domain_windows(handle, 41, 1, 0), 0);
        if (handle != nullptr) EXPECT_EQ(comm_destroy(handle), 0);
    }
    std::vector<CommAllocationSpan> query() {
        size_t count = 0;
        EXPECT_EQ(comm_query_domain_allocation(handle, 41, nullptr, 0, &count), 0);
        std::vector<CommAllocationSpan> views(count);
        EXPECT_EQ(comm_query_domain_allocation(handle, 41, views.data(), views.size(), &count), 0);
        return views;
    }
    CommHandle handle = nullptr;
    bool allocated = false;
    std::string root;
    uint64_t context = 0;
    uint64_t window = 0;
    size_t window_bytes = 0;
};

TEST_F(DomainAllocationViews, PublishedContextAndWindowHaveBackendOwnedBounds) {
    const auto views = query();
    ASSERT_EQ(views.size(), 2u);
    EXPECT_EQ(views[0].addr, context);
    EXPECT_EQ(views[0].bytes, sizeof(CommContext));
    EXPECT_EQ(views[1].addr, window);
    EXPECT_EQ(views[1].bytes, window_bytes);
}

TEST_F(DomainAllocationViews, WrongHandleOrAllocationCannotAttestAnOwner) {
    size_t count = 99;
    EXPECT_NE(comm_query_domain_allocation(nullptr, 41, nullptr, 0, &count), 0);
    EXPECT_EQ(count, 0u);
    count = 99;
    EXPECT_NE(comm_query_domain_allocation(handle, 42, nullptr, 0, &count), 0);
    EXPECT_EQ(count, 0u);
    auto other = comm_init(0, 1, nullptr, (root + "_other").c_str());
    ASSERT_NE(other, nullptr);
    count = 99;
    EXPECT_NE(comm_query_domain_allocation(other, 41, nullptr, 0, &count), 0);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(comm_destroy(other), 0);
}

TEST_F(DomainAllocationViews, ShortOutputPublishesNoPartialGrant) {
    CommAllocationSpan sentinel{123, 456};
    size_t count = 0;
    EXPECT_NE(comm_query_domain_allocation(handle, 41, &sentinel, 1, &count), 0);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(sentinel.addr, 123u);
    EXPECT_EQ(sentinel.bytes, 456u);
    EXPECT_NE(comm_query_domain_allocation(handle, 41, nullptr, 1, &count), 0);
    EXPECT_EQ(count, 0u);
    EXPECT_NE(comm_query_domain_allocation(handle, 41, &sentinel, 1, nullptr), 0);
}

TEST_F(DomainAllocationViews, QueryDoesNotChangeWindowContent) {
    auto *bytes = reinterpret_cast<uint8_t *>(window);
    bytes[0] = 0x5A;
    bytes[window_bytes - 1] = 0xA5;
    const auto views = query();
    EXPECT_EQ(views.size(), 2u);
    EXPECT_EQ(bytes[0], 0x5A);
    EXPECT_EQ(bytes[window_bytes - 1], 0xA5);
}

TEST_F(DomainAllocationViews, MutableContextContentsCannotExpandTheOwnedMapping) {
    auto *metadata = reinterpret_cast<CommContext *>(context);
    const auto saved_address = metadata->windowsIn[0];
    const auto saved_size = metadata->winSize;
    metadata->windowsIn[0] = 1;
    metadata->winSize = UINT64_MAX;
    const auto views = query();
    ASSERT_EQ(views.size(), 2u);
    EXPECT_EQ(views[1].addr, window);
    EXPECT_EQ(views[1].bytes, window_bytes);
    metadata->windowsIn[0] = saved_address;
    metadata->winSize = saved_size;
}

TEST_F(DomainAllocationViews, ReleasedAllocationHasNoPublishedViews) {
    ASSERT_EQ(comm_release_domain_windows(handle, 41, 1, 0), 0);
    allocated = false;
    size_t count = 99;
    EXPECT_NE(comm_query_domain_allocation(handle, 41, nullptr, 0, &count), 0);
    EXPECT_EQ(count, 0u);
}
}  // namespace
