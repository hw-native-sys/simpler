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

#include <cstddef>
#include <cstdlib>

#include <gtest/gtest.h>

#include "host/memory_allocator.h"

namespace {

TEST(MemoryAllocatorTest, FreshAllocatorHasZeroCommittedBytes) {
    MemoryAllocator a;
    EXPECT_EQ(a.committed_bytes(), 0u);
    EXPECT_EQ(a.get_allocation_count(), 0u);
}

TEST(MemoryAllocatorTest, AllocAccumulatesBytesAndFreeDecrements) {
    MemoryAllocator a;

    void *p = a.alloc(100);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(a.committed_bytes(), 100u);
    EXPECT_EQ(a.get_allocation_count(), 1u);

    void *q = a.alloc(256);
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(a.committed_bytes(), 100u + 256u);
    EXPECT_EQ(a.get_allocation_count(), 2u);

    a.free(p);
    EXPECT_EQ(a.committed_bytes(), 256u);
    EXPECT_EQ(a.get_allocation_count(), 1u);

    a.free(q);
    EXPECT_EQ(a.committed_bytes(), 0u);
    EXPECT_EQ(a.get_allocation_count(), 0u);
}

TEST(MemoryAllocatorTest, FreeOfUntrackedOrNullPointerIsNoop) {
    MemoryAllocator a;
    void *p = a.alloc(64);
    ASSERT_NE(p, nullptr);

    int stack_sentinel = 0;
    a.free(&stack_sentinel);  // untracked address -> no-op
    a.free(nullptr);          // nullptr -> no-op
    EXPECT_EQ(a.committed_bytes(), 64u);
    EXPECT_EQ(a.get_allocation_count(), 1u);

    a.free(p);
    EXPECT_EQ(a.committed_bytes(), 0u);
}

TEST(MemoryAllocatorTest, FinalizeReleasesAllAndZeroesCounter) {
    MemoryAllocator a;
    ASSERT_NE(a.alloc(128), nullptr);
    ASSERT_NE(a.alloc(512), nullptr);
    EXPECT_EQ(a.committed_bytes(), 128u + 512u);

    EXPECT_EQ(a.finalize(), 0);  // sim finalize is infallible
    EXPECT_EQ(a.committed_bytes(), 0u);
    EXPECT_EQ(a.get_allocation_count(), 0u);

    // Idempotent: a second finalize on an empty allocator is a no-op.
    EXPECT_EQ(a.finalize(), 0);
    EXPECT_EQ(a.committed_bytes(), 0u);
}

TEST(MemoryAllocatorTest, AbandonClearsTrackingWithoutCallingPlatformFree) {
    MemoryAllocator a;
    void *p = a.alloc(128);
    void *q = a.alloc(256);
    ASSERT_NE(p, nullptr);
    ASSERT_NE(q, nullptr);

    a.abandon_after_device_failure();
    EXPECT_EQ(a.committed_bytes(), 0u);
    EXPECT_EQ(a.get_allocation_count(), 0u);
    EXPECT_EQ(a.finalize(), 0);

    // The sim test has no device reset to reclaim abandoned allocations.
    std::free(p);
    std::free(q);
}

TEST(MemoryAllocatorTest, DestructorFreesLiveAllocationsWithoutLeak) {
    {
        MemoryAllocator a;
        ASSERT_NE(a.alloc(200), nullptr);
        ASSERT_NE(a.alloc(300), nullptr);
        EXPECT_EQ(a.committed_bytes(), 500u);
        // Out-of-scope destructor runs finalize() internally; the contract is
        // no leak / no crash (the counter is not queryable after destruction).
    }
    // A fresh allocator after one with live allocs went out of scope is clean.
    MemoryAllocator b;
    EXPECT_EQ(b.committed_bytes(), 0u);
}

}  // namespace
