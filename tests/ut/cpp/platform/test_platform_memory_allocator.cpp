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

/**
 * UT for the sim variant of src/a2a3/platform/sim/host/memory_allocator.cpp.
 *
 * The sim implementation wraps malloc/free and tracks live pointers in a
 * std::set, making it directly unit-testable without any hardware or CANN
 * runtime dependency.  Hardware (onboard) variants that call rtMalloc/rtFree
 * are intentionally NOT covered here -- they need an Ascend device and are
 * exercised by the hardware CI job.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

#include "host/memory_allocator.h"

namespace {

class MemoryAllocatorTest : public ::testing::Test {
protected:
    MemoryAllocator alloc_;
};

}  // namespace

// ---------- Happy path ----------

TEST_F(MemoryAllocatorTest, Alloc_ValidSize_ReturnsUsablePointer) {
    // Act
    void *p = alloc_.alloc(128);

    // Assert
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(alloc_.get_allocation_count(), 1U);

    // Use memory to ensure it is real, writable storage.
    std::memset(p, 0xAB, 128);

    // Cleanup
    EXPECT_EQ(alloc_.free(p), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

TEST_F(MemoryAllocatorTest, Alloc_MultipleTracksAll) {
    void *p1 = alloc_.alloc(16);
    void *p2 = alloc_.alloc(32);
    void *p3 = alloc_.alloc(64);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    ASSERT_NE(p3, nullptr);

    EXPECT_EQ(alloc_.get_allocation_count(), 3U);

    alloc_.free(p1);
    alloc_.free(p2);
    alloc_.free(p3);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

// ---------- Free: edge cases ----------

TEST_F(MemoryAllocatorTest, Free_Nullptr_ReturnsZeroNoop) {
    // Contract: free(nullptr) is a safe no-op.
    EXPECT_EQ(alloc_.free(nullptr), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

TEST_F(MemoryAllocatorTest, Free_UntrackedPointer_ReturnsZeroLeavesStateAlone) {
    // Allocate something to populate the set.
    void *tracked = alloc_.alloc(8);
    ASSERT_NE(tracked, nullptr);

    // An unrelated, untracked address must NOT be freed -- memory_allocator
    // only calls std::free() for pointers it allocated itself.
    int stack_int = 0;
    EXPECT_EQ(alloc_.free(&stack_int), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 1U);

    alloc_.free(tracked);
}

TEST_F(MemoryAllocatorTest, Free_SamePointerTwice_SecondCallIsNoop) {
    void *p = alloc_.alloc(8);
    ASSERT_NE(p, nullptr);

    EXPECT_EQ(alloc_.free(p), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);

    // Second free on the same (now untracked) address: no crash, no state change.
    EXPECT_EQ(alloc_.free(p), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

// ---------- finalize ----------

TEST_F(MemoryAllocatorTest, Finalize_FreesAllTrackedAllocations) {
    // Arrange
    (void)alloc_.alloc(16);
    (void)alloc_.alloc(32);
    ASSERT_EQ(alloc_.get_allocation_count(), 2U);

    // Act
    EXPECT_EQ(alloc_.finalize(), 0);

    // Assert
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

TEST_F(MemoryAllocatorTest, Finalize_IdempotentWhenEmpty) {
    EXPECT_EQ(alloc_.finalize(), 0);
    EXPECT_EQ(alloc_.finalize(), 0);
    EXPECT_EQ(alloc_.get_allocation_count(), 0U);
}

TEST_F(MemoryAllocatorTest, Destructor_CallsFinalizeAutomatically) {
    // Use a local-scope allocator to trigger RAII cleanup.
    size_t count_after_destruct = 0;
    {
        MemoryAllocator scoped;
        (void)scoped.alloc(16);
        (void)scoped.alloc(32);
        ASSERT_EQ(scoped.get_allocation_count(), 2U);
        // scoped goes out of scope here -- destructor must free the 2 allocations.
    }
    // We can't query the destroyed allocator, but reaching here without leak
    // reports under asan/ubsan is the observable signal.  A fresh allocator
    // starts at zero -- confirm basic post-condition.
    MemoryAllocator fresh;
    count_after_destruct = fresh.get_allocation_count();
    EXPECT_EQ(count_after_destruct, 0U);
}
