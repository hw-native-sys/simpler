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
// Unit tests for src/common/pto_runtime2/device_pool.h.
//
// DevicePool is a single-slot pooled buffer holder used by DeviceRunner to
// keep the PTO2 GM heap / shared memory alive across worker.run() dispatches.
// The pool's contract is:
//   - acquire(size) returns the existing pointer if it is already large
//     enough; otherwise frees the old buffer and allocates a fresh one.
//   - acquire(0) is a no-op that returns nullptr.
//   - release() frees the buffer and resets state.
//   - The destructor releases the buffer.
//
// These tests exercise the contract with mock alloc/free callables so they
// do not depend on any real device runtime.

#include <cstddef>
#include <unordered_set>

#include <gtest/gtest.h>

#include "pto_runtime2/device_pool.h"

using pto_runtime2::DevicePool;

namespace {

// Minimal allocator that hands out fresh host buffers and tracks how many
// times alloc / free were called. Each returned pointer is unique so tests
// can assert pointer identity without relying on the system allocator
// reusing the same address.
struct MockAllocator {
    int alloc_count = 0;
    int free_count = 0;
    std::unordered_set<void *> live;

    void *alloc(size_t size) {
        ++alloc_count;
        void *p = ::operator new(size);
        live.insert(p);
        return p;
    }

    void free(void *p) {
        ++free_count;
        EXPECT_EQ(live.count(p), 1u) << "free called on pointer not currently live";
        live.erase(p);
        ::operator delete(p);
    }
};

DevicePool make_pool(MockAllocator *m) {
    return DevicePool(
        [m](size_t n) {
            return m->alloc(n);
        },
        [m](void *p) {
            m->free(p);
        }
    );
}

TEST(DevicePoolTest, AcquireSameSizeReusesBuffer) {
    MockAllocator m;
    DevicePool pool = make_pool(&m);

    void *p1 = pool.acquire(128);
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(m.alloc_count, 1);
    EXPECT_EQ(m.free_count, 0);

    void *p2 = pool.acquire(128);
    EXPECT_EQ(p2, p1) << "same-size acquire must reuse existing buffer";
    EXPECT_EQ(m.alloc_count, 1);
    EXPECT_EQ(m.free_count, 0);
}

TEST(DevicePoolTest, AcquireLargerSizeTriggersRealloc) {
    MockAllocator m;
    DevicePool pool = make_pool(&m);

    void *p1 = pool.acquire(128);
    ASSERT_NE(p1, nullptr);

    void *p2 = pool.acquire(256);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p2, p1) << "grow must allocate a new buffer";
    EXPECT_EQ(m.alloc_count, 2);
    EXPECT_EQ(m.free_count, 1);
    EXPECT_EQ(pool.size(), 256u);
}

TEST(DevicePoolTest, AcquireSmallerSizeKeepsExistingBuffer) {
    MockAllocator m;
    DevicePool pool = make_pool(&m);

    void *p_big = pool.acquire(256);
    ASSERT_NE(p_big, nullptr);
    EXPECT_EQ(m.alloc_count, 1);

    void *p_small = pool.acquire(64);
    EXPECT_EQ(p_small, p_big) << "shrink must not reallocate";
    EXPECT_EQ(m.alloc_count, 1);
    EXPECT_EQ(m.free_count, 0);
    EXPECT_EQ(pool.size(), 256u);
}

TEST(DevicePoolTest, AcquireZeroReturnsNullAndDoesNotAllocate) {
    MockAllocator m;
    DevicePool pool = make_pool(&m);

    void *p = pool.acquire(0);
    EXPECT_EQ(p, nullptr);
    EXPECT_EQ(m.alloc_count, 0);
    EXPECT_EQ(m.free_count, 0);
}

TEST(DevicePoolTest, ReleaseFreesBufferAndReacquireAllocates) {
    MockAllocator m;
    DevicePool pool = make_pool(&m);

    void *p1 = pool.acquire(128);
    ASSERT_NE(p1, nullptr);

    pool.release();
    EXPECT_EQ(m.free_count, 1);
    EXPECT_EQ(pool.ptr(), nullptr);
    EXPECT_EQ(pool.size(), 0u);

    // release() on an empty pool must be a no-op.
    pool.release();
    EXPECT_EQ(m.free_count, 1);

    void *p2 = pool.acquire(64);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(m.alloc_count, 2);
}

TEST(DevicePoolTest, DestructorReleasesOutstandingBuffer) {
    MockAllocator m;
    {
        DevicePool pool = make_pool(&m);
        pool.acquire(128);
        EXPECT_EQ(m.alloc_count, 1);
        EXPECT_EQ(m.free_count, 0);
    }
    EXPECT_EQ(m.free_count, 1) << "destructor must free the outstanding buffer";
    EXPECT_TRUE(m.live.empty());
}

}  // namespace
