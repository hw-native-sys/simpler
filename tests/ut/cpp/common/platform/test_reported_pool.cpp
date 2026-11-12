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

#include <cstdlib>
#include <cstddef>
#include <cstdint>

#include <new>

#include <gtest/gtest.h>

#include "host/memory_allocator.h"
#include "host/reported_pool.h"

namespace {
thread_local int allocation_countdown = -1;
thread_local bool allocation_failed = false;
}  // namespace

void *operator new(std::size_t bytes) {
    if (allocation_countdown == 0) {
        allocation_countdown = -1;
        allocation_failed = true;
        throw std::bad_alloc();
    }
    if (allocation_countdown > 0) --allocation_countdown;
    if (void *storage = std::malloc(bytes == 0 ? 1 : bytes)) return storage;
    throw std::bad_alloc();
}
void operator delete(void *ptr) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }

namespace {

TEST(ReportedPoolTest, AcquireReleaseReusesIdleRegion) {
    MemoryAllocator alloc;
    ReportedPool pool(alloc);
    ReportedPool::Lease first;
    ASSERT_EQ(pool.acquire(1, 0, 128, &first), 0);
    ASSERT_NE(first.device_ptr, nullptr);
    EXPECT_FALSE(first.initialized);
    EXPECT_EQ(pool.live_count(), 1u);

    pool.mark_initialized(first);
    ASSERT_EQ(pool.release(first, true, true), 0);
    EXPECT_EQ(pool.live_count(), 1u);
    EXPECT_EQ(pool.idle_device_ptr(), first.device_ptr);
    EXPECT_TRUE(pool.idle_region_initialized());

    ReportedPool::Lease second;
    ASSERT_EQ(pool.acquire(2, 0, 128, &second), 0);
    EXPECT_EQ(second.device_ptr, first.device_ptr);
    EXPECT_TRUE(second.initialized);
    ASSERT_EQ(pool.release(second, true, false), 0);
    EXPECT_EQ(pool.live_count(), 0u);
    EXPECT_EQ(alloc.committed_bytes(), 0u);
    EXPECT_EQ(pool.finalize(), 0);
}

TEST(ReportedPoolTest, ReleaseWithoutProofKeepsTheEpoch) {
    MemoryAllocator alloc;
    ReportedPool pool(alloc);
    ReportedPool::Lease lease;
    ASSERT_EQ(pool.acquire(7, 1, 64, &lease), 0);
    ASSERT_EQ(pool.release(lease, false, true), 0);
    EXPECT_EQ(pool.live_count(), 1u);
    EXPECT_EQ(pool.idle_device_ptr(), nullptr);

    ReportedPool::Lease again;
    ASSERT_EQ(pool.acquire(7, 1, 64, &again), 0);
    EXPECT_EQ(again.device_ptr, lease.device_ptr);
    ASSERT_EQ(pool.release(again, true, false), 0);
    EXPECT_EQ(pool.live_count(), 0u);
    EXPECT_EQ(pool.finalize(), 0);
}

TEST(ReportedPoolTest, FinalizeFreesActiveAndIdleRegions) {
    MemoryAllocator alloc;
    ReportedPool pool(alloc);
    ReportedPool::Lease one;
    ReportedPool::Lease two;
    ASSERT_EQ(pool.acquire(1, 0, 64, &one), 0);
    ASSERT_EQ(pool.acquire(2, 1, 64, &two), 0);
    ASSERT_NE(one.device_ptr, two.device_ptr);
    ASSERT_EQ(pool.release(one, true, true), 0);
    EXPECT_EQ(pool.live_count(), 2u);
    EXPECT_EQ(pool.finalize(), 0);
    EXPECT_EQ(pool.live_count(), 0u);
    EXPECT_EQ(alloc.committed_bytes(), 0u);
}

TEST(ReportedPoolTest, RejectsInvalidAcquireArguments) {
    MemoryAllocator alloc;
    ReportedPool pool(alloc);
    ReportedPool::Lease lease;
    EXPECT_NE(pool.acquire(0, 0, 64, &lease), 0);
    EXPECT_NE(pool.acquire(1, 0, 0, &lease), 0);
    EXPECT_NE(pool.acquire(1, 0, 64, nullptr), 0);
    EXPECT_EQ(alloc.committed_bytes(), 0u);
}

TEST(ReportedPoolTest, HostAllocationFailureKeepsEveryRegionOwned) {
    for (bool reuse_idle : {false, true}) {
        bool reached_success = false;
        for (int fail_at = 0; fail_at < 16; ++fail_at) {
            MemoryAllocator alloc;
            ReportedPool pool(alloc);
            ReportedPool::Lease old;
            if (reuse_idle) {
                ASSERT_EQ(pool.acquire(1, 0, 128, &old), 0);
                pool.mark_initialized(old);
                ASSERT_EQ(pool.release(old, true), 0);
            }
            ReportedPool::Lease result = old;
            allocation_failed = false;
            allocation_countdown = fail_at;
            int rc = PTO_RUNTIME_ERR_INTERNAL;
            try {
                rc = pool.acquire(2, 0, 128, &result);
            } catch (const std::bad_alloc &) {}
            allocation_countdown = -1;
            if (!allocation_failed) {
                ASSERT_EQ(rc, 0);
                ASSERT_EQ(pool.finalize(), 0);
                EXPECT_EQ(alloc.committed_bytes(), 0u);
                reached_success = true;
                break;
            }
            EXPECT_NE(rc, 0);
            EXPECT_EQ(result.run_epoch, old.run_epoch);
            EXPECT_EQ(result.device_ptr, old.device_ptr);
            EXPECT_EQ(pool.live_count(), reuse_idle ? 1u : 0u);
            EXPECT_EQ(pool.idle_device_ptr(), old.device_ptr);
            EXPECT_EQ(pool.idle_region_initialized(), reuse_idle);
            EXPECT_EQ(alloc.committed_bytes(), reuse_idle ? 128u : 0u);
            ASSERT_EQ(pool.acquire(2, 0, 128, &result), 0);
            if (reuse_idle) EXPECT_EQ(result.device_ptr, old.device_ptr);
            ASSERT_EQ(pool.finalize(), 0);
            EXPECT_EQ(alloc.committed_bytes(), 0u);
        }
        EXPECT_TRUE(reached_success);
    }
}

}  // namespace
