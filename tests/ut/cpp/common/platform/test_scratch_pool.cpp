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
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "host/scratch_pool.h"

namespace {

struct Allocator {
    size_t allocations{0};
    size_t frees{0};
    size_t fail_at{0};
};

void *fake_alloc(void *ctx, size_t bytes) {
    auto *allocator = static_cast<Allocator *>(ctx);
    if (allocator->fail_at != 0 && allocator->allocations + 1 == allocator->fail_at) return nullptr;
    ++allocator->allocations;
    return std::malloc(bytes);
}

void fake_free(void *ctx, void *ptr) {
    ++static_cast<Allocator *>(ctx)->frees;
    std::free(ptr);
}

TEST(ScratchPool, CapacityAndBaseRemainFixedUntilContextClose) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);
    ScratchPool::Lease first, second;
    ASSERT_EQ(pool.acquire(4096, &first), 0);
    ASSERT_EQ(pool.acquire(4096, &second), 0);
    EXPECT_EQ(first.base, second.base);
    pool.release(first);
    pool.release(second);
    ScratchPool::Lease refused;
    EXPECT_EQ(pool.acquire(8192, &refused), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(pool.acquire(2048, &refused), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_FALSE(refused.valid());
    EXPECT_EQ(pool.current_base(), first.base);
    EXPECT_EQ(allocator.allocations, 1u);
    EXPECT_EQ(allocator.frees, 0u);
    ASSERT_EQ(pool.acquire(4096, &second), 0);
    EXPECT_EQ(second.base, first.base);
    pool.release(second);
    pool.reset();
    EXPECT_EQ(allocator.frees, 1u);
}

TEST(ScratchPool, FailedInitialAllocationCanBeRetried) {
    Allocator allocator;
    allocator.fail_at = 1;
    ScratchPool pool(fake_alloc, fake_free, &allocator);
    ScratchPool::Lease lease;
    EXPECT_NE(pool.acquire(4096, &lease), 0);
    EXPECT_FALSE(lease.valid());
    EXPECT_EQ(pool.current_base(), nullptr);
    allocator.fail_at = 0;
    ASSERT_EQ(pool.acquire(4096, &lease), 0);
    pool.release(lease);
}

TEST(ScratchPool, ConcurrentLeasesShareOneAllocation) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
        threads.emplace_back([&] {
            for (int j = 0; j < 100; ++j) {
                ScratchPool::Lease lease;
                ASSERT_EQ(pool.acquire(4096, &lease), 0);
                pool.release(lease);
            }
        });
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(pool.live_leases(), 0u);
    EXPECT_EQ(allocator.allocations, 1u);
    EXPECT_EQ(allocator.frees, 0u);
}
}  // namespace
