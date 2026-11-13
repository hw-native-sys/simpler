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

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

#include <new>

#include <gtest/gtest.h>

#include "host/scratch_pool.h"

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

TEST(ScratchPool, ReusesCurrentGenerationWhileItFits) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);

    ScratchPool::Lease first;
    ScratchPool::Lease second;
    ASSERT_EQ(pool.acquire(128, &first), 0);
    ASSERT_EQ(pool.acquire(64, &second), 0);
    EXPECT_EQ(first.generation, second.generation);
    EXPECT_EQ(first.base, second.base);
    EXPECT_EQ(allocator.allocations, 1u);
    EXPECT_EQ(pool.live_generations(), 1u);
    EXPECT_EQ(pool.live_leases(), 2u);

    pool.release(first);
    pool.release(second);
    EXPECT_EQ(allocator.frees, 0u);
    EXPECT_EQ(pool.live_generations(), 1u);
    EXPECT_EQ(pool.live_leases(), 0u);
}

TEST(ScratchPool, GrowthKeepsTheOldGenerationUntilItsLeaseEnds) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);

    ScratchPool::Lease small;
    ScratchPool::Lease large;
    ASSERT_EQ(pool.acquire(128, &small), 0);
    ASSERT_EQ(pool.acquire(256, &large), 0);
    EXPECT_NE(small.generation, large.generation);
    EXPECT_EQ(allocator.allocations, 2u);
    EXPECT_EQ(pool.live_generations(), 2u);

    pool.release(small);
    EXPECT_EQ(allocator.frees, 1u);
    EXPECT_EQ(pool.live_generations(), 1u);
    pool.release(large);
    EXPECT_EQ(allocator.frees, 1u);  // the current generation stays reusable
    EXPECT_EQ(pool.live_leases(), 0u);
}

TEST(ScratchPool, ALeaseOnARetiredGenerationKeepsItAlive) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);

    ScratchPool::Lease old;
    ASSERT_EQ(pool.acquire(128, &old), 0);
    ScratchPool::Lease current;
    ASSERT_EQ(pool.acquire(512, &current), 0);
    EXPECT_EQ(pool.live_generations(), 2u);

    // The old generation is retired but still leased, so a later grow neither
    // frees it nor hands its bytes to a new run.
    ScratchPool::Lease next;
    ASSERT_EQ(pool.acquire(1024, &next), 0);
    EXPECT_EQ(allocator.allocations, 3u);
    EXPECT_EQ(pool.live_generations(), 3u);
    EXPECT_EQ(allocator.frees, 0u);
}

TEST(ScratchPool, AcquireFailureLeavesExistingLeasesUsable) {
    Allocator allocator;
    allocator.fail_at = 2;
    ScratchPool pool(fake_alloc, fake_free, &allocator);

    ScratchPool::Lease first;
    ASSERT_EQ(pool.acquire(128, &first), 0);
    ScratchPool::Lease too_large;
    EXPECT_NE(pool.acquire(256, &too_large), 0);
    EXPECT_EQ(pool.live_generations(), 1u);
    EXPECT_EQ(pool.live_leases(), 1u);

    ScratchPool::Lease second;
    ASSERT_EQ(pool.acquire(64, &second), 0);
    EXPECT_EQ(first.generation, second.generation);
    pool.release(first);
    pool.release(second);
}

TEST(ScratchPool, InvalidLeasesAreIgnored) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);

    pool.release({});
    pool.release(ScratchPool::Lease{99, nullptr, 0});
    EXPECT_EQ(pool.live_generations(), 0u);
    EXPECT_EQ(pool.live_leases(), 0u);
    EXPECT_EQ(allocator.frees, 0u);
}

TEST(ScratchPool, ResetFreesEveryGeneration) {
    Allocator allocator;
    {
        ScratchPool pool(fake_alloc, fake_free, &allocator);
        ScratchPool::Lease first;
        ScratchPool::Lease second;
        ASSERT_EQ(pool.acquire(128, &first), 0);
        ASSERT_EQ(pool.acquire(256, &second), 0);
        EXPECT_EQ(allocator.allocations, 2u);
    }
    EXPECT_EQ(allocator.frees, 2u);
}

TEST(ScratchPool, ConcurrentAcquireAndReleaseShareOneGeneration) {
    Allocator allocator;
    ScratchPool pool(fake_alloc, fake_free, &allocator);
    constexpr int kThreads = 16;
    constexpr int kCalls = 128;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kCalls; ++i) {
                ScratchPool::Lease lease;
                if (pool.acquire(128, &lease) != 0) {
                    ++failures;
                    return;
                }
                pool.release(lease);
            }
        });
    }
    for (auto &thread : threads)
        thread.join();

    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(allocator.allocations, 1u);
    EXPECT_EQ(pool.live_generations(), 1u);
    EXPECT_EQ(pool.live_leases(), 0u);
}

TEST(ScratchPool, GrowthReclaimsIdleGenerationsBeforeReturning) {
    Allocator allocator;
    size_t retirements = 0;
    ScratchPool pool(fake_alloc, fake_free, &allocator, [&](uint64_t, void *) {
        EXPECT_EQ(retirements, allocator.frees);
        ++retirements;
    });
    for (size_t bytes : {128, 256, 512}) {
        ScratchPool::Lease lease;
        ASSERT_EQ(pool.acquire(bytes, &lease), 0);
        EXPECT_EQ(pool.live_generations(), 1u);
        EXPECT_EQ(allocator.frees, allocator.allocations - 1);
        EXPECT_EQ(retirements, allocator.frees);
        pool.release(lease);
    }
    pool.reset();
    EXPECT_EQ(allocator.frees, 3u);
    EXPECT_EQ(retirements, 3u);
}

TEST(ScratchPool, HostAllocationFailurePreservesPublicationAndAllowsRetry) {
    for (bool grow : {false, true}) {
        bool reached_success = false;
        for (int fail_at = 0; fail_at < 16; ++fail_at) {
            Allocator allocator;
            ScratchPool pool(fake_alloc, fake_free, &allocator);
            ScratchPool::Lease old;
            if (grow) ASSERT_EQ(pool.acquire(128, &old), 0);
            ScratchPool::Lease result = old;
            allocation_failed = false;
            allocation_countdown = fail_at;
            int rc = PTO_RUNTIME_ERR_INTERNAL;
            try {
                rc = pool.acquire(256, &result);
            } catch (const std::bad_alloc &) {}
            allocation_countdown = -1;
            if (!allocation_failed) {
                ASSERT_EQ(rc, 0);
                reached_success = true;
                break;
            }
            EXPECT_NE(rc, 0);
            EXPECT_EQ(result.generation, old.generation);
            EXPECT_EQ(result.base, old.base);
            EXPECT_EQ(pool.live_generations(), grow ? 1u : 0u);
            EXPECT_EQ(pool.live_leases(), grow ? 1u : 0u);
            EXPECT_EQ(pool.current_base(), old.base);
            EXPECT_EQ(pool.covers(128), grow);
            EXPECT_FALSE(pool.covers(256));
            EXPECT_EQ(allocator.allocations - allocator.frees, grow ? 1u : 0u);
            ASSERT_EQ(pool.acquire(256, &result), 0);
            pool.release(old);
            pool.release(result);
            pool.reset();
            EXPECT_EQ(allocator.allocations, allocator.frees);
        }
        EXPECT_TRUE(reached_success);
    }
}

}  // namespace
