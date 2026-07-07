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

#include "host/buffer_pool_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

namespace {

struct TestHeader {};
struct TestReadyEntry {};

struct TestReadyBufferInfo {
    void *dev_buffer_ptr{nullptr};
    uint32_t shard_marker{0};
};

struct TestModule {
    using DataHeader = TestHeader;
    using ReadyEntry = TestReadyEntry;
    using ReadyBufferInfo = TestReadyBufferInfo;

    static constexpr int kBufferKinds = 2;
    static constexpr int kCollectorThreadCount = 4;
};

void *ptr(uintptr_t value) { return reinterpret_cast<void *>(value); }

}  // namespace

TEST(BufferPoolManagerShardingTest, ReadyShardsAreIndependent) {
    using Manager = profiling_common::BufferPoolManager<TestModule>;
    static_assert(Manager::kCollectorShardCount == 4);

    Manager manager;
    manager.push_to_ready(TestReadyBufferInfo{ptr(0x1000), 0}, 0);
    manager.push_to_ready(TestReadyBufferInfo{ptr(0x2000), 1}, 1);
    manager.push_to_ready(TestReadyBufferInfo{ptr(0x5000), 5}, 5);  // normalizes to shard 1

    TestReadyBufferInfo out;
    EXPECT_FALSE(manager.try_pop_ready(out, 2));

    ASSERT_TRUE(manager.try_pop_ready(out, 0));
    EXPECT_EQ(out.dev_buffer_ptr, ptr(0x1000));
    EXPECT_EQ(out.shard_marker, 0u);
    EXPECT_FALSE(manager.try_pop_ready(out, 0));

    ASSERT_TRUE(manager.try_pop_ready(out, 1));
    EXPECT_EQ(out.dev_buffer_ptr, ptr(0x2000));
    EXPECT_EQ(out.shard_marker, 1u);
    ASSERT_TRUE(manager.try_pop_ready(out, 1));
    EXPECT_EQ(out.dev_buffer_ptr, ptr(0x5000));
    EXPECT_EQ(out.shard_marker, 5u);
    EXPECT_FALSE(manager.try_pop_ready(out, 1));
}

TEST(BufferPoolManagerShardingTest, DoneShardsRecycleByKind) {
    profiling_common::BufferPoolManager<TestModule> manager;

    manager.notify_copy_done(ptr(0x1000), /*kind=*/0, /*shard_index=*/0);
    manager.notify_copy_done(ptr(0x2000), /*kind=*/1, /*shard_index=*/1);
    manager.notify_copy_done(ptr(0x5000), /*kind=*/1, /*shard_index=*/5);  // normalizes to shard 1

    EXPECT_EQ(manager.drain_done_into_recycled(), 3u);
    EXPECT_EQ(manager.recycled_count(0), 1u);
    EXPECT_EQ(manager.recycled_count(1), 2u);

    EXPECT_EQ(manager.pop_recycled(0), ptr(0x1000));

    std::set<void *> kind_one;
    kind_one.insert(manager.pop_recycled(1, 1));
    kind_one.insert(manager.pop_recycled(1, 1));
    EXPECT_EQ(kind_one, (std::set<void *>{ptr(0x2000), ptr(0x5000)}));
}

TEST(BufferPoolManagerShardingTest, ReleaseOwnedBuffersVisitsAllShards) {
    profiling_common::BufferPoolManager<TestModule> manager;
    manager.push_recycled(/*kind=*/0, ptr(0x1000));
    manager.push_to_ready(TestReadyBufferInfo{ptr(0x2000), 2}, /*shard_index=*/2);
    manager.notify_copy_done(ptr(0x3000), /*kind=*/1, /*shard_index=*/3);

    std::vector<void *> released;
    manager.release_owned_buffers([&](void *p) {
        released.push_back(p);
    });

    EXPECT_EQ(
        std::set<void *>(released.begin(), released.end()), (std::set<void *>{ptr(0x1000), ptr(0x2000), ptr(0x3000)})
    );
    EXPECT_TRUE(manager.recycled_empty());

    TestReadyBufferInfo out;
    EXPECT_FALSE(manager.try_pop_ready(out, 2));
    EXPECT_EQ(manager.drain_done_into_recycled(), 0u);
}
