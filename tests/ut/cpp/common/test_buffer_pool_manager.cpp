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
#include "host/profiler_base.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <thread>
#include <utility>
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
    static constexpr uint32_t kReadyQueueSize = 4;
};

struct SplitCapacityModule {
    using DataHeader = TestHeader;
    using ReadyEntry = TestReadyEntry;
    using ReadyBufferInfo = TestReadyBufferInfo;

    static constexpr int kBufferKinds = 1;
    static constexpr int kCollectorThreadCount = 1;
    static constexpr uint32_t kReadyQueueSize = 2;
    static constexpr uint32_t kHostPoolQueueSize = 5;
};

struct SplitDoneRecycledCapacityModule {
    using DataHeader = TestHeader;
    using ReadyEntry = TestReadyEntry;
    using ReadyBufferInfo = TestReadyBufferInfo;

    static constexpr int kBufferKinds = 1;
    static constexpr int kCollectorThreadCount = 1;
    static constexpr uint32_t kReadyQueueSize = 2;
    static constexpr uint32_t kHostPoolQueueSize = 5;
    static constexpr uint32_t kHostRecycledQueueSize = 3;
};

struct AlgorithmFreeQueue {
    volatile uint32_t head{0};
    volatile uint32_t tail{1};
    volatile uint64_t buffer_ptrs[1]{};
};

struct AlgorithmHeader {
    AlgorithmFreeQueue free_queue;
};

struct AlgorithmReadyEntry {
    uint64_t buffer_ptr{0};
};

struct AlgorithmReadyBufferInfo {
    void *dev_buffer_ptr{nullptr};
    void *host_buffer_ptr{nullptr};
};

struct AlgorithmModule {
    using DataHeader = AlgorithmHeader;
    using ReadyEntry = AlgorithmReadyEntry;
    using ReadyBufferInfo = AlgorithmReadyBufferInfo;
    using FreeQueue = AlgorithmFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr int kCollectorThreadCount = 1;
    static constexpr uint32_t kReadyQueueSize = 1;
    static constexpr uint32_t kHostPoolQueueSize = 4;
    static constexpr uint32_t kSlotCount = 1;
    static constexpr const char *kSubsystemName = "AlgorithmModule";

    static DataHeader *header_from_shm(void *shared_mem_host) { return static_cast<DataHeader *>(shared_mem_host); }

    static int batch_size(int /*kind*/) { return 1; }

    static std::optional<profiling_common::EntrySite<AlgorithmModule>>
    resolve_entry(void * /*shm_host*/, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        return profiling_common::EntrySite<AlgorithmModule>{
            0,
            &header->free_queue,
            sizeof(uint64_t),
            AlgorithmReadyBufferInfo{reinterpret_cast<void *>(entry.buffer_ptr), nullptr},
        };
    }

    template <typename Cb>
    static void for_each_instance(void * /*shm_host*/, DataHeader *header, Cb &&cb) {
        cb(0, &header->free_queue, sizeof(uint64_t));
    }
};

struct WarmRecycledModule {
    using DataHeader = AlgorithmHeader;
    using ReadyEntry = AlgorithmReadyEntry;
    using ReadyBufferInfo = AlgorithmReadyBufferInfo;
    using FreeQueue = AlgorithmFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr int kCollectorThreadCount = 2;
    static constexpr uint32_t kReadyQueueSize = 1;
    static constexpr uint32_t kHostPoolQueueSize = 8;
    static constexpr uint32_t kHostRecycledQueueSize = 3;
    static constexpr uint32_t kSlotCount = 1;
    static constexpr const char *kSubsystemName = "WarmRecycledModule";

    static DataHeader *header_from_shm(void *shared_mem_host) { return static_cast<DataHeader *>(shared_mem_host); }

    static int batch_size(int /*kind*/) { return 1; }

    static int recycled_warm_target(int /*kind*/, int shard) { return shard + 1; }

    static std::optional<profiling_common::EntrySite<WarmRecycledModule>>
    resolve_entry(void * /*shm_host*/, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        return profiling_common::EntrySite<WarmRecycledModule>{
            0,
            &header->free_queue,
            sizeof(uint64_t),
            AlgorithmReadyBufferInfo{reinterpret_cast<void *>(entry.buffer_ptr), nullptr},
        };
    }

    template <typename Cb>
    static void for_each_instance(void * /*shm_host*/, DataHeader *header, Cb &&cb) {
        cb(0, &header->free_queue, sizeof(uint64_t));
    }
};

struct WarmBatchModule {
    using DataHeader = AlgorithmHeader;
    using ReadyEntry = AlgorithmReadyEntry;
    using ReadyBufferInfo = AlgorithmReadyBufferInfo;
    using FreeQueue = AlgorithmFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr int kCollectorThreadCount = 1;
    static constexpr uint32_t kReadyQueueSize = 1;
    static constexpr uint32_t kHostPoolQueueSize = 16;
    static constexpr uint32_t kHostRecycledQueueSize = 16;
    static constexpr uint32_t kSlotCount = 4;
    static constexpr const char *kSubsystemName = "WarmBatchModule";

    static DataHeader *header_from_shm(void *shared_mem_host) { return static_cast<DataHeader *>(shared_mem_host); }

    static int batch_size(int /*kind*/) { return 1; }

    static int recycled_warm_target(int /*kind*/, int /*shard*/) { return 6; }

    static std::optional<profiling_common::EntrySite<WarmBatchModule>>
    resolve_entry(void * /*shm_host*/, DataHeader *header, int /*q*/, const ReadyEntry &entry) {
        return profiling_common::EntrySite<WarmBatchModule>{
            0,
            &header->free_queue,
            sizeof(uint64_t),
            AlgorithmReadyBufferInfo{reinterpret_cast<void *>(entry.buffer_ptr), nullptr},
        };
    }

    template <typename Cb>
    static void for_each_instance(void * /*shm_host*/, DataHeader *header, Cb &&cb) {
        cb(0, &header->free_queue, sizeof(uint64_t));
    }
};

void *ptr(uintptr_t value) { return reinterpret_cast<void *>(value); }

}  // namespace

TEST(SpscRingTest, ConcurrentProducerConsumerPreservesFifo) {
    profiling_common::SpscRing<uint64_t, 1024> ring;
    constexpr uint64_t kItems = 100000;
    std::atomic<bool> producer_done{false};

    std::thread producer([&]() {
        for (uint64_t i = 0; i < kItems; i++) {
            while (!ring.push(i)) {
                std::this_thread::yield();
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    uint64_t expected = 0;
    while (expected < kItems) {
        uint64_t value = 0;
        if (ring.pop(value)) {
            ASSERT_EQ(value, expected);
            expected++;
            continue;
        }
        ASSERT_FALSE(producer_done.load(std::memory_order_acquire) && ring.empty());
        std::this_thread::yield();
    }

    producer.join();
    EXPECT_TRUE(ring.empty());
}

TEST(BufferPoolManagerShardingTest, ReadyShardsAreIndependent) {
    using Manager = profiling_common::BufferPoolManager<TestModule>;
    static_assert(Manager::kCollectorShardCount == 4);

    Manager manager;
    ASSERT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x1000), 0}, 0));
    ASSERT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x2000), 1}, 1));
    ASSERT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x5000), 5}, 5));  // normalizes to shard 1

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

TEST(BufferPoolManagerShardingTest, ReadyShardReportsFullAndPreservesOrder) {
    using Manager = profiling_common::BufferPoolManager<TestModule>;

    Manager manager;
    for (uintptr_t i = 0; i < Manager::kHostQueueCapacity; i++) {
        ASSERT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x1000 + i), static_cast<uint32_t>(i)}, 0));
    }
    EXPECT_FALSE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x9999), 99}, 0));

    TestReadyBufferInfo out;
    for (uintptr_t i = 0; i < Manager::kHostQueueCapacity; i++) {
        ASSERT_TRUE(manager.try_pop_ready(out, 0));
        EXPECT_EQ(out.dev_buffer_ptr, ptr(0x1000 + i));
        EXPECT_EQ(out.shard_marker, static_cast<uint32_t>(i));
    }
    EXPECT_FALSE(manager.try_pop_ready(out, 0));
}

TEST(BufferPoolManagerShardingTest, WaitPopReadyWakesOnProducer) {
    using namespace std::chrono_literals;
    profiling_common::BufferPoolManager<TestModule> manager;

    std::thread producer([&]() {
        std::this_thread::sleep_for(20ms);
        EXPECT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x7000), 7}, 2));
    });

    TestReadyBufferInfo out;
    ASSERT_TRUE(manager.wait_pop_ready(out, 500ms, 2));
    EXPECT_EQ(out.dev_buffer_ptr, ptr(0x7000));
    EXPECT_EQ(out.shard_marker, 7u);
    producer.join();
}

TEST(BufferPoolManagerShardingTest, DoneShardsRecycleByKind) {
    profiling_common::BufferPoolManager<TestModule> manager;

    manager.notify_copy_done(ptr(0x1000), /*kind=*/0, /*shard_index=*/0);
    manager.notify_copy_done(ptr(0x2000), /*kind=*/1, /*shard_index=*/1);
    manager.notify_copy_done(ptr(0x5000), /*kind=*/1, /*shard_index=*/5);  // normalizes to shard 1

    EXPECT_EQ(manager.drain_done_into_recycled(), 3u);
    EXPECT_EQ(manager.recycled_count(0), 1u);
    EXPECT_EQ(manager.recycled_count(1), 2u);
    EXPECT_EQ(manager.recycled_count(1, 1), 2u);

    EXPECT_EQ(manager.pop_recycled(0), ptr(0x1000));

    std::set<void *> kind_one;
    kind_one.insert(manager.pop_recycled(1, 1));
    kind_one.insert(manager.pop_recycled(1, 1));
    EXPECT_EQ(kind_one, (std::set<void *>{ptr(0x2000), ptr(0x5000)}));
}

TEST(BufferPoolManagerShardingTest, DrainDoneCanTargetOneShard) {
    profiling_common::BufferPoolManager<TestModule> manager;

    ASSERT_TRUE(manager.notify_copy_done(ptr(0x1000), /*kind=*/0, /*shard_index=*/0));
    ASSERT_TRUE(manager.notify_copy_done(ptr(0x2000), /*kind=*/0, /*shard_index=*/1));

    EXPECT_EQ(manager.drain_done_into_recycled(/*shard_index=*/1), 1u);
    EXPECT_EQ(manager.recycled_count(0, 0), 0u);
    EXPECT_EQ(manager.recycled_count(0, 1), 1u);
    EXPECT_EQ(manager.pop_recycled(0, 1), ptr(0x2000));

    EXPECT_EQ(manager.drain_done_into_recycled(/*shard_index=*/0), 1u);
    EXPECT_EQ(manager.pop_recycled(0, 0), ptr(0x1000));
}

TEST(BufferPoolManagerShardingTest, DoneShardCarriesKindAndDevicePointer) {
    profiling_common::BufferPoolManager<TestModule> manager;

    ASSERT_TRUE(manager.notify_copy_done(ptr(0x1000), /*kind=*/1, /*shard_index=*/2));

    profiling_common::DoneInfo out{};
    ASSERT_TRUE(manager.try_pop_done(out, /*shard_index=*/2));
    EXPECT_EQ(out.dev_ptr, ptr(0x1000));
    EXPECT_EQ(out.kind, 1);
    EXPECT_FALSE(manager.try_pop_done(out, /*shard_index=*/2));
}

TEST(BufferPoolManagerShardingTest, StartupRecyclePopCanUseAnyShard) {
    profiling_common::BufferPoolManager<TestModule> manager;

    ASSERT_TRUE(manager.push_recycled(/*kind=*/0, ptr(0x1000), /*shard_index=*/2));
    ASSERT_TRUE(manager.push_recycled(/*kind=*/0, ptr(0x2000), /*shard_index=*/3));

    std::set<void *> popped;
    popped.insert(manager.pop_recycled_for_startup(/*kind=*/0));
    popped.insert(manager.pop_recycled_for_startup(/*kind=*/0));
    EXPECT_EQ(popped, (std::set<void *>{ptr(0x1000), ptr(0x2000)}));
    EXPECT_EQ(manager.pop_recycled_for_startup(/*kind=*/0), nullptr);
}

TEST(BufferPoolManagerShardingTest, PoolQueuesUseCapacityIndependentFromReadyQueue) {
    using Manager = profiling_common::BufferPoolManager<SplitCapacityModule>;

    Manager manager;
    EXPECT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x1000), 0}, 0));
    EXPECT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x1001), 1}, 0));
    EXPECT_FALSE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x1002), 2}, 0));

    for (uintptr_t i = 0; i < SplitCapacityModule::kHostPoolQueueSize; i++) {
        EXPECT_TRUE(manager.push_recycled(/*kind=*/0, ptr(0x2000 + i), 0)) << i;
    }
    EXPECT_FALSE(manager.push_recycled(/*kind=*/0, ptr(0x2999), 0));

    profiling_common::BufferPoolManager<SplitCapacityModule> done_manager;
    for (uintptr_t i = 0; i < SplitCapacityModule::kHostPoolQueueSize; i++) {
        EXPECT_TRUE(done_manager.notify_copy_done(ptr(0x3000 + i), /*kind=*/0, 0)) << i;
    }
    EXPECT_FALSE(done_manager.notify_copy_done(ptr(0x3999), /*kind=*/0, 0));
}

TEST(BufferPoolManagerShardingTest, DoneAndRecycledQueuesCanUseDifferentCapacities) {
    using Manager = profiling_common::BufferPoolManager<SplitDoneRecycledCapacityModule>;

    Manager manager;
    for (uintptr_t i = 0; i < SplitDoneRecycledCapacityModule::kHostPoolQueueSize; i++) {
        EXPECT_TRUE(manager.notify_copy_done(ptr(0x3000 + i), /*kind=*/0, 0)) << i;
    }
    EXPECT_FALSE(manager.notify_copy_done(ptr(0x3999), /*kind=*/0, 0));

    for (uintptr_t i = 0; i < SplitDoneRecycledCapacityModule::kHostRecycledQueueSize; i++) {
        EXPECT_TRUE(manager.push_recycled(/*kind=*/0, ptr(0x4000 + i), 0)) << i;
    }
    EXPECT_FALSE(manager.push_recycled(/*kind=*/0, ptr(0x4999), 0));
}

TEST(BufferPoolManagerShardingTest, ReplenishRecycledPoolsAllocatesToRuntimeWatermarks) {
    using Manager = profiling_common::BufferPoolManager<WarmRecycledModule>;

    Manager manager;
    AlgorithmHeader header{};
    profiling_common::MemoryOps ops;
    ops.alloc = [](size_t size) {
        return std::malloc(size);
    };
    ops.reg = [](void *dev_ptr, size_t /*size*/, int /*device_id*/, void **host_ptr_out) {
        *host_ptr_out = dev_ptr;
        return 0;
    };
    ops.free_ = [](void *dev_ptr) {
        std::free(dev_ptr);
        return 0;
    };
    manager.set_memory_context(std::move(ops), nullptr, &header, sizeof(header), 0);

    EXPECT_EQ(profiling_common::ProfilerAlgorithms<WarmRecycledModule>::replenish_recycled_pools(manager, &header), 3u);
    EXPECT_EQ(manager.recycled_count(0, 0), 1u);
    EXPECT_EQ(manager.recycled_count(0, 1), 2u);
    EXPECT_EQ(profiling_common::ProfilerAlgorithms<WarmRecycledModule>::replenish_recycled_pools(manager, &header), 0u);

    manager.release_owned_buffers([](void *p) {
        std::free(p);
    });
    manager.clear_mappings();
}

TEST(BufferPoolManagerShardingTest, ReplenishRecycledPoolsUsesSlotSizedBatchForSmallGap) {
    using Manager = profiling_common::BufferPoolManager<WarmBatchModule>;

    Manager manager;
    AlgorithmHeader header{};
    int alloc_calls = 0;
    profiling_common::MemoryOps ops;
    ops.alloc = [&](size_t size) {
        alloc_calls++;
        return std::malloc(size);
    };
    ops.reg = [](void *dev_ptr, size_t /*size*/, int /*device_id*/, void **host_ptr_out) {
        *host_ptr_out = dev_ptr;
        return 0;
    };
    ops.free_ = [](void *dev_ptr) {
        std::free(dev_ptr);
        return 0;
    };
    manager.set_memory_context(std::move(ops), nullptr, &header, sizeof(header), 0);

    EXPECT_EQ(profiling_common::ProfilerAlgorithms<WarmBatchModule>::replenish_recycled_pools(manager, &header), 6u);
    EXPECT_EQ(manager.recycled_count(0, 0), 6u);
    EXPECT_EQ(alloc_calls, 1);

    EXPECT_NE(manager.pop_recycled(0, 0), nullptr);
    EXPECT_NE(manager.pop_recycled(0, 0), nullptr);
    EXPECT_EQ(manager.recycled_count(0, 0), 4u);

    EXPECT_EQ(profiling_common::ProfilerAlgorithms<WarmBatchModule>::replenish_recycled_pools(manager, &header), 4u);
    EXPECT_EQ(manager.recycled_count(0, 0), 8u);
    EXPECT_EQ(alloc_calls, 2);

    manager.release_owned_buffers([](void *p) {
        std::free(p);
    });
    manager.clear_mappings();
}

TEST(BufferPoolManagerShardingTest, ProcessEntryReadyFullDoesNotPublishDoneFromDrainThread) {
    using Manager = profiling_common::BufferPoolManager<AlgorithmModule>;

    Manager manager;
    AlgorithmHeader header{};
    void *dev_ptr = ptr(0x7000);
    manager.register_mapping(dev_ptr, dev_ptr);

    ASSERT_TRUE(manager.push_to_ready(AlgorithmReadyBufferInfo{ptr(0x1111), ptr(0x1111)}, 0));

    profiling_common::ProfilerAlgorithms<AlgorithmModule>::process_entry(
        manager, &header, 0, AlgorithmReadyEntry{reinterpret_cast<uint64_t>(dev_ptr)}
    );

    profiling_common::DoneInfo done{};
    EXPECT_FALSE(manager.try_pop_done(done, 0));

    AlgorithmReadyBufferInfo ready{};
    ASSERT_TRUE(manager.try_pop_ready(ready, 0));
    EXPECT_EQ(ready.dev_buffer_ptr, ptr(0x1111));
    EXPECT_FALSE(manager.try_pop_ready(ready, 0));

    std::vector<void *> released;
    manager.release_owned_buffers([&](void *p) {
        released.push_back(p);
    });
    EXPECT_EQ(released, (std::vector<void *>{dev_ptr}));
}

TEST(BufferPoolManagerShardingTest, BlockBatchCarvesRangeMappingsAndReleasesBaseOnce) {
    using Manager = profiling_common::BufferPoolManager<TestModule>;

    Manager manager;
    profiling_common::MemoryOps ops;
    ops.alloc = [](size_t size) {
        return std::malloc(size);
    };
    ops.reg = [](void *dev_ptr, size_t /*size*/, int /*device_id*/, void **host_ptr_out) {
        *host_ptr_out = dev_ptr;
        return 0;
    };
    ops.free_ = [](void *dev_ptr) {
        std::free(dev_ptr);
        return 0;
    };
    manager.set_memory_context(std::move(ops), nullptr, nullptr, 0, 0);

    ASSERT_EQ(manager.allocate_recycled_batch(/*kind=*/0, /*buffer_size=*/32, /*count=*/3, /*shard_index=*/1), 3u);

    std::vector<void *> buffers;
    for (int i = 0; i < 3; i++) {
        void *p = manager.pop_recycled(/*kind=*/0, /*shard_index=*/1);
        ASSERT_NE(p, nullptr);
        buffers.push_back(p);
    }
    EXPECT_EQ(manager.pop_recycled(/*kind=*/0, /*shard_index=*/1), nullptr);

    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[1]) - reinterpret_cast<uintptr_t>(buffers[0]), 64u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffers[2]) - reinterpret_cast<uintptr_t>(buffers[1]), 64u);
    EXPECT_EQ(manager.resolve_host_ptr(buffers[0]), buffers[0]);

    void *inner_dev = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffers[1]) + 7);
    EXPECT_EQ(manager.resolve_host_ptr(inner_dev), inner_dev);

    for (void *p : buffers) {
        ASSERT_TRUE(manager.push_recycled(/*kind=*/0, p, /*shard_index=*/1));
    }

    std::vector<void *> released;
    manager.release_owned_buffers([&](void *p) {
        released.push_back(p);
        std::free(p);
    });
    ASSERT_EQ(released.size(), 1u);
    EXPECT_EQ(released[0], buffers[0]);
    manager.clear_mappings();
}

TEST(BufferPoolManagerShardingTest, FreeBufferAllowsAllocationAddressReuse) {
    using Manager = profiling_common::BufferPoolManager<TestModule>;

    Manager manager;
    std::vector<void *> released;
    profiling_common::MemoryOps ops;
    ops.free_ = [&](void *dev_ptr) {
        released.push_back(dev_ptr);
        return 0;
    };
    manager.set_memory_context(std::move(ops), nullptr, nullptr, 0, 0);

    void *dev_ptr = ptr(0x7000);
    manager.register_mapping(dev_ptr, nullptr);
    manager.free_buffer(dev_ptr);

    manager.register_mapping(dev_ptr, nullptr);
    manager.free_buffer(dev_ptr);

    EXPECT_EQ(released, (std::vector<void *>{dev_ptr, dev_ptr}));
}

TEST(BufferPoolManagerShardingTest, ReleaseOwnedBuffersVisitsAllShards) {
    profiling_common::BufferPoolManager<TestModule> manager;
    ASSERT_TRUE(manager.push_recycled(/*kind=*/0, ptr(0x1000)));
    ASSERT_TRUE(manager.push_to_ready(TestReadyBufferInfo{ptr(0x2000), 2}, /*shard_index=*/2));
    ASSERT_TRUE(manager.notify_copy_done(ptr(0x3000), /*kind=*/1, /*shard_index=*/3));
    ASSERT_TRUE(manager.retire_unqueued_buffer(/*kind=*/1, ptr(0x4000), /*shard_index=*/2));

    std::vector<void *> released;
    manager.release_owned_buffers([&](void *p) {
        released.push_back(p);
    });

    EXPECT_EQ(
        std::set<void *>(released.begin(), released.end()),
        (std::set<void *>{ptr(0x1000), ptr(0x2000), ptr(0x3000), ptr(0x4000)})
    );
    EXPECT_TRUE(manager.recycled_empty());

    TestReadyBufferInfo out;
    EXPECT_FALSE(manager.try_pop_ready(out, 2));
    EXPECT_EQ(manager.drain_done_into_recycled(), 0u);
}
