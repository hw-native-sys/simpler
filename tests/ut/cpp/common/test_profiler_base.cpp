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
 * ProfilerBase start/stop and thread-fanout tests.
 *
 * The framework runs `min(aicpu_thread_num, Module::kMaxCollectorThreads)`
 * drain+collector threads while scanning `aicpu_thread_num` device ready
 * queues. Those two counts are equal for the scheduler-fed subsystems
 * (L2Swimlane / TensorDump / PMU) but differ for the orchestrator-only ones
 * (DepGen / ScopeStats), whose single producer writes the LAST queue while
 * only one shard exists. Both shapes are covered here.
 */

#include "common/dfx_backpressure_device.h"
#include "host/profiler_base.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kReadyQueueSize = 8;
constexpr uint32_t kSlotCount = 4;

struct TestFreeQueue {
    volatile uint32_t head{0};
    volatile uint32_t tail{0};
    volatile uint64_t buffer_ptrs[kSlotCount]{};
};

struct TestReadyEntry {
    uint64_t buffer_ptr{0};
    uint32_t buffer_seq{0};
};

// Mirrors the real subsystems: the ready-queue array is dimensioned by the
// PLATFORM max (the device shm layout stride is fixed there), while only the
// first `aicpu_thread_num` rows ever get a producer.
struct TestHeader {
    TestReadyEntry queues[PLATFORM_MAX_AICPU_THREADS][kReadyQueueSize];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];
    TestFreeQueue free_queue;
    DfxBackpressureHeader backpressure;
};

struct TestReadyBufferInfo {
    void *dev_buffer_ptr{nullptr};
    void *host_buffer_ptr{nullptr};
    uint32_t producer_queue{0};
};

// Base traits; the two Module flavours below differ only in their shard cap.
template <int kMaxThreads>
struct TestModuleBase {
    using DataHeader = TestHeader;
    using ReadyEntry = TestReadyEntry;
    using ReadyBufferInfo = TestReadyBufferInfo;
    using FreeQueue = TestFreeQueue;

    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = ::kReadyQueueSize;
    static constexpr uint32_t kHostPoolQueueSize = 64;
    static constexpr uint32_t kSlotCount = ::kSlotCount;
    static constexpr const char *kSubsystemName = "TestModule";
    static constexpr int kMaxCollectorThreads = kMaxThreads;

    static DataHeader *header_from_shm(void *shm) { return static_cast<DataHeader *>(shm); }

    static int batch_size(int /*kind*/) { return 1; }

    static std::optional<profiling_common::EntrySite<TestModuleBase>>
    resolve_entry(void * /*shm*/, DataHeader *header, int q, const ReadyEntry &entry) {
        return profiling_common::EntrySite<TestModuleBase>{
            0,
            &header->free_queue,
            sizeof(uint64_t),
            TestReadyBufferInfo{reinterpret_cast<void *>(entry.buffer_ptr), nullptr, static_cast<uint32_t>(q)},
        };
    }

    template <typename Cb>
    static void for_each_instance(void * /*shm*/, DataHeader *header, Cb &&cb) {
        cb(0, &header->free_queue, sizeof(uint64_t));
    }
};

// Scheduler-fed shape: one shard per AICPU thread.
using PerThreadModule = TestModuleBase<PLATFORM_MAX_AICPU_THREADS>;
// Orchestrator-only shape (DepGen / ScopeStats): a single shard no matter how
// many AICPU threads run.
using SingleShardModule = TestModuleBase<1>;

template <typename Module>
class TestCollector : public profiling_common::ProfilerBase<TestCollector<Module>, Module> {
public:
    using Base = profiling_common::ProfilerBase<TestCollector<Module>, Module>;
    using ReadyBufferInfo = typename Module::ReadyBufferInfo;

    // Deliberately short: a false-positive idle timeout must fail the test fast
    // rather than stall it.
    static constexpr int kIdleTimeoutSec = 2;
    static constexpr const char *kSubsystemName = "TestCollector";

    void on_buffer_collected(const ReadyBufferInfo &info, int collector_shard) {
        collected_.fetch_add(1, std::memory_order_relaxed);
        last_producer_queue_.store(info.producer_queue, std::memory_order_relaxed);
        last_shard_.store(collector_shard, std::memory_order_relaxed);
    }

    int collected() const { return collected_.load(std::memory_order_relaxed); }
    uint32_t last_producer_queue() const { return last_producer_queue_.load(std::memory_order_relaxed); }
    int last_shard() const { return last_shard_.load(std::memory_order_relaxed); }

    // Stand-in for a real Derived::init(): latch the thread count and hand the
    // base an identity-mapped (SVM-style) memory context.
    void init(int aicpu_thread_num, void *shm) {
        this->set_aicpu_thread_num(aicpu_thread_num);
        this->set_memory_context(
            [](size_t size) {
                return std::malloc(size);
            },
            /*register_cb=*/nullptr, /*free_cb=*/nullptr, /*copy_to_device=*/nullptr, /*copy_from_device=*/nullptr, shm,
            shm, sizeof(TestHeader), /*device_id=*/0
        );
    }

private:
    std::atomic<int> collected_{0};
    std::atomic<uint32_t> last_producer_queue_{0};
    std::atomic<int> last_shard_{-1};
};

// Publish one buffer on device queue `q`, exactly as DeviceProfilerEngine's
// enqueue_ready does: write the entry, then advance the tail. The buffer must
// already be known to the manager — process_entry drops any entry whose device
// pointer has no host mapping.
template <typename Collector>
void publish(Collector &collector, TestHeader &header, int q, uint64_t *buffer) {
    collector.manager().register_mapping(buffer, buffer);  // SVM-style identity map
    uint32_t tail = header.queue_tails[q];
    header.queues[q][tail].buffer_ptr = reinterpret_cast<uint64_t>(buffer);
    header.queues[q][tail].buffer_seq = 0;
    header.queue_tails[q] = (tail + 1) % kReadyQueueSize;
}

template <typename Collector>
bool wait_for_collected(const Collector &c, int expected, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (c.collected() >= expected) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return c.collected() >= expected;
}

}  // namespace

// One drain+collector pair per AICPU thread when the module allows it.
TEST(ProfilerBaseTest, ShardCountFollowsAicpuThreadNum) {
    TestHeader header{};
    TestCollector<PerThreadModule> collector;
    collector.init(2, &header);

    EXPECT_EQ(collector.manager().shard_count(), 2);
}

// A module capped at one shard keeps one, however many AICPU threads run.
TEST(ProfilerBaseTest, ShardCountIsCappedByModuleMax) {
    TestHeader header{};
    TestCollector<SingleShardModule> collector;
    collector.init(PLATFORM_MAX_AICPU_THREADS, &header);

    EXPECT_EQ(collector.manager().shard_count(), 1);
}

// The DepGen / ScopeStats shape: the sole producer is the orchestrator, which
// writes the LAST queue (index aicpu_thread_num - 1), while only shard 0
// exists. The single drain thread must still scan every live queue to find it —
// bounding the scan by the shard count instead of the queue count would miss
// the producer entirely and silently collect nothing.
TEST(ProfilerBaseTest, SingleDrainThreadScansEveryLiveQueue) {
    constexpr int kThreads = 3;
    constexpr int kOrchQueue = kThreads - 1;

    TestHeader header{};
    uint64_t buffer = 0;

    TestCollector<SingleShardModule> collector;
    collector.init(kThreads, &header);
    ASSERT_EQ(collector.manager().shard_count(), 1);

    collector.start(nullptr);
    publish(collector, header, kOrchQueue, &buffer);

    EXPECT_TRUE(wait_for_collected(collector, 1, std::chrono::seconds(5)));
    collector.stop();

    EXPECT_EQ(collector.collected(), 1);
    EXPECT_EQ(collector.last_producer_queue(), static_cast<uint32_t>(kOrchQueue));
    EXPECT_EQ(collector.last_shard(), 0);  // folded onto the only shard
}

// Every live queue's buffers reach a collector when shards and queues are 1:1.
TEST(ProfilerBaseTest, EveryLiveQueueIsDrained) {
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS;

    TestHeader header{};
    uint64_t buffers[kThreads]{};

    TestCollector<PerThreadModule> collector;
    collector.init(kThreads, &header);
    collector.start(nullptr);

    for (int q = 0; q < kThreads; q++) {
        publish(collector, header, q, &buffers[q]);
    }

    EXPECT_TRUE(wait_for_collected(collector, kThreads, std::chrono::seconds(5)));
    collector.stop();
    EXPECT_EQ(collector.collected(), kThreads);
}

// A subsystem that emits nothing for a whole run is a valid shape: stop() must
// bring the collector down via execution_complete_, NOT via the idle-timeout
// hang detector. The guard that used to skip arming the timeout only applied
// when shard_count > 1, so a single-shard subsystem (DepGen / ScopeStats) would
// trip it. kIdleTimeoutSec is 2s here and stop() is called well after that, so
// a regression shows up as a hang or an early-abandoned shard, not a flake.
TEST(ProfilerBaseTest, SilentRunDoesNotTripIdleTimeout) {
    TestHeader header{};
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);

    collector.start(nullptr);
    std::this_thread::sleep_for(std::chrono::seconds(3));  // > kIdleTimeoutSec

    // The collector must still be alive and able to take a late buffer.
    uint64_t buffer = 0;
    publish(collector, header, 1, &buffer);
    EXPECT_TRUE(wait_for_collected(collector, 1, std::chrono::seconds(5)));

    collector.stop();
    EXPECT_EQ(collector.collected(), 1);
}
