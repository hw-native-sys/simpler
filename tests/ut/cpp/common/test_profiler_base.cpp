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
 * (ChipSwimlane / ArgsDump / PMU) but differ for the orchestrator-only ones
 * (DepGen / ScopeStats), whose single producer writes the LAST queue while
 * only one shard exists. Both shapes are covered here.
 */

#include "host/profiler_base.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
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

template <typename Module, int IdleTimeoutSeconds = 2>
class TestCollector : public profiling_common::ProfilerBase<TestCollector<Module, IdleTimeoutSeconds>, Module> {
public:
    using Base = profiling_common::ProfilerBase<TestCollector<Module, IdleTimeoutSeconds>, Module>;
    using ReadyBufferInfo = typename Module::ReadyBufferInfo;

    // Deliberately short: a false-positive idle timeout must fail the test fast
    // rather than stall it.
    static constexpr int kIdleTimeoutSec = IdleTimeoutSeconds;
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
    void
    init(int aicpu_thread_num, void *shm, std::function<int(void *, const void *, size_t)> copy_from_device = nullptr) {
        this->set_aicpu_thread_num(aicpu_thread_num);
        this->set_memory_context(
            [](size_t size) {
                return std::malloc(size);
            },
            /*register_cb=*/nullptr, /*free_cb=*/nullptr, /*copy_to_device=*/nullptr, std::move(copy_from_device), shm,
            shm, sizeof(TestHeader), /*device_id=*/0
        );
    }

private:
    std::atomic<int> collected_{0};
    std::atomic<uint32_t> last_producer_queue_{0};
    std::atomic<int> last_shard_{-1};
};

// The buffer mapping is immutable while collector threads run.
template <typename Collector>
void register_buffer(Collector &collector, uint64_t *buffer) {
    collector.manager().register_mapping(buffer, buffer);  // SVM-style identity map
}

// Publish one buffer on device queue `q` with DeviceProfilerEngine's entry-before-tail ordering.
void publish(TestHeader &header, int q, uint64_t *buffer) {
    uint32_t tail = header.queue_tails[q];
    header.queues[q][tail].buffer_ptr = reinterpret_cast<uint64_t>(buffer);
    header.queues[q][tail].buffer_seq = 0;
    wmb();
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
    register_buffer(collector, &buffer);

    collector.start(nullptr);
    publish(header, kOrchQueue, &buffer);

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
    for (int q = 0; q < kThreads; q++) {
        register_buffer(collector, &buffers[q]);
    }
    collector.start(nullptr);

    for (int q = 0; q < kThreads; q++) {
        publish(header, q, &buffers[q]);
    }

    EXPECT_TRUE(wait_for_collected(collector, kThreads, std::chrono::seconds(5)));
    collector.stop();
    EXPECT_EQ(collector.collected(), kThreads);
}

TEST(ProfilerBaseTest, StopWakesSilentCollector) {
    using namespace std::chrono_literals;

    TestHeader header{};
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);
    collector.start(nullptr);
    std::this_thread::sleep_for(10ms);

    const auto start = std::chrono::steady_clock::now();
    collector.stop();
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(elapsed, 50ms);
}

TEST(ProfilerBaseTest, QuiesceWakesSilentCollector) {
    using namespace std::chrono_literals;

    TestHeader header{};
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);
    collector.start(nullptr);
    std::this_thread::sleep_for(10ms);

    const auto start = std::chrono::steady_clock::now();
    collector.quiesce();
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(elapsed, 50ms);
    collector.stop();
}

// notify_ready_waiters() walks shards 0..shard_count_. At one shard that loop is
// indistinguishable from one that only ever notifies shard 0, so the two tests
// above hold no bound on it. Here every live shard carries a collector that
// never saw a buffer: any shard the notify misses falls back to the 100 ms tick
// and blows the bound, whichever control path published the state.
TEST(ProfilerBaseTest, LifecycleControlWakesEverySilentCollectorShard) {
    using namespace std::chrono_literals;
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS;

    TestHeader header{};
    TestCollector<PerThreadModule> collector;
    collector.init(kThreads, &header);
    ASSERT_EQ(collector.manager().shard_count(), kThreads);
    collector.start(nullptr);
    std::this_thread::sleep_for(10ms);

    const auto quiesce_start = std::chrono::steady_clock::now();
    collector.quiesce();
    const auto quiesce_elapsed = std::chrono::steady_clock::now() - quiesce_start;

    const auto stop_start = std::chrono::steady_clock::now();
    collector.stop();
    const auto stop_elapsed = std::chrono::steady_clock::now() - stop_start;

    EXPECT_LT(quiesce_elapsed, 50ms);
    EXPECT_LT(stop_elapsed, 50ms);
}

// A subsystem that emits nothing for a whole run is a valid shape: stop() must
// bring the collector down via execution_complete_, NOT via the idle-timeout
// hang detector. The guard that used to skip arming the timeout only applied
// when shard_count > 1, so a single-shard subsystem (DepGen / ScopeStats) would
// trip it. kIdleTimeoutSec is 2s here and stop() is called well after that, so
// a regression shows up as a hang or an early-abandoned shard, not a flake.
TEST(ProfilerBaseTest, SilentRunDoesNotTripIdleTimeout) {
    TestHeader header{};
    uint64_t buffer = 0;
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);
    register_buffer(collector, &buffer);

    collector.start(nullptr);
    std::this_thread::sleep_for(std::chrono::seconds(3));  // > kIdleTimeoutSec

    // The collector must still be alive and able to take a late buffer.
    publish(header, 1, &buffer);
    EXPECT_TRUE(wait_for_collected(collector, 1, std::chrono::seconds(5)));

    collector.stop();
    EXPECT_EQ(collector.collected(), 1);
}

TEST(ProfilerBaseTest, CollectorStaysAliveAfterArmedIdleTimeout) {
    using namespace std::chrono_literals;

    TestHeader header{};
    uint64_t first_buffer = 0;
    uint64_t late_buffer = 0;
    TestCollector<SingleShardModule, 0> collector;
    collector.init(2, &header);
    register_buffer(collector, &first_buffer);
    register_buffer(collector, &late_buffer);
    collector.start(nullptr);

    publish(header, 1, &first_buffer);
    EXPECT_TRUE(wait_for_collected(collector, 1, 5s));

    // Once traffic has armed the idle detector, a zero-second timeout fires on
    // the next empty poll. The collector must report it without exiting.
    std::this_thread::sleep_for(250ms);

    publish(header, 1, &late_buffer);
    EXPECT_TRUE(wait_for_collected(collector, 2, 5s));

    collector.stop();
    EXPECT_EQ(collector.collected(), 2);
}

// quiesce() gives stop()'s drain guarantee without retiring the threads: on
// return every published buffer has reached on_buffer_collected, and the
// collector is still able to take more. This is what lets the collectors stay
// resident across runs instead of being started and joined per run.
TEST(ProfilerBaseTest, QuiesceDrainsWithoutRetiringThreads) {
    constexpr int kThreads = PLATFORM_MAX_AICPU_THREADS;

    TestHeader header{};
    uint64_t first[kThreads]{};
    uint64_t second[kThreads]{};

    TestCollector<PerThreadModule> collector;
    collector.init(kThreads, &header);
    for (int q = 0; q < kThreads; q++) {
        register_buffer(collector, &first[q]);
        register_buffer(collector, &second[q]);
    }
    collector.start(nullptr);

    for (int q = 0; q < kThreads; q++) {
        publish(header, q, &first[q]);
    }
    collector.quiesce();
    // No wait_for_collected here on purpose: quiesce() must have delivered
    // everything by the time it returns, so polling for it would hide a
    // handshake that reports too early.
    EXPECT_EQ(collector.collected(), kThreads);

    for (int q = 0; q < kThreads; q++) {
        publish(header, q, &second[q]);
    }
    collector.quiesce();
    EXPECT_EQ(collector.collected(), 2 * kThreads);

    collector.stop();
    EXPECT_EQ(collector.collected(), 2 * kThreads);
}

TEST(ProfilerBaseTest, QuiesceAcknowledgementRequiresPostRequestSweep) {
    using namespace std::chrono_literals;

    TestHeader header{};
    uint64_t buffer = 0;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    int queue_zero_reads = 0;
    bool first_sweep_paused = false;
    bool second_sweep_paused = false;
    bool release_first_sweep = false;
    bool release_second_sweep = false;

    auto gated_copy = [&](void * /*dst*/, const void *src, size_t /*size*/) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        if (src == &header.queue_heads[0]) {
            queue_zero_reads++;
            if (queue_zero_reads == 2) {
                second_sweep_paused = true;
                gate_cv.notify_all();
                gate_cv.wait(lock, [&]() {
                    return release_second_sweep;
                });
            }
        } else if (src == &header.queue_heads[1] && !first_sweep_paused) {
            first_sweep_paused = true;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&]() {
                return release_first_sweep;
            });
        }
        return 0;
    };

    TestCollector<SingleShardModule> collector;
    collector.init(2, &header, gated_copy);
    register_buffer(collector, &buffer);
    collector.start(nullptr);

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        if (!gate_cv.wait_for(lock, 2s, [&]() {
                return first_sweep_paused;
            })) {
            release_first_sweep = true;
            release_second_sweep = true;
            lock.unlock();
            gate_cv.notify_all();
            collector.stop();
            FAIL() << "drain did not reach the first sweep gate";
        }
    }

    // Queue zero is published only after this sweep has already observed it empty.
    publish(header, 0, &buffer);
    std::atomic<bool> quiesce_started{false};
    std::atomic<bool> quiesce_done{false};
    std::thread quiesce_thread([&]() {
        quiesce_started.store(true, std::memory_order_release);
        collector.quiesce();
        quiesce_done.store(true, std::memory_order_release);
    });
    while (!quiesce_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(100ms);

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_first_sweep = true;
    }
    gate_cv.notify_all();

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        if (!gate_cv.wait_for(lock, 2s, [&]() {
                return second_sweep_paused;
            })) {
            release_second_sweep = true;
            lock.unlock();
            gate_cv.notify_all();
            quiesce_thread.join();
            collector.stop();
            FAIL() << "drain did not reach the post-request sweep gate";
        }
    }

    std::this_thread::sleep_for(250ms);
    const bool completed_before_post_request_sweep = quiesce_done.load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_second_sweep = true;
    }
    gate_cv.notify_all();

    quiesce_thread.join();
    EXPECT_FALSE(completed_before_post_request_sweep);
    EXPECT_EQ(collector.collected(), 1);
    collector.stop();
}

// A subsystem that emitted nothing still has to complete the handshake. The
// collector loop skips its idle bookkeeping for a shard that has never seen a
// buffer, so an ack placed behind that guard would leave quiesce() waiting
// forever on a silent run — a hang, not a wrong count.
TEST(ProfilerBaseTest, QuiesceCompletesOnASilentCollector) {
    TestHeader header{};
    uint64_t buffer = 0;
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);
    register_buffer(collector, &buffer);
    collector.start(nullptr);

    collector.quiesce();
    EXPECT_EQ(collector.collected(), 0);

    // Still live afterwards.
    publish(header, 1, &buffer);
    collector.quiesce();
    EXPECT_EQ(collector.collected(), 1);

    collector.stop();
}

// quiesce() before start() and after stop() are both no-ops rather than hangs:
// there are no workers to answer the handshake in either state.
TEST(ProfilerBaseTest, QuiesceIsANoOpWithoutRunningThreads) {
    TestHeader header{};
    TestCollector<SingleShardModule> collector;
    collector.init(2, &header);

    collector.quiesce();  // before start()

    collector.start(nullptr);
    collector.stop();

    collector.quiesce();  // after stop()
    EXPECT_EQ(collector.collected(), 0);
}
