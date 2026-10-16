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

#include "host/pmu_collector.h"

#include "support/test_task_id.h"

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

void *test_alloc(size_t size) { return std::calloc(1, size); }

int test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

}  // namespace

TEST(PmuCollectorTest, PreservesShardFilesWhenFinalCsvCannotBeOpened) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("pmu_collector_failure_test_" + std::to_string(::getpid()));
    const std::filesystem::path csv_path = test_dir / "pmu.csv";
    const std::filesystem::path shard_path = test_dir / "pmu.csv.shard0.tmp";
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(csv_path));

    PmuCollector collector;
    collector.begin_run(csv_path.string(), PmuEventType::PIPE_UTILIZATION);
    ASSERT_EQ(collector.init(1, 1, test_alloc, nullptr, test_free, 0), 0);

    PmuBuffer buffer{};
    buffer.count = 1;
    PmuReadyBufferInfo info{};
    info.core_index = 0;
    info.thread_index = 0;
    info.host_buffer_ptr = &buffer;
    collector.on_buffer_collected(info, 0);
    ASSERT_TRUE(std::filesystem::exists(shard_path));

    auto *state = get_pmu_buffer_state(collector.get_pmu_shm_device_ptr(), 0);
    state->total_record_count = 1;
    collector.reconcile_counters();
    EXPECT_TRUE(std::filesystem::exists(shard_path));
    EXPECT_TRUE(std::filesystem::is_directory(csv_path));

    collector.finalize(nullptr, test_free);
    EXPECT_TRUE(std::filesystem::exists(shard_path));
    std::filesystem::remove_all(test_dir);
}

TEST(PmuCollectorTest, MergesConcurrentShardWritesIntoFinalCsv) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("pmu_collector_merge_test_" + std::to_string(::getpid()));
    const std::filesystem::path csv_path = test_dir / "pmu.csv";
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(test_dir));

    constexpr int kRecordsPerShard = 64;
    constexpr int kShardCount = PmuModule::kMaxCollectorThreads;
    PmuCollector collector;
    collector.begin_run(csv_path.string(), PmuEventType::PIPE_UTILIZATION);
    ASSERT_EQ(collector.init(1, kShardCount, test_alloc, nullptr, test_free, 0), 0);

    std::atomic<int> ready_workers{0};
    std::atomic<bool> start_workers{false};
    std::vector<std::thread> workers;
    workers.reserve(kShardCount);
    for (int shard = 0; shard < kShardCount; shard++) {
        workers.emplace_back([&collector, &ready_workers, &start_workers, shard] {
            PmuBuffer buffer{};
            buffer.count = 1;
            PmuReadyBufferInfo info{};
            info.core_index = 0;
            info.thread_index = static_cast<uint32_t>(shard);
            info.host_buffer_ptr = &buffer;

            ready_workers.fetch_add(1, std::memory_order_release);
            while (!start_workers.load(std::memory_order_acquire)) {}
            for (int record = 0; record < kRecordsPerShard; record++) {
                buffer.records[0].task_id = simpler::ut::test_task_id(shard * kRecordsPerShard + record);
                collector.on_buffer_collected(info, shard);
            }
        });
    }
    while (ready_workers.load(std::memory_order_acquire) != kShardCount) {}
    start_workers.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }

    auto *state = get_pmu_buffer_state(collector.get_pmu_shm_device_ptr(), 0);
    state->total_record_count = kShardCount * kRecordsPerShard;
    collector.reconcile_counters();

    EXPECT_TRUE(std::filesystem::is_regular_file(csv_path));
    for (int shard = 0; shard < kShardCount; shard++) {
        EXPECT_FALSE(std::filesystem::exists(csv_path.string() + ".shard" + std::to_string(shard) + ".tmp"));
    }
    std::ifstream csv(csv_path);
    std::string line;
    int line_count = 0;
    while (std::getline(csv, line)) {
        line_count++;
    }
    EXPECT_EQ(line_count, 1 + kShardCount * kRecordsPerShard);

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(test_dir);
}

TEST(PmuCollectorTest, FinalizeReleasesItsSharedHeaderExactlyOnce) {
    std::vector<void *> allocations;
    std::vector<void *> releases;
    auto alloc = [&](size_t size) -> void * {
        void *pointer = std::calloc(1, size);
        allocations.push_back(pointer);
        return pointer;
    };
    auto release = [&](void *pointer) -> int {
        releases.push_back(pointer);
        return 0;
    };
    {
        PmuCollector collector;
        collector.begin_run("/tmp/issue1710-pmu-release.csv", PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector.init(1, 1, alloc, nullptr, release, 0), 0);
        void *header = collector.get_pmu_shm_device_ptr();
        ASSERT_NE(header, nullptr);
        collector.finalize(nullptr, release);
        EXPECT_EQ(std::count(releases.begin(), releases.end(), header), 1)
            << "the independently allocated shared header must be released";
        collector.finalize(nullptr, release);
        EXPECT_EQ(std::count(releases.begin(), releases.end(), header), 1);
    }
    for (void *pointer : allocations)
        std::free(pointer);
}

namespace {
std::vector<void *> i1710_registrations;
std::vector<void *> i1710_unregistrations;
int i1710_register(void *pointer, size_t, int, void **host) {
    i1710_registrations.push_back(pointer);
    *host = pointer;
    return 0;
}
int i1710_unregister(void *pointer, int) {
    i1710_unregistrations.push_back(pointer);
    return 0;
}
}  // namespace

TEST(PmuCollectorTest, FinalizeUnregistersTheSharedHeaderBeforeReleasingIt) {
    i1710_registrations.clear();
    i1710_unregistrations.clear();
    std::vector<void *> allocations;
    std::vector<void *> releases;
    auto alloc = [&](size_t size) -> void * {
        void *pointer = std::calloc(1, size);
        allocations.push_back(pointer);
        return pointer;
    };
    auto release = [&](void *pointer) -> int {
        const int registered = std::count(i1710_registrations.begin(), i1710_registrations.end(), pointer);
        EXPECT_EQ(std::count(i1710_unregistrations.begin(), i1710_unregistrations.end(), pointer), registered);
        releases.push_back(pointer);
        return 0;
    };
    {
        PmuCollector collector;
        collector.begin_run("/tmp/issue1710-pmu-map-release.csv", PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector.init(1, 1, alloc, i1710_register, release, 0), 0);
        void *header = collector.get_pmu_shm_device_ptr();
        ASSERT_EQ(std::count(i1710_registrations.begin(), i1710_registrations.end(), header), 1);
        collector.finalize(i1710_unregister, release);
        EXPECT_EQ(std::count(i1710_unregistrations.begin(), i1710_unregistrations.end(), header), 1);
        EXPECT_EQ(std::count(releases.begin(), releases.end(), header), 1);
        collector.finalize(i1710_unregister, release);
        EXPECT_EQ(std::count(i1710_unregistrations.begin(), i1710_unregistrations.end(), header), 1);
    }
    for (void *pointer : allocations)
        std::free(pointer);
}

TEST(PmuCollectorTest, FinalizeReleasesEveryOwnedAllocationExactlyOnce) {
    std::vector<void *> allocations;
    std::vector<void *> releases;
    auto alloc = [&](size_t size) -> void * {
        void *pointer = std::calloc(1, size);
        allocations.push_back(pointer);
        return pointer;
    };
    auto release = [&](void *pointer) -> int {
        releases.push_back(pointer);
        return 0;
    };
    {
        PmuCollector collector;
        collector.begin_run("/tmp/issue1710-pmu-owned-allocations.csv", PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector.init(2, 1, alloc, nullptr, release, 0), 0);
        collector.finalize(nullptr, release);
        for (void *pointer : allocations) {
            EXPECT_EQ(std::count(releases.begin(), releases.end(), pointer), 1)
                << "every allocation owned by the collector must reach its release callback";
        }
        const size_t release_count = releases.size();
        collector.finalize(nullptr, release);
        EXPECT_EQ(releases.size(), release_count);
    }
    for (void *pointer : allocations)
        std::free(pointer);
}

namespace {

struct PmuAllocationLedger {
    std::vector<void *> allocations;
    std::vector<void *> registrations;
    std::vector<void *> unregistrations;
    std::vector<void *> releases;
    int allocation_attempts = 0;
    int fail_allocation = 0;
    int registration_attempts = 0;
    int fail_registration = 0;

    ~PmuAllocationLedger() {
        for (void *pointer : allocations)
            std::free(pointer);
    }

    void *allocate(size_t size) {
        if (++allocation_attempts == fail_allocation) return nullptr;
        void *pointer = std::calloc(1, size);
        allocations.push_back(pointer);
        return pointer;
    }

    int release(void *pointer) {
        releases.push_back(pointer);
        return 0;
    }

    void expect_released_once() const {
        for (void *pointer : allocations) {
            EXPECT_EQ(std::count(releases.begin(), releases.end(), pointer), 1);
        }
        for (void *pointer : registrations) {
            EXPECT_EQ(std::count(unregistrations.begin(), unregistrations.end(), pointer), 1);
        }
    }
};

thread_local PmuAllocationLedger *pmu_test_ledger = nullptr;

int register_pmu_test_memory(void *pointer, size_t, int, void **host) {
    if (++pmu_test_ledger->registration_attempts == pmu_test_ledger->fail_registration) return 19;
    pmu_test_ledger->registrations.push_back(pointer);
    *host = pointer;
    return 0;
}

int unregister_pmu_test_memory(void *pointer, int device) {
    EXPECT_EQ(device, 7);
    pmu_test_ledger->unregistrations.push_back(pointer);
    return 0;
}

}  // namespace

TEST(PmuCollectorTest, RepeatedInitializeFinalizeBalancesEveryAllocationAndMapping) {
    PmuAllocationLedger ledger;
    pmu_test_ledger = &ledger;
    PmuCollector collector;
    auto alloc = [&](size_t size) {
        return ledger.allocate(size);
    };
    auto release = [&](void *pointer) {
        return ledger.release(pointer);
    };
    for (int round = 0; round < 10; ++round) {
        collector.begin_run("/tmp/pmu_lifecycle_test.csv", PmuEventType::PIPE_UTILIZATION);
        const bool registered = round % 2 == 0;
        ASSERT_EQ(collector.init(2, 1, alloc, registered ? register_pmu_test_memory : nullptr, release, 7), 0);
        collector.finalize(unregister_pmu_test_memory, release);
        ledger.expect_released_once();
        const size_t released = ledger.releases.size();
        collector.finalize(unregister_pmu_test_memory, release);
        EXPECT_EQ(ledger.releases.size(), released);
    }
}

TEST(PmuCollectorTest, FinalizeUsesSuppliedQuarantineCallbackInsteadOfStoredAllocator) {
    PmuAllocationLedger ledger;
    PmuCollector collector;
    auto alloc = [&](size_t size) {
        return ledger.allocate(size);
    };
    auto release = [&](void *pointer) {
        return ledger.release(pointer);
    };
    collector.begin_run("/tmp/pmu_quarantine_test.csv", PmuEventType::PIPE_UTILIZATION);
    ASSERT_EQ(collector.init(2, 1, alloc, nullptr, release, 7), 0);
    std::vector<void *> quarantined;
    auto quarantine = [&](void *pointer) {
        quarantined.push_back(pointer);
        return 0;
    };
    collector.finalize(nullptr, quarantine);
    EXPECT_TRUE(ledger.releases.empty());
    for (void *pointer : ledger.allocations) {
        EXPECT_EQ(std::count(quarantined.begin(), quarantined.end(), pointer), 1);
    }
    const size_t count = quarantined.size();
    collector.finalize(nullptr, quarantine);
    EXPECT_EQ(quarantined.size(), count);
}

TEST(PmuCollectorTest, FailedInitializationCanBeFinalizedAndRetriedAtEveryAllocation) {
    int allocation_count = 0;
    {
        PmuAllocationLedger ledger;
        PmuCollector collector;
        auto alloc = [&](size_t size) {
            return ledger.allocate(size);
        };
        auto release = [&](void *pointer) {
            return ledger.release(pointer);
        };
        collector.begin_run("/tmp/pmu_allocation_failure_test.csv", PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector.init(2, 1, alloc, nullptr, release, 7), 0);
        allocation_count = ledger.allocation_attempts;
        collector.finalize(nullptr, release);
    }
    for (int fail_at = 1; fail_at <= allocation_count; ++fail_at) {
        SCOPED_TRACE(fail_at);
        PmuAllocationLedger ledger;
        pmu_test_ledger = &ledger;
        ledger.fail_allocation = fail_at;
        PmuCollector collector;
        auto alloc = [&](size_t size) {
            return ledger.allocate(size);
        };
        auto release = [&](void *pointer) {
            return ledger.release(pointer);
        };
        collector.begin_run("/tmp/pmu_allocation_failure_test.csv", PmuEventType::PIPE_UTILIZATION);
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        PmuRegisterCallback register_callback = nullptr;
#else
        PmuRegisterCallback register_callback = register_pmu_test_memory;
#endif
        EXPECT_NE(collector.init(2, 1, alloc, register_callback, release, 7), 0);
        collector.finalize(unregister_pmu_test_memory, release);
        ledger.expect_released_once();
        ledger.fail_allocation = 0;
        ASSERT_EQ(collector.init(2, 1, alloc, register_callback, release, 7), 0);
        collector.finalize(unregister_pmu_test_memory, release);
        ledger.expect_released_once();
    }
}

#if !defined(SIMPLER_TEST_PMU_ARCH_A5)
TEST(PmuCollectorTest, FailedRegistrationCanBeFinalizedAndRetried) {
    for (int fail_at : {1, 2, 3}) {
        SCOPED_TRACE(fail_at);
        PmuAllocationLedger ledger;
        pmu_test_ledger = &ledger;
        ledger.fail_registration = fail_at;
        PmuCollector collector;
        auto alloc = [&](size_t size) {
            return ledger.allocate(size);
        };
        auto release = [&](void *pointer) {
            return ledger.release(pointer);
        };
        collector.begin_run("/tmp/pmu_registration_failure_test.csv", PmuEventType::PIPE_UTILIZATION);
        EXPECT_NE(collector.init(2, 1, alloc, register_pmu_test_memory, release, 7), 0);
        collector.finalize(unregister_pmu_test_memory, release);
        ledger.expect_released_once();
        ledger.fail_registration = 0;
        ASSERT_EQ(collector.init(2, 1, alloc, register_pmu_test_memory, release, 7), 0);
        collector.finalize(unregister_pmu_test_memory, release);
        ledger.expect_released_once();
    }
}
#endif
