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

#include <gtest/gtest.h>

#include <atomic>
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
    ASSERT_EQ(
        collector.init(1, 1, csv_path.string(), PmuEventType::PIPE_UTILIZATION, test_alloc, nullptr, test_free, 0), 0
    );

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
    ASSERT_EQ(
        collector.init(
            1, kShardCount, csv_path.string(), PmuEventType::PIPE_UTILIZATION, test_alloc, nullptr, test_free, 0
        ),
        0
    );

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
                buffer.records[0].task_id = static_cast<uint64_t>(shard * kRecordsPerShard + record);
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
