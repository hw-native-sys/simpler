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

#include "host/args_dump_collector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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

TEST(ArgsDumpCollectorTest, MergesConcurrentShardRecordsIntoManifest) {
    const std::filesystem::path test_dir =
        std::filesystem::temp_directory_path() / ("args_dump_collector_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(test_dir);
    ASSERT_TRUE(std::filesystem::create_directories(test_dir));

    constexpr int kRecordsPerShard = 32;
    constexpr int kShardCount = DumpModule::kCollectorThreadCount;
    ArgsDumpCollector collector;
    ASSERT_EQ(
        collector.initialize(1, 0, test_alloc, nullptr, test_free, test_dir.string(), DumpArgsLevel::FULL_JSON_ONLY), 0
    );

    std::vector<DumpMetaBuffer> buffers(kShardCount);
    std::atomic<int> ready_workers{0};
    std::atomic<bool> start_workers{false};
    std::vector<std::thread> workers;
    workers.reserve(kShardCount);
    for (int shard = 0; shard < kShardCount; shard++) {
        workers.emplace_back([&collector, &buffers, &ready_workers, &start_workers, shard] {
            DumpMetaBuffer &buffer = buffers[shard];
            buffer.count = kRecordsPerShard;
            for (int record = 0; record < kRecordsPerShard; record++) {
                ArgsDumpRecord &entry = buffer.records[record];
                entry.task_id = static_cast<uint64_t>(shard * kRecordsPerShard + record);
                entry.role = static_cast<uint8_t>(ArgsDumpRole::INPUT);
                entry.stage = static_cast<uint8_t>(ArgsDumpStage::BEFORE_DISPATCH);
                entry.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
                entry.arg_index = static_cast<uint32_t>(record);
                entry.func_ids[0] = static_cast<uint16_t>(shard);
                entry.func_count = 1;
            }

            DumpReadyBufferInfo info{};
            info.thread_index = 0;
            info.host_buffer_ptr = &buffer;
            ready_workers.fetch_add(1, std::memory_order_release);
            while (!start_workers.load(std::memory_order_acquire)) {}
            collector.on_buffer_collected(info, shard);
        });
    }
    while (ready_workers.load(std::memory_order_acquire) != kShardCount) {}
    start_workers.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }

    ASSERT_EQ(collector.export_dump_files(), 0);
    const std::filesystem::path manifest_path = test_dir / "args_dump" / "args_dump.json";
    std::ifstream manifest_file(manifest_path);
    ASSERT_TRUE(manifest_file.is_open());
    const std::string manifest{std::istreambuf_iterator<char>(manifest_file), std::istreambuf_iterator<char>()};
    EXPECT_NE(manifest.find("\"total_args\": " + std::to_string(kShardCount * kRecordsPerShard)), std::string::npos);

    collector.finalize(nullptr, test_free);
    std::filesystem::remove_all(test_dir);
}
