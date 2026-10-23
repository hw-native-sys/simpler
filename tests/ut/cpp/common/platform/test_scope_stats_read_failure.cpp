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

#include "host/scope_stats_collector.h"

#include <gtest/gtest.h>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "host/profiling_copy.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace {

const volatile void *refused_source = nullptr;
int refused_copies = 0;

class InspectableScopeStats : public ScopeStatsCollector {
public:
    void *host_shared() const { return shm_host_; }
};

void *allocate(size_t size) { return std::calloc(1, size); }

int release(void *ptr) {
    std::free(ptr);
    return 0;
}

}  // namespace

// Both architectures use detached host shadows in this transport test. Only
// the held-buffer copy fails; the shared accounting read still succeeds.
int profiling_copy_to_device(volatile void *dst, const void *src, size_t bytes) {
    std::memcpy(const_cast<void *>(dst), src, bytes);
    return 0;
}

int profiling_copy_from_device(volatile void *dst, const volatile void *src, size_t bytes) {
    if (src == refused_source) {
        ++refused_copies;
        return -1;
    }
    std::memcpy(const_cast<void *>(dst), const_cast<const void *>(src), bytes);
    return 0;
}

std::function<int(void *, const void *, size_t)> profiling_copy_to_device_or_null() {
    return profiling_copy_to_device_for_ops;
}

std::function<int(void *, const void *, size_t)> profiling_copy_from_device_or_null() {
    return profiling_copy_from_device_for_ops;
}

TEST(ScopeStatsReadFailureTest, HeldBufferReadFailureCannotRecoverOrExportThePreviousShadow) {
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, allocate, nullptr, release, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
    set_platform_scope_stats_run_terminal(0);
    set_platform_run_result(0, 11);
    set_scope_stats_enabled(true);
    scope_stats_aicpu_set_orch_thread_idx(0);
    collector.begin_run(11);
    scope_stats_aicpu_begin_run();
    scope_stats_set_pending_site("previous.cpp", 7);
    scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
    ASSERT_FALSE(collector.reconcile_counters());
    ASSERT_EQ(collector.total_collected(), 1u);
    auto *state = get_scope_stats_buffer_state(shm, 0);
    const uint64_t held = state->current_buf_ptr;
    ASSERT_NE(held, 0u);

    collector.begin_run(22);
    set_platform_run_result(0, 22);
    scope_stats_aicpu_begin_run();
    ASSERT_EQ(state->run_epoch, 22u);
    ASSERT_EQ(reinterpret_cast<ScopeStatsBuffer *>(held)->count, 0u);
    refused_source = reinterpret_cast<const void *>(held);
    refused_copies = 0;
    const auto root = std::filesystem::temp_directory_path() / ("scope_stats_held_copy_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    EXPECT_FALSE(collector.reconcile_counters());
    EXPECT_EQ(refused_copies, 1);
    EXPECT_EQ(collector.total_collected(), 0u);
    EXPECT_NE(collector.write_jsonl(root.string()), 0);
    EXPECT_FALSE(std::filesystem::exists(root));

    refused_source = nullptr;
    EXPECT_TRUE(collector.reconcile_counters());
    EXPECT_EQ(collector.write_jsonl(root.string()), 0);
    set_platform_scope_stats_base(0);
    set_platform_run_result(0, 0);
    collector.finalize(nullptr, release);
    std::filesystem::remove_all(root);
}

TEST(ScopeStatsReadFailureTest, ExportUsesTheCheckedSnapshotInsteadOfTheLiveHeader) {
    InspectableScopeStats collector;
    ASSERT_EQ(collector.init(1, allocate, nullptr, release, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
    set_platform_scope_stats_run_terminal(0);
    set_platform_run_result(0, 33);
    set_scope_stats_enabled(true);
    scope_stats_aicpu_set_orch_thread_idx(0);
    collector.begin_run(33);
    scope_stats_aicpu_begin_run();
    scope_stats_set_pending_site("frozen.cpp", 3);
    scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
    ASSERT_FALSE(collector.reconcile_counters());
    ASSERT_EQ(collector.total_collected(), 1u);
    auto *header = get_scope_stats_header(collector.host_shared());
    auto *state = get_scope_stats_buffer_state(collector.host_shared(), 0);
    header->fatal_latched = 1;
    state->total_record_count = 99;
    state->dropped_record_count = 98;
    const auto root = std::filesystem::temp_directory_path() / ("scope_stats_snapshot_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    ASSERT_EQ(collector.write_jsonl(root.string()), 0);
    std::ifstream input(root / "scope_stats/scope_stats.jsonl");
    const std::string data{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_NE(data.find("\"fatal\": false"), std::string::npos);
    EXPECT_NE(data.find("\"total\": 1"), std::string::npos);
    EXPECT_NE(data.find("\"dropped\": 0"), std::string::npos);
    set_platform_scope_stats_base(0);
    set_platform_run_result(0, 0);
    collector.finalize(nullptr, release);
    std::filesystem::remove_all(root);
}
