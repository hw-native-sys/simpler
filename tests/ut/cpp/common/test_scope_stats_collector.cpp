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
#include "host/scope_stats_host_capture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace {

void *test_alloc(size_t size) { return std::calloc(1, size); }

int test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

void fill_record(ScopeStatsRecord &rec, const char *site, int line, int16_t phase) {
    std::memset(&rec, 0, sizeof(rec));
    std::snprintf(rec.site_file_basename, sizeof(rec.site_file_basename), "%s", site);
    rec.site_line = line;
    rec.phase = phase;
    rec.depth = 1;
    rec.ring_id = 1;
    rec.task_start = 10;
    rec.task_end = 14;
    rec.heap_start = 1024;
    rec.heap_end = 4096;
    rec.dep_pool_start = 2;
    rec.dep_pool_end = 5;
    rec.tensormap_used = 7;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(ScopeStatsCollectorTest, ReconcileRecoversUnflushedCurrentBuffer) {
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, test_alloc, nullptr, test_free, 0), 0);

    auto *header = get_scope_stats_header(collector.get_scope_stats_shm_device_ptr());
    auto *state = get_scope_stats_buffer_state(collector.get_scope_stats_shm_device_ptr(), 0);
    ASSERT_EQ(header->num_instances, 1u);

    const uint32_t head = state->free_queue.head;
    const uint32_t tail = state->free_queue.tail;
    ASSERT_LT(head, tail);

    uint64_t buf_dev = state->free_queue.buffer_ptrs[head % PLATFORM_SCOPE_STATS_SLOT_COUNT];
    ASSERT_NE(buf_dev, 0u);
    state->free_queue.head = head + 1;
    state->current_buf_ptr = buf_dev;
    state->current_buf_seq = 7;
    state->total_record_count = 2;

    auto *buf = reinterpret_cast<ScopeStatsBuffer *>(buf_dev);
    buf->count = 2;
    fill_record(buf->records[0], "recover.cpp", 123, SCOPE_STATS_PHASE_BEGIN);
    fill_record(buf->records[1], "recover.cpp", 123, SCOPE_STATS_PHASE_END);

    EXPECT_FALSE(collector.reconcile_counters());
    EXPECT_EQ(collector.total_collected(), 2u);

    std::filesystem::path out_dir = std::filesystem::temp_directory_path() /
                                    ("scope_stats_collector_test_" + std::to_string(::getpid()) + "_recover");
    std::filesystem::remove_all(out_dir);
    ASSERT_EQ(collector.write_jsonl(out_dir.string()), 0);

    std::string jsonl = read_file(out_dir / "scope_stats" / "scope_stats.jsonl");
    ASSERT_FALSE(jsonl.empty());
    EXPECT_NE(jsonl.find("\"total\": 2"), std::string::npos);
    EXPECT_NE(jsonl.find("\"site\": \"recover.cpp:123\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"phase\": \"begin\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"phase\": \"end\""), std::string::npos);

    // A second reconcile on the same un-flushed pointer should not append the
    // same buffer again.
    EXPECT_FALSE(collector.reconcile_counters());
    EXPECT_EQ(collector.total_collected(), 2u);

    // A later abnormal run may reuse the same current_buf_ptr. A changed
    // device total means this is a new in-flight buffer snapshot, not the
    // duplicate reconcile above.
    state->total_record_count = 3;
    buf->count = 1;
    fill_record(buf->records[0], "recover_again.cpp", 456, SCOPE_STATS_PHASE_BEGIN);
    EXPECT_FALSE(collector.reconcile_counters());
    EXPECT_EQ(collector.total_collected(), 3u);

    std::filesystem::remove_all(out_dir);
    collector.finalize(nullptr, test_free);
}

TEST(ScopeStatsHostCaptureTest, DisabledCaptureProducesNoRecords) {
    ScopeStatsHostCapture capture;
    capture.begin_capture(/*task_window_cap=*/128, /*heap_cap=*/4096, /*tensormap_cap=*/256);
    capture.set_pending_site("disabled.cpp", 17);
    capture.begin(0, 0, 0, 0, 0, 0, 0, 0);
    capture.end(0, 0, 1, 0, 64, 0, 0, 1);

    EXPECT_TRUE(capture.records().empty());
    EXPECT_EQ(capture.write_jsonl("/tmp"), -3);
}

TEST(ScopeStatsHostCaptureTest, NestedCapturePreservesSitesCapacitiesWrapsAndFatal) {
    ScopeStatsHostCapture capture;
    capture.set_enabled(true);
    capture.begin_capture(/*task_window_cap=*/128, /*heap_cap=*/4096, /*tensormap_cap=*/256);

    capture.begin(0, 0, 0, 0, 0, 0, 0, 0);
    capture.set_pending_site("/tmp/source/nested_scope.cpp", 42);
    capture.begin(0, 0, 1, 0, 1024, 0, 0, 1);
    capture.note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    capture.note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    capture.end(0, 0, 5, 0, 512, 0, 0, 3);
    capture.on_fatal();
    capture.end(0, 0, 5, 0, 512, 0, 0, 3);

    const auto &records = capture.records();
    ASSERT_EQ(records.size(), 4u);
    EXPECT_EQ(records[0].depth, 0);
    EXPECT_EQ(records[1].depth, 1);
    EXPECT_EQ(records[1].phase, SCOPE_STATS_PHASE_BEGIN);
    EXPECT_EQ(records[2].phase, SCOPE_STATS_PHASE_END);
    EXPECT_STREQ(records[1].site_file_basename, "nested_scope.cpp");
    EXPECT_EQ(records[1].site_line, 42);
    EXPECT_EQ(records[2].heap_end, 2 * 4096 + 512);

    const auto &header = capture.header();
    EXPECT_EQ(header.task_window_cap[0], 128);
    EXPECT_EQ(header.heap_cap[0], 4096u);
    EXPECT_EQ(header.dep_pool_cap[0], 0);
    EXPECT_EQ(header.tensormap_cap, 256);
    EXPECT_EQ(header.fatal_latched, 1u);

    std::filesystem::path out_dir =
        std::filesystem::temp_directory_path() / ("scope_stats_host_capture_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(out_dir);
    ASSERT_EQ(capture.write_jsonl(out_dir.c_str()), 0);
    std::string jsonl = read_file(out_dir / "scope_stats" / "scope_stats.jsonl");
    EXPECT_NE(jsonl.find("\"fatal\": true"), std::string::npos);
    EXPECT_NE(jsonl.find("\"total\": 4"), std::string::npos);
    EXPECT_NE(jsonl.find("\"site\": \"nested_scope.cpp:42\""), std::string::npos);
    std::filesystem::remove_all(out_dir);
}

TEST(ScopeStatsHostCaptureTest, BeginCaptureClearsPriorRunState) {
    ScopeStatsHostCapture capture;
    capture.set_enabled(true);
    capture.begin_capture(64, 2048, 32);
    capture.begin(0, 0, 0, 0, 0, 0, 0, 0);
    capture.end(0, 0, 1, 0, 128, 0, 0, 1);
    ASSERT_EQ(capture.records().size(), 2u);

    capture.begin_capture(32, 1024, 16);
    EXPECT_TRUE(capture.records().empty());
    EXPECT_EQ(capture.header().task_window_cap[0], 32);
    EXPECT_EQ(capture.header().heap_cap[0], 1024u);
    EXPECT_EQ(capture.header().fatal_latched, 0u);
}
