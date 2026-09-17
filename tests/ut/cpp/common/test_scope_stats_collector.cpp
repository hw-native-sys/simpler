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

// ---------------------------------------------------------------------------
// Run identity: a buffer carries the run that acquired it, so records stay
// attributed after that buffer has been returned and reused (§2.4 step 1).
//
// Both sides are production code: the device module's own acquire / append /
// flush functions stamp and publish, and the host collector's own reconcile /
// accessors read it back. Consumer lag is simply not collecting between the
// two runs, so nothing here relaxes admission or reorders runs.
//
// What these cannot show is device cache visibility — every store here is
// plain host memory. The stamp's publication ordering comes from the engine's
// fence in claim_free() and is asserted on device separately.
// ---------------------------------------------------------------------------

namespace {

// One run's worth of device-side production: publish identity, append `n`
// begin/end pairs through the real producer, then run-end flush.
void produce_run(uint64_t run_epoch, void *shm_dev, int n, const char *site) {
    set_platform_run_result(/*region_base=*/0, run_epoch);
    set_scope_stats_enabled(true);
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm_dev));
    scope_stats_aicpu_set_orch_thread_idx(0);
    for (int i = 0; i < n; i++) {
        scope_stats_set_pending_site(site, 100 + i);
        scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
        scope_stats_end(0, 1, 3, 1024, 3072, 1, 3, 4);
    }
    scope_stats_aicpu_flush_buffers();
}

uint64_t epoch_of_current_buffer(void *shm) {
    auto *state = get_scope_stats_buffer_state(shm, 0);
    auto *buf = reinterpret_cast<const ScopeStatsBuffer *>(state->current_buf_ptr);
    return buf != nullptr ? buf->run_epoch : 0;
}

uint64_t published_buffer_at(void *shm, uint32_t index) {
    return get_scope_stats_header(shm)->queues[0][index].buffer_ptr;
}

}  // namespace

TEST(ScopeStatsRunIdentityTest, LaggingConsumerKeepsEachRunsTailBufferIntact) {
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, test_alloc, nullptr, test_free, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();
    auto *state = get_scope_stats_buffer_state(shm, 0);

    // Run 1 produces one begin/end pair — far short of a full buffer — and
    // flushes it to the ready queue. Nothing consumes it.
    produce_run(/*run_epoch=*/11, shm, /*n=*/1, "run_one.cpp");
    const uint32_t tail_after_first = get_scope_stats_header(shm)->queue_tails[0];
    ASSERT_EQ(tail_after_first, 1u) << "run 1's tail buffer never reached the ready queue";
    const uint64_t first_buf = published_buffer_at(shm, 0);
    ASSERT_NE(first_buf, 0u);
    const auto *published = reinterpret_cast<const ScopeStatsBuffer *>(first_buf);

    EXPECT_EQ(published->run_epoch, 11u);
    EXPECT_EQ(published->count, 2u);
    EXPECT_EQ(state->current_buf_ptr, 0u) << "flush must release the buffer it handed to the consumer";

    // Run 2 acquires while run 1's buffer is still unconsumed, repeating the
    // same site keys on purpose.
    produce_run(/*run_epoch=*/22, shm, /*n=*/1, "run_one.cpp");

    const uint32_t tail_after_second = get_scope_stats_header(shm)->queue_tails[0];
    ASSERT_EQ(tail_after_second, 2u);
    const uint64_t second_buf = published_buffer_at(shm, 1);
    EXPECT_NE(second_buf, first_buf) << "a successor took a buffer the consumer still holds";
    EXPECT_EQ(reinterpret_cast<const ScopeStatsBuffer *>(second_buf)->run_epoch, 22u);
    // Run 1's payload is untouched by run 2.
    EXPECT_EQ(published->run_epoch, 11u);
    EXPECT_EQ(published->count, 2u);

    collector.finalize(nullptr, test_free);
}

TEST(ScopeStatsRunIdentityTest, HostCopyKeepsItsRunAfterTheBufferIsReused) {
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, test_alloc, nullptr, test_free, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();

    // Leave a partial buffer as the *current* one (no flush) — the shape
    // reconcile_counters() recovers host-side.
    set_platform_run_result(0, 33);
    set_scope_stats_enabled(true);
    set_platform_scope_stats_base(reinterpret_cast<uint64_t>(shm));
    scope_stats_aicpu_set_orch_thread_idx(0);
    scope_stats_set_pending_site("reuse.cpp", 7);
    scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
    ASSERT_EQ(epoch_of_current_buffer(shm), 33u) << "acquiring a buffer must stamp the acquiring run";

    auto *state = get_scope_stats_buffer_state(shm, 0);
    const uint64_t reused_buf = state->current_buf_ptr;
    EXPECT_FALSE(collector.reconcile_counters());
    ASSERT_EQ(collector.collected_for_run(33), 1u);

    // Hand the storage back and let a later run re-stamp the very same buffer.
    state->current_buf_ptr = 0;
    auto *buf = reinterpret_cast<ScopeStatsBuffer *>(reused_buf);
    buf->count = 0;
    buf->run_epoch = 44;

    EXPECT_EQ(collector.collected_for_run(33), 1u);
    EXPECT_EQ(collector.collected_for_run(44), 0u);
    const auto records = collector.collected_records();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].run_epoch, 33u);

    collector.finalize(nullptr, test_free);
}

TEST(ScopeStatsRunIdentityTest, TwoPartialRunsAccountWithoutACapacityTerm) {
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(1, test_alloc, nullptr, test_free, 0), 0);
    void *shm = collector.get_scope_stats_shm_device_ptr();
    auto *state = get_scope_stats_buffer_state(shm, 0);

    // One pair per run, so every buffer is far short of capacity — the case a
    // `total - seq * capacity` derivation cannot account for.
    produce_run(/*run_epoch=*/1, shm, /*n=*/1, "count_one.cpp");
    const uint32_t total_after_first = state->total_record_count;
    produce_run(/*run_epoch=*/2, shm, /*n=*/1, "count_two.cpp");

    EXPECT_EQ(total_after_first, 2u);
    EXPECT_EQ(state->total_record_count, 4u);
    EXPECT_EQ(state->dropped_record_count, 0u);

    // published + in-flight + dropped accounts for everything produced, with
    // no capacity term anywhere in the sum.
    const ScopeStatsDataHeader *header = get_scope_stats_header(shm);
    uint32_t published_records = 0;
    for (uint32_t i = 0; i < header->queue_tails[0]; i++) {
        published_records += reinterpret_cast<const ScopeStatsBuffer *>(published_buffer_at(shm, i))->count;
    }
    const uint32_t in_flight =
        state->current_buf_ptr != 0 ? reinterpret_cast<const ScopeStatsBuffer *>(state->current_buf_ptr)->count : 0;
    EXPECT_EQ(published_records + in_flight + state->dropped_record_count, state->total_record_count);

    collector.finalize(nullptr, test_free);
}
