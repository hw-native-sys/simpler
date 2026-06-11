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
 * Regression tests for the scope_stats collector's monotonic heap accounting
 * (issue #996).
 *
 * The collector samples the heap ring's reclaim/alloc pointers as wrapping byte
 * offsets in [0, heap_cap). Subtracting two boundary snapshots loses every wrap
 * in between, so before the fix a scope whose heap throughput exceeded heap_cap
 * reported wrong scope_alloc / high-water deltas (and they were unrecoverable
 * from the two wrapped samples). The allocator now reports each wrap via
 * scope_stats_note_heap_wrap(side); the collector unrolls the sampled offsets
 * into monotonic cumulative bytes so every host-side delta is exact for any
 * wrap count.
 *
 * These tests drive the collector directly (no device): they feed
 * scope_stats_begin / note_heap_wrap / scope_stats_end and assert the unrolled
 * heap_start/heap_end committed to the buffer, reproducing the multi-wrap case
 * #996 asked for. The metric expressions mirror simpler_setup/tools/
 * scope_stats_plot.py exactly (scope_alloc = end.head - begin.head, etc.).
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "aicpu/scope_stats_collector_aicpu.h"
#include "common/scope_stats.h"
// Collector compiled in here (not linked via CMake).
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "../../../../src/common/platform/shared/aicpu/scope_stats_collector_aicpu.cpp"

namespace {

constexpr uint64_t kHeapCap = 4096;  // small cap so a few KB of throughput wraps it.
constexpr int kRing = 0;

// Drives one collector instance over a host-owned shared region plus a single
// free buffer, mirroring what the host collector hands the device at init.
class ScopeStatsCollectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        size_t shm_size = calc_scope_stats_shm_size(/*num_instances=*/1);
        base_ = std::aligned_alloc(64, shm_size);
        buf_ = static_cast<ScopeStatsBuffer *>(std::aligned_alloc(64, sizeof(ScopeStatsBuffer)));
        std::memset(base_, 0, shm_size);
        std::memset(buf_, 0, sizeof(ScopeStatsBuffer));

        set_scope_stats_enabled(true);
        set_platform_scope_stats_base(reinterpret_cast<uint64_t>(base_));
        scope_stats_aicpu_set_orch_thread_idx(0);
        scope_stats_set_ring_capacity(kRing, /*window_cap=*/8, kHeapCap, /*dep_pool_cap=*/1024);

        // Hand the device one free buffer to pop into (host is the producer).
        ScopeStatsBufferState *state = get_scope_stats_buffer_state(base_, 0);
        state->free_queue.buffer_ptrs[0] = reinterpret_cast<uint64_t>(buf_);
        state->free_queue.head = 0;
        state->free_queue.tail = 1;
    }

    void TearDown() override {
        set_scope_stats_enabled(false);
        set_platform_scope_stats_base(0);
        std::free(base_);
        std::free(buf_);
    }

    const ScopeStatsRecord &record(uint32_t i) const { return buf_->records[i]; }

    void *base_ = nullptr;
    ScopeStatsBuffer *buf_ = nullptr;
};

// A scope that wraps the alloc pointer twice and the reclaim pointer once must
// report monotonic cumulative bytes, so scope_alloc recovers the full
// throughput even though it exceeds heap_cap. This is the case #996 reported.
TEST_F(ScopeStatsCollectorTest, MultiWrapScopeUnrollsToMonotonicBytes) {
    // Total over the scope: 10000 bytes allocated, 8000 reclaimed.
    //   alloc:   10000 = 2 * 4096 + 1808  -> 2 wraps, raw offset 1808
    //   reclaim:  8000 = 1 * 4096 + 3904  -> 1 wrap,  raw offset 3904
    constexpr uint64_t kAllocRaw = 1808;
    constexpr uint64_t kReclaimRaw = 3904;

    scope_stats_set_pending_site("test_scope_stats_collector.cpp", __LINE__);
    scope_stats_begin(kRing, 0, 0, /*heap_start=*/0, /*heap_end=*/0, 0, 0, 0);

    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_RECLAIM);

    scope_stats_end(kRing, 0, 0, /*heap_start=*/kReclaimRaw, /*heap_end=*/kAllocRaw, 0, 0, 0);

    ASSERT_EQ(buf_->count, 2u);
    const ScopeStatsRecord &begin = record(0);
    const ScopeStatsRecord &end = record(1);

    // Unrolled to monotonic cumulative bytes (raw + wraps * cap).
    EXPECT_EQ(begin.heap_start, 0u);
    EXPECT_EQ(begin.heap_end, 0u);
    EXPECT_EQ(end.heap_start, 8000u);
    EXPECT_EQ(end.heap_end, 10000u);

    // Metric expressions mirror scope_stats_plot.py.
    uint64_t scope_alloc = end.heap_end - begin.heap_end;
    uint64_t real_occupancy = end.heap_end - end.heap_start;
    uint64_t high_water = end.heap_end - begin.heap_start;

    EXPECT_EQ(scope_alloc, 10000u);
    EXPECT_GT(scope_alloc, kHeapCap);  // the multi-wrap throughput the old code lost.
    EXPECT_EQ(real_occupancy, 2000u);
    EXPECT_LE(real_occupancy, kHeapCap);
    EXPECT_EQ(high_water, 10000u);
}

// When the ring is exactly full, the raw offsets coincide (modular occupancy
// folds to 0) but the unrolled occupancy is heap_cap. The debug invariant
// assert must accept this instead of firing — exercised here because the UT
// build keeps assertions enabled (no NDEBUG).
TEST_F(ScopeStatsCollectorTest, FullRingOccupancyDoesNotTripInvariant) {
    // alloc 8192 (2 wraps, raw 0), reclaim 4096 (1 wrap, raw 0) -> occ == cap.
    scope_stats_set_pending_site("test_scope_stats_collector.cpp", __LINE__);
    scope_stats_begin(kRing, 0, 0, /*heap_start=*/0, /*heap_end=*/0, 0, 0, 0);

    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_RECLAIM);

    scope_stats_end(kRing, 0, 0, /*heap_start=*/0, /*heap_end=*/0, 0, 0, 0);

    ASSERT_EQ(buf_->count, 2u);
    const ScopeStatsRecord &end = record(1);
    EXPECT_EQ(end.heap_start, kHeapCap);                 // 1 * cap
    EXPECT_EQ(end.heap_end, 2 * kHeapCap);               // 2 * cap
    EXPECT_EQ(end.heap_end - end.heap_start, kHeapCap);  // full ring, exactly cap.
}

// A ring that never wraps must pass raw offsets through unchanged (the wrap
// accounting stays dormant), so a non-migrated path is unaffected.
TEST_F(ScopeStatsCollectorTest, NoWrapPassesOffsetsThrough) {
    scope_stats_set_pending_site("test_scope_stats_collector.cpp", __LINE__);
    scope_stats_begin(kRing, 0, 0, /*heap_start=*/0, /*heap_end=*/512, 0, 0, 0);
    scope_stats_end(kRing, 0, 0, /*heap_start=*/512, /*heap_end=*/2048, 0, 0, 0);

    ASSERT_EQ(buf_->count, 2u);
    EXPECT_EQ(record(0).heap_end, 512u);
    EXPECT_EQ(record(1).heap_start, 512u);
    EXPECT_EQ(record(1).heap_end, 2048u);
}

}  // namespace
