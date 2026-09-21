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
 * Unit tests for ChipTensorMap from tensormap.h / tensormap.cpp
 *
 * Tests hash-table-based producer lookup with overlap detection:
 * - Hash function distribution (golden-ratio multiplicative hash)
 * - Insert / lookup / cleanup lifecycle
 * - Overlap detection: fast-path (is_all_offset_zero) and slow-path (offsets)
 * - Lazy invalidation (stale entries skipped, not truncated)
 * - Multi-ring isolation in the same hash chain
 * - Lookup returns all matches (no silent 16-result cap post-#669)
 * - Entry pool allocation and free-list recycling
 * - cleanup_retired correctness across task windows
 */

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <vector>

#include "utils/device_arena.h"
#include "orchestration_api.h"
#include "tensormap.h"
#include "tensormap_and_ringbuffer/task_id.h"

// =============================================================================
// Helpers
// =============================================================================

// Test-local mirror of the old stack-buffered lookup result. PR #669 removed
// LookupResult in favor of a callback-based API; these tests collect
// matches into a vector-like struct so assertions remain readable.
struct TestLookupResult {
    struct Entry {
        ChipTensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    std::vector<Entry> entries;
    int count = 0;
};

static void run_lookup(ChipTensorMap &tmap, const simpler::tmr::Tensor &tensor, TestLookupResult &out) {
    tmap.lookup(tensor, [&](ChipTensorMapEntry &e, OverlapStatus s) -> bool {
        out.entries.push_back({&e, s});
        out.count++;
        return true;
    });
}

static simpler::tmr::Tensor make_test_tensor(uint64_t addr, uint32_t shape0, uint32_t ndims = 1, int32_t version = 0) {
    uint32_t shapes[MAX_TENSOR_DIMS] = {shape0};
    return simpler::tmr::make_tensor_external(
        reinterpret_cast<void *>(addr), shapes, ndims, DataType::FLOAT32, false, version
    );
}

static simpler::tmr::Tensor make_test_tensor_2d(uint64_t addr, uint32_t s0, uint32_t s1, int32_t version = 0) {
    uint32_t shapes[MAX_TENSOR_DIMS] = {s0, s1};
    return simpler::tmr::make_tensor_external(
        reinterpret_cast<void *>(addr), shapes, 2, DataType::FLOAT32, false, version
    );
}

// =============================================================================
// Fixture
// =============================================================================

class TensorMapTest : public ::testing::Test {
protected:
    static constexpr int32_t NUM_BUCKETS = 16;
    static constexpr int32_t POOL_SIZE = 64;
    static constexpr int32_t WINDOW_SIZE = 32;

    ChipTensorMap tmap{};
    DeviceArena arena;

    void SetUp() override {
        int32_t window_sizes[CHIP_MAX_RING_DEPTH] = {WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE};
        auto layout = ChipTensorMap::reserve_layout(arena, NUM_BUCKETS, POOL_SIZE, window_sizes);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(tmap.init_data_from_layout(layout, arena));
        tmap.wire_arena_pointers(layout, arena);
    }

    void TearDown() override {
        tmap.destroy();
        arena.release();
    }
};

// =============================================================================
// Initialization
// =============================================================================

TEST_F(TensorMapTest, InitValidState) {
    EXPECT_EQ(tmap.num_buckets, NUM_BUCKETS);
    EXPECT_EQ(tmap.pool_size, POOL_SIZE);
    EXPECT_EQ(tmap.next_entry_idx, 0);
    EXPECT_EQ(tmap.free_num, 0);
    EXPECT_EQ(tmap.valid_count(), 0);
}

TEST_F(TensorMapTest, InitWithPowerOfTwoBucketsSucceeds) {
    // The reject path for non-power-of-2 bucket counts is enforced via an
    // always_assert inside reserve_layout. It is not asserted here because
    // EXPECT_DEATH cannot run reliably in release builds where always_assert
    // may compile out. Cover only the accepted (power-of-2) shape.
    ChipTensorMap ok{};
    DeviceArena ok_arena;
    int32_t ws[CHIP_MAX_RING_DEPTH] = {8, 8, 8, 8};
    auto layout = ChipTensorMap::reserve_layout(ok_arena, 8, 64, ws);
    ASSERT_NE(ok_arena.commit(), nullptr);
    EXPECT_TRUE(ok.init_data_from_layout(layout, ok_arena));
    ok.wire_arena_pointers(layout, ok_arena);
    ok.destroy();
}

// =============================================================================
// Hash function
// =============================================================================

TEST_F(TensorMapTest, HashDeterministic) {
    uint64_t addr = 0x1000;
    EXPECT_EQ(tmap.hash(addr), tmap.hash(addr));
}

TEST_F(TensorMapTest, HashDistributesAlignedAddresses) {
    std::set<uint32_t> hit_buckets;
    // Aligned addresses (64KB stride) should still distribute across buckets
    for (uint64_t i = 0; i < 64; i++) {
        uint64_t addr = i * 65536;
        hit_buckets.insert(tmap.hash(addr));
    }
    // With golden-ratio hash, 64 aligned addresses across 16 buckets
    // should hit at least 12 distinct buckets
    EXPECT_GE(hit_buckets.size(), 12u) << "Aligned addresses must distribute well";
}

TEST_F(TensorMapTest, HashBoundedByBucketCount) {
    for (uint64_t addr = 0; addr < 1000; addr++) {
        EXPECT_LT(tmap.hash(addr), static_cast<uint32_t>(NUM_BUCKETS));
    }
}

// =============================================================================
// Insert and lookup: basic
// =============================================================================

TEST_F(TensorMapTest, InsertThenLookupFindsProducer) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    TaskId tid = TaskId::make(0, 0);
    tmap.insert(t, tid);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, tid);
}

TEST_F(TensorMapTest, LookupEmptyReturnsZero) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    EXPECT_EQ(result.count, 0);
}

TEST_F(TensorMapTest, InsertMultipleSameBuffer) {
    simpler::tmr::Tensor t1 = make_test_tensor(0x1000, 256);
    simpler::tmr::Tensor t2 = make_test_tensor(0x1000, 128);
    TaskId tid1 = TaskId::make(0, 0);
    TaskId tid2 = TaskId::make(0, 1);

    tmap.insert(t1, tid1);
    tmap.insert(t2, tid2);

    TestLookupResult result;
    run_lookup(tmap, t1, result);
    // Both entries share same buffer_addr, so both should be found
    EXPECT_EQ(result.count, 2);
}

TEST_F(TensorMapTest, InsertDifferentBuffersNoCollision) {
    simpler::tmr::Tensor t1 = make_test_tensor(0x1000, 256);
    simpler::tmr::Tensor t2 = make_test_tensor(0x2000, 256);
    tmap.insert(t1, TaskId::make(0, 0));
    tmap.insert(t2, TaskId::make(0, 1));

    TestLookupResult r1;
    run_lookup(tmap, t1, r1);
    EXPECT_EQ(r1.count, 1);
    EXPECT_EQ(r1.entries[0].entry->producer_task_id, TaskId::make(0, 0));

    TestLookupResult r2;
    run_lookup(tmap, t2, r2);
    EXPECT_EQ(r2.count, 1);
    EXPECT_EQ(r2.entries[0].entry->producer_task_id, TaskId::make(0, 1));
}

// =============================================================================
// Overlap detection: fast path (is_all_offset_zero)
// =============================================================================

TEST_F(TensorMapTest, OverlapFastPathCovered) {
    // Producer output: shape [256], consumer input: shape [512]
    // Consumer covers producer -> COVERED
    simpler::tmr::Tensor producer = make_test_tensor(0x1000, 256);
    simpler::tmr::Tensor consumer = make_test_tensor(0x1000, 512);
    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);
}

TEST_F(TensorMapTest, OverlapFastPathOther) {
    // Producer output: shape [512], consumer input: shape [256]
    // Consumer does NOT cover producer -> OTHER
    simpler::tmr::Tensor producer = make_test_tensor(0x1000, 512);
    simpler::tmr::Tensor consumer = make_test_tensor(0x1000, 256);
    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

TEST_F(TensorMapTest, OverlapFastPathExactMatch) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);
}

// =============================================================================
// Overlap detection: slow path (offsets via view)
// =============================================================================

TEST_F(TensorMapTest, OverlapSlowPathNoOverlap) {
    // Producer writes [0..128), consumer reads [128..256) -> NO_OVERLAP
    simpler::tmr::Tensor base = make_test_tensor_2d(0x1000, 256, 1);
    uint32_t prod_shapes[] = {128, 1};
    uint32_t prod_offsets[] = {0, 0};
    simpler::tmr::Tensor producer = base.view(prod_shapes, prod_offsets);

    uint32_t con_shapes[] = {128, 1};
    uint32_t con_offsets[] = {128, 0};
    simpler::tmr::Tensor consumer = base.view(con_shapes, con_offsets);

    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    EXPECT_EQ(result.count, 0) << "Non-overlapping regions must return no results";
}

TEST_F(TensorMapTest, OverlapSlowPathPartialOverlap) {
    // Producer writes [0..192), consumer reads [64..256) -> overlapping, OTHER
    simpler::tmr::Tensor base = make_test_tensor_2d(0x1000, 256, 1);
    uint32_t prod_shapes[] = {192, 1};
    uint32_t prod_offsets[] = {0, 0};
    simpler::tmr::Tensor producer = base.view(prod_shapes, prod_offsets);

    uint32_t con_shapes[] = {192, 1};
    uint32_t con_offsets[] = {64, 0};
    simpler::tmr::Tensor consumer = base.view(con_shapes, con_offsets);

    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

TEST_F(TensorMapTest, OverlapSlowPathCovered) {
    // Producer writes [64..192), consumer reads [0..256) -> consumer covers producer
    simpler::tmr::Tensor base = make_test_tensor_2d(0x1000, 256, 1);
    uint32_t prod_shapes[] = {128, 1};
    uint32_t prod_offsets[] = {64, 0};
    simpler::tmr::Tensor producer = base.view(prod_shapes, prod_offsets);

    uint32_t con_shapes[] = {256, 1};
    uint32_t con_offsets[] = {0, 0};
    simpler::tmr::Tensor consumer = base.view(con_shapes, con_offsets);

    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::COVERED);
}

// =============================================================================
// Version-based overlap detection
// =============================================================================

TEST_F(TensorMapTest, VersionMismatchReturnsOther) {
    // Producer v0, consumer v1 -> always OTHER regardless of shape match
    simpler::tmr::Tensor producer = make_test_tensor(0x1000, 256, 1, 0);
    simpler::tmr::Tensor consumer = make_test_tensor(0x1000, 256, 1, 1);

    tmap.insert(producer, TaskId::make(0, 0));

    TestLookupResult result;
    run_lookup(tmap, consumer, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].overlap_status, OverlapStatus::OTHER);
}

// =============================================================================
// Lazy invalidation
// =============================================================================

TEST_F(TensorMapTest, StaleEntriesSkippedDuringLookup) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 0));
    tmap.insert(t, TaskId::make(0, 1));

    // Advance validity to skip task 0
    tmap.sync_validity(0, 1);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make(0, 1));
}

TEST_F(TensorMapTest, StaleEntriesNotTruncatedAcrossRings) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    // Ring 0, task 0 and Ring 1, task 0 -> same bucket
    tmap.insert(t, TaskId::make(0, 0));
    tmap.insert(t, TaskId::make(1, 0));

    // Invalidate ring 0 only
    tmap.sync_validity(0, 1);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    // Ring 1 task 0 still valid, ring 0 task 0 invalidated
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make(1, 0));
}

// =============================================================================
// cleanup_retired
// =============================================================================

TEST_F(TensorMapTest, CleanupRetiredRemovesEntriesForRetiredTasks) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 0));
    tmap.insert(t, TaskId::make(0, 1));
    tmap.insert(t, TaskId::make(0, 2));
    EXPECT_EQ(tmap.valid_count(), 3);

    // Cleanup tasks [0, 2) on ring 0
    tmap.cleanup_retired(0, 0, 2);

    EXPECT_EQ(tmap.valid_count(), 1);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make(0, 2));
}

TEST_F(TensorMapTest, CleanupRetiredPreservesOtherRings) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 0));
    tmap.insert(t, TaskId::make(1, 0));

    tmap.cleanup_retired(0, 0, 1);

    EXPECT_EQ(tmap.valid_count(), 1);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make(1, 0));
}

TEST_F(TensorMapTest, CleanupRetiredFreesEntriesToPool) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 0));
    EXPECT_EQ(tmap.free_num, 0);
    EXPECT_EQ(tmap.next_entry_idx, 1);

    tmap.cleanup_retired(0, 0, 1);

    EXPECT_EQ(tmap.free_num, 1) << "Cleaned entry should be in free list";

    // New insert should reuse free entry instead of allocating fresh
    tmap.insert(t, TaskId::make(0, 1));
    EXPECT_EQ(tmap.free_num, 0);
    EXPECT_EQ(tmap.next_entry_idx, 1) << "Should reuse freed entry, not allocate new";
}

// A later task that reuses a slot (local_id + WINDOW_SIZE) before cleanup has
// run on the earlier task chains its entries under the same task_entry_head.
// cleanup_retired retiring only the earlier task must free that earlier task's
// entries alone and leave the still-live (later) task's entries intact.
TEST_F(TensorMapTest, CleanupRetiredSparesLaterTaskReusingSlot) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    // Task 0 and task 0 + WINDOW_SIZE share slot 0 (local_id & (WINDOW_SIZE-1)).
    tmap.insert(t, TaskId::make(0, 0));
    tmap.insert(t, TaskId::make(0, WINDOW_SIZE));
    ASSERT_EQ(tmap.valid_count(), 2);

    // Retire only task 0.
    tmap.cleanup_retired(0, 0, 1);

    // Only task 0's entry is freed; task WINDOW_SIZE's entry survives.
    EXPECT_EQ(tmap.valid_count(), 1);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make(0, WINDOW_SIZE));
}

// =============================================================================
// Multi-ring isolation
// =============================================================================

TEST_F(TensorMapTest, MultiRingIndependentLookup) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make(0, 5));
    tmap.insert(t, TaskId::make(1, 3));
    tmap.insert(t, TaskId::make(2, 7));

    TestLookupResult result;
    run_lookup(tmap, t, result);
    EXPECT_EQ(result.count, 3);

    // Invalidate ring 0 up to task 6 and ring 2 up to task 8
    tmap.sync_validity(0, 6);
    tmap.sync_validity(2, 8);

    TestLookupResult result2;
    run_lookup(tmap, t, result2);
    EXPECT_EQ(result2.count, 1);
    EXPECT_EQ(result2.entries[0].entry->producer_task_id, TaskId::make(1, 3));
}

// =============================================================================
// Lookup returns all matches (PR #669 removed the 16-slot cap)
// =============================================================================

TEST_F(TensorMapTest, LookupReturnsAllMatches) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    // Insert 20 entries for the same buffer (was capped at 16 before #669)
    for (int i = 0; i < 20; i++) {
        tmap.insert(t, TaskId::make(0, i));
    }

    TestLookupResult result;
    run_lookup(tmap, t, result);
    EXPECT_EQ(result.count, 20) << "Lookup must return every overlapping entry, no silent cap";
}

// =============================================================================
// Entry pool lifecycle
// =============================================================================

TEST_F(TensorMapTest, PoolExhaustionAsserts) {
    // With pool_size=64, inserting 64 entries should work, 65th should fail
    for (int i = 0; i < POOL_SIZE; i++) {
        simpler::tmr::Tensor t = make_test_tensor(0x1000 + i * 0x100, 256);
        tmap.insert(t, TaskId::make(0, i));
    }
    EXPECT_EQ(tmap.next_entry_idx, POOL_SIZE);
    EXPECT_EQ(tmap.free_num, 0);

    // 65th insert should trigger always_assert (pool overflow)
    simpler::tmr::Tensor overflow = make_test_tensor(0x9000, 256);
    EXPECT_THROW(tmap.insert(overflow, TaskId::make(0, POOL_SIZE)), std::runtime_error);
}

TEST_F(TensorMapTest, FreeListRecycling) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    // Insert and cleanup 10 entries
    for (int i = 0; i < 10; i++) {
        tmap.insert(t, TaskId::make(0, i));
    }
    tmap.cleanup_retired(0, 0, 10);
    EXPECT_EQ(tmap.free_num, 10);

    // Re-insert should use free list
    for (int i = 10; i < 20; i++) {
        tmap.insert(t, TaskId::make(0, i));
    }
    EXPECT_EQ(tmap.free_num, 0);
    EXPECT_EQ(tmap.next_entry_idx, 10) << "No new pool entries consumed when free list available";
}

// =============================================================================
// Task chain integrity (per-task entry list)
// =============================================================================

TEST_F(TensorMapTest, PerTaskEntryListTracksMultipleOutputs) {
    simpler::tmr::Tensor t1 = make_test_tensor(0x1000, 256);
    simpler::tmr::Tensor t2 = make_test_tensor(0x2000, 128);
    TaskId tid = TaskId::make(0, 5);

    tmap.insert(t1, tid);
    tmap.insert(t2, tid);
    EXPECT_EQ(tmap.valid_count(), 2);

    // Cleanup task 5 should remove both entries
    tmap.cleanup_retired(0, 5, 6);
    EXPECT_EQ(tmap.valid_count(), 0);
    EXPECT_EQ(tmap.free_num, 2);
}

// =============================================================================
// Bucket chain integrity (doubly-linked list)
// =============================================================================

TEST_F(TensorMapTest, RemoveMiddleEntryPreservesChain) {
    simpler::tmr::Tensor t = make_test_tensor(0x1000, 256);
    TaskId tid0 = TaskId::make(0, 0);
    TaskId tid1 = TaskId::make(0, 1);
    TaskId tid2 = TaskId::make(0, 2);

    tmap.insert(t, tid0);
    tmap.insert(t, tid1);
    tmap.insert(t, tid2);

    // Remove middle entry (task 1)
    tmap.cleanup_retired(0, 1, 2);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    EXPECT_EQ(result.count, 2);

    std::set<uint32_t> found_locals;
    for (int i = 0; i < result.count; i++) {
        found_locals.insert(result.entries[i].entry->producer_task_id.local_id());
    }
    EXPECT_TRUE(found_locals.count(0));
    EXPECT_TRUE(found_locals.count(2));
}

// =============================================================================
// TaskId encoding/decoding
// =============================================================================

TEST(TaskIdTest, MakeAndDecode) {
    auto tid = TaskId::make(3, 42);
    EXPECT_EQ(tid.ring(), 3);
    EXPECT_EQ(tid.local_id(), 42u);
}

TEST(TaskIdTest, InvalidSentinel) {
    auto inv = TaskId::invalid();
    EXPECT_FALSE(inv.is_valid());
    EXPECT_EQ(inv.raw, UINT64_MAX);
}

TEST(TaskIdTest, Equality) {
    auto a = TaskId::make(1, 100);
    auto b = TaskId::make(1, 100);
    auto c = TaskId::make(2, 100);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(TaskIdTest, RingIdMaxValue) {
    auto tid = TaskId::make(255, 0);
    EXPECT_EQ(tid.ring(), 255);
    EXPECT_EQ(tid.local_id(), 0u);
}

TEST(TaskIdTest, LocalIdMaxValue) {
    auto tid = TaskId::make(0, UINT32_MAX);
    EXPECT_EQ(tid.ring(), 0);
    EXPECT_EQ(tid.local_id(), UINT32_MAX);
}
