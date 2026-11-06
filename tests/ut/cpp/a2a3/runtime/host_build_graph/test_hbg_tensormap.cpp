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
 * Entry-lifetime tests for the host_build_graph copy of ChipTensorMap.
 *
 * This is a distinct type from the tensormap_and_ringbuffer ChipTensorMap
 * covered by a2a3/test_tensormap.cpp: single-ring, no entry epochs, and — the
 * subject of this file — no completion-watermark retirement. A registered
 * output stays visible until dependency computation explicitly removes it as
 * semantically covered; time and task-slot aliases alone never invalidate it.
 * The hash / overlap surface is shared logic already exercised by that suite.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <new>
#include <vector>

#include "host_build_graph/host/tensormap.h"
#include "host_build_graph/task_id.h"

namespace {
// Only nothrow array allocations in this executable participate. The counter
// is disarmed before gtest reports an assertion or allocates its diagnostics.
int fail_array_after = -1;
bool fail_array_allocation() {
    if (fail_array_after < 0) return false;
    return fail_array_after-- == 0;
}
}  // namespace

void *operator new[](std::size_t bytes, const std::nothrow_t &) noexcept {
    if (fail_array_allocation()) return nullptr;
    try {
        return ::operator new[](bytes);
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

void *operator new[](std::size_t bytes, std::align_val_t alignment, const std::nothrow_t &) noexcept {
    if (fail_array_allocation()) return nullptr;
    try {
        return ::operator new[](bytes, alignment);
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

namespace {

struct TestLookupResult {
    struct Entry {
        ChipTensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    std::vector<Entry> entries;
    int count = 0;
};

void run_lookup(ChipTensorMap &tmap, const simpler::hbg::Tensor &tensor, TestLookupResult &out) {
    tmap.lookup(tensor, [&](ChipTensorMapEntry &e, OverlapStatus s) -> bool {
        out.entries.push_back({&e, s});
        out.count++;
        return true;
    });
}

simpler::hbg::Tensor make_test_tensor(uint64_t addr, uint32_t shape0) {
    uint32_t shapes[MAX_TENSOR_DIMS] = {shape0};
    return simpler::hbg::make_tensor_external(reinterpret_cast<void *>(addr), shapes, 1, DataType::FLOAT32, false, 0);
}

class HbgTensorMapTest : public ::testing::Test {
protected:
    static constexpr int32_t NUM_BUCKETS = 16;
    static constexpr int32_t POOL_SIZE = 64;
    // Task chains this map is dimensioned for. A local id is its own chain index
    // here, so it is also the exclusive upper bound on a producer id these tests
    // may insert under.
    static constexpr int32_t MAX_TASKS = 64;
    static constexpr int32_t LAST_TASK = MAX_TASKS - 1;

    ChipTensorMap tmap{};

    void SetUp() override { ASSERT_TRUE(tmap.init(NUM_BUCKETS, POOL_SIZE, MAX_TASKS)); }
};

// Completion progress alone cannot make an older producer disappear. Direct
// inserts remain visible until dependency computation explicitly removes one.
TEST_F(HbgTensorMapTest, EveryProducerOfARegionStaysVisible) {
    simpler::hbg::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make_global(0));
    tmap.insert(t, TaskId::make_global(1));
    tmap.insert(t, TaskId::make_global(2));
    EXPECT_EQ(tmap.valid_count(), 3);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 3);
    std::vector<TaskId> producers;
    producers.reserve(result.entries.size());
    for (const auto &e : result.entries) {
        producers.push_back(e.entry->producer_task_id);
    }
    EXPECT_NE(std::find(producers.begin(), producers.end(), TaskId::make_global(0)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), TaskId::make_global(1)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), TaskId::make_global(2)), producers.end());
}

// A local id is its own task chain index -- nothing masks with the chain count -- so
// two distinct producers never share a chain, and the highest id the map is
// dimensioned for reaches its own chain rather than folding onto a lower one.
TEST_F(HbgTensorMapTest, DistinctLocalIdsGetDistinctTaskChains) {
    simpler::hbg::Tensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, TaskId::make_global(0));
    tmap.insert(t, TaskId::make_global(LAST_TASK));

    EXPECT_EQ(tmap.valid_count(), 2);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 2);
    std::vector<TaskId> producers;
    producers.reserve(result.entries.size());
    for (const auto &e : result.entries) {
        producers.push_back(e.entry->producer_task_id);
    }
    EXPECT_NE(std::find(producers.begin(), producers.end(), TaskId::make_global(0)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), TaskId::make_global(LAST_TASK)), producers.end());

    // Each producer's chain head is its own entry, and LAST_TASK reaches the last
    // reserved chain rather than folding onto a lower one. Read the heads directly:
    // an entry count cannot tell one chain per producer from one shared chain.
    ASSERT_NE(tmap.task_entry_heads[0], nullptr);
    ASSERT_NE(tmap.task_entry_heads[LAST_TASK], nullptr);
    EXPECT_EQ(tmap.task_entry_heads[0]->producer_task_id, TaskId::make_global(0));
    EXPECT_EQ(tmap.task_entry_heads[LAST_TASK]->producer_task_id, TaskId::make_global(LAST_TASK));

    // Unlinking a chain's only entry empties that chain and leaves the other intact.
    tmap.remove_entry(*tmap.task_entry_heads[LAST_TASK]);
    EXPECT_EQ(tmap.valid_count(), 1);
    EXPECT_EQ(tmap.task_entry_heads[LAST_TASK], nullptr);
    ASSERT_NE(tmap.task_entry_heads[0], nullptr);
    EXPECT_EQ(tmap.task_entry_heads[0]->producer_task_id, TaskId::make_global(0));
}

// Without an explicit semantic removal, direct inserts consume one pool entry
// each. free_entries() is what the orchestrator's pre-registration capacity
// check reads, so it must track those inserts exactly.
TEST_F(HbgTensorMapTest, PoolOccupancyOnlyGrows) {
    EXPECT_EQ(tmap.current_used(), 0);
    EXPECT_EQ(tmap.pool_capacity(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), POOL_SIZE);

    for (int32_t i = 0; i < 8; i++) {
        tmap.insert(make_test_tensor(0x1000 + 0x100 * i, 64), TaskId::make_global(i));
        EXPECT_EQ(tmap.current_used(), i + 1);
        EXPECT_EQ(tmap.free_entries(), POOL_SIZE - (i + 1));
    }
}

// A recorder thread empties its map between bodies instead of allocating a new one, so
// reset() must leave it indistinguishable from a freshly init'd map: nothing found, the
// whole pool free, and the next body's inserts starting from entry 0. A leftover bucket
// chain here would give the next Definition a producer the body never had.
TEST_F(HbgTensorMapTest, ResetLeavesTheMapAsFreshlyInitialized) {
    simpler::hbg::Tensor t = make_test_tensor(0x1000, 256);
    for (int32_t i = 0; i < 8; i++) {
        tmap.insert(make_test_tensor(0x1000 + 0x100 * i, 64), TaskId::make_global(i));
    }
    tmap.insert(t, TaskId::make_global(9));
    ASSERT_EQ(tmap.current_used(), 9);

    tmap.reset();

    EXPECT_EQ(tmap.current_used(), 0);
    EXPECT_EQ(tmap.valid_count(), 0);
    EXPECT_EQ(tmap.free_entries(), POOL_SIZE);
    EXPECT_EQ(tmap.pool_capacity(), POOL_SIZE);
    TestLookupResult after_reset;
    run_lookup(tmap, t, after_reset);
    EXPECT_EQ(after_reset.count, 0);

    // And it is usable, not merely empty: the second body's producer is the only one a
    // lookup can reach.
    tmap.insert(t, TaskId::make_global(3));
    EXPECT_EQ(tmap.current_used(), 1);
    TestLookupResult second_body;
    run_lookup(tmap, t, second_body);
    ASSERT_EQ(second_body.count, 1);
    EXPECT_EQ(second_body.entries[0].entry->producer_task_id, TaskId::make_global(3));
}

// Reset is reachable any number of times, including on a map that was never inserted
// into, and does not depend on how many task chains it reserved -- it keeps the sizes
// init() reserved.
TEST_F(HbgTensorMapTest, ResetIsIdempotentAndKeepsReservedSizes) {
    tmap.reset();
    tmap.reset();
    EXPECT_EQ(tmap.pool_capacity(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), POOL_SIZE);

    // The last reserved chain is still a working chain after two resets.
    simpler::hbg::Tensor t = make_test_tensor(0x2000, 128);
    tmap.insert(t, TaskId::make_global(LAST_TASK));
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make_global(LAST_TASK));
}

// Filling the current chunks leaves every producer visible. Device completion
// cannot reclaim them; output registration must explicitly request stable growth
// rather than wait for asynchronous reclaim.
TEST_F(HbgTensorMapTest, ExhaustedPoolStaysExhausted) {
    for (int32_t i = 0; i < POOL_SIZE; i++) {
        tmap.insert(make_test_tensor(0x10000 + 0x100 * i, 64), TaskId::make_global(i % MAX_TASKS));
    }
    EXPECT_EQ(tmap.current_used(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), 0);
    EXPECT_EQ(tmap.valid_count(), POOL_SIZE);
}

}  // namespace

TEST(HbgTensorMapReuse, ClearsReachabilityWithoutReconstructingEntries) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init(16, 64, 32));
    auto *entries = map.entry_pool.get();
    auto *entry = map.new_entry();
    entry->producer_task_id = TaskId::make_global(7);
    map.buckets[2] = entry;
    map.task_entry_heads[7] = entry;
    ASSERT_TRUE(map.init(16, 64, 32));
    EXPECT_EQ(map.entry_pool.get(), entries);
    EXPECT_EQ(map.current_used(), 0);
    EXPECT_EQ(map.next_entry_idx, 0);
    EXPECT_EQ(map.free_num, 0);
    EXPECT_EQ(map.buckets[2], nullptr);
    EXPECT_EQ(map.task_entry_heads[7], nullptr);
    // Retention, not a default-construction pass over the unused entries.
    EXPECT_EQ(entries[0].producer_task_id, TaskId::make_global(7));
    ASSERT_TRUE(map.init(32, 128, 64));
    EXPECT_EQ(map.pool_capacity(), 128);
    EXPECT_EQ(map.max_tasks, 64);
    EXPECT_EQ(map.current_used(), 0);
}

TEST(HbgTensorMapReuse, TaskCapacityChangesKeepTheEntryPool) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init(16, 64, 32));
    auto *entries = map.entry_pool.get();
    auto *buckets = map.buckets.get();
    auto *free_list = map.free_entry_list.get();
    for (int capacity : {64, 8, 32, 64}) {
        ASSERT_TRUE(map.init(16, 64, capacity));
        EXPECT_EQ(map.entry_pool.get(), entries);
        ASSERT_EQ(map.buckets.get(), buckets);
        EXPECT_EQ(map.free_entry_list.get(), free_list);
        EXPECT_EQ(map.max_tasks, capacity);
        EXPECT_EQ(map.current_used(), 0);
    }
    auto *heads = map.task_entry_heads.get();
    fail_array_after = 0;
    const bool reused = map.init(16, 64, 8);
    fail_array_after = -1;
    ASSERT_TRUE(reused);
    EXPECT_EQ(map.task_entry_heads.get(), heads);
}

TEST(HbgTensorMapReuse, FailedResizePreservesTheMapAndCanBeRetried) {
    for (int failure = 0; failure < 4; ++failure) {
        ChipTensorMap map;
        ASSERT_TRUE(map.init(16, 64, 32));
        auto *entries = map.entry_pool.get();
        auto *buckets = map.buckets.get();
        auto *heads = map.task_entry_heads.get();
        auto *free_list = map.free_entry_list.get();
        auto tensor = make_test_tensor(0x1000, 64);
        map.insert(tensor, TaskId::make_global(7));
        fail_array_after = failure;
        const bool resized = map.init(32, 128, 64);
        fail_array_after = -1;
        ASSERT_FALSE(resized);
        EXPECT_EQ(map.entry_pool.get(), entries);
        ASSERT_EQ(map.buckets.get(), buckets);
        EXPECT_EQ(map.task_entry_heads.get(), heads);
        EXPECT_EQ(map.free_entry_list.get(), free_list);
        EXPECT_EQ(map.num_buckets, 16);
        EXPECT_EQ(map.pool_capacity(), 64);
        EXPECT_EQ(map.max_tasks, 32);
        TestLookupResult result;
        run_lookup(map, tensor, result);
        ASSERT_EQ(result.count, 1);
        EXPECT_EQ(result.entries[0].entry->producer_task_id, TaskId::make_global(7));
        ASSERT_TRUE(map.init(32, 128, 64));
        EXPECT_EQ(map.current_used(), 0);
        result = {};
        run_lookup(map, tensor, result);
        EXPECT_EQ(result.count, 0);
    }
}

TEST(HbgTensorMapReuse, FailedFirstAllocationLeavesAnEmptyOwnerAndCanBeRetried) {
    for (int failure = 0; failure < 4; ++failure) {
        ChipTensorMap map;
        fail_array_after = failure;
        const bool initialized = map.init(16, 64, 32);
        fail_array_after = -1;
        ASSERT_FALSE(initialized);
        EXPECT_EQ(map.buckets, nullptr);
        EXPECT_EQ(map.entry_pool, nullptr);
        EXPECT_EQ(map.free_entry_list, nullptr);
        EXPECT_EQ(map.task_entry_heads, nullptr);
        EXPECT_EQ(map.num_buckets, 0);
        EXPECT_EQ(map.pool_capacity(), 0);
        EXPECT_EQ(map.max_tasks, 0);
        ASSERT_TRUE(map.init(16, 64, 32));
        EXPECT_EQ(map.current_used(), 0);
    }
}

TEST(HbgTensorMapGrowth, ExtendsPastTheOldDefaultWithoutMovingProducers) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init_default(32));
    ASSERT_GT(map.pool_capacity(), 0);
    auto *first = map.entry_pool.get();
    const int32_t ENTRIES = map.pool_capacity() + 1;
    for (int32_t i = 0; i < ENTRIES; ++i) {
        ASSERT_TRUE(map.ensure_free_entries(1));
        map.insert(make_test_tensor(0x10000ULL + 0x100ULL * i, 64), TaskId::make_global(0));
    }
    EXPECT_EQ(map.entry_pool.get(), first);
    EXPECT_EQ(map.current_used(), ENTRIES);
    EXPECT_EQ(map.valid_count(), ENTRIES);
    TestLookupResult result;
    run_lookup(map, make_test_tensor(0x10000, 64), result);
    ASSERT_EQ(result.count, 1);
    EXPECT_EQ(result.entries[0].entry, first);
    EXPECT_EQ(first->producer_task_id, TaskId::make_global(0));
}

TEST(HbgTensorMapGrowth, PreservesTheFreeStackAcrossGrowth) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init(16, 4, 32));
    for (int32_t i = 0; i < 4; ++i) {
        map.insert(make_test_tensor(0x10000 + 0x100 * i, 64), TaskId::make_global(i));
    }
    auto *first = map.entry_pool.get();
    map.remove_from_task(first[0]);
    map.free_entry(first[0]);
    ASSERT_EQ(map.current_used(), 3);
    ASSERT_TRUE(map.ensure_free_entries(3));
    ASSERT_EQ(map.pool_capacity(), 8);
    for (int32_t i = 0; i < 3; ++i) {
        map.insert(make_test_tensor(0x20000 + 0x100 * i, 64), TaskId::make_global(8 + i));
    }
    EXPECT_EQ(map.current_used(), 6);
    EXPECT_EQ(map.valid_count(), 6);
    TestLookupResult reused;
    run_lookup(map, make_test_tensor(0x20000, 64), reused);
    ASSERT_EQ(reused.count, 1);
    EXPECT_EQ(reused.entries[0].entry, first);
    EXPECT_EQ(reused.entries[0].entry->producer_task_id, TaskId::make_global(8));
    TestLookupResult retained;
    run_lookup(map, make_test_tensor(0x10100, 64), retained);
    ASSERT_EQ(retained.count, 1);
    EXPECT_EQ(retained.entries[0].entry, first + 1);
}

TEST(HbgTensorMapGrowth, RetainsGrownChunksAcrossResetAndMatchingInit) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init(16, 4, 32));
    ASSERT_TRUE(map.ensure_free_entries(17));
    const int32_t capacity = map.pool_capacity();
    auto *entries = map.entry_pool.get();
    auto *free_list = map.free_entry_list.get();
    map.insert(make_test_tensor(0x10000, 64), TaskId::make_global(7));
    map.reset();
    fail_array_after = 0;
    const bool reinitialized = map.init(16, 4, 8);
    fail_array_after = -1;
    ASSERT_TRUE(reinitialized);
    EXPECT_EQ(map.pool_capacity(), capacity);
    EXPECT_EQ(map.entry_pool.get(), entries);
    EXPECT_EQ(map.free_entry_list.get(), free_list);
    EXPECT_EQ(map.current_used(), 0);
    EXPECT_EQ(map.valid_count(), 0);
    EXPECT_EQ(map.task_entry_heads[7], nullptr);
    for (int32_t i = 0; i < capacity; ++i) {
        map.insert(make_test_tensor(0x20000 + 0x100 * i, 64), TaskId::make_global(0));
    }
    EXPECT_EQ(map.current_used(), capacity);
    EXPECT_EQ(map.valid_count(), capacity);
}

TEST(HbgTensorMapGrowth, AllocationFailurePreservesAllReachableEntriesAndCanRetry) {
    // Pending chunk owners, entry chunk, directory and free-list allocation.
    for (int failure = 0; failure < 4; ++failure) {
        ChipTensorMap map;
        ASSERT_TRUE(map.init(16, 4, 32));
        for (int32_t i = 0; i < 4; ++i) {
            map.insert(make_test_tensor(0x10000 + 0x100 * i, 64), TaskId::make_global(i));
        }
        auto *entries = map.entry_pool.get();
        auto *free_list = map.free_entry_list.get();
        fail_array_after = failure;
        const bool grew = map.ensure_free_entries(1);
        fail_array_after = -1;
        ASSERT_FALSE(grew) << failure;
        EXPECT_EQ(map.pool_capacity(), 4);
        EXPECT_EQ(map.current_used(), 4);
        EXPECT_EQ(map.entry_pool.get(), entries);
        EXPECT_EQ(map.free_entry_list.get(), free_list);
        TestLookupResult result;
        run_lookup(map, make_test_tensor(0x10000, 64), result);
        ASSERT_EQ(result.count, 1);
        EXPECT_EQ(result.entries[0].entry, entries);
        EXPECT_EQ(map.task_entry_heads[0], entries);
        ASSERT_TRUE(map.ensure_free_entries(1));
        map.insert(make_test_tensor(0x20000, 64), TaskId::make_global(8));
        EXPECT_EQ(map.valid_count(), 5);
    }
}

TEST(HbgTensorMapGrowth, MultiChunkFailureDoesNotPartiallyPublishCapacity) {
    // Four added chunks plus pending owners, directory and free-list allocation.
    for (int failure = 0; failure < 7; ++failure) {
        ChipTensorMap map;
        ASSERT_TRUE(map.init(16, 4, 32));
        map.insert(make_test_tensor(0x10000, 64), TaskId::make_global(0));
        auto *entries = map.entry_pool.get();
        fail_array_after = failure;
        const bool grew = map.ensure_free_entries(16);
        fail_array_after = -1;
        ASSERT_FALSE(grew) << failure;
        EXPECT_EQ(map.pool_capacity(), 4);
        EXPECT_EQ(map.current_used(), 1);
        EXPECT_EQ(map.entry_pool.get(), entries);
        EXPECT_EQ(map.task_entry_heads[0], entries);
        ASSERT_TRUE(map.ensure_free_entries(16));
        EXPECT_GE(map.pool_capacity(), 17);
    }
}

TEST(HbgTensorMapGrowth, InvalidDemandDoesNotAllocateOrChangeState) {
    ChipTensorMap map;
    ASSERT_TRUE(map.init(16, 4, 32));
    map.insert(make_test_tensor(0x10000, 64), TaskId::make_global(0));
    fail_array_after = 0;
    const bool negative = map.ensure_free_entries(-1);
    const bool overflow = map.ensure_free_entries(std::numeric_limits<int32_t>::max());
    const bool available = map.ensure_free_entries(3);
    const int remaining_failure_counter = fail_array_after;
    fail_array_after = -1;
    EXPECT_FALSE(negative);
    EXPECT_FALSE(overflow);
    EXPECT_TRUE(available);
    EXPECT_EQ(remaining_failure_counter, 0);
    EXPECT_EQ(map.pool_capacity(), 4);
    EXPECT_EQ(map.current_used(), 1);
}
