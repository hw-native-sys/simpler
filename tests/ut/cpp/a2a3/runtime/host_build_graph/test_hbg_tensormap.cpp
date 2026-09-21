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
#include <vector>

#include "host_build_graph/tensormap.h"
#include "host_build_graph/task_id.h"

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

// Filling the pool drives free_entries() to zero. No device-completion watermark
// can free it, so ensure_tensormap_capacity() must fail immediately instead of
// waiting for asynchronous reclaim that HBG does not have.
TEST_F(HbgTensorMapTest, ExhaustedPoolStaysExhausted) {
    for (int32_t i = 0; i < POOL_SIZE; i++) {
        tmap.insert(make_test_tensor(0x10000 + 0x100 * i, 64), TaskId::make_global(i % MAX_TASKS));
    }
    EXPECT_EQ(tmap.current_used(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), 0);
    EXPECT_EQ(tmap.valid_count(), POOL_SIZE);
}

}  // namespace
