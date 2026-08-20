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
 * Entry-lifetime tests for the host_build_graph copy of PTO2TensorMap.
 *
 * This is a distinct type from the tensormap_and_ringbuffer PTO2TensorMap
 * covered by a2a3/test_tensormap.cpp: single-ring, no entry epochs, and — the
 * subject of this file — no completion-watermark retirement. A registered
 * output stays visible until dependency computation explicitly removes it as
 * semantically covered; time and task-slot aliases alone never invalidate it.
 * The hash / overlap surface is shared logic already exercised by that suite.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "utils/device_arena.h"
#include "pto_tensormap.h"

namespace {

struct TestLookupResult {
    struct Entry {
        PTO2TensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    std::vector<Entry> entries;
    int count = 0;
};

void run_lookup(PTO2TensorMap &tmap, const ChipTensor &tensor, TestLookupResult &out) {
    tmap.lookup(tensor, [&](PTO2TensorMapEntry &e, OverlapStatus s) -> bool {
        out.entries.push_back({&e, s});
        out.count++;
        return true;
    });
}

ChipTensor make_test_tensor(uint64_t addr, uint32_t shape0) {
    uint32_t shapes[MAX_TENSOR_DIMS] = {shape0};
    return make_tensor_external(reinterpret_cast<void *>(addr), shapes, 1, DataType::FLOAT32, false, 0);
}

class HbgTensorMapTest : public ::testing::Test {
protected:
    static constexpr int32_t NUM_BUCKETS = 16;
    static constexpr int32_t POOL_SIZE = 64;
    static constexpr int32_t TASK_CAPACITY = 32;

    PTO2TensorMap tmap{};
    DeviceArena arena;

    void SetUp() override {
        auto layout = PTO2TensorMap::reserve_layout(arena, NUM_BUCKETS, POOL_SIZE, TASK_CAPACITY);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(tmap.init_data_from_layout(layout, arena));
        tmap.wire_arena_pointers(layout, arena);
    }

    void TearDown() override {
        tmap.destroy();
        arena.release();
    }
};

// Completion progress alone cannot make an older producer disappear. Direct
// inserts remain visible until dependency computation explicitly removes one.
TEST_F(HbgTensorMapTest, EveryProducerOfARegionStaysVisible) {
    ChipTensor t = make_test_tensor(0x1000, 256);
    tmap.insert(t, PTO2TaskId::make(0, 0));
    tmap.insert(t, PTO2TaskId::make(0, 1));
    tmap.insert(t, PTO2TaskId::make(0, 2));
    EXPECT_EQ(tmap.valid_count(), 3);

    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 3);
    std::vector<PTO2TaskId> producers;
    for (const auto &e : result.entries) {
        producers.push_back(e.entry->producer_task_id);
    }
    EXPECT_NE(std::find(producers.begin(), producers.end(), PTO2TaskId::make(0, 0)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), PTO2TaskId::make(0, 1)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), PTO2TaskId::make(0, 2)), producers.end());
}

// Two tasks whose local ids alias to the same task slot both keep their entries;
// slot reuse is not retirement.
TEST_F(HbgTensorMapTest, SlotAliasingTasksBothKeepTheirEntries) {
    ChipTensor t = make_test_tensor(0x1000, 256);
    // Task 0 and task 0 + TASK_CAPACITY share slot 0 (local_id & (TASK_CAPACITY-1)).
    tmap.insert(t, PTO2TaskId::make(0, 0));
    tmap.insert(t, PTO2TaskId::make(0, TASK_CAPACITY));

    EXPECT_EQ(tmap.valid_count(), 2);
    TestLookupResult result;
    run_lookup(tmap, t, result);
    ASSERT_EQ(result.count, 2);
    std::vector<PTO2TaskId> producers;
    for (const auto &e : result.entries) {
        producers.push_back(e.entry->producer_task_id);
    }
    EXPECT_NE(std::find(producers.begin(), producers.end(), PTO2TaskId::make(0, 0)), producers.end());
    EXPECT_NE(std::find(producers.begin(), producers.end(), PTO2TaskId::make(0, TASK_CAPACITY)), producers.end());
}

// Without an explicit semantic removal, direct inserts consume one pool entry
// each. free_entries() is what the orchestrator's pre-registration capacity
// check reads, so it must track those inserts exactly.
TEST_F(HbgTensorMapTest, PoolOccupancyOnlyGrows) {
    EXPECT_EQ(tmap.current_used(), 0);
    EXPECT_EQ(tmap.pool_capacity(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), POOL_SIZE);

    for (int32_t i = 0; i < 8; i++) {
        tmap.insert(make_test_tensor(0x1000 + 0x100 * i, 64), PTO2TaskId::make(0, i));
        EXPECT_EQ(tmap.current_used(), i + 1);
        EXPECT_EQ(tmap.free_entries(), POOL_SIZE - (i + 1));
    }
}

// Filling the pool drives free_entries() to zero. No device-completion watermark
// can free it, so ensure_tensormap_capacity() must fail immediately instead of
// waiting for asynchronous reclaim that HBG does not have.
TEST_F(HbgTensorMapTest, ExhaustedPoolStaysExhausted) {
    for (int32_t i = 0; i < POOL_SIZE; i++) {
        tmap.insert(make_test_tensor(0x10000 + 0x100 * i, 64), PTO2TaskId::make(0, i % TASK_CAPACITY));
    }
    EXPECT_EQ(tmap.current_used(), POOL_SIZE);
    EXPECT_EQ(tmap.free_entries(), 0);
    EXPECT_EQ(tmap.valid_count(), POOL_SIZE);
}

}  // namespace
