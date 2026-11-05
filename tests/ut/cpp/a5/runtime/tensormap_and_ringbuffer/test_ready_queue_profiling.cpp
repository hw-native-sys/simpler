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
#include <gtest/gtest.h>

#include "scheduler/scheduler.h"
#include "utils/device_arena.h"

namespace {
constexpr LaneMask kAicAiv0{true, true, false};
constexpr LaneMask kAicAiv1{true, false, true};
}  // namespace

class ProfiledReadyQueue : public testing::Test {
protected:
    ChipReadyQueue queue;
    DeviceArena arena;
    ChipTaskSlotState slots[5]{};

    void SetUp() override {
        const size_t offset = ready_queue_reserve_layout(arena, 4);
        ASSERT_NE(arena.commit(), nullptr);
        ASSERT_TRUE(ready_queue_init_data_from_layout(&queue, arena, offset, 4, true));
        ready_queue_wire_arena_pointers(&queue, arena, offset);
        for (auto &slot : slots)
            slot.active_mask = kAicAiv0;
    }
    void TearDown() override {
        ready_queue_destroy(&queue);
        arena.release();
    }
};

TEST_F(ProfiledReadyQueue, SingleAndEmptyCallsCountTheirActualAtomics) {
    uint64_t atomics = 0, wait = 0;
    ASSERT_TRUE(queue.push(&slots[0], atomics, wait));
    EXPECT_EQ(atomics, 5u);
    EXPECT_EQ(queue.pop(atomics, wait), &slots[0]);
    EXPECT_EQ(atomics, 12u);
    EXPECT_EQ(queue.pop(atomics, wait), nullptr);
    EXPECT_EQ(atomics, 14u);
    EXPECT_TRUE(queue.reserved_mix_lanes().is_empty());
}

TEST_F(ProfiledReadyQueue, FailedPushCountsReservationAndRollback) {
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(queue.push(&slots[i]));
    slots[4].active_mask = kAicAiv1;
    uint64_t atomics = 0, wait = 0;
    EXPECT_FALSE(queue.push(&slots[4], atomics, wait));
    EXPECT_EQ(atomics, 4u);
    EXPECT_EQ(queue.lane_demand_.count(LaneMask::aic()), 4u);
    EXPECT_EQ(queue.lane_demand_.count(LaneMask::aiv1()), 0u);
}

TEST_F(ProfiledReadyQueue, BatchPopCountsOneSubForEachConsumedCell) {
    ASSERT_TRUE(queue.push(&slots[0]));
    ASSERT_TRUE(queue.push(&slots[1]));
    LaneTaggedSlot out[2];
    uint64_t atomics = 0, wait = 0;
    ASSERT_EQ(queue.pop_batch(out, 2, atomics, wait), 2);
    EXPECT_EQ(out[0].ptr(), &slots[0]);
    EXPECT_EQ(out[1].ptr(), &slots[1]);
    EXPECT_EQ(atomics, 8u);
    EXPECT_TRUE(queue.reserved_mix_lanes().is_empty());
}
