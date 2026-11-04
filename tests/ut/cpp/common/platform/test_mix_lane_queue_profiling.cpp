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
#include <utility>
#include "scheduler/scheduler.h"
#include "support/test_task_id.h"

namespace {
template <typename Queue>
auto push_profiled_mix(Queue &queue, ChipTaskSlotState *slot, uint64_t &ops, uint64_t &wait, int)
    -> decltype(queue.push_mix(slot, TaskId::invalid(), ops, wait), std::pair<bool, bool>{}) {
    return {queue.push_mix(slot, TaskId::invalid(), ops, wait), true};
}
template <typename Queue>
std::pair<bool, bool> push_profiled_mix(Queue &queue, ChipTaskSlotState *slot, uint64_t &, uint64_t &, long) {
    return {queue.push_mix(slot), false};
}
}  // namespace

TEST(MixLaneProfiling, TrackingRmwIsCountedAndCapturedOnBatchPop) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot cells[8]{};
    queue.capacity = 8;
    queue.mask = 7;
    queue.slots = cells;
    queue.enqueue_pos.store(0);
    queue.dequeue_pos.store(0);
    queue.mixed_lanes.reset();
    for (int i = 0; i < 8; ++i)
        cells[i].sequence.store(i);
    ChipTaskSlotState plain{}, mixed{};
    mixed.active_mask = {true, true, false};
    uint64_t plain_push = 0, mix_push = 0, wait = 0;
    ASSERT_TRUE(queue.push(&plain, plain_push, wait));
    CapturedReadySlot values[1];
    uint64_t plain_pop = 0, mix_pop = 0;
    ASSERT_EQ(queue.pop_shaped_batch(values, nullptr, 1, ResourceShape::AIC, plain_pop, wait), 1);
    EXPECT_EQ(values[0].ptr(), &plain);
    const auto [pushed, measured] = push_profiled_mix(queue, &mixed, mix_push, wait, 0);
    ASSERT_TRUE(pushed);
    if (measured) EXPECT_EQ(mix_push, plain_push + 1);
    EXPECT_EQ(queue.mix_demand(), mixed.active_mask);
    ASSERT_EQ(queue.pop_shaped_batch(values, nullptr, 1, ResourceShape::MIX, mix_pop, wait), 1);
    EXPECT_EQ(mix_pop, plain_pop + 1);
    EXPECT_EQ(values[0].ptr(), &mixed);
    EXPECT_EQ(values[0].tag(), mixed.active_mask);
    EXPECT_EQ(queue.mix_demand(), LaneMask::none());
}

TEST(MixLaneProfiling, BatchPopUsesOneDemandRmwAndKeepsOriginalIds) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot cells[8]{};
    queue.capacity = 8;
    queue.mask = 7;
    queue.slots = cells;
    for (int i = 0; i < 8; ++i)
        cells[i].sequence.store(i);
    ChipTaskSlotState tasks[3]{};
    CapturedReadySlot values[5];
    TaskId ids[5];
    const LaneMask masks[] = {{true, true, false}, {true, false, true}, {false, true, true}};
    auto seed_wrap = [&] {
        queue.enqueue_pos.store(6);
        queue.dequeue_pos.store(6);
        for (int i = 0; i < 8; ++i)
            cells[i].sequence.store(i < 6 ? i + 8 : i);
    };
    seed_wrap();
    for (auto &task : tasks)
        ASSERT_TRUE(queue.push(&task));
    uint64_t plain_ops = 0, mix_ops = 0, wait = 0;
    ASSERT_EQ(queue.pop_shaped_batch(values, ids, 5, ResourceShape::AIC, plain_ops, wait), 3);
    for (int i = 0; i < 3; ++i) {
        tasks[i].active_mask = masks[i];
        values[i] = CapturedReadySlot(&tasks[i], tasks[i].active_mask);
        ids[i] = simpler::ut::test_task_id(i + 1);
    }
    seed_wrap();
    ASSERT_FALSE(queue.push_mix_batch(values, ids, 9));
    EXPECT_EQ(queue.mix_demand(), LaneMask::none());
    ASSERT_TRUE(queue.push_mix_batch(values, ids, 3));
    EXPECT_EQ(queue.mixed_lanes.count(LaneMask::aic()), 2u);
    EXPECT_EQ(queue.mixed_lanes.count(LaneMask::aiv0()), 2u);
    EXPECT_EQ(queue.mixed_lanes.count(LaneMask::aiv1()), 2u);
    for (auto &task : tasks)
        task.active_mask = LaneMask::aic();
    ASSERT_EQ(queue.pop_shaped_batch(values, ids, 5, ResourceShape::MIX, mix_ops, wait), 3);
    EXPECT_EQ(mix_ops, plain_ops + 1);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(values[i].ptr(), &tasks[i]);
        EXPECT_EQ(values[i].tag(), masks[i]);
        EXPECT_EQ(ids[i], simpler::ut::test_task_id(i + 1));
    }
    EXPECT_EQ(queue.mix_demand(), LaneMask::none());
    EXPECT_EQ(queue.size(), 0u);
}

TEST(MixLaneProfiling, PlainPushReplacesAReusedCellsGenerationSnapshot) {
    ChipReadyQueue queue{};
    ChipReadyQueueSlot cells[2]{};
    queue.capacity = 2;
    queue.mask = 1;
    queue.slots = cells;
    for (int i = 0; i < 2; ++i) {
        cells[i].sequence.store(i);
        cells[i].task_id_snapshot = simpler::ut::test_task_id(99);
    }
    ChipTaskSlotState task{};
    uint64_t push_ops = 0, pop_ops = 0, wait = 0;
    ASSERT_TRUE(queue.push(&task, push_ops, wait));
    CapturedReadySlot value;
    TaskId id;
    ASSERT_EQ(queue.pop_shaped_batch(&value, &id, 1, ResourceShape::AIC, pop_ops, wait), 1);
    EXPECT_EQ(value.ptr(), &task);
    EXPECT_EQ(id, TaskId::invalid());
    EXPECT_EQ(queue.mix_demand(), LaneMask::none());
}
