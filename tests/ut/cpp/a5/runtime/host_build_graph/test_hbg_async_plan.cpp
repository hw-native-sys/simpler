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

#include <cstring>
#include <set>

#include "async_plan.h"

namespace {
constexpr uint64_t kBudget = 64 * 1024 * 1024;
constexpr uint64_t kEpoch = 17;
constexpr uint64_t kNone = SCHEDULER_ASYNC_INDEX_NONE;

SchedulerTaskMetadata task(uint8_t mask, uint16_t blocks = 1, bool sync_start = false) {
    SchedulerTaskMetadata result{};
    result.active_mask = LaneMask((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
    result.flags = SCHEDULER_TASK_EXECUTABLE;
    if (result.active_mask.count() > 1) result.flags |= SCHEDULER_TASK_MIX;
    if (blocks > 1) result.flags |= SCHEDULER_TASK_SPMD;
    if (sync_start) result.flags |= SCHEDULER_TASK_SYNC_START;
    result.logical_block_num = blocks;
    result.timing_slot = -1;
    return result;
}

template <typename T>
T record(const SchedulerAsyncHostPlan &plan, uint64_t offset, uint64_t index = 0) {
    T value{};
    std::memcpy(&value, plan.image.data() + offset + index * sizeof(T), sizeof(T));
    return value;
}

TEST(HbgAsyncPlan, EveryPhysicalExecutionHasIndependentDefaultCapacity) {
    const std::vector<SchedulerTaskMetadata> metadata{task(1), task(6, 4), task(7, 4, true)};
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, kEpoch, kBudget, &plan, nullptr));
    EXPECT_EQ(plan.layout.group_count, 21u);
    EXPECT_EQ(plan.layout.finish_count, 3u);
    EXPECT_EQ(plan.layout.condition_capacity, 21u * MAX_COMPLETIONS_PER_TASK);
    uint64_t next_entries = plan.layout.entries_offset;
    for (size_t ref = 0; ref < metadata.size(); ++ref) {
        const auto map = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset, ref);
        EXPECT_EQ(map.groups_per_block, metadata[ref].active_mask.count());
        for (uint32_t block = 0; block < map.block_count; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                const auto index = scheduler_async_group_index(map, block, lane);
                if (!metadata[ref].active_mask.has(static_cast<SubtaskSlot>(lane))) {
                    EXPECT_EQ(index, kNone);
                    continue;
                }
                const auto identity =
                    record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, index);
                EXPECT_EQ(identity.task_ref, ref);
                EXPECT_EQ(identity.block_idx, block);
                EXPECT_EQ(identity.subtask_slot, lane);
                EXPECT_EQ(identity.capacity, MAX_COMPLETIONS_PER_TASK);
                EXPECT_EQ(identity.entries_offset, next_entries);
                EXPECT_EQ(next_entries % 128, 0u);
                next_entries += MAX_COMPLETIONS_PER_TASK * sizeof(DeferredCompletionEntry);
            }
        }
    }
    EXPECT_EQ(next_entries, plan.layout.entries_offset + plan.layout.entries_bytes);
}

TEST(HbgAsyncPlan, ControlAndInlineNodesNeedNoCompletionStorage) {
    auto control = task(1);
    control.flags = 0;
    auto inlined = task(1);
    inlined.flags |= SCHEDULER_TASK_INLINE;
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state({control, inlined}, kEpoch, 0, &plan, nullptr));
    EXPECT_TRUE(plan.image.empty());
}

TEST(HbgAsyncPlan, VectorExecutionsShareOneRunOwnedGuardPool) {
    SchedulerAsyncHostPlan scalar, vector;
    ASSERT_TRUE(scheduler_plan_async_state({task(1)}, kEpoch, kBudget, &scalar, nullptr));
    ASSERT_TRUE(scheduler_plan_async_state({task(6, 4)}, kEpoch, kBudget, &vector, nullptr));
    EXPECT_EQ(scalar.layout.sdma_guard_count, 0u);
    const auto context = record<AsyncSdmaGuardContext>(vector, vector.layout.sdma_guard_context_offset);
    EXPECT_EQ(context.run_epoch, kEpoch);
    EXPECT_EQ(context.guard_count, ASYNC_SDMA_CHANNELS);
    EXPECT_EQ(context.bytes, (ASYNC_SDMA_CHANNELS + 1) * 128u);
}

TEST(HbgAsyncPlan, BudgetFailureLeavesOutputUnchangedAndInitializationUsesTheSameLayout) {
    const std::vector<SchedulerTaskMetadata> metadata{task(7, 3)};
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, kEpoch, kBudget, &plan, nullptr));
    const auto saved = plan.image;
    EXPECT_FALSE(scheduler_plan_async_state(metadata, kEpoch, plan.image.size() - 1, &plan, nullptr));
    EXPECT_EQ(saved, plan.image);
    uint64_t offset = 17, total = 19;
    ASSERT_TRUE(scheduler_append_async_layout(129, plan.layout, kBudget, &offset, &total, nullptr));
    EXPECT_EQ(offset, 256u);
    EXPECT_EQ(total, offset + plan.image.size());
    std::vector<std::byte> image(plan.image.size());
    ASSERT_TRUE(scheduler_initialize_async_state(metadata, plan.layout, image.data(), image.size(), nullptr));
    EXPECT_EQ(image, saved);
    auto invalid = plan.layout;
    ++invalid.entries_offset;
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, invalid, image.data(), image.size(), nullptr));
    EXPECT_EQ(image, saved);
    EXPECT_FALSE(scheduler_append_async_layout(UINT64_MAX, plan.layout, kBudget, &offset, &total, nullptr));
}

TEST(HbgAsyncPlan, ReusedStorageInitializesControlPrefixWithoutTouchingConditions) {
    const std::vector<SchedulerTaskMetadata> metadata{task(6, 4)};
    SchedulerAsyncHostPlan fresh;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, kEpoch, kBudget, &fresh, nullptr));
    const auto &layout = fresh.layout;
    ASSERT_LE(layout.sdma_guard_context_offset + (layout.sdma_guard_count + 1) * 128, layout.entries_offset);
    ASSERT_EQ(layout.entries_offset + layout.entries_bytes, layout.total_bytes);
    std::vector<std::byte> reused(layout.total_bytes, std::byte{0xa5});
    ASSERT_TRUE(scheduler_initialize_async_state(metadata, layout, reused.data(), reused.size(), nullptr));
    EXPECT_EQ(std::memcmp(reused.data(), fresh.image.data(), layout.entries_offset), 0);
    for (size_t i = layout.entries_offset; i < reused.size(); ++i) {
        ASSERT_EQ(reused[i], std::byte{0xa5}) << i;
    }
    SchedulerAsyncHostPlan next;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, kEpoch + 1, kBudget, &next, nullptr));
    ASSERT_TRUE(scheduler_initialize_async_state(metadata, next.layout, reused.data(), reused.size(), nullptr));
    EXPECT_EQ(std::memcmp(reused.data(), next.image.data(), next.layout.entries_offset), 0);
    for (size_t i = next.layout.entries_offset; i < reused.size(); ++i) {
        ASSERT_EQ(reused[i], std::byte{0xa5}) << i;
    }
}

TEST(HbgAsyncPlan, InvalidExecutionShapeAndEpochAreRejected) {
    for (auto invalid : {task(0), task(2, 0), task(1, 1, true)}) {
        SchedulerAsyncHostPlan plan;
        EXPECT_FALSE(scheduler_plan_async_state({invalid}, kEpoch, kBudget, &plan, nullptr));
    }
    SchedulerAsyncHostPlan plan;
    EXPECT_FALSE(scheduler_plan_async_state({task(1)}, 0, kBudget, &plan, nullptr));
}
}  // namespace
