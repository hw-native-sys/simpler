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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "async_completion_plan.h"
#include "async_condition_writer.h"
#include "backend/sdma/sdma_submission_guard.h"

namespace {

using namespace pto2::resident;

AsyncTaskPlanInput task(int64_t ref, uint32_t blocks, LaneMask active) {
    AsyncTaskPlanInput input;
    input.task_ref = ref;
    input.block_count = blocks;
    input.active_mask = active;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (active.has(static_cast<SubtaskSlot>(lane))) input.capacities[lane] = static_cast<uint16_t>(lane + 1);
    return input;
}

struct Image {
    std::vector<uint8_t> backing;
    void *base;

    explicit Image(uint64_t size) :
        backing(size + 191, 0xa5) {
        const uintptr_t aligned = (reinterpret_cast<uintptr_t>(backing.data()) + 127) & ~uintptr_t(127);
        base = reinterpret_cast<void *>(aligned + 64);
    }
};

constexpr uint64_t kPrefix = sizeof(SchedulerRunControl);

uint64_t group_offset(const AsyncCompletionHostPlan &plan, uint32_t index = 0) {
    return plan.layout.groups_offset + static_cast<uint64_t>(index) * sizeof(AsyncGroupData);
}

const AsyncGroupData &group_at(const Image &image, const AsyncCompletionHostPlan &plan, uint32_t index) {
    return *scheduler_state_at<AsyncGroupData>(image.base, group_offset(plan, index));
}

TEST(HbgAsyncPlan, SdmaGuardsAreSharedByExecutionsAndIsolatedByCluster) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 1, 1, {task(0, 2, LaneMask::both_aiv())}, &plan, &error, 32));
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    EXPECT_EQ(plan.layout.sdma_guard_count, 32u);
    EXPECT_EQ(plan.layout.sdma_guards_offset % ASYNC_RECORD_ALIGNMENT, 0u);
    AsyncSdmaSubmissionGuard *first = nullptr;
    for (uint32_t index = 0; index < plan.layout.group_count; ++index) {
        AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 1, group_offset(plan, index), index};
        auto *guard = sdma_submission_guard(writer, 7);
        ASSERT_NE(guard, nullptr);
        if (first == nullptr) first = guard;
        EXPECT_EQ(guard, first);
        EXPECT_NE(sdma_submission_guard(writer, 8), guard);
        EXPECT_EQ(guard->word, 0u);
    }
    AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 1, group_offset(plan), 0};
    EXPECT_EQ(sdma_submission_guard(writer, 32), nullptr);
    EXPECT_FALSE(seal_conditions(writer));
}

TEST(HbgAsyncPlan, SdmaGuardSerializesSessionStateAndRetainsFailure) {
    AsyncSdmaSubmissionGuard guard{};
    uint64_t post_id = 0;
    std::atomic<bool> failed{false};
    std::vector<std::thread> submitters;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        submitters.emplace_back([&]() {
            for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
                SdmaGuardResult result;
                do {
                    result = sdma_try_acquire(guard);
                } while (result == SdmaGuardResult::BUSY);
                if (result != SdmaGuardResult::ACQUIRED) {
                    failed = true;
                    return;
                }
                ++post_id;
                if (!sdma_release(guard, true)) failed = true;
            }
        });
    }
    for (auto &thread : submitters)
        thread.join();
    EXPECT_FALSE(failed);
    EXPECT_EQ(post_id, 4000u);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::ACQUIRED);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::BUSY);
    EXPECT_TRUE(sdma_release(guard, false));
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::FAILED);
    EXPECT_EQ(guard.word, 2u);
}

TEST(HbgAsyncPlan, SynchronousAndControlOnlyRunsAppendNoStorage) {
    AsyncCompletionHostPlan plan;
    std::string error;
    auto synchronous = task(1, 4, LaneMask::aic());
    synchronous.capacities[0] = 0;
    EXPECT_TRUE(plan_async_completion(kPrefix, 1, 3, {synchronous}, &plan, &error)) << error;
    EXPECT_EQ(plan.total_bytes, kPrefix);
    EXPECT_EQ(plan.layout_offset, 0u);
    EXPECT_TRUE(plan.layout.group_count == 0);
    EXPECT_TRUE(initialize_async_completion(nullptr, 0, plan));
    EXPECT_TRUE(plan_async_completion(kPrefix, 2, 3, {}, &plan, &error)) << error;
    EXPECT_EQ(plan.total_bytes, kPrefix);
}

TEST(HbgAsyncPlan, HeadersDoNotStrideAcrossUnusedConditionCapacity) {
    std::vector<AsyncTaskPlanInput> inputs;
    for (int64_t ref = 0; ref < 64; ++ref) {
        auto input = task(ref, 1, LaneMask::aic());
        input.capacities[0] = MAX_COMPLETIONS_PER_TASK;
        inputs.push_back(input);
    }
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 7, 64, inputs, &plan, &error)) << error;
    ASSERT_EQ(plan.layout.group_count, 64u);
    EXPECT_EQ(group_offset(plan, 1), group_offset(plan) + sizeof(AsyncGroupData));
    const uint64_t headers_end = group_offset(plan, plan.layout.group_count - 1) + sizeof(AsyncGroupData);
    EXPECT_LE(headers_end, plan.initialized_bytes);
    EXPECT_LT(plan.initialized_bytes, plan.total_bytes);
    EXPECT_LT(headers_end - kPrefix, 64 * (sizeof(AsyncGroupData) + sizeof(SchedulerAsyncTaskJoin)) + 8192);
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    for (uint32_t index = 0; index < plan.layout.group_count; ++index) {
        const auto &group = group_at(image, plan, index);
        EXPECT_EQ(group.capacity, MAX_COMPLETIONS_PER_TASK);
        EXPECT_GE(group.entries_offset, plan.initialized_bytes);
        EXPECT_LE(group.entries_offset + async_group_entry_bytes(group.capacity), plan.total_bytes);
    }
}

TEST(HbgAsyncPlan, ExpandedTasksBlocksAndLanesHaveIndependentStableRecords) {
    const std::vector<AsyncTaskPlanInput> inputs{
        task(0, 2, LaneMask::aic()), task(2, 3, LaneMask::all()), task(5, 2, LaneMask::both_aiv())
    };
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 19, 7, inputs, &plan, &error)) << error;
    ASSERT_EQ(plan.layout.group_count, 15u);
    ASSERT_EQ(plan.tasks.size(), 3u);
    EXPECT_EQ(plan.tasks[0].data.group_base, 0u);
    EXPECT_EQ(plan.tasks[1].data.group_base, 2u);
    EXPECT_EQ(plan.tasks[2].data.group_base, 11u);
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const auto *map = scheduler_state_at<uint32_t>(image.base, plan.layout.task_map_offset);
    EXPECT_EQ(map[0], 0u);
    EXPECT_EQ(map[1], ASYNC_INVALID_INDEX);
    EXPECT_EQ(map[2], 1u);
    EXPECT_EQ(map[5], 2u);
    EXPECT_EQ(map[6], ASYNC_INVALID_INDEX);
    uint32_t group_index = 0;
    for (const auto &input : inputs) {
        for (uint32_t block = 0; block < input.block_count; ++block) {
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!input.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                const uint64_t offset = group_offset(plan, group_index);
                const auto *data = scheduler_state_at<AsyncGroupData>(image.base, offset);
                EXPECT_EQ(data->task_ref, input.task_ref);
                EXPECT_EQ(data->block_index, block);
                EXPECT_EQ(data->subtask_slot, lane);
                EXPECT_EQ(data->group_index, group_index);
                EXPECT_EQ(offset % ASYNC_RECORD_ALIGNMENT, 0u);
                AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 19, offset, group_index};
                EXPECT_TRUE(seal_conditions(writer));
                if (group_index != 0) EXPECT_GE(offset, group_offset(plan, group_index - 1) + sizeof(AsyncGroupData));
                ++group_index;
            }
        }
    }
    const auto *leases = scheduler_state_at<SchedulerAsyncLease>(image.base, plan.layout.group_leases_offset);
    for (uint32_t index = 0; index < plan.layout.group_count; ++index)
        EXPECT_EQ(leases[index].word, UINT64_C(0xa5a5a5a5a5a5a5a5));
    const auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(image.base, plan.layout.joins_offset);
    EXPECT_EQ(joins[1].word, 0u);
    EXPECT_EQ(joins[2].word, 0u);
    EXPECT_EQ(plan.tasks[1].data.block_count, 3u);
    EXPECT_EQ(plan.tasks[2].data.block_count, 2u);
    for (uint64_t i = plan.initialized_bytes; i < plan.total_bytes; ++i)
        EXPECT_EQ(static_cast<uint8_t *>(image.base)[i], 0xa5);
    for (uint64_t i = 0; i < kPrefix; ++i)
        EXPECT_EQ(static_cast<uint8_t *>(image.base)[i], 0xa5);
}

TEST(HbgAsyncPlan, MixedCapabilityTaskCountsOnlyDeferredLanesAndAllBlocks) {
    auto input = task(0, 5, LaneMask::all());
    input.capacities[1] = 0;
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 2, 1, {input}, &plan, &error)) << error;
    EXPECT_EQ(plan.layout.group_count, 10u);
    EXPECT_TRUE(plan.tasks[0].data.async_mask.has_aic());
    EXPECT_FALSE(plan.tasks[0].data.async_mask.has_aiv0());
    EXPECT_TRUE(plan.tasks[0].data.async_mask.has_aiv1());
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    auto *join = scheduler_state_at<SchedulerAsyncTaskJoin>(image.base, plan.layout.joins_offset);
    EXPECT_EQ(join->word, 0u);
}

TEST(HbgAsyncPlan, HeterogeneousCapacitiesKeepEveryTokenRangeIndependent) {
    auto first = task(2, 3, LaneMask::all());
    first.capacities = {1, 0, 64};
    auto second = task(0, 2, LaneMask::all());
    second.capacities = {63, 4, 1};
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 7, 3, {first, second}, &plan, &error)) << error;
    ASSERT_EQ(plan.layout.group_count, 12u);
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const std::array<uint16_t, 12> capacities{1, 64, 1, 64, 1, 64, 63, 4, 1, 63, 4, 1};
    uint64_t previous_end = plan.initialized_bytes;
    for (uint32_t index = 0; index < capacities.size(); ++index) {
        const auto &data = group_at(image, plan, index);
        EXPECT_EQ(data.capacity, capacities[index]);
        EXPECT_GE(data.entries_offset, previous_end);
        EXPECT_EQ(data.entries_offset % ASYNC_RECORD_ALIGNMENT, 0u);
        EXPECT_EQ(data.entries_bytes, async_group_entry_bytes(capacities[index]));
        previous_end = data.entries_offset + data.entries_bytes;
        EXPECT_LE(previous_end, plan.total_bytes);
        AsyncWriter writer{static_cast<uint8_t *>(image.base), plan.total_bytes, 7, group_offset(plan, index), index};
        for (uint32_t condition = 0; condition < capacities[index]; ++condition) {
            const CompletionToken token{0x10000, condition, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
            ASSERT_TRUE(register_completion_condition(writer, token));
        }
        ASSERT_TRUE(seal_conditions(writer));
    }
    EXPECT_EQ(previous_end, plan.total_bytes);
}

TEST(HbgAsyncPlan, LargestSupportedSpmdAndMixShapesPreserveTheLastRecord) {
    for (const auto lanes : {LaneMask::aic(), LaneMask::all()}) {
        const uint32_t blocks = SCHEDULER_ASYNC_JOIN_COUNT_MAX / lanes.count();
        auto input = task(0, blocks, lanes);
        input.capacities.fill(MAX_COMPLETIONS_PER_TASK);
        AsyncCompletionHostPlan plan;
        std::string error;
        ASSERT_TRUE(plan_async_completion(kPrefix, 8, 1, {input}, &plan, &error)) << error;
        ASSERT_EQ(plan.layout.group_count, blocks * lanes.count());
        Image image(plan.total_bytes);
        ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
        const uint32_t last_index = plan.layout.group_count - 1;
        const auto &last = group_at(image, plan, last_index);
        EXPECT_EQ(last.block_index, blocks - 1);
        EXPECT_EQ(last.group_index, last_index);
        EXPECT_EQ(last.capacity, MAX_COMPLETIONS_PER_TASK);
        EXPECT_EQ(last.entries_offset + last.entries_bytes, plan.total_bytes);
        AsyncWriter writer{
            static_cast<uint8_t *>(image.base), plan.total_bytes, 8, group_offset(plan, last_index), last_index
        };
        for (uint32_t condition = 0; condition < MAX_COMPLETIONS_PER_TASK; ++condition) {
            const CompletionToken token{0x10000, condition, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
            ASSERT_TRUE(register_completion_condition(writer, token));
        }
        ASSERT_TRUE(seal_conditions(writer));
        EXPECT_EQ(group_at(image, plan, last_index - 1).count, 0u);
    }
}

TEST(HbgAsyncPlan, NonmonotonicRefsRetainTheirMappingAndStillRejectDuplicates) {
    std::vector<AsyncTaskPlanInput> inputs{task(2, 2, LaneMask::aic()), task(0, 1, LaneMask::both_aiv())};
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 7, 3, inputs, &plan, &error)) << error;
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const auto *map = scheduler_state_at<uint32_t>(image.base, plan.layout.task_map_offset);
    EXPECT_EQ(map[2], 0u);
    EXPECT_EQ(map[0], 1u);
    EXPECT_EQ(scheduler_state_at<AsyncGroupData>(image.base, group_offset(plan))->task_ref, 2);
    EXPECT_EQ(
        scheduler_state_at<AsyncGroupData>(image.base, group_offset(plan, plan.layout.group_count - 1))->task_ref, 0
    );
    const uint64_t bytes = plan.total_bytes;
    inputs[1].task_ref = 2;
    EXPECT_FALSE(plan_async_completion(kPrefix, 7, 3, inputs, &plan, &error));
    EXPECT_EQ(plan.total_bytes, bytes);
}

TEST(HbgAsyncPlan, EveryMixMaskIncludesOnlyItsActiveLanes) {
    for (uint32_t mask : {3u, 5u, 6u, 7u}) {
        const LaneMask lanes((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        AsyncCompletionHostPlan plan;
        std::string error;
        ASSERT_TRUE(plan_async_completion(kPrefix, 3, 1, {task(0, 4, lanes)}, &plan, &error)) << error;
        EXPECT_EQ(plan.layout.group_count, 4u * static_cast<uint32_t>(lanes.count()));
        EXPECT_EQ(plan.tasks[0].data.async_mask, lanes);
        Image image(plan.total_bytes);
        ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
        for (uint32_t index = 0; index < plan.layout.group_count; ++index)
            EXPECT_TRUE(lanes.has(static_cast<SubtaskSlot>(group_at(image, plan, index).subtask_slot)));
    }
}

TEST(HbgAsyncPlan, RelocationPreservesIndependentExecutionRecords) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 9, 1, {task(0, 2, LaneMask::aiv0())}, &plan, &error)) << error;
    Image first(plan.total_bytes), relocated(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(first.base, plan.total_bytes, plan));
    std::memcpy(relocated.base, first.base, plan.initialized_bytes);
    AsyncWriter writer{static_cast<uint8_t *>(relocated.base), plan.total_bytes, 9, group_offset(plan), 0};
    CompletionToken token{0x10000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    ASSERT_TRUE(register_completion_condition(writer, token));
    EXPECT_EQ(scheduler_state_at<AsyncGroupData>(first.base, group_offset(plan))->count, 0u);
    EXPECT_EQ(scheduler_state_at<AsyncGroupData>(relocated.base, group_offset(plan, 1))->count, 0u);
    EXPECT_TRUE(seal_conditions(writer));
}

TEST(HbgAsyncPlan, DirectoriesHideUninitializedLeasesAcrossWordBoundaries) {
    AsyncCompletionHostPlan plan;
    std::vector<AsyncTaskPlanInput> inputs;
    for (int64_t ref = 0; ref < 65; ++ref)
        inputs.push_back(task(ref, 1, LaneMask::aic()));
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 7, 65, inputs, &plan, &error)) << error;
    Image image(plan.total_bytes);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    const auto *checks =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(image.base, plan.layout.check_directory_offset);
    const auto *finishes =
        scheduler_state_at<SchedulerAsyncDirectoryWord>(image.base, plan.layout.finish_directory_offset);
    for (uint32_t word = 0; word < 2; ++word) {
        EXPECT_EQ(checks[word].bits, 0u);
        EXPECT_EQ(finishes[word].bits, 0u);
    }
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(image.base, plan.layout.finish_leases_offset);
    for (uint32_t index = 0; index < 65; ++index)
        EXPECT_EQ(leases[index].word, UINT64_C(0xa5a5a5a5a5a5a5a5));
    EXPECT_EQ(scheduler_async_condition_mask(0), 0u);
    EXPECT_EQ(scheduler_async_condition_mask(1), 1u);
    EXPECT_EQ(scheduler_async_condition_mask(63), UINT64_MAX >> 1);
    EXPECT_EQ(scheduler_async_condition_mask(64), UINT64_MAX);
}

TEST(HbgAsyncPlan, RejectsOverflowsDuplicateRefsAndInvalidJoinCapacityWithoutReplacingOutput) {
    AsyncCompletionHostPlan plan;
    plan.total_bytes = 99;
    std::string error;
    const auto input = task(0, 1, LaneMask::aic());
    EXPECT_FALSE(plan_async_completion(UINT64_MAX - 127, 1, 1, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 0, 1, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {input, input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 0, {input}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {task(0, 10923, LaneMask::all())}, &plan, &error));
    EXPECT_FALSE(plan_async_completion(kPrefix, 1, 1, {task(0, 0, LaneMask::aic())}, &plan, &error));
    EXPECT_EQ(plan.total_bytes, 99u);
}

TEST(HbgAsyncPlan, InitializationChecksDestinationBeforeWritingAndResetsRetiredStorage) {
    AsyncCompletionHostPlan plan;
    std::string error;
    ASSERT_TRUE(plan_async_completion(kPrefix, 5, 1, {task(0, 1, LaneMask::aic())}, &plan, &error));
    Image image(plan.total_bytes);
    EXPECT_FALSE(initialize_async_completion(image.base, plan.total_bytes - 1, plan));
    EXPECT_FALSE(initialize_async_completion(static_cast<uint8_t *>(image.base) + 1, plan.total_bytes, plan));
    EXPECT_EQ(static_cast<uint8_t *>(image.base)[plan.begin_offset], 0xa5);
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    auto *data = scheduler_state_at<AsyncGroupData>(image.base, group_offset(plan));
    data->count = 1;
    data->state = AsyncPublicationState::FAILED;
    ASSERT_TRUE(plan_async_completion(kPrefix, 6, 1, {task(0, 1, LaneMask::aic())}, &plan, &error));
    ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
    EXPECT_EQ(data->run_epoch, 6u);
    EXPECT_EQ(data->count, 0u);
    EXPECT_EQ(static_cast<AsyncPublicationState>(data->state), AsyncPublicationState::OPEN);
}

TEST(HbgAsyncPlan, UrmaAdmissionRequiresSingleAivExecution) {
    for (uint32_t blocks : {1u, 2u}) {
        for (auto mask : {LaneMask::all(), LaneMask::both_aiv(), LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()}) {
            AsyncCompletionHostPlan plan;
            std::string error;
            ASSERT_TRUE(plan_async_completion(kPrefix, 1, 1, {task(0, blocks, mask)}, &plan, &error));
            Image image(plan.total_bytes);
            ASSERT_TRUE(initialize_async_completion(image.base, plan.total_bytes, plan));
            for (uint32_t i = 0; i < plan.layout.group_count; ++i)
                EXPECT_EQ(
                    group_at(image, plan, i).urma_single_aiv,
                    blocks == 1 && (mask == LaneMask::aiv0() || mask == LaneMask::aiv1()) ? 1u : 0u
                );
        }
    }
}

}  // namespace
