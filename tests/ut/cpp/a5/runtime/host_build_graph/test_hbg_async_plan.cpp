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

CoreAsyncCapability deferred(uint32_t count = 1, uint32_t types = CORE_ASYNC_CONDITION_COUNTER) {
    return {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::DEFERRED, count, types};
}

SchedulerAsyncCapabilities capabilities(uint8_t deferred_mask = 0, uint32_t count = 1) {
    SchedulerAsyncCapabilities result{};
    for (uint32_t lane = 0; lane < 3; ++lane)
        result[lane] = (deferred_mask & (1u << lane)) != 0 ?
                           deferred(count) :
                           CoreAsyncCapability{CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::SYNCHRONOUS, 0, 0};
    return result;
}

template <typename T>
T record(const SchedulerAsyncHostPlan &plan, uint64_t offset, uint64_t index = 0) {
    T value{};
    std::memcpy(&value, plan.image.data() + offset + index * sizeof(T), sizeof(T));
    return value;
}

TEST(HbgAsyncPlan, PureSynchronousExecutionNeedsNoAsyncImage) {
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(
        {task(1), task(6, 4), task(7, 4, true)}, {capabilities(), capabilities(), capabilities()}, kEpoch, 0, {}, &plan,
        nullptr
    ));
    EXPECT_TRUE(plan.image.empty());
    EXPECT_EQ(plan.layout.total_bytes, 0u);
    EXPECT_EQ(plan.layout.group_count, 0u);
    EXPECT_EQ(plan.layout.finish_count, 0u);
    EXPECT_EQ(plan.layout.run_epoch, kEpoch);
}

TEST(HbgAsyncPlan, SdmaGuardStorageIsRunOwnedZeroedAndIncludedInTheBudget) {
    SchedulerAsyncHostPlan baseline, guarded;
    const std::vector<SchedulerTaskMetadata> metadata{task(6)};
    const std::vector<SchedulerAsyncCapabilities> caps(1);
    ASSERT_TRUE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, {}, &baseline, nullptr));
    SchedulerAsyncAuxiliaryBytes extra{};
    extra.sdma_guard_count = ASYNC_SDMA_CHANNELS;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, extra, &guarded, nullptr));
    const auto offset = guarded.layout.sdma_guard_context_offset;
    const auto header = record<AsyncSdmaGuardContext>(guarded, offset);
    EXPECT_EQ(header.run_epoch, kEpoch);
    EXPECT_EQ(header.guard_count, ASYNC_SDMA_CHANNELS);
    EXPECT_EQ(header.bytes, (ASYNC_SDMA_CHANNELS + 1) * 128u);
    EXPECT_EQ(guarded.image.size() - baseline.image.size(), header.bytes);
    for (uint64_t queue = 0; queue < header.guard_count; ++queue)
        EXPECT_EQ(record<AsyncSdmaSubmissionGuard>(guarded, offset + sizeof(header), queue).word, 0u);
    EXPECT_FALSE(
        scheduler_plan_async_state(metadata, caps, kEpoch, guarded.image.size() - 1, extra, &baseline, nullptr)
    );
    extra.sdma_guard_count = ASYNC_SDMA_CHANNELS + 1;
    EXPECT_FALSE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, extra, &baseline, nullptr));
    extra.sdma_guard_count = 1;
    EXPECT_FALSE(scheduler_plan_async_state({task(2)}, {capabilities()}, kEpoch, kBudget, extra, &baseline, nullptr));
}

TEST(HbgAsyncPlan, UnknownReserves64ForEveryPhysicalExecutionWithoutChangingDescriptor) {
    for (uint8_t mask : {1, 2, 4, 7}) {
        for (uint16_t blocks : {1, 4}) {
            for (bool sync_start : {false, true}) {
                if (sync_start && blocks == 1) continue;
                const SchedulerAsyncCapabilities caps{};
                SchedulerAsyncHostPlan plan;
                ASSERT_TRUE(scheduler_plan_async_state(
                    {task(mask, blocks, sync_start)}, {caps}, kEpoch, kBudget, {}, &plan, nullptr
                ));
                EXPECT_EQ(plan.layout.group_count, blocks * task(mask).active_mask.count());
                EXPECT_EQ(plan.layout.condition_capacity, 64 * plan.layout.group_count);
                for (uint64_t index = 0; index < plan.layout.group_count; ++index) {
                    const auto identity =
                        record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, index);
                    EXPECT_EQ(identity.capacity, 64u);
                    EXPECT_EQ(identity.condition_types, CORE_ASYNC_CONDITION_TYPES);
                }
                EXPECT_EQ(caps[0].mode, CoreAsyncMode::UNKNOWN);
                EXPECT_EQ(caps[0].max_conditions, 0u);
            }
        }
    }
}

TEST(HbgAsyncPlan, EveryActivePhysicalExecutionGetsAUniqueGroup) {
    for (uint8_t mask : {1, 2, 4, 3, 5, 6, 7}) {
        for (uint16_t blocks : {1, 4}) {
            for (bool sync_start : {false, true}) {
                if (sync_start && blocks == 1) continue;
                SCOPED_TRACE(testing::Message() << "mask=" << +mask << " blocks=" << blocks << " sync=" << sync_start);
                SchedulerAsyncHostPlan plan;
                ASSERT_TRUE(scheduler_plan_async_state(
                    {task(mask, blocks, sync_start)}, {capabilities(mask, 3)}, kEpoch, kBudget, {}, &plan, nullptr
                ));
                const auto map = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset);
                EXPECT_EQ(plan.layout.group_count, blocks * task(mask).active_mask.count());
                EXPECT_EQ(plan.layout.finish_count, 1u);
                EXPECT_EQ(plan.layout.condition_capacity, 3 * plan.layout.group_count);
                EXPECT_EQ(map.async_lane_mask, mask);
                std::set<uint64_t> groups;
                for (uint32_t block = 0; block < blocks; ++block) {
                    for (uint32_t lane = 0; lane < 3; ++lane) {
                        const uint64_t index = scheduler_async_group_index(map, block, lane);
                        if ((mask & (1u << lane)) == 0) {
                            EXPECT_EQ(index, kNone);
                            continue;
                        }
                        ASSERT_LT(index, plan.layout.group_count);
                        EXPECT_TRUE(groups.insert(index).second);
                        const auto identity =
                            record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, index);
                        EXPECT_EQ(identity.run_epoch, kEpoch);
                        EXPECT_EQ(identity.task_ref, 0);
                        EXPECT_EQ(identity.block_idx, block);
                        EXPECT_EQ(identity.subtask_slot, lane);
                        EXPECT_EQ(identity.finish_index, 0u);
                        EXPECT_EQ(identity.capacity, 3u);
                    }
                }
                EXPECT_EQ(groups.size(), plan.layout.group_count);
                EXPECT_EQ(scheduler_async_group_index(map, blocks, 0), kNone);
                EXPECT_EQ(scheduler_async_group_index(map, 0, 3), kNone);
                const auto join = record<SchedulerAsyncTaskJoin>(plan, plan.layout.finish_join_offset);
                EXPECT_EQ(scheduler_async_join_blocks(join.word), blocks);
                EXPECT_EQ(scheduler_async_join_groups(join.word), 0u);
            }
        }
    }
}

TEST(HbgAsyncPlan, MixedSynchronousLanesAllocateOnlyDeclaredGroups) {
    SchedulerAsyncHostPlan plan;
    auto caps = capabilities(5);
    caps[0] = deferred(1);
    caps[2] = deferred(64, CORE_ASYNC_CONDITION_SDMA_POST_DONE);
    ASSERT_TRUE(scheduler_plan_async_state({task(7, 4)}, {caps}, kEpoch, kBudget, {}, &plan, nullptr));
    const auto map = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset);
    EXPECT_EQ(plan.layout.group_count, 8u);
    EXPECT_EQ(plan.layout.condition_capacity, 260u);
    EXPECT_EQ(map.groups_per_block, 2u);
    EXPECT_EQ(map.async_lane_mask, 5u);
    EXPECT_EQ(scheduler_async_group_index(map, 0, 1), kNone);
    for (uint32_t block = 0; block < 4; ++block) {
        EXPECT_EQ(scheduler_async_group_index(map, block, 0), 2 * block);
        EXPECT_EQ(scheduler_async_group_index(map, block, 2), 2 * block + 1);
    }
    const auto first = record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, 0);
    const auto second = record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, 1);
    EXPECT_EQ(first.capacity, 1u);
    EXPECT_EQ(second.capacity, 64u);
    EXPECT_EQ(second.condition_types, CORE_ASYNC_CONDITION_SDMA_POST_DONE);
    EXPECT_EQ(second.entries_offset - first.entries_offset, 128u);
}

TEST(HbgAsyncPlan, ExpandedGraphReferencesDoNotAliasTraceOrKernelIdentity) {
    auto first = task(1), second = task(1);
    first.trace_index_base = second.trace_index_base = UINT32_MAX;
    first.kernel_ids[0] = second.kernel_ids[0] = 13;
    auto control = task(0);
    control.flags = 0;
    auto inline_task = task(0);
    inline_task.flags |= SCHEDULER_TASK_INLINE;
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(
        {control, first, task(2), inline_task, second}, {{}, capabilities(1), capabilities(), {}, capabilities(1)},
        kEpoch, kBudget, {}, &plan, nullptr
    ));
    EXPECT_EQ(plan.layout.task_count, 5u);
    EXPECT_EQ(plan.layout.group_count, 2u);
    for (uint64_t index : {0, 2, 3}) {
        const auto map = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset, index);
        EXPECT_EQ(map.group_base, kNone);
        EXPECT_EQ(map.finish_index, kNone);
    }
    const auto a = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset, 1);
    const auto b = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset, 4);
    EXPECT_NE(a.group_base, b.group_base);
    EXPECT_NE(a.finish_index, b.finish_index);
    EXPECT_EQ(record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, a.group_base).task_ref, 1);
    EXPECT_EQ(record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, b.group_base).task_ref, 4);
}

TEST(HbgAsyncPlan, ImageBeginsDormantEmptyAndUnbound) {
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state({task(7, 24)}, {capabilities(7, 64)}, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_EQ(plan.layout.check_directory_words, 2u);
    for (uint64_t i = 0; i < plan.layout.check_directory_words; ++i)
        EXPECT_EQ(record<SchedulerAsyncDirectoryWord>(plan, plan.layout.check_directory_offset, i).bits, 0u);
    EXPECT_EQ(record<SchedulerAsyncDirectoryWord>(plan, plan.layout.finish_directory_offset).bits, 0u);
    EXPECT_EQ(record<SchedulerAsyncLease>(plan, plan.layout.finish_lease_offset).word, SchedulerAsyncLease{}.word);
    const auto finish = record<SchedulerAsyncFinishProgress>(plan, plan.layout.finish_progress_offset);
    EXPECT_EQ(finish.completion.resolve.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(finish.completion.detached_next, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(finish.completion.publication.pending_count, 0u);
    for (uint64_t i = 0; i < plan.layout.group_count; ++i) {
        const auto id = record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, i);
        const auto writer = record<SchedulerAsyncGroupWriter>(plan, plan.layout.group_writer_offset, i);
        EXPECT_EQ(writer.count, 0u);
        EXPECT_EQ(writer.error_code, 0);
        EXPECT_EQ(writer.reservation.sealed, 0u);
        EXPECT_EQ(
            record<SchedulerAsyncLease>(plan, plan.layout.group_lease_offset, i).word, SchedulerAsyncLease{}.word
        );
        const auto progress = record<SchedulerAsyncGroupProgress>(plan, plan.layout.group_progress_offset, i);
        EXPECT_EQ(progress.remaining_mask, 0u);
        EXPECT_EQ(progress.failure_code, 0);
    }
}

TEST(HbgAsyncPlan, AllRegionsAndPerGroupEntriesHaveDisjointCacheLines) {
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(
        {task(7, 4)}, {capabilities(7, 5)}, kEpoch, kBudget, {257, 1, HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY}, &plan,
        nullptr
    ));
    const auto &l = plan.layout;
    uint64_t end = sizeof(SchedulerAsyncLayout);
    const std::pair<uint64_t, uint64_t> regions[] = {
        {l.task_map_offset, l.task_count * sizeof(SchedulerAsyncTaskMap)},
        {l.group_identity_offset, l.group_count * sizeof(SchedulerAsyncGroupIdentity)},
        {l.group_writer_offset, l.group_count * sizeof(SchedulerAsyncGroupWriter)},
        {l.group_lease_offset, l.group_count * sizeof(SchedulerAsyncLease)},
        {l.group_progress_offset, l.group_count * sizeof(SchedulerAsyncGroupProgress)},
        {l.finish_identity_offset, l.finish_count * sizeof(SchedulerAsyncFinishIdentity)},
        {l.finish_lease_offset, l.finish_count * sizeof(SchedulerAsyncLease)},
        {l.finish_join_offset, l.finish_count * sizeof(SchedulerAsyncTaskJoin)},
        {l.finish_progress_offset, l.finish_count * sizeof(SchedulerAsyncFinishProgress)},
        {l.check_directory_offset, l.check_directory_words * sizeof(SchedulerAsyncDirectoryWord)},
        {l.finish_directory_offset, l.finish_directory_words * sizeof(SchedulerAsyncDirectoryWord)},
        {l.entries_offset, l.entries_bytes},
        {l.backend_offset, l.backend_bytes},
        {l.sdma_guard_context_offset, (l.sdma_guard_count + 1) * 128}
    };
    for (const auto &[offset, size] : regions) {
        EXPECT_EQ(offset % 128, 0u);
        EXPECT_GE(offset, end);
        end = offset + size;
        EXPECT_LE(end, plan.image.size());
    }
    EXPECT_EQ(l.total_bytes, plan.image.size());
    EXPECT_EQ(l.total_bytes % 128, 0u);
    uint64_t entries_end = l.entries_offset;
    for (uint64_t i = 0; i < l.group_count; ++i) {
        const auto id = record<SchedulerAsyncGroupIdentity>(plan, l.group_identity_offset, i);
        EXPECT_EQ(id.entries_offset % 128, 0u);
        EXPECT_GE(id.entries_offset, entries_end);
        entries_end = id.entries_offset + id.capacity * sizeof(DeferredCompletionEntry);
    }
    EXPECT_LE(entries_end, l.entries_offset + l.entries_bytes);
}

TEST(HbgAsyncPlan, DirectoryWordBoundaryAndUnusedRecordBytesAreDeterministic) {
    for (uint16_t groups : {64, 65}) {
        SchedulerAsyncHostPlan plan;
        ASSERT_TRUE(
            scheduler_plan_async_state({task(1, groups)}, {capabilities(1)}, kEpoch, kBudget, {}, &plan, nullptr)
        );
        EXPECT_EQ(plan.layout.check_directory_words, groups == 64 ? 1u : 2u);
        for (uint64_t index = 0; index < groups; ++index) {
            const auto id = record<SchedulerAsyncGroupIdentity>(plan, plan.layout.group_identity_offset, index);
            EXPECT_EQ(id.block_idx, index);
            EXPECT_EQ(id.run_epoch, kEpoch);
            const uint64_t lease = plan.layout.group_lease_offset + index * sizeof(SchedulerAsyncLease);
            for (uint64_t byte = sizeof(uint64_t); byte < sizeof(SchedulerAsyncLease); ++byte)
                EXPECT_EQ(plan.image[lease + byte], std::byte{0});
        }
        const auto finish = record<SchedulerAsyncFinishIdentity>(plan, plan.layout.finish_identity_offset);
        EXPECT_EQ(finish.group_count, groups);
        const auto progress = record<SchedulerAsyncFinishProgress>(plan, plan.layout.finish_progress_offset);
        for (uint8_t byte : progress.completion.resolve.reserved)
            EXPECT_EQ(byte, 0u);
        for (uint64_t byte = sizeof(uint64_t); byte < sizeof(SchedulerAsyncTaskJoin); ++byte)
            EXPECT_EQ(plan.image[plan.layout.finish_join_offset + byte], std::byte{0});
        SchedulerAsyncHostPlan duplicate;
        ASSERT_TRUE(
            scheduler_plan_async_state({task(1, groups)}, {capabilities(1)}, kEpoch, kBudget, {}, &duplicate, nullptr)
        );
        EXPECT_EQ(duplicate.image, plan.image);
    }
}

TEST(HbgAsyncPlan, DiagnosticsUseOptionalRunStorageWithoutChangingDefaultCapacity) {
    SchedulerAsyncHostPlan plain, measured, synchronous;
    const SchedulerAsyncAuxiliaryBytes diagnostics{0, 0, HBG_ASYNC_CONTRACT_UNQUALIFIED, true};
    ASSERT_TRUE(scheduler_plan_async_state({task(2, 8)}, {capabilities(2, 64)}, kEpoch, kBudget, {}, &plain, nullptr));
    ASSERT_TRUE(scheduler_plan_async_state(
        {task(2, 8)}, {capabilities(2, 64)}, kEpoch, kBudget, diagnostics, &measured, nullptr
    ));
    EXPECT_EQ(plain.layout.group_trace_offset, 0u);
    EXPECT_EQ(plain.layout.finish_trace_offset, 0u);
    EXPECT_EQ(plain.layout.condition_capacity, measured.layout.condition_capacity);
    EXPECT_EQ(plain.layout.group_count, measured.layout.group_count);
    EXPECT_GT(measured.layout.total_bytes, plain.layout.total_bytes);
    EXPECT_GE(measured.layout.group_trace_offset, plain.layout.total_bytes);
    EXPECT_GE(
        measured.layout.finish_trace_offset,
        measured.layout.group_trace_offset + measured.layout.group_count * sizeof(SchedulerAsyncGroupTrace)
    );
    ASSERT_TRUE(
        scheduler_plan_async_state({task(2)}, {capabilities()}, kEpoch, kBudget, diagnostics, &synchronous, nullptr)
    );
    EXPECT_TRUE(synchronous.image.empty());
    EXPECT_EQ(synchronous.layout.group_trace_offset, 0u);
}

TEST(HbgAsyncPlan, ExactBudgetSucceedsAndRefusalPreservesPreviousImage) {
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state({task(1)}, {capabilities(1)}, kEpoch, kBudget, {}, &plan, nullptr));
    const uint64_t size = plan.layout.total_bytes;
    ASSERT_TRUE(scheduler_plan_async_state({task(1)}, {capabilities(1)}, kEpoch, size, {}, &plan, nullptr));
    const auto saved = plan.image;
    std::string error;
    EXPECT_FALSE(scheduler_plan_async_state({task(1)}, {capabilities(1)}, 18, size - 1, {}, &plan, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(plan.image, saved);
    EXPECT_EQ(plan.layout.run_epoch, kEpoch);
    for (const auto aux :
         {SchedulerAsyncAuxiliaryBytes{UINT64_MAX, 0, 0}, SchedulerAsyncAuxiliaryBytes{0, UINT64_MAX, 0},
          SchedulerAsyncAuxiliaryBytes{0, 0, UINT64_MAX}}) {
        EXPECT_FALSE(scheduler_plan_async_state({task(1)}, {capabilities(1)}, 18, UINT64_MAX, aux, &plan, &error));
        EXPECT_EQ(plan.image, saved);
    }
}

TEST(HbgAsyncPlan, RejectsMalformedCapabilityOnAnActiveLane) {
    const CoreAsyncCapability invalid[] = {
        {1, CoreAsyncMode::UNKNOWN, 1, 0},     {1, CoreAsyncMode::UNKNOWN, 0, 1},
        {2, CoreAsyncMode::DEFERRED, 1, 1},    {1, static_cast<CoreAsyncMode>(3), 0, 0},
        {1, CoreAsyncMode::SYNCHRONOUS, 1, 0}, {1, CoreAsyncMode::DEFERRED, 0, 1},
        {1, CoreAsyncMode::DEFERRED, 65, 1},   {1, CoreAsyncMode::DEFERRED, 1, 0},
        {1, CoreAsyncMode::DEFERRED, 1, 4}
    };
    for (auto cap : invalid) {
        auto row = capabilities();
        row[0] = cap;
        SchedulerAsyncHostPlan plan;
        EXPECT_FALSE(scheduler_plan_async_state({task(1)}, {row}, kEpoch, kBudget, {}, &plan, nullptr));
        EXPECT_TRUE(scheduler_plan_async_state({task(2)}, {row}, kEpoch, kBudget, {}, &plan, nullptr));
    }
}

TEST(HbgAsyncPlan, RejectsMalformedPhysicalExecutionShapes) {
    std::vector<SchedulerTaskMetadata> invalid{task(0), task(1, 0), task(7, 10923), task(1, 1, true)};
    auto bad = task(1);
    uint8_t raw = 0x08;
    std::memcpy(&bad.active_mask, &raw, sizeof(raw));
    invalid.push_back(bad);
    raw = 0x09;
    std::memcpy(&bad.active_mask, &raw, sizeof(raw));
    invalid.push_back(bad);
    bad = task(3);
    bad.flags &= ~SCHEDULER_TASK_MIX;
    invalid.push_back(bad);
    bad = task(1, 4);
    bad.flags &= ~SCHEDULER_TASK_SPMD;
    invalid.push_back(bad);
    bad = task(1);
    bad.flags |= SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD;
    invalid.push_back(bad);
    bad = task(3);
    bad.flags |= SCHEDULER_TASK_HAS_PREDICATE;
    invalid.push_back(bad);
    bad = task(1, 4);
    bad.flags |= SCHEDULER_TASK_HAS_PREDICATE;
    invalid.push_back(bad);
    for (const auto &meta : invalid) {
        SchedulerAsyncHostPlan plan;
        EXPECT_FALSE(scheduler_plan_async_state({meta}, {capabilities(7)}, kEpoch, kBudget, {}, &plan, nullptr));
    }
}

TEST(HbgAsyncPlan, MaximumLegalMixedExecutionCountFitsJoinBound) {
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state({task(7, 10922)}, {capabilities(7)}, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_EQ(plan.layout.group_count, 32766u);
    const auto map = record<SchedulerAsyncTaskMap>(plan, plan.layout.task_map_offset);
    EXPECT_EQ(scheduler_async_group_index(map, 10921, 2), 32765u);
    const auto join = record<SchedulerAsyncTaskJoin>(plan, plan.layout.finish_join_offset);
    EXPECT_EQ(scheduler_async_join_blocks(join.word), 10922u);
}

TEST(HbgAsyncPlan, ValidatesInputsBeforePublishingThePlan) {
    SchedulerAsyncHostPlan plan;
    EXPECT_FALSE(scheduler_plan_async_state({}, {}, 0, kBudget, {}, &plan, nullptr));
    EXPECT_FALSE(scheduler_plan_async_state({task(1)}, {}, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_FALSE(scheduler_plan_async_state({}, {}, kEpoch, kBudget, {}, nullptr, nullptr));
    EXPECT_FALSE(scheduler_plan_async_state({task(1)}, {capabilities()}, kEpoch, kBudget, {1, 0, 0}, &plan, nullptr));
    EXPECT_TRUE(scheduler_plan_async_state({}, {}, kEpoch, 0, {}, &plan, nullptr));
    EXPECT_TRUE(plan.image.empty());
}

TEST(HbgAsyncPlan, DirectInitializationMatchesTheImageWithoutWritingTheBackingPrefix) {
    const std::vector<SchedulerTaskMetadata> metadata{task(1), task(7, 4, true)};
    const std::vector<SchedulerAsyncCapabilities> caps{capabilities(), capabilities(5, 3)};
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, {}, &plan, nullptr));
    SchedulerAsyncLayout layout;
    ASSERT_TRUE(scheduler_plan_async_layout(metadata, caps, kEpoch, kBudget, {}, &layout, nullptr));
    EXPECT_EQ(std::memcmp(&layout, &plan.layout, sizeof(layout)), 0);
    uint64_t offset = 0, total = 0;
    ASSERT_TRUE(scheduler_append_async_layout(385, layout, kBudget, &offset, &total, nullptr));
    EXPECT_EQ(offset, 512u);
    EXPECT_EQ(total, offset + layout.total_bytes);
    std::vector<std::byte> backing(total + 17, std::byte{0xA5});
    ASSERT_TRUE(
        scheduler_initialize_async_state(metadata, caps, layout, backing.data() + offset, layout.total_bytes, nullptr)
    );
    EXPECT_EQ(std::memcmp(backing.data() + offset, plan.image.data(), plan.image.size()), 0);
    for (uint64_t byte = 0; byte < offset; ++byte)
        EXPECT_EQ(backing[byte], std::byte{0xA5});
    for (uint64_t byte = total; byte < backing.size(); ++byte)
        EXPECT_EQ(backing[byte], std::byte{0xA5});
}

TEST(HbgAsyncPlan, TotalBackingBudgetIncludesThePrefixAndAlignment) {
    SchedulerAsyncLayout layout;
    ASSERT_TRUE(scheduler_plan_async_layout({task(1)}, {capabilities(1)}, kEpoch, kBudget, {}, &layout, nullptr));
    const uint64_t exact = 256 + layout.total_bytes;
    uint64_t offset = 13, total = 17;
    EXPECT_FALSE(scheduler_append_async_layout(129, layout, exact - 1, &offset, &total, nullptr));
    EXPECT_EQ(offset, 13u);
    EXPECT_EQ(total, 17u);
    ASSERT_TRUE(scheduler_append_async_layout(129, layout, exact, &offset, &total, nullptr));
    EXPECT_EQ(offset, 256u);
    EXPECT_EQ(total, exact);
    EXPECT_FALSE(scheduler_append_async_layout(UINT64_MAX, layout, UINT64_MAX, &offset, &total, nullptr));
    EXPECT_EQ(offset, 256u);
    EXPECT_EQ(total, exact);
    EXPECT_FALSE(scheduler_append_async_layout(0, layout, kBudget, &offset, &total, nullptr));
    EXPECT_FALSE(scheduler_append_async_layout(129, layout, kBudget, nullptr, &total, nullptr));
}

TEST(HbgAsyncPlan, SynchronousLayoutLeavesTheBackingAndLocatorUnchanged) {
    SchedulerAsyncLayout layout;
    ASSERT_TRUE(scheduler_plan_async_layout({task(7, 4)}, {capabilities()}, kEpoch, 0, {}, &layout, nullptr));
    uint64_t offset = 7, total = 9;
    ASSERT_TRUE(scheduler_append_async_layout(385, layout, 385, &offset, &total, nullptr));
    EXPECT_EQ(offset, 0u);
    EXPECT_EQ(total, 385u);
    std::byte untouched{0xA5};
    EXPECT_TRUE(scheduler_initialize_async_state({task(7, 4)}, {capabilities()}, layout, &untouched, 0, nullptr));
    EXPECT_EQ(untouched, std::byte{0xA5});
    EXPECT_FALSE(scheduler_append_async_layout(385, layout, 384, &offset, &total, nullptr));
    EXPECT_EQ(offset, 0u);
    EXPECT_EQ(total, 385u);
}

TEST(HbgAsyncPlan, InitializationRejectsLayoutOrCapabilityDriftBeforeWriting) {
    const std::vector<SchedulerTaskMetadata> metadata{task(1)};
    const std::vector<SchedulerAsyncCapabilities> caps{capabilities(1)};
    SchedulerAsyncLayout layout;
    ASSERT_TRUE(scheduler_plan_async_layout(metadata, caps, kEpoch, kBudget, {}, &layout, nullptr));
    std::vector<std::byte> backing(layout.total_bytes, std::byte{0xA5});
    const auto saved = backing;
    auto corrupt = layout;
    corrupt.group_progress_offset = 0;
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, caps, corrupt, backing.data(), backing.size(), nullptr));
    EXPECT_EQ(backing, saved);
    corrupt = layout;
    ++corrupt.version;
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, caps, corrupt, backing.data(), backing.size(), nullptr));
    EXPECT_EQ(backing, saved);
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, caps, layout, backing.data(), backing.size() - 1, nullptr));
    EXPECT_EQ(backing, saved);
    EXPECT_FALSE(scheduler_initialize_async_state(
        metadata, {capabilities(1, 2)}, layout, backing.data(), backing.size(), nullptr
    ));
    EXPECT_EQ(backing, saved);
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, caps, layout, nullptr, backing.size(), nullptr));
}
TEST(HbgAsyncPlan, MixedAndSyncStartTasksNeedNoConsumerScratch) {
    const std::vector<SchedulerTaskMetadata> metadata{
        task(1), task(6, 3), task(7, 2, true), task(2, 5), task(1, 2, true)
    };
    const std::vector<SchedulerAsyncCapabilities> caps{
        capabilities(1), capabilities(), capabilities(), capabilities(), capabilities()
    };
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, {}, &plan, nullptr));
    EXPECT_EQ(plan.layout.group_count, 1u);
    EXPECT_EQ(plan.layout.finish_count, 1u);
    EXPECT_EQ(plan.layout.backend_bytes, 0u);
    EXPECT_EQ(plan.layout.total_bytes, plan.layout.entries_offset + plan.layout.entries_bytes);
}

TEST(HbgAsyncPlan, BackendDeclarationsHaveNoConsumerPrefix) {
    const std::vector<SchedulerTaskMetadata> metadata{task(7, 2, true)};
    const std::vector<SchedulerAsyncCapabilities> caps{capabilities(2)};
    SchedulerAsyncHostPlan plan;
    ASSERT_TRUE(scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, {129, 0, 0}, &plan, nullptr));
    EXPECT_EQ(plan.layout.backend_bytes, 129u);
    EXPECT_EQ(plan.layout.backend_offset % 128, 0u);
    for (uint64_t i = 0; i < plan.layout.backend_bytes; ++i)
        EXPECT_EQ(plan.image[plan.layout.backend_offset + i], std::byte{0});
    SchedulerAsyncLayout exact{};
    ASSERT_TRUE(
        scheduler_plan_async_layout(metadata, caps, kEpoch, plan.layout.total_bytes, {129, 0, 0}, &exact, nullptr)
    );
    EXPECT_FALSE(
        scheduler_plan_async_layout(metadata, caps, kEpoch, plan.layout.total_bytes - 1, {129, 0, 0}, &exact, nullptr)
    );
    std::vector<std::byte> rebuilt(plan.image.size(), std::byte{0xA5});
    ASSERT_TRUE(scheduler_initialize_async_state(metadata, caps, plan.layout, rebuilt.data(), rebuilt.size(), nullptr));
    EXPECT_EQ(rebuilt, plan.image);
    auto bad = plan.layout;
    --bad.backend_offset;
    const auto saved = rebuilt;
    EXPECT_FALSE(scheduler_initialize_async_state(metadata, caps, bad, rebuilt.data(), rebuilt.size(), nullptr));
    EXPECT_EQ(rebuilt, saved);
}

}  // namespace
