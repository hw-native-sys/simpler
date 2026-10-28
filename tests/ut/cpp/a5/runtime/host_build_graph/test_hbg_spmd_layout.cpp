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
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_memory.h"

namespace {

SchedulerSpmdRequirements requirements(uint64_t workers, uint64_t cooperative, uint64_t mix) {
    return {workers, workers / 3, cooperative, mix};
}

SchedulerAdmissionRecord *record_at(void *base, const AicoreSchedulerLayout &layout, uint64_t task) {
    return scheduler_state_at<SchedulerAdmissionRecord>(
        base, layout.admission_records_offset + task * layout.admission_record_stride
    );
}

void set_metadata(
    void *base, const AicoreSchedulerLayout &layout, uint64_t task, uint16_t blocks, uint8_t mask,
    uint8_t extra_flags = 0
) {
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
    auto &entry = metadata[task];
    entry.logical_block_num = blocks;
    entry.active_mask = mask;
    entry.flags = SCHEDULER_TASK_EXECUTABLE | extra_flags;
    if (blocks > 1) entry.flags |= SCHEDULER_TASK_SPMD;
    if (__builtin_popcount(mask) > 1) entry.flags |= SCHEDULER_TASK_MIX;
    entry.timing_slot = -1;
    for (uint8_t lane = 0; lane < 3; ++lane)
        entry.kernel_ids[lane] = (mask & (1U << lane)) != 0 ? 1 : UINT16_MAX;
}

void expect_no_protocol_storage(const AicoreSchedulerLayout &layout) {
    EXPECT_EQ(layout.admission_control_offset, 0u);
    EXPECT_EQ(layout.admission_records_offset, 0u);
    EXPECT_EQ(layout.admission_record_stride, 0u);
    EXPECT_EQ(layout.admission_member_word_count, 0u);
    EXPECT_EQ(layout.admission_log_offset, 0u);
    EXPECT_EQ(layout.group_mailboxes_offset, 0u);
    EXPECT_EQ(layout.mix_close_seals_offset, 0u);
    EXPECT_EQ(layout.mix_close_seal_count, 0u);
}

void expect_summary_configuration(void *base, const AicoreSchedulerLayout &layout, uint64_t epoch) {
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, layout.graph_summary_offset);
    EXPECT_EQ(summary->mix_task_count, layout.spmd.mix_task_count);
    EXPECT_EQ(summary->cooperative_task_count, layout.spmd.cooperative_task_count);
    EXPECT_EQ(summary->worker_count, layout.spmd.worker_count);
    EXPECT_EQ(summary->cluster_count, layout.spmd.cluster_count);
    EXPECT_EQ(summary->admission_member_word_count, layout.admission_member_word_count);
    EXPECT_EQ(summary->admission_records_offset, layout.admission_records_offset);
    EXPECT_EQ(summary->admission_record_stride, layout.admission_record_stride);
    EXPECT_EQ(summary->admission_log_offset, layout.admission_log_offset);
    EXPECT_EQ(summary->admission_control_offset, layout.admission_control_offset);
    EXPECT_EQ(summary->group_mailboxes_offset, layout.group_mailboxes_offset);
    EXPECT_EQ(summary->mix_close_seals_offset, layout.mix_close_seals_offset);
    EXPECT_EQ(summary->run_epoch, epoch);
}

void expect_empty_protocol(void *base, const AicoreSchedulerLayout &layout) {
    expect_summary_configuration(base, layout, 0);
    const auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        SCOPED_TRACE(task);
        EXPECT_EQ(controls[task].next_block, 0u);
        EXPECT_EQ(controls[task].claim_tickets, 0u);
        EXPECT_EQ(controls[task].completed_blocks, 0u);
        EXPECT_EQ(controls[task].admission_state, 0u);
        EXPECT_EQ(controls[task].admission_claim_lock, 0u);
    }
    if (layout.spmd.cooperative_task_count == 0) return;
    const auto *control = scheduler_state_at<SchedulerAdmissionControl>(base, layout.admission_control_offset);
    EXPECT_EQ(control->commit_lock, 0u);
    EXPECT_EQ(control->commit_tail, 0u);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        SCOPED_TRACE(task);
        const auto *record = record_at(base, layout, task);
        EXPECT_EQ(record->task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(record->run_epoch, 0u);
        EXPECT_EQ(record->order_key, UINT64_MAX);
        EXPECT_EQ(record->logical_block_num, 0u);
        EXPECT_EQ(record->member_count, 0u);
        EXPECT_EQ(record->active_mask, 0u);
        EXPECT_EQ(record->flags, 0u);
        EXPECT_EQ(record->mix_close_seals_offset, 0u);
        ASSERT_EQ(
            record->member_words_offset,
            layout.admission_records_offset + task * layout.admission_record_stride + sizeof(SchedulerAdmissionRecord)
        );
        const auto *members = scheduler_state_at<uint64_t>(base, record->member_words_offset);
        for (uint64_t word = 0; word < layout.admission_member_word_count; ++word)
            EXPECT_EQ(members[word], 0u);
    }
    const auto *log = scheduler_state_at<int64_t>(base, layout.admission_log_offset);
    for (uint64_t index = 0; index < layout.spmd.cooperative_task_count; ++index)
        EXPECT_EQ(log[index], SCHEDULER_TASK_REF_INVALID);
    if (layout.spmd.mix_task_count == 0) return;
    const auto *mailboxes = scheduler_state_at<SchedulerGroupMailbox>(base, layout.group_mailboxes_offset);
    for (uint64_t cluster = 0; cluster < layout.spmd.cluster_count; ++cluster) {
        SCOPED_TRACE(cluster);
        const auto &mailbox = mailboxes[cluster];
        EXPECT_EQ(mailbox.descriptor.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(mailbox.descriptor.order_key, UINT64_MAX);
        EXPECT_EQ(mailbox.descriptor.run_epoch, 0u);
        EXPECT_EQ(mailbox.descriptor.group_seq, 0u);
        EXPECT_EQ(mailbox.descriptor.block_idx, UINT32_MAX);
        EXPECT_EQ(mailbox.descriptor.cluster_index, UINT32_MAX);
        EXPECT_EQ(mailbox.published_seq, 0u);
        for (uint64_t lane = 0; lane < 3; ++lane) {
            EXPECT_EQ(mailbox.descriptor.lane_subtasks[lane], UINT8_MAX);
            EXPECT_EQ(mailbox.lanes[lane].arrived_seq, 0u);
            EXPECT_EQ(mailbox.lanes[lane].released_seq, 0u);
        }
    }
    const auto *seals = scheduler_state_at<uint64_t>(base, layout.mix_close_seals_offset);
    for (uint64_t index = 0; index < layout.mix_close_seal_count; ++index)
        EXPECT_EQ(seals[index], 0u);
}

TEST(HbgSpmdLayout, PreservesCompactMetadataAndAtomicLineBoundaries) {
    EXPECT_EQ(sizeof(SchedulerTaskMetadata), 16u);
    EXPECT_EQ(alignof(SchedulerTaskMetadata), 16u);
    EXPECT_EQ(offsetof(SchedulerTaskMetadata, timing_slot), 12u);
    EXPECT_EQ(offsetof(SchedulerTaskMetadata, logical_block_num), 14u);
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskMetadata::timing_slot), int16_t>));
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskMetadata::logical_block_num), uint16_t>));
    EXPECT_EQ(sizeof(SchedulerTaskControl), 512u);
    EXPECT_EQ(alignof(SchedulerTaskControl), 128u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, claim_tickets), 24u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, next_block), 128u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, completed_blocks), 256u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, admission_state), 384u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, admission_claim_lock), 40u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, next_fanin_index), 64u);
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskControl::next_block), volatile uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskControl::completed_blocks), volatile uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskControl::admission_state), volatile uint64_t>));
    EXPECT_TRUE((std::is_same_v<decltype(SchedulerTaskControl::admission_claim_lock), volatile uint64_t>));
}

TEST(HbgSpmdLayout, TaskBlockCountersOwnSeparate128ByteRegions) {
    constexpr size_t cursor = offsetof(SchedulerTaskControl, next_block);
    constexpr size_t completed = offsetof(SchedulerTaskControl, completed_blocks);
    EXPECT_EQ(cursor % 128, 0u);
    EXPECT_EQ(completed % 128, 0u);
    EXPECT_NE(cursor / 128, completed / 128);
    EXPECT_NE(cursor / 128, offsetof(SchedulerTaskControl, state) / 128);
    EXPECT_NE(completed / 128, offsetof(SchedulerTaskControl, next_fanin_index) / 128);
    EXPECT_LE(cursor + 128, sizeof(SchedulerTaskControl));
    EXPECT_LE(completed + 128, sizeof(SchedulerTaskControl));
    EXPECT_EQ(sizeof(SchedulerTaskControl) % 128, 0u);
}

TEST(HbgSpmdLayout, AdmissionPhaseOwnsAnIsolatedBypassPublicationRegion) {
    constexpr size_t phase = offsetof(SchedulerTaskControl, admission_state);
    EXPECT_EQ(phase % 128, 0u);
    for (size_t atomic :
         {offsetof(SchedulerTaskControl, state), offsetof(SchedulerTaskControl, claim_tickets),
          offsetof(SchedulerTaskControl, next_block), offsetof(SchedulerTaskControl, completed_blocks)})
        EXPECT_NE(phase / 128, atomic / 128);
    EXPECT_LE(phase + 128, sizeof(SchedulerTaskControl));
}

TEST(HbgSpmdLayout, CounterUpdatesLeaveOtherTaskAndMetadataBytesUntouched) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 0, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    std::array<uint8_t, 3 * sizeof(SchedulerTaskControl)> expected{};
    std::memcpy(expected.data(), controls, expected.size());
    for (uint64_t task = 0; task < 3; ++task) {
        const uint64_t cursor = (UINT64_C(1) << 32) + task;
        const uint64_t completed = (UINT64_C(1) << 33) + task;
        EXPECT_EQ(scheduler_gm_compare_exchange(controls[task].next_block, UINT64_C(0), cursor), 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(controls[task].completed_blocks, completed), 0u);
        const size_t row = task * sizeof(SchedulerTaskControl);
        std::memcpy(expected.data() + row + offsetof(SchedulerTaskControl, next_block), &cursor, sizeof(cursor));
        std::memcpy(
            expected.data() + row + offsetof(SchedulerTaskControl, completed_blocks), &completed, sizeof(completed)
        );
        EXPECT_EQ(std::memcmp(expected.data(), controls, expected.size()), 0);
    }
}

TEST(HbgSpmdLayout, AdmissionPublicationAndLockLeaveAllOtherTaskBytesUntouched) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 0, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    std::array<uint8_t, 3 * sizeof(SchedulerTaskControl)> expected{};
    std::memcpy(expected.data(), controls, expected.size());
    for (uint64_t task = 0; task < 3; ++task) {
        const uint64_t phase = UINT64_MAX - task;
        const uint64_t lock = (UINT64_C(1) << 32) + task;
        EXPECT_EQ(scheduler_gm_compare_exchange(controls[task].admission_claim_lock, UINT64_C(0), lock), 0u);
        scheduler_publish_signal(controls[task].admission_state, phase);
        EXPECT_EQ(scheduler_gm_query(controls[task].admission_state), phase);
        const size_t row = task * sizeof(SchedulerTaskControl);
        std::memcpy(expected.data() + row + offsetof(SchedulerTaskControl, admission_state), &phase, sizeof(phase));
        std::memcpy(expected.data() + row + offsetof(SchedulerTaskControl, admission_claim_lock), &lock, sizeof(lock));
        EXPECT_EQ(std::memcmp(expected.data(), controls, expected.size()), 0);
        scheduler_gm_store(controls[task].admission_claim_lock, UINT64_C(0));
        const uint64_t unlocked = 0;
        std::memcpy(
            expected.data() + row + offsetof(SchedulerTaskControl, admission_claim_lock), &unlocked, sizeof(unlocked)
        );
        EXPECT_EQ(std::memcmp(expected.data(), controls, expected.size()), 0);
    }
}

TEST(HbgSpmdLayout, IsolatesAdmissionAndMailboxAtomicOwners) {
    EXPECT_EQ(sizeof(SchedulerAdmissionRecord), 64u);
    EXPECT_EQ(sizeof(SchedulerAdmissionControl), 256u);
    EXPECT_EQ(alignof(SchedulerAdmissionControl), 128u);
    EXPECT_EQ(offsetof(SchedulerAdmissionControl, commit_lock), 0u);
    EXPECT_EQ(offsetof(SchedulerAdmissionControl, commit_tail), 128u);
    EXPECT_EQ(sizeof(SchedulerGroupDescriptor), 128u);
    EXPECT_EQ(sizeof(SchedulerGroupLaneAck), 128u);
    EXPECT_EQ(sizeof(SchedulerGroupMailbox), 640u);
    EXPECT_EQ(alignof(SchedulerGroupMailbox), 128u);
    EXPECT_EQ(offsetof(SchedulerGroupMailbox, descriptor), 0u);
    EXPECT_EQ(offsetof(SchedulerGroupMailbox, published_seq), 128u);
    EXPECT_EQ(offsetof(SchedulerGroupMailbox, lanes), 256u);
    EXPECT_EQ(sizeof(SchedulerGraphSummary), 128u);
    EXPECT_TRUE((std::is_standard_layout_v<SchedulerAdmissionRecord>));
    EXPECT_TRUE((std::is_trivially_copyable_v<SchedulerAdmissionRecord>));
    EXPECT_TRUE((std::is_standard_layout_v<SchedulerGroupMailbox>));
    EXPECT_TRUE((std::is_trivially_copyable_v<SchedulerGroupMailbox>));
}

TEST(HbgSpmdLayout, ZeroRequirementsPreserveLegacyAllocation) {
    SchedulerSpmdRequirements spmd;
    EXPECT_EQ(spmd.worker_count, 0u);
    EXPECT_EQ(spmd.cluster_count, 0u);
    EXPECT_EQ(spmd.cooperative_task_count, 0u);
    EXPECT_EQ(spmd.mix_task_count, 0u);
    AicoreSchedulerLayout implicit{};
    AicoreSchedulerLayout explicit_zero{};
    ASSERT_TRUE(scheduler_plan_layout(5, 2, 3, &implicit));
    ASSERT_TRUE(scheduler_plan_layout(5, 2, 3, &explicit_zero, false, 0, spmd));
    EXPECT_EQ(std::memcmp(&implicit, &explicit_zero, sizeof(implicit)), 0);
    expect_no_protocol_storage(implicit);
    scheduler_test::SchedulerStateBuffer storage(implicit);
    ASSERT_NE(storage.base(), nullptr);
    expect_empty_protocol(storage.base(), implicit);
}

class HbgSpmdWorkerLayout : public ::testing::TestWithParam<uint64_t> {};

TEST_P(HbgSpmdWorkerLayout, OrdinaryGraphHasNoProtocolAllocation) {
    AicoreSchedulerLayout ordinary{};
    AicoreSchedulerLayout legacy{};
    ASSERT_TRUE(scheduler_plan_layout(5, 2, 3, &ordinary, false, 0, requirements(GetParam(), 0, 0)));
    ASSERT_TRUE(scheduler_plan_layout(5, 2, 3, &legacy));
    expect_no_protocol_storage(ordinary);
    EXPECT_EQ(ordinary.total_size, legacy.total_size);
    scheduler_test::SchedulerStateBuffer storage(ordinary);
    ASSERT_NE(storage.base(), nullptr);
    expect_empty_protocol(storage.base(), ordinary);
}

TEST_P(HbgSpmdWorkerLayout, PureSpmdUsesActualMembershipWidthWithoutMixStorage) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(9, 5, 4, &layout, false, 0, requirements(GetParam(), 3, 0)));
    EXPECT_EQ(layout.admission_member_word_count, (GetParam() + 63) / 64);
    EXPECT_EQ(layout.admission_record_stride, 128u);
    EXPECT_EQ(layout.admission_records_offset % 128, 0u);
    EXPECT_EQ(layout.admission_control_offset % 128, 0u);
    EXPECT_NE(layout.admission_log_offset, 0u);
    EXPECT_EQ(layout.group_mailboxes_offset, 0u);
    EXPECT_EQ(layout.mix_close_seals_offset, 0u);
    EXPECT_EQ(layout.mix_close_seal_count, 0u);
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    expect_empty_protocol(storage.base(), layout);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto *record = record_at(storage.base(), layout, task);
        const uint64_t row = layout.admission_records_offset + task * layout.admission_record_stride;
        EXPECT_EQ(row % 128, 0u);
        EXPECT_LE(
            record->member_words_offset + layout.admission_member_word_count * sizeof(uint64_t),
            row + layout.admission_record_stride
        );
    }
}

TEST_P(HbgSpmdWorkerLayout, MixRegionsAreAlignedDisjointAndSizedToActualTopology) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(9, 9, 9, &layout, false, 0, requirements(GetParam(), 3, 2)));
    EXPECT_EQ(layout.spmd.worker_count, GetParam());
    EXPECT_EQ(layout.spmd.cluster_count, GetParam() / 3);
    EXPECT_EQ(layout.mix_close_seal_count, 2 * (GetParam() / 3));
    EXPECT_EQ(layout.group_mailboxes_offset % 128, 0u);
    EXPECT_EQ(layout.mix_close_seals_offset % alignof(uint64_t), 0u);
    std::vector<std::pair<uint64_t, uint64_t>> regions{
        {layout.admission_control_offset, sizeof(SchedulerAdmissionControl)},
        {layout.admission_records_offset, layout.task_count * layout.admission_record_stride},
        {layout.admission_log_offset, layout.spmd.cooperative_task_count * sizeof(int64_t)},
        {layout.group_mailboxes_offset, layout.spmd.cluster_count * sizeof(SchedulerGroupMailbox)},
        {layout.mix_close_seals_offset, layout.mix_close_seal_count * sizeof(uint64_t)},
        {layout.graph_summary_offset, sizeof(SchedulerGraphSummary)},
        {layout.task_metadata_offset, layout.task_count * sizeof(SchedulerTaskMetadata)},
        {layout.task_controls_offset, layout.task_count * sizeof(SchedulerTaskControl)},
    };
    std::sort(regions.begin(), regions.end());
    for (size_t index = 0; index < regions.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_LE(regions[index].first + regions[index].second, layout.total_size);
        if (index + 1 < regions.size())
            EXPECT_LE(regions[index].first + regions[index].second, regions[index + 1].first);
    }
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    expect_empty_protocol(storage.base(), layout);
}

TEST_P(HbgSpmdWorkerLayout, SecondInitializationErasesEveryDirtyProtocolByte) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(7, 7, 7, &layout, true, 31, requirements(GetParam(), 4, 2)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    constexpr uint64_t device_base = UINT64_C(0x400000000);
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
    std::vector<uint8_t> expected(static_cast<size_t>(layout.total_size));
    std::memcpy(expected.data(), storage.base(), expected.size());
    std::memset(storage.base(), 0xa5, expected.size());
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
    EXPECT_EQ(std::memcmp(expected.data(), storage.base(), expected.size()), 0);
    expect_empty_protocol(storage.base(), layout);
}

INSTANTIATE_TEST_SUITE_P(ActualTopology, HbgSpmdWorkerLayout, ::testing::Values(3u, 63u, 66u, 96u, 108u));

TEST(HbgSpmdLayout, RetainedAllocationRebindsLargeSmallLargeWithoutTouchingInactiveTail) {
    AicoreSchedulerLayout large{};
    AicoreSchedulerLayout small{};
    ASSERT_TRUE(scheduler_plan_layout(9, 9, 9, &large, true, 57, requirements(108, 4, 2)));
    ASSERT_TRUE(scheduler_plan_layout(2, 2, 2, &small, false, 5, requirements(3, 1, 0)));
    ASSERT_GT(large.total_size, small.total_size);
    ASSERT_GE(large.group_mailboxes_offset, small.total_size);
    ASSERT_GE(large.mix_close_seals_offset, small.total_size);
    scheduler_test::SchedulerStateBuffer retained(large);
    scheduler_test::SchedulerStateBuffer fresh_small(small);
    scheduler_test::SchedulerStateBuffer fresh_large(large);
    ASSERT_NE(retained.base(), nullptr);
    ASSERT_NE(fresh_small.base(), nullptr);
    ASSERT_NE(fresh_large.base(), nullptr);
    constexpr uint64_t device_base = UINT64_C(0x400000000);
    const auto fill_large_metadata = [&large](void *base, uint16_t blocks, uint8_t pure_mask) {
        for (uint64_t task = 0; task < large.task_count; ++task)
            set_metadata(base, large, task, 1, 1);
        set_metadata(base, large, 1, blocks, pure_mask);
        set_metadata(base, large, 3, blocks, 3);
        set_metadata(base, large, 5, blocks, 2);
        set_metadata(base, large, 8, blocks, 7);
    };
    ASSERT_TRUE(scheduler_init_data_from_layout(retained.base(), large, device_base));
    fill_large_metadata(retained.base(), 3, 1);
    ASSERT_TRUE(scheduler_init_spmd_task_records(retained.base(), large, 117));
    auto *old_mailboxes = scheduler_state_at<SchedulerGroupMailbox>(retained.base(), large.group_mailboxes_offset);
    for (uint64_t cluster = 0; cluster < large.spmd.cluster_count; ++cluster) {
        old_mailboxes[cluster].published_seq = cluster + 400;
        old_mailboxes[cluster].descriptor.run_epoch = 117;
        for (uint64_t lane = 0; lane < 3; ++lane) {
            old_mailboxes[cluster].lanes[lane].arrived_seq = cluster + lane + 200;
            old_mailboxes[cluster].lanes[lane].released_seq = cluster + lane + 100;
        }
    }
    auto *old_seals = scheduler_state_at<uint64_t>(retained.base(), large.mix_close_seals_offset);
    for (uint64_t index = 0; index < large.mix_close_seal_count; ++index)
        old_seals[index] = index + 1;
    auto *retained_bytes = static_cast<uint8_t *>(retained.base());
    retained_bytes[small.total_size] = 0xa5;
    retained_bytes[large.total_size - 1] = 0x5a;
    const std::vector<uint8_t> inactive_tail(retained_bytes + small.total_size, retained_bytes + large.total_size);

    ASSERT_TRUE(scheduler_init_data_from_layout(retained.base(), small, device_base));
    ASSERT_TRUE(scheduler_init_data_from_layout(fresh_small.base(), small, device_base));
    EXPECT_EQ(std::memcmp(retained.base(), fresh_small.base(), static_cast<size_t>(small.total_size)), 0);
    expect_empty_protocol(retained.base(), small);
    for (void *base : {retained.base(), fresh_small.base()}) {
        set_metadata(base, small, 0, 2, 2);
        set_metadata(base, small, 1, 1, 1);
        ASSERT_TRUE(scheduler_init_spmd_task_records(base, small, 232));
    }
    EXPECT_EQ(std::memcmp(retained.base(), fresh_small.base(), static_cast<size_t>(small.total_size)), 0);
    EXPECT_EQ(std::memcmp(retained_bytes + small.total_size, inactive_tail.data(), inactive_tail.size()), 0);
    expect_summary_configuration(retained.base(), small, 232);
    EXPECT_EQ(record_at(retained.base(), small, 0)->logical_block_num, 2u);
    EXPECT_EQ(record_at(retained.base(), small, 0)->active_mask, 2u);
    EXPECT_EQ(record_at(retained.base(), small, 1)->task_ref, SCHEDULER_TASK_REF_INVALID);

    ASSERT_TRUE(scheduler_init_data_from_layout(retained.base(), large, device_base));
    ASSERT_TRUE(scheduler_init_data_from_layout(fresh_large.base(), large, device_base));
    EXPECT_EQ(std::memcmp(retained.base(), fresh_large.base(), static_cast<size_t>(large.total_size)), 0);
    expect_empty_protocol(retained.base(), large);
    for (void *base : {retained.base(), fresh_large.base()}) {
        fill_large_metadata(base, 7, 4);
        ASSERT_TRUE(scheduler_init_spmd_task_records(base, large, 349));
    }
    EXPECT_EQ(std::memcmp(retained.base(), fresh_large.base(), static_cast<size_t>(large.total_size)), 0);
    expect_summary_configuration(retained.base(), large, 349);
    EXPECT_EQ(record_at(retained.base(), large, 0)->task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(record_at(retained.base(), large, 1)->logical_block_num, 7u);
    EXPECT_EQ(record_at(retained.base(), large, 1)->active_mask, 4u);
    EXPECT_EQ(record_at(retained.base(), large, 1)->run_epoch, 0u);
    EXPECT_EQ(
        record_at(retained.base(), large, 8)->mix_close_seals_offset,
        large.mix_close_seals_offset + large.spmd.cluster_count * sizeof(uint64_t)
    );
}

class HbgSpmdInvalidRequirements : public ::testing::TestWithParam<SchedulerSpmdRequirements> {};

TEST_P(HbgSpmdInvalidRequirements, RejectsWithoutChangingOutput) {
    AicoreSchedulerLayout layout{};
    std::memset(&layout, 0xa5, sizeof(layout));
    std::array<uint8_t, sizeof(layout)> before{};
    std::memcpy(before.data(), &layout, sizeof(layout));
    EXPECT_FALSE(scheduler_plan_layout(5, 5, 5, &layout, false, 0, GetParam()));
    EXPECT_EQ(std::memcmp(before.data(), &layout, sizeof(layout)), 0);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidTopologyAndCounts, HbgSpmdInvalidRequirements,
    ::testing::Values(
        SchedulerSpmdRequirements{0, 0, 1, 0}, SchedulerSpmdRequirements{0, 1, 0, 0},
        SchedulerSpmdRequirements{1, 0, 0, 0}, SchedulerSpmdRequirements{2, 0, 0, 0},
        SchedulerSpmdRequirements{3, 0, 0, 0}, SchedulerSpmdRequirements{3, 2, 0, 0},
        SchedulerSpmdRequirements{64, 21, 1, 0}, SchedulerSpmdRequirements{109, 36, 1, 0},
        SchedulerSpmdRequirements{111, 37, 1, 0}, SchedulerSpmdRequirements{108, 35, 1, 0},
        SchedulerSpmdRequirements{108, 36, 6, 0}, SchedulerSpmdRequirements{108, 36, 1, 2},
        SchedulerSpmdRequirements{108, 36, 0, 1}, SchedulerSpmdRequirements{UINT64_MAX, UINT64_MAX, 1, 0},
        SchedulerSpmdRequirements{108, 36, UINT64_MAX, 0}, SchedulerSpmdRequirements{108, 36, 5, UINT64_MAX}
    )
);

TEST(HbgSpmdLayout, RejectsExistingCountAndTraceOverflowWithoutChangingOutput) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 1, &layout, false, 0, requirements(3, 1, 1)));
    std::array<uint8_t, sizeof(layout)> before{};
    std::memcpy(before.data(), &layout, sizeof(layout));
    for (const auto &counts : std::vector<std::array<uint64_t, 4>>{
             {static_cast<uint64_t>(INT32_MAX) + 1, 0, 0, 0},
             {UINT64_MAX, 0, 0, 0},
             {1, 2, 0, 0},
             {1, 0, 2, 0},
             {1, 1, 1, UINT64_MAX},
             {1, 1, 1, UINT64_MAX / sizeof(SchedulerTaskTrace) + 1},
         }) {
        SCOPED_TRACE(counts[0]);
        SCOPED_TRACE(counts[3]);
        EXPECT_FALSE(
            scheduler_plan_layout(counts[0], counts[1], counts[2], &layout, false, counts[3], requirements(3, 1, 1))
        );
        EXPECT_EQ(std::memcmp(before.data(), &layout, sizeof(layout)), 0);
    }
    EXPECT_FALSE(scheduler_plan_layout(1, 1, 1, nullptr, false, 0, requirements(3, 1, 1)));
}

TEST(HbgSpmdLayout, EmptyGraphWithKnownTopologyHasNoProtocolStorage) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(0, 0, 0, &layout, false, 0, requirements(108, 0, 0)));
    expect_no_protocol_storage(layout);
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, UINT64_MAX));
    expect_summary_configuration(storage.base(), layout, UINT64_MAX);
}

TEST(HbgSpmdLayout, AdmissionLogScalesWithCooperativeTasksNotAllTasksOrWorkerCapacity) {
    AicoreSchedulerLayout one_cooperative{};
    AicoreSchedulerLayout seventeen_cooperative{};
    ASSERT_TRUE(scheduler_plan_layout(65, 65, 0, &one_cooperative, false, 0, requirements(108, 1, 0)));
    ASSERT_TRUE(scheduler_plan_layout(65, 65, 0, &seventeen_cooperative, false, 0, requirements(108, 17, 0)));
    EXPECT_EQ(one_cooperative.admission_records_offset, seventeen_cooperative.admission_records_offset);
    EXPECT_EQ(one_cooperative.admission_record_stride, seventeen_cooperative.admission_record_stride);
    EXPECT_EQ(one_cooperative.admission_log_offset, seventeen_cooperative.admission_log_offset);
    EXPECT_EQ(seventeen_cooperative.total_size - one_cooperative.total_size, 128u);
    EXPECT_EQ(
        one_cooperative.admission_log_offset - one_cooperative.admission_records_offset,
        65 * one_cooperative.admission_record_stride
    );
}

TEST(HbgSpmdLayout, DeviceRelocationKeepsAllProtocolReferencesAsOffsets) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &layout, false, 0, requirements(66, 2, 1)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    for (uint64_t device_base : {UINT64_C(0x400000000), UINT64_C(0x800000000)}) {
        SCOPED_TRACE(device_base);
        ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, device_base));
        set_metadata(storage.base(), layout, 0, 3, 1);
        set_metadata(storage.base(), layout, 1, 2, 7);
        set_metadata(storage.base(), layout, 2, 1, 2);
        ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, 9));
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const auto *record = record_at(storage.base(), layout, task);
            EXPECT_EQ(
                record->member_words_offset, layout.admission_records_offset + task * layout.admission_record_stride +
                                                 sizeof(SchedulerAdmissionRecord)
            );
            EXPECT_LT(record->member_words_offset, layout.total_size);
        }
        EXPECT_EQ(record_at(storage.base(), layout, 0)->mix_close_seals_offset, 0u);
        EXPECT_EQ(record_at(storage.base(), layout, 1)->mix_close_seals_offset, layout.mix_close_seals_offset);
        expect_summary_configuration(storage.base(), layout, 9);
        const auto *payload = scheduler_state_at<DispatchPayload>(storage.base(), layout.dispatch_payloads_offset);
        EXPECT_EQ(
            payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX],
            device_base + layout.dispatch_payloads_offset + offsetof(DispatchPayload, local_context)
        );
    }
}

TEST(HbgSpmdTaskRecords, InitializesOnlyCooperativeTasksAndPacksOneSealRowPerMixTask) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(8, 8, 8, &layout, false, 0, requirements(96, 4, 2)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 1, 1);
    set_metadata(storage.base(), layout, 1, 5, 1, SCHEDULER_TASK_HAS_FANIN);
    set_metadata(storage.base(), layout, 2, 1, 6);
    set_metadata(storage.base(), layout, 3, 2, 2, SCHEDULER_TASK_SYNC_START);
    set_metadata(storage.base(), layout, 4, 17, 7);
    set_metadata(storage.base(), layout, 5, 0, 0, SCHEDULER_TASK_INLINE);
    set_metadata(storage.base(), layout, 7, 1, 4);
    constexpr uint64_t epoch = 123;
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, epoch));
    expect_summary_configuration(storage.base(), layout, epoch);
    const auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(storage.base(), layout.task_metadata_offset);
    for (uint64_t task : {1u, 2u, 3u, 4u}) {
        SCOPED_TRACE(task);
        const auto *record = record_at(storage.base(), layout, task);
        EXPECT_EQ(record->task_ref, static_cast<int64_t>(task));
        EXPECT_EQ(record->logical_block_num, metadata[task].logical_block_num);
        EXPECT_EQ(record->active_mask, metadata[task].active_mask);
        EXPECT_EQ(record->flags, metadata[task].flags);
        EXPECT_EQ(record->run_epoch, 0u);
        EXPECT_EQ(record->order_key, UINT64_MAX);
        EXPECT_EQ(record->member_count, 0u);
        const auto *words = scheduler_state_at<uint64_t>(storage.base(), record->member_words_offset);
        for (uint64_t word = 0; word < layout.admission_member_word_count; ++word)
            EXPECT_EQ(words[word], 0u);
    }
    for (uint64_t task : {0u, 5u, 6u, 7u})
        EXPECT_EQ(record_at(storage.base(), layout, task)->task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(record_at(storage.base(), layout, 1)->mix_close_seals_offset, 0u);
    EXPECT_EQ(record_at(storage.base(), layout, 3)->mix_close_seals_offset, 0u);
    EXPECT_EQ(record_at(storage.base(), layout, 2)->mix_close_seals_offset, layout.mix_close_seals_offset);
    EXPECT_EQ(
        record_at(storage.base(), layout, 4)->mix_close_seals_offset,
        layout.mix_close_seals_offset + layout.spmd.cluster_count * sizeof(uint64_t)
    );
    const auto *control =
        scheduler_state_at<SchedulerAdmissionControl>(storage.base(), layout.admission_control_offset);
    EXPECT_EQ(control->commit_lock, 0u);
    EXPECT_EQ(control->commit_tail, 0u);
    const auto *log = scheduler_state_at<int64_t>(storage.base(), layout.admission_log_offset);
    for (uint64_t index = 0; index < layout.spmd.cooperative_task_count; ++index)
        EXPECT_EQ(log[index], SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgSpmdTaskRecords, EpochIsSuppliedExactlyWithoutHostPublicationOrIncrement) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 0, requirements(3, 1, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    for (uint64_t epoch : {UINT64_C(0), UINT64_C(1), UINT64_MAX - 1, UINT64_MAX}) {
        SCOPED_TRACE(epoch);
        ASSERT_TRUE(
            scheduler_init_data_from_layout(storage.base(), layout, reinterpret_cast<uint64_t>(storage.base()))
        );
        set_metadata(storage.base(), layout, 0, 2, 1);
        ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, epoch));
        expect_summary_configuration(storage.base(), layout, epoch);
        EXPECT_EQ(record_at(storage.base(), layout, 0)->run_epoch, 0u);
        EXPECT_EQ(record_at(storage.base(), layout, 0)->order_key, UINT64_MAX);
    }
}

struct InvalidTaskShape {
    uint16_t blocks;
    uint8_t mask;
    uint8_t flags;
};

class HbgSpmdInvalidTaskShape : public ::testing::TestWithParam<InvalidTaskShape> {};

TEST_P(HbgSpmdInvalidTaskShape, RejectsBeforePublishingAnyRecordOrEpoch) {
    AicoreSchedulerLayout layout{};
    const uint64_t declared_mix = (GetParam().flags & SCHEDULER_TASK_MIX) != 0 ? 1 : 0;
    ASSERT_TRUE(scheduler_plan_layout(2, 2, 2, &layout, false, 0, requirements(108, 2, declared_mix)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 2, 1);
    set_metadata(storage.base(), layout, 1, GetParam().blocks, GetParam().mask);
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(storage.base(), layout.task_metadata_offset);
    metadata[1].flags = GetParam().flags;
    std::vector<uint8_t> before(static_cast<size_t>(layout.total_size));
    std::memcpy(before.data(), storage.base(), before.size());
    EXPECT_FALSE(scheduler_init_spmd_task_records(storage.base(), layout, 99));
    EXPECT_EQ(std::memcmp(before.data(), storage.base(), before.size()), 0);
}

INSTANTIATE_TEST_SUITE_P(
    ShapeAndFlags, HbgSpmdInvalidTaskShape,
    ::testing::Values(
        InvalidTaskShape{0, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX},
        InvalidTaskShape{2, 0, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD},
        InvalidTaskShape{2, 8, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD},
        InvalidTaskShape{2, UINT8_MAX, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX},
        InvalidTaskShape{2, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD},
        InvalidTaskShape{2, 1, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX},
        InvalidTaskShape{2, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX},
        InvalidTaskShape{1, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX},
        InvalidTaskShape{1, 1, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SYNC_START},
        InvalidTaskShape{1, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX | SCHEDULER_TASK_SYNC_START},
        InvalidTaskShape{2, 1, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_HAS_PREDICATE},
        InvalidTaskShape{
            2, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX | SCHEDULER_TASK_HAS_PREDICATE
        },
        InvalidTaskShape{1, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX | SCHEDULER_TASK_HAS_PREDICATE},
        InvalidTaskShape{16384, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX},
        InvalidTaskShape{10923, 7, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX},
        InvalidTaskShape{UINT16_MAX, 3, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX}
    )
);

TEST(HbgSpmdTaskRecords, CooperativeAndMixCountMismatchDoesNotPartiallyInitialize) {
    for (const auto &spmd : {
             requirements(3, 1, 0),
             requirements(3, 1, 1),
             requirements(3, 2, 0),
             requirements(3, 2, 2),
             requirements(3, 3, 1),
         }) {
        SCOPED_TRACE(spmd.cooperative_task_count);
        SCOPED_TRACE(spmd.mix_task_count);
        AicoreSchedulerLayout layout{};
        ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &layout, false, 0, spmd));
        scheduler_test::SchedulerStateBuffer storage(layout);
        ASSERT_NE(storage.base(), nullptr);
        set_metadata(storage.base(), layout, 0, 2, 1);
        set_metadata(storage.base(), layout, 1, 1, 3);
        set_metadata(storage.base(), layout, 2, 1, 2);
        std::vector<uint8_t> before(static_cast<size_t>(layout.total_size));
        std::memcpy(before.data(), storage.base(), before.size());
        EXPECT_FALSE(scheduler_init_spmd_task_records(storage.base(), layout, 55));
        EXPECT_EQ(std::memcmp(before.data(), storage.base(), before.size()), 0);
    }
}

TEST(HbgSpmdTaskRecords, MissingProtocolStorageRejectsCooperativeMetadata) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 0, requirements(3, 0, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 2, 1);
    EXPECT_FALSE(scheduler_init_spmd_task_records(storage.base(), layout, 1));
    expect_summary_configuration(storage.base(), layout, 0);
}

TEST(HbgSpmdTaskRecords, OrdinaryPredicateAndSkippedEntriesNeedNoAdmissionStorage) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(4, 4, 4, &layout, false, 0, requirements(3, 0, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 1, 1, SCHEDULER_TASK_HAS_PREDICATE);
    set_metadata(storage.base(), layout, 1, 1, 2, SCHEDULER_TASK_HAS_PREDICATE);
    set_metadata(storage.base(), layout, 2, 0, 0, SCHEDULER_TASK_INLINE);
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, 37));
    expect_summary_configuration(storage.base(), layout, 37);
    expect_no_protocol_storage(layout);
}

TEST(HbgSpmdTaskRecords, RejectsNullStorage) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 0, requirements(3, 1, 0)));
    EXPECT_FALSE(scheduler_init_spmd_task_records(nullptr, layout, 1));
}

TEST(HbgSpmdTaskRecords, RejectsUnalignedStorageWithoutChangingState) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 0, requirements(3, 1, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 2, 1);
    std::vector<uint8_t> before(static_cast<size_t>(layout.total_size));
    std::memcpy(before.data(), storage.base(), before.size());
    for (uint64_t displacement : {1u, 8u, 64u, 127u}) {
        SCOPED_TRACE(displacement);
        EXPECT_FALSE(
            scheduler_init_spmd_task_records(static_cast<uint8_t *>(storage.base()) + displacement, layout, 1)
        );
        EXPECT_EQ(std::memcmp(before.data(), storage.base(), before.size()), 0);
    }
}

TEST(HbgSpmdTaskRecords, RejectsTimingSlotsOutsideTaskAttrsRangeWithoutPartialPublication) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(2, 2, 0, &layout, false, 0, requirements(3, 2, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, 2, 1);
    set_metadata(storage.base(), layout, 1, 2, 1);
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(storage.base(), layout.task_metadata_offset);
    for (int16_t timing_slot : {-2, INT16_MIN, 16, INT16_MAX}) {
        SCOPED_TRACE(timing_slot);
        metadata[1].timing_slot = timing_slot;
        std::vector<uint8_t> before(static_cast<size_t>(layout.total_size));
        std::memcpy(before.data(), storage.base(), before.size());
        EXPECT_FALSE(scheduler_init_spmd_task_records(storage.base(), layout, 5));
        EXPECT_EQ(std::memcmp(before.data(), storage.base(), before.size()), 0);
    }
}

TEST(HbgSpmdTaskRecords, AcceptsMaximumRepresentablePureAndMixSubtaskCounts) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(7, 7, 7, &layout, false, 0, requirements(108, 7, 4)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    uint64_t task = 0;
    for (uint8_t mask : {1u, 2u, 4u})
        set_metadata(storage.base(), layout, task++, INT16_MAX, mask);
    for (uint8_t mask : {3u, 5u, 6u})
        set_metadata(storage.base(), layout, task++, 16383, mask);
    set_metadata(storage.base(), layout, task, 10922, 7);
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, 11));
    for (task = 0; task < layout.task_count; ++task) {
        SCOPED_TRACE(task);
        const auto *record = record_at(storage.base(), layout, task);
        EXPECT_EQ(record->task_ref, static_cast<int64_t>(task));
        EXPECT_EQ(record->logical_block_num, task < 3 ? INT16_MAX : (task < 6 ? 16383 : 10922));
    }
}

TEST(HbgSpmdTaskRecords, DispatchMetadataAlwaysCopiesLogicalBlockCountAndTimingSlot) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 0, requirements(3, 1, 0)));
    scheduler_test::SchedulerStateBuffer storage(layout);
    ASSERT_NE(storage.base(), nullptr);
    set_metadata(storage.base(), layout, 0, INT16_MAX, 1);
    auto *source = scheduler_state_at<SchedulerTaskMetadata>(storage.base(), layout.task_metadata_offset);
    source->trace_index_base = UINT32_MAX;
    SchedulerLocalState local{};
    local.config.task_metadata_offset = layout.task_metadata_offset;
    for (int16_t timing_slot : {-1, 0, 15}) {
        source->timing_slot = timing_slot;
        for (uint64_t profiling : {0u, 1u, 3u}) {
            SCOPED_TRACE(timing_slot);
            SCOPED_TRACE(profiling);
            const auto loaded = scheduler_load_dispatch_metadata(storage.base(), &local, 0, profiling);
            EXPECT_EQ(loaded.logical_block_num, INT16_MAX);
            EXPECT_EQ(loaded.timing_slot, timing_slot);
            EXPECT_EQ(loaded.active_mask, source->active_mask);
            EXPECT_EQ(loaded.flags, source->flags);
            for (uint8_t lane = 0; lane < 3; ++lane)
                EXPECT_EQ(loaded.kernel_ids[lane], source->kernel_ids[lane]);
            EXPECT_EQ(loaded.trace_index_base, profiling == 0 ? 0u : UINT32_MAX);
        }
    }
}

struct SubtaskCountCase {
    uint64_t blocks;
    uint32_t active;
    uint32_t expected;
};

class HbgSpmdValidSubtaskCount : public ::testing::TestWithParam<SubtaskCountCase> {};

TEST_P(HbgSpmdValidSubtaskCount, AcceptsRepresentablePositiveProduct) {
    uint32_t result = UINT32_MAX;
    ASSERT_TRUE(scheduler_subtask_count_checked(GetParam().blocks, GetParam().active, &result));
    EXPECT_EQ(result, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    NarrowWireBoundary, HbgSpmdValidSubtaskCount,
    ::testing::Values(
        SubtaskCountCase{1, 1, 1}, SubtaskCountCase{1, 2, 2}, SubtaskCountCase{1, 3, 3},
        SubtaskCountCase{INT16_MAX, 1, INT16_MAX}, SubtaskCountCase{16383, 2, 32766}, SubtaskCountCase{10922, 3, 32766},
        SubtaskCountCase{108, 3, 324}
    )
);

class HbgSpmdInvalidSubtaskCount : public ::testing::TestWithParam<SubtaskCountCase> {};

TEST_P(HbgSpmdInvalidSubtaskCount, FailurePreservesOutput) {
    uint32_t result = UINT32_C(0xa5a5a5a5);
    EXPECT_FALSE(scheduler_subtask_count_checked(GetParam().blocks, GetParam().active, &result));
    EXPECT_EQ(result, UINT32_C(0xa5a5a5a5));
}

INSTANTIATE_TEST_SUITE_P(
    ZeroInvalidWidthAndOverflow, HbgSpmdInvalidSubtaskCount,
    ::testing::Values(
        SubtaskCountCase{0, 1, 0}, SubtaskCountCase{0, 3, 0}, SubtaskCountCase{1, 0, 0}, SubtaskCountCase{1, 4, 0},
        SubtaskCountCase{1, UINT32_MAX, 0}, SubtaskCountCase{32768, 1, 0}, SubtaskCountCase{16384, 2, 0},
        SubtaskCountCase{10923, 3, 0}, SubtaskCountCase{UINT64_MAX, 1, 0}, SubtaskCountCase{UINT64_MAX, 3, 0},
        SubtaskCountCase{UINT64_MAX / 3 + 1, 3, 0}
    )
);

struct TraceCountCase {
    uint64_t prefix;
    uint64_t added;
    uint64_t expected;
};

class HbgSpmdValidTraceCount : public ::testing::TestWithParam<TraceCountCase> {};

TEST_P(HbgSpmdValidTraceCount, AllowsLastAddressableTraceIncludingCapacityEndpoint) {
    uint64_t result = UINT64_MAX;
    ASSERT_TRUE(scheduler_trace_count_checked(GetParam().prefix, GetParam().added, &result));
    EXPECT_EQ(result, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    Uint32IndexBoundary, HbgSpmdValidTraceCount,
    ::testing::Values(
        TraceCountCase{0, 1, 1}, TraceCountCase{0, UINT32_MAX, UINT32_MAX},
        TraceCountCase{0, UINT64_C(1) << 32, UINT64_C(1) << 32}, TraceCountCase{UINT32_MAX, 1, UINT64_C(1) << 32},
        TraceCountCase{UINT32_MAX - 1, 2, UINT64_C(1) << 32}, TraceCountCase{UINT32_MAX - 1, 1, UINT32_MAX},
        TraceCountCase{17, 32766, 32783}
    )
);

class HbgSpmdInvalidTraceCount : public ::testing::TestWithParam<TraceCountCase> {};

TEST_P(HbgSpmdInvalidTraceCount, FailurePreservesOutput) {
    uint64_t result = UINT64_C(0xa5a5a5a5a5a5a5a5);
    EXPECT_FALSE(scheduler_trace_count_checked(GetParam().prefix, GetParam().added, &result));
    EXPECT_EQ(result, UINT64_C(0xa5a5a5a5a5a5a5a5));
}

INSTANTIATE_TEST_SUITE_P(
    ZeroAddInvalidPrefixAndOverflow, HbgSpmdInvalidTraceCount,
    ::testing::Values(
        TraceCountCase{0, 0, 0}, TraceCountCase{UINT32_MAX, 0, 0}, TraceCountCase{UINT32_MAX, 2, 0},
        TraceCountCase{UINT64_C(1) << 32, 1, 0}, TraceCountCase{UINT64_MAX, 1, 0},
        TraceCountCase{0, (UINT64_C(1) << 32) + 1, 0}, TraceCountCase{0, UINT64_MAX, 0},
        TraceCountCase{1, UINT64_MAX, 0}
    )
);

TEST(HbgSpmdCheckedCounts, RejectsNullOutputPointers) {
    EXPECT_FALSE(scheduler_subtask_count_checked(1, 1, nullptr));
    EXPECT_FALSE(scheduler_trace_count_checked(0, 1, nullptr));
}

class HbgProtocolTraceLayout : public ::testing::TestWithParam<uint64_t> {};

TEST_P(HbgProtocolTraceLayout, ReservesIndexedRecordsRatherThanPerWorkerCopiesOfTheBlockHistory) {
    AicoreSchedulerLayout plain{};
    AicoreSchedulerLayout full{};
    const auto spmd = requirements(GetParam(), 2, 1);
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &plain, true, 7, spmd));
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &full, true, 7, spmd, true));
    EXPECT_EQ(plain.protocol_layout_offset, 0u);
    EXPECT_EQ(plain.protocol.magic, 0u);
    EXPECT_EQ(full.protocol_layout_offset, plain.total_size);
    const auto &p = full.protocol;
    EXPECT_EQ(p.magic, SCHEDULER_PROTOCOL_MAGIC);
    EXPECT_EQ(p.task_count, 3u);
    EXPECT_EQ(p.worker_count, GetParam());
    EXPECT_EQ(p.cooperative_task_count, 2u);
    EXPECT_EQ(p.member_count, 2 * GetParam());
    EXPECT_EQ(p.trace_count, 7u);
    const uint64_t expected = sizeof(SchedulerProtocolLayout) + 3 * sizeof(SchedulerProtocolTaskTrace) +
                              2 * GetParam() * sizeof(SchedulerProtocolMemberTrace) +
                              7 * sizeof(SchedulerProtocolBlockTrace) + 7 * sizeof(SchedulerProtocolExecutionTrace) +
                              GetParam() * sizeof(SchedulerProtocolWorkerTrace);
    EXPECT_EQ(full.total_size - plain.total_size, expected);
    EXPECT_EQ(p.storage_end_offset, full.total_size);
    std::vector<std::pair<uint64_t, uint64_t>> regions{
        {full.protocol_layout_offset, sizeof(SchedulerProtocolLayout)},
        {p.tasks_offset, p.task_count * sizeof(SchedulerProtocolTaskTrace)},
        {p.members_offset, p.member_count * sizeof(SchedulerProtocolMemberTrace)},
        {p.blocks_offset, p.trace_count * sizeof(SchedulerProtocolBlockTrace)},
        {p.executions_offset, p.trace_count * sizeof(SchedulerProtocolExecutionTrace)},
        {p.workers_offset, p.worker_count * sizeof(SchedulerProtocolWorkerTrace)},
    };
    std::sort(regions.begin(), regions.end());
    for (size_t index = 0; index < regions.size(); ++index) {
        EXPECT_EQ(regions[index].first % 128, 0u);
        EXPECT_LE(regions[index].first + regions[index].second, full.total_size);
        if (index + 1 < regions.size())
            EXPECT_LE(regions[index].first + regions[index].second, regions[index + 1].first);
    }
    scheduler_test::SchedulerStateBuffer storage(full);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(storage.base(), full.graph_summary_offset);
    EXPECT_EQ(summary->protocol_layout_offset, full.protocol_layout_offset);
    const auto *wire = scheduler_state_at<SchedulerProtocolLayout>(storage.base(), full.protocol_layout_offset);
    EXPECT_EQ(std::memcmp(wire, &p, sizeof(p)), 0);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(storage.base(), p.workers_offset);
    for (uint64_t worker = 0; worker < p.worker_count; ++worker) {
        EXPECT_EQ(workers[worker].worker_index, worker);
        EXPECT_EQ(workers[worker].next_sequence, 0u);
        EXPECT_EQ(workers[worker].finished, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(ActualWorkers, HbgProtocolTraceLayout, ::testing::Values(3u, 63u, 66u, 96u, 108u));

TEST(HbgProtocolTrace, PreservesRunAndTraceWireSizesAndSeparatesConcurrentWriters) {
    EXPECT_EQ(sizeof(SchedulerGraphSummary), 128u);
    EXPECT_EQ(sizeof(SchedulerRunControl), 512u);
    EXPECT_EQ(sizeof(SchedulerTaskTrace), 384u);
    EXPECT_EQ(offsetof(SchedulerGraphSummary, protocol_layout_offset), 96u);
    EXPECT_EQ(offsetof(SchedulerProtocolTaskTrace, completion), 64u);
    EXPECT_EQ(offsetof(SchedulerProtocolBlockTrace, lanes), 256u);
    EXPECT_EQ(sizeof(SchedulerProtocolLaneTrace), 256u);
}

TEST(HbgProtocolTrace, RetainedResetErasesAllEvidenceAndBindsTheLiteralNewEpoch) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &layout, false, 7, requirements(6, 2, 1), true));
    scheduler_test::SchedulerStateBuffer storage(layout);
    const auto init = [&](uint64_t epoch) {
        EXPECT_TRUE(scheduler_init_data_from_layout(storage.base(), layout, UINT64_C(0x400000000)));
        set_metadata(storage.base(), layout, 0, 2, 1);
        set_metadata(storage.base(), layout, 1, 2, 3);
        set_metadata(storage.base(), layout, 2, 1, 2);
        EXPECT_TRUE(scheduler_init_spmd_task_records(storage.base(), layout, epoch));
    };
    for (uint64_t epoch : {UINT64_C(0), UINT64_MAX}) {
        init(epoch);
        std::vector<uint8_t> expected(static_cast<size_t>(layout.total_size));
        std::memcpy(expected.data(), storage.base(), expected.size());
        std::memset(storage.base(), 0xa5, expected.size());
        init(epoch);
        EXPECT_EQ(std::memcmp(storage.base(), expected.data(), expected.size()), 0);
        const auto *p = scheduler_state_at<SchedulerProtocolLayout>(storage.base(), layout.protocol_layout_offset);
        EXPECT_EQ(p->run_epoch, epoch);
        const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(storage.base(), p->workers_offset);
        for (uint64_t worker = 0; worker < 6; ++worker) {
            EXPECT_EQ(workers[worker].run_epoch, epoch);
            EXPECT_EQ(workers[worker].next_sequence, 0u);
            EXPECT_EQ(workers[worker].error_flags, 0u);
            EXPECT_EQ(workers[worker].finished, 0u);
        }
    }
}

TEST(HbgProtocolTrace, AdditionalTraceStorageOverflowPreservesTheCallersLayout) {
    const uint64_t records = UINT64_MAX / 1024;
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, records, requirements(3, 1, 0)));
    const auto previous = layout;
    EXPECT_FALSE(scheduler_plan_layout(1, 1, 0, &layout, false, records, requirements(3, 1, 0), true));
    EXPECT_EQ(std::memcmp(&layout, &previous, sizeof(layout)), 0);
}

TEST(HbgProtocolTrace, DisabledRunDoesNotExposeOrClearAFormerLargerProtocolRegion) {
    AicoreSchedulerLayout full{}, plain{};
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &full, false, 7, requirements(6, 2, 1), true));
    ASSERT_TRUE(scheduler_plan_layout(3, 3, 3, &plain, false, 7, requirements(6, 2, 1)));
    scheduler_test::SchedulerStateBuffer storage(full);
    auto *bytes = static_cast<uint8_t *>(storage.base());
    std::memset(bytes + plain.total_size, 0xa5, static_cast<size_t>(full.total_size - plain.total_size));
    ASSERT_TRUE(scheduler_init_data_from_layout(storage.base(), plain, UINT64_C(0x400000000)));
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(storage.base(), plain.graph_summary_offset);
    EXPECT_EQ(summary->protocol_layout_offset, 0u);
    EXPECT_EQ(bytes[plain.total_size], 0xa5);
    EXPECT_EQ(bytes[full.total_size - 1], 0xa5);
}

TEST(HbgProtocolTrace, RequiresAnActualTopologyEvenForOrdinaryOnlyGraphs) {
    AicoreSchedulerLayout layout{};
    EXPECT_FALSE(scheduler_plan_layout(1, 1, 0, &layout, false, 1, {}, true));
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, false, 1, requirements(3, 0, 0), true));
    EXPECT_EQ(layout.protocol.member_count, 0u);
    EXPECT_EQ(layout.protocol.worker_count, 3u);
}

}  // namespace
