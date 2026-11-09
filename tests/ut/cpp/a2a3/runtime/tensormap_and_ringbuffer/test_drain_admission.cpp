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

#include "support/mix_lane_priority_fixture.h"

class DrainAdmissionTestPeer {
public:
    static int held(SchedulerContext &context) { return context.retained_ready_[0].count; }
    static void add_idle_peer(SchedulerContext &context) {
        context.active_sched_threads_ = 2;
        auto &peer = context.core_trackers_[1];
        peer.init(1);
        peer.set_cluster(0, 3, 4, 5);
    }
    static void dispatch_peer(SchedulerContext &context) {
        bool progress = false, pushed = false;
        context.dispatch_ready_tasks(1, context.core_trackers_[1], false, progress, pushed);
        EXPECT_FALSE(progress);
        EXPECT_FALSE(pushed);
    }
    static void occupy(SchedulerContext &context, int lane, bool pending) {
        if (!pending) context.core_trackers_[0].change_core_state(lane);
        context.core_trackers_[0].set_pending_occupied(lane);
    }
    static void release(SchedulerContext &context, int lane, bool pending) {
        context.core_trackers_[0].clear_pending_occupied(lane);
        if (!pending) context.core_trackers_[0].change_core_state(lane);
    }
    static void early(SchedulerContext &context) {
        bool progress = false, pushed = false;
        context.try_early_dispatch(0, context.core_trackers_[0], false, progress, pushed);
    }
    static void hold(SchedulerContext &context, ChipTaskSlotState &slot) {
        int32_t start = 0;
        const int32_t count = slot.claim_block_range(slot.logical_block_num, 1, start);
        ASSERT_EQ(count, 1);
        context.retain_ready_range(0, &slot, start, count);
    }
};

TEST_P(LanePriority, ClosedDrainRetainsClaimUntilReopen) {
    const auto [bits, pending, early] = GetParam();
    (void)early;
    enqueue_mix();
    SchedulerContextTestPeer::drain(*context, -1);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    ASSERT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(SchedulerContextTestPeer::placed(*context, lane, pending), nullptr);
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    SchedulerContextTestPeer::drain(*context, 0);
    observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(
            SchedulerContextTestPeer::placed(*context, lane, pending), bits & (1 << lane) ? &mixed.slot : nullptr
        );
}

TEST_P(LanePriority, RejectedRangeKeepsRemainingBlocksAvailableToPeers) {
    const auto [bits, pending, early] = GetParam();
    (void)bits;
    (void)early;
    mixed.slot.logical_block_num = 3;
    mixed.slot.total_required_subtasks = 3 * mixed.slot.active_mask.count();
    enqueue_mix();
    SchedulerContextTestPeer::drain(*context, -1);
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    ASSERT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    CapturedReadySlot entry;
    ASSERT_EQ(sched->ready_queues[2].pop_shaped_batch(&entry, nullptr, 1, ResourceShape::MIX), 1);
    ASSERT_EQ(entry.ptr(), &mixed.slot);
    EXPECT_EQ(entry.tag(), mixed.slot.active_mask);
    int32_t start = -1;
    EXPECT_EQ(entry.ptr()->claim_block_range(3, 3, start), 2);
    EXPECT_EQ(start, 1);
    SchedulerContextTestPeer::drain(*context, 0);
    observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 3);
}

TEST_P(LanePriority, NonFittingRegularMixReturnsWithoutClaiming) {
    const auto [bits, pending, early] = GetParam();
    (void)early;
    const int lane = bits & 1 ? 0 : 1;
    SchedulerContextTestPeer::block_lane(*context, lane, &pure.slot);
    enqueue_mix();
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, false, observed);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    EXPECT_EQ(sched->ready_queues[2].size(), 1u);
    EXPECT_EQ(sched->ready_queues[2].mix_demand(), mixed.slot.active_mask);
}

TEST_P(LanePriority, RetainedMixReservesOnlyItsLanesBelowReadySync) {
    DrainAdmissionTestPeer::hold(*context, mixed.slot);
    LaneMask observed = LaneMask::none();
    EXPECT_EQ(SchedulerContextTestPeer::allowed(*context, 0, observed), LaneMask::all());
    EXPECT_EQ(
        SchedulerContextTestPeer::allowed(*context, 1, observed), LaneMask::all().without(mixed.slot.active_mask)
    );
    observed = LaneMask::none();
    EXPECT_EQ(
        SchedulerContextTestPeer::allowed(*context, 2, observed), LaneMask::all().without(mixed.slot.active_mask)
    );
}

TEST_P(LanePriority, RetainedReadyPrecedesOverlappingEarlyMix) {
    const auto [bits, pending, full_chain] = GetParam();
    const int consumer_bits = bits == 6 ? 3 : 6;
    const int blocked_lane = bits == 6 ? 2 : 0;
    pure.init(consumer_bits, 1, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    DrainAdmissionTestPeer::hold(*context, mixed.slot);
    DrainAdmissionTestPeer::occupy(*context, blocked_lane, pending);
    ASSERT_TRUE(sched->early_dispatch_queues[2].push_mix(&pure.slot, pure.descriptor.task_id));

    SchedulerContextTestPeer::ready(*context, false);
    ASSERT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    LaneMask observed = LaneMask::none();
    if (full_chain) DrainAdmissionTestPeer::early(*context);
    else SchedulerContextTestPeer::leaf(*context, ResourceShape::MIX, pending, true, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(sched->early_dispatch_queues[2].size(), 1u);
    EXPECT_EQ(pure.payload.staged_core_mask[0].load(), 0u);

    DrainAdmissionTestPeer::release(*context, blocked_lane, pending);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    for (int lane = 0; lane < 3; ++lane)
        if (bits & (1 << lane)) EXPECT_EQ(SchedulerContextTestPeer::placed(*context, lane, pending), &mixed.slot);
}

TEST_P(LanePriority, EmptyEarlyPassAllowsLaterPublication) {
    const auto [bits, pending, early] = GetParam();
    pure.init(2, 1, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    DrainAdmissionTestPeer::early(*context);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    ASSERT_TRUE(sched->early_dispatch_queues[1].push_tagged(&pure.slot, pure.descriptor.task_id));
    DrainAdmissionTestPeer::early(*context);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 1);
    EXPECT_EQ(SchedulerContextTestPeer::placed(*context, 1, pending), &pure.slot);
}

TEST_P(LanePriority, RetainedPendingReplayDoesNotWaitForIdlePeer) {
    const auto [bits, pending, early] = GetParam();
    if (!pending) GTEST_SKIP() << "Requires occupied running slots";
    DrainAdmissionTestPeer::add_idle_peer(*context);
    mixed.slot.task_attrs.set_early_resolve(true);
    DrainAdmissionTestPeer::hold(*context, mixed.slot);
    DrainAdmissionTestPeer::dispatch_peer(*context);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(mixed.payload.published_block_count.load(), 1);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(SchedulerContextTestPeer::placed(*context, lane, true), bits & (1 << lane) ? &mixed.slot : nullptr);
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 1);
    EXPECT_EQ(mixed.payload.published_block_count.load(), 1);
    EXPECT_EQ(sched->ready_queues[2].size(), 0u);
}

TEST_P(LanePriority, SharedPendingMixStillYieldsToIdlePeer) {
    const auto [bits, pending, early] = GetParam();
    if (!pending) GTEST_SKIP() << "Requires occupied running slots";
    DrainAdmissionTestPeer::add_idle_peer(*context);
    enqueue_mix();
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(sched->ready_queues[2].size(), 1u);
}

TEST_P(LanePriority, PmuStillBlocksRetainedPendingReplay) {
    const auto [bits, pending, early] = GetParam();
    if (!pending) GTEST_SKIP() << "Requires occupied running slots";
    DrainAdmissionTestPeer::add_idle_peer(*context);
    DrainAdmissionTestPeer::hold(*context, mixed.slot);
    SchedulerContextTestPeer::ready(*context, true);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 1);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(SchedulerContextTestPeer::placed(*context, lane, true), nullptr);
}

TEST_P(LanePriority, CompletedRetainedReplayKeepsSharedPeerPolicy) {
    const auto [bits, pending, early] = GetParam();
    if (pending) GTEST_SKIP() << "Requires idle running slots";
    DrainAdmissionTestPeer::add_idle_peer(*context);
    DrainAdmissionTestPeer::hold(*context, mixed.slot);
    pure.init(bits, 1, 1);
    ASSERT_TRUE(sched->ready_queues[2].push_mix(&pure.slot, pure.descriptor.task_id));
    SchedulerContextTestPeer::ready(*context, false);
    EXPECT_EQ(DrainAdmissionTestPeer::held(*context), 0);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(sched->ready_queues[2].size(), 1u);
    for (int lane = 0; lane < 3; ++lane)
        EXPECT_EQ(SchedulerContextTestPeer::placed(*context, lane, true), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    DrainLanes, LanePriority, testing::Combine(testing::Values(3, 5, 6, 7), testing::Bool(), testing::Bool())
);
