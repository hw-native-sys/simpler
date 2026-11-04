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

TEST_P(LanePriority, PurePlacementsUseOnlySparePhysicalLanes) {
    const auto [bits, pending, early] = GetParam();
    enqueue_mix(early);
    const int pure_bits = bits == 6 ? 1 : 2;
    pure.init(pure_bits, 2, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    const auto shape = pure.slot.active_mask.to_shape();
    auto &q =
        early ? sched->early_dispatch_queues[static_cast<int>(shape)] : sched->ready_queues[static_cast<int>(shape)];
    ASSERT_TRUE(q.push_shaped(&pure.slot, shape, pure.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::leaf(*context, shape, pending, early, observed);
    const int expected = bits == 7 ? 0 : 1;
    EXPECT_EQ(pure.slot.next_block_idx.load(), expected);
    for (int lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(
            SchedulerContextTestPeer::placed(*context, lane, pending),
            expected && !(bits & (1 << lane)) ? &pure.slot : nullptr
        ) << lane;
    }
}

TEST_P(LanePriority, PriorityPrefixAndAttemptCarryAreDirectional) {
    const auto [bits, pending, early] = GetParam();
    (void)pending;
    (void)early;
    mixed.slot.task_attrs.set_sync_start();
    ASSERT_TRUE(sched->ready_sync_queues[2].push_mix(&mixed.slot));
    TestTask later;
    later.init(bits == 3 ? 5 : 3, 1, 2);
    ASSERT_TRUE(sched->early_dispatch_queues[2].push_mix(&later.slot, later.descriptor.task_id));
    LaneMask observed = LaneMask::none();
    EXPECT_EQ(SchedulerContextTestPeer::allowed(*context, 0, observed), LaneMask::all().without(mask_of(bits)));
    CapturedReadySlot popped;
    ASSERT_EQ(sched->ready_sync_queues[2].pop_shaped_batch(&popped, nullptr, 1, ResourceShape::MIX), 1);
    EXPECT_EQ(SchedulerContextTestPeer::allowed(*context, 1, observed), LaneMask::all().without(mask_of(bits)));
    observed = LaneMask::none();
    EXPECT_EQ(SchedulerContextTestPeer::allowed(*context, 1, observed), LaneMask::all());
    EXPECT_EQ(
        SchedulerContextTestPeer::allowed(*context, 2, observed), LaneMask::all().without(later.slot.active_mask)
    );
    SchedulerContextTestPeer::drain(*context, -1);
    EXPECT_EQ(SchedulerContextTestPeer::allowed(*context, 0, observed), LaneMask::none());
}

TEST_P(LanePriority, LaterComplementaryDemandClosesRemainingLane) {
    const auto [bits, pending, early] = GetParam();
    enqueue_mix(early);
    LaneMask observed = LaneMask::none();
    const int priority = early ? 2 : 1;
    (void)SchedulerContextTestPeer::allowed(*context, priority, observed);
    TestTask later;
    later.init(bits == 3 ? 5 : 3, 1, 2);
    auto &q = early ? sched->early_dispatch_queues[2] : sched->ready_queues[2];
    ASSERT_TRUE(q.push_mix(&later.slot));
    pure.init(bits == 6 ? 1 : 2, 1, 1);
    pure.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    auto shape = pure.slot.active_mask.to_shape();
    auto &pure_q =
        early ? sched->early_dispatch_queues[static_cast<int>(shape)] : sched->ready_queues[static_cast<int>(shape)];
    ASSERT_TRUE(pure_q.push_shaped(&pure.slot, shape, pure.descriptor.task_id));
    SchedulerContextTestPeer::leaf(*context, shape, pending, early, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
}

TEST_P(LanePriority, FullRequeueReportsOverflowAndStopsInsteadOfSpinning) {
    auto &q = sched->ready_queues[1];
    pure.init(2, 1, 1);
    for (int i = 0; i < capacity; ++i)
        ASSERT_TRUE(q.push(&pure.slot));
    const auto value = CapturedReadySlot(&pure.slot, LaneMask::none());
    EXPECT_FALSE(SchedulerContextTestPeer::requeue(*context, q, &value, 1, ResourceShape::AIV));
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_TRUE(SchedulerContextTestPeer::stopped(*context));
    EXPECT_EQ(header.sched_error_thread.load(), 0);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u);
    EXPECT_FALSE(SchedulerContextTestPeer::requeue(*context, q, &value, 1, ResourceShape::AIV, 3));
    EXPECT_EQ(header.sched_error_thread.load(), 0);
    EXPECT_EQ(header.sched_error_bitmap.load(), (1u << 0) | (1u << 3));
}

TEST_P(LanePriority, CapturedMaskAndIdSurviveMutationAndHybridPureEntry) {
    auto &q = sched->early_sync_start_queue;
    pure.init(2, 1, 1);
    ASSERT_TRUE(q.push_sync_start(&mixed.slot, mixed.descriptor.task_id));
    ASSERT_TRUE(q.push_sync_start(&pure.slot, pure.descriptor.task_id));
    const LaneMask captured = mixed.slot.active_mask;
    mixed.slot.active_mask = LaneMask::all();
    EXPECT_EQ(q.mix_demand(), captured);
    TaskId id;
    const auto value = q.pop_sync_start(&id);
    EXPECT_EQ(value.ptr(), &mixed.slot);
    EXPECT_EQ(id, mixed.descriptor.task_id);
    EXPECT_EQ(value.tag(), captured);
    EXPECT_EQ(q.mix_demand(), LaneMask::none());
    ASSERT_TRUE(q.push_sync_start(value, id));
    EXPECT_EQ(q.mix_demand(), captured);
    const auto zero = q.pop_sync_start(&id);
    EXPECT_EQ(zero.ptr(), &pure.slot);
    EXPECT_EQ(zero.tag(), LaneMask::none());
    EXPECT_EQ(q.mix_demand(), captured);
    (void)q.pop_sync_start(&id);
    EXPECT_EQ(q.mix_demand(), LaneMask::none());
}

TEST_P(LanePriority, PureSyncCohortNeverPartiallyClaimsOrEscapesThroughDrain) {
    const auto [bits, pending, early] = GetParam();
    (void)pending;
    (void)early;
    SchedulerContextTestPeer::bind(*context, *sched, callables, registers.data(), registers.size() / 3, false);
    mixed.slot.task_attrs.set_sync_start();
    ASSERT_TRUE(sched->ready_sync_queues[2].push_mix(&mixed.slot));
    pure.init(2, 2, 1);
    pure.slot.task_attrs.set_sync_start();
    ASSERT_TRUE(sched->ready_sync_queues[1].push(&pure.slot));
    LaneMask observed = LaneMask::none();
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 0);
    EXPECT_EQ(SchedulerContextTestPeer::drain_pending(*context), 0);
    EXPECT_EQ(sched->ready_sync_queues[1].size(), 1);
    CapturedReadySlot value;
    ASSERT_EQ(sched->ready_sync_queues[2].pop_shaped_batch(&value, nullptr, 1, ResourceShape::MIX), 1);
    observed = LaneMask::none();
    SchedulerContextTestPeer::sync_leaf(*context, observed);
    EXPECT_EQ(pure.slot.next_block_idx.load(), 2);
    EXPECT_EQ(SchedulerContextTestPeer::placed(*context, 1, false), &pure.slot);
    EXPECT_EQ(SchedulerContextTestPeer::placed(*context, 2, false), &pure.slot);
}

TEST_P(LanePriority, NormalOrderRetainsResidualAndPmuIdleOpportunity) {
    const auto [bits, pending, early] = GetParam();
    (void)pending;
    // Rebind idle placement independently of the parameterized pending-only leaf cases.
    const size_t words = registers.size() / 3;
    SchedulerContextTestPeer::bind(*context, *sched, callables, registers.data(), words, false);
    enqueue_mix();
    const int blocked = bits & 1 ? 0 : 1;
    SchedulerContextTestPeer::block_lane(*context, blocked, &mixed.slot);
    pure.init(bits == 6 ? 1 : 2, 1, 1);
    ASSERT_TRUE(sched->ready_queues[static_cast<int>(pure.slot.active_mask.to_shape())].push(&pure.slot));
    SchedulerContextTestPeer::ready(*context, early);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(pure.slot.next_block_idx.load(), bits == 7 ? 0 : 1);
}

TEST_P(LanePriority, ConcurrentMixedQueueReleasesEveryCapturedContribution) {
    auto &q = sched->ready_queues[2];
    TestTask other;
    other.init(5, 1, 2);
    constexpr int per_producer = 1024;
    std::atomic<int> popped{0};
    auto producer = [&](TestTask &task) {
        for (int i = 0; i < per_producer; ++i)
            while (!q.push_mix(&task.slot, task.descriptor.task_id))
                std::this_thread::yield();
    };
    auto consumer = [&]() {
        CapturedReadySlot values[3];
        TaskId ids[3];
        while (popped.load() < 2 * per_producer) {
            const int count = q.pop_shaped_batch(values, ids, 3, ResourceShape::MIX);
            for (int i = 0; i < count; ++i) {
                EXPECT_TRUE(values[i].ptr() == &mixed.slot || values[i].ptr() == &other.slot);
                EXPECT_EQ(values[i].tag(), values[i].ptr()->active_mask);
                EXPECT_EQ(ids[i], values[i].ptr() == &mixed.slot ? mixed.descriptor.task_id : other.descriptor.task_id);
            }
            popped.fetch_add(count);
        }
    };
    std::thread p0(producer, std::ref(mixed)), p1(producer, std::ref(other)), c0(consumer), c1(consumer);
    p0.join();
    p1.join();
    c0.join();
    c1.join();
    EXPECT_EQ(popped.load(), 2 * per_producer);
    EXPECT_EQ(q.mix_demand(), LaneMask::none());
    EXPECT_EQ(q.size(), 0);
}

TEST_P(LanePriority, ReadySyncResidualProtectsTheLowerRegularTier) {
    const auto [bits, pending, pmu] = GetParam();
    (void)pending;
    SchedulerContextTestPeer::bind(*context, *sched, callables, registers.data(), registers.size() / 3, false);
    mixed.slot.task_attrs.set_sync_start();
    ASSERT_TRUE(sched->ready_sync_queues[2].push_mix(&mixed.slot));
    SchedulerContextTestPeer::block_lane(*context, bits & 1 ? 0 : 1, &mixed.slot);
    pure.init(bits == 6 ? 1 : 2, 1, 1);
    ASSERT_TRUE(sched->ready_queues[static_cast<int>(pure.slot.active_mask.to_shape())].push(&pure.slot));
    SchedulerContextTestPeer::ready(*context, pmu);
    EXPECT_EQ(mixed.slot.next_block_idx.load(), 0);
    EXPECT_EQ(pure.slot.next_block_idx.load(), bits == 7 ? 0 : 1);
}

TEST_P(LanePriority, CancelledEarlySyncRequeuesOriginalTagAndIdentity) {
    mixed.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    const auto captured =
        CapturedReadySlot(&mixed.slot, (mixed.slot.active_mask).is_mix() ? (mixed.slot.active_mask) : LaneMask::none());
    const auto original_id = mixed.descriptor.task_id;
    ASSERT_TRUE(SchedulerState::try_claim_early_sync_drain(mixed.payload));
    mixed.init(7, 1, 7);
    sched->cancel_early_sync_drain(mixed.slot, captured, original_id);
    EXPECT_EQ(sched->early_sync_start_queue.mix_demand(), captured.tag());
    TaskId id;
    const auto popped = sched->early_sync_start_queue.pop_sync_start(&id);
    EXPECT_EQ(popped.ptr(), &mixed.slot);
    EXPECT_EQ(popped.tag(), captured.tag());
    EXPECT_EQ(id, original_id);
    EXPECT_NE(id, mixed.descriptor.task_id);
    EXPECT_EQ(sched->early_sync_start_queue.mix_demand(), LaneMask::none());
}

INSTANTIATE_TEST_SUITE_P(
    PhysicalLanes, LanePriority, testing::Combine(testing::Values(3, 5, 6, 7), testing::Bool(), testing::Bool())
);
