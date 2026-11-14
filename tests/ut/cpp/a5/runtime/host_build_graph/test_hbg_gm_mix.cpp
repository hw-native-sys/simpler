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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "async_completion_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_async_finish.h"
#include "scheduler/scheduler_gm_mix.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "scheduler/scheduler_args_dump.h"

#include "hbg_gm_mix_test_support.h"

namespace {

using namespace scheduler_test::gm_mix;

TEST(HbgGmMixArgsDump, LastSnapshotHoldsExistingCompletionAcrossThreeWaves) {
    GmMixFixture fixture({{3, LaneMask::all()}, {1, LaneMask::aiv0()}}, false, 1);
    fixture.graph.executable(1, 1, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *local = fixture.locals[0];
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    SchedulerArgsDumpTask dump{};
    dump.task_ref = 0;
    dump.active_mask = 7;
    dump.expected_participants = 9;
    ASSERT_EQ(scheduler_args_dump_enter(&dump), SchedulerArgsDumpArrival::Publisher);
    dump.publisher_before = scheduler_args_dump_publisher(0);
    SchedulerArgsDumpAck before{11, 0, 0, 1, 1, 0, 0, 0};
    before.status = scheduler_args_dump_ack_commit(0, 0, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_seal(&dump, ArgsDumpStage::BEFORE_DISPATCH, before, 11, 1));
    std::array<uint32_t, 9> visible{};
    GmExecution held{};
    for (uint32_t block = 0; block < 3; ++block) {
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            ASSERT_EQ(scheduler_args_dump_enter(&dump), SchedulerArgsDumpArrival::Ready);
            const uint32_t index = block * 3 + worker;
            const auto arrival = scheduler_args_dump_return(&dump, [&] {
                visible[index] = index + 1;
                return true;
            });
            if (index == 8) {
                ASSERT_EQ(arrival, SchedulerArgsDumpArrival::Publisher);
                held = execution;
            } else {
                ASSERT_EQ(arrival, SchedulerArgsDumpArrival::Waiting);
                ASSERT_TRUE(fixture.complete(execution));
            }
        }
        ASSERT_EQ(fixture.retire(0), block == 2 ? SchedulerGmMixResult::WAIT : SchedulerGmMixResult::PROGRESS);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), 1);
        EXPECT_EQ(scheduler_observe_next_waiter(fixture.control(1)), SCHEDULER_WAKE_LIST_OPEN);
        EXPECT_EQ(local->pending_completed, 0u);
    }
    EXPECT_EQ(visible, (std::array<uint32_t, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 8u);
    dump.publisher_after = scheduler_args_dump_publisher(held.worker);
    SchedulerArgsDumpAck after{11, 0, 1, scheduler_args_dump_publisher(held.worker), 1, 0, 0, 0};
    after.status = scheduler_args_dump_ack_commit(0, 1, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_seal(&dump, ArgsDumpStage::AFTER_COMPLETION, after, 11, 1));
    ASSERT_TRUE(fixture.complete(held));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.locals[held.worker]->pending_completed, 1u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 9u);
    uint32_t successor_entries = 0;
    for (uint32_t step = 0; step < 8; ++step) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 1);
            ++successor_entries;
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(successor_entries, 1u);
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

class HbgGmMixResourceScope : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixResourceScope, UnusedVectorDoesNotBlockAOneCubeOneVectorGroup) {
    const LaneMask mask = GetParam();
    const uint32_t vector = mask == kAicAiv0 ? 1 : 2;
    const uint32_t unused = 3 - vector;
    GmMixFixture fixture({{2, mask}}, false, 1);
    fixture.locals[unused]->slots[1].state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.admissions[unused].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[unused].scan_cursor, 1u);
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        EXPECT_EQ(fixture.mailbox(0)->descriptor.resource_mask, mask);
        EXPECT_EQ(fixture.mailbox(0)->lanes[vector].assigned_seq, block + 1);
        EXPECT_EQ(fixture.mailbox(0)->lanes[unused].assigned_seq, 0u);
        bool progress = false;
        ASSERT_TRUE(fixture.observe(vector, &progress));
        ASSERT_TRUE(progress);
        ASSERT_TRUE(fixture.arrive(0, &progress));
        ASSERT_TRUE(fixture.prepare(0, &progress));
        EXPECT_FALSE(progress);
        GmExecution pending{};
        EXPECT_FALSE(fixture.begin(0, &pending));
        ASSERT_TRUE(fixture.arrive(vector, &progress));
        ASSERT_TRUE(progress);
        for (uint32_t worker : {0u, vector}) {
            ASSERT_TRUE(fixture.prepare(worker, &progress));
            ASSERT_TRUE(progress);
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            EXPECT_EQ(execution.block, block);
            EXPECT_EQ(execution.blocks, 2u);
            ASSERT_TRUE(fixture.complete(execution));
        }
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    }
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u * mask.count());
    EXPECT_EQ(fixture.mailbox(0)->lanes[unused].arrived_seq, 0u);
    EXPECT_EQ(fixture.mailbox(0)->lanes[unused].released_seq, 0u);
    EXPECT_EQ(fixture.locals[unused]->slots[1].state, SchedulerDispatchSlotState::FILLING);
    for (uint32_t worker : {0u, vector}) {
        bool closed = false;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

INSTANTIATE_TEST_SUITE_P(SingleVectorMasks, HbgGmMixResourceScope, ::testing::Values(kAicAiv0, kAicAiv1));

TEST(HbgGmMixResourceScopeTransition, SkippedSingleVectorGroupsCanRejoinTheNextFullGroup) {
    const std::vector<GmShape> shapes{
        {2, kAicAiv0}, {3, kAicAiv1}, {1, kAicAiv0}, {2, LaneMask::all()}, {2, LaneMask::both_aiv()}
    };
    GmMixFixture fixture(shapes, false, 1);
    ASSERT_TRUE(fixture.bootstrap_admission());
    for (uint32_t task = 0; task < shapes.size(); ++task)
        ASSERT_TRUE(fixture.admit(task));
    for (uint32_t task = 0; task < shapes.size(); ++task) {
        const LaneMask resources = shapes[task].mask.merged_with(LaneMask::aic());
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (resources.has(static_cast<SubtaskSlot>(lane)))
                EXPECT_EQ(fixture.admissions[lane].current_task_ref, task);
            else EXPECT_NE(fixture.admissions[lane].current_task_ref, task);
        }
        for (uint32_t block = 0; block < shapes[task].blocks; ++block) {
            ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(0));
            bool progress = false;
            for (uint32_t lane = 1; lane < 3; ++lane)
                if (resources.has(static_cast<SubtaskSlot>(lane))) ASSERT_TRUE(fixture.observe(lane, &progress));
            for (uint32_t lane = 0; lane < 3; ++lane)
                if (resources.has(static_cast<SubtaskSlot>(lane))) ASSERT_TRUE(fixture.arrive(lane, &progress));
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
                ASSERT_TRUE(fixture.prepare(lane, &progress));
                ASSERT_TRUE(progress);
                GmExecution execution{};
                const bool active = shapes[task].mask.has(static_cast<SubtaskSlot>(lane));
                ASSERT_EQ(fixture.begin(lane, &execution), active);
                if (active) ASSERT_TRUE(fixture.complete(execution));
            }
            ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        }
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
            bool closed = false;
            ASSERT_TRUE(fixture.close(lane, &closed));
            ASSERT_TRUE(closed);
        }
    }
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    for (const auto &admission : fixture.admissions) {
        EXPECT_EQ(admission.scan_cursor, shapes.size());
        EXPECT_EQ(admission.gm_mix.group_seq, 10u);
    }
}

TEST(HbgGmMixResourceScopeTransition, FutureMemberIgnoresEarlierGroupsThatDoNotUseItsLane) {
    GmMixFixture fixture({{2, kAicAiv0}, {1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.admissions[2].current_task_ref, 1);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    for (uint32_t poll = 0; poll < 4; ++poll) {
        bool progress = true;
        ASSERT_TRUE(fixture.observe(2, &progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(fixture.admissions[2].current_task_ref, 1);
        EXPECT_EQ(fixture.admissions[2].gm_mix.group_seq, 0u);
        EXPECT_EQ(fixture.mailbox(0)->lanes[2].arrived_seq, 0u);
    }
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgGmMixResourceScopeTransition, UnnotifiedVectorNeverReadsAChangingForeignDescriptor) {
    GmMixFixture fixture({{2, kAicAiv0}, {1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    // A nonmember has no lifetime claim on a predecessor's reusable descriptor.
    fixture.mailbox(0)->descriptor.run_epoch = UINT64_MAX;
    fixture.mailbox(0)->descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
    fixture.mailbox(0)->descriptor.previous_lane_seq[2] = UINT64_MAX;
    bool progress = true;
    ASSERT_TRUE(fixture.observe(2, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[2].current_task_ref, 1);
    EXPECT_EQ(fixture.admissions[2].gm_mix.group_seq, 0u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

class HbgGmMixContinuous : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixContinuous, TicketExhaustionDoesNotSealAnUnpublishedRemoteReservation) {
    for (uint16_t count : {3u, 4u, 5u}) {
        GmMixFixture fixture({{count, GetParam()}, {1, GetParam()}}, true);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.admit(1));
        ASSERT_TRUE(fixture.refresh_all());
        scheduler_gm_store(fixture.control(0)->claim_tickets, static_cast<uint64_t>(count));
        scheduler_gm_store(fixture.control(0)->next_block, static_cast<uint64_t>(count - 1));
        scheduler_gm_store(
            fixture.control(0)->completion_state,
            (fixture.control(0)->completion_state & SCHEDULER_COMPLETION_TOTAL_MASK) | static_cast<uint64_t>(count - 1)
        );
        for (uint32_t repeat = 0; repeat < 16; ++repeat)
            ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::EXHAUSTED);
        EXPECT_EQ(fixture.control(0)->claim_tickets, count + 1u);
        EXPECT_EQ(fixture.mailbox(1)->published_seq, 0u);
        for (uint32_t worker : {3u, 4u, 5u}) {
            if (!fixture.participant(worker)) continue;
            bool closed = true;
            ASSERT_TRUE(fixture.close(worker, &closed));
            EXPECT_FALSE(closed);
        }
        EXPECT_EQ(fixture.seal(0, 1), 0u);
        scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(1));
        for (uint32_t worker : {3u, 4u, 5u}) {
            if (!fixture.participant(worker)) continue;
            bool closed = false;
            ASSERT_TRUE(fixture.close(worker, &closed));
            EXPECT_TRUE(closed);
            EXPECT_FALSE(fixture.admissions[worker].ticket_exhausted);
            ASSERT_TRUE(fixture.refresh(worker));
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
        }
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), count - 1u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST_P(HbgGmMixContinuous, ReleasedGroupPublishesNextBlockInTheSameFill) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));

    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count()
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.block_idx, 1u);
    GmExecution execution{};
    EXPECT_FALSE(fixture.begin(0, &execution));
}

TEST_P(HbgGmMixContinuous, LastLocalGroupPublishesSuccessorWhileRemotePredecessorStillRuns) {
    GmMixFixture fixture({{2, GetParam()}, {1, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count()
    );
    EXPECT_NE(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);

    for (uint32_t worker : {1u, 2u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        ASSERT_TRUE(fixture.fill(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 2u);
    }
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 0);
}

TEST_P(HbgGmMixContinuous, MissingReleaseStopsContinuousProgressWithoutTakingAnotherBlock) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    const uint32_t late_lane = GetParam().has_aiv1() ? 2 : 1;
    GmExecution late{};
    ASSERT_TRUE(fixture.begin(late_lane, &late));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == late_lane || !GetParam().has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count() - 1u
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
}

TEST_P(HbgGmMixContinuous, ErrorBeforeRetirementDoesNotPublishOrContribute) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    scheduler_publish_error(fixture.run(), UINT64_C(1));
    EXPECT_FALSE(fixture.fill(0));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count()
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
}

TEST(HbgGmMix, VectorCompletionAdmitsMixWithoutAReadyQueueRoundTrip) {
    GmMixFixture fixture({{1, LaneMask::aiv0()}, {2, kAicAiv0}});
    fixture.graph.executable(1, 0, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0, 1));
    ASSERT_TRUE(fixture.fill(1));
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(1, &producer));
    ASSERT_EQ(producer.task, 0);
    ASSERT_TRUE(fixture.complete(producer));
    EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    for (const auto &candidate : fixture.admissions[1].candidates)
        EXPECT_EQ(candidate.ready.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    ASSERT_TRUE(fixture.fill(1));
    // The AIV owns admission only. AIC still publishes each MIX group and
    // every participating lane takes its admission permit.
    std::set<std::pair<uint32_t, uint8_t>> executed;
    for (uint32_t step = 0; step < 100; ++step) {
        for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution next{};
            if (!fixture.begin(worker, &next)) continue;
            ASSERT_EQ(next.task, 1);
            EXPECT_TRUE(executed.emplace(next.block, next.subtask).second);
            ASSERT_TRUE(fixture.complete(next));
        }
        if (fixture.control(1)->state == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(executed.size(), 4u);
    EXPECT_EQ(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgGmMixContinuous, DirectSuccessorFromRetirementKeepsItsAdmissionRequirement) {
    for (uint16_t blocks : {1, 2, 8, 65}) {
        SCOPED_TRACE(blocks);
        GmMixFixture fixture({{1, GetParam()}, {blocks, LaneMask::aic()}});
        fixture.graph.executable(1, 0, {0});
        fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1),
            SchedulerRouteResult::WAITING
        );
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_TRUE(fixture.fill(0));
        ASSERT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
        if (blocks > 1) {
            ASSERT_EQ(
                scheduler_gm_query(fixture.control(1)->admission_state),
                static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
            );
            for (const auto &slot : fixture.locals[0]->slots)
                if (slot.task_ref == 1 && slot.state == SchedulerDispatchSlotState::READY)
                    EXPECT_TRUE(scheduler_slot_uses_admission(fixture.locals[0], slot));
        }
        std::set<uint32_t> executed;
        for (uint32_t step = 0; step < 200; ++step) {
            for (uint32_t worker = 0; worker < 6; ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution execution{};
                if (!fixture.begin(worker, &execution)) continue;
                ASSERT_EQ(execution.task, 1);
                EXPECT_EQ(execution.blocks, blocks);
                EXPECT_EQ(
                    scheduler_slot_uses_admission(
                        fixture.locals[worker], fixture.locals[worker]->slots[execution.slot]
                    ),
                    blocks > 1
                );
                EXPECT_TRUE(executed.insert(execution.block).second);
                ASSERT_TRUE(fixture.complete(execution));
            }
            if (scheduler_gm_load(fixture.control(1)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
        }
        EXPECT_EQ(executed.size(), blocks);
        EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    }
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixContinuous, ::testing::ValuesIn(kMixMasks));

TEST(HbgGmMix, ZeroWorkContinuousProgressHasABoundedNumberOfObligations) {
    GmMixFixture fixture({{1, LaneMask::all()}, {1, LaneMask::all()}, {1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    for (uint32_t task = 0; task < 3; ++task) {
        ASSERT_TRUE(fixture.admit(task));
        ASSERT_TRUE(fixture.refresh(3));
        ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(1));
        for (uint32_t worker : {4u, 5u})
            ASSERT_TRUE(fixture.refresh(worker));
        ASSERT_TRUE(fixture.rendezvous(1));
        ASSERT_TRUE(fixture.finish_lanes(1));
        ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
        ASSERT_TRUE(fixture.task_done(task));
        for (uint32_t worker : {3u, 4u, 5u}) {
            bool closed = false;
            ASSERT_TRUE(fixture.close(worker, &closed));
            ASSERT_TRUE(closed);
        }
    }
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].scan_cursor, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(2, 0), UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 3u);
}

class HbgGmMixSyncStart : public ::testing::TestWithParam<LaneMask> {};

class HbgGmMixIdlePoll : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t>> {};

void expect_ready_pool_drained(GmMixFixture &fixture) {
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.storage.ready_pool_offset);
    for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
        const auto queue = scheduler_ready_pool_queue(fixture.base(), pool, type);
        EXPECT_EQ(scheduler_gm_query(queue.control->head), scheduler_gm_query(queue.control->reserved_tail));
    }
}

TEST_P(HbgGmMixIdlePoll, UnpublishedClaimIsIdleButDescriptorAndRetainedSealWakeTheVector) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    auto *local = fixture.locals[lane];
    const bool member = fixture.participant(lane);
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, member ? 0 : SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[lane].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local), member);
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local), member);
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
}

TEST_P(HbgGmMixIdlePoll, WaitingForPartnersDoesNotHideAnOrdinaryReadyTask) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{1, mask}, {1, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    auto *local = fixture.locals[lane];
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.enqueue(1, lane));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local));
    ASSERT_TRUE(fixture.fill(lane));
    GmExecution execution{};
    const bool entered = fixture.begin(lane, &execution);
    EXPECT_EQ(entered, !fixture.participant(lane));
    if (entered) {
        EXPECT_EQ(execution.task, 1);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(scheduler_completion_progress(fixture.control(1)->completion_state), entered ? 1u : 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndVectors, HbgGmMixIdlePoll,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint32_t>(1, 2))
);

TEST_P(HbgGmMixSyncStart, IdlePollRequiresAllArrivalAndReleaseAcknowledgements) {
    GmMixFixture fixture({{1, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    for (uint32_t lane : {1u, 2u}) {
        if (!fixture.participant(lane)) continue;
        EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
        ASSERT_TRUE(fixture.observe(lane, &progress));
        ASSERT_TRUE(fixture.arrive(lane, &progress));
    }
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.rendezvous(0));
    if (GetParam().has_aic()) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    for (uint32_t lane = 1; lane < 3; ++lane) {
        if (!GetParam().has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0]));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count());
}

struct GmServiceCase {
    uint32_t worker;
    SchedulerGmMixPhase phase;
    bool record_protocol{true};
};

void prepare_service_phase(GmMixFixture &fixture, const GmServiceCase &scenario) {
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    if (scenario.phase == SchedulerGmMixPhase::IDLE) return;
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (scenario.phase == SchedulerGmMixPhase::CLAIMED) return;
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    if (scenario.worker != 0) ASSERT_TRUE(fixture.observe(scenario.worker, &progress));
    if (scenario.phase == SchedulerGmMixPhase::PUBLISHED) return;
    if (scenario.phase == SchedulerGmMixPhase::ARRIVED) {
        ASSERT_TRUE(fixture.arrive(scenario.worker, &progress));
        return;
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    if (scenario.phase == SchedulerGmMixPhase::RELEASED) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(scenario.worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
}

class HbgGmMixServiceError : public ::testing::TestWithParam<GmServiceCase> {};

TEST_P(HbgGmMixServiceError, ServiceRejectsPublishedErrorBeforeAnyStateSpecificWork) {
    GmMixFixture fixture({{2, LaneMask::all()}});
    const auto scenario = GetParam();
    if (!scenario.record_protocol)
        for (auto *local : fixture.locals)
            local->profiling = nullptr;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
    ASSERT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, scenario.phase);
    const auto before = fixture.admissions[scenario.worker].gm_mix;
    const uint64_t cursor = fixture.control(0)->next_block;
    const uint64_t published = fixture.mailbox(0)->published_seq;
    const uint8_t ready_mask = fixture.locals[scenario.worker]->local_ready_mask;
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    bool progress = true;
    const auto service = [&](auto record) {
        return scheduler_service_gm_mix<decltype(record)::value>(
            fixture.graph.graph(), fixture.base(), fixture.locals[scenario.worker], fixture.run(), 0, &progress
        );
    };
    EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, before.phase);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.group_seq, before.group_seq);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.slot_index, before.slot_index);
    EXPECT_EQ(fixture.control(0)->next_block, cursor);
    EXPECT_EQ(
        scheduler_completion_progress(fixture.control(0)->completion_state),
        scenario.phase == SchedulerGmMixPhase::RELEASED ? 1u : 0u
    );
    EXPECT_EQ(fixture.mailbox(0)->published_seq, published);
    EXPECT_EQ(fixture.locals[scenario.worker]->local_ready_mask, ready_mask);
    EXPECT_EQ(fixture.seal(0, 0), 0u);
    EXPECT_EQ(fixture.run()->error_worker_index, 4u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

TEST_P(HbgGmMixServiceError, AHealthyServicePassDoesNotCacheTheNextPassEpochOrErrorPermission) {
    const auto scenario = GetParam();
    for (uint32_t fault = 0; fault < 3; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{2, LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
        auto *local = fixture.locals[scenario.worker];
        bool progress = false;
        const auto service = [&](auto record) {
            return scheduler_service_gm_mix<decltype(record)::value>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            );
        };
        ASSERT_TRUE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        const auto before = fixture.admissions[scenario.worker].gm_mix;
        const uint64_t cursor = fixture.control(0)->next_block;
        const uint64_t completed = scheduler_completion_progress(fixture.control(0)->completion_state);
        const uint64_t published = fixture.mailbox(0)->published_seq;
        const uint8_t ready = local->local_ready_mask;
        if (fault == 0) {
            ++fixture.admissions[scenario.worker].epoch;
        } else if (fault == 1) {
            ++scheduler_state_at<SchedulerGraphSummary>(fixture.base(), local->config.graph_summary_offset)->run_epoch;
        } else {
            scheduler_record_error(
                fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
            );
        }
        progress = true;
        EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        EXPECT_FALSE(progress);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, before.phase);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.group_seq, before.group_seq);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.slot_index, before.slot_index);
        EXPECT_EQ(fixture.control(0)->next_block, cursor);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), completed);
        EXPECT_EQ(fixture.mailbox(0)->published_seq, published);
        EXPECT_EQ(local->local_ready_mask, ready);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
        EXPECT_EQ(
            fixture.run()->error_site,
            static_cast<uint64_t>(
                fault == 2 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
            )
        );
    }
}

TEST_P(HbgGmMixServiceError, ServiceValidatesTheWholeContextBeforePhaseSpecificWork) {
    const auto scenario = GetParam();
    for (uint32_t invalid = 0; invalid < 5; ++invalid) {
        SCOPED_TRACE(invalid);
        GmMixFixture fixture({{2, LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
        auto *local = fixture.locals[scenario.worker];
        auto &admission = fixture.admissions[scenario.worker];
        const auto before = admission.gm_mix;
        const uint64_t cursor = fixture.control(0)->next_block;
        const uint64_t published = fixture.mailbox(0)->published_seq;
        const uint8_t ready = local->local_ready_mask;
        if (invalid == 0) admission.current_task_ref = SCHEDULER_TASK_REF_INVALID;
        else if (invalid == 1) admission.current_flags &= ~SCHEDULER_TASK_MIX;
        else if (invalid == 2) admission.gm_mix.mailbox_offset = 0;
        else if (invalid == 3) admission.gm_mix.cluster_index = local->config.cluster_count;
        else local->config.self_lane = PLATFORM_CORES_PER_BLOCKDIM;
        bool progress = true;
        const auto service = [&](auto record) {
            return scheduler_service_gm_mix<decltype(record)::value>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            );
        };
        EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        EXPECT_FALSE(progress);
        EXPECT_EQ(admission.gm_mix.phase, before.phase);
        EXPECT_EQ(admission.gm_mix.group_seq, before.group_seq);
        EXPECT_EQ(admission.gm_mix.slot_index, before.slot_index);
        EXPECT_EQ(fixture.control(0)->next_block, cursor);
        EXPECT_EQ(
            scheduler_completion_progress(fixture.control(0)->completion_state),
            scenario.phase == SchedulerGmMixPhase::RELEASED ? 1u : 0u
        );
        EXPECT_EQ(fixture.mailbox(0)->published_seq, published);
        EXPECT_EQ(local->local_ready_mask, ready);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

std::vector<GmServiceCase> gm_service_cases() {
    std::vector<GmServiceCase> cases{
        GmServiceCase{0, SchedulerGmMixPhase::IDLE},      GmServiceCase{0, SchedulerGmMixPhase::CLAIMED},
        GmServiceCase{0, SchedulerGmMixPhase::PUBLISHED}, GmServiceCase{0, SchedulerGmMixPhase::ARRIVED},
        GmServiceCase{0, SchedulerGmMixPhase::RUNNING},   GmServiceCase{0, SchedulerGmMixPhase::RELEASED},
        GmServiceCase{1, SchedulerGmMixPhase::IDLE},      GmServiceCase{1, SchedulerGmMixPhase::PUBLISHED},
        GmServiceCase{1, SchedulerGmMixPhase::ARRIVED},   GmServiceCase{1, SchedulerGmMixPhase::RUNNING},
        GmServiceCase{1, SchedulerGmMixPhase::RELEASED}
    };
    const auto recorded = cases;
    for (auto scenario : recorded) {
        scenario.record_protocol = false;
        cases.push_back(scenario);
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(CoordinatorAndVectorPhases, HbgGmMixServiceError, ::testing::ValuesIn(gm_service_cases()));

class HbgGmMixBoundedValidation : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, bool>> {};

TEST_P(HbgGmMixBoundedValidation, AValidatedStaticContextNeverSuppressesFreshErrorAndEpochChecks) {
    const auto [operation, fault, record_protocol] = GetParam();
    for (LaneMask mask : kMixMasks) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!scheduler_mix_resource_mask(mask).has(static_cast<SubtaskSlot>(worker))) continue;
            SCOPED_TRACE(lane_names(mask));
            SCOPED_TRACE(worker);
            GmMixFixture fixture({{2, mask}});
            ASSERT_TRUE(fixture.bootstrap_admission());
            ASSERT_TRUE(fixture.admit(0));
            ASSERT_TRUE(fixture.refresh_all());
            auto *local = fixture.locals[worker];
            ASSERT_TRUE(scheduler_gm_mix_context_valid(fixture.graph.graph(), fixture.base(), local, fixture.run()));
            if (fault == 0) {
                ++fixture.admissions[worker].epoch;
            } else if (fault == 1) {
                ++scheduler_state_at<SchedulerGraphSummary>(fixture.base(), local->config.graph_summary_offset)
                      ->run_epoch;
            } else {
                scheduler_record_error(
                    fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                    SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
                );
            }
            bool progress = true;
            const auto apply = [&](auto record) {
                constexpr bool enabled = decltype(record)::value;
                if (operation == 0)
                    return scheduler_gm_mix_close<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 1)
                    return scheduler_gm_mix_claim<enabled, true>(
                               fixture.graph.graph(), fixture.base(), local, fixture.run()
                           ) != SchedulerBlockClaimResult::ERROR;
                if (operation == 2)
                    return scheduler_gm_mix_publish<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run()
                    );
                if (operation == 3)
                    return scheduler_gm_mix_observe<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 4)
                    return scheduler_gm_mix_arrive<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 5)
                    return scheduler_gm_mix_prepare_lane<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
                    );
                return scheduler_gm_mix_retire<enabled, true>(
                           fixture.graph.graph(), fixture.base(), local, fixture.run()
                       ) != SchedulerGmMixResult::ERROR;
            };
            EXPECT_FALSE(record_protocol ? apply(std::true_type{}) : apply(std::false_type{}));
            EXPECT_EQ(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 0u);
            EXPECT_EQ(fixture.control(0)->next_block, 0u);
            EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
            EXPECT_EQ(fixture.mailbox(0)->published_seq, 0u);
            EXPECT_EQ(local->local_ready_mask, 0u);
            EXPECT_EQ(fixture.seal(0, 0), 0u);
            EXPECT_EQ(
                fixture.run()->error_site,
                static_cast<uint64_t>(
                    fault == 2 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
                )
            );
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    StageAndHealth, HbgGmMixBoundedValidation,
    ::testing::Combine(::testing::Range(0u, 7u), ::testing::Range(0u, 3u), ::testing::Bool())
);

TEST(HbgGmMix, ServiceCarriesNewPublicationThroughArrivalWithoutAnExtraPoll) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!fixture.participant(worker)) continue;
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].arrived_seq, 1u);
            EXPECT_EQ(
                fixture.admissions[worker].gm_mix.phase,
                worker == (mask.has_aiv1() ? 2u : 1u) ? SchedulerGmMixPhase::RUNNING : SchedulerGmMixPhase::ARRIVED
            );
        }
        for (uint32_t worker = 0; worker < 2; ++worker) {
            if (!fixture.participant(worker) || fixture.admissions[worker].gm_mix.phase != SchedulerGmMixPhase::ARRIVED)
                continue;
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(
                fixture.admissions[worker].gm_mix.phase, mask.has(static_cast<SubtaskSlot>(worker)) ?
                                                             SchedulerGmMixPhase::RUNNING :
                                                             SchedulerGmMixPhase::RELEASED
            );
        }
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    }
}

TEST(HbgGmMix, ServiceRetiresOnlyAfterAllReleasesAndPublishesTheNextGroupWithoutWaiting) {
    GmMixFixture fixture({{3, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, {0, SchedulerGmMixPhase::RELEASED}));
    bool progress = true;
    ASSERT_TRUE(
        scheduler_service_gm_mix(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 1u);
    for (uint32_t worker : {1u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(
        scheduler_service_gm_mix(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 3u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
    ASSERT_TRUE(
        scheduler_service_gm_mix(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
}

using GmServiceProfilingCase = std::tuple<LaneMask, uint64_t, bool>;
class HbgGmMixServiceProfiling : public ::testing::TestWithParam<GmServiceProfilingCase> {};

TEST_P(HbgGmMixServiceProfiling, ResourceArrivalAndReleaseGuardsHoldAcrossTwoWaves) {
    const auto [mask, level, record_protocol] = GetParam();
    GmMixFixture fixture({{2, mask}}, level == 4, 1);
    if (!record_protocol) {
        ASSERT_EQ(level, 0u);
        for (auto *local : fixture.locals)
            local->profiling = nullptr;
    }
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());

    const auto service = [&](uint32_t worker, bool *progress) {
        const auto invoke = [&](auto record) {
            return scheduler_service_gm_mix<decltype(record)::value>(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), level, progress
            );
        };
        return record_protocol ? invoke(std::true_type{}) : invoke(std::false_type{});
    };
    const auto complete = [&](const GmExecution &execution) {
        const auto invoke = [&](auto record) {
            return fixture.complete<decltype(record)::value>(execution, level);
        };
        return record_protocol ? invoke(std::true_type{}) : invoke(std::false_type{});
    };
    bool progress = false;
    ASSERT_TRUE(service(0, &progress));
    ASSERT_TRUE(progress);
    std::set<std::pair<uint32_t, uint8_t>> identities;
    for (uint32_t block = 0; block < 2; ++block) {
        SCOPED_TRACE(block);
        EXPECT_EQ(fixture.mailbox(0)->published_seq, block + 1);
        EXPECT_EQ(fixture.control(0)->next_block, block + 1);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), block * mask.count());
        EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        for (uint32_t worker = 1; worker < 3; ++worker) {
            if (!fixture.participant(worker)) continue;
            ASSERT_TRUE(service(worker, &progress));
            EXPECT_TRUE(progress);
        }
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!fixture.participant(worker)) continue;
            if (fixture.admissions[worker].gm_mix.phase == SchedulerGmMixPhase::ARRIVED)
                ASSERT_TRUE(service(worker, &progress));
        }
        GmExecution last{};
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!mask.has(static_cast<SubtaskSlot>(worker))) continue;
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            EXPECT_EQ(execution.task, 0);
            EXPECT_EQ(execution.block, block);
            EXPECT_EQ(execution.blocks, 2u);
            EXPECT_EQ(execution.subtask, worker);
            EXPECT_TRUE(identities.emplace(block, execution.subtask).second);
            if (worker == (mask.has_aiv1() ? 2u : 1u)) last = execution;
            else ASSERT_TRUE(complete(execution));
        }
        ASSERT_TRUE(service(0, &progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(
            scheduler_completion_progress(fixture.control(0)->completion_state), (block + 1u) * mask.count() - 1u
        );
        EXPECT_EQ(fixture.mailbox(0)->published_seq, block + 1);
        ASSERT_TRUE(complete(last));
        ASSERT_TRUE(service(0, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), (block + 1u) * mask.count());
    }
    for (uint32_t worker = 1; worker < 3; ++worker) {
        if (!fixture.participant(worker)) {
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].assigned_seq, 0u);
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].arrived_seq, 0u);
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].released_seq, 0u);
            continue;
        }
        ASSERT_TRUE(service(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
    EXPECT_EQ(identities.size(), static_cast<size_t>(2 * mask.count()));
    EXPECT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(0)->state, static_cast<uint64_t>(SchedulerTaskState::DONE));

    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

std::vector<GmServiceProfilingCase> gm_service_profiling_cases() {
    std::vector<GmServiceProfilingCase> cases;
    for (LaneMask mask : kMixMasks)
        for (uint64_t level = 0; level <= 4; ++level) {
            cases.emplace_back(mask, level, true);
            if (level == 0) cases.emplace_back(mask, level, false);
        }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(MasksAndLevels, HbgGmMixServiceProfiling, ::testing::ValuesIn(gm_service_profiling_cases()));

using GmNoRecordAckCase = std::tuple<LaneMask, bool, uint32_t, uint32_t>;
class HbgGmMixNoRecordAck : public ::testing::TestWithParam<GmNoRecordAckCase> {};

TEST_P(HbgGmMixNoRecordAck, FutureAckCannotPermitEntryOrMailboxReuse) {
    const auto [mask, releases, actor, corrupt_lane] = GetParam();
    GmMixFixture fixture({{2, mask}}, false, 1);
    for (auto *local : fixture.locals)
        local->profiling = nullptr;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    if (releases) {
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
    } else {
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!fixture.participant(lane)) continue;
            if (lane != 0) ASSERT_TRUE(fixture.observe(lane, &progress));
            ASSERT_TRUE(fixture.arrive(lane, &progress));
        }
    }
    auto &ack = releases ? fixture.mailbox(0)->lanes[corrupt_lane].released_seq :
                           fixture.mailbox(0)->lanes[corrupt_lane].arrived_seq;
    ASSERT_EQ(ack, 1u);
    ack = 2;
    const auto before = fixture.admissions[actor].gm_mix;
    EXPECT_FALSE(fixture.fill<false>(actor));
    EXPECT_EQ(fixture.admissions[actor].gm_mix.phase, before.phase);
    EXPECT_EQ(fixture.admissions[actor].gm_mix.group_seq, before.group_seq);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), releases ? mask.count() : 0u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
    EXPECT_EQ(fixture.seal(0, 0), 0u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
    for (auto *local : fixture.locals)
        EXPECT_EQ(local->local_ready_mask, 0u);
}

std::vector<GmNoRecordAckCase> gm_no_record_ack_cases() {
    std::vector<GmNoRecordAckCase> cases;
    for (LaneMask mask : kMixMasks) {
        const LaneMask resources = scheduler_mix_resource_mask(mask);
        for (uint32_t corrupt_lane = 0; corrupt_lane < 3; ++corrupt_lane) {
            if (!resources.has(static_cast<SubtaskSlot>(corrupt_lane))) continue;
            cases.emplace_back(mask, true, 0, corrupt_lane);
            for (uint32_t actor = 0; actor < 3; ++actor)
                if (resources.has(static_cast<SubtaskSlot>(actor)))
                    cases.emplace_back(mask, false, actor, corrupt_lane);
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(RequiredResources, HbgGmMixNoRecordAck, ::testing::ValuesIn(gm_no_record_ack_cases()));

struct GmPendingEntryCase {
    GmShape shape;
    uint32_t worker;
    bool prepared;
    bool protocol;
    uint32_t failure{0};
    bool reversed_task_ids{false};
};

void prepare_pending_mix_entry(GmMixFixture &fixture, const GmPendingEntryCase &scenario) {
    if (!scenario.protocol)
        for (auto *local : fixture.locals)
            local->profiling = nullptr;
    const uint32_t current_task = scenario.reversed_task_ids ? 1 : 0;
    const uint32_t later_task = current_task ^ 1U;
    ASSERT_TRUE(fixture.admit(current_task));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    const uint32_t worker = scenario.worker;
    ASSERT_TRUE(fixture.claim_pending(later_task, worker));
    if (!scenario.prepared) return;
    const uint32_t peer = worker + 3;
    ASSERT_TRUE(fixture.claim_pending(2, peer));
    auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    ASSERT_EQ(scheduler_gm_compare_exchange(shared->commit_lock, UINT64_C(0), static_cast<uint64_t>(peer + 1)), 0u);
    ASSERT_EQ(
        scheduler_try_commit_admission(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &fixture.admissions[worker]
        ),
        SchedulerAdmissionResult::RETRY
    );
    ASSERT_EQ(shared->commit_tail, 1u);
}

class HbgGmPendingEarlierEntry : public ::testing::TestWithParam<GmPendingEntryCase> {};

TEST_P(HbgGmPendingEarlierEntry, CurrentGroupCanFinishWithoutPublishingItsLaterCandidate) {
    const auto scenario = GetParam();
    const LaneMask pending_mask = scenario.worker == 0 ? LaneMask::all() : LaneMask::aiv0();
    const uint32_t current_task = scenario.reversed_task_ids ? 1 : 0;
    const uint32_t later_task = current_task ^ 1U;
    std::vector<GmShape> shapes{{2, pending_mask}, {2, pending_mask}, {2, pending_mask}};
    shapes[current_task] = scenario.shape;
    GmMixFixture fixture(shapes, scenario.protocol);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_pending_mix_entry(fixture, scenario));
    auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    std::set<uint8_t> executed;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!scenario.shape.mask.has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.task, current_task);
        EXPECT_EQ(execution.block, 0u);
        EXPECT_EQ(execution.blocks, scenario.shape.blocks);
        EXPECT_EQ(execution.subtask, lane);
        EXPECT_TRUE(executed.insert(execution.subtask).second);
        ASSERT_TRUE(fixture.complete(execution));
    }
    const auto result = fixture.retire(0);
    EXPECT_EQ(result, SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_progress(fixture.control(current_task)->completion_state), scenario.shape.mask.count()
    );
    EXPECT_EQ(fixture.control(later_task)->next_block, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(later_task)->completion_state), 0u);
    EXPECT_EQ(fixture.admissions[scenario.worker].pending_task_ref, later_task);
    EXPECT_EQ(
        fixture.control(later_task)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(shared->commit_lock, scenario.prepared ? scenario.worker + 4u : 0u);
    EXPECT_EQ(executed.size(), static_cast<size_t>(scenario.shape.mask.count()));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

class HbgGmPendingEntryGuard : public ::testing::TestWithParam<GmPendingEntryCase> {};

TEST_P(HbgGmPendingEntryGuard, PendingCandidateDoesNotBypassErrorEpochOrGroupIdentity) {
    const auto scenario = GetParam();
    const LaneMask pending_mask = scenario.worker == 0 ? LaneMask::all() : LaneMask::aiv0();
    GmMixFixture fixture({scenario.shape, {2, pending_mask}, {2, pending_mask}}, scenario.protocol);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_pending_mix_entry(fixture, scenario));
    auto &admission = fixture.admissions[scenario.worker];
    auto *local = fixture.locals[scenario.worker];
    const auto slot = admission.gm_mix.slot_index;
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    if (scenario.failure == 0) {
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[scenario.worker + 3],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
    } else if (scenario.failure == 1) {
        ++admission.epoch;
    } else if (scenario.failure == 2) {
        admission.gm_mix.subtask_slot ^= 1U;
    } else {
        ++admission.gm_mix.block_idx;
    }
    GmExecution denied{};
    EXPECT_FALSE(fixture.begin(scenario.worker, &denied));
    EXPECT_EQ(denied.task, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    EXPECT_EQ(admission.pending_task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
}

std::vector<GmPendingEntryCase> gm_pending_entry_cases(bool failures) {
    std::vector<GmPendingEntryCase> cases;
    for (LaneMask mask : kMixMasks)
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!mask.has(static_cast<SubtaskSlot>(worker))) continue;
            for (bool prepared : {false, true})
                for (bool protocol : {false, true}) {
                    if (failures) {
                        for (uint32_t failure = 0; failure < 4; ++failure)
                            cases.push_back({{2, mask}, worker, prepared, protocol, failure});
                    } else {
                        for (uint16_t blocks : {1, 2, 65})
                            cases.push_back({{blocks, mask}, worker, prepared, protocol});
                        cases.push_back({{2, mask, true}, worker, prepared, protocol});
                        if (prepared) cases.push_back({{2, mask}, worker, prepared, protocol, 0, true});
                    }
                }
        }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    ActualResourcesAndPendingPhases, HbgGmPendingEarlierEntry, ::testing::ValuesIn(gm_pending_entry_cases(false))
);
INSTANTIATE_TEST_SUITE_P(
    ActualResourcesAndPendingPhases, HbgGmPendingEntryGuard, ::testing::ValuesIn(gm_pending_entry_cases(true))
);

TEST(HbgGmMix, PublishedRunErrorRejectsNewClaimAndDescriptorPublication) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.claim(1), SchedulerBlockClaimResult::ERROR);
        EXPECT_FALSE(fixture.publish(0));
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        EXPECT_EQ(fixture.mailbox(0)->published_seq, 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
        EXPECT_EQ(fixture.admissions[3].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

TEST(HbgGmMix, PublishedRunErrorRejectsAlreadyPreparedKernelEntryOnEveryActiveLane) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
            auto *local = fixture.locals[lane];
            const auto slot = fixture.admissions[lane].gm_mix.slot_index;
            ASSERT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
            bool permit = true;
            EXPECT_FALSE(scheduler_ordered_can_enter_slot(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit
            ));
            EXPECT_FALSE(permit);
            EXPECT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
        }
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST(HbgGmMix, MixPrefixSuffixErrorRejectsNewAdmissionAndPreviouslyPermittedKernelEntry) {
    GmMixFixture fixture(
        {{2, LaneMask::all()}, {1, LaneMask::aic()}, {1, LaneMask::all()}, {1, LaneMask::all()}, {1, LaneMask::all()}}
    );
    for (uint32_t task = 2; task <= 4; ++task) {
        fixture.graph.executable(task, 0, {1});
        fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = false;
        ASSERT_TRUE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        ASSERT_TRUE(permit);
    }

    auto *resolver = fixture.locals[3];
    for (uint32_t task : {4u, 3u, 2u})
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), resolver, fixture.run(), task),
            SchedulerRouteResult::WAITING
        );
    fixture.graph.executable(4, 0, {1, 4});
    ASSERT_EQ(fixture.graph.payload(4).fanin_count, 2);
    scheduler_gm_store(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::DONE));

    ASSERT_FALSE(scheduler_resolve_completion(
        fixture.graph.graph(), fixture.base(), resolver, fixture.run(), 1, 0, true, nullptr, UINT32_MAX
    ));

    ASSERT_EQ(scheduler_read_error(fixture.run()), static_cast<uint64_t>(SchedulerGraphResult::INVALID_FANIN_ID));
    const uint64_t first_error = scheduler_read_error(fixture.run());
    const uint64_t first_site = fixture.run()->error_site;
    const uint64_t first_task = fixture.run()->error_task_ref;
    ASSERT_EQ(first_task, 4u);

    // A published prefix is not rolled back. A consumer that already owns it
    // must still pass the real admission gate after a later suffix failure.
    SchedulerReadyClaim candidate{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        fixture.graph.graph(), fixture.base(), resolver, fixture.run(), static_cast<uint32_t>(SchedulerReadyClass::AIC),
        &candidate
    ));
    ASSERT_EQ(candidate.task_ref, 2);
    EXPECT_EQ(
        scheduler_try_claim_admission(
            fixture.graph.graph(), fixture.base(), resolver, fixture.run(), &fixture.admissions[3], candidate
        ),
        SchedulerAdmissionResult::ERROR
    );
    EXPECT_EQ(fixture.admissions[3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(2)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(shared->commit_lock, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = true;
        EXPECT_FALSE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        EXPECT_FALSE(permit);
        EXPECT_EQ(fixture.locals[lane]->slots[slot].state, SchedulerDispatchSlotState::READY);
    }
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), first_error);
    EXPECT_EQ(fixture.run()->error_site, first_site);
    EXPECT_EQ(fixture.run()->error_task_ref, first_task);
}

TEST(HbgGmMix, PublishedRunErrorDoesNotTurnReleasedLanesIntoSuccessfulBlockCompletion) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count());
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
    }
}

TEST(HbgGmMix, PublishedRunErrorDoesNotSealALocallyCompletedSyncTask) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        bool closed = true;
        EXPECT_FALSE(fixture.close(0, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.seal(0, 0), 0u);
        EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count());
    }
}

TEST_P(HbgGmMixSyncStart, LocalSealAndNextTaskDoNotWaitForRemoteClusterClaim) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{2, mask, true}, {2, mask, true}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::RETRY);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    // The retained seal remains valid after the next task replaces the mailbox.
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 2u);
    for (uint32_t worker : {1u, 2u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.refresh(worker));
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        mask.count()
    );
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    EXPECT_EQ(fixture.mailbox(1)->descriptor.task_ref, 0);
    EXPECT_EQ(fixture.mailbox(1)->descriptor.block_idx, 1u);
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgGmMixSyncStart, ZeroWorkClusterClosesOnlyAfterExhaustionWhileLastPublisherStillOwesItsGroup) {
    GmMixFixture fixture({{2, GetParam(), true}}, false, 3);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(6, &closed));
    EXPECT_FALSE(closed);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.close(6, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 2), UINT64_C(0)), 1u);
    for (uint32_t worker : {7u, 8u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    for (uint32_t worker : {0u, 1u, 2u, 3u, 4u, 5u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
    }
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixSyncStart, ::testing::ValuesIn(kMixMasks));

class HbgGmMixSingleVectorAbi : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixSingleVectorAbi, EvenAndOddBlocksKeepTheLegacySubtaskToPhysicalLaneMapping) {
    const LaneMask mask = GetParam();
    const uint8_t vector_lane = mask.has_aiv0() ? 1 : 2;
    GmMixFixture fixture({{3, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t block = 0; block < 2; ++block) {
        SCOPED_TRACE(block);
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        const auto &descriptor = fixture.mailbox(0)->descriptor;
        EXPECT_EQ(descriptor.block_idx, block);
        EXPECT_EQ(descriptor.lane_subtasks[0], 0u);
        EXPECT_EQ(descriptor.lane_subtasks[vector_lane], vector_lane);
        EXPECT_EQ(descriptor.lane_subtasks[3 - vector_lane], UINT8_MAX);
        ASSERT_TRUE(fixture.rendezvous(0));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.subtask != 0) {
                EXPECT_EQ(worker, vector_lane);
                EXPECT_EQ(execution.subtask, vector_lane);
                EXPECT_EQ(execution.physical_subblock, vector_lane - 1u);
            }
            ASSERT_TRUE(fixture.complete(execution));
        }
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    }
}

TEST_P(HbgGmMixSingleVectorAbi, VectorObserverRejectsAValidMaskRelocatedToTheOtherPhysicalLane) {
    const LaneMask mask = GetParam();
    const uint8_t vector_lane = mask.has_aiv0() ? 1 : 2;
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    auto &descriptor = fixture.mailbox(0)->descriptor;
    descriptor.lane_subtasks[vector_lane] = UINT8_MAX;
    descriptor.lane_subtasks[3 - vector_lane] = vector_lane;
    bool progress = false;
    EXPECT_FALSE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
}

INSTANTIATE_TEST_SUITE_P(SingleVectorMasks, HbgGmMixSingleVectorAbi, ::testing::Values(kAicAiv0, kAicAiv1));

class HbgGmMixShapes : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgGmMixShapes, RealOwnDispatchAndCompletionKeepEveryLogicalBlockAndSubtaskExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}});

    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.enqueue(0));
    std::map<std::pair<uint32_t, uint8_t>, uint32_t> visits;
    std::set<uint32_t> single_vector_lanes;
    const uint64_t expected = blocks * static_cast<uint32_t>(mask.count());
    uint64_t executions = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 0);
            ASSERT_EQ(execution.blocks, blocks);
            ASSERT_LT(execution.block, blocks);
            ASSERT_LT(execution.subtask, 3);
            ASSERT_TRUE(mask.has(static_cast<SubtaskSlot>(execution.subtask)));
            EXPECT_TRUE(fixture.locals[worker]->slots[execution.slot].gm_mix);
            EXPECT_EQ(fixture.mailbox(worker / 3)->descriptor.lane_subtasks[worker % 3], execution.subtask);
            if (execution.subtask != 0) {
                EXPECT_EQ(worker % 3, execution.subtask);
                EXPECT_EQ(execution.physical_subblock, worker % 3 - 1);
                if (mask.has_aic() && !mask.has_both_aiv()) single_vector_lanes.insert(worker % 3);
            }
            ++visits[{execution.block, execution.subtask}];
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_gm_load(fixture.control(0)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(executions, expected);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            EXPECT_EQ(visits[std::make_pair(block, subtask)], mask.has(static_cast<SubtaskSlot>(subtask)) ? 1u : 0u);
    if (mask.has_aic() && !mask.has_both_aiv())
        EXPECT_EQ(single_vector_lanes, (std::set<uint32_t>{mask.has_aiv0() ? 1U : 2U}));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks == 2 ? 0u : blocks);
    if (blocks > 2) {
        EXPECT_GE(fixture.control(0)->claim_tickets, blocks);
        EXPECT_LE(fixture.control(0)->claim_tickets, blocks + 2);
    } else {
        EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    }
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        expected
    );
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    for (uint32_t worker = 0; worker < 6; ++worker)
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndClusterCapacity, HbgGmMixShapes,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint16_t>(1, 2, 3, 4, 5))
);

TEST(HbgGmMix, ClaimedLastBlockWithoutDescriptorCannotLetItsVectorMembersClose) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = true;
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_FALSE(progress);
        bool closed = true;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
}

class HbgGmMixInactiveLane : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixInactiveLane, OnlyDeclaredResourceLanesArriveAndInactiveCoordinatorNeverRunsAKernel) {
    GmMixFixture fixture({{1, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    const auto &descriptor = fixture.mailbox(0)->descriptor;
    ASSERT_EQ(descriptor.resource_mask, GetParam().merged_with(LaneMask::aic()));
    uint32_t inactive = UINT32_MAX;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (descriptor.lane_subtasks[lane] == UINT8_MAX) inactive = lane;
    ASSERT_LT(inactive, 3u);
    if (!fixture.participant(inactive)) {
        EXPECT_EQ(fixture.admissions[inactive].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.rendezvous(0));
        EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].arrived_seq, 0u);
        EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].released_seq, 0u);
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count());
        return;
    }
    bool progress = false;
    for (uint32_t lane = 1; lane < 3; ++lane)
        ASSERT_TRUE(fixture.observe(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (lane != inactive) ASSERT_TRUE(fixture.arrive(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == inactive) continue;
        ASSERT_TRUE(fixture.prepare(lane, &progress));
        EXPECT_EQ(fixture.locals[lane]->local_ready_mask, 0u);
    }
    ASSERT_TRUE(fixture.arrive(inactive, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        ASSERT_TRUE(fixture.prepare(lane, &progress));
    EXPECT_EQ(fixture.admissions[inactive].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.locals[inactive]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[inactive].arrived_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[inactive].released_seq, UINT64_C(0)), 1u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        GmExecution execution{};
        if (lane == inactive) {
            EXPECT_FALSE(fixture.begin(lane, &execution));
            continue;
        }
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.subtask, descriptor.lane_subtasks[lane]);
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count()
    );
}

INSTANTIATE_TEST_SUITE_P(
    MasksWithInactiveResourceMember, HbgGmMixInactiveLane, ::testing::Values(kAicAiv0, kAicAiv1, LaneMask::both_aiv())
);

class HbgGmMixOrdinaryResource : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t, uint32_t>> {};

TEST_P(HbgGmMixOrdinaryResource, ArrivesOnlyAfterRealOrdinarySlotCompletionIsConsumed) {
    const auto [mask, lane, pending_slot] = GetParam();
    const LaneMask ordinary_mask = lane == 0 ? LaneMask::aic() : LaneMask::aiv0();
    GmMixFixture fixture({{1, ordinary_mask}, {1, ordinary_mask}, {1, mask}}, true);
    auto *local = fixture.locals[lane];
    const uint32_t core_type = lane == 0 ? 0 : 1;
    const uint32_t first_task = lane == 2 ? 1 : 0;
    SchedulerReadyBatch batch{};
    for (uint32_t task : {first_task, 1U - first_task}) {
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), task),
            SchedulerRouteResult::READY_TO_ENQUEUE
        );
        ASSERT_TRUE(scheduler_ready_batch_append(fixture.base(), local, task, &batch));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(fixture.base(), local, core_type, &batch));
    // Direct retention can populate both slots independently of queue prefetch.
    for (uint32_t task : {first_task, 1U - first_task}) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(scheduler_claim_ready_for_slot(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), core_type, &claim
        ));
        ASSERT_EQ(claim.task_ref, task);
        ASSERT_TRUE(scheduler_ordered_retain_candidate(&fixture.admissions[lane], claim));
    }
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.fill(lane));
    ASSERT_EQ(local->local_ready_mask, 3u);
    GmExecution ordinary{};
    ASSERT_TRUE(fixture.begin(lane, &ordinary));
    ASSERT_EQ(ordinary.slot, 0u);
    if (pending_slot == 1) {
        ASSERT_TRUE(fixture.complete(ordinary));
        ASSERT_TRUE(fixture.begin(lane, &ordinary));
    }
    ASSERT_EQ(ordinary.slot, pending_slot);
    ASSERT_FALSE(scheduler_slot_uses_admission(local, local->slots[pending_slot]));
    ASSERT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::EXECUTING);
    const auto *ordinary_trace = scheduler_state_at<SchedulerProtocolExecutionTrace>(
                                     fixture.base(), fixture.storage.layout.protocol.executions_offset
                                 ) +
                                 ordinary.task;
    ASSERT_NE(ordinary_trace->returned.sequence, 0u);
    ASSERT_EQ(ordinary_trace->completion_consumed.sequence, 0u);

    ASSERT_TRUE(fixture.admit(2, 3));
    ASSERT_TRUE(fixture.refresh_all());
    if (!fixture.participant(lane, 2)) {
        EXPECT_EQ(fixture.admissions[lane].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        ASSERT_TRUE(fixture.task_done(2));
        EXPECT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::EXECUTING);
        EXPECT_EQ(fixture.mailbox(0)->lanes[lane].arrived_seq, 0u);
        EXPECT_EQ(fixture.mailbox(0)->lanes[lane].released_seq, 0u);
        ASSERT_TRUE(fixture.complete(ordinary));
        ASSERT_TRUE(fixture.fill(lane));
        GmExecution remaining{};
        if (pending_slot == 0) {
            ASSERT_TRUE(fixture.begin(lane, &remaining));
            ASSERT_EQ(remaining.task, 1 - ordinary.task);
            ASSERT_TRUE(fixture.complete(remaining));
        }
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        return;
    }
    ASSERT_TRUE(scheduler_ordered_park_ordinary(fixture.graph.graph(), local, fixture.run()));
    ASSERT_EQ(local->local_ready_mask, 0u);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t member : {1U, 2U})
        if (fixture.participant(member, 2)) ASSERT_TRUE(fixture.observe(member, &progress));
    for (uint32_t member : {0U, 1U, 2U}) {
        if (member == lane || !fixture.participant(member, 2)) continue;
        ASSERT_TRUE(fixture.arrive(member, &progress));
        EXPECT_TRUE(progress);
    }

    {
        EXPECT_EQ(ordinary_trace->completion_consumed.sequence, 0u);
        progress = true;
        ASSERT_TRUE(fixture.arrive(lane, &progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(fixture.admissions[lane].gm_mix.phase, SchedulerGmMixPhase::PUBLISHED);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].arrived_seq, UINT64_C(0)), 0u);
        EXPECT_EQ(local->slots[pending_slot].task_ref, ordinary.task);
        EXPECT_EQ(local->slots[pending_slot].block_idx, ordinary.block);
        EXPECT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::EXECUTING);
        for (uint32_t member : {0U, 1U, 2U}) {
            if (!fixture.participant(member, 2)) continue;
            progress = true;
            ASSERT_TRUE(fixture.prepare(member, &progress));
            EXPECT_FALSE(progress);
            EXPECT_EQ(fixture.locals[member]->local_ready_mask, 0u);
        }
        EXPECT_EQ(
            scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(2)->completion_state, UINT64_C(0))), 0u
        );
    }

    ASSERT_TRUE(fixture.complete(ordinary));
    ASSERT_EQ(local->slots[pending_slot].state, SchedulerDispatchSlotState::FREE);
    EXPECT_NE(ordinary_trace->completion_consumed.sequence, 0u);
    ASSERT_TRUE(fixture.arrive(lane, &progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].arrived_seq, UINT64_C(0)), 1u);
    for (uint32_t member : {0U, 1U, 2U}) {
        if (!fixture.participant(member, 2)) continue;
        ASSERT_TRUE(fixture.prepare(member, &progress));
        EXPECT_TRUE(progress);
        const bool active = mask.has(static_cast<SubtaskSlot>(member));
        EXPECT_EQ(
            fixture.admissions[member].gm_mix.phase,
            active ? SchedulerGmMixPhase::RUNNING : SchedulerGmMixPhase::RELEASED
        );
        EXPECT_EQ(fixture.locals[member]->local_ready_mask != 0, active);
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(2));
    for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
        if (!fixture.participant(worker, 2)) continue;
        bool closed = false;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    ASSERT_TRUE(fixture.fill(lane));
    GmExecution resumed{};
    if (pending_slot == 0) {
        ASSERT_TRUE(fixture.begin(lane, &resumed));
        EXPECT_EQ(resumed.task, 1 - ordinary.task);
        EXPECT_FALSE(scheduler_slot_uses_admission(local, local->slots[resumed.slot]));
        ASSERT_TRUE(fixture.complete(resumed));
    }
    EXPECT_FALSE(fixture.begin(lane, &resumed));
    for (uint32_t task : {0U, 1U, 2U})
        EXPECT_EQ(scheduler_gm_load(fixture.control(task)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(2)->completion_state, UINT64_C(0))),
        mask.count()
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    EveryMaskLaneAndOwnSlot, HbgGmMixOrdinaryResource,
    ::testing::Combine(
        ::testing::ValuesIn(kMixMasks), ::testing::Values<uint32_t>(0, 1, 2), ::testing::Values<uint32_t>(0, 1)
    )
);

TEST(HbgGmMix, KernelReturnWithoutCompletionReleaseDoesNotRetireTheGroup) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    std::array<GmExecution, 3> executions{};
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.begin(worker, &executions[worker]));
    ASSERT_TRUE(fixture.complete(executions[0]));
    ASSERT_TRUE(fixture.complete(executions[1]));
    const auto &late = executions[2];
    EXPECT_EQ(fixture.locals[2]->slots[late.slot].state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 2u
    );
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.locals[2]->pending_completed, 1u);
}

TEST(HbgGmMixProtocol, ZeroWorkMembersRecordTheActualSealWithoutInventingCursorReads) {
    GmMixFixture fixture({{1, LaneMask::both_aiv()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    bool closed = false;
    ASSERT_TRUE(fixture.close(4, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(members[4].closed.sequence, 0u);
    for (uint32_t worker = 3; worker < 6; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(members[worker].observed.sequence, 1u);
        EXPECT_EQ(members[worker].closed.sequence, 2u);
        EXPECT_EQ(members[worker].close_seal, 1u);
        EXPECT_EQ(members[worker].last_group_seq, 0u);
        EXPECT_EQ(members[worker].closed_cursor, worker == 3 ? UINT64_C(1) : UINT64_MAX);
        EXPECT_EQ(members[worker].pending_mask, 0u);
        EXPECT_EQ(members[worker].worker_index, worker);
    }
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(members[0].closed.sequence, 0u);
    EXPECT_EQ(members[1].closed.sequence, 0u);
}

TEST(HbgGmMix, AZeroWorkClusterPublishesSealOneWhileAnotherClusterStillOwnsTheLastBlock) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::EXHAUSTED);
    bool closed = false;
    ASSERT_TRUE(fixture.close(3, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    for (uint32_t worker : {4u, 5u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
}

TEST(HbgGmMix, FastClusterPublishesBWhileRemoteAIsRunningAndLateVectorConsumesAsRetainedSeal) {
    GmMixFixture fixture({{2, LaneMask::all()}, {1, kAicAiv0}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1, 3));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    GmExecution remote{};
    ASSERT_TRUE(fixture.begin(5, &remote));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.admissions[0].current_task_ref, 1);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(fixture.admissions[5].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[5].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = false;
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        if (fixture.participant(worker, 1)) ASSERT_TRUE(fixture.observe(worker, &progress));
        else {
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].assigned_seq, 1u);
            ASSERT_TRUE(fixture.close(worker, &progress));
        }
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 1u);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
        ASSERT_TRUE(fixture.refresh(worker));
        if (!fixture.participant(worker, 1)) {
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 1u);
            continue;
        }
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 1);
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 2u);
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(1));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
}

TEST(HbgGmMix, ReleasedAcknowledgementsFromThePreviousGroupCannotRetireTheNextGroup) {
    GmMixFixture fixture({{3, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution cube{};
    ASSERT_TRUE(fixture.begin(0, &cube));
    ASSERT_TRUE(fixture.complete(cube));
    for (uint32_t lane : {1u, 2u})
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[lane].released_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 4u
    );
    for (uint32_t lane : {1u, 2u}) {
        GmExecution vector{};
        ASSERT_TRUE(fixture.begin(lane, &vector));
        EXPECT_EQ(vector.block, 1u);
        ASSERT_TRUE(fixture.complete(vector));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 6u
    );
}

TEST(HbgGmMix, RetiredSlotCannotReleaseOrContributeTwice) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution first{};
    ASSERT_TRUE(fixture.begin(1, &first));
    ASSERT_TRUE(fixture.complete(first));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 1u
    );
    for (uint32_t worker : {0u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    EXPECT_FALSE(scheduler_complete_own_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), first.slot, 0
    ));
    EXPECT_EQ(fixture.mailbox(0)->lanes[1].released_seq, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 3u);
}

TEST(HbgGmMix, ReleaseBeforeRealSlotRetirementFailsWithoutPublishingAnAcknowledgement) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(1, &execution));
    EXPECT_FALSE(scheduler_gm_mix_release_lane(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(),
        fixture.locals[1]->slots[execution.slot], execution.slot
    ));
    EXPECT_EQ(fixture.locals[1]->slots[execution.slot].state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
}

TEST(HbgGmMix, CoordinatorPreparesOnlyItsOwnPayloadAndVectorsMaterializeTheirOwnSlots) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t worker : {1u, 2u})
        ASSERT_TRUE(fixture.observe(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_TRUE(progress);
    for (uint32_t worker : {1u, 2u}) {
        const auto *payload = scheduler_state_at<DispatchPayload>(
            fixture.base(), fixture.locals[worker]->dispatch_payload_offset(worker, 0)
        );
        EXPECT_EQ(payload->function_bin_addr, 0u);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
        ASSERT_TRUE(fixture.prepare(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_NE(payload->function_bin_addr, 0u);
        EXPECT_EQ(payload->local_context.block_num, 1);
        EXPECT_EQ(payload->global_context.sub_block_id, worker - 1);
    }
}

class HbgGmMixHighSequence : public ::testing::TestWithParam<uint64_t> {};

TEST_P(HbgGmMixHighSequence, GroupSequenceIsIndependentOfTheNarrowOwnSlotGeneration) {
    const uint64_t previous = GetParam();
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    scheduler_gm_store(fixture.mailbox(0)->published_seq, previous);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        fixture.admissions[worker].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].released_seq, previous);
    }
    for (auto &sequence : fixture.admissions[0].gm_mix.vector_last_published_seq)
        sequence = previous;
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), previous + 1);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_LT(fixture.admissions[worker].gm_mix.slot_index, SCHEDULER_PENDING_SLOT_COUNT);
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    bool closed = false;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), previous + 2);
}

INSTANTIATE_TEST_SUITE_P(
    WideCounterBoundaries, HbgGmMixHighSequence, ::testing::Values(UINT64_C(1) << 32, UINT64_MAX - 2)
);

TEST(HbgGmMix, SequenceExhaustionFailsBeforeClaimingAnUnpublishableBlock) {
    for (uint64_t sequence : {UINT64_MAX - 1, UINT64_MAX}) {
        GmMixFixture fixture({{1, LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        fixture.admissions[0].gm_mix.group_seq = sequence;
        EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::ERROR);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.group_seq, sequence);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

enum class DescriptorFault {
    EPOCH,
    TASK,
    ORDER,
    SEQUENCE,
    BLOCK,
    BLOCK_COUNT,
    CLUSTER,
    ACTIVE_MASK,
    RESOURCE_MASK,
    FLAGS,
    WRONG_CORE_TYPE,
    DUPLICATED_SUBTASK,
    MISSING_SUBTASK,
    WRONG_VECTOR_PAIR,
    SKIPPED_SEQUENCE,
    OWNER_PREDECESSOR,
    VECTOR_PREDECESSOR,
    FUTURE_PREDECESSOR,
};

class HbgGmMixDescriptorFault : public ::testing::TestWithParam<DescriptorFault> {};

TEST_P(HbgGmMixDescriptorFault, CorruptPublishedDescriptorCannotAcquireAVectorLane) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    auto &descriptor = fixture.mailbox(0)->descriptor;
    switch (GetParam()) {
    case DescriptorFault::EPOCH:
        ++descriptor.run_epoch;
        break;
    case DescriptorFault::TASK:
        descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
        break;
    case DescriptorFault::ORDER:
        ++descriptor.order_key;
        break;
    case DescriptorFault::SEQUENCE:
        ++descriptor.group_seq;
        break;
    case DescriptorFault::BLOCK:
        descriptor.block_idx = descriptor.logical_block_num;
        break;
    case DescriptorFault::BLOCK_COUNT:
        ++descriptor.logical_block_num;
        break;
    case DescriptorFault::CLUSTER:
        ++descriptor.cluster_index;
        break;
    case DescriptorFault::ACTIVE_MASK:
        descriptor.active_mask = kAicAiv0;
        break;
    case DescriptorFault::RESOURCE_MASK:
        descriptor.resource_mask = kAicAiv0;
        break;
    case DescriptorFault::FLAGS:
        descriptor.flags ^= SCHEDULER_TASK_SPMD;
        break;
    case DescriptorFault::WRONG_CORE_TYPE:
        std::swap(descriptor.lane_subtasks[0], descriptor.lane_subtasks[1]);
        break;
    case DescriptorFault::DUPLICATED_SUBTASK:
        descriptor.lane_subtasks[2] = 1;
        break;
    case DescriptorFault::MISSING_SUBTASK:
        descriptor.lane_subtasks[2] = UINT8_MAX;
        break;
    case DescriptorFault::WRONG_VECTOR_PAIR:
        std::swap(descriptor.lane_subtasks[1], descriptor.lane_subtasks[2]);
        break;
    case DescriptorFault::SKIPPED_SEQUENCE:
        descriptor.group_seq = 3;
        descriptor.previous_lane_seq[0] = 2;
        descriptor.previous_lane_seq[1] = 1;
        scheduler_gm_store(fixture.mailbox(0)->published_seq, UINT64_C(3));
        scheduler_gm_store(fixture.mailbox(0)->lanes[1].assigned_seq, UINT64_C(3));
        break;
    case DescriptorFault::OWNER_PREDECESSOR:
        ++descriptor.previous_lane_seq[0];
        break;
    case DescriptorFault::VECTOR_PREDECESSOR:
        ++descriptor.previous_lane_seq[1];
        break;
    case DescriptorFault::FUTURE_PREDECESSOR:
        descriptor.previous_lane_seq[2] = UINT64_MAX;
        break;
    }
    bool progress = true;
    EXPECT_FALSE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[1].gm_mix.group_seq, 0u);
    EXPECT_EQ(fixture.admissions[1].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].arrived_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    IdentityAndPlacement, HbgGmMixDescriptorFault,
    ::testing::Values(
        DescriptorFault::EPOCH, DescriptorFault::TASK, DescriptorFault::ORDER, DescriptorFault::SEQUENCE,
        DescriptorFault::BLOCK, DescriptorFault::BLOCK_COUNT, DescriptorFault::CLUSTER, DescriptorFault::ACTIVE_MASK,
        DescriptorFault::RESOURCE_MASK, DescriptorFault::FLAGS, DescriptorFault::WRONG_CORE_TYPE,
        DescriptorFault::DUPLICATED_SUBTASK, DescriptorFault::MISSING_SUBTASK, DescriptorFault::WRONG_VECTOR_PAIR,
        DescriptorFault::SKIPPED_SEQUENCE, DescriptorFault::OWNER_PREDECESSOR, DescriptorFault::VECTOR_PREDECESSOR,
        DescriptorFault::FUTURE_PREDECESSOR
    )
);

struct AckPrefixCase {
    bool releases;
    uint64_t sequence;
    std::array<int8_t, 3> offsets;
};

std::vector<AckPrefixCase> ack_prefix_cases() {
    std::vector<AckPrefixCase> result;
    for (bool releases : {false, true})
        for (uint64_t sequence : {UINT64_C(1), (UINT64_C(1) << 32) + 9, UINT64_MAX - 1})
            for (int8_t first : {-1, 0, 1})
                for (int8_t second : {-1, 0, 1})
                    for (int8_t third : {-1, 0, 1})
                        result.push_back({releases, sequence, {first, second, third}});
    return result;
}

class HbgGmAckPrefix : public ::testing::TestWithParam<AckPrefixCase> {};

TEST_P(HbgGmAckPrefix, WaitsOnOldPrefixAndRejectsFuturePrefixBeforePublishingEvidence) {
    const auto scenario = GetParam();
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    fixture.admissions[0].gm_mix.group_seq = scenario.sequence;
    int8_t first_difference = 0;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const int8_t delta = scenario.offsets[lane];
        if (first_difference == 0 && delta != 0) first_difference = delta;
        const uint64_t value = delta < 0 ? scenario.sequence - 1 : scenario.sequence + delta;
        auto &ack = scenario.releases ? fixture.mailbox(0)->lanes[lane].released_seq :
                                        fixture.mailbox(0)->lanes[lane].arrived_seq;
        scheduler_gm_store(ack, value);
    }
    bool all = true;
    const bool valid = scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), scenario.releases, &all
    );
    EXPECT_EQ(valid, first_difference <= 0);
    EXPECT_EQ(all, first_difference == 0);
    EXPECT_EQ(scheduler_read_error(fixture.run()) != 0, first_difference > 0);
    if (first_difference > 0)
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
}

INSTANTIATE_TEST_SUITE_P(OldCurrentFutureAndFullWidth, HbgGmAckPrefix, ::testing::ValuesIn(ack_prefix_cases()));

TEST_P(HbgGmAckPrefix, ReusedPrefilterStillStronglyChecksEveryAcknowledgement) {
    const auto scenario = GetParam();
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    fixture.admissions[0].gm_mix.group_seq = scenario.sequence;
    int8_t first_difference = 0;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const int8_t delta = scenario.offsets[lane];
        if (first_difference == 0 && delta != 0) first_difference = delta;
        auto &ack = scenario.releases ? fixture.mailbox(0)->lanes[lane].released_seq :
                                        fixture.mailbox(0)->lanes[lane].arrived_seq;
        scheduler_gm_store(ack, delta < 0 ? scenario.sequence - 1 : scenario.sequence + delta);
    }
    bool all = true;
    const bool valid = scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), scenario.releases, &all, true
    );
    EXPECT_EQ(valid, first_difference <= 0);
    EXPECT_EQ(all, first_difference == 0);
    EXPECT_EQ(scheduler_read_error(fixture.run()) != 0, first_difference > 0);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
}

class HbgGmMixPollAckHint : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t>> {};

TEST_P(HbgGmMixPollAckHint, ArrivalHintIsClearedOnEarlyReturnAndCannotPermitChangedAcknowledgement) {
    const auto [mask, worker] = GetParam();
    const uint32_t cluster = worker / 3;
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    bool progress = false;
    for (uint32_t lane = 1; lane < 3; ++lane)
        if (fixture.participant(lane)) ASSERT_TRUE(fixture.observe(cluster * 3 + lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (fixture.participant(lane)) ASSERT_TRUE(fixture.arrive(cluster * 3 + lane, &progress));
    auto *local = fixture.locals[worker];
    auto phase = SchedulerGmMixPhase::IDLE;
    if (!fixture.participant(worker)) {
        EXPECT_FALSE(
            scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase)
        );
        EXPECT_EQ(phase, SchedulerGmMixPhase::IDLE);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(local->local_ready_mask, 0u);
        return;
    }
    ASSERT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase));
    EXPECT_EQ(phase, SchedulerGmMixPhase::ARRIVED);
    local->local_ready_mask = 1;
    ASSERT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase));
    EXPECT_EQ(phase, SchedulerGmMixPhase::IDLE);
    local->local_ready_mask = 0;
    const uint32_t checked_lane = mask.has_aiv1() ? 2 : 1;
    fixture.mailbox(cluster)->lanes[checked_lane].arrived_seq = 0;
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase));
    EXPECT_EQ(phase, SchedulerGmMixPhase::IDLE);
    fixture.mailbox(cluster)->lanes[checked_lane].arrived_seq = 1;
    ASSERT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase));
    ASSERT_EQ(phase, SchedulerGmMixPhase::ARRIVED);
    fixture.mailbox(cluster)->lanes[checked_lane].arrived_seq = 2;
    EXPECT_FALSE(
        scheduler_gm_mix_prepare_lane(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress, true)
    );
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(local->local_ready_mask, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
}

TEST_P(HbgGmMixPollAckHint, ReleaseHintOnlyBelongsToCoordinatorAndStillRejectsLateError) {
    const auto [mask, worker] = GetParam();
    const uint32_t cluster = worker / 3;
    GmMixFixture fixture({{1, mask}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    ASSERT_TRUE(fixture.rendezvous(cluster));
    ASSERT_TRUE(fixture.finish_lanes(cluster));
    auto phase = SchedulerGmMixPhase::ARRIVED;
    auto *local = fixture.locals[worker];
    EXPECT_EQ(
        scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, nullptr, &phase),
        worker % 3 == 0
    );
    EXPECT_EQ(phase, worker % 3 == 0 ? SchedulerGmMixPhase::RELEASED : SchedulerGmMixPhase::IDLE);
    scheduler_publish_error(fixture.run(), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    bool progress = false;
    EXPECT_FALSE(
        scheduler_service_gm_mix(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress, phase)
    );
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count());
    EXPECT_EQ(
        fixture.admissions[worker].current_task_ref, fixture.participant(worker) ? 0 : SCHEDULER_TASK_REF_INVALID
    );
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndWorkers, HbgGmMixPollAckHint,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values(0u, 1u, 2u, 3u, 4u, 5u))
);

class HbgGmLateAckReadyCandidate : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t>> {};

TEST_P(HbgGmLateAckReadyCandidate, RepeatedReadyDiscoveryCannotPrepareBeforeEveryResourceLaneArrives) {
    const auto [mask, delayed] = GetParam();
    uint32_t observer = (delayed + 1) % 3;
    while (!mask.merged_with(LaneMask::aic()).has(static_cast<SubtaskSlot>(observer)))
        observer = (observer + 1) % 3;
    const LaneMask ordinary_mask = observer == 0 ? LaneMask::aic() : LaneMask::aiv0();
    GmMixFixture fixture({{1, mask}, {1, ordinary_mask}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t lane : {1u, 2u})
        if (fixture.participant(lane)) ASSERT_TRUE(fixture.observe(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (lane != delayed && fixture.participant(lane)) ASSERT_TRUE(fixture.arrive(lane, &progress));
    ASSERT_TRUE(fixture.enqueue(1, observer));
    auto *block =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    if (!fixture.participant(delayed)) {
        ASSERT_TRUE(fixture.fill(observer));
        EXPECT_NE(block->lanes[observer].arrivals_observed.sequence, 0u);
        EXPECT_EQ(block->lanes[observer].arrived_seq[delayed], 0u);
        EXPECT_EQ(block->lanes[delayed].arrive_after.sequence, 0u);
        EXPECT_EQ(fixture.locals[observer]->local_ready_mask != 0, mask.has(static_cast<SubtaskSlot>(observer)));
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        return;
    }
    for (uint32_t poll = 0; poll < 8; ++poll) {
        EXPECT_TRUE(
            scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[observer])
        );
        ASSERT_TRUE(fixture.fill(observer));
        EXPECT_EQ(fixture.admissions[observer].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        EXPECT_EQ(fixture.locals[observer]->local_ready_mask, 0u);
        EXPECT_EQ(block->lanes[observer].arrivals_observed.sequence, 0u);
        EXPECT_EQ(block->lanes[observer].release_before.sequence, 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    }
    ASSERT_TRUE(fixture.arrive(delayed, &progress));
    ASSERT_TRUE(fixture.fill(observer));
    EXPECT_NE(block->lanes[observer].arrivals_observed.sequence, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(block->lanes[observer].arrived_seq[lane], fixture.participant(lane) ? 1u : 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    AllMasksAndDelayedLanes, HbgGmLateAckReadyCandidate,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint32_t>(0, 1, 2))
);

TEST(HbgGmMix, FutureArrivedAcknowledgementIsRejectedInsteadOfSatisfyingTheBarrier) {
    GmMixFixture fixture({{1, LaneMask::all()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t worker : {1u, 2u})
        ASSERT_TRUE(fixture.observe(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].arrived_seq, UINT64_C(2));
    EXPECT_FALSE(fixture.prepare(0, &progress));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->lanes[0].arrivals_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
}

TEST(HbgGmMix, FutureReleasedAcknowledgementIsRejectedBeforeResourceRetirement) {
    GmMixFixture fixture({{1, LaneMask::all()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(2));
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
    EXPECT_NE(record->lanes[2].release_after.sequence, 0u);
}

TEST(HbgGmMix, WrongRetainedSealCannotAdvanceALateVectorFrontier) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    scheduler_gm_store(fixture.seal(0, 0), UINT64_C(1));
    EXPECT_FALSE(fixture.close(1, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 0u);
}

TEST(HbgGmMix, ExitDrainNeverClaimsOrPreparesUnexecutedWork) {
    GmMixFixture fixture({{2, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
}

TEST(HbgGmMix, ExitDrainConsumesOutstandingSealsWithoutPublishingAnotherGroup) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    bool done = true;
    ASSERT_TRUE(fixture.drain(1, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    for (uint32_t worker = 0; worker < 6; ++worker) {
        done = false;
        for (uint32_t step = 0; step < 3 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->published_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(1)->published_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 0), UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
}

TEST(HbgGmMix, ExitDrainReleasesArrivedInactiveCoordinatorAfterLogicalCompletion) {
    for (bool protocol : {false, true}) {
        SCOPED_TRACE(protocol);
        GmMixFixture fixture({{1, LaneMask::both_aiv()}}, protocol, 1);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        bool progress = false;
        ASSERT_TRUE(fixture.arrive(0, &progress));
        ASSERT_TRUE(fixture.prepare(0, &progress));
        ASSERT_FALSE(progress);
        ASSERT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        ASSERT_EQ(fixture.admissions[0].gm_mix.subtask_slot, UINT8_MAX);
        for (uint32_t worker : {1u, 2u}) {
            ASSERT_TRUE(fixture.observe(worker, &progress));
            ASSERT_TRUE(fixture.arrive(worker, &progress));
        }
        for (uint32_t worker : {1u, 2u}) {
            ASSERT_TRUE(fixture.prepare(worker, &progress));
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            ASSERT_TRUE(fixture.complete(execution));
            scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
        }
        ASSERT_TRUE(fixture.task_done(0));
        ASSERT_EQ(fixture.run()->resolved_task_count, 1u);
        ASSERT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);

        bool done = true;
        ASSERT_TRUE(fixture.drain(0, &done));
        EXPECT_FALSE(done);
        EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 1u);
        for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
            done = false;
            for (uint32_t step = 0; step < 3 && !done; ++step)
                ASSERT_TRUE(fixture.drain(worker, &done));
            EXPECT_TRUE(done);
        }
        EXPECT_EQ(scheduler_gm_query(fixture.seal(0, 0)), 2u);
        EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u);
        EXPECT_EQ(scheduler_gm_query(fixture.mailbox(0)->published_seq), 1u);
        EXPECT_EQ(scheduler_gm_query(fixture.control(0)->next_block), 1u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgGmMix, ExitDrainInactiveCoordinatorWaitsForArrivalsAndRejectsFutureAcks) {
    GmMixFixture fixture({{1, LaneMask::both_aiv()}}, true, 1);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.observe(worker, &progress));
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    }
    scheduler_gm_store(fixture.mailbox(0)->lanes[1].arrived_seq, UINT64_C(2));
    EXPECT_FALSE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
}

TEST(HbgGmMix, ExitDrainWaitsForLateReleaseThenClosesTheObligation) {
    GmMixFixture fixture({{1, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution late{};
    ASSERT_TRUE(fixture.begin(2, &late));
    for (uint32_t worker : {0u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.mailbox(0)->lanes[2].released_seq, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);

    ASSERT_TRUE(fixture.complete(late));
    for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
        done = false;
        for (uint32_t step = 0; step < 3 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 3u);
    EXPECT_EQ(scheduler_gm_query(fixture.mailbox(0)->published_seq), 1u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST(HbgGmMix, MixFaninWaitsForLastSubtaskAndLastVectorReleasesItOnce) {
    GmMixFixture fixture({{2, LaneMask::both_aiv()}, {1, kAicAiv0}});
    fixture.graph.executable(1, 0, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), 1);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    EXPECT_EQ(fixture.locals[0]->pending_completed, 0u);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.locals[5]->pending_completed, 1u);
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.storage.ready_pool_offset);
    const auto queue =
        scheduler_ready_pool_queue(fixture.base(), pool, static_cast<uint32_t>(SchedulerReadyClass::AIC));
    const uint64_t head = scheduler_gm_query(queue.control->head);
    ASSERT_LT(head, queue.capacity);
    EXPECT_EQ(scheduler_gm_query(queue.entries[head].task_ref), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
    );
    EXPECT_EQ(scheduler_gm_query(fixture.control(1)->next_block), 0u);
    uint32_t successor_executions = 0;
    for (uint32_t step = 0; step < 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            EXPECT_EQ(execution.task, 1);
            EXPECT_EQ(execution.block, 0u);
            EXPECT_EQ(execution.blocks, 1u);
            ++successor_executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_gm_load(fixture.control(1)->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(successor_executions, 2u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 4u
    );
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(1)->completion_state, UINT64_C(0))), 2u
    );
    for (auto *local : fixture.locals)
        scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 2u);
}

class HbgGmMixExitInFlight : public ::testing::TestWithParam<SchedulerGmMixPhase> {};

TEST_P(HbgGmMixExitInFlight, ExitDrainRejectsAnOutstandingGroupWithoutClaimingMoreWork) {
    GmMixFixture fixture({{2, LaneMask::all()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (GetParam() != SchedulerGmMixPhase::CLAIMED) ASSERT_TRUE(fixture.publish(0));
    if (GetParam() == SchedulerGmMixPhase::ARRIVED || GetParam() == SchedulerGmMixPhase::RUNNING) {
        bool progress = false;
        for (uint32_t worker : {1u, 2u})
            ASSERT_TRUE(fixture.observe(worker, &progress));
        for (uint32_t worker = 0; worker < 3; ++worker)
            ASSERT_TRUE(fixture.arrive(worker, &progress));
        if (GetParam() == SchedulerGmMixPhase::RUNNING)
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.prepare(worker, &progress));
    }
    ASSERT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
    bool done = true;
    EXPECT_FALSE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
    EXPECT_EQ(
        scheduler_completion_progress(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    UnretiredGroupPhases, HbgGmMixExitInFlight,
    ::testing::Values(
        SchedulerGmMixPhase::CLAIMED, SchedulerGmMixPhase::PUBLISHED, SchedulerGmMixPhase::ARRIVED,
        SchedulerGmMixPhase::RUNNING
    )
);

class HbgGmMixOwnPrepare : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixOwnPrepare, HasMixDoesNotRequireSsbufForAnOwnPayloadPreparation) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{1, mask}, {1, LaneMask::all()}});
    for (uint8_t subtask = 0; subtask < 3; ++subtask) {
        if (!mask.has(static_cast<SubtaskSlot>(subtask))) continue;
        auto *local = fixture.locals[subtask];

        local->slots[0].state = SchedulerDispatchSlotState::FILLING;
        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        ASSERT_TRUE(scheduler_prepare_dispatch_slot(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), {0}, ready, fixture.storage.metadata[0], 0,
            subtask
        ));
        const auto *payload =
            scheduler_state_at<DispatchPayload>(fixture.base(), local->dispatch_payload_offset(subtask, 0));
        EXPECT_NE(payload->function_bin_addr, 0u);
        EXPECT_EQ(payload->local_context.block_num, 1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PureAndMixedPayloads, HbgGmMixOwnPrepare,
    ::testing::Values(LaneMask::aic(), LaneMask::aiv0(), kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all())
);

class HbgGmProtocolEvents : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmProtocolEvents, CapturesPublicationDescriptorActualAcksAndUniqueBlockCompletion) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{1, mask}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    auto &record = blocks[0];
    ASSERT_NE(record.owner.claim_after.sequence, 0u);
    EXPECT_EQ(record.owner.publish_before.sequence, 0u);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    bool progress = true;
    ASSERT_TRUE(fixture.observe(mask.has_aiv0() ? 1 : 2, &progress));
    EXPECT_FALSE(progress);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_GT(record.owner.publish_before.sequence, record.owner.claim_after.sequence);
    EXPECT_GT(record.owner.publish_after.sequence, record.owner.publish_before.sequence);
    EXPECT_EQ(record.owner.group_seq, 1u);
    EXPECT_EQ(record.owner.active_mask, mask);
    EXPECT_EQ(record.owner.logical_block_num, 1u);
    EXPECT_GT(record.lanes[0].descriptor_ready.sequence, record.owner.publish_after.sequence);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto &event = record.lanes[lane];
        if (!fixture.participant(lane)) {
            EXPECT_EQ(event.descriptor_ready.sequence, 0u);
            EXPECT_EQ(event.arrive_after.sequence, 0u);
            EXPECT_EQ(event.release_after.sequence, 0u);
            continue;
        }
        EXPECT_EQ(event.worker_index, lane);
        EXPECT_EQ(event.lane, lane);
        EXPECT_EQ(event.run_epoch, 47u);
        EXPECT_EQ(event.group_seq, 1u);
        EXPECT_EQ(event.active_mask, mask);
        EXPECT_EQ(event.resource_mask, mask.merged_with(LaneMask::aic()));
        EXPECT_EQ(event.logical_block_num, 1u);
        EXPECT_GT(event.arrive_before.sequence, event.descriptor_ready.sequence);
        EXPECT_GT(event.arrive_after.sequence, event.arrive_before.sequence);
        EXPECT_GT(event.arrivals_observed.sequence, event.arrive_after.sequence);
        for (uint32_t peer = 0; peer < 3; ++peer) {
            EXPECT_EQ(event.arrived_seq[peer], fixture.participant(peer) ? 1u : 0u);
            EXPECT_EQ(event.previous_lane_seq[peer], 0u);
            EXPECT_EQ(event.lane_subtasks[peer], mask.has(static_cast<SubtaskSlot>(peer)) ? peer : UINT8_MAX);
        }
        if (mask.has(static_cast<SubtaskSlot>(lane))) EXPECT_EQ(event.release_before.sequence, 0u);
        else EXPECT_GT(event.release_before.sequence, event.arrivals_observed.sequence);
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(record.owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record.owner.completion_before.sequence, 0u);
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!fixture.participant(lane)) {
            EXPECT_EQ(record.owner.released_seq[lane], 0u);
            continue;
        }
        EXPECT_GT(record.lanes[lane].release_after.sequence, record.lanes[lane].release_before.sequence);
        EXPECT_EQ(record.owner.released_seq[lane], 1u);
    }
    EXPECT_GT(record.owner.releases_observed.sequence, record.lanes[0].release_after.sequence);
    EXPECT_GT(record.owner.completion_before.sequence, record.owner.releases_observed.sequence);
    EXPECT_GT(record.owner.completion_after.sequence, record.owner.completion_before.sequence);
    EXPECT_EQ(record.owner.completion_ticket, UINT64_MAX);
    EXPECT_EQ(record.owner.worker_index, 0u);
    ASSERT_TRUE(fixture.task_done(0));
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto *executions = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    const auto &last_execution = executions[mask.count() - 1];
    EXPECT_GT(tasks[0].completion.before.sequence, last_execution.contribution_after.sequence);
    EXPECT_EQ(last_execution.contribution_ticket, mask.count() - 1u);
    EXPECT_GT(tasks[0].completion.after.sequence, tasks[0].completion.before.sequence);
    EXPECT_EQ(tasks[0].completion.order_key, 0u);
    EXPECT_EQ(tasks[0].completion.worker_index, mask.has_aiv1() ? 2u : 1u);
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(workers[lane].error_flags, 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmProtocolEvents, ::testing::ValuesIn(kMixMasks));

TEST(HbgGmMixProtocol, OldAcksCannotCreateNewGroupEvidenceAndMailboxReuseKeepsEarlierRecords) {
    GmMixFixture fixture({{3, LaneMask::all()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto first = blocks[0];
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t lane : {1u, 2u})
        ASSERT_TRUE(fixture.observe(lane, &progress));
    ASSERT_TRUE(fixture.arrive(0, &progress));
    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(blocks[3].lanes[0].arrivals_observed.sequence, 0u);
    EXPECT_EQ(blocks[3].lanes[0].release_before.sequence, 0u);
    for (uint32_t lane : {1u, 2u})
        ASSERT_TRUE(fixture.arrive(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        ASSERT_TRUE(fixture.prepare(lane, &progress));
    const uint64_t checked_sequence = blocks[3].lanes[0].arrivals_observed.sequence;
    ASSERT_NE(checked_sequence, 0u);
    bool all = false;
    ASSERT_TRUE(scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), false, &all
    ));
    EXPECT_TRUE(all);
    EXPECT_EQ(blocks[3].lanes[0].arrivals_observed.sequence, checked_sequence);
    for (uint32_t lane = 0; lane < 3; ++lane)
        for (uint32_t peer = 0; peer < 3; ++peer)
            EXPECT_EQ(blocks[3].lanes[lane].arrived_seq[peer], 2u);
    for (uint32_t lane : {0u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(blocks[3].owner.releases_observed.sequence, 0u);
    EXPECT_EQ(blocks[3].owner.completion_before.sequence, 0u);
    GmExecution last{};
    ASSERT_TRUE(fixture.begin(2, &last));
    ASSERT_TRUE(fixture.complete(last));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    for (uint32_t peer = 0; peer < 3; ++peer)
        EXPECT_EQ(blocks[3].owner.released_seq[peer], 2u);
    EXPECT_EQ(blocks[3].owner.completion_ticket, UINT64_MAX);
    EXPECT_EQ(blocks[0].owner.claim_before.sequence, first.owner.claim_before.sequence);
    EXPECT_EQ(blocks[0].owner.completion_after.sequence, first.owner.completion_after.sequence);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(blocks[0].lanes[lane].descriptor_ready.sequence, first.lanes[lane].descriptor_ready.sequence);
        EXPECT_EQ(blocks[0].lanes[lane].release_after.sequence, first.lanes[lane].release_after.sequence);
        EXPECT_EQ(blocks[0].lanes[lane].group_seq, 1u);
    }
}

TEST(HbgGmMixProtocol, SameSequenceInDifferentClustersDoesNotAliasBlockOrLaneRecords) {
    GmMixFixture fixture({{2, LaneMask::both_aiv()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_TRUE(fixture.finish_lanes(cluster));
        ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
        const auto &record = blocks[cluster * 2];
        EXPECT_EQ(record.owner.cluster_index, cluster);
        EXPECT_EQ(record.owner.worker_index, cluster * 3);
        EXPECT_EQ(record.owner.block_idx, cluster);
        EXPECT_EQ(record.owner.group_seq, 1u);
        EXPECT_EQ(record.owner.completion_ticket, UINT64_MAX);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            EXPECT_EQ(record.lanes[lane].worker_index, cluster * 3 + lane);
            EXPECT_EQ(record.lanes[lane].cluster_index, cluster);
            EXPECT_EQ(record.lanes[lane].block_idx, cluster);
            EXPECT_EQ(record.lanes[lane].group_seq, 1u);
            EXPECT_NE(record.lanes[lane].release_after.sequence, 0u);
        }
    }
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
}

TEST(HbgGmMixProtocol, FullWidthGroupSequenceIsPreservedInObservedAckValues) {
    GmMixFixture fixture({{1, LaneMask::all()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    const uint64_t previous = UINT64_MAX - 2;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        fixture.admissions[lane].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].released_seq, previous);
    }
    for (auto &sequence : fixture.admissions[0].gm_mix.vector_last_published_seq)
        sequence = previous;
    scheduler_gm_store(fixture.mailbox(0)->published_seq, previous);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.group_seq, UINT64_MAX - 1);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(record->owner.released_seq[lane], UINT64_MAX - 1);
        EXPECT_EQ(record->lanes[lane].group_seq, UINT64_MAX - 1);
        for (uint32_t peer = 0; peer < 3; ++peer)
            EXPECT_EQ(record->lanes[lane].arrived_seq[peer], UINT64_MAX - 1);
    }
}

TEST(HbgGmMixProtocol, ReplayedPublicationCannotOverwriteItsFirstInterval) {
    GmMixFixture fixture({{1, LaneMask::all()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto first = record->owner;
    scheduler_protocol_group_publish(fixture.base(), fixture.locals[0], true);
    scheduler_protocol_group_publish(fixture.base(), fixture.locals[0], false);
    EXPECT_EQ(record->owner.publish_before.sequence, first.publish_before.sequence);
    EXPECT_EQ(record->owner.publish_after.sequence, first.publish_after.sequence);
    EXPECT_EQ(record->owner.group_seq, first.group_seq);
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
}

class HbgGmMixReadyProbe : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixReadyProbe, EmptyReadySnapshotStillRefreshesAdmissionAndRetiresThenClaims) {
    GmMixFixture fixture({{3, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    ASSERT_EQ(fixture.admissions[0].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    bool empty = false;
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    EXPECT_TRUE(empty);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    empty = false;
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    EXPECT_TRUE(empty);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count());
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgGmMixReadyProbe, MissingReleaseDoesNotAdvanceAndTheNextFillSeesNewReady) {
    GmMixFixture fixture({{3, GetParam()}, {1, GetParam()}});
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, {0, SchedulerGmMixPhase::RUNNING}));
    const uint32_t late_lane = GetParam().has_aiv1() ? 2 : 1;
    GmExecution late{};
    ASSERT_TRUE(fixture.begin(late_lane, &late));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == late_lane || !GetParam().has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    bool empty = false;
    EXPECT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    EXPECT_TRUE(empty);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count() - 1u);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
    ASSERT_TRUE(fixture.enqueue(1));
    empty = true;
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    EXPECT_FALSE(empty);
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count() - 1u);
    EXPECT_EQ(fixture.mailbox(0)->lanes[late_lane].released_seq, 0u);
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count());
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.mailbox(0)->published_seq, 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMasks, HbgGmMixReadyProbe, ::testing::ValuesIn(kMixMasks));

struct GmReadyWaitCase {
    LaneMask mask;
    uint32_t worker;
    bool releases;
    bool remote_producer;
};

std::vector<GmReadyWaitCase> gm_ready_wait_cases() {
    std::vector<GmReadyWaitCase> result;
    for (LaneMask mask : kMixMasks)
        for (uint32_t worker = 0; worker < 6; ++worker) {
            if (!mask.merged_with(LaneMask::aic()).has(static_cast<SubtaskSlot>(worker % 3))) continue;
            for (bool remote : {false, true}) {
                result.push_back({mask, worker, false, remote});
                result.push_back({mask, worker, true, remote});
            }
        }
    return result;
}

void prepare_ready_wait(GmMixFixture &fixture, const GmReadyWaitCase &scenario, GmExecution *late) {
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    const uint32_t cluster = scenario.worker / 3;
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    if (scenario.releases) {
        ASSERT_TRUE(fixture.rendezvous(cluster));
        const uint32_t late_lane = scenario.worker % 3 == 0 ? (scenario.mask.has_aiv1() ? 2 : 1) :
                                   scenario.mask.has_aic()  ? 0 :
                                                              3 - scenario.worker % 3;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!scenario.mask.has(static_cast<SubtaskSlot>(lane))) continue;
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(cluster * 3 + lane, &execution));
            if (lane == late_lane) *late = execution;
            else ASSERT_TRUE(fixture.complete(execution));
        }
    } else {
        bool progress = false;
        if (scenario.worker % 3 != 0) ASSERT_TRUE(fixture.observe(scenario.worker, &progress));
        ASSERT_TRUE(fixture.arrive(scenario.worker, &progress));
    }
    ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
    ASSERT_EQ(
        fixture.admissions[scenario.worker].gm_mix.phase,
        scenario.releases ? SchedulerGmMixPhase::RELEASED : SchedulerGmMixPhase::ARRIVED
    );
}

uint32_t ready_wait_task(const GmReadyWaitCase &scenario) { return scenario.worker % 3 == 1 ? 2 : 1; }

void enqueue_ready_wait(GmMixFixture &fixture, const GmReadyWaitCase &scenario) {
    const uint32_t task = ready_wait_task(scenario);
    const uint32_t producer = scenario.remote_producer ? (scenario.worker < 3 ? 3 : 0) : scenario.worker / 3 * 3;
    ASSERT_TRUE(fixture.enqueue(task, producer));
}

class HbgGmReadyWaitProbe : public ::testing::TestWithParam<GmReadyWaitCase> {};

TEST_P(HbgGmReadyWaitProbe, GroupSignalsAndReadyAreFreshAfterAnEmptyProbe) {
    const auto scenario = GetParam();
    GmMixFixture fixture(
        {{3, scenario.mask}, {1, (scenario.worker % 3 == 0 ? scenario.mask : LaneMask::aiv0())}, {1, LaneMask::aiv0()}}
    );
    GmExecution late{};
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_ready_wait(fixture, scenario, &late));
    auto *local = fixture.locals[scenario.worker];
    bool empty = false;
    auto phase = SchedulerGmMixPhase::IDLE;
    ASSERT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty, &phase));
    ASSERT_TRUE(empty);
    if (scenario.releases) {
        ASSERT_TRUE(fixture.complete(late));
        if (scenario.worker % 3 != 0) {
            const uint32_t cluster = scenario.worker / 3;
            ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
            ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(cluster));
        }
    } else {
        const uint32_t cluster = scenario.worker / 3;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const uint32_t peer = cluster * 3 + lane;
            if (peer == scenario.worker || !fixture.participant(peer)) continue;
            bool progress = false;
            if (lane != 0) ASSERT_TRUE(fixture.observe(peer, &progress));
            ASSERT_TRUE(fixture.arrive(peer, &progress));
        }
    }
    ASSERT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty, &phase));
    EXPECT_TRUE(empty);
    EXPECT_EQ(
        phase, !scenario.releases       ? SchedulerGmMixPhase::ARRIVED :
               scenario.worker % 3 == 0 ? SchedulerGmMixPhase::RELEASED :
                                          SchedulerGmMixPhase::IDLE
    );
    EXPECT_EQ(
        scheduler_completion_progress(fixture.control(0)->completion_state),
        scenario.releases ? scenario.mask.count() : 0u
    );
    ASSERT_NO_FATAL_FAILURE(enqueue_ready_wait(fixture, scenario));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty, &phase));
    EXPECT_FALSE(empty);
    EXPECT_EQ(phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgGmReadyWaitProbe, PublishedErrorAfterAnEmptyProbeStopsFillBeforeAnyNewOwnership) {
    const auto scenario = GetParam();
    GmMixFixture fixture(
        {{2, scenario.mask}, {1, (scenario.worker % 3 == 0 ? scenario.mask : LaneMask::aiv0())}, {1, LaneMask::aiv0()}}
    );
    GmExecution late{};
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_ready_wait(fixture, scenario, &late));
    auto *local = fixture.locals[scenario.worker];
    bool empty = false;
    ASSERT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty));
    ASSERT_TRUE(empty);
    ASSERT_NO_FATAL_FAILURE(enqueue_ready_wait(fixture, scenario));
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[5],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    EXPECT_FALSE(fixture.fill(scenario.worker));
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(
        scheduler_completion_progress(fixture.control(0)->completion_state),
        scenario.releases ? scenario.mask.count() - 1u : 0u
    );
    EXPECT_EQ(fixture.mailbox(scenario.worker / 3)->published_seq, 1u);
    EXPECT_EQ(
        fixture.control(ready_wait_task(scenario))->admission_state,
        static_cast<uint64_t>(
            scheduler_task_requires_admission(fixture.storage.metadata[ready_wait_task(scenario)]) ?
                SchedulerAdmissionState::INITIALIZED :
                SchedulerAdmissionState::UNCLAIMED
        )
    );
    for (const auto &candidate : fixture.admissions[scenario.worker].candidates)
        EXPECT_TRUE(scheduler_task_ref_is_null(candidate.ready.task_ref));
    EXPECT_EQ(local->local_ready_mask, 0u);
    EXPECT_EQ(fixture.run()->error_worker_index, 5u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

INSTANTIATE_TEST_SUITE_P(
    ResourceMembersAndPublicationRaces, HbgGmReadyWaitProbe, ::testing::ValuesIn(gm_ready_wait_cases())
);

class HbgGmImmediateReadyProbe : public ::testing::TestWithParam<std::tuple<LaneMask, bool>> {};

TEST_P(HbgGmImmediateReadyProbe, SingleBlockAndSyncStartDoNotDeferNewReady) {
    const auto [mask, sync] = GetParam();
    GmMixFixture fixture({{static_cast<uint16_t>(sync ? 2 : 1), mask, sync}, {1, mask}});
    GmExecution late{};
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_NO_FATAL_FAILURE(prepare_ready_wait(fixture, {mask, 0, false, 0}, &late));
    bool empty = false;
    ASSERT_FALSE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    ASSERT_TRUE(empty);
    ASSERT_TRUE(fixture.enqueue(1));
    EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty));
    EXPECT_FALSE(empty);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndShapes, HbgGmImmediateReadyProbe, ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Bool())
);

TEST(HbgGmMixReadyProbeOwnership, CandidatePendingAndLocalReadyNeverReuseAnEmptyHint) {
    for (uint32_t owned = 0; owned < 3; ++owned) {
        SCOPED_TRACE(owned);
        GmMixFixture fixture({{2, LaneMask::all()}, {1, owned == 0 ? LaneMask::aic() : LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, {0, SchedulerGmMixPhase::RUNNING}));
        auto *local = fixture.locals[0];
        if (owned != 2) {
            GmExecution executing{};
            ASSERT_TRUE(fixture.begin(0, &executing));
            ASSERT_EQ(local->local_ready_mask, 0u);
            ASSERT_TRUE(fixture.enqueue(1));
            if (owned == 0) {
                ASSERT_TRUE(fixture.fill(0));
                bool retained = false;
                for (const auto &candidate : fixture.admissions[0].candidates)
                    retained = retained || candidate.ready.task_ref == 1;
                ASSERT_TRUE(retained);
            } else {
                SchedulerReadyClaim candidate{};
                ASSERT_TRUE(scheduler_claim_ready_for_slot(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(),
                    static_cast<uint32_t>(SchedulerReadyClass::AIC), &candidate
                ));
                ASSERT_EQ(candidate.task_ref, 1);
                ASSERT_EQ(
                    scheduler_try_claim_admission(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &fixture.admissions[0], candidate
                    ),
                    SchedulerAdmissionResult::PROGRESS
                );
                ASSERT_EQ(fixture.admissions[0].pending_task_ref, 1);
            }
        } else {
            ASSERT_NE(local->local_ready_mask, 0u);
        }
        ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
        const uint8_t ready_mask = local->local_ready_mask;
        bool empty = true;
        EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty));
        EXPECT_FALSE(empty);
        ASSERT_TRUE(fixture.fill(0));
        if (owned == 1) {
            EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        } else if (owned == 2) {
            EXPECT_EQ(local->local_ready_mask, ready_mask);
        } else {
            bool retained = false;
            for (const auto &candidate : fixture.admissions[0].candidates)
                retained = retained || candidate.ready.task_ref == 1;
            EXPECT_TRUE(retained);
        }
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgGmMixReadyProbeOwnership, ReadyFromEitherProducerKeepsDiscoveryEnabled) {
    for (bool remote : {false, true}) {
        SCOPED_TRACE(remote);
        GmMixFixture fixture({{2, LaneMask::all()}, {1, LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
        ASSERT_TRUE(fixture.enqueue(1, remote ? 3 : 0));
        bool empty = true;
        EXPECT_TRUE(
            scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], &empty)
        );
        EXPECT_FALSE(empty);

        bool failed = false;
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &failed);
        EXPECT_FALSE(failed);
        EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));

        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgGmMixReadyProbeOwnership, InvalidDiscoveryConfigurationCannotBeHiddenByAnEmptyHint) {
    for (uint32_t invalid = 0; invalid < 2; ++invalid) {
        SCOPED_TRACE(invalid);
        GmMixFixture fixture({{2, LaneMask::all()}});
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, {0, SchedulerGmMixPhase::RUNNING}));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_NO_FATAL_FAILURE(expect_ready_pool_drained(fixture));
        auto *local = fixture.locals[0];
        if (invalid == 0) {
            const auto &pool =
                *scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.storage.ready_pool_offset);
            const auto queue =
                scheduler_ready_pool_queue(fixture.base(), pool, static_cast<uint32_t>(SchedulerReadyClass::AIC));
            scheduler_gm_store(queue.control->head, queue.capacity + 1);
        } else local->config.self_core_type = SCHEDULER_READY_QUEUE_COUNT;
        bool empty = true;
        EXPECT_TRUE(scheduler_ordered_poll_may_progress(fixture.graph.graph(), fixture.base(), local, &empty));
        EXPECT_FALSE(empty);
        EXPECT_FALSE(fixture.fill(0));
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 3u);
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        EXPECT_EQ(fixture.mailbox(0)->published_seq, 1u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    }
}

using GmMultiWaveCompletionCase = std::tuple<LaneMask, uint16_t, uint32_t, uint32_t>;

std::vector<GmMultiWaveCompletionCase> gm_multiwave_completion_cases() {
    std::vector<GmMultiWaveCompletionCase> cases;
    for (LaneMask mask : kMixMasks)
        for (uint16_t blocks : {3, 17, 65})
            for (uint32_t clusters : {1, 2, 4}) {
                cases.emplace_back(mask, std::max<uint16_t>(blocks, clusters + 1), clusters, 0);
                if (clusters != 1)
                    cases.emplace_back(mask, std::max<uint16_t>(blocks, clusters + 1), clusters, clusters - 1);
            }
    return cases;
}

class HbgGmMixMultiWaveCompletion : public ::testing::TestWithParam<GmMultiWaveCompletionCase> {};

TEST_P(HbgGmMixMultiWaveCompletion, EveryReturnPrecedesExactlyOneSuccessorAcrossAllEligibleClusters) {
    const auto [mask, blocks, clusters, first] = GetParam();
    GmMixFixture fixture({{blocks, mask}, {1, LaneMask::aiv0()}}, false, clusters);
    fixture.graph.executable(1, 1, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0, first * 3));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[first * 3], fixture.run(), 1),
        SchedulerRouteResult::WAITING
    );
    std::map<std::pair<uint32_t, uint8_t>, uint32_t> visits;
    for (uint32_t block = 0; block < 2; ++block) {
        ASSERT_EQ(fixture.claim(first), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(first));
        ASSERT_TRUE(fixture.rendezvous(first));
        ASSERT_TRUE(fixture.finish_lanes(first));
        ASSERT_EQ(fixture.retire(first), SchedulerGmMixResult::PROGRESS);
        const uint64_t completed = scheduler_completion_progress(fixture.control(0)->completion_state);
        EXPECT_LE(completed, (block + 1u) * mask.count());
        ASSERT_EQ(fixture.retire(first), SchedulerGmMixResult::WAIT);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), completed);
        ASSERT_NE(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            if (mask.has(static_cast<SubtaskSlot>(subtask))) ++visits[{block, subtask}];
    }
    uint32_t successor_entries = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t index = 0; index < clusters * 3; ++index) {
            const uint32_t worker = (index + step + first * 3) % (clusters * 3);
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.task == 0) {
                ASSERT_LT(execution.block, blocks);
                ++visits[{execution.block, execution.subtask}];
            } else {
                ASSERT_EQ(execution.task, 1);
                EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), blocks * mask.count());
                EXPECT_EQ(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
                EXPECT_EQ(visits.size(), blocks * static_cast<uint32_t>(mask.count()));
                ++successor_entries;
            }
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (fixture.control(1)->state == static_cast<int64_t>(SchedulerTaskState::DONE)) break;
    }
    EXPECT_EQ(successor_entries, 1u);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            EXPECT_EQ(visits[std::make_pair(block, subtask)], mask.has(static_cast<SubtaskSlot>(subtask)) ? 1u : 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), blocks * mask.count());
    EXPECT_EQ(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    for (uint32_t worker = 0; worker < clusters * 3; ++worker) {
        bool done = false;
        for (uint32_t step = 0; step < 16 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, fixture.admissions[worker].capacity);
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    }
    EXPECT_EQ(fixture.run()->resolved_task_count, 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgGmMixMultiWaveCompletion, CompletedPrefixCannotAdvanceAfterErrorEpochCursorOrSealFailure) {
    const auto [mask, blocks, clusters, first] = GetParam();
    for (uint32_t fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{blocks, mask}}, false, clusters);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0, first * 3));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(first), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(first));
        ASSERT_TRUE(fixture.rendezvous(first));
        ASSERT_TRUE(fixture.finish_lanes(first));
        ASSERT_EQ(fixture.retire(first), SchedulerGmMixResult::PROGRESS);
        const uint64_t completed_prefix = scheduler_completion_progress(fixture.control(0)->completion_state);
        ASSERT_LE(completed_prefix, mask.count());
        if (fault == 0) scheduler_publish_error(fixture.run(), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
        else if (fault == 1) ++fixture.admissions[first * 3].epoch;
        else if (fault == 2) fixture.control(0)->next_block = blocks + 1;
        else {
            fixture.control(0)->next_block = blocks;
            fixture.seal(0, first) = 1;
        }
        EXPECT_FALSE(fixture.fill(first * 3));
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), completed_prefix);
        EXPECT_NE(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MasksSizesClustersAndClaimants, HbgGmMixMultiWaveCompletion, ::testing::ValuesIn(gm_multiwave_completion_cases())
);

class HbgGmMixDisabledProtocol : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HbgGmMixDisabledProtocol, BlockWritersDoNotInspectInvalidArgumentsOrChangeSharedState) {
    GmMixFixture fixture({{2, LaneMask::all()}}, true);
    auto *local = fixture.locals[0];
    ASSERT_TRUE(scheduler_protocol_enabled(local));
    if (GetParam() == 0) local = nullptr;
    else if (GetParam() == 1) local->profiling = nullptr;
    else local->profiling->protocol_layout_offset = 0;
    ASSERT_FALSE(scheduler_protocol_enabled(local));
    const auto *begin = static_cast<const uint8_t *>(fixture.base());
    const auto *end = begin + fixture.storage.layout.protocol.storage_end_offset;
    ASSERT_GT(end, begin);
    const std::vector<uint8_t> before(begin, end);
    scheduler_protocol_block_claim(fixture.base(), local, UINT32_MAX, UINT64_MAX, UINT64_MAX);
    scheduler_protocol_group_descriptor(fixture.base(), local, nullptr);
    for (bool first : {false, true}) {
        scheduler_protocol_block_complete(fixture.base(), local, UINT32_MAX, first, UINT64_MAX);
        scheduler_protocol_group_publish(fixture.base(), local, first);
        scheduler_protocol_group_acks(fixture.base(), local, first, nullptr);
        for (bool release : {false, true})
            scheduler_protocol_lane_publication(fixture.base(), local, release, first);
    }
    EXPECT_EQ(std::vector<uint8_t>(begin, end), before);
}

INSTANTIATE_TEST_SUITE_P(NullLocalNullProfileAndNoLayout, HbgGmMixDisabledProtocol, ::testing::Values(0u, 1u, 2u));

// Admission ownership is independent of the type of the core that resolves
// the last dependency. Publication alone must not allocate an execution slot.
TEST(HbgDirectAdmission, ReadyCooperativeSuccessorCommitsBeforeCompletionReturns) {
    {
        for (const auto mask : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::both_aiv(), LaneMask::all()}) {
            for (uint32_t owner : {0u, 1u, 2u}) {
                SCOPED_TRACE(lane_names(mask));
                SCOPED_TRACE(owner);
                GmMixFixture fixture({{2, LaneMask::aic()}, {2, mask}});
                fixture.graph.executable(1, 0, {0});
                fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
                ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
                ASSERT_TRUE(fixture.bootstrap_admission());
                ASSERT_EQ(
                    scheduler_route_task(
                        fixture.graph.graph(), fixture.base(), fixture.locals[owner], fixture.run(), 1
                    ),
                    SchedulerRouteResult::WAITING
                );
                ASSERT_TRUE(fixture.admit(0));
                fixture.control(0)->next_block = 2;
                fixture.control(0)->claim_seal = 2;
                fixture.control(0)->completion_state |= 2;
                ASSERT_TRUE(scheduler_complete_task(
                    fixture.graph.graph(), fixture.base(), fixture.locals[owner], fixture.run(), 0, false, 0, nullptr
                ));
                auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
                    fixture.base(), fixture.storage.layout.admission_control_offset
                );
                EXPECT_EQ(shared->commit_tail, 2u);
                EXPECT_EQ(
                    fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
                );
                EXPECT_EQ(fixture.control(1)->next_block, 0u);
                EXPECT_EQ(scheduler_completion_progress(fixture.control(1)->completion_state), 0u);
                for (auto *local : fixture.locals)
                    EXPECT_EQ(local->local_ready_mask, 0u);
                for (uint32_t queue = 0; queue < 2; ++queue) {
                    auto *holder = fixture.locals[queue == 0 ? 0 : 1];
                    SchedulerReadyClaim empty{};
                    ASSERT_TRUE(scheduler_claim_ready_for_slot(
                        fixture.graph.graph(), fixture.base(), holder, fixture.run(), queue, &empty
                    ));
                    EXPECT_EQ(empty.task_ref, SCHEDULER_TASK_REF_INVALID);
                }
            }
        }
    }
}

TEST(HbgDirectAdmission, ContendedCommitRetainsExactlyOnePendingOwner) {
    {
        GmMixFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::all()}});
        fixture.graph.executable(1, 0, {0});
        fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        {
            ASSERT_TRUE(fixture.bootstrap_admission());
            EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        }
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1),
            SchedulerRouteResult::WAITING
        );
        ASSERT_TRUE(fixture.admit(0));
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        shared->commit_lock = 1;
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state |= 2;
        ASSERT_TRUE(scheduler_complete_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, false, 0, nullptr
        ));
        EXPECT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, 1);
        EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        for (const auto &candidate : fixture.admissions[1].candidates)
            EXPECT_EQ(candidate.ready.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        shared->commit_lock = 0;
        EXPECT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        EXPECT_EQ(shared->commit_tail, 2u);
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgDirectAdmission, UnfinishedSecondFaninCannotBeAdmitted) {
    {
        GmMixFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aiv0()}, {2, LaneMask::all()}});
        fixture.graph.executable(2, 0, {0, 1});
        fixture.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 2),
            SchedulerRouteResult::WAITING
        );
        ASSERT_TRUE(fixture.admit(0));
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state |= 2;
        ASSERT_TRUE(scheduler_complete_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, false, 0, nullptr
        ));
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        EXPECT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        ASSERT_TRUE(fixture.admit(1, 1));
        fixture.control(1)->next_block = 2;
        fixture.control(1)->claim_seal = 2;
        fixture.control(1)->completion_state |= 2;
        ASSERT_TRUE(scheduler_complete_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1, false, 0, nullptr
        ));
        EXPECT_EQ(shared->commit_tail, 3u);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
    }
}

TEST(HbgDirectAdmission, OneDirectSuccessorLeavesExcessWorkQueuedExactlyOnce) {
    {
        GmMixFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::all()}, {2, LaneMask::all()}});
        for (uint32_t task : {1u, 2u}) {
            fixture.graph.executable(task, 0, {0});
            fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
        }
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.bootstrap_admission());
        for (uint32_t task : {1u, 2u})
            ASSERT_EQ(
                scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), task),
                SchedulerRouteResult::WAITING
            );
        ASSERT_TRUE(fixture.admit(0));
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state |= 2;
        ASSERT_TRUE(scheduler_complete_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, false, 0, nullptr
        ));
        unsigned admitted = 0;
        int64_t queued = -1;
        for (uint32_t task : {1u, 2u}) {
            if (fixture.control(task)->admission_state == static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED))
                ++admitted;
            else queued = task;
        }
        ASSERT_EQ(admitted, 1u);
        ASSERT_NE(queued, -1);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(scheduler_claim_ready_for_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &ready
        ));
        EXPECT_EQ(ready.task_ref, queued);
        ready = {};
        ASSERT_TRUE(scheduler_claim_ready_for_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &ready
        ));
        EXPECT_EQ(ready.task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}
TEST(HbgCompletionPacking, MaximumSupportedShapesPreserveTotalAndAsyncVotesUntilTheUniqueLastReturn) {
    for (const auto mask : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::all()}) {
        const auto blocks = static_cast<uint16_t>(INT16_MAX / mask.count());
        const uint32_t total = blocks * mask.count();
        std::vector<uint8_t> subtasks;
        for (uint8_t lane = 0; lane < 3; ++lane)
            if (mask.has(static_cast<SubtaskSlot>(lane))) subtasks.push_back(lane);
        for (const bool asynchronous : {false, true}) {
            GmMixFixture fixture({{blocks, mask}});
            auto *local = fixture.locals[0];
            SchedulerAsyncLocalState async{};
            local->async = &async;
            SchedulerLocalSlotState slot{};
            slot.task_ref = 0;
            for (uint32_t index = 0; index < total; ++index) {
                slot.block_idx = index / subtasks.size();
                slot.subtask_slot = subtasks[index % subtasks.size()];
                async.work_activated = asynchronous && index % 3 == 0;
                const auto result = scheduler_contribute_subtask<false>(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0
                );
                const auto expected = index + 1 != total ? SchedulerSubtaskFinishResult::PARTIAL :
                                      asynchronous       ? SchedulerSubtaskFinishResult::TASK_LAST_CHECK_ASYNC :
                                                           SchedulerSubtaskFinishResult::TASK_LAST;
                ASSERT_EQ(result, expected);
            }
            const uint64_t state = fixture.control(0)->completion_state;
            EXPECT_EQ(scheduler_completion_count(state), total);
            EXPECT_EQ(scheduler_completion_total(state), total);
            EXPECT_EQ(state >> 32, asynchronous ? (total + 2) / 3 : 0u);
            EXPECT_EQ(
                (scheduler_contribute_subtask<false>(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0
                )),
                SchedulerSubtaskFinishResult::ERROR
            );
        }
    }
}

class HbgGmMixSingleWave : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixSingleWave, LocalCloseAndNextTaskProgressBeforeTheRemoteClusterClaims) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{2, mask}, {2, mask}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.admit(1));
    for (uint32_t worker = 0; worker < 3; ++worker) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.refresh(worker));
    }
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_TRUE(fixture.admissions[0].single_wave_claim_confirmed != 0);
    EXPECT_FALSE(fixture.admissions[0].sync_claimed);
    EXPECT_FALSE(fixture.admissions[0].single_wave_retired);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));

    const uint32_t late_lane = mask.has_aiv1() ? 2 : 1;
    GmExecution late{};
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        if (lane == late_lane) late = execution;
        else ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    bool closed = true;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.seal(0, 0), 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count() - 1u);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::RETRY);

    ASSERT_TRUE(fixture.complete(late));
    EXPECT_FALSE(fixture.admissions[0].single_wave_retired);
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.seal(0, 0), 0u);
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_TRUE(fixture.admissions[0].single_wave_retired);
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::RETRY);
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_FALSE(fixture.admissions[0].single_wave_claim_confirmed != 0);
    EXPECT_EQ(fixture.seal(0, 0), 2u);
    EXPECT_FALSE(fixture.admissions[0].single_wave_retired);
    const auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_EQ(members[0].closed_cursor, UINT64_MAX);
    EXPECT_EQ(members[0].local_claim_confirmed, 1u);
    EXPECT_EQ(members[0].claim_seal_seen, 0u);
    EXPECT_EQ(members[0].claim_cursor_seen, 0u);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count());
    EXPECT_EQ(fixture.admissions[3].current_task_ref, SCHEDULER_TASK_REF_INVALID);

    // The fast cluster can reuse its mailbox for P while the remote cluster
    // has not even adopted A. A's vector members still close via its retained seal.
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.admissions[0].current_task_ref, 1);
    EXPECT_FALSE(fixture.admissions[0].single_wave_claim_confirmed != 0);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_TRUE(fixture.admissions[0].single_wave_claim_confirmed != 0);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 2u);
    for (uint32_t worker : {1u, 2u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 1);
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_FALSE(fixture.admissions[worker].single_wave_claim_confirmed != 0);
    }
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(1)->completion_state), mask.count());
    EXPECT_EQ(fixture.control(1)->claim_seal, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), mask.count());

    // The slow cluster owes one block from each order; neither the fast
    // cluster's local close nor its next task may erase those obligations.
    for (uint32_t task = 0; task < 2; ++task) {
        for (uint32_t worker = 3; worker < 6; ++worker) {
            if (!fixture.participant(worker)) continue;
            ASSERT_TRUE(fixture.refresh(worker));
            ASSERT_EQ(fixture.admissions[worker].current_task_ref, task);
            EXPECT_FALSE(fixture.admissions[worker].single_wave_claim_confirmed != 0);
        }
        ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
        EXPECT_EQ(fixture.control(task)->next_block, 0u);
        EXPECT_EQ(fixture.control(task)->claim_tickets, 0u);
        EXPECT_EQ(fixture.control(task)->claim_seal, 0u);
        ASSERT_TRUE(fixture.publish(1));
        EXPECT_EQ(fixture.mailbox(1)->descriptor.task_ref, task);
        EXPECT_EQ(fixture.mailbox(1)->descriptor.block_idx, 1u);
        EXPECT_EQ(fixture.mailbox(1)->descriptor.group_seq, task + 1u);
        ASSERT_TRUE(fixture.rendezvous(1));
        ASSERT_TRUE(fixture.finish_lanes(1));
        ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
        for (uint32_t worker = 3; worker < 6; ++worker) {
            if (!fixture.participant(worker)) continue;
            ASSERT_TRUE(fixture.close(worker, &closed));
            EXPECT_TRUE(closed);
            EXPECT_FALSE(fixture.admissions[worker].single_wave_claim_confirmed != 0);
        }
        EXPECT_EQ(scheduler_completion_progress(fixture.control(task)->completion_state), 2u * mask.count());
        EXPECT_EQ(fixture.control(task)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    }
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    for (auto *local : fixture.locals)
        scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(fixture.run()->resolved_task_count, 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgGmMixSingleWave, InvalidReceiptCannotRetireOrPublishAClusterCloseSeal) {
    for (uint32_t fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{2, GetParam()}}, true);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        EXPECT_FALSE(fixture.admissions[0].single_wave_retired);
        if (fault == 0) fixture.admissions[0].single_wave_claim_confirmed = 0;
        if (fault == 1) fixture.admissions[0].single_wave_claim_confirmed = 2;
        if (fault == 2) fixture.admissions[0].single_wave_claim_confirmed = 3;
        if (fault == 3) fixture.admissions[0].single_wave_retired = true;
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), GetParam().count());
        EXPECT_EQ(fixture.seal(0, 0), 0u);
        EXPECT_NE(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
        bool closed = true;
        EXPECT_FALSE(fixture.close(0, &closed));
        EXPECT_FALSE(closed);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST_P(HbgGmMixSingleWave, LastPhysicalVectorContributionChecksFixedAssignment) {
    for (uint32_t fault = 0; fault < 4; ++fault) {
        SCOPED_TRACE(fault);
        const auto mask = GetParam();
        GmMixFixture fixture({{2, mask}}, true);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t cluster = 0; cluster < 2; ++cluster) {
            ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(cluster));
            ASSERT_TRUE(fixture.rendezvous(cluster));
        }
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        const uint32_t last = mask.has_aiv1() ? 5 : 4;
        GmExecution held{};
        for (uint32_t worker = 3; worker < 6; ++worker) {
            if (!mask.has(static_cast<SubtaskSlot>(worker % 3))) continue;
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            if (worker == last) held = execution;
            else ASSERT_TRUE(fixture.complete(execution));
        }
        ASSERT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u * mask.count() - 1u);
        auto *local = fixture.locals[held.worker];
        if (fault == 0) local->slots[held.slot].block_idx = 0;
        if (fault == 1) local->slots[held.slot].block_idx = 2;
        if (fault == 2) local->admission->gm_mix.cluster_index = 0;
        if (fault == 3) local->admission->gm_mix.cluster_index = 2;
        EXPECT_FALSE(fixture.complete(held));
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u * mask.count() - 1u);
        EXPECT_FALSE(fixture.task_done(0));
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
        EXPECT_EQ(fixture.seal(0, 1), 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixSingleWave, ::testing::ValuesIn(kMixMasks));

TEST(HbgGmMixSingleWaveCoordinator, VectorOnlyCoordinatorCanRunAnOrdinaryPublisherBeforeRemoteClaim) {
    GmMixFixture fixture({{2, LaneMask::both_aiv()}, {1, LaneMask::aic()}}, true);
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.refresh(worker));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution coordinator{};
    EXPECT_FALSE(fixture.begin(0, &coordinator));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_FALSE(fixture.admissions[0].single_wave_claim_confirmed != 0);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);

    ASSERT_TRUE(fixture.enqueue(1));
    ASSERT_TRUE(fixture.fill(0));
    GmExecution publisher{};
    ASSERT_TRUE(fixture.begin(0, &publisher));
    EXPECT_EQ(publisher.task, 1);
    EXPECT_EQ(publisher.subtask, 0);
    EXPECT_FALSE(fixture.locals[0]->slots[publisher.slot].gm_mix);
    ASSERT_TRUE(fixture.complete(publisher));
    EXPECT_EQ(fixture.control(1)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u);
    EXPECT_EQ(fixture.admissions[3].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    for (uint32_t worker = 3; worker < 6; ++worker)
        ASSERT_TRUE(fixture.refresh(worker));
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    for (auto *local : fixture.locals)
        scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 4u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

}  // namespace

TEST(HbgGmMixPreparation, ArrivalPreparesLayoutWithoutReservingOrPublishingASlot) {
    GmMixFixture fixture({{2, LaneMask::all()}}, true);
    // This dependency-free fixture has unused aligned fanin storage inside
    // its wire image; bind one scalar there to keep relative offsets valid.
    auto &source = fixture.graph.payload(0);
    auto *fanins = source.fanin_data();
    auto *scalar = reinterpret_cast<uint64_t *>(fanins);
    source.bind_regions(nullptr, scalar, fanins);
    source.scalar_count = 1;
    *scalar = 111;
    ASSERT_TRUE(fixture.bootstrap_admission());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    ASSERT_TRUE(progress);
    const auto &group = fixture.admissions[0].gm_mix;
    EXPECT_EQ(group.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(group.prepared_arguments_seq, group.group_seq);
    const auto *metadata = scheduler_task_metadata_at(fixture.base(), fixture.locals[0], 0);
    EXPECT_EQ(group.prepared_metadata.logical_block_num, metadata->logical_block_num);
    EXPECT_EQ(group.prepared_metadata.kernel_ids[0], metadata->kernel_ids[0]);
    EXPECT_EQ(group.prepared_metadata.trace_index_base, metadata->trace_index_base);
    EXPECT_EQ(group.slot_index, UINT8_MAX);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    for (const auto &slot : fixture.locals[0]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(fixture.mailbox(0)->lanes[1].arrived_seq, 0u);
    EXPECT_EQ(fixture.mailbox(0)->lanes[2].arrived_seq, 0u);
    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(group.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    for (const auto &slot : fixture.locals[0]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(group.prepared_arguments.scalar_count, 1);
    EXPECT_EQ(group.prepared_arguments.scalars, scalar);
    *scalar = 222;
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(group.phase, SchedulerGmMixPhase::RUNNING);
    auto *payload = scheduler_state_at<DispatchPayload>(
        fixture.base(), fixture.locals[0]->slot_payload_offset(SchedulerFreeSlotClaim{group.slot_index})
    );
    EXPECT_EQ(payload->args[0], 222u);
    ASSERT_TRUE(fixture.finish_lanes(0));
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
}

TEST(HbgGmMixPreparation, AdvanceLayoutUsesFreshValuesAndRejectsStaleIdentity) {
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(mode);
        GmMixFixture fixture({{2, LaneMask::all()}}, true);
        // This dependency-free fixture has unused aligned fanin storage inside
        // its wire image; bind one scalar there to keep relative offsets valid.
        auto &source = fixture.graph.payload(0);
        auto *fanins = source.fanin_data();
        auto *scalar = reinterpret_cast<uint64_t *>(fanins);
        source.bind_regions(nullptr, scalar, fanins);
        source.scalar_count = 1;
        *scalar = 111;
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        uint64_t hinted_scalar = 333;
        auto &hint = fixture.admissions[0];
        hint.advance_task = mode == 2 ? 1 : 0;
        hint.advance_epoch = hint.epoch + (mode == 1 ? 1 : 0);
        hint.advance_arguments = {};
        hint.advance_arguments.scalars = &hinted_scalar;
        hint.advance_arguments.scalar_count = 1;
        bool progress = false;
        ASSERT_TRUE(fixture.arrive(0, &progress));
        ASSERT_TRUE(progress);
        const auto &group = fixture.admissions[0].gm_mix;
        EXPECT_EQ(group.phase, SchedulerGmMixPhase::ARRIVED);
        EXPECT_EQ(group.prepared_arguments_seq, group.group_seq);
        const auto *metadata = scheduler_task_metadata_at(fixture.base(), fixture.locals[0], 0);
        EXPECT_EQ(group.prepared_metadata.logical_block_num, metadata->logical_block_num);
        EXPECT_EQ(group.prepared_metadata.kernel_ids[0], metadata->kernel_ids[0]);
        EXPECT_EQ(group.prepared_metadata.trace_index_base, metadata->trace_index_base);
        EXPECT_EQ(group.slot_index, UINT8_MAX);
        EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
        for (const auto &slot : fixture.locals[0]->slots)
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(fixture.mailbox(0)->lanes[1].arrived_seq, 0u);
        EXPECT_EQ(fixture.mailbox(0)->lanes[2].arrived_seq, 0u);
        ASSERT_TRUE(fixture.prepare(0, &progress));
        EXPECT_FALSE(progress);
        EXPECT_EQ(group.phase, SchedulerGmMixPhase::ARRIVED);
        EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
        for (const auto &slot : fixture.locals[0]->slots)
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(group.prepared_arguments.scalar_count, 1);
        EXPECT_EQ(group.prepared_arguments.scalars, mode == 0 ? &hinted_scalar : scalar);
        *scalar = 222;
        hinted_scalar = 444;
        ASSERT_TRUE(fixture.rendezvous(0));
        EXPECT_EQ(group.phase, SchedulerGmMixPhase::RUNNING);
        auto *payload = scheduler_state_at<DispatchPayload>(
            fixture.base(), fixture.locals[0]->slot_payload_offset(SchedulerFreeSlotClaim{group.slot_index})
        );
        EXPECT_EQ(payload->args[0], mode == 0 ? 444u : 222u);
        ASSERT_TRUE(fixture.finish_lanes(0));
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    }
}

TEST(HbgGmMixPartialWave, UnselectedClusterAdvancesWithoutCompletingSelectedBlocks) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask}, {3, mask}}, true, 3);
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.admit(1));
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!fixture.participant(lane)) continue;
            EXPECT_EQ(fixture.admissions[lane].current_task_ref, 0);
            EXPECT_EQ(fixture.admissions[3 + lane].current_task_ref, 0);
            EXPECT_EQ(fixture.admissions[6 + lane].current_task_ref, 1);
            for (const auto &slot : fixture.locals[6 + lane]->slots)
                EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        }
        EXPECT_EQ(fixture.seal(0, 2), 0u);
        EXPECT_FALSE(fixture.task_done(0));
        for (uint32_t cluster = 0; cluster < 2; ++cluster) {
            ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
            EXPECT_EQ(fixture.admissions[cluster * 3].gm_mix.block_idx, cluster);
            ASSERT_TRUE(fixture.publish(cluster));
            ASSERT_TRUE(fixture.rendezvous(cluster));
            if (cluster == 1) EXPECT_FALSE(fixture.task_done(0));
            ASSERT_TRUE(fixture.finish_lanes(cluster));
            ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!fixture.participant(cluster * 3 + lane)) continue;
                bool closed = false;
                ASSERT_TRUE(fixture.close(cluster * 3 + lane, &closed));
                EXPECT_TRUE(closed);
            }
        }
        EXPECT_TRUE(fixture.task_done(0));
        EXPECT_EQ(fixture.control(0)->next_block, 0u);
        EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
        EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
        EXPECT_EQ(scheduler_completion_progress(fixture.control(0)->completion_state), 2u * mask.count());
        EXPECT_EQ(fixture.seal(0, 2), 0u);
        EXPECT_EQ(fixture.admissions[6].current_task_ref, 1);
    }
}

namespace {

using namespace pto2::resident;

class HbgGmMixPartialWaveDeferred : public ::testing::TestWithParam<bool> {};

TEST_P(HbgGmMixPartialWaveDeferred, DeferredCompletionHoldsConsumerAfterPhysicalReturn) {
    constexpr uint64_t epoch = 47;
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        AsyncTaskPlanInput input{};
        input.task_ref = 0;
        input.block_count = 2;
        input.active_mask = mask;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (mask.has(static_cast<SubtaskSlot>(lane))) input.capacities[lane] = 1;
        }
        AsyncCompletionHostPlan plan;
        GmMixFixture fixture({{2, mask}, {3, mask}, {1, LaneMask::aic()}}, true, 3, [&](AicoreSchedulerLayout &layout) {
            std::string error;
            EXPECT_TRUE(plan_async_completion(layout.total_size, epoch, 3, {input}, &plan, &error)) << error;
            layout.total_size = plan.total_bytes;
            layout.async_layout_offset = plan.layout_offset;
        });
        ASSERT_TRUE(initialize_async_completion(fixture.base(), fixture.storage.layout.total_size, plan));
        fixture.graph.executable(2, 0, {0});
        fixture.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
        std::array<SchedulerAsyncLocalState, 9> async{};
        for (uint32_t worker = 0; worker < async.size(); ++worker) {
            ASSERT_TRUE(scheduler_initialize_async(
                fixture.base(), plan.total_bytes, plan.layout_offset, epoch, 3, async[worker]
            ));
            fixture.locals[worker]->async = &async[worker];
        }
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 2),
            SchedulerRouteResult::WAITING
        );
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.admit(1));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.admissions[6].current_task_ref, 1);

        alignas(128) uint32_t signal = 0;
        const uint32_t executions = 2 * mask.count();
        const uint32_t deferred_execution = GetParam() ? executions - 1 : 0;
        uint32_t completed = 0;
        for (uint32_t cluster = 0; cluster < 2; ++cluster) {
            ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(cluster));
            ASSERT_TRUE(fixture.rendezvous(cluster));
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
                const uint32_t worker = cluster * 3 + lane;
                GmExecution execution{};
                ASSERT_TRUE(fixture.begin(worker, &execution));
                GlobalContext context{};
                AsyncWriter writer{};
                ASSERT_TRUE(scheduler_async_begin_execution(
                    &async[worker], execution.task, execution.block, execution.subtask, context, writer
                ));
                if (completed == deferred_execution) {
                    const CompletionToken token{
                        reinterpret_cast<uint64_t>(&signal), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
                    };
                    ASSERT_TRUE(register_completion_condition(writer, token));
                }
                ASSERT_TRUE(
                    scheduler_async_seal_execution<true>(
                        &async[worker], execution.task, execution.block, execution.subtask, writer
                    )
                );
                ASSERT_TRUE(fixture.complete(execution));
                ++completed;
                EXPECT_EQ(fixture.locals[worker]->slots[execution.slot].state, SchedulerDispatchSlotState::FREE);
            }
            ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
        }
        ASSERT_EQ(completed, executions);
        ASSERT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 2u * mask.count());
        auto *join = scheduler_state_at<SchedulerAsyncTaskJoin>(fixture.base(), plan.layout.joins_offset);
        ASSERT_TRUE(scheduler_async_pending::closed(join->word));
        ASSERT_EQ(scheduler_async_pending::pending(join->word), 1u);
        ASSERT_FALSE(fixture.task_done(0));
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), fixture.locals[0], 0));

        ASSERT_EQ(fixture.claim(2), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(2));
        ASSERT_TRUE(fixture.rendezvous(2));
        ASSERT_TRUE(fixture.finish_lanes(2));
        ASSERT_EQ(fixture.retire(2), SchedulerGmMixResult::PROGRESS);
        EXPECT_FALSE(fixture.task_done(0));

        bool progress = false;
        ASSERT_TRUE(scheduler_async_check_step(async[6], 1, 16, SchedulerAsyncCompletionObserver{}, progress));
        ASSERT_TRUE(scheduler_async_finish_step(
            fixture.graph.graph(), fixture.base(), fixture.locals[6], fixture.run(), async[6], progress
        ));
        ASSERT_FALSE(fixture.task_done(0));
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), fixture.locals[6], 0));
        __atomic_store_n(&signal, 1u, __ATOMIC_RELEASE);
        for (uint32_t step = 0; step < 32 && !fixture.task_done(0); ++step) {
            bool progress = false;
            ASSERT_TRUE(scheduler_async_check_step(async[6], 1, 16, SchedulerAsyncCompletionObserver{}, progress));
            ASSERT_TRUE(scheduler_async_finish_step(
                fixture.graph.graph(), fixture.base(), fixture.locals[6], fixture.run(), async[6], progress
            ));
        }
        ASSERT_TRUE(fixture.task_done(0));
        ASSERT_TRUE(scheduler_ready_pool_may_have_work(fixture.base(), fixture.locals[6], 0));
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(scheduler_claim_ready_for_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[6], fixture.run(), 0, &ready
        ));
        EXPECT_EQ(ready.task_ref, 2);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), fixture.locals[6], 0));
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(EarlyAndLastReturn, HbgGmMixPartialWaveDeferred, ::testing::Bool());

}  // namespace
