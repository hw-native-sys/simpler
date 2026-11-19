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

#include "hbg_gm_mix_test_support.h"

namespace {
using namespace scheduler_test::gm_mix;

TEST(HbgDirectDispatch, SelfClaimSelectsItsOriginalExecutionRecord) {
    GmMixFixture fixture({{2, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    uint32_t selected = UINT32_MAX;
    bool failed = false;
    ASSERT_TRUE(
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &selected)
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(selected, SCHEDULER_EXECUTION_TOKEN_COUNT);
    const auto &owner = local->execution_tokens[selected];
    EXPECT_EQ(&local->execution_at(selected), static_cast<const SchedulerLocalSlotState *>(&owner));
    EXPECT_EQ(owner.state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(scheduler_prepared_execute_owner(owner.prepared.claimed_state), 1u);
    EXPECT_EQ(scheduler_execution_token_count(local), 1u);
    EXPECT_EQ(hbg_test_eligible_mask(local), 1u << selected);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
}

TEST(HbgDirectDispatch, ExistingEligibleOwnerExecutesBeforeAnotherGlobalClaim) {
    GmMixFixture fixture({{1, LaneMask::aiv0()}, {1, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    EXPECT_FALSE(scheduler_execution_token_has_task(local, SCHEDULER_TASK_REF_INVALID));
    ASSERT_TRUE(fixture.fill(1));
    EXPECT_TRUE(scheduler_execution_token_has_task(local, 0));
    uint32_t owner = UINT32_MAX;
    ASSERT_TRUE(hbg_test_find_eligible(local, &owner));
    uint32_t selected = UINT32_MAX;
    bool failed = false;
    scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &selected);
    ASSERT_FALSE(failed);
    EXPECT_EQ(selected, owner);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(scheduler_execution_token_count(local), 1u);
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(1, &execution));
    ASSERT_EQ(execution.slot, owner);
    ASSERT_TRUE(fixture.complete(execution));
    EXPECT_EQ(local->execution_at(owner).state, SchedulerDispatchSlotState::FREE);
    EXPECT_FALSE(scheduler_execution_token_has_task(local, 0));
    EXPECT_FALSE(scheduler_execution_token_has_task(local, SCHEDULER_TASK_REF_INVALID));
}

void change_binding_shape(__gm__ int64_t *args) {
    auto *context = reinterpret_cast<LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX]);
    context->block_idx = 77;
    context->block_num = 99;
}

TEST(HbgDirectDispatch, BeginReportsTheEntryIdentityWhenTheKernelChangesItsBinding) {
    GmMixFixture fixture({{1, LaneMask::aiv0()}});
    fixture.storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&change_binding_shape);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.fill(1));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(1, &execution));
    EXPECT_EQ(execution.block, 0u);
    EXPECT_EQ(execution.blocks, 1u);
    auto *local = fixture.locals[1];
    auto *binding = scheduler_execution_binding_at(
        fixture.base(), local->config.prepared, local->execution_at(execution.slot).prepared.execution_index
    );
    EXPECT_EQ(binding->local_context.block_idx, 77);
    EXPECT_EQ(binding->local_context.block_num, 99);
}

TEST(HbgClusterGroupOwnership, PendingReservationCreditsItsOwnTaskWhenAnotherTaskRequestsPublication) {
    for (const uint32_t request : {1u, 2u}) {
        SCOPED_TRACE(request);
        const GmShape mix{1, LaneMask::all()};
        GmMixFixture fixture({mix, mix, mix}, false, 1);
        ASSERT_TRUE(fixture.prepare_calls());
        auto *proxy = fixture.locals[1];
        bool published = false;
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &published));
        ASSERT_TRUE(published);
        ASSERT_TRUE(fixture.ordinary_arm(1, 0, &published));
        ASSERT_FALSE(published);
        ASSERT_EQ(proxy->cluster_pending.task_ref, 1);
        for (uint32_t pass = 0; pass < 3; ++pass)
            for (uint32_t lane = 0; lane < 3; ++lane)
                ASSERT_TRUE(fixture.service_cooperative(lane));
        for (const uint32_t lane : {0u, 2u, 1u}) {
            GmExecution first{};
            ASSERT_TRUE(fixture.begin(lane, &first));
            ASSERT_EQ(first.task, 0);
            ASSERT_TRUE(fixture.complete(first));
        }
        ASSERT_TRUE(fixture.task_done(0));
        int64_t head = SCHEDULER_TASK_REF_INVALID;
        bool progress = false;
        for (uint32_t probe = 0; probe < 2 && head != 2; ++probe)
            ASSERT_TRUE(
                scheduler_issue_head(fixture.graph.graph(), fixture.base(), proxy, fixture.run(), &head, &progress)
            );
        ASSERT_EQ(head, 2);
        ASSERT_TRUE(fixture.ordinary_arm(request, 0, &published));
        EXPECT_EQ(published, request == 1);
        EXPECT_EQ(fixture.signal(0)->task_ref, 1);
        EXPECT_TRUE(scheduler_cluster_mix_armed(fixture.signal(0)->armed_generation));
        EXPECT_EQ(proxy->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control(2)->next_block, 0u);
        const uint32_t generation = fixture.signal(0)->armed_generation;
        ASSERT_TRUE(fixture.ordinary_arm(1, 0, &published));
        EXPECT_FALSE(published);
        EXPECT_EQ(fixture.signal(0)->armed_generation, generation);
        for (uint32_t pass = 0; pass < 3; ++pass)
            for (uint32_t lane = 0; lane < 3; ++lane)
                ASSERT_TRUE(fixture.service_cooperative(lane));
        for (const uint32_t lane : {0u, 2u, 1u}) {
            GmExecution reserved{};
            ASSERT_TRUE(fixture.begin(lane, &reserved));
            ASSERT_EQ(reserved.task, 1);
            ASSERT_TRUE(fixture.complete(reserved));
        }
        EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 3u);
        EXPECT_TRUE(fixture.task_done(1));
        EXPECT_EQ(scheduler_completion_count(fixture.control(2)->completion_state), 0u);
    }
}

class HbgClusterMixMultiWave : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgClusterMixMultiWave, EveryReturnPrecedesExactlyOneSuccessor) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}, {1, LaneMask::aiv0()}}, false, 2);
    fixture.graph.executable(1, 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());

    std::set<std::pair<uint32_t, uint8_t>> visits;
    uint32_t successor_entries = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t index = 0; index < 6; ++index) {
            const uint32_t worker = (index + step) % 6;
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.task == 0) {
                EXPECT_LT(execution.block, blocks);
                EXPECT_TRUE(visits.emplace(execution.block, execution.subtask).second);
            } else {
                ASSERT_EQ(execution.task, 1);
                EXPECT_TRUE(fixture.task_done(0));
                EXPECT_EQ(visits.size(), blocks * mask.count());
                ++successor_entries;
            }
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (fixture.task_done(1)) break;
    }
    EXPECT_EQ(successor_entries, 1u);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t lane = 0; lane < 3; ++lane)
            EXPECT_EQ(visits.count({block, lane}), mask.has(static_cast<SubtaskSlot>(lane)) ? 1u : 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), blocks * mask.count());
    EXPECT_TRUE(fixture.task_done(1));
    for (uint32_t worker = 0; worker < 6; ++worker) {
        bool done = false;
        for (uint32_t pass = 0; pass < 4 && !done; ++pass)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndWaves, HbgClusterMixMultiWave,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint16_t>(3, 17, 65))
);

class HbgClusterMixShapes : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgClusterMixShapes, RealOwnDispatchAndCompletionKeepEveryLogicalBlockAndSubtaskExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}});

    ASSERT_TRUE(fixture.prepare_calls());

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
            EXPECT_TRUE(fixture.locals[worker]->execution_at(execution.slot).cluster_mix);
            EXPECT_EQ(worker % 3, execution.subtask);
            if (execution.subtask != 0) {
                EXPECT_EQ(worker % 3, execution.subtask);
                EXPECT_EQ(execution.physical_subblock, worker % 3 - 1);
                if (mask.has_aic() && !mask.has_both_aiv()) single_vector_lanes.insert(worker % 3);
            }
            ++visits[{execution.block, execution.subtask}];
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_completion_done(scheduler_gm_query(fixture.control(0)->completion_state))) break;
    }
    EXPECT_EQ(executions, expected);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            EXPECT_EQ(visits[std::make_pair(block, subtask)], mask.has(static_cast<SubtaskSlot>(subtask)) ? 1u : 0u);
    if (mask.has_aic() && !mask.has_both_aiv())
        EXPECT_EQ(single_vector_lanes, (std::set<uint32_t>{mask.has_aiv0() ? 1U : 2U}));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks);
    EXPECT_EQ(fixture.control(0)->claim_seal, blocks);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), expected
    );
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndClusterCapacity, HbgClusterMixShapes,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint16_t>(1, 2, 3, 4, 5))
);

TEST(HbgClusterMixFloor, InterleavedPureSyncTasksConsumeNoFloorRows) {
    GmMixFixture fixture(
        {{2, LaneMask::aic(), true}, {1, LaneMask::both_aiv()}, {2, LaneMask::aiv0(), true}, {1, LaneMask::all()}},
        true, 2
    );
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_EQ(fixture.storage.layout.protocol.floor_count, 6u);
    for (uint32_t pass = 0; pass < 64 && scheduler_test::resolved_count(fixture.run()) != 4; ++pass) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution entry{};
            if (!fixture.begin(worker, &entry)) continue;
            ASSERT_TRUE(fixture.complete(entry));
        }
    }
    ASSERT_EQ(scheduler_test::resolved_count(fixture.run()), 4u);
    const auto *floors =
        scheduler_state_at<SchedulerProtocolFloorTrace>(fixture.base(), fixture.storage.layout.protocol.floors_offset);
    const std::array<uint32_t, 3> prior{UINT32_MAX, 1, 3};
    const std::array<uint32_t, 3> floor{1, 3, 4};
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        const uint32_t proxy = 3 * cluster + 1;
        bool drained = false;
        for (uint32_t pass = 0; pass < 4 && !drained; ++pass)
            ASSERT_TRUE(fixture.drain(proxy, &drained));
        ASSERT_TRUE(drained);
        for (uint32_t row = 0; row < prior.size(); ++row) {
            const auto &record = floors[2 * row + cluster];
            EXPECT_NE(record.before.sequence, 0u);
            EXPECT_GT(record.after.sequence, record.before.sequence);
            EXPECT_EQ(record.prior_mix_ref, prior[row]);
            EXPECT_EQ(record.floor, floor[row]);
            EXPECT_EQ(record.worker_index, proxy);
        }
    }
}

TEST(HbgCompletionPacking, GroupCreditsLeaveDeferredDebtForTheUniqueFinish) {
    for (uint64_t credits : {2u, 3u}) {
        SchedulerTaskControl control{};
        control.completion_state = scheduler_completion_encode(2 * credits, 2 * credits, 2 * credits + 1);
        uint64_t previous = UINT64_MAX;
        EXPECT_EQ(scheduler_contribute_returns(&control, credits, &previous), SchedulerSubtaskFinishResult::PARTIAL);
        EXPECT_EQ(previous, 0u);
        EXPECT_EQ(scheduler_completion_remaining(control.completion_state), credits);
        EXPECT_EQ(scheduler_completion_debt(control.completion_state), credits + 1);
        EXPECT_EQ(
            scheduler_contribute_returns(&control, credits, &previous), SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED
        );
        EXPECT_EQ(previous, credits);
        EXPECT_EQ(scheduler_completion_remaining(control.completion_state), 0u);
        EXPECT_EQ(scheduler_completion_debt(control.completion_state), 1u);
        EXPECT_FALSE(scheduler_completion_done(control.completion_state));
    }
}

TEST(HbgCompletionPacking, MaximumSupportedShapesIgnoreUnrelatedAsyncActivityUntilTheUniqueLastReturn) {
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
                                                           SchedulerSubtaskFinishResult::TASK_LAST;
                ASSERT_EQ(result, expected);
            }
            const uint64_t state = fixture.control(0)->completion_state;
            EXPECT_EQ(scheduler_completion_count(state), total);
            EXPECT_EQ(scheduler_completion_total(state), total);
            EXPECT_EQ(scheduler_completion_remaining(state), 0u);
            EXPECT_EQ(scheduler_completion_debt(state), 0u);
            EXPECT_TRUE(scheduler_completion_done(state));
            EXPECT_EQ(
                (scheduler_contribute_subtask<false>(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0
                )),
                SchedulerSubtaskFinishResult::ERROR
            );
        }
    }
}

TEST(HbgClusterMixArgsDump, LastSnapshotHoldsExistingCompletionAcrossThreeWaves) {
    GmMixFixture fixture({{3, LaneMask::all()}, {1, LaneMask::aiv0()}}, false, 1);
    fixture.graph.executable(1, 1, {0});
    EXPECT_FALSE(fixture.task_done(1));
    ASSERT_TRUE(fixture.prepare_calls());
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
        bool armed = false;
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
        ASSERT_TRUE(armed);
        for (uint32_t pass = 0; pass < 3; ++pass)
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.service_cooperative(worker));
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
        bool progress = false;
        ASSERT_TRUE(fixture.ordinary_service(1, &progress));
        ASSERT_TRUE(fixture.ordinary_service(1, &progress));
        EXPECT_EQ(scheduler_cluster_mix_armed(fixture.signal(0)->armed_generation), block == 2);
        EXPECT_FALSE(fixture.task_done(0));
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 0u);
    }
    EXPECT_EQ(visible, (std::array<uint32_t, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 6u);
    dump.publisher_after = scheduler_args_dump_publisher(held.worker);
    SchedulerArgsDumpAck after{11, 0, 1, scheduler_args_dump_publisher(held.worker), 1, 0, 0, 0};
    after.status = scheduler_args_dump_ack_commit(0, 1, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_seal(&dump, ArgsDumpStage::AFTER_COMPLETION, after, 11, 1));
    ASSERT_TRUE(fixture.complete(held));
    bool progress = false;
    ASSERT_TRUE(fixture.ordinary_service(1, &progress));
    ASSERT_TRUE(fixture.ordinary_service(1, &progress));
    EXPECT_FALSE(scheduler_cluster_mix_armed(fixture.signal(0)->armed_generation));
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 9u);
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

TEST(HbgClusterGroupReturn, RejectsEarlyAndDuplicateReleaseAndRunErrors) {
    for (bool sync : {false, true}) {
        for (const auto mask : kMixMasks) {
            SCOPED_TRACE(lane_names(mask));
            GmMixFixture fixture({{2, mask, sync}});
            ASSERT_TRUE(fixture.prepare_calls());
            bool armed = false;
            ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
            ASSERT_TRUE(armed);
            for (uint32_t pass = 0; pass < 3; ++pass)
                for (uint32_t lane = 0; lane < 3; ++lane)
                    ASSERT_TRUE(fixture.service_cooperative(lane));
            const uint32_t lane = scheduler_mix_accountant_lane(mask);
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(lane, &execution));
            auto *local = fixture.locals[lane];
            const auto &owner = local->execution_at(execution.slot);
            EXPECT_FALSE(scheduler_cluster_mix_release(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), owner, execution.slot
            ));
            EXPECT_EQ(fixture.signals[0].finished[lane], 0u);
            EXPECT_FALSE(fixture.task_done(0));
            EXPECT_NE(fixture.run()->scheduler_error, 0u);
            ASSERT_TRUE((scheduler_complete_own_slot<false, true, false>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 0
            )));
            EXPECT_EQ(fixture.signals[0].finished[lane], fixture.signals[0].armed_generation + 1);
            EXPECT_FALSE(fixture.complete(execution));
            EXPECT_EQ(fixture.signals[0].accounted_generation, 0u);
            EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        }
    }
}
TEST(HbgClusterGroupReturn, RunErrorSuppressesTheLiveAccountantsCredit) {
    GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool published = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &published));
    ASSERT_TRUE(published);
    for (uint32_t pass = 0; pass < 3; ++pass)
        for (uint32_t lane = 0; lane < 3; ++lane)
            ASSERT_TRUE(fixture.service_cooperative(lane));
    GmExecution accountant{};
    ASSERT_TRUE(fixture.begin(1, &accountant));
    for (uint32_t lane : {0u, 2u}) {
        GmExecution peer{};
        ASSERT_TRUE(fixture.begin(lane, &peer));
        ASSERT_TRUE(fixture.complete(peer));
    }
    const auto graph = fixture.graph.graph();
    scheduler_record_error(fixture.run(), 0, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, fixture.locals[0]);
    EXPECT_FALSE(fixture.complete(accountant));
    EXPECT_EQ(fixture.locals[1]->execution_at(accountant.slot).state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(fixture.signal(0)->accounted_generation, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_FALSE(fixture.task_done(0));
}

TEST(HbgClusterGroupOwnership, FullSelfWindowDoesNotConsumeTheEarlierMixSource) {
    std::vector<GmShape> shapes{{1, LaneMask::all()}};
    shapes.resize(SCHEDULER_EXECUTION_TOKEN_COUNT + 1, GmShape{1, LaneMask::aiv0()});
    GmMixFixture fixture(shapes, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    auto *proxy = fixture.locals[1];
    auto *member = fixture.locals[2];
    bool advanced = false;
    ASSERT_TRUE(
        scheduler_cluster_mix_claim_head(fixture.graph.graph(), fixture.base(), proxy, fixture.run(), 0, &advanced)
    );
    ASSERT_TRUE(advanced);
    for (uint32_t task = 1; task < shapes.size(); ++task)
        ASSERT_TRUE(fixture.claim_ordinary(task, 2));
    ASSERT_EQ(scheduler_execution_token_count(member), SCHEDULER_EXECUTION_TOKEN_COUNT);
    std::array<SchedulerPreparedHandle, SCHEDULER_EXECUTION_TOKEN_COUNT> handles;
    for (uint32_t id = 0; id < handles.size(); ++id)
        handles[id] = member->execution_at(id).prepared;
    ASSERT_TRUE(
        scheduler_cluster_mix_progress_pending(fixture.graph.graph(), fixture.base(), proxy, fixture.run(), &advanced)
    );
    for (uint32_t pass = 0; pass < 3; ++pass)
        for (uint32_t lane = 0; lane < 3; ++lane)
            ASSERT_TRUE(fixture.service_cooperative(lane));
    for (uint32_t worker = 0; worker < 3; ++worker) {
        GmExecution mix{};
        ASSERT_TRUE(fixture.begin(worker, &mix));
        ASSERT_EQ(mix.slot, SCHEDULER_CLUSTER_MIX_EXECUTION);
        ASSERT_TRUE(fixture.complete(mix));
    }
    ASSERT_TRUE(fixture.service_cooperative(1));
    EXPECT_TRUE(fixture.task_done(0));
    for (uint32_t id = 0; id < handles.size(); ++id) {
        EXPECT_EQ(member->execution_at(id).prepared.execution_index, handles[id].execution_index);
        EXPECT_EQ(member->execution_at(id).prepared.claimed_state, handles[id].claimed_state);
    }
    for (uint32_t i = 0; i < handles.size(); ++i) {
        ASSERT_TRUE(fixture.fill(2));
        GmExecution owned{};
        ASSERT_TRUE(fixture.begin(2, &owned));
        ASSERT_LT(owned.slot, SCHEDULER_EXECUTION_TOKEN_COUNT);
        ASSERT_TRUE(fixture.complete(owned));
    }
    EXPECT_EQ(scheduler_execution_token_count(member), 0u);
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), shapes.size());
}

TEST(HbgClusterDescriptor, RejectsMalformedGroupIdentityOrUnsupportedPredicate) {
    for (uint32_t fault = 0; fault < 5; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
        ASSERT_TRUE(fixture.prepare_calls());
        bool published = false;
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &published));
        ASSERT_TRUE(published);
        auto *signal = fixture.signal(0);
        switch (fault) {
        case 0:
            ++signal->run_epoch;
            break;
        case 1:
            ++signal->rank;
            break;
        case 2:
            ++signal->block_idx;
            break;
        case 3:
            signal->anchor_state = 0;
            break;
        case 4:
            fixture.storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
            break;
        }
        bool progress = false;
        EXPECT_FALSE(scheduler_cluster_mix_service(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress
        ));
        EXPECT_EQ(signal->arrived[0], 0u);
        EXPECT_EQ(signal->accounted_generation, 0u);
        EXPECT_EQ(
            fixture.locals[0]->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).state, SchedulerDispatchSlotState::FREE
        );
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR));
    }
}

TEST(HbgClusterGroupOwnership, CancelAnUnpublishedAnchorWithoutCreatingAnExecutionOrACompletion) {
    GmMixFixture fixture({{2, LaneMask::all()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *proxy = fixture.locals[1];
    bool progress = false;
    ASSERT_TRUE(
        scheduler_cluster_mix_claim_head(fixture.graph.graph(), fixture.base(), proxy, fixture.run(), 0, &progress)
    );
    ASSERT_TRUE(progress);
    const auto handle = proxy->cluster_pending.anchor;
    auto *control = scheduler_prepared_control_at(fixture.base(), proxy->config.prepared, handle.execution_index);
    EXPECT_EQ(scheduler_prepared_phase(control->state), SchedulerPreparedPhase::GROUP_RESERVED);
    ASSERT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), proxy));
    EXPECT_EQ(scheduler_prepared_phase(control->state), SchedulerPreparedPhase::FAILED);
    EXPECT_EQ(fixture.signal(0)->armed_generation, 0u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
}
}  // namespace
