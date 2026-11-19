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
class HbgClusterOwnedGroups : public ::testing::TestWithParam<std::tuple<LaneMask, bool>> {};

TEST_P(HbgClusterOwnedGroups, ActualMembersCompleteWithoutASecondProxyVisit) {
    const auto [active, sync] = GetParam();
    constexpr uint32_t blocks = 2;
    GmMixFixture fixture({{blocks, active, sync}}, false, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    auto &storage = fixture.storage;
    auto &locals = fixture.locals;
    auto &signals = fixture.signals;
    auto &graph = fixture.graph;
    auto *base = fixture.base();
    auto *control = fixture.control(0);
    const uint32_t accountant = scheduler_mix_accountant_lane(active);
    for (uint32_t round = 0; round < blocks; ++round) {
        const uint32_t cluster = sync ? round : 0;
        const auto first = cluster * 3;
        auto *proxy = locals[first + 1];
        auto &signal = signals[cluster];
        bool advanced = false;
        ASSERT_TRUE(scheduler_cluster_mix_claim_head(graph.graph(), base, proxy, storage.run_control, 0, &advanced));
        ASSERT_TRUE(advanced);
        ASSERT_TRUE(scheduler_cluster_mix_progress_pending(graph.graph(), base, proxy, storage.run_control, &advanced));
        ASSERT_TRUE(advanced);
        const uint32_t generation = signal.armed_generation;
        ASSERT_TRUE(scheduler_cluster_mix_armed(generation));
        ASSERT_EQ(signal.active_mask, active.raw());
        ASSERT_EQ(
            signal.anchor_state, scheduler_prepared_state(SchedulerPreparedPhase::GROUP_ARMED, proxy->worker_id())
        );
        // Deliberately withhold all noncomputing lanes, including the AIV0 publisher of an AIC+AIV1 group.
        for (uint32_t turn = 0; turn < 3; ++turn)
            for (uint32_t lane = 0; lane < 3; ++lane)
                if (active.has(static_cast<SubtaskSlot>(lane)))
                    ASSERT_TRUE(scheduler_cluster_mix_service(
                        graph.graph(), base, locals[first + lane], storage.run_control, 0, &advanced
                    ));
        for (uint32_t lane = 0; lane < 3; ++lane) {
            auto *local = locals[first + lane];
            if (!active.has(static_cast<SubtaskSlot>(lane))) {
                EXPECT_EQ(signal.arrived[lane], 0u);
                EXPECT_EQ(signal.finished[lane], 0u);
                EXPECT_EQ(local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).state, SchedulerDispatchSlotState::FREE);
                continue;
            }
            GmExecution unit{};
            ASSERT_TRUE(fixture.begin(first + lane, &unit));
            ASSERT_EQ(unit.slot, SCHEDULER_CLUSTER_MIX_EXECUTION);
        }
        auto finish = [&](uint32_t lane) {
            return scheduler_complete_own_slot(
                graph.graph(), base, locals[first + lane], storage.run_control, SCHEDULER_CLUSTER_MIX_EXECUTION, 0
            );
        };
        // The accountant returns before the other actual members.
        ASSERT_TRUE(finish(accountant));
        EXPECT_LT(signal.accounted_generation, generation);
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (lane != accountant && active.has(static_cast<SubtaskSlot>(lane))) ASSERT_TRUE(finish(lane));
        ASSERT_TRUE(scheduler_cluster_mix_service(
            graph.graph(), base, locals[first + accountant], storage.run_control, 0, &advanced
        ));
        ASSERT_EQ(signal.accounted_generation, generation);
        EXPECT_EQ(scheduler_completion_count(control->completion_state), (round + 1) * active.count());
        EXPECT_EQ(signal.armed_generation, generation);
        ASSERT_TRUE(scheduler_cluster_mix_progress_pending(graph.graph(), base, proxy, storage.run_control, &advanced));
        ASSERT_TRUE(advanced);
        EXPECT_EQ(signal.armed_generation, generation + 1);
        if (sync && round == 0) {
            ASSERT_TRUE(
                scheduler_cluster_mix_claim_head(graph.graph(), base, proxy, storage.run_control, 0, &advanced)
            );
            EXPECT_FALSE(advanced);
            EXPECT_EQ(control->claim_seal, 0u);
            EXPECT_EQ(signals[1].armed_generation, 0u);
        }
    }
    EXPECT_TRUE(scheduler_completion_done(control->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(storage.run_control), 1u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}
TEST(HbgClusterOrder, EarlierProducerCanEnterWhileSyncConsumersAndLaterMixShareVectorLanes) {
    GmMixFixture fixture(
        {{1, LaneMask::aic()}, {2, LaneMask::aiv0(), true}, {1, LaneMask{true, true, false}}}, false, 1
    );
    fixture.graph.executable(1, 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 0));
    ASSERT_TRUE(fixture.progress_ordinary(0));
    ASSERT_TRUE(fixture.claim_ordinary(1, 1));
    ASSERT_TRUE(fixture.claim_ordinary(1, 2));
    ASSERT_EQ(fixture.control(1)->claim_seal, 2u);
    for (uint32_t pass = 0; pass < 8 && fixture.signal(0)->armed_generation == 0; ++pass)
        ASSERT_TRUE(fixture.fill(1));
    ASSERT_EQ(fixture.signal(0)->armed_generation, 1u);
    ASSERT_EQ(fixture.signal(0)->task_ref, 2);
    bool progress = false;
    for (uint32_t worker : {1u, 0u, 1u, 0u})
        ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
    ASSERT_EQ(fixture.signal(0)->arrived[0], 0u);
    ASSERT_EQ(fixture.signal(0)->arrived[1], 0u);
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_EQ(producer.task, 0);
    ASSERT_TRUE(fixture.complete(producer));
    for (uint32_t pass = 0; pass < 16 && !fixture.task_done(2); ++pass) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution unit{};
            if (fixture.begin(worker, &unit)) ASSERT_TRUE(fixture.complete(unit));
        }
    }
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_TRUE(fixture.task_done(2));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgClusterOrder, PureAiv1SyncStartUsesBothVectorWorkersWithLogicalSubtaskTwo) {
    GmMixFixture fixture({{2, LaneMask::aiv1(), true}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 1));
    ASSERT_TRUE(fixture.claim_ordinary(0, 2));
    std::set<uint32_t> blocks;
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.progress_ordinary(worker));
        GmExecution unit{};
        ASSERT_TRUE(fixture.begin(worker, &unit));
        EXPECT_EQ(unit.subtask, 2u);
        EXPECT_EQ(unit.physical_subblock, 1u);
        EXPECT_TRUE(blocks.insert(unit.block).second);
        ASSERT_TRUE(fixture.complete(unit));
    }
    EXPECT_EQ(blocks, (std::set<uint32_t>{0, 1}));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgClusterOrder, ActualMemberFinAndAccountantReceiptProtectTheirOwnGeneration) {
    SchedulerClusterSignal signal{};
    scheduler_cluster_signal_clear(&signal, 47);
    SchedulerLocalState local{};
    local.cluster_signal = &signal;
    scheduler_cluster_store_field(&signal.rank, UINT32_C(10));
    scheduler_cluster_store_field(&signal.active_mask, UINT32_C(5));
    scheduler_cluster_store_release(&signal.armed_generation, UINT32_C(1));
    local.config.self_lane = 1;
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    local.config.self_lane = 2;
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 10));
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
    scheduler_cluster_store_release(&signal.finished[2], UINT32_C(1));
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    local.config.self_lane = 0;
    scheduler_cluster_store_release(&signal.finished[0], UINT32_C(1));
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
    scheduler_cluster_store_release(&signal.accounted_generation, UINT32_C(1));
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    // Another block of rank 10 cannot use the prior generation's FIN or receipt.
    scheduler_cluster_store_release(&signal.armed_generation, UINT32_C(3));
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
    local.config.self_lane = 2;
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
}

INSTANTIATE_TEST_SUITE_P(
    PhysicalMasks, HbgClusterOwnedGroups,
    ::testing::Combine(
        ::testing::Values(
            LaneMask{true, true, false}, LaneMask{true, false, true}, LaneMask::both_aiv(), LaneMask::all()
        ),
        ::testing::Bool()
    )
);
}  // namespace
