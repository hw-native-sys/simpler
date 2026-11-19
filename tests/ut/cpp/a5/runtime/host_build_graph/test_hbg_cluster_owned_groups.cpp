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

#include <atomic>
#include <set>
#include <thread>

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
    std::set<uint32_t> tickets;
    for (uint32_t round = 0; round < blocks; ++round) {
        const uint32_t cluster = sync ? round : 0;
        const auto first = cluster * 3;
        auto *proxy = locals[first + 1];
        auto &signal = signals[cluster];
        bool advanced = false;
        ASSERT_TRUE(scheduler_claim_for_build(graph.graph(), base, proxy, storage.run_control, 0, &advanced));
        ASSERT_TRUE(advanced);
        const uint32_t block = proxy->cluster_pending.block_idx;
        ASSERT_LT(block, blocks);
        ASSERT_TRUE(tickets.insert(block).second);
        ASSERT_TRUE(scheduler_progress_mix_publication(graph.graph(), base, proxy, storage.run_control, &advanced));
        ASSERT_TRUE(advanced);
        const uint32_t generation = signal.groups[0].armed_generation;
        ASSERT_TRUE(scheduler_cluster_mix_armed(generation));
        ASSERT_EQ(signal.groups[0].active_mask, active.raw());
        const auto &context = *proxy->config.shared_context;
        EXPECT_EQ(
            context.mix_proxy_published[0], scheduler_mix_proxy_state(SchedulerMixProxyPhase::PUBLISHED, 0, block)
        );
        EXPECT_EQ(signal.groups[0].block_idx, block);
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
                EXPECT_EQ(signal.groups[0].arrived[lane], 0u);
                EXPECT_EQ(signal.groups[0].finished[lane], 0u);
                EXPECT_EQ(local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).state, SchedulerDispatchSlotState::FREE);
                continue;
            }
            const auto &handle = local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).prepared;
            const auto *cell = scheduler_prepared_control_at(base, local->config.prepared, handle.execution_index);
            const auto worker = local->worker_id();
            EXPECT_EQ(
                cell->state,
                scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker) | SCHEDULER_PREPARED_MIX_OWNED
            );
            auto unentered = handle;
            EXPECT_FALSE(scheduler_return_prepared_execution(base, local->config.prepared, &unentered));
            GmExecution unit{};
            ASSERT_TRUE(fixture.begin(first + lane, &unit));
            ASSERT_EQ(unit.slot, SCHEDULER_CLUSTER_MIX_EXECUTION);
            EXPECT_EQ(
                cell->state,
                scheduler_prepared_state(SchedulerPreparedPhase::ENTERED, worker) | SCHEDULER_PREPARED_MIX_OWNED
            );
        }
        auto finish = [&](uint32_t lane) {
            auto *local = locals[first + lane];
            const auto index = local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).prepared.execution_index;
            const bool ok = scheduler_complete_lane_execution(
                graph.graph(), base, local, storage.run_control, SCHEDULER_CLUSTER_MIX_EXECUTION, 0
            );
            const auto state = scheduler_prepared_control_at(base, local->config.prepared, index)->state;
            EXPECT_EQ(
                state, scheduler_prepared_state(SchedulerPreparedPhase::RETURNED, local->worker_id()) |
                           SCHEDULER_PREPARED_MIX_OWNED
            );
            return ok;
        };
        // The accountant returns before the other actual members.
        ASSERT_TRUE(finish(accountant));
        EXPECT_LT(signal.groups[0].accounted_generation, generation);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (lane == accountant || !active.has(static_cast<SubtaskSlot>(lane))) continue;
            ASSERT_TRUE(finish(lane));
            // The descriptor is still published. A returning member may see it again before the accountant runs.
            ASSERT_TRUE(scheduler_cluster_mix_service(
                graph.graph(), base, locals[first + lane], storage.run_control, 0, &advanced
            ));
            EXPECT_FALSE(advanced);
            GmExecution duplicate{};
            EXPECT_FALSE(fixture.begin(first + lane, &duplicate));
        }
        ASSERT_TRUE(scheduler_cluster_mix_service(
            graph.graph(), base, locals[first + accountant], storage.run_control, 0, &advanced
        ));
        ASSERT_EQ(signal.groups[0].accounted_generation, generation);
        EXPECT_EQ(scheduler_completion_count(control->completion_state), (round + 1) * active.count());
        EXPECT_EQ(signal.groups[0].armed_generation, generation);
        ASSERT_TRUE(scheduler_progress_mix_publication(graph.graph(), base, proxy, storage.run_control, &advanced));
        ASSERT_TRUE(advanced);
        EXPECT_EQ(signal.groups[0].armed_generation, generation + 1);
        if (sync && round == 0) {
            ASSERT_TRUE(scheduler_claim_for_build(graph.graph(), base, proxy, storage.run_control, 0, &advanced));
            EXPECT_FALSE(advanced);
            EXPECT_EQ(control->claim_seal, 0u);
            EXPECT_EQ(signals[1].groups[0].armed_generation, 0u);
        }
    }
    EXPECT_TRUE(scheduler_completion_done(control->completion_state));
    EXPECT_EQ(scheduler_test::resolved_count(storage.run_control), 1u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}
TEST(HbgClusterOwnedGroups, CompetingProxiesCanOnlyDistributeOneCopyOfTheSameMixBlock) {
    const auto lanes = LaneMask::both_aiv();
    GmMixFixture fixture({{1, lanes}}, true, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    const auto graph = fixture.graph.graph();
    auto *base = fixture.base();
    auto *run = fixture.run();

    // Both AIV0s have selected the same open issue head before either can take its only ticket.
    std::atomic<uint32_t> selected{0};
    std::array<bool, 2> ok{}, claimed{};
    const auto claim = [&](uint32_t cluster) {
        auto *proxy = fixture.locals[3 * cluster + 1];
        int64_t head = SCHEDULER_TASK_REF_INVALID;
        bool progress = false;
        ok[cluster] = scheduler_issue_head(graph, base, proxy, run, &head, &progress) && head == 0;
        selected.fetch_add(1, std::memory_order_acq_rel);
        while (selected.load(std::memory_order_acquire) < 2)
            std::this_thread::yield();
        if (ok[cluster]) ok[cluster] = scheduler_claim_for_build(graph, base, proxy, run, head, &claimed[cluster]);
    };
    std::thread peer(claim, 1);
    claim(0);
    peer.join();
    ASSERT_TRUE(ok[0] && ok[1]);
    ASSERT_NE(claimed[0], claimed[1]);
    const uint32_t winner = claimed[0] ? 0 : 1;
    const uint32_t other = 1 - winner;
    ASSERT_EQ(fixture.control(0)->next_block, 1u);
    ASSERT_EQ(fixture.control(0)->claim_seal, 1u);
    ASSERT_EQ(fixture.locals[3 * winner + 1]->cluster_pending.block_idx, 0u);
    ASSERT_EQ(fixture.locals[3 * other + 1]->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);

    bool progress = false;
    for (uint32_t cluster : {other, winner}) {
        ASSERT_TRUE(scheduler_progress_mix_publication(graph, base, fixture.locals[3 * cluster + 1], run, &progress));
        ASSERT_EQ(progress, cluster == winner);
    }
    const auto &descriptor = fixture.signal(winner)->groups[0];
    ASSERT_TRUE(scheduler_cluster_mix_armed(descriptor.armed_generation));
    ASSERT_EQ(descriptor.task_ref, 0);
    ASSERT_EQ(descriptor.block_idx, 0u);
    ASSERT_FALSE(scheduler_cluster_mix_armed(fixture.signal(winner)->groups[1].armed_generation));
    for (const auto &unused : fixture.signal(other)->groups)
        ASSERT_FALSE(scheduler_cluster_mix_armed(unused.armed_generation));

    // A physical lane reads only its own cluster's SSBUF. Re-observing an armed generation must not reacquire it.
    const auto service_members = [&] {
        for (uint32_t worker : {1u, 2u, 4u, 5u})
            ASSERT_TRUE(scheduler_cluster_mix_service(graph, base, fixture.locals[worker], run, 0, &progress));
    };
    for (uint32_t pass = 0; pass < 3; ++pass)
        ASSERT_NO_FATAL_FAILURE(service_members());
    for (uint32_t lane : {2u, 1u}) {
        const uint32_t owner = 3 * winner + lane;
        GmExecution execution{};
        ASSERT_FALSE(fixture.begin(3 * other + lane, &execution));
        ASSERT_TRUE(fixture.begin(owner, &execution));
        ASSERT_EQ(execution.task, 0);
        ASSERT_EQ(execution.block, 0u);
        ASSERT_EQ(execution.subtask, lane);
        const auto index = fixture.locals[owner]->execution_at(execution.slot).prepared.execution_index;
        const auto *state = scheduler_prepared_control_at(base, fixture.locals[owner]->config.prepared, index);
        ASSERT_TRUE(fixture.complete(execution));
        ASSERT_NO_FATAL_FAILURE(service_members());
        EXPECT_FALSE(fixture.begin(owner, &execution));
        EXPECT_FALSE(fixture.begin(3 * other + lane, &execution));
        EXPECT_EQ(
            state->state,
            scheduler_prepared_state(SchedulerPreparedPhase::RETURNED, owner) | SCHEDULER_PREPARED_MIX_OWNED
        );
        const auto &trace = scheduler_state_at<SchedulerProtocolExecutionTrace>(
            base, fixture.storage.layout.protocol.executions_offset
        )[index];
        EXPECT_EQ(trace.worker_index, owner);
        EXPECT_EQ(trace.group_seq, descriptor.armed_generation);
        EXPECT_EQ(trace.publication_count, 1u);
        EXPECT_NE(trace.entry.sequence, 0u);
        EXPECT_NE(trace.returned.sequence, 0u);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), lane == 1 ? lanes.count() : 0u);
    }
    EXPECT_EQ(descriptor.accounted_generation, descriptor.armed_generation);
    EXPECT_EQ(scheduler_test::resolved_count(run), 1u);
    EXPECT_EQ(scheduler_read_error(run), 0u);
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
    for (uint32_t pass = 0; pass < 8 && fixture.signal(0)->groups[0].armed_generation == 0; ++pass)
        ASSERT_TRUE(fixture.fill(1));
    ASSERT_EQ(fixture.signal(0)->groups[0].armed_generation, 1u);
    ASSERT_EQ(fixture.signal(0)->groups[0].task_ref, 2);
    bool progress = false;
    for (uint32_t worker : {1u, 0u, 1u, 0u})
        ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
    ASSERT_EQ(fixture.signal(0)->groups[0].arrived[0], 0u);
    ASSERT_EQ(fixture.signal(0)->groups[0].arrived[1], 0u);
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

TEST(HbgClusterOrder, MixArrivalWaitsForAnEarlierTokenClaimedWhilePeersBecomeOrderReady) {
    constexpr uint32_t vector_sync = 0, ordinary = 1, mix = 2;
    GmMixFixture fixture({{2, LaneMask::aiv0(), true}, {1, LaneMask::aic()}, {1, kAicAiv0}}, false, 1);
    ASSERT_TRUE(fixture.claim_ordinary(vector_sync, 1));
    ASSERT_TRUE(fixture.claim_ordinary(vector_sync, 2));
    bool published = false;
    for (uint32_t pass = 0; pass < 3 && !published; ++pass)
        ASSERT_TRUE(fixture.ordinary_arm(mix, 0, &published));
    ASSERT_TRUE(published);
    auto &group = fixture.signal(0)->groups[0];
    const uint32_t generation = group.armed_generation;
    ASSERT_TRUE(scheduler_cluster_mix_armed(generation));
    ASSERT_TRUE(fixture.service_cooperative(1));
    ASSERT_TRUE(fixture.service_cooperative(0));
    ASSERT_EQ(fixture.locals[0]->mix_groups[0].phase, SchedulerGmMixPhase::ORDER_READY);
    ASSERT_EQ(group.order_ready[0], generation);
    ASSERT_EQ(group.order_ready[1], 0u);
    ASSERT_TRUE(fixture.claim_ordinary(ordinary, 0));

    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.progress_ordinary(worker));
        GmExecution sync{};
        ASSERT_TRUE(fixture.begin(worker, &sync));
        ASSERT_EQ(sync.task, vector_sync);
        ASSERT_TRUE(fixture.complete(sync));
    }
    ASSERT_TRUE(fixture.service_cooperative(1));
    ASSERT_EQ(group.order_ready[1], generation);
    ASSERT_TRUE(fixture.service_cooperative(0));
    ASSERT_EQ(group.arrived[0], 0u);

    ASSERT_TRUE(fixture.progress_ordinary(0));
    GmExecution earlier{};
    ASSERT_TRUE(fixture.begin(0, &earlier));
    ASSERT_EQ(earlier.task, ordinary);
    ASSERT_TRUE(fixture.complete(earlier));
    ASSERT_TRUE(fixture.service_cooperative(0));
    ASSERT_EQ(group.arrived[0], generation);
    for (uint32_t worker : {0u, 1u}) {
        ASSERT_TRUE(fixture.service_cooperative(worker));
        GmExecution member{};
        ASSERT_TRUE(fixture.begin(worker, &member));
        ASSERT_EQ(member.task, mix);
        ASSERT_TRUE(fixture.complete(member));
    }
    EXPECT_TRUE(fixture.task_done(mix));
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
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

TEST_P(HbgClusterOwnedGroups, MembersClaimBothDependentMixExecutionsBeforeTheProducerReturns) {
    const auto [mask, sync] = GetParam();
    const uint16_t blocks = sync ? 2 : 1;
    const uint32_t workers = 3 * blocks;
    GmMixFixture fixture({{1, LaneMask::aic()}, {blocks, mask, sync}, {blocks, mask, sync}}, false, blocks + 1);
    const uint8_t role = static_cast<uint8_t>(__builtin_ctz(mask.raw()));
    fixture.graph.executable(1, role, {0});
    fixture.graph.executable(2, role, {1});
    ASSERT_TRUE(fixture.claim_ordinary(0, workers));
    ASSERT_TRUE(fixture.progress_ordinary(workers));
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(workers, &producer));

    // The producer occupies another cluster. This cluster can claim two MIX groups while the first awaits data.
    for (int pass = 0; pass < 12; ++pass)
        for (uint32_t worker = 0; worker < workers; ++worker)
            ASSERT_TRUE(fixture.fill(worker));
    for (uint32_t worker = 0; worker < workers; ++worker) {
        if (!mask.has(static_cast<SubtaskSlot>(worker % 3))) continue;
        std::set<int64_t> claimed;
        for (uint32_t slot = SCHEDULER_CLUSTER_MIX_EXECUTION; slot < SCHEDULER_LOCAL_EXECUTION_COUNT; ++slot) {
            const auto &owner = fixture.locals[worker]->execution_at(slot);
            EXPECT_EQ(
                owner.prepared.claimed_state,
                scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, worker) | SCHEDULER_PREPARED_MIX_OWNED
            );
            EXPECT_EQ(owner.state, SchedulerDispatchSlotState::FILLING);
            EXPECT_TRUE(claimed.insert(owner.task_ref).second);
        }
        EXPECT_EQ(claimed, (std::set<int64_t>{1, 2}));
        GmExecution premature{};
        EXPECT_FALSE(fixture.begin(worker, &premature));
    }
    ASSERT_TRUE(fixture.complete(producer));
    for (int64_t task : {1, 2}) {
        std::set<uint32_t> members;
        for (int pass = 0; pass < 12 && members.size() != blocks * mask.count(); ++pass) {
            for (uint32_t worker = 0; worker < workers; ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution execution{};
                if (!fixture.begin(worker, &execution)) continue;
                ASSERT_EQ(execution.task, task);
                ASSERT_TRUE(members.insert(worker).second);
                ASSERT_TRUE(fixture.complete(execution));
            }
        }
        ASSERT_EQ(members.size(), blocks * mask.count());
        for (uint32_t worker = 0; worker < workers; ++worker)
            ASSERT_TRUE(fixture.service_cooperative(worker));
        EXPECT_TRUE(fixture.task_done(task));
    }
    EXPECT_EQ(scheduler_test::resolved_count(fixture.run()), 3u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

class HbgEarlierProducer : public ::testing::TestWithParam<bool> {};

TEST_P(HbgEarlierProducer, LaterMixAllowsOrdinaryProducerToReleaseTheEarlierTask) {
    const bool earlier_sync = GetParam();
    constexpr int64_t remote = 0, ready = 1, producer = 2, relay = 3, first = 4, later = 5;
    GmMixFixture f(
        {{1, LaneMask::aic()},
         {1, LaneMask::aic()},
         {1, LaneMask::aiv0()},
         {1, LaneMask::aic()},
         earlier_sync ? GmShape{2, LaneMask::aic(), true} : GmShape{1, kAicAiv0},
         {1, kAicAiv1}},
        false, 2
    );
    f.graph.executable(producer, 1, {remote});
    f.graph.executable(relay, 0, {ready});
    f.graph.executable(first, 0, {producer});
    f.graph.executable(later, 0, {relay});
    const auto *meta = f.storage.metadata;
    const std::vector<SchedulerTaskMetadata> metadata(meta, meta + f.storage.layout.task_count);
    SchedulerTaskCatalogPlan plan;
    ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 2, f.storage.catalog_begin, &plan));
    ASSERT_TRUE(scheduler_order_task_catalog(f.graph.graph(), metadata, {}, &plan));
    ASSERT_EQ(plan.task_order, (std::vector<int64_t>{remote, ready, producer, relay, first, later}));
    ASSERT_TRUE(f.prepare_calls());

    // Delay the other cluster so an ordinary producer holds a self-claim slot on this cluster's AIV1.
    ASSERT_TRUE(f.claim_ordinary(remote, 3));
    ASSERT_TRUE(f.progress_ordinary(3));
    GmExecution running{};
    ASSERT_TRUE(f.begin(3, &running));
    for (const auto task : {ready, relay}) {
        ASSERT_TRUE(f.claim_ordinary(task, 0));
        ASSERT_TRUE(f.progress_ordinary(0));
        GmExecution execution{};
        ASSERT_TRUE(f.begin(0, &execution));
        ASSERT_EQ(execution.task, task);
        ASSERT_TRUE(f.complete(execution));
    }
    ASSERT_TRUE(f.claim_ordinary(producer, 2));
    auto service = [&](bool remote_returned) {
        for (uint32_t worker : {1U, 0U, 2U, 3U, 4U, 5U}) {
            if (!remote_returned && worker >= 3) continue;
            bool failed = false;
            uint32_t selected = UINT32_MAX;
            scheduler_find_local_execution(f.graph.graph(), f.base(), f.locals[worker], f.run(), 0, &failed, &selected);
            ASSERT_FALSE(failed);
            if (selected == UINT32_MAX) continue;
            GmExecution execution{};
            ASSERT_TRUE(f.begin(worker, &execution));
            ASSERT_TRUE(f.complete(execution));
        }
    };
    auto &newer = f.signal(0)->groups[earlier_sync ? 0 : 1];
    for (unsigned pass = 0; pass < 12 && !newer.dependencies_generation; ++pass)
        service(false);
    ASSERT_EQ(newer.task_ref, later);
    ASSERT_EQ(newer.dependencies_generation, newer.armed_generation);

    // Both groups need the AIC. AIV1 must run the producer before the earlier group can release the AIC to the later
    // group.
    ASSERT_TRUE(f.complete(running));
    for (unsigned pass = 0; pass < 16 && !f.task_done(later); ++pass)
        service(true);
    EXPECT_TRUE(f.task_done(producer));
    EXPECT_TRUE(f.task_done(first));
    EXPECT_TRUE(f.task_done(later));
    EXPECT_EQ(scheduler_read_error(f.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(EarlierMixOrPureSync, HbgEarlierProducer, ::testing::Bool());

TEST(HbgClusterGroupOwnership, InvalidMemberCallPublishesFailedWithoutArriving) {
    GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool published = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &published));
    ASSERT_TRUE(published);
    auto *member = fixture.locals[2];
    uint64_t index = UINT64_MAX;
    ASSERT_TRUE(scheduler_prepared_execution_index(fixture.base(), member->config.prepared, 0, 0, 2, &index));
    scheduler_prepared_call_at(fixture.base(), member->config.prepared, index)->function_bin_addr = 0;
    bool progress = false;
    EXPECT_FALSE(
        scheduler_cluster_mix_service(fixture.graph.graph(), fixture.base(), member, fixture.run(), 0, &progress)
    );
    const auto state = scheduler_prepared_control_at(fixture.base(), member->config.prepared, index)->state;
    EXPECT_EQ(
        state,
        scheduler_prepared_state(SchedulerPreparedPhase::FAILED, member->worker_id()) | SCHEDULER_PREPARED_MIX_OWNED
    );
    EXPECT_EQ(fixture.signal(0)->groups[0].arrived[2], 0u);
    EXPECT_EQ(fixture.signal(0)->groups[0].finished[2], 0u);
}

TEST(HbgClusterOrder, ActualMemberFinAndAccountantReceiptProtectTheirOwnGeneration) {
    SchedulerClusterSignal signal{};
    scheduler_cluster_signal_clear(&signal, 47);
    SchedulerLocalState local{};
    local.cluster_signal = &signal;
    scheduler_cluster_store_field(&signal.groups[0].rank, UINT32_C(10));
    scheduler_cluster_store_field(&signal.groups[0].active_mask, UINT32_C(5));
    scheduler_cluster_store_release(&signal.groups[0].armed_generation, UINT32_C(1));
    local.config.self_lane = 1;
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    local.config.self_lane = 2;
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 10));
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
    scheduler_cluster_store_release(&signal.groups[0].finished[2], UINT32_C(1));
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    local.config.self_lane = 0;
    scheduler_cluster_store_release(&signal.groups[0].finished[0], UINT32_C(1));
    EXPECT_TRUE(scheduler_cluster_owes_before(&local, 11));
    scheduler_cluster_store_release(&signal.groups[0].accounted_generation, UINT32_C(1));
    EXPECT_FALSE(scheduler_cluster_owes_before(&local, 11));
    // Another block of rank 10 cannot use the prior generation's FIN or receipt.
    scheduler_cluster_store_release(&signal.groups[0].armed_generation, UINT32_C(3));
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
