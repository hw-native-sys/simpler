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

TEST(HbgMixDependencies, LateMemberRequiresASuccessfulPeerPublicationForItsGeneration) {
    for (bool ordered : {false, true}) {
        GmMixFixture f({{1, LaneMask::aic()}, {2, LaneMask::all()}});
        f.graph.executable(1, 0, {0});
        ASSERT_TRUE(f.prepare_calls());
        auto *local = f.locals[2];
        auto &group = local->mix_groups[0];
        group.generation = 3;
        group.task_ref = 1;
        group.requires_peer_order = ordered;
        ASSERT_EQ(scheduler_node_fanin_view(f.graph.graph(), 1, &group.fanin), SchedulerGraphResult::OK);
        auto &signal = f.signal(0)->groups[0];
        auto *witness = ordered ? signal.order_ready : signal.finished;
        scheduler_cluster_store_release(&witness[0], 1);
        scheduler_cluster_store_release(&witness[1], 4);
        bool progress = false;
        ASSERT_TRUE(scheduler_poll_mix_fanin(f.graph.graph(), f.base(), local, f.run(), &progress, 0, true));
        EXPECT_FALSE(progress);
        EXPECT_EQ(group.next_fanin, 0u);
        group.wait = {};
        scheduler_cluster_store_release(&witness[1], 3);
        ASSERT_TRUE(scheduler_poll_mix_fanin(f.graph.graph(), f.base(), local, f.run(), &progress, 0, true));
        EXPECT_TRUE(progress);
        EXPECT_EQ(group.next_fanin, group.fanin.count);
        EXPECT_NE(scheduler_cached_task_proof(local, 0), SchedulerTaskProof::DONE);
    }
}

TEST(HbgFixedOwnerWave, SingleCrossEngineInputChecksSupplyBeforeBindingAndCompletionBeforeEntry) {
    GmMixFixture f({{1, LaneMask::aic()}, {4, LaneMask::aiv0()}});
    f.graph.executable(1, 0, {0});
    f.storage.metadata[1].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    auto *consumer = f.locals[1];
    const SchedulerCatalogEntry row{1, 0};
    ASSERT_EQ(consumer->config.engine_catalog.count, 1u);
    consumer->config.engine_catalog.entries = &row;
    ASSERT_TRUE(f.fill(1));
    EXPECT_FALSE(scheduler_execution_token_has_task(consumer, 1));
    ASSERT_TRUE(f.fill(0));
    ASSERT_TRUE(f.fill(1));
    EXPECT_TRUE(scheduler_execution_token_has_task(consumer, 1));
    GmExecution entry{};
    EXPECT_FALSE(f.begin(1, &entry));
    ASSERT_TRUE(f.begin(0, &entry));
    ASSERT_TRUE(f.complete(entry));
    ASSERT_TRUE(f.fill(1));
    ASSERT_TRUE(f.begin(1, &entry));
    EXPECT_EQ(entry.task, 1);
}

TEST(HbgFixedOwnerWave, StaticSupplyStillWaitsForTheLastProducerBlockToFinish) {
    GmMixFixture f({{2, LaneMask::aic()}, {4, LaneMask::aiv0()}});
    f.graph.executable(1, 0, {0});
    for (uint32_t ref = 0; ref < 2; ++ref)
        f.storage.metadata[ref].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    auto *consumer = f.locals[1];
    const SchedulerCatalogEntry row{1, SCHEDULER_SUPPLY_STATIC};
    ASSERT_EQ(consumer->config.engine_catalog.count, 1u);
    consumer->config.engine_catalog.entries = &row;
    ASSERT_TRUE(f.fill(1));
    EXPECT_TRUE(scheduler_execution_token_has_task(consumer, 1));
    EXPECT_EQ(consumer->issue_binding.next_fanin, 0u);
    for (uint32_t worker : {0u, 3u}) {
        GmExecution entry{};
        EXPECT_FALSE(f.begin(1, &entry));
        ASSERT_TRUE(f.fill(worker));
        ASSERT_TRUE(f.begin(worker, &entry));
        ASSERT_EQ(entry.task, 0);
        ASSERT_TRUE(f.complete(entry));
        ASSERT_TRUE(f.fill(1));
    }
    GmExecution entry{};
    ASSERT_TRUE(f.begin(1, &entry));
    EXPECT_EQ(entry.task, 1);
    EXPECT_TRUE(f.task_done(0));
}

TEST(HbgFixedOwnerWave, TerminalMixCanBePublishedBeforeItsCrossEnginePredecessorReturns) {
    GmMixFixture f({{2, LaneMask::aic()}, {2, LaneMask::all()}, {2, LaneMask::all(), true}});
    f.graph.executable(1, 0, {0});
    f.graph.executable(2, 0, {1});
    for (uint32_t ref = 0; ref < 3; ++ref)
        f.storage.metadata[ref].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    for (uint32_t worker : {1u, 2u, 4u, 5u}) {
        auto *local = f.locals[worker];
        local->config.engine_catalog.terminal_mix_begin = 0;
        ASSERT_TRUE(f.fill(worker));
        EXPECT_EQ(local->issue_cursor, local->is_mix_coordinator() ? 1u : 2u);
        EXPECT_TRUE(scheduler_task_ref_is_null(local->issue_binding.task_ref));
        GmExecution entry{};
        EXPECT_FALSE(f.begin(worker, &entry));
    }
    bool seen[3][2][3]{};
    uint32_t executions = 0;
    for (uint32_t turn = 0; turn < 32; ++turn) {
        for (uint32_t worker : {4u, 5u, 1u, 2u, 3u, 0u}) {
            ASSERT_TRUE(f.fill(worker));
            GmExecution entry{};
            if (!f.begin(worker, &entry)) continue;
            if (entry.task > 0) EXPECT_TRUE(f.task_done(entry.task - 1));
            ASSERT_LT(static_cast<uint64_t>(entry.task), 3u);
            ASSERT_LT(entry.block, 2u);
            ASSERT_LT(entry.subtask, 3u);
            EXPECT_FALSE(seen[entry.task][entry.block][entry.subtask]);
            seen[entry.task][entry.block][entry.subtask] = true;
            ++executions;
            ASSERT_TRUE(f.complete(entry));
        }
    }
    EXPECT_TRUE(f.task_done(2));
    EXPECT_EQ(executions, 14u);
}

TEST(HbgDirectDispatch, SelfClaimSelectsItsOriginalExecutionRecord) {
    GmMixFixture fixture({{2, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    uint32_t selected = UINT32_MAX;
    bool failed = false;
    ASSERT_TRUE(scheduler_find_local_execution(
        fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &selected
    ));
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
    scheduler_find_local_execution(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &selected);
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

TEST(HbgClusterGroupOwnership, TimeoutSnapshotDistinguishesFaninWaitFromKernelEntryAtL4) {
    for (bool l4 : {false, true}) {
        SCOPED_TRACE(l4);
        GmMixFixture fixture({{1, LaneMask::aiv0()}, {1, LaneMask::all()}}, l4, 1);
        fixture.graph.executable(1, 0, {0});
        ASSERT_TRUE(fixture.prepare_calls());
        GmExecution producer{};
        for (unsigned i = 0; i < 8; ++i) {
            ASSERT_TRUE(fixture.fill(1));
            if (fixture.begin(1, &producer)) break;
        }
        ASSERT_EQ(producer.task, 0);
        auto *member = fixture.locals[0];
        uint64_t index = UINT64_MAX;
        ASSERT_TRUE(scheduler_prepared_execution_index(fixture.base(), member->config.prepared, 1, 0, 0, &index));
        auto *cell = scheduler_prepared_control_at(fixture.base(), member->config.prepared, index);
        for (unsigned i = 0; i < 8; ++i) {
            for (unsigned worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.fill(worker));
        }
        ASSERT_EQ(member->mix_groups[0].task_ref, 1);
        ASSERT_EQ(member->mix_groups[0].prepared.execution_index, index);
        GmExecution consumer{};
        ASSERT_FALSE(fixture.begin(0, &consumer));
        EXPECT_STREQ(scheduler_prepared_mix_entry(cell->state, "unknown"), l4 ? "false" : "unknown");
        ASSERT_TRUE(fixture.complete(producer));
        bool entered = false;
        for (unsigned i = 0; i < 8 && !entered; ++i) {
            for (unsigned worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.fill(worker));
            entered = fixture.begin(0, &consumer);
        }
        ASSERT_TRUE(entered);
        ASSERT_EQ(consumer.task, 1);
        EXPECT_STREQ(scheduler_prepared_mix_entry(cell->state, "unknown"), l4 ? "true" : "unknown");
        ASSERT_TRUE(fixture.complete(consumer));
        EXPECT_STREQ(scheduler_prepared_mix_entry(cell->state, "unknown"), "true");
    }
}

TEST(HbgDirectDispatch, MemberCanSelectFreshMixBeforeClaimingAnOrdinaryTicket) {
    GmMixFixture f({{1, LaneMask::all()}, {1, LaneMask::aiv0()}}, false, 1);
    f.storage.metadata[0].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    bool progress = false;
    ASSERT_TRUE(f.fill(1));
    ASSERT_TRUE(scheduler_cluster_mix_armed(f.signal(0)->groups[0].armed_generation));
    ASSERT_EQ(f.signal(0)->groups[0].task_ref, 0);
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        f.graph.graph(), f.base(), f.locals[2], f.run(), 0, &progress, nullptr, &selected
    ));
    EXPECT_EQ(selected, SCHEDULER_CLUSTER_MIX_EXECUTION);
    EXPECT_EQ(scheduler_execution_token_count(f.locals[2]), 0u);
    EXPECT_EQ(f.control(1)->next_block, 0u);
}

TEST(HbgDirectDispatch, FreshMixAfterOrdinaryBindPreservesTheTicketUntilResume) {
    GmMixFixture f({{1, LaneMask::all()}, {1, LaneMask::aiv0()}}, false, 1);
    f.storage.metadata[0].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    ASSERT_TRUE(f.claim_ordinary(1, 2));
    auto *local = f.locals[2];
    const auto handle = local->execution_tokens[0].prepared;
    bool progress = false;
    ASSERT_TRUE(f.fill(1));
    ASSERT_TRUE(scheduler_cluster_mix_armed(f.signal(0)->groups[0].armed_generation));
    ASSERT_EQ(f.signal(0)->groups[0].task_ref, 0);
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(scheduler_pick_new_mix_member(f.graph.graph(), f.base(), local, f.run(), 0, &selected, &progress));
    ASSERT_EQ(selected, SCHEDULER_CLUSTER_MIX_EXECUTION);
    ASSERT_EQ(local->execution_tokens[0].state, SchedulerDispatchSlotState::FILLING);
    EXPECT_EQ(local->execution_tokens[0].prepared.execution_index, handle.execution_index);
    EXPECT_EQ(local->execution_tokens[0].prepared.claimed_state, handle.claimed_state);
    GmExecution execution{};
    ASSERT_TRUE(f.begin(2, selected, &execution));
    ASSERT_EQ(execution.task, 0);
    ASSERT_TRUE(f.complete(execution));
    uint32_t ordinary_returns = 0;
    for (uint32_t turn = 0; turn < 8 && !f.task_done(1); ++turn) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(f.fill(worker));
            if (!f.begin(worker, &execution)) continue;
            if (execution.task == 1) {
                EXPECT_EQ(worker, 2u);
                EXPECT_EQ(execution.block, 0u);
                ++ordinary_returns;
            }
            ASSERT_TRUE(f.complete(execution));
        }
    }
    EXPECT_TRUE(f.task_done(0));
    EXPECT_TRUE(f.task_done(1));
    EXPECT_EQ(ordinary_returns, 1u);
}

TEST(HbgDirectDispatch, ReceivingMixRecognizesAnAlreadyBoundOrdinaryConsumer) {
    GmMixFixture f({{1, LaneMask::aic()}, {1, LaneMask::both_aiv()}, {3, LaneMask::aiv0()}}, false, 1);
    f.graph.executable(1, 1, {0});
    f.graph.executable(2, 1, {1});
    ASSERT_TRUE(f.claim_ordinary(0, 0));
    ASSERT_TRUE(f.progress_ordinary(0));
    GmExecution producer{};
    ASSERT_TRUE(f.begin(0, &producer));
    auto *signal = f.signal(0);
    for (int i = 0; i < 8 && !scheduler_cluster_mix_armed(signal->groups[0].armed_generation); ++i)
        ASSERT_TRUE(f.fill(1));
    ASSERT_TRUE(scheduler_cluster_mix_armed(signal->groups[0].armed_generation));
    ASSERT_TRUE(f.claim_ordinary(2, 2));
    auto *peer = f.locals[2];
    ASSERT_EQ(peer->mix_groups[0].phase, SchedulerGmMixPhase::IDLE);
    ASSERT_EQ(peer->execution_tokens[0].wait.predecessor, 1);
    ASSERT_TRUE(f.fill(2));
    EXPECT_EQ(peer->mix_groups[0].phase, SchedulerGmMixPhase::PUBLISHED);
    EXPECT_TRUE(peer->mix_groups[0].ordinary_consumer_waits);
    EXPECT_EQ(peer->execution_tokens[0].task_ref, 2);
}

TEST(HbgDirectDispatch, OwnedConsumerWaitsForItsMixAndResumesTheSameTicket) {
    GmMixFixture f({{1, LaneMask::aic()}, {1, LaneMask::both_aiv()}, {1, LaneMask::aiv0()}}, false, 1);
    f.graph.executable(1, 1, {0});
    f.graph.executable(2, 1, {1});
    ASSERT_TRUE(f.claim_ordinary(0, 0));
    ASSERT_TRUE(f.progress_ordinary(0));
    GmExecution producer{};
    ASSERT_TRUE(f.begin(0, &producer));

    auto *local = f.locals[1];
    auto &token = local->execution_tokens[0];
    for (int i = 0; i < 8 && token.wait.predecessor != 1; ++i)
        ASSERT_TRUE(f.fill(1));
    ASSERT_EQ(local->mix_groups[0].phase, SchedulerGmMixPhase::PUBLISHED);
    ASSERT_EQ(token.task_ref, 2);
    ASSERT_EQ(token.wait.predecessor, 1);
    EXPECT_TRUE(local->mix_groups[0].ordinary_consumer_waits);
    const auto ticket = token.prepared.execution_index;
    ASSERT_TRUE(f.fill(1));
    GmExecution execution{};
    EXPECT_FALSE(f.begin(1, &execution));
    ASSERT_TRUE(f.complete(producer));

    unsigned mix_returns = 0;
    for (int i = 0; i < 12; ++i) {
        for (uint32_t worker : {1u, 2u}) {
            ASSERT_TRUE(f.fill(worker));
            if (!f.begin(worker, &execution)) continue;
            if (execution.task == 2) {
                EXPECT_EQ(mix_returns, 2u);
                EXPECT_EQ(worker, 1u);
                EXPECT_EQ(token.prepared.execution_index, ticket);
                ASSERT_TRUE(f.complete(execution));
                return;
            }
            ASSERT_EQ(execution.task, 1);
            ASSERT_TRUE(f.complete(execution));
            ++mix_returns;
        }
    }
    FAIL() << "The dependent ordinary ticket did not resume";
}

TEST(HbgDirectDispatch, NewMixWaitingOnTheOrdinarySlotLetsItsPredecessorExecute) {
    GmMixFixture f({{1, LaneMask::aic()}, {1, LaneMask::all()}}, false, 1);
    f.graph.executable(1, 0, {0});
    f.storage.metadata[0].flags |= SCHEDULER_TASK_FIXED_OWNER;
    f.storage.metadata[1].flags |= SCHEDULER_TASK_FIXED_OWNER;
    ASSERT_TRUE(f.prepare_calls());
    bool progress = false;
    ASSERT_TRUE(f.fill(1));
    ASSERT_TRUE(scheduler_cluster_mix_armed(f.signal(0)->groups[0].armed_generation));
    ASSERT_EQ(f.signal(0)->groups[0].task_ref, 1);
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        f.graph.graph(), f.base(), f.locals[0], f.run(), 0, &progress, nullptr, &selected
    ));
    ASSERT_LT(selected, SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(f.locals[0]->mix_groups[0].phase, SchedulerGmMixPhase::PUBLISHED);
    GmExecution execution{};
    ASSERT_TRUE(f.begin(0, selected, &execution));
    ASSERT_EQ(execution.task, 0);
    ASSERT_TRUE(f.complete(execution));
    for (uint32_t turn = 0; turn < 8 && !f.task_done(1); ++turn) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(f.fill(worker));
            if (f.begin(worker, &execution)) {
                ASSERT_EQ(execution.task, 1);
                ASSERT_TRUE(f.complete(execution));
            }
        }
    }
    EXPECT_TRUE(f.task_done(1));
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
    EXPECT_EQ(fixture.storage.completed_gm_records(), 2u);
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
    EXPECT_EQ(fixture.storage.completed_gm_records(), 1u);
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
    for (uint32_t pass = 0; pass < 64 && fixture.storage.completed_gm_records() != 4; ++pass) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution entry{};
            if (!fixture.begin(worker, &entry)) continue;
            ASSERT_TRUE(fixture.complete(entry));
        }
    }
    ASSERT_EQ(fixture.storage.completed_gm_records(), 4u);
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

TEST(HbgDirectDispatch, PureSyncWaitsForItsEarlierOwnerBeforeJoiningACrossClusterBarrier) {
    constexpr int64_t remote = 0, ready = 1, producer = 2, relay = 3, mixed = 4, collective = 5;
    GmMixFixture f(
        {{1, LaneMask::aic()},
         {1, LaneMask::aic()},
         {1, LaneMask::aiv0()},
         {1, LaneMask::aic()},
         {1, kAicAiv0},
         {4, LaneMask::aiv0(), true}},
        false, 2
    );
    f.graph.executable(producer, 1, {remote});
    f.graph.executable(relay, 0, {ready});
    f.graph.executable(mixed, 0, {producer});
    f.graph.executable(collective, 1, {relay});
    ASSERT_TRUE(f.prepare_calls());
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
    // AIV1 in cluster 0 owns the producer. AIV0 in cluster 1 must first execute the dependent MIX.
    ASSERT_TRUE(f.claim_ordinary(producer, 2));
    bool published = false;
    for (unsigned pass = 0; pass < 8 && !published; ++pass) {
        bool progress = false;
        ASSERT_TRUE(scheduler_cluster_mix_discover_next(
            f.graph.graph(), f.base(), f.locals[4], f.run(), &progress, nullptr, &published
        ));
    }
    ASSERT_TRUE(published);
    ASSERT_EQ(f.signal(1)->groups[0].task_ref, mixed);

    std::map<uint32_t, GmExecution> at_barrier;
    auto service = [&](bool remote_returned) {
        for (uint32_t worker : {1U, 0U, 2U, 4U, 3U, 5U}) {
            if ((!remote_returned && worker >= 3) || at_barrier.count(worker)) continue;
            bool failed = false;
            uint32_t selected = UINT32_MAX;
            scheduler_find_local_execution(f.graph.graph(), f.base(), f.locals[worker], f.run(), 0, &failed, &selected);
            ASSERT_FALSE(failed);
            if (selected == UINT32_MAX) continue;
            GmExecution execution{};
            ASSERT_TRUE(f.begin(worker, &execution));
            ASSERT_EQ(execution.slot, selected);
            if (execution.task == collective) at_barrier.emplace(worker, execution);
            else ASSERT_TRUE(f.complete(execution));
        }
        // Hardware synchronization permits a return only after all four AIVs have entered the kernel.
        if (at_barrier.size() == 4) {
            for (const auto &entry : at_barrier)
                ASSERT_TRUE(f.complete(entry.second));
            at_barrier.clear();
        }
    };
    constexpr bool spare_slot = SCHEDULER_EXECUTION_TOKEN_COUNT > 1;
    const auto collective_claimed = [&] {
        return scheduler_execution_token_has_task(f.locals[2], collective);
    };
    for (unsigned pass = 0; pass < 12 && (!at_barrier.count(1) || collective_claimed() != spare_slot); ++pass)
        service(false);
    EXPECT_TRUE(scheduler_execution_token_has_task(f.locals[2], producer));
    EXPECT_EQ(collective_claimed(), spare_slot);
    EXPECT_TRUE(at_barrier.count(1));
    EXPECT_FALSE(at_barrier.count(2));

    ASSERT_TRUE(f.complete(running));
    for (unsigned pass = 0; pass < 20 && !f.task_done(collective); ++pass)
        service(true);
    EXPECT_TRUE(f.task_done(producer));
    EXPECT_TRUE(f.task_done(mixed));
    EXPECT_TRUE(f.task_done(collective));
    EXPECT_TRUE(at_barrier.empty());
    EXPECT_EQ(scheduler_read_error(f.run()), 0u);
}

TEST(HbgDirectDispatch, PureSyncAdmissionDependsOnTheSpaceBesideAnEarlierOwner) {
    GmMixFixture f({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}, {4, LaneMask::aiv0(), true}}, false, 2);
    f.graph.executable(1, 1, {0});
    GmExecution outstanding{};
    ASSERT_TRUE(f.claim_ordinary(0, 0));
    ASSERT_TRUE(f.progress_ordinary(0));
    ASSERT_TRUE(f.begin(0, &outstanding));
    ASSERT_TRUE(f.claim_ordinary(1, 2));
    std::set<uint32_t> arrived;
    std::vector<GmExecution> held;
    const auto join = [&](uint32_t worker) {
        if (arrived.count(worker)) return;
        bool failed = false;
        uint32_t selected = UINT32_MAX;
        scheduler_find_local_execution(f.graph.graph(), f.base(), f.locals[worker], f.run(), 0, &failed, &selected);
        ASSERT_FALSE(failed);
        if (selected == UINT32_MAX) return;
        GmExecution unit{};
        ASSERT_TRUE(f.begin(worker, selected, &unit));
        ASSERT_EQ(unit.slot, selected);
        ASSERT_EQ(unit.task, 2);
        arrived.insert(worker);
        held.push_back(unit);
    };
    const auto complete_collective = [&] {
        ASSERT_EQ(arrived.size(), 4u);
        EXPECT_EQ(f.control(2)->claim_seal, 4u);
        for (const auto &unit : held)
            ASSERT_TRUE(f.complete(unit));
        EXPECT_TRUE(f.task_done(2));
    };
    constexpr bool spare_slot = SCHEDULER_EXECUTION_TOKEN_COUNT > 1;
    constexpr uint32_t initial_arrivals = spare_slot ? 4 : 3;
    for (unsigned pass = 0; pass < 16 && arrived.size() < initial_arrivals; ++pass)
        for (uint32_t worker : {2U, 1U, 4U, 5U})
            ASSERT_NO_FATAL_FAILURE(join(worker));
    ASSERT_EQ(arrived.size(), initial_arrivals);
    EXPECT_EQ(arrived.count(2) != 0, spare_slot);
    EXPECT_TRUE(scheduler_execution_token_has_task(f.locals[2], 1));
    EXPECT_FALSE(f.task_done(0));
    if constexpr (spare_slot) ASSERT_NO_FATAL_FAILURE(complete_collective());

    ASSERT_TRUE(f.complete(outstanding));
    ASSERT_TRUE(f.progress_ordinary(2));
    GmExecution dependent{};
    ASSERT_TRUE(f.begin(2, &dependent));
    ASSERT_EQ(dependent.task, 1);
    ASSERT_TRUE(f.complete(dependent));
    if constexpr (!spare_slot) {
        for (unsigned pass = 0; pass < 16 && !arrived.count(2); ++pass)
            ASSERT_NO_FATAL_FAILURE(join(2));
        ASSERT_NO_FATAL_FAILURE(complete_collective());
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
                const auto result = scheduler_publish_fanout<false>(
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
                (scheduler_publish_fanout<false>(fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0)),
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
        ASSERT_TRUE(scheduler_claim_for_execute(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &armed
        ));
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
        EXPECT_EQ(scheduler_cluster_mix_armed(fixture.signal(0)->groups[0].armed_generation), block == 2);
        EXPECT_FALSE(fixture.task_done(0));
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        EXPECT_EQ(fixture.storage.completed_gm_records(), 0u);
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
    EXPECT_FALSE(scheduler_cluster_mix_armed(fixture.signal(0)->groups[0].armed_generation));
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.storage.completed_gm_records(), 1u);
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
            ASSERT_TRUE(scheduler_claim_for_execute(
                fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &armed
            ));
            ASSERT_TRUE(armed);
            for (uint32_t pass = 0; pass < 3; ++pass)
                for (uint32_t lane = 0; lane < 3; ++lane)
                    ASSERT_TRUE(fixture.service_cooperative(lane));
            const uint32_t lane = scheduler_mix_accountant_lane(mask);
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(lane, &execution));
            auto *local = fixture.locals[lane];
            const auto &owner = local->execution_at(execution.slot);
            const auto *cell =
                scheduler_prepared_control_at(fixture.base(), local->config.prepared, owner.prepared.execution_index);
            EXPECT_FALSE(scheduler_release_mix_lane(fixture.graph.graph(), fixture.base(), local, fixture.run()));
            EXPECT_EQ(fixture.signals[0].groups[0].finished[lane], 0u);
            EXPECT_FALSE(fixture.task_done(0));
            EXPECT_NE(fixture.run()->scheduler_error, 0u);
            ASSERT_TRUE((scheduler_complete_lane_execution<false, false>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 0
            )));
            EXPECT_EQ(
                cell->state,
                scheduler_prepared_state(SchedulerPreparedPhase::RETURNED, lane) | SCHEDULER_PREPARED_MIX_OWNED
            );
            EXPECT_EQ(fixture.signals[0].groups[0].finished[lane], fixture.signals[0].groups[0].armed_generation + 1);
            EXPECT_NE(local->mix_groups[0].settled_generation, fixture.signals[0].groups[0].armed_generation);
            EXPECT_FALSE(fixture.complete(execution));
            EXPECT_EQ(fixture.signals[0].groups[0].accounted_generation, 0u);
            EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        }
    }
}
TEST(HbgClusterGroupReturn, RunErrorSuppressesTheLiveAccountantsCredit) {
    GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool published = false;
    ASSERT_TRUE(scheduler_claim_for_execute(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &published
    ));
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
    EXPECT_EQ(fixture.signal(0)->groups[0].accounted_generation, 0u);
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
    ASSERT_TRUE(scheduler_claim_for_execute(fixture.graph.graph(), fixture.base(), proxy, fixture.run(), 0, &advanced));
    ASSERT_TRUE(advanced);
    for (uint32_t task = 1; task < shapes.size(); ++task)
        ASSERT_TRUE(fixture.claim_ordinary(task, 2));
    ASSERT_EQ(scheduler_execution_token_count(member), SCHEDULER_EXECUTION_TOKEN_COUNT);
    std::array<SchedulerPreparedHandle, SCHEDULER_EXECUTION_TOKEN_COUNT> handles;
    for (uint32_t id = 0; id < handles.size(); ++id)
        handles[id] = member->execution_at(id).prepared;
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
    EXPECT_EQ(fixture.storage.completed_gm_records(), shapes.size());
}

TEST(HbgClusterDescriptor, RejectsMalformedGroupIdentityOrUnsupportedPredicate) {
    for (uint32_t fault = 0; fault < 5; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
        ASSERT_TRUE(fixture.prepare_calls());
        bool published = false;
        ASSERT_TRUE(scheduler_claim_for_execute(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &published
        ));
        ASSERT_TRUE(published);
        auto *signal = fixture.signal(0);
        switch (fault) {
        case 0:
            ++signal->run_epoch;
            break;
        case 1:
            ++signal->groups[0].rank;
            break;
        case 2:
            ++signal->groups[0].block_idx;
            break;
        case 3:
            signal->groups[0].active_mask = kAicAiv1.raw();
            break;
        case 4:
            fixture.storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
            break;
        }
        bool progress = false;
        EXPECT_FALSE(scheduler_cluster_mix_service(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress
        ));
        EXPECT_EQ(signal->groups[0].arrived[0], 0u);
        EXPECT_EQ(signal->groups[0].accounted_generation, 0u);
        EXPECT_EQ(
            fixture.locals[0]->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).state, SchedulerDispatchSlotState::FREE
        );
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR));
    }
}

}  // namespace
