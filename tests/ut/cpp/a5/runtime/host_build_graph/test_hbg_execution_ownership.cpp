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

#include <array>
#include <set>

#include "hbg_scheduler_test_support.h"
#include "prepared_execution_plan.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_execution_tokens.h"
#include "scheduler/scheduler_prepared_execution.h"

namespace {

struct OwnershipFixture {
    alignas(128) std::array<uint8_t, 1024> bytes{};
    SchedulerPreparedLayout *prepared;

    OwnershipFixture() {
        AicoreSchedulerLayout layout{};
        layout.total_size = bytes.size();
        layout.task_count = 1;
        layout.prepared_layout_offset = 64;
        prepared = scheduler_state_at<SchedulerPreparedLayout>(base(), layout.prepared_layout_offset);
        *prepared = {17, 1, 1, 128, 256, 512, 384, SCHEDULER_PREPARED_VERSION};
        *scheduler_state_at<SchedulerExecutionPlan>(base(), prepared->plans_offset) = {0, 1, 1, 1, 1};
        const uint64_t functions[3]{1, 0, 0};
        EXPECT_TRUE(scheduler_prepare_host_calls(
            base(), layout, reinterpret_cast<uint64_t>(base()), 0, functions, 0, nullptr, 0, 0
        ));
    }

    void *base() { return bytes.data(); }
    SchedulerPreparedControl *control() { return scheduler_prepared_control_at(base(), prepared, 0); }
    PreparedAccessResult claim(uint32_t worker, SchedulerPreparedHandle *handle) {
        return scheduler_acquire_execution_token(base(), prepared, worker, 0, 0, 0, handle);
    }
};

TEST(HbgExecutionOwnership, FreshTicketCannotBeConsumedTwiceAndBindingIsReceived) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle first{}, duplicate{};
    ASSERT_EQ(fixture.claim(0, &first), PreparedAccessResult::READY);
    EXPECT_NE(first.claimed_state & SCHEDULER_PREPARED_BINDING_READY, 0u);
    EXPECT_EQ(scheduler_prepared_generation(first.claimed_state), 0u);
    EXPECT_EQ(fixture.claim(1, &duplicate), PreparedAccessResult::ERROR);
    EXPECT_EQ(duplicate.claimed_state, 0u);
    EXPECT_EQ(fixture.control()->state, first.claimed_state);
    EXPECT_TRUE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &first));
    EXPECT_EQ(fixture.claim(0, &duplicate), PreparedAccessResult::ERROR);
}

TEST(HbgExecutionOwnership, InvalidCallFailsClosedAfterClaim) {
    OwnershipFixture fixture;
    scheduler_prepared_call_at(fixture.base(), fixture.prepared, 0)->run_epoch = 18;
    SchedulerPreparedHandle owner{};
    EXPECT_EQ(fixture.claim(0, &owner), PreparedAccessResult::ERROR);
    EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &owner));
}

TEST(HbgExecutionOwnership, OnlyTheFixedOwnerCanCancelAnUnstartedExecution) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{}, duplicate{};
    ASSERT_EQ(fixture.claim(0, &owner), PreparedAccessResult::READY);
    const auto stale = owner;
    EXPECT_FALSE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 1, &owner));
    EXPECT_EQ(fixture.control()->state, stale.claimed_state);
    ASSERT_TRUE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 0, &owner));
    EXPECT_EQ(owner.claimed_state, 0u);
    EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
    auto returned = stale;
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &returned));
    EXPECT_EQ(fixture.claim(1, &duplicate), PreparedAccessResult::ERROR);
}

struct RefillFixture {
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<16> graph;
    SchedulerLocalState *local{nullptr};

    explicit RefillFixture(uint32_t tasks) :
        storage(tasks, 6, 2),
        graph(tasks) {
        for (uint32_t task = 0; task < tasks; ++task)
            graph.executable(task, 0);
        for (uint32_t worker = 0; worker < 6; ++worker)
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage.contexts[worker].cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&scheduler_test::protocol_noop_kernel);
    }

    void initialize() {
        storage.prepare_catalog();
        ASSERT_TRUE(storage.prepare_calls(graph.graph()));
        local = storage.local_context(&storage.contexts[0]);
    }
    void *base() { return storage.scheduler_state->base(); }
    SchedulerTaskControl *control(uint32_t task) {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    void reserve(uint32_t task, uint32_t worker, SchedulerPreparedHandle *handle) {
        const uint64_t block = scheduler_gm_fetch_add(control(task)->next_block, UINT64_C(1));
        ASSERT_LT(block, storage.metadata[task].logical_block_num);
        ASSERT_EQ(
            scheduler_acquire_execution_token(base(), local->config.prepared, worker, task, block, 0, handle),
            PreparedAccessResult::READY
        );
    }
    // Seed a known ownership boundary using the actual ticket and Call claim.
    // Ordered issuance and subsequent entry remain production operations.
    void own(uint32_t task) {
        auto entry =
            std::find_if(std::begin(local->execution_tokens), std::end(local->execution_tokens), [](auto &token) {
                return token.phase == SchedulerExecutionTokenPhase::EMPTY;
            });
        ASSERT_NE(entry, std::end(local->execution_tokens));
        ASSERT_NO_FATAL_FAILURE(reserve(task, 0, &entry->prepared));
        entry->task_ref = task;
        entry->block_idx = static_cast<uint32_t>(control(task)->next_block - 1);
        entry->subtask_slot = 0;
        entry->phase = SchedulerExecutionTokenPhase::WAIT_FANIN;
        ASSERT_EQ(scheduler_node_fanin_view(graph.graph(), task, &entry->fanin), SchedulerGraphResult::OK);
    }
    std::set<int64_t> pending_tasks() const {
        std::set<int64_t> tasks;
        for (const auto &token : local->execution_tokens)
            if (token.phase != SchedulerExecutionTokenPhase::EMPTY) tasks.insert(token.task_ref);
        return tasks;
    }
    bool refill(bool *progress) {
        return scheduler_refill_execution_tokens(graph.graph(), base(), local, storage.run_control, 0, progress);
    }
    bool progress(uint32_t *slot) {
        bool advanced = false;
        return scheduler_progress_execution_tokens(
            graph.graph(), base(), local, storage.run_control, 0, slot, &advanced
        );
    }
    void execute(uint32_t index, SchedulerLocalState *executor = nullptr) {
        if (executor == nullptr) executor = local;
        ASSERT_LT(index, SCHEDULER_PENDING_SLOT_COUNT);
        scheduler_local_ready_publish(executor, index);
        ASSERT_TRUE(scheduler_local_ready_execute(executor, index));
        auto *call = scheduler_prepared_call_at(
            base(), executor->config.prepared, executor->slots[index].prepared.execution_index
        );
        auto kernel = reinterpret_cast<void (*)(__gm__ int64_t *)>(call->function_bin_addr);
        kernel(reinterpret_cast<__gm__ int64_t *>(call->args));
        ASSERT_TRUE(scheduler_complete_own_slot<false>(graph.graph(), base(), executor, storage.run_control, index, 0));
    }
};

TEST(HbgExecutionRefill, AWorkerCannotSkipTheUnissuedRemainderOfItsCurrentTask) {
    RefillFixture fixture(4);
    fixture.storage.metadata[0].logical_block_num = 2;
    fixture.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
    for (uint32_t task = 1; task < 4; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    ASSERT_EQ(fixture.pending_tasks(), (std::set<int64_t>{0}));
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    SchedulerPreparedHandle peer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &peer));
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 4u);
}

TEST(HbgExecutionRefill, FullWaitingWindowKeepsEveryOwnerUntilTheProducerCompletes) {
    RefillFixture fixture(6);
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerPreparedHandle producer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &producer));
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    ASSERT_EQ(fixture.pending_tasks(), (std::set<int64_t>{1, 2, 3, 4}));
    std::array<SchedulerPreparedHandle, 4> held{};
    for (uint32_t index = 0; index < 4; ++index)
        held[index] = fixture.local->execution_tokens[index].prepared;
    ASSERT_TRUE(fixture.refill(&advanced));
    EXPECT_FALSE(advanced);
    EXPECT_EQ(fixture.control(5)->next_block, 0u);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    EXPECT_EQ(slot, UINT32_MAX);
    for (uint32_t index = 0; index < 4; ++index) {
        const auto &current = fixture.local->execution_tokens[index].prepared;
        EXPECT_EQ(current.execution_index, held[index].execution_index);
        EXPECT_EQ(current.claimed_state, held[index].claimed_state);
        EXPECT_EQ(scheduler_prepared_owner(current.claimed_state), 0u);
    }
    auto *peer = fixture.storage.local_context(&fixture.storage.contexts[3]);
    auto &pending = peer->slots[0];
    pending.prepared = producer;
    pending.task_ref = 0;
    pending.subtask_slot = 0;
    pending.block_idx = 0;
    pending.execution_token = true;
    pending.predicate = SchedulerPredicateState::PASS;
    pending.state = SchedulerDispatchSlotState::READY;
    ASSERT_NO_FATAL_FAILURE(fixture.execute(0, peer));
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(scheduler_prepared_owner(fixture.local->slots[slot].prepared.claimed_state), 0u);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 3u);
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
}

TEST(HbgExecutionRefill, UnissuedProducerIsClaimedBeforeAnyOfItsConsumers) {
    RefillFixture fixture(6);
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{0}));
    for (uint32_t task = 1; task < 6; ++task)
        EXPECT_EQ(fixture.control(task)->next_block, 0u);
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{0, 1, 2, 3}));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 0);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
}

TEST(HbgExecutionRefill, ReturnedSpmdContinuesBeforeCollectingUnreadyTasks) {
    RefillFixture fixture(12);
    fixture.storage.metadata[0].logical_block_num = 6;
    for (uint32_t task = 1; task < 12; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    ASSERT_NO_FATAL_FAILURE(fixture.own(0));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));

    bool failed = false;
    slot = UINT32_MAX;
    scheduler_fill_own_slots<false>(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 0);
    EXPECT_EQ(fixture.local->slots[slot].block_idx, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
}

TEST(HbgExecutionRefill, SkipsExhaustedEntriesAndReturnsAsSoonAsAReadyTicketIsFound) {
    RefillFixture fixture(6);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    std::array<SchedulerPreparedHandle, 3> other_core{};
    for (uint32_t task = 0; task < other_core.size(); ++task)
        ASSERT_NO_FATAL_FAILURE(fixture.reserve(task, 3, &other_core[task]));
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{3}));
    for (uint32_t task = 0; task < 4; ++task) {
        EXPECT_EQ(fixture.control(task)->next_block, 1u);
    }
    EXPECT_EQ(fixture.control(4)->next_block, 0u);
    EXPECT_EQ(fixture.control(5)->next_block, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
}

}  // namespace
