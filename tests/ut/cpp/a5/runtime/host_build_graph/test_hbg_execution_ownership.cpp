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
#include <condition_variable>
#include <mutex>
#include <set>
#include <thread>

#include "hbg_scheduler_test_support.h"
#include "prepared_execution_plan.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_execution_tokens.h"
#include "scheduler/scheduler_prepared_execution.h"

namespace {

struct OwnershipFixture {
    alignas(128) std::array<uint8_t, 1024> bytes{};
    alignas(128) volatile uint64_t parked_count{0};
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
    PreparedAccessResult resume(uint32_t worker, uint64_t observed, SchedulerPreparedHandle *handle) {
        return scheduler_resume_parked_execution(base(), prepared, worker, 0, 0, 0, observed, handle, parked_count);
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

TEST(HbgExecutionOwnership, ParkingIsNotClaimableUntilTheOwnerPublishesIt) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{}, receiver{};
    ASSERT_EQ(fixture.claim(0, &owner), PreparedAccessResult::READY);
    EXPECT_FALSE(scheduler_begin_park_prepared_execution(fixture.base(), fixture.prepared, 1, &owner));
    ASSERT_TRUE(scheduler_begin_park_prepared_execution(fixture.base(), fixture.prepared, 0, &owner));
    const uint64_t parking = fixture.control()->state;
    EXPECT_EQ(scheduler_prepared_phase(parking), SchedulerPreparedPhase::PARKING);
    EXPECT_EQ(fixture.parked_count, 0u);
    EXPECT_EQ(fixture.resume(1, parking, &receiver), PreparedAccessResult::EXHAUSTED);
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &owner));
    ASSERT_TRUE(scheduler_publish_parked_execution(fixture.base(), fixture.prepared, 0, &owner, fixture.parked_count));
    EXPECT_EQ(owner.claimed_state, 0u);
    EXPECT_EQ(fixture.parked_count, 1u);
    const uint64_t parked = fixture.control()->state;
    ASSERT_EQ(fixture.resume(1, parked, &receiver), PreparedAccessResult::READY);
    EXPECT_EQ(fixture.parked_count, 0u);
    EXPECT_EQ(scheduler_prepared_generation(receiver.claimed_state), 1u);
    EXPECT_TRUE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &receiver));
}

TEST(HbgExecutionOwnership, CompetingReceiversHaveOneWinnerAndOneCountContribution) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{};
    ASSERT_EQ(fixture.claim(0, &owner), PreparedAccessResult::READY);
    ASSERT_TRUE(scheduler_park_prepared_execution(fixture.base(), fixture.prepared, 0, &owner, fixture.parked_count));
    const uint64_t parked = fixture.control()->state;
    std::array<SchedulerPreparedHandle, 2> handles{};
    std::array<PreparedAccessResult, 2> results{};
    std::mutex mutex;
    std::condition_variable event;
    uint32_t arrived = 0;
    bool start = false;
    auto compete = [&](uint32_t index) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++arrived;
            event.notify_all();
            event.wait(lock, [&] {
                return start;
            });
        }
        results[index] = fixture.resume(index + 1, parked, &handles[index]);
    };
    std::thread first(compete, 0), second(compete, 1);
    {
        std::unique_lock<std::mutex> lock(mutex);
        event.wait(lock, [&] {
            return arrived == 2;
        });
        start = true;
        event.notify_all();
    }
    first.join();
    second.join();
    const uint32_t winner = results[0] == PreparedAccessResult::READY ? 0 : 1;
    EXPECT_EQ(results[winner], PreparedAccessResult::READY);
    EXPECT_EQ(results[1 - winner], PreparedAccessResult::EXHAUSTED);
    EXPECT_EQ(handles[1 - winner].claimed_state, 0u);
    EXPECT_EQ(fixture.parked_count, 0u);
    EXPECT_EQ(fixture.control()->state, handles[winner].claimed_state);
}

TEST(HbgExecutionOwnership, SameWorkerReacquireRejectsThePreviousGeneration) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{};
    ASSERT_EQ(fixture.claim(0, &owner), PreparedAccessResult::READY);
    SchedulerPreparedHandle stale = owner;
    ASSERT_TRUE(scheduler_park_prepared_execution(fixture.base(), fixture.prepared, 0, &owner, fixture.parked_count));
    ASSERT_EQ(fixture.resume(0, fixture.control()->state, &owner), PreparedAccessResult::READY);
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &stale));
    EXPECT_FALSE(scheduler_park_prepared_execution(fixture.base(), fixture.prepared, 0, &stale, fixture.parked_count));
    EXPECT_EQ(fixture.control()->state, owner.claimed_state);
    EXPECT_EQ(fixture.parked_count, 0u);
    EXPECT_TRUE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &owner));
}

TEST(HbgExecutionOwnership, GenerationExhaustionCannotWrapIntoAnOldOwner) {
    OwnershipFixture fixture;
    const uint64_t exhausted = scheduler_prepared_state(SchedulerPreparedPhase::PARKED, UINT32_MAX, UINT32_MAX);
    fixture.control()->state = exhausted;
    fixture.parked_count = 1;
    SchedulerPreparedHandle owner{};
    EXPECT_EQ(fixture.resume(0, exhausted, &owner), PreparedAccessResult::ERROR);
    EXPECT_EQ(fixture.control()->state, exhausted);
    EXPECT_EQ(fixture.parked_count, 1u);
    EXPECT_EQ(owner.claimed_state, 0u);
}

TEST(HbgExecutionOwnership, InvalidCallFailsClosedAfterClaim) {
    OwnershipFixture fixture;
    scheduler_prepared_call_at(fixture.base(), fixture.prepared, 0)->run_epoch = 18;
    SchedulerPreparedHandle owner{};
    EXPECT_EQ(fixture.claim(0, &owner), PreparedAccessResult::ERROR);
    EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &owner));
    EXPECT_FALSE(scheduler_park_prepared_execution(fixture.base(), fixture.prepared, 0, &owner, fixture.parked_count));
    EXPECT_EQ(fixture.parked_count, 0u);
}

TEST(HbgExecutionOwnership, OwnerCancellationDoesNotReturnOrPublishTheExecution) {
    for (bool parking : {false, true}) {
        OwnershipFixture fixture;
        SchedulerPreparedHandle owner{};
        ASSERT_EQ(fixture.claim(0, &owner), PreparedAccessResult::READY);
        if (parking) ASSERT_TRUE(scheduler_begin_park_prepared_execution(fixture.base(), fixture.prepared, 0, &owner));
        SchedulerPreparedHandle stale = owner;
        EXPECT_FALSE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 1, &owner));
        ASSERT_TRUE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 0, &owner));
        EXPECT_EQ(owner.claimed_state, 0u);
        EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
        EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &stale));
        if (parking)
            EXPECT_FALSE(
                scheduler_publish_parked_execution(fixture.base(), fixture.prepared, 0, &stale, fixture.parked_count)
            );
        EXPECT_EQ(fixture.parked_count, 0u);
    }
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
        local->execution_scan_cursor = 0;
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
    // Discovery, replacement and subsequent entry remain production operations.
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

TEST(HbgExecutionRefill, WideLookaheadPreservesFirstWaveAndFourOwnedUnits) {
    RefillFixture fixture(6);
    fixture.storage.metadata[1].logical_block_num = 6;
    fixture.graph.executable(1, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    ASSERT_EQ(fixture.local->config.cluster_count, 2u);
    std::array<SchedulerPreparedHandle, 6> outside{};
    for (uint32_t task : {0u, 2u, 3u, 4u, 5u})
        ASSERT_NO_FATAL_FAILURE(fixture.reserve(task, 3, &outside[task]));
    ASSERT_NO_FATAL_FAILURE(fixture.own(1));
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    EXPECT_FALSE(advanced);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 1u);

    ASSERT_NO_FATAL_FAILURE(fixture.reserve(1, 3, &outside[1]));
    for (uint32_t expected = 2; expected <= SCHEDULER_EXECUTION_TOKEN_COUNT; ++expected) {
        ASSERT_TRUE(fixture.refill(&advanced));
        ASSERT_EQ(scheduler_execution_token_count(fixture.local), expected);
    }
    EXPECT_EQ(fixture.control(1)->next_block, 5u);
    ASSERT_TRUE(fixture.refill(&advanced));
    EXPECT_EQ(fixture.control(1)->next_block, 5u);
    EXPECT_EQ(fixture.control(1)->parked_count, 0u);
    std::set<uint32_t> blocks;
    for (const auto &token : fixture.local->execution_tokens)
        blocks.insert(token.block_idx);
    EXPECT_EQ(blocks, (std::set<uint32_t>{0, 2, 3, 4}));

    auto *peer = fixture.storage.local_context(&fixture.storage.contexts[3]);
    auto &producer = peer->slots[0];
    producer.prepared = outside[0];
    producer.task_ref = 0;
    producer.subtask_slot = 0;
    producer.block_idx = 0;
    producer.execution_token = true;
    producer.predicate = SchedulerPredicateState::PASS;
    producer.state = SchedulerDispatchSlotState::READY;
    ASSERT_NO_FATAL_FAILURE(fixture.execute(0, peer));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
    for (const auto &token : fixture.local->execution_tokens)
        if (token.phase != SchedulerExecutionTokenPhase::EMPTY) EXPECT_EQ(token.next_fanin, token.fanin.count);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 1);
    EXPECT_EQ(fixture.control(1)->next_block, 5u);
}

TEST(HbgExecutionRefill, SameTaskWindowYieldsToAnIndependentNotifier) {
    RefillFixture fixture(6);
    fixture.storage.metadata[1].logical_block_num = 6;
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerPreparedHandle peer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(1, 3, &peer));
    for (uint32_t unit = 0; unit < SCHEDULER_EXECUTION_TOKEN_COUNT; ++unit)
        ASSERT_NO_FATAL_FAILURE(fixture.own(1));
    fixture.local->execution_scan_cursor = 0;
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    EXPECT_TRUE(advanced);
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{0, 1}));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(fixture.control(1)->next_block, 5u);
    EXPECT_EQ(fixture.control(1)->parked_count, 1u);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    ASSERT_EQ(fixture.local->slots[slot].task_ref, 0);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 1);
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
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
    fixture.local->execution_scan_cursor = 1;

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

TEST(HbgExecutionRefill, ExistingReadyOwnershipPrecedesContinuation) {
    RefillFixture fixture(3);
    fixture.storage.metadata[0].logical_block_num = 3;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    ASSERT_NO_FATAL_FAILURE(fixture.own(0));
    ASSERT_NO_FATAL_FAILURE(fixture.own(1));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    ASSERT_EQ(fixture.local->slots[slot].task_ref, 0);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    bool failed = false;
    slot = UINT32_MAX;
    scheduler_fill_own_slots<false>(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 1);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
}

TEST(HbgExecutionRefill, ContinuationYieldsToDirectoryNotifierBeforeTaskExhaustion) {
    RefillFixture fixture(12);
    fixture.storage.metadata[0].logical_block_num = 12;
    for (uint32_t task = 1; task < 12; ++task)
        if (task != 10) fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    ASSERT_NO_FATAL_FAILURE(fixture.own(0));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    fixture.local->execution_scan_cursor = 1;
    bool notified = false;
    for (uint32_t slice = 0; slice < 6 && !notified; ++slice) {
        bool failed = false;
        slot = UINT32_MAX;
        scheduler_fill_own_slots<false>(
            fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
        );
        ASSERT_FALSE(failed);
        if (slot == UINT32_MAX) continue;
        ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
        const int64_t task = fixture.local->slots[slot].task_ref;
        ASSERT_TRUE(task == 0 || task == 10);
        notified = task == 10;
        ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    }
    EXPECT_TRUE(notified);
    EXPECT_LT(fixture.control(0)->next_block, fixture.storage.metadata[0].logical_block_num);
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
        EXPECT_EQ(fixture.control(task)->parked_count, 0u);
    }
    EXPECT_EQ(fixture.control(4)->next_block, 0u);
    EXPECT_EQ(fixture.control(5)->next_block, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
}

TEST(HbgExecutionRefill, OneSearchFillsDistinctPendingTasksWithinFourIncludingCurrent) {
    RefillFixture fixture(6);
    fixture.storage.metadata[0].logical_block_num = 2;
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    ASSERT_NO_FATAL_FAILURE(fixture.own(0));
    uint32_t current = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&current));
    ASSERT_LT(current, SCHEDULER_PENDING_SLOT_COUNT);
    fixture.local->slots[current].state = SchedulerDispatchSlotState::EXECUTING;
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{1, 2, 3}));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(4)->next_block, 0u);
    EXPECT_EQ(fixture.control(5)->next_block, 0u);
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{1, 2, 3}));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
    EXPECT_EQ(fixture.local->slots[current].state, SchedulerDispatchSlotState::EXECUTING);
}

TEST(HbgExecutionOwnership, MandatoryParkedSweepCoversEveryCellRegardlessOfHints) {
    for (uint32_t blocks : {1u, 2u, 5u, 32u, 64u}) {
        for (uint32_t worker : {0u, 31u, 81u}) {
            for (uint32_t hints = 0; hints < 4; ++hints) {
                std::set<uint32_t> mandatory_cells;
                for (uint64_t round = 0; round < blocks * SCHEDULER_PARKED_SWEEP_INTERVAL; ++round) {
                    const bool mandatory =
                        round % SCHEDULER_PARKED_SWEEP_INTERVAL == SCHEDULER_PARKED_SWEEP_INTERVAL - 1;
                    const bool positive = hints == 1 || (hints == 2 && mandatory) || (hints == 3 && !mandatory);
                    const auto plan = scheduler_parked_probe_plan(worker, round, blocks, positive);
                    ASSERT_LE(plan.count, 2u);
                    for (uint32_t probe = 0; probe < plan.count; ++probe)
                        EXPECT_LT(plan.blocks[probe], blocks);
                    if (plan.count == 2) EXPECT_NE(plan.blocks[0], plan.blocks[1]);
                    if (mandatory) {
                        ASSERT_GT(plan.count, 0u);
                        mandatory_cells.insert(plan.blocks[0]);
                    }
                }
                EXPECT_EQ(mandatory_cells.size(), blocks);
            }
        }
    }
}

TEST(HbgExecutionRefill, PositiveParkedHintDoesNotRepeatTheSameBusyCell) {
    RefillFixture fixture(1);
    fixture.storage.metadata[0].logical_block_num = 2;
    fixture.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    std::array<SchedulerPreparedHandle, 2> owners{};
    for (auto &owner : owners)
        ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &owner));
    ASSERT_TRUE(scheduler_park_prepared_execution(
        fixture.base(), fixture.local->config.prepared, 3, &owners[1], fixture.control(0)->parked_count
    ));
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 0u);
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_TRUE(progress);
    ASSERT_EQ(scheduler_execution_token_count(fixture.local), 1u);
    EXPECT_EQ(fixture.local->execution_tokens[0].block_idx, 1u);
    EXPECT_EQ(scheduler_prepared_generation(fixture.local->execution_tokens[0].prepared.claimed_state), 1u);
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_EQ(fixture.control(0)->parked_count, 0u);
}

TEST(HbgExecutionRefill, FullWaitingWindowDoesNotParkWithoutAReadyReplacement) {
    RefillFixture fixture(6);
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerPreparedHandle producer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &producer));
    std::array<SchedulerPreparedHandle, 4> original{};
    for (uint32_t task = 1; task <= 4; ++task) {
        ASSERT_NO_FATAL_FAILURE(fixture.own(task));
        original[task - 1] = fixture.local->execution_tokens[task - 1].prepared;
    }
    fixture.local->execution_scan_cursor = 5;
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(fixture.control(5)->next_block, 0u);
    for (uint32_t task = 1; task <= 4; ++task) {
        EXPECT_EQ(fixture.control(task)->parked_count, 0u);
        EXPECT_EQ(
            scheduler_prepared_control_at(
                fixture.base(), fixture.local->config.prepared, original[task - 1].execution_index
            )
                ->state,
            original[task - 1].claimed_state
        );
    }
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
}

TEST(HbgExecutionRefill, HiddenReadyNotifierReplacesOneWaiterAndUnlocksTheWindow) {
    RefillFixture fixture(5);
    for (uint32_t task = 1; task < 5; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    for (uint32_t task = 1; task <= 4; ++task)
        ASSERT_NO_FATAL_FAILURE(fixture.own(task));
    fixture.local->execution_scan_cursor = 1;
    bool progress = false;
    ASSERT_TRUE(fixture.refill(&progress));
    EXPECT_TRUE(progress);
    ASSERT_EQ(fixture.pending_tasks().count(0), 1u);
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
    uint64_t parked = 0;
    for (uint32_t task = 1; task <= 4; ++task) {
        parked += fixture.control(task)->parked_count;
        EXPECT_EQ(fixture.control(task)->next_block, 1u);
    }
    EXPECT_EQ(parked, 1u);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    ASSERT_EQ(fixture.local->slots[slot].task_ref, 0);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
    slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_GT(fixture.local->slots[slot].task_ref, 0);
    EXPECT_EQ(scheduler_read_error(fixture.storage.run_control), 0u);
}

}  // namespace
