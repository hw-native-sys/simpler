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
#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <set>
#include <utility>

#include "scheduler/scheduler_memory.h"

namespace {
thread_local std::function<void(void *)> receive_observer;
thread_local std::function<void(void *)> writeback_observer;
thread_local std::function<void(volatile uint64_t *)> prepared_cas_observer;
thread_local const volatile void *query_target = nullptr;
thread_local uint32_t query_count = 0;
thread_local const volatile uint32_t *notify_target = nullptr;
thread_local uint32_t notify_count = 0;

template <typename Word>
auto observe_query(volatile Word &word, int order = __ATOMIC_ACQUIRE) {
    if (&word == query_target) ++query_count;
    return scheduler_gm_query(word, order);
}

uint32_t observe_notify_query(volatile uint32_t &word) {
    if (&word == notify_target) ++notify_count;
    return scheduler_notify_query(word);
}

class ScopedQueryObserver {
public:
    explicit ScopedQueryObserver(const volatile void *address) {
        query_target = address;
        query_count = 0;
    }
    ~ScopedQueryObserver() { query_target = nullptr; }
};

class ScopedNotifyObserver {
public:
    explicit ScopedNotifyObserver(const volatile uint32_t *address) {
        notify_target = address;
        notify_count = 0;
    }
    ~ScopedNotifyObserver() { notify_target = nullptr; }
};

uint64_t observe_prepared_cas(
    volatile uint64_t &word, uint64_t expected, uint64_t desired, int success = __ATOMIC_ACQ_REL,
    int failure = __ATOMIC_ACQUIRE
) {
    if (prepared_cas_observer) prepared_cas_observer(&word);
    return scheduler_gm_compare_exchange(word, expected, desired, success, failure);
}

class ScopedPreparedCasObserver {
public:
    explicit ScopedPreparedCasObserver(std::function<void(volatile uint64_t *)> observer) {
        prepared_cas_observer = std::move(observer);
    }
    ~ScopedPreparedCasObserver() { prepared_cas_observer = {}; }
};

void observe_binding_writeback(void *address) {
    if (writeback_observer) writeback_observer(address);
    scheduler_writeback_cache_line(address);
}

void observe_binding_writeback(volatile void *address) { observe_binding_writeback(const_cast<void *>(address)); }

class ScopedBindingWritebackObserver {
public:
    explicit ScopedBindingWritebackObserver(std::function<void(void *)> observer) {
        writeback_observer = std::move(observer);
    }
    ~ScopedBindingWritebackObserver() { writeback_observer = {}; }
};

void observe_receive_cache_line(void *address) {
    if (receive_observer) {
        auto observer = std::exchange(receive_observer, {});
        observer(address);
        receive_observer = std::move(observer);
    }
    scheduler_invalidate_cache_line(address);
}

void observe_receive_cache_line(volatile void *address) { observe_receive_cache_line(const_cast<void *>(address)); }

class ScopedReceiveObserver {
public:
    explicit ScopedReceiveObserver(std::function<void(void *)> observer) { receive_observer = std::move(observer); }
    ~ScopedReceiveObserver() { receive_observer = {}; }
};
}  // namespace

#define scheduler_invalidate_cache_line observe_receive_cache_line
#define scheduler_gm_compare_exchange observe_prepared_cas
#include "scheduler/scheduler_prepared_execution.h"
#undef scheduler_gm_compare_exchange
#undef scheduler_invalidate_cache_line

#define scheduler_writeback_cache_line observe_binding_writeback
#define scheduler_gm_query observe_query
#define scheduler_notify_query observe_notify_query
#include "hbg_scheduler_test_support.h"
#include "prepared_execution_plan.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_execution_tokens.h"
#include "scheduler/scheduler_prepared_execution.h"
#include "test_hbg_owned_execution.h"
#undef scheduler_writeback_cache_line
#undef scheduler_gm_query
#undef scheduler_notify_query

namespace {

struct RefillFixture {
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<16> graph;
    SchedulerLocalState *local{nullptr};

    explicit RefillFixture(uint32_t tasks, uint32_t capacity_per_task = 2) :
        storage(tasks, 6, capacity_per_task),
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
    void reserve(uint32_t task, uint32_t worker, SchedulerExecutionToken *token) {
        auto *owner = storage.local_context(&storage.contexts[worker]);
        const auto blocks = storage.metadata[task].logical_block_num;
        ASSERT_TRUE(scheduler_claim_ticket(
            graph.graph(), base(), owner, storage.run_control, task, blocks, &token->block_idx, &token->closes_supply
        ));
        ASSERT_LT(token->block_idx, blocks);
        token->task_ref = task;
        token->subtask_slot = 0;
        token->phase = SchedulerExecutionTokenPhase::TICKETED;
        token->state = SchedulerDispatchSlotState::FILLING;
        ASSERT_EQ(scheduler_node_fanin_view(graph.graph(), task, &token->fanin), SchedulerGraphResult::OK);
        ASSERT_TRUE(scheduler_bind_ticketed_execution(graph.graph(), base(), owner, storage.run_control, *token, 0));
    }
    void own(uint32_t task) {
        auto entry =
            std::find_if(std::begin(local->execution_tokens), std::end(local->execution_tokens), [](auto &token) {
                return token.phase == SchedulerExecutionTokenPhase::EMPTY;
            });
        ASSERT_NE(entry, std::end(local->execution_tokens));
        ASSERT_NO_FATAL_FAILURE(reserve(task, 0, entry));
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
    void fill_window(uint32_t count) {
        for (uint32_t slice = 0; slice < graph.graph().task_count * SCHEDULER_EXECUTION_TOKEN_COUNT &&
                                 scheduler_execution_token_count(local) < count;
             ++slice) {
            const auto before = pending_tasks();
            const auto head = local->issue_cursor;
            const auto task = local->issue_task_ref;
            const auto fanin = local->issue_next_fanin;
            bool advanced = false;
            ASSERT_TRUE(refill(&advanced));
            ASSERT_TRUE(advanced);
            ASSERT_TRUE(
                pending_tasks() != before || local->issue_cursor > head || local->issue_task_ref != task ||
                local->issue_next_fanin > fanin
            );
        }
        ASSERT_EQ(scheduler_execution_token_count(local), count);
    }
    bool progress(uint32_t *slot) {
        bool advanced = false;
        return scheduler_progress_execution_tokens(
            graph.graph(), base(), local, storage.run_control, 0, slot, &advanced
        );
    }
    void execute(uint32_t index, SchedulerLocalState *executor = nullptr) {
        if (executor == nullptr) executor = local;
        ASSERT_LT(index, SCHEDULER_LOCAL_EXECUTION_COUNT);
        ASSERT_TRUE(hbg_test_begin_execution(executor, index));
        auto *call = scheduler_prepared_call_at(
            base(), executor->config.prepared, executor->execution_at(index).prepared.execution_index
        );
        auto kernel = reinterpret_cast<void (*)(__gm__ int64_t *)>(call->function_bin_addr);
        kernel(reinterpret_cast<__gm__ int64_t *>(call->args));
        ASSERT_TRUE(
            scheduler_complete_lane_execution<false>(graph.graph(), base(), executor, storage.run_control, index, 0)
        );
    }
};

TEST(HbgExecutionOwnership, FreshTicketCannotBeConsumedTwiceAndBindingIsReceived) {
    RefillFixture fixture(1);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    const auto *prepared = fixture.local->config.prepared;
    auto *binding = scheduler_execution_binding_at(fixture.base(), prepared, 0);
    std::vector<void *> received;
    ScopedReceiveObserver observer([&](void *address) {
        if (address == &binding->local_context || address == &binding->global_context) received.push_back(address);
    });
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    ASSERT_TRUE(advanced);
    auto &token = fixture.local->execution_tokens[0];
    ASSERT_EQ(token.phase, SchedulerExecutionTokenPhase::WAIT_FANIN);
    ASSERT_TRUE(scheduler_bind_ticketed_execution(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, token, 0
    ));
    EXPECT_EQ(received, (std::vector<void *>{&binding->local_context, &binding->global_context}));
    const auto claimed = scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, 0) | SCHEDULER_PREPARED_TICKET_OWNED;
    EXPECT_EQ(token.prepared.claimed_state, claimed);
    SchedulerPreparedHandle duplicate{};
    EXPECT_FALSE(scheduler_claim_for_execution(fixture.base(), prepared, 3, 0, 0, 0, &duplicate));
    EXPECT_EQ(duplicate.claimed_state, 0u);
    EXPECT_EQ(scheduler_prepared_control_at(fixture.base(), prepared, 0)->state, claimed);
}

TEST(HbgExecutionOwnership, InvalidCallFailsClosedAfterClaim) {
    RefillFixture fixture(1);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    const auto *prepared = fixture.local->config.prepared;
    ++scheduler_prepared_call_at(fixture.base(), prepared, 0)->run_epoch;
    bool advanced = false;
    EXPECT_FALSE(fixture.refill(&advanced));
    auto &owner = fixture.local->execution_tokens[0].prepared;
    EXPECT_EQ(scheduler_prepared_phase(owner.claimed_state), SchedulerPreparedPhase::FAILED);
    EXPECT_EQ(scheduler_prepared_control_at(fixture.base(), prepared, 0)->state, owner.claimed_state);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
    EXPECT_TRUE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.local, &owner));
}

TEST(HbgExecutionOwnership, TicketOwnedRetirementKeepsExactStateValidation) {
    RefillFixture fixture(1);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    ASSERT_TRUE(advanced);
    const auto owner = fixture.local->execution_tokens[0].prepared;
    const auto *prepared = fixture.local->config.prepared;
    const auto &call = *scheduler_prepared_call_at(fixture.base(), prepared, 0);
    const auto &plan = *scheduler_state_at<SchedulerExecutionPlan>(fixture.base(), prepared->plans_offset);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    const uint64_t completed = fixture.control(0)->completion_state;
    ASSERT_TRUE(scheduler_prepared_retirement_inferred(owner.claimed_state, *prepared, 0, call, plan, completed));
    for (uint32_t bit : {24u, 26u, 32u, 63u}) {
        auto malformed = owner;
        malformed.claimed_state |= UINT64_C(1) << bit;
        EXPECT_FALSE(scheduler_claim_for_execution(fixture.base(), prepared, 0, 0, 0, 0, &malformed));
        EXPECT_FALSE(
            scheduler_prepared_retirement_inferred(malformed.claimed_state, *prepared, 0, call, plan, completed)
        );
    }
}

TEST(HbgExecutionOwnership, OnlyTheFixedOwnerCanCancelAnUnstartedExecution) {
    RefillFixture fixture(1);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    ASSERT_TRUE(advanced);
    auto &owner = fixture.local->execution_tokens[0].prepared;
    const auto *record = scheduler_prepared_control_at(fixture.base(), fixture.local->config.prepared, 0);
    auto stale = owner;
    auto *peer = fixture.storage.local_context(&fixture.storage.contexts[3]);
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), peer, &owner));
    ASSERT_EQ(record->state, stale.claimed_state);
    ASSERT_TRUE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.local, &owner));
    EXPECT_EQ(owner.claimed_state, 0u);
    EXPECT_EQ(scheduler_prepared_phase(record->state), SchedulerPreparedPhase::FAILED);
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.local, &stale));
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
}

TEST(HbgExecutionRefill, ContinuationExhaustionIsReusedWithoutAdvancingAnUnsealedHead) {
    RefillFixture fixture(1);
    fixture.storage.metadata[0].logical_block_num = 2;
    fixture.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerIssueSnapshot snapshot{};
    int64_t head = SCHEDULER_TASK_REF_INVALID;
    bool progress = false;
    ASSERT_TRUE(scheduler_issue_head(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &head, &progress, &snapshot
    ));
    ASSERT_EQ(head, 0);
    ASSERT_EQ(fixture.local->issue_task_ref, 0);
    fixture.control(0)->next_block = 2;
    fixture.local->continuation_task_ref = 0;
    ScopedQueryObserver observer(&fixture.control(0)->next_block);
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &progress, &snapshot
    ));
    EXPECT_TRUE(fixture.local->issue_ticket_exhausted);
    EXPECT_EQ(query_count, 1u);
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &progress, &snapshot
    ));
    EXPECT_EQ(query_count, 1u);
    EXPECT_TRUE(snapshot.evaluated);
    EXPECT_EQ(fixture.local->issue_cursor, 0u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    fixture.control(0)->claim_seal = 2;
    snapshot = {};
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &progress, &snapshot
    ));
    EXPECT_EQ(fixture.local->issue_cursor, 1u);
    EXPECT_EQ(query_count, 1u);
}

TEST(HbgExecutionRefill, ChangedHeadClaimsSuccessorWithoutInheritingContinuationFanin) {
    RefillFixture fixture(2);
    fixture.storage.metadata[0].logical_block_num = 2;
    fixture.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
    fixture.graph.executable(1, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());

    bool advanced = false;
    ASSERT_TRUE(fixture.refill(&advanced));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_EXECUTION_TOKEN_COUNT);
    SchedulerExecutionToken other{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &other));
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    ASSERT_EQ(fixture.local->continuation_task_ref, 0);

    bool failed = false;
    EXPECT_TRUE(scheduler_find_local_execution(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
    ));
    EXPECT_FALSE(failed);
    EXPECT_EQ(slot, UINT32_MAX);
    EXPECT_TRUE(scheduler_execution_token_has_task(fixture.local, 1));
    EXPECT_EQ(fixture.local->execution_tokens[0].next_fanin, 0u);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
}

TEST(HbgExecutionRefill, RetirementSnapshotReadsTheCounterWithoutPublishingProgress) {
    RefillFixture fixture(1);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    uint32_t value = 0;
    auto *counters = scheduler_state_at<SchedulerTaskCompletionCounter>(
        fixture.base(), fixture.storage.layout.completion_counters_offset
    );
    counters[0] = {reinterpret_cast<uint64_t>(&value), reinterpret_cast<uint64_t>(&value), 1, 1};
    fixture.local->config.completion_counters_offset = fixture.storage.layout.completion_counters_offset;
    scheduler_remember_task_proof(fixture.local, 0, SchedulerTaskProof::SUPPLY_CLOSED);
    ScopedNotifyObserver observer(&value);
    scheduler_snapshot_counter_completions(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control
    );
    EXPECT_EQ(notify_count, 1u);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
    value = 1;
    scheduler_snapshot_counter_completions(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control
    );
    EXPECT_EQ(notify_count, 2u);
    EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
}

TEST(HbgExecutionRefill, TicketOwnerKeepsLifecycleWithoutASecondOwnershipAtomic) {
    for (bool cancel : {false, true}) {
        RefillFixture fixture(1);
        fixture.storage.metadata[0].logical_block_num = 2;
        fixture.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
        ASSERT_NO_FATAL_FAILURE(fixture.initialize());
        auto *record = scheduler_prepared_control_at(fixture.base(), fixture.local->config.prepared, 0);
        uint32_t exchanges = 0;
        ScopedPreparedCasObserver observer([&](volatile uint64_t *address) {
            if (address == &record->state) ++exchanges;
        });
        bool advanced = false;
        ASSERT_TRUE(fixture.refill(&advanced));
        ASSERT_TRUE(advanced);
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(scheduler_prepared_owner(record->state), 0u);
        EXPECT_EQ(scheduler_prepared_phase(record->state), SchedulerPreparedPhase::CLAIMED);
        EXPECT_EQ(exchanges, 0u);
        if (cancel) {
            auto stale = fixture.local->execution_tokens[0].prepared;
            ASSERT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), fixture.local));
            EXPECT_EQ(scheduler_prepared_phase(record->state), SchedulerPreparedPhase::FAILED);
            EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.local, &stale));
        } else {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(fixture.progress(&slot));
            ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
            ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
            EXPECT_EQ(scheduler_prepared_phase(record->state), SchedulerPreparedPhase::RETURNED);
        }
        EXPECT_EQ(exchanges, 0u);
        EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
    }
}

TEST(HbgExecutionRefill, LastClaimOpensSuccessorPreparationBeforeReceivingItsPayload) {
    for (bool invalid_call : {false, true}) {
        SCOPED_TRACE(invalid_call);
        RefillFixture fixture(2);
        fixture.graph.executable(1, 0, {0});
        ASSERT_NO_FATAL_FAILURE(fixture.initialize());
        auto *peer = fixture.storage.local_context(&fixture.storage.contexts[3]);
        uint64_t index = UINT64_MAX;
        ASSERT_TRUE(
            scheduler_prepared_execution_index(fixture.base(), fixture.local->config.prepared, 0, 0, 0, &index)
        );
        auto *call = scheduler_prepared_call_at(fixture.base(), fixture.local->config.prepared, index);
        if (invalid_call) call->function_bin_addr = 0;
        uint32_t observations = 0;
        ScopedReceiveObserver observer([&](void *address) {
            const auto *binding = scheduler_execution_binding_at(fixture.base(), fixture.local->config.prepared, index);
            if (address != &binding->local_context || observations != 0) return;
            ++observations;
            EXPECT_EQ(fixture.control(0)->claim_seal, 1u);
            EXPECT_EQ(fixture.local->execution_tokens[0].phase, SchedulerExecutionTokenPhase::TICKETED);
            EXPECT_EQ(
                scheduler_prepared_phase(fixture.local->execution_tokens[0].prepared.claimed_state),
                SchedulerPreparedPhase::CLAIMED
            );
            bool progress = false;
            ASSERT_TRUE(scheduler_refill_execution_tokens(
                fixture.graph.graph(), fixture.base(), peer, fixture.storage.run_control, 0, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(fixture.control(1)->next_block, 1u);
            EXPECT_EQ(peer->execution_tokens[0].task_ref, 1);
            EXPECT_EQ(peer->execution_tokens[0].phase, SchedulerExecutionTokenPhase::WAIT_FANIN);
            EXPECT_EQ(peer->execution_tokens[0].next_fanin, 0u);
            EXPECT_EQ(hbg_test_eligible_mask(peer), 0u);
            EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
        });
        bool progress = false;
        EXPECT_EQ(fixture.refill(&progress), !invalid_call);
        ASSERT_EQ(observations, 1u);
        if (invalid_call) {
            EXPECT_NE(fixture.storage.run_control->scheduler_error, 0u);
            EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
            EXPECT_FALSE(scheduler_completion_done(fixture.control(1)->completion_state));
            EXPECT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), fixture.local));
            EXPECT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), peer));
            EXPECT_EQ(scheduler_execution_token_count(fixture.local), 0u);
            EXPECT_EQ(scheduler_execution_token_count(peer), 0u);
        } else {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(fixture.progress(&slot));
            ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
            ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
            ASSERT_TRUE(scheduler_progress_execution_tokens(
                fixture.graph.graph(), fixture.base(), peer, fixture.storage.run_control, 0, &slot, &progress
            ));
            ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
            EXPECT_EQ(peer->execution_at(slot).task_ref, 1);
            EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
        }
    }
}

TEST(HbgExecutionRefill, WaitingTokenUsesTheHostBindingWhenAllPredecessorsComplete) {
    RefillFixture fixture(3);
    fixture.graph.executable(2, 0, {0, 1});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerExecutionToken producers[2]{};
    for (uint32_t task = 0; task < 2; ++task)
        ASSERT_NO_FATAL_FAILURE(fixture.reserve(task, 3, &producers[task]));
    uint64_t index = UINT64_MAX;
    ASSERT_TRUE(scheduler_prepared_execution_index(fixture.base(), fixture.local->config.prepared, 2, 0, 0, &index));
    auto *binding = scheduler_execution_binding_at(fixture.base(), fixture.local->config.prepared, index);
    uint32_t writebacks = 0;
    uint32_t local_writebacks = 0;
    ScopedBindingWritebackObserver observer([&](void *address) {
        if (address == &binding->global_context) ++writebacks;
        if (address == &binding->local_context) ++local_writebacks;
    });
    ASSERT_NO_FATAL_FAILURE(fixture.own(2));
    EXPECT_EQ(writebacks, 0u);
    EXPECT_EQ(local_writebacks, 0u);
    for (uint32_t predecessor = 0; predecessor < 2; ++predecessor) {
        for (uint32_t attempt = 0; attempt < 2; ++attempt) {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(fixture.progress(&slot));
            EXPECT_EQ(slot, UINT32_MAX);
            EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{2}));
            EXPECT_EQ(writebacks, 0u);
        }
        uint64_t previous = 0;
        ASSERT_EQ(
            scheduler_contribute_returns(fixture.control(predecessor), 1, &previous),
            SchedulerSubtaskFinishResult::TASK_LAST
        );
    }
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
    EXPECT_EQ(fixture.local->execution_at(slot).task_ref, 2);
    EXPECT_EQ(writebacks, 0u);
    EXPECT_EQ(local_writebacks, 0u);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 0u);
}

std::set<int64_t> expected_window_tasks(int64_t first) {
    std::set<int64_t> expected;
    for (uint32_t offset = 0; offset < SCHEDULER_EXECUTION_TOKEN_COUNT; ++offset)
        expected.insert(first + offset);
    return expected;
}

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
    int64_t head = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_issue_head(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &head, &progress
    ));
    EXPECT_EQ(head, 0);
    EXPECT_EQ(fixture.local->issue_cursor, 0u);
    SchedulerExecutionToken peer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &peer));
    ASSERT_TRUE(scheduler_issue_head(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &head, &progress
    ));
    // Rechecking the held token can end the first probe just after crossing the seal.
    EXPECT_EQ(fixture.local->issue_cursor, 1u);
    ASSERT_TRUE(scheduler_issue_head(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &head, &progress
    ));
    EXPECT_EQ(head, 1);
    ASSERT_NO_FATAL_FAILURE(fixture.fill_window(SCHEDULER_EXECUTION_TOKEN_COUNT));
    EXPECT_EQ(fixture.pending_tasks(), expected_window_tasks(0));
    EXPECT_EQ(fixture.control(0)->next_block, 2u);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(0)->completion_state));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT);
}

TEST(HbgExecutionRefill, FullWaitingWindowKeepsEveryOwnerUntilTheProducerCompletes) {
    RefillFixture fixture(6);
    for (uint32_t task = 1; task < 6; ++task)
        fixture.graph.executable(task, 0, {0});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    auto *peer = fixture.storage.local_context(&fixture.storage.contexts[3]);
    bool failed = false;
    uint32_t producer_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_find_local_execution(
        fixture.graph.graph(), fixture.base(), peer, fixture.storage.run_control, 0, &failed, &producer_slot
    ));
    ASSERT_FALSE(failed);
    ASSERT_LT(producer_slot, SCHEDULER_EXECUTION_TOKEN_COUNT);
    uint32_t slot = UINT32_MAX;
    for (uint32_t pass = 0;
         pass < 12 && scheduler_execution_token_count(fixture.local) < SCHEDULER_EXECUTION_TOKEN_COUNT; ++pass) {
        scheduler_find_local_execution(
            fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
        );
        ASSERT_FALSE(failed);
        ASSERT_EQ(slot, UINT32_MAX);
    }
    bool advanced = false;
    ASSERT_EQ(fixture.pending_tasks(), expected_window_tasks(1));
    std::array<SchedulerPreparedHandle, SCHEDULER_EXECUTION_TOKEN_COUNT> held{};
    for (uint32_t index = 0; index < held.size(); ++index)
        held[index] = fixture.local->execution_tokens[index].prepared;
    ASSERT_TRUE(fixture.refill(&advanced));
    EXPECT_FALSE(advanced);
    EXPECT_EQ(fixture.control(SCHEDULER_EXECUTION_TOKEN_COUNT + 1)->next_block, 0u);
    ASSERT_TRUE(fixture.progress(&slot));
    EXPECT_EQ(slot, UINT32_MAX);
    for (uint32_t index = 0; index < held.size(); ++index) {
        const auto &current = fixture.local->execution_tokens[index].prepared;
        EXPECT_EQ(current.execution_index, held[index].execution_index);
        EXPECT_EQ(current.claimed_state, held[index].claimed_state);
        EXPECT_EQ(scheduler_prepared_owner(current.claimed_state), 0u);
    }
    ASSERT_NO_FATAL_FAILURE(fixture.execute(producer_slot, peer));
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
    if constexpr (SCHEDULER_EXECUTION_TOKEN_COUNT > 1) {
        const uint32_t first = slot;
        ASSERT_TRUE(fixture.progress(&slot));
        ASSERT_LT(slot, SCHEDULER_EXECUTION_TOKEN_COUNT);
        EXPECT_NE(slot, first);
        EXPECT_EQ(fixture.local->execution_at(slot).prepared.execution_index, held[slot].execution_index);
    }
    EXPECT_EQ(scheduler_prepared_owner(fixture.local->execution_at(slot).prepared.claimed_state), 0u);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), SCHEDULER_EXECUTION_TOKEN_COUNT - 1);
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
    ASSERT_NO_FATAL_FAILURE(fixture.fill_window(SCHEDULER_EXECUTION_TOKEN_COUNT));
    EXPECT_EQ(fixture.pending_tasks(), expected_window_tasks(0));
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
    EXPECT_EQ(fixture.local->execution_at(slot).task_ref, 0);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
}

TEST(HbgExecutionRefill, IndependentClaimSectorsAssignEveryBlockExactlyOnce) {
    for (const uint16_t count : {1, 3, 8, 80, 129}) {
        SCOPED_TRACE(count);
        RefillFixture fixture(1, count);
        fixture.storage.metadata[0].logical_block_num = count;
        ASSERT_NO_FATAL_FAILURE(fixture.initialize());
        std::array<SchedulerLocalState *, 6> locals{};
        for (uint32_t worker = 0; worker < locals.size(); ++worker)
            locals[worker] = fixture.storage.local_context(&fixture.storage.contexts[worker]);
        std::array<std::vector<uint32_t>, 6> claimed;
        std::array<std::thread, 6> workers;
        std::atomic<bool> start{false};
        std::atomic<uint32_t> closing_owners{0};
        for (uint32_t worker = 0; worker < workers.size(); ++worker) {
            workers[worker] = std::thread([&, worker] {
                while (!start.load(std::memory_order_acquire)) {}
                for (;;) {
                    uint32_t block = count;
                    bool closes = false;
                    ASSERT_TRUE(scheduler_claim_ticket(
                        fixture.graph.graph(), fixture.base(), locals[worker], fixture.storage.run_control, 0, count,
                        &block, &closes
                    ));
                    if (block == count) break;
                    claimed[worker].push_back(block);
                    if (closes) {
                        closing_owners.fetch_add(1, std::memory_order_relaxed);
                        ASSERT_TRUE(scheduler_close_ordinary_supply(
                            fixture.graph.graph(), fixture.base(), locals[worker], fixture.storage.run_control, 0,
                            closes, count, block
                        ));
                    }
                }
            });
        }
        start.store(true, std::memory_order_release);
        for (auto &worker : workers)
            worker.join();
        std::vector<uint32_t> blocks;
        for (const auto &owned : claimed)
            blocks.insert(blocks.end(), owned.begin(), owned.end());
        std::sort(blocks.begin(), blocks.end());
        ASSERT_EQ(blocks.size(), count);
        for (uint32_t block = 0; block < count; ++block)
            EXPECT_EQ(blocks[block], block);
        EXPECT_EQ(closing_owners.load(), 1u);
        EXPECT_EQ(fixture.control(0)->next_block, count);
        EXPECT_EQ(fixture.control(0)->claim_seal, count);
        EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
    }
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
    ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));

    bool failed = false;
    slot = UINT32_MAX;
    scheduler_find_local_execution<false>(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, 0, &failed, &slot
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(slot, SCHEDULER_LOCAL_EXECUTION_COUNT);
    EXPECT_EQ(fixture.local->execution_at(slot).task_ref, 0);
    EXPECT_EQ(fixture.local->execution_at(slot).block_idx, 4u);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
}

TEST(HbgExecutionRefill, SkipsExhaustedEntriesAndReturnsAsSoonAsAReadyTicketIsFound) {
    RefillFixture fixture(6);
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    std::array<SchedulerExecutionToken, 3> other_core{};
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
