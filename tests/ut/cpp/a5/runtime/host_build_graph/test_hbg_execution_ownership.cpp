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
#undef scheduler_writeback_cache_line
#undef scheduler_gm_query
#undef scheduler_notify_query

namespace {

struct OwnershipFixture {
    alignas(128) std::array<uint8_t, 2048> bytes{};
    SchedulerPreparedLayout *prepared;

    OwnershipFixture() {
        AicoreSchedulerLayout layout{};
        layout.total_size = bytes.size();
        layout.task_count = 1;
        layout.prepared_layout_offset = 64;
        prepared = scheduler_state_at<SchedulerPreparedLayout>(base(), layout.prepared_layout_offset);
        *prepared = {17, 1, 1, 128, 256, 512, 384, 1024};
        scheduler_state_at<SchedulerTaskControl>(base(), prepared->task_controls_offset)->completion_state =
            scheduler_completion_initial_state(1);
        *scheduler_state_at<SchedulerExecutionPlan>(base(), prepared->plans_offset) = {0, 1, 1, 1, 1};
        const uint64_t functions[3]{1, 0, 0};
        EXPECT_TRUE(scheduler_prepare_host_calls(
            base(), layout, reinterpret_cast<uint64_t>(base()), 0, functions, 0, nullptr, 0, 0
        ));
    }

    void *base() { return bytes.data(); }
    SchedulerPreparedControl *control() { return scheduler_prepared_control_at(base(), prepared, 0); }
    bool claim(uint32_t worker, SchedulerPreparedHandle *handle) {
        return scheduler_acquire_execution_token(base(), prepared, worker, 0, 0, 0, handle);
    }
};

TEST(HbgExecutionOwnership, FreshTicketCannotBeConsumedTwiceAndBindingIsReceived) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle first{}, duplicate{};
    std::vector<void *> received;
    ScopedReceiveObserver observer([&](void *address) {
        received.push_back(address);
    });
    ASSERT_TRUE(fixture.claim(0, &first));
    auto *binding = scheduler_execution_binding_at(fixture.base(), fixture.prepared, 0);
    EXPECT_EQ(received, (std::vector<void *>{&binding->local_context, &binding->global_context}));
    ASSERT_TRUE(fixture.claim(0, &first));
    EXPECT_EQ(received.size(), 2u);
    EXPECT_EQ(first.claimed_state, scheduler_prepared_state(SchedulerPreparedPhase::CLAIMED, 0));
    EXPECT_FALSE(fixture.claim(1, &duplicate));
    EXPECT_EQ(duplicate.claimed_state, 0u);
    EXPECT_EQ(fixture.control()->state, first.claimed_state);
    const auto &call = *scheduler_prepared_call_at(fixture.base(), fixture.prepared, 0);
    const auto &plan = *scheduler_state_at<SchedulerExecutionPlan>(fixture.base(), fixture.prepared->plans_offset);
    const uint64_t completed = scheduler_completion_completed_state(1);
    ASSERT_TRUE(
        scheduler_prepared_retirement_inferred(first.claimed_state, *fixture.prepared, 0, call, plan, completed)
    );
    for (uint32_t bit : {24u, 32u, 63u}) {
        SchedulerPreparedHandle malformed = first;
        malformed.claimed_state |= UINT64_C(1) << bit;
        EXPECT_FALSE(fixture.claim(0, &malformed));
        EXPECT_FALSE(
            scheduler_prepared_retirement_inferred(malformed.claimed_state, *fixture.prepared, 0, call, plan, completed)
        );
        EXPECT_EQ(fixture.control()->state, first.claimed_state);
    }
    EXPECT_TRUE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &first));
    EXPECT_FALSE(fixture.claim(0, &duplicate));
}

TEST(HbgExecutionOwnership, InvalidCallFailsClosedAfterClaim) {
    OwnershipFixture fixture;
    scheduler_prepared_call_at(fixture.base(), fixture.prepared, 0)->run_epoch = 18;
    SchedulerPreparedHandle owner{};
    EXPECT_FALSE(fixture.claim(0, &owner));
    EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &owner));
}

TEST(HbgExecutionOwnership, TicketOwnedRetirementKeepsExactStateValidation) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{};
    ASSERT_TRUE(scheduler_claim_execution_token<true>(fixture.base(), fixture.prepared, 0, 0, 0, 0, &owner));
    const auto &call = *scheduler_prepared_call_at(fixture.base(), fixture.prepared, 0);
    const auto &plan = *scheduler_state_at<SchedulerExecutionPlan>(fixture.base(), fixture.prepared->plans_offset);
    const uint64_t completed = scheduler_completion_completed_state(1);
    EXPECT_TRUE(
        scheduler_prepared_retirement_inferred(owner.claimed_state, *fixture.prepared, 0, call, plan, completed)
    );
    for (uint32_t bit : {24u, 26u, 32u, 63u}) {
        EXPECT_FALSE(scheduler_prepared_retirement_inferred(
            owner.claimed_state | (UINT64_C(1) << bit), *fixture.prepared, 0, call, plan, completed
        ));
    }
}

TEST(HbgExecutionOwnership, OnlyTheFixedOwnerCanCancelAnUnstartedExecution) {
    OwnershipFixture fixture;
    SchedulerPreparedHandle owner{}, duplicate{};
    ASSERT_TRUE(fixture.claim(0, &owner));
    const auto stale = owner;
    EXPECT_FALSE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 1, &owner));
    EXPECT_EQ(fixture.control()->state, stale.claimed_state);
    ASSERT_TRUE(scheduler_fail_prepared_execution(fixture.base(), fixture.prepared, 0, &owner));
    EXPECT_EQ(owner.claimed_state, 0u);
    EXPECT_EQ(scheduler_prepared_phase(fixture.control()->state), SchedulerPreparedPhase::FAILED);
    auto returned = stale;
    EXPECT_FALSE(scheduler_return_prepared_execution(fixture.base(), fixture.prepared, &returned));
    EXPECT_FALSE(fixture.claim(1, &duplicate));
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
        ASSERT_TRUE(scheduler_acquire_execution_token(base(), local->config.prepared, worker, task, block, 0, handle));
        ASSERT_TRUE(scheduler_close_ordinary_supply(
            graph.graph(), base(), storage.local_context(&storage.contexts[worker]), storage.run_control, task,
            static_cast<uint32_t>(block), storage.metadata[task].logical_block_num
        ));
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
        ASSERT_TRUE(scheduler_prepare_token_binding(base(), local, *entry, 0));
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
    ASSERT_TRUE(scheduler_try_execution_continuation(
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

TEST(HbgExecutionRefill, CounterAccountingReusesOnlyAnObservedDoneProof) {
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
    bool progress = false;
    ASSERT_TRUE(scheduler_service_counter_accounting(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(notify_count, 1u);
    EXPECT_EQ(fixture.local->pending_completed, 0u);
    value = 1;
    EXPECT_EQ(
        scheduler_observe_cached_completion(fixture.base(), fixture.local, 0), SchedulerCompletionObservation::DONE
    );
    EXPECT_EQ(notify_count, 2u);
    ASSERT_TRUE(scheduler_service_counter_accounting(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(notify_count, 2u);
    EXPECT_EQ(fixture.local->pending_completed, 1u);
    EXPECT_TRUE(scheduler_completion_done(fixture.control(0)->completion_state));
    ASSERT_TRUE(scheduler_service_counter_accounting(
        fixture.graph.graph(), fixture.base(), fixture.local, fixture.storage.run_control, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.local->pending_completed, 1u);
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
            EXPECT_FALSE(scheduler_fail_prepared_execution(fixture.base(), fixture.local->config.prepared, 0, &stale));
        } else {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(fixture.progress(&slot));
            ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
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
            EXPECT_EQ(peer->local_ready_mask, 0u);
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
            ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
            ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
            ASSERT_TRUE(scheduler_progress_execution_tokens(
                fixture.graph.graph(), fixture.base(), peer, fixture.storage.run_control, 0, &slot, &progress
            ));
            ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
            EXPECT_EQ(peer->slots[slot].task_ref, 1);
            EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
        }
    }
}

TEST(HbgExecutionRefill, WaitingTokenRebindsAfterEachPredecessorCompletes) {
    RefillFixture fixture(3);
    fixture.graph.executable(2, 0, {0, 1});
    ASSERT_NO_FATAL_FAILURE(fixture.initialize());
    SchedulerPreparedHandle producers[2]{};
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
    EXPECT_EQ(writebacks, 1u);
    EXPECT_EQ(local_writebacks, 0u);
    for (uint32_t predecessor = 0; predecessor < 2; ++predecessor) {
        for (uint32_t attempt = 0; attempt < 2; ++attempt) {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(fixture.progress(&slot));
            EXPECT_EQ(slot, UINT32_MAX);
            EXPECT_EQ(fixture.pending_tasks(), (std::set<int64_t>{2}));
        }
        uint64_t previous = 0;
        ASSERT_EQ(
            scheduler_contribute_returns(fixture.control(predecessor), 1, &previous),
            SchedulerSubtaskFinishResult::TASK_LAST
        );
    }
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(&slot));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(fixture.local->slots[slot].task_ref, 2);
    EXPECT_EQ(writebacks, 1u);
    EXPECT_EQ(local_writebacks, 0u);
    ASSERT_NO_FATAL_FAILURE(fixture.execute(slot));
    EXPECT_EQ(scheduler_execution_token_count(fixture.local), 0u);
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
    SchedulerPreparedHandle peer{};
    ASSERT_NO_FATAL_FAILURE(fixture.reserve(0, 3, &peer));
    ASSERT_NO_FATAL_FAILURE(fixture.fill_window(SCHEDULER_EXECUTION_TOKEN_COUNT));
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
    ASSERT_NO_FATAL_FAILURE(fixture.fill_window(SCHEDULER_EXECUTION_TOKEN_COUNT));
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
    ASSERT_NO_FATAL_FAILURE(fixture.fill_window(SCHEDULER_EXECUTION_TOKEN_COUNT));
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
