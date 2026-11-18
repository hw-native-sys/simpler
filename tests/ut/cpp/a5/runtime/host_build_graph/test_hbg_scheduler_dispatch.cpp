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

#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include "scheduler/scheduler_dispatch.h"
#include "hbg_scheduler_test_support.h"
#include "hbg_gm_mix_test_support.h"

namespace {
using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;

struct DispatchFixture {
    FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<32> graph;
    SchedulerLocalState *worker{nullptr};

    explicit DispatchFixture(uint32_t count = 1, uint32_t traces_per_task = 1) :
        storage(count, 3, traces_per_task),
        graph(count) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage.contexts[worker].cluster_worker_ids[lane] = lane;
        }
        for (uint32_t task = 0; task < count; ++task) {
            graph.executable(task, 0);
            storage.metadata[task].timing_slot = -1;
        }
    }
    void *base() { return storage.scheduler_state->base(); }
    SchedulerLocalState *local() {
        if (worker == nullptr) worker = storage.local_context(&storage.contexts[0]);
        return worker;
    }
    bool prepare() {
        storage.prepare_catalog();
        return storage.prepare_calls(graph.graph());
    }
    bool take(uint32_t &slot, uint64_t level = 0) {
        slot = UINT32_MAX;
        for (uint64_t visit = 0; visit <= storage.layout.task_count; ++visit) {
            bool failed = false;
            scheduler_fill_own_slots(graph.graph(), base(), local(), storage.run_control, level, &failed, &slot);
            if (failed) return false;
            if (slot != UINT32_MAX) return true;
        }
        return false;
    }
    bool complete(uint32_t slot, uint64_t level = 0) {
        if (local()->slots[slot].state != SchedulerDispatchSlotState::READY) return false;
        local()->slots[slot].state = SchedulerDispatchSlotState::EXECUTING;
        return scheduler_complete_own_slot(graph.graph(), base(), local(), storage.run_control, slot, level);
    }
    SchedulerTaskControl *control(uint32_t task) { return scheduler_task_control_at(base(), local(), task); }
};

void bind_counter(DispatchFixture &f, uint32_t task, uint32_t &value, uint32_t threshold, uint32_t limit) {
    auto *counters =
        scheduler_state_at<SchedulerTaskCompletionCounter>(f.base(), f.storage.layout.completion_counters_offset);
    counters[task] = {reinterpret_cast<uint64_t>(&value), reinterpret_cast<uint64_t>(&value), threshold, limit};
    for (uint32_t worker = 0; worker < 3; ++worker)
        f.storage.contexts[worker].completion_counters_offset = f.storage.layout.completion_counters_offset;
    f.local()->config.completion_counters_offset = f.storage.layout.completion_counters_offset;
}

TEST(SchedulerTokenDispatch, AddChainReleasesSuccessorsBeforeAccountingAndKeepsOldThresholdsValid) {
    DispatchFixture f(2, 2);
    for (uint32_t task = 0; task < 2; ++task) {
        f.storage.metadata[task].logical_block_num = 2;
        f.storage.metadata[task].flags |= SCHEDULER_TASK_SPMD;
    }
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.prepare());
    uint32_t value = 0;
    bind_counter(f, 0, value, 2, 4);
    bind_counter(f, 1, value, 4, 4);
    const auto first = scheduler_completion_source(f.base(), f.local(), 0);
    for (uint32_t task = 0; task < 2; ++task) {
        for (uint32_t block = 0; block < 2; ++block) {
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(f.take(slot));
            ASSERT_EQ(f.local()->slots[slot].task_ref, task);
            auto stale = f.local()->slots[slot].prepared;
            ASSERT_TRUE(f.complete(slot));
            EXPECT_FALSE(scheduler_return_prepared_execution(f.base(), f.local()->config.prepared, &stale));
            EXPECT_EQ(value, task * 2 + block + 1);
        }
        EXPECT_EQ(scheduler_completion_remaining(f.control(task)->completion_state), 2u);
        EXPECT_EQ(scheduler_observe_task_completion(first), SchedulerCompletionObservation::DONE);
    }
    bool progress = false;
    ASSERT_TRUE(
        scheduler_service_counter_accounting(f.graph.graph(), f.base(), f.local(), f.storage.run_control, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_TRUE(scheduler_completion_done(f.control(0)->completion_state));
    EXPECT_EQ(f.local()->pending_completed, 1u);
    ASSERT_TRUE(
        scheduler_service_counter_accounting(f.graph.graph(), f.base(), f.local(), f.storage.run_control, &progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_EQ(value, 4u);
    EXPECT_EQ(f.local()->pending_completed, 1u);
    value = 5;
    EXPECT_EQ(scheduler_observe_task_completion(first), SchedulerCompletionObservation::INVALID);
}

TEST(SchedulerTokenDispatch, AddMixContributionCountsOnlyActiveLanesAtTheExistingReleaseBoundary) {
    DispatchFixture f(1, 4);
    auto &metadata = f.storage.metadata[0];
    metadata.logical_block_num = 2;
    metadata.active_mask = LaneMask::both_aiv();
    metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX;
    metadata.kernel_ids[0] = UINT16_MAX;
    metadata.kernel_ids[1] = metadata.kernel_ids[2] = 1;
    f.graph.storage()[0].task.kernel_id[0] = INVALID_KERNEL_ID;
    f.graph.storage()[0].task.kernel_id[1] = f.graph.storage()[0].task.kernel_id[2] = 1;
    ASSERT_TRUE(f.prepare());
    for (bool async_lane : {false, true}) {
        SCOPED_TRACE(async_lane);
        uint32_t value = 0;
        const uint32_t credits_per_block = async_lane ? 3 : 2;
        bind_counter(f, 0, value, 2 * credits_per_block, 2 * credits_per_block);
        const auto *counter = scheduler_completion_counter_at(f.base(), f.storage.layout.completion_counters_offset, 0);
        for (uint32_t block = 0; block < 2; ++block) {
            if (async_lane)
                ASSERT_TRUE(scheduler_counter_publish_condition(
                    f.base(), *counter, 0, block, 2, f.storage.layout.task_metadata_offset, 0, 2, false
                ));
            EXPECT_EQ(value, block * credits_per_block + (async_lane ? 1 : 0));
            EXPECT_EQ(scheduler_completion_counter_observe(*counter), 0);
            ASSERT_TRUE(
                scheduler_contribute_mix_group<false>(
                    f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, block, LaneMask::both_aiv()
                )
            );
            EXPECT_EQ(value, (block + 1) * credits_per_block);
        }
        EXPECT_EQ(scheduler_completion_counter_observe(*counter), 1);
        EXPECT_EQ(scheduler_completion_remaining(f.control(0)->completion_state), 4u);
    }
}

TEST(SchedulerTokenDispatch, CntNotifyCompletionReleasesAllReadersWithoutConsumingTheSignal) {
    DispatchFixture f(2);
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.prepare());
    uint32_t notification = 0;
    auto *addresses = scheduler_state_at<uint64_t>(f.base(), f.storage.layout.done_notifications_offset);
    addresses[0] = reinterpret_cast<uint64_t>(&notification);
    const auto source = scheduler_completion_source(f.base(), f.local(), 0);
    ASSERT_EQ(scheduler_observe_task_completion(source), SchedulerCompletionObservation::PENDING);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(f.take(slot));
    ASSERT_EQ(f.local()->slots[slot].task_ref, 0);
    ASSERT_TRUE(f.complete(slot));
    ASSERT_EQ(notification, 1u);
    EXPECT_EQ(scheduler_observe_task_completion(source), SchedulerCompletionObservation::DONE);
    EXPECT_EQ(scheduler_observe_task_completion(source), SchedulerCompletionObservation::DONE);
    EXPECT_EQ(notification, 1u);
    ASSERT_TRUE(f.take(slot));
    EXPECT_EQ(f.local()->slots[slot].task_ref, 1);
}

TEST(SchedulerTokenDispatch, CntNotifyWaitKeepsItsAddressAndDoesNotConfirmCanonicalGmState) {
    DispatchFixture f(2);
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.prepare());
    uint32_t notification = 0;
    auto *addresses = scheduler_state_at<uint64_t>(f.base(), f.storage.layout.done_notifications_offset);
    addresses[0] = reinterpret_cast<uint64_t>(&notification);
    SchedulerFaninView fanin{};
    ASSERT_EQ(scheduler_node_fanin_view(f.graph.graph(), 1, &fanin), SchedulerGraphResult::OK);
    SchedulerFaninWait wait{};
    uint32_t next = 0;
    bool ready = false;
    ASSERT_TRUE(scheduler_observe_fanin_prefix(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 1, fanin, next, &ready, &wait
    ));
    EXPECT_FALSE(ready);
    EXPECT_EQ(wait.source.notification, &notification);
    // Poison unrelated proof storage after binding: the saved SRAM source
    // alone authorizes this wait, and the address table is not read twice.
    addresses[0] = 0;
    f.control(0)->completion_state = 0;
    scheduler_notify_done(notification);
    ASSERT_TRUE(scheduler_observe_fanin_prefix(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 1, fanin, next, &ready, &wait
    ));
    EXPECT_TRUE(ready);
    EXPECT_EQ(next, 1u);
    EXPECT_EQ(f.storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerTokenDispatch, FreshReadyRefillSelectsOneExecutionWithoutAnotherScan) {
    DispatchFixture f;
    ASSERT_TRUE(f.prepare());
    bool progress = false;
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_refill_execution_tokens(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &progress, nullptr, &slot
    ));
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_TRUE(progress);
    EXPECT_EQ(f.control(0)->next_block, 1u);
    EXPECT_EQ(scheduler_cached_task_proof(f.local(), 0), SchedulerTaskProof::SUPPLY_CLOSED);
    EXPECT_FALSE(scheduler_completion_done(f.control(0)->completion_state));
    EXPECT_EQ(f.local()->local_ready_mask, 0u);
    EXPECT_EQ(f.local()->slots[slot].task_ref, 0);
    EXPECT_EQ(scheduler_execution_token_count(f.local()), 1u);
    ASSERT_TRUE(f.complete(slot));
    EXPECT_TRUE(scheduler_completion_done(f.control(0)->completion_state));
    EXPECT_EQ(scheduler_execution_token_count(f.local()), 0u);
}

TEST(SchedulerTokenDispatch, ReturnReleasesItsSlotAndFlushCountsCompletionOnce) {
    DispatchFixture f;
    ASSERT_TRUE(f.prepare());
    uint32_t slot;
    ASSERT_TRUE(f.take(slot));
    EXPECT_EQ(f.local()->local_ready_mask, 0u);
    const auto handle = f.local()->slots[slot].prepared;
    f.local()->slots[slot].state = SchedulerDispatchSlotState::EXECUTING;
    ASSERT_TRUE(
        (scheduler_complete_own_slot<true, true>(f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot, 0))
    );
    EXPECT_EQ(f.local()->slots[slot].state, SchedulerDispatchSlotState::FREE);
    EXPECT_TRUE(scheduler_completion_done(f.control(0)->completion_state));
    EXPECT_EQ(
        scheduler_prepared_phase(
            scheduler_prepared_control_at(f.base(), f.local()->config.prepared, handle.execution_index)->state
        ),
        SchedulerPreparedPhase::CLAIMED
    );
    EXPECT_TRUE(scheduler_prepared_already_returned(f.base(), f.local()->config.prepared, handle));
    auto stale = handle;
    EXPECT_FALSE(scheduler_return_prepared_execution(f.base(), f.local()->config.prepared, &stale));
    EXPECT_FALSE(
        scheduler_fail_prepared_execution(f.base(), f.local()->config.prepared, f.local()->worker_id(), &stale)
    );
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(f.base(), f.local(), &stale));
    EXPECT_EQ(f.local()->pending_completed, 1u);
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 0u);
    EXPECT_FALSE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot, 0));
    EXPECT_EQ(f.local()->pending_completed, 1u);
    scheduler_flush_completions(f.storage.run_control, f.local());
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerTokenDispatch, OnlyTheLastOrdinaryReturnUsesTheCanonicalRetirementProof) {
    DispatchFixture f(1, 2);
    f.storage.metadata[0].logical_block_num = 2;
    f.storage.metadata[0].flags |= SCHEDULER_TASK_SPMD;
    ASSERT_TRUE(f.prepare());
    for (uint32_t block = 0; block < 2; ++block) {
        uint32_t slot = UINT32_MAX;
        ASSERT_TRUE(f.take(slot));
        const auto handle = f.local()->slots[slot].prepared;
        EXPECT_EQ(f.local()->slots[slot].block_idx, block);
        f.local()->slots[slot].state = SchedulerDispatchSlotState::EXECUTING;
        ASSERT_TRUE((scheduler_complete_own_slot<false, true>(
            f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot, 0
        )));
        const auto raw =
            scheduler_prepared_control_at(f.base(), f.local()->config.prepared, handle.execution_index)->state;
        EXPECT_EQ(
            scheduler_prepared_phase(raw),
            block == 0 ? SchedulerPreparedPhase::RETURNED : SchedulerPreparedPhase::CLAIMED
        );
        EXPECT_EQ(scheduler_prepared_already_returned(f.base(), f.local()->config.prepared, handle), block == 1);
        EXPECT_EQ(scheduler_cached_task_proof(f.local(), 0) == SchedulerTaskProof::DONE, block == 1);
    }
    EXPECT_TRUE(scheduler_completion_done(f.control(0)->completion_state));
    EXPECT_EQ(f.local()->pending_completed, 1u);
}

TEST(SchedulerTokenDispatch, InvalidReturnPreservesPreviouslyCompletedWork) {
    DispatchFixture f(2);
    ASSERT_TRUE(f.prepare());
    uint32_t slot;
    ASSERT_TRUE(f.take(slot));
    ASSERT_TRUE(f.complete(slot));
    ASSERT_TRUE(f.take(slot));
    ASSERT_EQ(f.local()->slots[slot].state, SchedulerDispatchSlotState::READY);
    EXPECT_FALSE(scheduler_complete_own_slot(f.graph.graph(), f.base(), f.local(), f.storage.run_control, slot, 0));
    EXPECT_EQ(
        f.storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_SLOT_BINDING_INVALID)
    );
    EXPECT_EQ(f.local()->pending_completed, 1u);
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
    EXPECT_FALSE(scheduler_completion_done(f.control(1)->completion_state));
}

TEST(SchedulerTokenDispatch, UnprofiledExecutionNeedsNoProfilingStorage) {
    DispatchFixture f;
    ASSERT_TRUE(f.prepare());
    f.local()->profiling = nullptr;
    uint32_t slot;
    ASSERT_TRUE(f.take(slot));
    ASSERT_TRUE(f.complete(slot));
    EXPECT_EQ(f.local()->profiling, nullptr);
    scheduler_flush_completions(f.storage.run_control, f.local());
    EXPECT_EQ(f.storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerTokenDispatch, ReusingTheSlotKeepsEachTraceBoundToItsExecution) {
    DispatchFixture f(2);
    f.graph.executable(1, 0, {0});
    ASSERT_TRUE(f.prepare());
    auto *traces = scheduler_state_at<SchedulerTaskTrace>(f.base(), f.storage.layout.trace_cells_offset);
    uint32_t reused_slot = UINT32_MAX;
    for (uint32_t task = 0; task < 2; ++task) {
        uint32_t slot;
        ASSERT_TRUE(f.take(slot, 3));
        if (task == 0) reused_slot = slot;
        EXPECT_EQ(slot, reused_slot);
        EXPECT_EQ(f.local()->slots[slot].task_ref, task);
        EXPECT_EQ(traces[task].task_ref, task);
        EXPECT_EQ(traces[task].worker_id, 0u);
        auto &staged = f.local()->profiling->executor_traces[slot];
        staged.kernel_start_cycles = 100 + task;
        staged.kernel_end_cycles = 200 + task;
        ASSERT_TRUE(f.complete(slot, 3));
        EXPECT_EQ(traces[task].kernel_start_cycles, 100 + task);
        EXPECT_EQ(traces[task].kernel_end_cycles, 200 + task);
        EXPECT_EQ(traces[task].valid, 1u);
    }
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
}

TEST(SchedulerTokenDispatch, HostBoundCallsDoNotConsultTheMutableCallableTable) {
    DispatchFixture f(2);
    f.storage.metadata[1].kernel_ids[0] = scheduler_test::kFixtureCallableCount - 1;
    f.storage.callable_addresses[scheduler_test::kFixtureCallableCount - 1] = 0x3000;
    ASSERT_TRUE(f.prepare());
    f.storage.callable_addresses[1] = f.storage.callable_addresses[scheduler_test::kFixtureCallableCount - 1] = 0;
    const uint64_t expected[]{0x1000, 0x3000};
    for (uint32_t task = 0; task < 2; ++task) {
        uint32_t slot;
        ASSERT_TRUE(f.take(slot));
        const auto &execution = f.local()->slots[slot];
        EXPECT_EQ(execution.task_ref, task);
        EXPECT_EQ(
            scheduler_prepared_call_at(f.base(), f.local()->config.prepared, execution.prepared.execution_index)
                ->function_bin_addr,
            expected[task]
        );
        ASSERT_TRUE(f.complete(slot));
    }
}

TEST(SchedulerHostCall, InvalidCallableIsRejectedBeforePublication) {
    DispatchFixture f;
    f.storage.metadata[0].kernel_ids[0] = scheduler_test::kFixtureCallableCount;
    EXPECT_FALSE(f.prepare());
    EXPECT_EQ(scheduler_prepared_control_at(f.base(), f.local()->config.prepared, 0)->state, 0u);
}

TEST(SchedulerTokenDispatch, ACorruptLaterCallCannotRewriteAnExistingExecution) {
    DispatchFixture f(2);
    ASSERT_TRUE(f.prepare());
    auto &token = f.local()->execution_tokens[0];
    const uint64_t block = scheduler_gm_fetch_add(f.control(0)->next_block, UINT64_C(1));
    ASSERT_EQ(block, 0u);
    token.task_ref = 0;
    token.block_idx = static_cast<uint32_t>(block);
    token.subtask_slot = 0;
    token.phase = SchedulerExecutionTokenPhase::TICKETED;
    ASSERT_EQ(scheduler_node_fanin_view(f.graph.graph(), 0, &token.fanin), SchedulerGraphResult::OK);
    ASSERT_TRUE(
        scheduler_execution_token_receive(f.graph.graph(), f.base(), f.local(), f.storage.run_control, token, 0)
    );
    ASSERT_TRUE(scheduler_close_ordinary_supply(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, token.block_idx, 1
    ));
    uint32_t slot = UINT32_MAX;
    bool progress = false;
    ASSERT_TRUE(scheduler_progress_execution_tokens(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &slot, &progress
    ));
    ASSERT_NE(slot, UINT32_MAX);
    ASSERT_EQ(f.local()->slots[slot].task_ref, 0);
    const auto first = f.local()->slots[slot].prepared;
    ASSERT_EQ(f.control(1)->next_block, 0u);
    ASSERT_EQ(
        scheduler_prepared_phase(scheduler_prepared_control_at(f.base(), f.local()->config.prepared, 1)->state),
        SchedulerPreparedPhase::BUILT
    );
    scheduler_prepared_call_at(f.base(), f.local()->config.prepared, 1)->function_bin_addr = 0;
    EXPECT_FALSE(
        scheduler_refill_execution_tokens(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &progress)
    );
    EXPECT_EQ(f.local()->slots[slot].prepared.execution_index, first.execution_index);
    EXPECT_EQ(f.local()->slots[slot].prepared.claimed_state, first.claimed_state);
    EXPECT_EQ(f.local()->slots[slot].state, SchedulerDispatchSlotState::READY);
    EXPECT_NE(f.storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerTokenDispatch, HostDoneAndConfirmedFaninPrefixDoNotReleaseTheRemainingDependency) {
    DispatchFixture f(3);
    f.storage.metadata[0] = {};
    f.graph.executable(2, 0, {0, 1});
    ASSERT_TRUE(f.prepare());
    SchedulerPreparedHandle predecessor{};
    const uint64_t block = scheduler_gm_fetch_add(f.control(1)->next_block, UINT64_C(1));
    ASSERT_EQ(block, 0u);
    ASSERT_TRUE(scheduler_acquire_execution_token(f.base(), f.local()->config.prepared, 1, 1, block, 0, &predecessor));
    ASSERT_EQ(scheduler_prepared_owner(predecessor.claimed_state), 1u);
    ASSERT_TRUE(scheduler_close_ordinary_supply(
        f.graph.graph(), f.base(), f.storage.local_context(&f.storage.contexts[1]), f.storage.run_control, 1,
        static_cast<uint32_t>(block), 1
    ));
    ASSERT_FALSE(scheduler_completion_done(f.control(1)->completion_state));
    f.control(0)->completion_state = 0;
    bool progress = false;
    ASSERT_TRUE(
        scheduler_refill_execution_tokens(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &progress)
    );
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_progress_execution_tokens(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &slot, &progress
    ));
    EXPECT_EQ(slot, UINT32_MAX);
    EXPECT_EQ(scheduler_execution_token_count(f.local()), 1u);
    EXPECT_EQ(f.control(2)->next_block, 1u);
    EXPECT_EQ(f.control(0)->completion_state, 0u);
    EXPECT_EQ(f.local()->execution_tokens[0].task_ref, 2);
    EXPECT_EQ(f.local()->execution_tokens[0].next_fanin, 1u);
    EXPECT_EQ(f.local()->execution_tokens[0].fanin.count, 2u);
    ASSERT_TRUE(scheduler_return_prepared_execution(f.base(), f.local()->config.prepared, &predecessor));
    scheduler_gm_store(f.control(1)->completion_state, scheduler_completion_completed_state(1));
    ASSERT_TRUE(scheduler_progress_execution_tokens(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &slot, &progress
    ));
    ASSERT_NE(slot, UINT32_MAX);
    EXPECT_EQ(f.local()->slots[slot].task_ref, 2);
    EXPECT_EQ(f.control(2)->next_block, 1u);
    EXPECT_EQ(f.storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerTokenDispatch, TaskProofDoesNotReleaseAsyncDebtCollidingTasksOrAnotherRun) {
    DispatchFixture completed;
    ASSERT_TRUE(completed.prepare());
    scheduler_gm_store(completed.control(0)->claim_seal, UINT64_C(1));
    scheduler_gm_store(completed.control(0)->completion_state, scheduler_completion_completed_state(1));
    bool completed_supply = false;
    ASSERT_TRUE(scheduler_task_supply_closed(
        completed.graph.graph(), completed.base(), completed.local(), completed.storage.run_control, 0,
        &completed_supply
    ));
    EXPECT_TRUE(completed_supply);
    EXPECT_EQ(scheduler_cached_task_proof(completed.local(), 0), SchedulerTaskProof::DONE);
    EXPECT_EQ(
        scheduler_observe_cached_completion(completed.base(), completed.local(), 0),
        SchedulerCompletionObservation::DONE
    );

    DispatchFixture f(SCHEDULER_TASK_PROOF_CACHE_COUNT + 1);
    ASSERT_TRUE(f.prepare());
    scheduler_gm_store(f.control(0)->completion_state, scheduler_completion_encode(1, 0, 1));
    scheduler_gm_store(f.control(0)->claim_seal, UINT64_C(1));
    bool closed = false;
    ASSERT_TRUE(scheduler_task_supply_closed(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_cached_task_proof(f.local(), 0), SchedulerTaskProof::SUPPLY_CLOSED);
    EXPECT_EQ(scheduler_observe_cached_completion(f.base(), f.local(), 0), SchedulerCompletionObservation::PENDING);
    scheduler_gm_store(f.control(0)->completion_state, scheduler_completion_completed_state(1));
    EXPECT_EQ(scheduler_observe_cached_completion(f.base(), f.local(), 0), SchedulerCompletionObservation::DONE);
    scheduler_remember_task_proof(f.local(), 0, SchedulerTaskProof::SUPPLY_CLOSED);
    EXPECT_EQ(scheduler_cached_task_proof(f.local(), 0), SchedulerTaskProof::DONE);
    EXPECT_EQ(scheduler_observe_cached_completion(f.base(), f.local(), 0), SchedulerCompletionObservation::DONE);
    EXPECT_EQ(
        scheduler_observe_cached_completion(f.base(), f.local(), SCHEDULER_TASK_PROOF_CACHE_COUNT),
        SchedulerCompletionObservation::PENDING
    );
    scheduler_gm_store(f.control(SCHEDULER_TASK_PROOF_CACHE_COUNT)->claim_seal, UINT64_C(1));
    ASSERT_TRUE(scheduler_task_supply_closed(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, SCHEDULER_TASK_PROOF_CACHE_COUNT, &closed
    ));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_cached_task_proof(f.local(), 0), SchedulerTaskProof::NONE);
    scheduler_gm_store(f.control(SCHEDULER_TASK_PROOF_CACHE_COUNT)->completion_state, UINT64_C(0));
    EXPECT_EQ(
        scheduler_observe_cached_completion(f.base(), f.local(), SCHEDULER_TASK_PROOF_CACHE_COUNT),
        SchedulerCompletionObservation::INVALID
    );
    DispatchFixture next_run;
    ASSERT_TRUE(next_run.prepare());
    EXPECT_EQ(scheduler_cached_task_proof(next_run.local(), 0), SchedulerTaskProof::NONE);
    EXPECT_EQ(
        scheduler_observe_cached_completion(next_run.base(), next_run.local(), 0),
        SchedulerCompletionObservation::PENDING
    );
}

TEST(SchedulerTokenDispatch, InvalidFaninCannotGrantExecutionPermission) {
    DispatchFixture f(2);
    f.graph.executable(1, 0, {1});
    ASSERT_TRUE(f.prepare());
    // Ordered issue validates task 1 only after the valid task 0 has an owner.
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(f.take(slot));
    ASSERT_EQ(f.local()->slots[slot].task_ref, 0);
    ASSERT_TRUE(f.complete(slot));
    ASSERT_EQ(f.local()->pending_completed, 1u);
    bool progress = false;
    EXPECT_FALSE(
        scheduler_refill_execution_tokens(f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &progress)
    );
    slot = UINT32_MAX;
    EXPECT_FALSE(scheduler_progress_execution_tokens(
        f.graph.graph(), f.base(), f.local(), f.storage.run_control, 0, &slot, &progress
    ));
    EXPECT_EQ(slot, UINT32_MAX);
    EXPECT_EQ(f.local()->pending_completed, 1u);
    EXPECT_EQ(f.control(1)->next_block, 0u);
    EXPECT_FALSE(scheduler_completion_done(f.control(1)->completion_state));
    EXPECT_NE(f.storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerTokenDispatch, PredicateFailureSkipsAndMalformedPredicateStopsExecution) {
    for (bool malformed : {false, true}) {
        DispatchFixture f;
        int32_t value = 0;
        f.storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
        f.graph.predicate(
            0, malformed ? 0 : reinterpret_cast<uint64_t>(&value), sizeof(value), static_cast<uint8_t>(PredicateOp::GT),
            0
        );
        ASSERT_TRUE(f.prepare());
        uint32_t slot;
        EXPECT_EQ(f.take(slot), !malformed);
        if (malformed) {
            EXPECT_EQ(slot, UINT32_MAX);
            EXPECT_EQ(f.local()->pending_completed, 0u);
            EXPECT_NE(f.storage.run_control->scheduler_error, 0u);
        } else {
            ASSERT_NE(slot, UINT32_MAX);
            EXPECT_EQ(f.local()->slots[slot].predicate, SchedulerPredicateState::FAIL);
            ASSERT_TRUE(f.complete(slot));
            EXPECT_TRUE(scheduler_completion_done(f.control(0)->completion_state));
            EXPECT_EQ(f.local()->pending_completed, 1u);
        }
    }
}

TEST(SchedulerSlotCompletion, InitializationDoesNotPublishAnExecutableSlot) {
    SchedulerLocalState local{};
    EXPECT_EQ(local.slots[0].state, SchedulerDispatchSlotState::EMPTY);
    for (auto &slot : local.slots)
        scheduler_initialize_free_slot(&slot);
    uint32_t slot = UINT32_MAX;
    EXPECT_FALSE(scheduler_local_ready_peek(&local, &slot));
    EXPECT_FALSE(scheduler_local_ready_execute(&local, 0));
    EXPECT_EQ(local.slots[0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(local.slots[1].state, SchedulerDispatchSlotState::FREE);
}

TEST(SchedulerLocalReady, OlderDispatchPrecedesReusedLowerSlot) {
    SchedulerLocalState local{};
    for (auto &slot : local.slots)
        slot.state = SchedulerDispatchSlotState::READY;
    scheduler_local_ready_publish(&local, 0);
    scheduler_local_ready_publish(&local, 1);
    ASSERT_TRUE(scheduler_local_ready_execute(&local, 0));
    scheduler_initialize_free_slot(&local.slots[0]);
    local.slots[0].state = SchedulerDispatchSlotState::READY;
    scheduler_local_ready_publish(&local, 0);
    uint32_t head = UINT32_MAX;
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 1u);
    ASSERT_TRUE(scheduler_local_ready_execute(&local, 1));
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &head));
    EXPECT_EQ(head, 0u);
    ASSERT_TRUE(scheduler_local_ready_execute(&local, 0));
    EXPECT_EQ(local.local_ready_mask, 0u);
}

TEST(SchedulerLocalReady, InvalidSlotStateIsNotReconstructedAsReady) {
    SchedulerLocalState local{};
    local.slots[0].state = SchedulerDispatchSlotState::FREE;
    scheduler_local_ready_publish(&local, 0);
    uint32_t slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_local_ready_peek(&local, &slot));
    EXPECT_EQ(slot, 0u);
    EXPECT_FALSE(scheduler_local_ready_execute(&local, slot));
    EXPECT_EQ(local.slots[0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(local.local_ready_mask, 1u);
}

void configure_cached_cluster(FixtureStorage &storage) {
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        auto &context = storage.contexts[lane];
        context.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
        for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
            context.cluster_worker_ids[peer] = peer;
    }
}

TEST(SchedulerLocalConfig, SnapshotsImmutableConfigurationAndReloadsOnNextRun) {
    FixtureStorage storage(2, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0});
    GraphBuffer graph_buffer(2);
    storage.prepare_catalog();
    const SchedulerGraphView graph = graph_buffer.graph();
    for (uint32_t self_lane : {1u, 2u}) {
        configure_cached_cluster(storage);
        SchedulerLocalState local{};
        auto &context = storage.contexts[self_lane];
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        EXPECT_EQ(local.config.self_lane, self_lane);
        EXPECT_EQ(local.config.worker_ids[0], 0u);
        EXPECT_EQ(local.worker_id(), self_lane);
        EXPECT_EQ(local.config.runtime_worker_count, 3u);
        const uint64_t pmu_offset = context.pmu_cells_offset;
        context.pmu_cells_offset = pmu_offset + 64;
        EXPECT_EQ(local.config.pmu_cells_offset, pmu_offset);
        SchedulerLocalState next_run{};
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &next_run));
        EXPECT_EQ(next_run.config.pmu_cells_offset, pmu_offset + 64);
        EXPECT_EQ(next_run.worker_id(), self_lane);
        EXPECT_EQ(next_run.profiling, nullptr);
        EXPECT_EQ(next_run.pending_completed, 0u);
    }
}

TEST(SchedulerLocalConfig, RejectsTruncatedOffsetsAndInvalidCoreTypes) {
    FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0});
    GraphBuffer graph_buffer(1);
    storage.prepare_catalog();
    const auto graph = graph_buffer.graph();
    configure_cached_cluster(storage);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    EXPECT_EQ(local.worker_id(), UINT64_MAX);
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, nullptr));
    auto invalid_graph = graph;
    ++invalid_graph.task_count;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &invalid_graph, &local));
    auto *summary =
        scheduler_state_at<SchedulerGraphSummary>(storage.scheduler_state->base(), context.graph_summary_offset);
    const uint64_t task_catalog_offset = summary->task_catalog_offset;
    summary->task_catalog_offset = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    summary->task_catalog_offset = task_catalog_offset;
    auto *pool = scheduler_state_at<SchedulerTaskCatalogHeader>(storage.scheduler_state->base(), task_catalog_offset);
    const uint64_t prepared_offset = summary->prepared_layout_offset;
    for (const uint64_t offset : {prepared_offset, UINT64_C(0)}) {
        summary->prepared_layout_offset = offset;
        context.prepared_layout_offset = offset;
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        pool->task_count = graph.task_count + 1;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        pool->task_count = graph.task_count;
    }
    summary->prepared_layout_offset = prepared_offset;
    context.prepared_layout_offset = prepared_offset;
    const uint64_t runtime_worker_count = context.runtime_worker_count;
    for (uint64_t invalid_count : {UINT64_C(0), uint64_t{SCHEDULER_WORKER_CAPACITY + 1}}) {
        context.runtime_worker_count = invalid_count;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    }
    context.runtime_worker_count = runtime_worker_count;
    const uint64_t worker_index = context.worker_index;
    context.worker_index = runtime_worker_count;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.worker_index = worker_index;
    for (auto field :
         {&SchedulerWorkerContext::worker_contexts_offset, &SchedulerWorkerContext::task_metadata_offset}) {
        const uint64_t original = context.*field;
        context.*field = UINT64_C(1) << 32;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        context.*field = original;
    }
    storage.contexts[0].core_type = 2;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
}

TEST(SchedulerLocalConfig, RejectsInvalidTopologyBeforeMailboxAccess) {
    FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0});
    GraphBuffer graph_buffer(1);
    storage.prepare_catalog();
    const SchedulerGraphView graph = graph_buffer.graph();
    configure_cached_cluster(storage);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    context.runtime_worker_count = SCHEDULER_WORKER_CAPACITY + 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.runtime_worker_count = 3;
    context.cluster_worker_ids[1] = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.cluster_worker_ids[1] = 1;
    context.worker_index = 3;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.worker_index = 2;
    EXPECT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
}

TEST(SchedulerPredicate, DistinguishesFailedAndMalformedPredicates) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(8) int64_t value = 5;
    struct PredicateCase {
        PredicateOp op;
        int64_t pass_target;
        int64_t fail_target;
    };
    constexpr std::array<PredicateCase, 6> cases{{
        {PredicateOp::EQ, 5, 6},
        {PredicateOp::NE, 6, 5},
        {PredicateOp::GT, 4, 5},
        {PredicateOp::LT, 6, 5},
        {PredicateOp::GE, 5, 6},
        {PredicateOp::LE, 5, 4},
    }};
    constexpr std::array<uint8_t, 4> element_sizes{1, 2, 4, 8};
    for (uint8_t element_size : element_sizes) {
        for (const PredicateCase &predicate_case : cases) {
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.pass_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.fail_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::FAIL);
        }
    }

    graph.predicate(0, 0, 0, static_cast<uint8_t>(PredicateOp::NONE));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);

    graph.predicate(0, 0, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 3, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 8, UINT8_C(0xff));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value) + 1, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
}

using namespace scheduler_test::gm_mix;
class SchedulerMix : public ::testing::TestWithParam<LaneMask> {};

void arm_ordinary_mix(GmMixFixture &fixture, uint64_t profiling = 0) {
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            bool progress = false;
            ASSERT_TRUE(fixture.ordinary_service(worker, &progress, profiling));
        }
    }
}

TEST_P(SchedulerMix, PayloadAndCompletionTraceFollowEachLogicalSubtask) {
    const auto mask = GetParam();
    GmMixFixture fixture({{1, mask}}, true, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_NO_FATAL_FAILURE(arm_ordinary_mix(fixture, 3));
    std::set<uint64_t> execution_indices;
    std::set<uint32_t> subtasks;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        const auto subtask = mask.has(static_cast<SubtaskSlot>(worker)) ? worker : UINT8_MAX;
        if (subtask == UINT8_MAX) continue;
        auto *local = fixture.locals[worker];
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.subtask, subtask);
        EXPECT_TRUE(subtasks.insert(subtask).second);
        EXPECT_TRUE(execution_indices.insert(local->slots[execution.slot].prepared.execution_index).second);
        EXPECT_EQ(execution.physical_subblock, worker == 2 ? 1u : 0u);
        auto &staged = local->profiling->executor_traces[execution.slot];
        staged.kernel_start_cycles = 100 + worker;
        staged.kernel_end_cycles = 200 + worker;
        ASSERT_TRUE(
            scheduler_complete_own_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 3)
        );
        auto *traces =
            scheduler_state_at<SchedulerTaskTrace>(fixture.base(), fixture.storage.layout.trace_cells_offset);
        const auto &trace =
            traces[scheduler_trace_index_at(fixture.base(), local, 0, static_cast<SubtaskSlot>(subtask), 0)];
        EXPECT_EQ(trace.worker_id, worker);
        EXPECT_EQ(trace.kernel_start_cycles, 100 + worker);
        EXPECT_EQ(trace.valid, 1u);
        EXPECT_EQ(local->slots[execution.slot].state, SchedulerDispatchSlotState::FREE);
    }
    EXPECT_EQ(subtasks.size(), mask.count());
    bool retired = false;
    ASSERT_TRUE(fixture.ordinary_service(0, &retired));
    EXPECT_TRUE(retired);
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), mask.count());
}

TEST_P(SchedulerMix, InvalidPreparedCallPublishesNoPayloadOrCompletion) {
    GmMixFixture fixture({{1, GetParam()}}, true, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    const auto *prepared = fixture.locals[0]->config.prepared;
    for (uint64_t index = 0; index < prepared->execution_count; ++index)
        scheduler_prepared_call_at(fixture.base(), prepared, index)->function_bin_addr = 0;
    bool valid = true;
    for (uint32_t worker = 0; worker < 3 && valid; ++worker) {
        bool progress = false;
        valid = fixture.ordinary_service(worker, &progress);
    }
    EXPECT_FALSE(valid);
    for (const auto *local : fixture.locals) {
        EXPECT_EQ(local->local_ready_mask, 0u);
        EXPECT_EQ(local->pending_completed, 0u);
    }
    EXPECT_NE(fixture.run()->scheduler_error, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
}

INSTANTIATE_TEST_SUITE_P(CurrentGmMasks, SchedulerMix, ::testing::ValuesIn(kMixMasks));

}  // namespace
