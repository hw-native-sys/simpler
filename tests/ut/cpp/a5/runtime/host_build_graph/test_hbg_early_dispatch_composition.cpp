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

#include "scheduler/scheduler_scalar_policy.h"

namespace {
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = true;
};
}  // namespace

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "async_completion_plan.h"
#include "hbg_early_dispatch_test_support.h"
#include "scheduler/scheduler_async_control.h"
#include "scheduler/scheduler_async_finish.h"

namespace {

using scheduler_test::early_dispatch::Execution;
using scheduler_test::early_dispatch::Fixture;
using scheduler_test::early_dispatch::Shape;
using namespace pto2::resident;

constexpr LaneMask kAicAiv0{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};

AsyncTaskPlanInput producer_plan() {
    AsyncTaskPlanInput input;
    input.task_ref = 0;
    input.block_count = 1;
    input.active_mask = LaneMask::aic();
    input.capacities[0] = 1;
    return input;
}

// Two tasks on one cluster: an async producer whose kernel return carries a
// pending counter condition, and an ordinary consumer that pre-dispatches
// behind it. The consumer's dependency gate tracks the deferred completion,
// never the physical kernel return.
struct AsyncGateFixture {
    alignas(128) std::array<uint32_t, 8> signal{};
    AsyncCompletionHostPlan plan;
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<8> graph;
    std::array<SchedulerAsyncLocalState, 3> async{};
    std::array<AsyncWriter, 3> executions{};
    std::array<SchedulerLocalState *, 3> locals{};

    AsyncGateFixture() :
        storage(
            2, 3, 1, {}, false,
            [&](AicoreSchedulerLayout &layout) {
                std::string error;
                EXPECT_TRUE(plan_async_completion(layout.total_size, 1, 2, {producer_plan()}, &plan, &error)) << error;
                layout.total_size = plan.total_bytes;
                layout.async_layout_offset = plan.layout_offset;
            }
        ),
        graph(2) {
        EXPECT_TRUE(initialize_async_completion(base(), storage.layout.total_size, plan));
        scheduler_state_at<SchedulerAsyncLayout>(base(), plan.layout_offset)->image_bytes = storage.layout.total_size;
        scheduler_state_at<SchedulerGraphSummary>(base(), storage.layout.graph_summary_offset)->run_epoch =
            plan.layout.run_epoch;
        graph.executable(0, 0);
        graph.executable(1, 1, {0});
        auto &consumer_metadata = storage.metadata[1];
        consumer_metadata.active_mask = LaneMask::aiv0();
        consumer_metadata.kernel_ids[0] = UINT16_MAX;
        consumer_metadata.kernel_ids[1] = 1;
        consumer_metadata.kernel_ids[2] = UINT16_MAX;
        consumer_metadata.timing_slot = -1;
        storage.metadata[0].timing_slot = -1;
        storage.prepare_catalog();
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&scheduler_test::early_dispatch::counting_kernel);
        if (!storage.prepare_calls(graph.graph())) throw std::logic_error("cannot prepare fixture calls");
        for (uint32_t worker = 0; worker < 3; ++worker) {
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage.contexts[worker].cluster_worker_ids[lane] = lane;
            EXPECT_TRUE(scheduler_initialize_async(
                base(), storage.layout.total_size, plan.layout_offset, 1, 2, storage.layout.task_controls_offset,
                async[worker]
            ));
            storage.test_graphs[worker] = graph.graph();
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            locals[worker]->async = &async[worker];
        }
    }

    void *base() { return storage.scheduler_state->base(); }
    SchedulerTaskControl *control(uint64_t task) {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    bool done(uint64_t task) {
        return scheduler_completion_done(scheduler_gm_fetch_add(control(task)->completion_state, UINT64_C(0)));
    }
    bool refill(uint32_t worker) {
        bool progress = false;
        return scheduler_refill_execution_tokens(
            graph.graph(), base(), locals[worker], storage.run_control, 0, &progress
        );
    }
    bool claim(uint64_t task, uint32_t worker) {
        for (uint64_t visit = 0; visit <= 4; ++visit) {
            if (!refill(worker)) return false;
            if (scheduler_execution_token_has_task(locals[worker], task)) return true;
        }
        return scheduler_execution_token_has_task(locals[worker], task);
    }
    bool progress(uint32_t worker, uint32_t *selected) {
        bool made = false;
        return scheduler_progress_execution_tokens(
            graph.graph(), base(), locals[worker], storage.run_control, 0, selected, &made
        );
    }
    AsyncWriter start(uint32_t worker = 0) {
        auto &context = storage.prepared_global_context(0, 0, 0);
        EXPECT_TRUE(scheduler_async_begin_prebound_execution(&async[worker], 0, context, executions[worker]));
        EXPECT_EQ(context.async_present, 1u);
        return {
            reinterpret_cast<uint8_t *>(context.async_image_address), context.async_image_bytes,
            context.async_run_epoch, context.async_group_offset, context.async_group_index
        };
    }
    void register_counter(const AsyncWriter &writer) {
        const CompletionToken token{
            reinterpret_cast<uint64_t>(&signal[0]), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
        };
        EXPECT_TRUE(register_completion_condition(writer, token));
    }
    bool check(uint32_t worker) {
        bool made = false;
        SchedulerAsyncCompletion completion{};
        EXPECT_TRUE(scheduler_async_check_step(
            async[worker], worker == 0 ? 1 : 2, 8, SchedulerAsyncCompletionObserver{}, made, completion
        ));
        if (completion.result == SchedulerAsyncCompletionResult::FINISH_OWNER)
            EXPECT_TRUE(
                scheduler_async_complete_task(graph.graph(), base(), locals[worker], storage.run_control, completion)
            );
        return made;
    }
};

// The async producer's physical return with a pending condition does not
// release the pre-dispatched consumer; only the deferred FINISH does, and the
// consumer's block then executes exactly once.
TEST(HbgEarlyDispatchComposition, AsyncProducerReleasesConsumerOnlyAtDeferredFinish) {
    AsyncGateFixture fixture;
    ASSERT_TRUE(fixture.claim(0, 0));
    EXPECT_EQ(fixture.control(0)->claim_seal, 1u);
    ASSERT_TRUE(fixture.claim(1, 1));
    auto *consumer = fixture.locals[1];
    ASSERT_TRUE(scheduler_execution_token_has_task(consumer, 1));
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress(1, &selected));
    EXPECT_EQ(selected, UINT32_MAX);
    EXPECT_EQ(hbg_test_eligible_mask(consumer), 0u);

    // Run the producer: register, seal and physically return with the counter
    // condition still pending.
    ASSERT_TRUE(fixture.progress(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    auto *producer = fixture.locals[0];
    producer->execution_at(selected).state = SchedulerDispatchSlotState::EXECUTING;
    const auto writer = fixture.start();
    fixture.register_counter(writer);
    ASSERT_TRUE(scheduler_async_seal_execution(&fixture.async[0], 0, 0, 0, fixture.executions[0]));
    ASSERT_TRUE(
        scheduler_complete_lane_execution<false>(
            fixture.graph.graph(), fixture.base(), producer, fixture.storage.run_control, selected, 0
        )
    );
    // The kernel returned, but completion is deferred: the consumer's gate
    // must still hold.
    EXPECT_FALSE(fixture.done(0));
    for (uint32_t poll = 0; poll < 4; ++poll) {
        ASSERT_TRUE(fixture.progress(1, &selected));
        EXPECT_EQ(selected, UINT32_MAX);
        EXPECT_EQ(hbg_test_eligible_mask(consumer), 0u);
    }

    // Only the signal lets the deferred close publish DONE, and only now may
    // the consumer enter.
    fixture.signal[0] = 1;
    ASSERT_TRUE(fixture.check(1));
    ASSERT_TRUE(fixture.done(0));

    ASSERT_TRUE(fixture.progress(1, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    const uint64_t runs_before = scheduler_test::early_dispatch::kernel_run_count();
    consumer->execution_at(selected).state = SchedulerDispatchSlotState::EXECUTING;
    const auto index = consumer->execution_at(selected).prepared.execution_index;
    auto *call = scheduler_prepared_call_at(fixture.base(), consumer->config.prepared, index);
    scheduler_execute_protocol_task(
        fixture.base(), consumer, selected, call->function_bin_addr, call->args,
        scheduler_execution_binding_at(fixture.base(), consumer->config.prepared, index)->local_context
    );
    EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), runs_before + 1);
    ASSERT_TRUE(
        scheduler_complete_lane_execution<false>(
            fixture.graph.graph(), fixture.base(), consumer, fixture.storage.run_control, selected, 0
        )
    );
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(1)->completion_state, UINT64_C(0))), 1u
    );
    EXPECT_EQ(scheduler_gm_query(fixture.storage.run_control->scheduler_error), 0u);
}

// A sync-start SPMD consumer pre-dispatches every member block behind the
// running producer; no member enters before the dependency completes, and
// each member executes exactly once.
TEST(HbgEarlyDispatchComposition, SyncStartMembersPreDispatchBehindTheDependencyGate) {
    Fixture fixture({{1, LaneMask::aic()}, {2, LaneMask::aiv0(), true}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    EXPECT_EQ(fixture.control(1)->claim_seal, 0u);
    ASSERT_TRUE(fixture.prefetch(1, 2));
    EXPECT_EQ(fixture.control(1)->claim_seal, 2u);
    for (const uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(scheduler_execution_token_has_task(fixture.locals[worker], 1));
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(fixture.progress_tokens(worker, &selected));
        EXPECT_EQ(selected, UINT32_MAX);
        EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[worker]), 0u);
    }

    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_TRUE(fixture.complete(producer));

    std::set<uint32_t> blocks;
    for (const uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.progress_tokens(worker, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        Execution member{};
        ASSERT_TRUE(fixture.begin(worker, &member));
        ASSERT_EQ(member.task, 1);
        EXPECT_TRUE(blocks.insert(member.block).second);
        ASSERT_TRUE(fixture.complete(member));
    }
    EXPECT_EQ(blocks, (std::set<uint32_t>{0, 1}));
    EXPECT_EQ(fixture.completions(1), 2u);
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// Candidate admission: only ordinary executable tasks use execution-token
// pre-dispatch. The refill claim consults this predicate before issuing any
// ticket, so MIX and inline work can never occupy an execution token.
TEST(HbgEarlyDispatchComposition, AdmissionMatrixExcludesMixAndInlineFromPredispatch) {
    EXPECT_TRUE(scheduler_task_uses_execution_tokens(SCHEDULER_TASK_EXECUTABLE));
    EXPECT_TRUE(scheduler_task_uses_execution_tokens(SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD));
    EXPECT_TRUE(scheduler_task_uses_execution_tokens(
        SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START
    ));
    EXPECT_FALSE(scheduler_task_uses_execution_tokens(SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX));
    EXPECT_FALSE(scheduler_task_uses_execution_tokens(SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE));
    EXPECT_FALSE(scheduler_task_uses_execution_tokens(SCHEDULER_TASK_INLINE));

    // An ordinary graph around the admission boundary still drains cleanly.
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prepare());
    ASSERT_TRUE(fixture.drain(64));
    EXPECT_TRUE(fixture.done(0));
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

}  // namespace
