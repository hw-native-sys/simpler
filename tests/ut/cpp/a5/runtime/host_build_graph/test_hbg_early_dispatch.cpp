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

#include <cstdint>
#include <utility>
#include <vector>

#include "hbg_early_dispatch_test_support.h"

namespace {

using scheduler_test::early_dispatch::Execution;
using scheduler_test::early_dispatch::Fixture;
using scheduler_test::early_dispatch::Shape;

constexpr uint32_t kAicProducer = 0;
constexpr uint32_t kAivConsumer = 1;

// The consumer claims its ticket and binds its prepared Call while the
// producer is still running; the kernel starts only after the producer's
// completion is observed, and the block executes exactly once.
TEST(HbgEarlyDispatch, ConsumerIsPreparedBeforeProducerCompletesAndEntersOnlyAfterRelease) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(kAicProducer, kAicProducer));
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 1u);
    EXPECT_FALSE(fixture.done(0));

    ASSERT_TRUE(fixture.prefetch(1, kAivConsumer));
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.control(1)->claim_seal, 1u);
    auto *consumer = fixture.locals[kAivConsumer];
    ASSERT_TRUE(scheduler_execution_token_has_task(consumer, 1));
    const auto &token = [&]() -> const SchedulerExecutionToken & {
        for (const auto &owned : consumer->execution_tokens)
            if (owned.task_ref == 1) return owned;
        ADD_FAILURE() << "consumer token missing";
        return consumer->execution_tokens[0];
    }();
    EXPECT_EQ(token.phase, SchedulerExecutionTokenPhase::WAIT_FANIN);
    EXPECT_EQ(token.state, SchedulerDispatchSlotState::FILLING);
    EXPECT_EQ(token.predicate, SchedulerPredicateState::UNEVALUATED);
    // Pre-dispatch state: the Call is claimed and its payload received while
    // the producer has not completed.
    EXPECT_EQ(scheduler_prepared_owner(token.prepared.claimed_state), consumer->worker_id());
    EXPECT_EQ(scheduler_prepared_phase(token.prepared.claimed_state), SchedulerPreparedPhase::CLAIMED);

    for (uint32_t poll = 0; poll < 8; ++poll) {
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(fixture.progress_tokens(kAivConsumer, &selected));
        EXPECT_EQ(selected, UINT32_MAX);
        EXPECT_EQ(hbg_test_eligible_mask(consumer), 0u);
        EXPECT_EQ(token.next_fanin, 0u);
    }
    Execution premature{};
    EXPECT_FALSE(fixture.begin(kAivConsumer, &premature));

    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(kAicProducer, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(kAicProducer, &producer));
    ASSERT_EQ(producer.task, 0);
    ASSERT_TRUE(fixture.complete(producer));
    ASSERT_TRUE(fixture.done(0));

    ASSERT_TRUE(fixture.progress_tokens(kAivConsumer, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution consumer_execution{};
    ASSERT_TRUE(fixture.begin(kAivConsumer, &consumer_execution));
    ASSERT_EQ(consumer_execution.task, 1);
    ASSERT_NE(consumer_execution.function, 0u);
    ASSERT_TRUE(fixture.complete(consumer_execution));

    ASSERT_EQ(fixture.entries.size(), 2u);
    EXPECT_EQ(fixture.entries[0], std::make_pair(int64_t{0}, 0u));
    EXPECT_EQ(fixture.entries[1], std::make_pair(int64_t{1}, 0u));
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// A consumer with two predecessors is released only when every fanin edge has
// observed completion, in either completion order.
TEST(HbgEarlyDispatch, MultiFaninConsumerWaitsForEveryProducerInEitherCompletionOrder) {
    for (const uint32_t first : {0u, 1u}) {
        Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
        fixture.dependency(2, {0, 1});
        ASSERT_TRUE(fixture.prefetch(0, 0));
        ASSERT_TRUE(fixture.prefetch(1, 3));
        ASSERT_TRUE(fixture.prefetch(2, 1));
        EXPECT_EQ(fixture.control(2)->next_block, 1u);

        const uint32_t second = first == 0 ? 1 : 0;
        const uint32_t first_worker = first == 0 ? 0 : 3;
        const uint32_t second_worker = first == 0 ? 3 : 0;
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(fixture.progress_tokens(first_worker, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        Execution producer{};
        ASSERT_TRUE(fixture.begin(first_worker, &producer));
        ASSERT_EQ(producer.task, static_cast<int64_t>(first));
        ASSERT_TRUE(fixture.complete(producer));

        ASSERT_TRUE(fixture.progress_tokens(1, &selected));
        EXPECT_EQ(selected, UINT32_MAX);
        EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[1]), 0u);
        Execution premature{};
        EXPECT_FALSE(fixture.begin(1, &premature));

        ASSERT_TRUE(fixture.progress_tokens(second_worker, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        ASSERT_TRUE(fixture.begin(second_worker, &producer));
        ASSERT_EQ(producer.task, static_cast<int64_t>(second));
        ASSERT_TRUE(fixture.complete(producer));

        ASSERT_TRUE(fixture.progress_tokens(1, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        Execution consumer{};
        ASSERT_TRUE(fixture.begin(1, &consumer));
        ASSERT_EQ(consumer.task, 2);
        ASSERT_TRUE(fixture.complete(consumer));
        EXPECT_EQ(fixture.completions(2), 1u);
        EXPECT_TRUE(fixture.done(2));
        EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
    }
}

// The producer finished before the consumer was ever claimed: the claim
// observes the settled fanin and becomes executable on the same pass.
TEST(HbgEarlyDispatch, CompletionBeforePredispatchReleasesImmediately) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_TRUE(fixture.complete(producer));
    ASSERT_TRUE(fixture.done(0));

    ASSERT_TRUE(fixture.prefetch(1, 1));
    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution consumer{};
    ASSERT_TRUE(fixture.begin(1, &consumer));
    ASSERT_EQ(consumer.task, 1);
    ASSERT_TRUE(fixture.complete(consumer));
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// The producer completes after the claim but before the consumer's first
// fanin observation: the first poll releases the prepared block.
TEST(HbgEarlyDispatch, CompletionDuringPredispatchWindowReleasesOnFirstObservation) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    auto *consumer = fixture.locals[1];
    ASSERT_TRUE(scheduler_execution_token_has_task(consumer, 1));

    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_TRUE(fixture.complete(producer));

    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution consumer_execution{};
    ASSERT_TRUE(fixture.begin(1, &consumer_execution));
    ASSERT_EQ(consumer_execution.task, 1);
    ASSERT_TRUE(fixture.complete(consumer_execution));
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// The predicate operand is read only after the dependencies complete: a value
// published during the pre-dispatch window decides the entry, and a failing
// predicate still fulfils exactly one return.
TEST(HbgEarlyDispatch, PredicateIsEvaluatedOnlyAfterDependenciesComplete) {
    for (const bool passes : {true, false}) {
        Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0(), false, SCHEDULER_TASK_HAS_PREDICATE}});
        fixture.dependency(1, {0});
        alignas(64) uint32_t operand = 0;
        fixture.graph.predicate(1, reinterpret_cast<uint64_t>(&operand), 4, 1, 100);
        ASSERT_TRUE(fixture.prefetch(0, 0));
        ASSERT_TRUE(fixture.prefetch(1, 1));
        auto *consumer = fixture.locals[1];
        ASSERT_TRUE(scheduler_execution_token_has_task(consumer, 1));
        uint32_t selected = UINT32_MAX;
        ASSERT_TRUE(fixture.progress_tokens(1, &selected));
        EXPECT_EQ(selected, UINT32_MAX);
        for (const auto &owned : consumer->execution_tokens)
            if (owned.task_ref == 1) EXPECT_EQ(owned.predicate, SchedulerPredicateState::UNEVALUATED);

        if (passes) operand = 100;
        ASSERT_TRUE(fixture.progress_tokens(0, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        Execution producer{};
        ASSERT_TRUE(fixture.begin(0, &producer));
        ASSERT_TRUE(fixture.complete(producer));

        const uint64_t runs_before = scheduler_test::early_dispatch::kernel_run_count();
        ASSERT_TRUE(fixture.progress_tokens(1, &selected));
        ASSERT_NE(selected, UINT32_MAX);
        Execution consumer_execution{};
        ASSERT_TRUE(fixture.begin(1, &consumer_execution));
        ASSERT_EQ(consumer_execution.task, 1);
        if (passes) {
            EXPECT_NE(consumer_execution.function, 0u);
            EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), runs_before + 1);
        } else {
            EXPECT_EQ(consumer_execution.function, 0u);
            EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), runs_before);
        }
        ASSERT_TRUE(fixture.complete(consumer_execution));
        EXPECT_EQ(fixture.completions(1), 1u);
        EXPECT_TRUE(fixture.done(1));
        EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
    }
}

// Cross-run safety relies on the previous run's workers having fully exited
// before the same allocation is re-initialized (the retirement fence in
// RUNTIME_LOGIC.md). Once the drained run is fully retired, its handles and
// completions cannot release work in the next run: reinitialization resets
// claims, the new epoch re-declares the same A -> B edge from the launch
// description, and the graph repeats the full pre-dispatch lifecycle.
TEST(HbgEarlyDispatch, ReinitializedRunRejectsStaleHandlesAfterFullRetirement) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    auto *consumer = fixture.locals[1];
    SchedulerPreparedHandle stale{};
    for (const auto &owned : consumer->execution_tokens)
        if (owned.task_ref == 1) stale = owned.prepared;
    ASSERT_NE(stale.execution_index, UINT64_MAX);
    ASSERT_TRUE(fixture.drain(64));
    ASSERT_TRUE(fixture.done(1));
    const uint64_t old_epoch = fixture.run_epoch;

    // The retired generation no longer accepts the drained run's handle.
    EXPECT_TRUE(scheduler_prepared_already_returned(fixture.base(), fixture.locals[1]->config.prepared, stale));
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.locals[1], &stale));
    EXPECT_FALSE(scheduler_prepared_layout_valid(
        fixture.base(), fixture.storage.layout.total_size, fixture.storage.layout.prepared_layout_offset,
        fixture.storage.layout.task_count, old_epoch + 1
    ));

    fixture.configure(old_epoch + 1);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(fixture.completions(0), 0u);
    EXPECT_EQ(fixture.completions(1), 0u);
    EXPECT_TRUE(scheduler_prepared_layout_valid(
        fixture.base(), fixture.storage.layout.total_size, fixture.storage.layout.prepared_layout_offset,
        fixture.storage.layout.task_count, old_epoch + 1
    ));
    // The stale handle from the previous epoch cannot cancel or fail the fresh
    // BUILT record: the transition requires the exact live state. This check
    // covers the moment before the new run re-claims the record.
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.locals[1], &stale));

    // The new run re-declares the same A -> B edge (configure() rebuilds the
    // graph from the launch description, which drops the dependency list, so
    // the launch restates it) and repeats the pre-dispatch lifecycle: B is
    // claimed while A is incomplete, cannot enter its kernel, and enters
    // exactly once after A completes.
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    ASSERT_TRUE(scheduler_execution_token_has_task(fixture.locals[1], 1));
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    EXPECT_EQ(selected, UINT32_MAX);
    EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[1]), 0u);
    Execution premature{};
    EXPECT_FALSE(fixture.begin(1, &premature));
    ASSERT_TRUE(fixture.progress_tokens(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_EQ(producer.task, 0);
    ASSERT_TRUE(fixture.complete(producer));
    ASSERT_TRUE(fixture.done(0));

    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    const uint64_t runs_before = scheduler_test::early_dispatch::kernel_run_count();
    Execution consumer_execution{};
    ASSERT_TRUE(fixture.begin(1, &consumer_execution));
    ASSERT_EQ(consumer_execution.task, 1);
    EXPECT_NE(consumer_execution.function, 0u);
    EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), runs_before + 1);
    ASSERT_TRUE(fixture.complete(consumer_execution));
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_TRUE(fixture.done(0));
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// A failed run cancels its owned unstarted pre-dispatch to FAILED without
// manufacturing a return, a completion contribution or a publication. The next
// epoch reuses the allocation, keeps the declared A -> B edge, and B still
// gates on A in the repeated pre-dispatch lifecycle.
TEST(HbgEarlyDispatch, FailedRunReclaimsUnstartedWorkWithoutManufacturingAReturn) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    auto *consumer = fixture.locals[1];
    SchedulerPreparedHandle owned_handle{};
    for (const auto &owned : consumer->execution_tokens)
        if (owned.task_ref == 1) owned_handle = owned.prepared;
    ASSERT_NE(owned_handle.execution_index, UINT64_MAX);

    scheduler_publish_error(fixture.run(), 1);
    ASSERT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), consumer));
    EXPECT_EQ(scheduler_execution_token_count(consumer), 0u);
    auto *control =
        scheduler_prepared_control_at(fixture.base(), consumer->config.prepared, owned_handle.execution_index);
    const uint64_t word = scheduler_gm_query(control->state);
    EXPECT_EQ(scheduler_prepared_phase(word), SchedulerPreparedPhase::FAILED);
    EXPECT_EQ(fixture.completions(1), 0u);
    EXPECT_FALSE(fixture.done(1));
    EXPECT_EQ(fixture.control(1)->next_block, 1u);

    // Another worker owns nothing here; its cancellation must not move the
    // failed handle to RETURNED.
    SchedulerPreparedHandle foreign = owned_handle;
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.locals[2], &foreign));
    EXPECT_EQ(scheduler_gm_query(control->state), word);

    fixture.configure(fixture.run_epoch + 1);
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
    // The reused allocation keeps the declared A -> B edge: B pre-dispatches
    // behind A, cannot enter while A is incomplete, and enters exactly once
    // after A completes.
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    ASSERT_TRUE(scheduler_execution_token_has_task(fixture.locals[1], 1));
    uint32_t selected = UINT32_MAX;
    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    EXPECT_EQ(selected, UINT32_MAX);
    Execution premature{};
    EXPECT_FALSE(fixture.begin(1, &premature));

    ASSERT_TRUE(fixture.progress_tokens(0, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    Execution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_EQ(producer.task, 0);
    ASSERT_TRUE(fixture.complete(producer));
    ASSERT_TRUE(fixture.done(0));

    ASSERT_TRUE(fixture.progress_tokens(1, &selected));
    ASSERT_NE(selected, UINT32_MAX);
    const uint64_t runs_before = scheduler_test::early_dispatch::kernel_run_count();
    Execution consumer_execution{};
    ASSERT_TRUE(fixture.begin(1, &consumer_execution));
    ASSERT_EQ(consumer_execution.task, 1);
    EXPECT_NE(consumer_execution.function, 0u);
    EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), runs_before + 1);
    ASSERT_TRUE(fixture.complete(consumer_execution));
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_TRUE(fixture.done(0));
    EXPECT_TRUE(fixture.done(1));
}

}  // namespace
