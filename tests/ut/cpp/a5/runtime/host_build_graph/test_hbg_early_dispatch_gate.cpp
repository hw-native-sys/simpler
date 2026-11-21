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
#include <set>
#include <utility>
#include <vector>

#include "hbg_early_dispatch_test_support.h"
#include "scheduler/scheduler_issue.h"

namespace {

using scheduler_test::early_dispatch::Execution;
using scheduler_test::early_dispatch::Fixture;
using scheduler_test::early_dispatch::Shape;

// Wide fan-out: 48 independent producer/consumer gates plus independent ready
// work. Pre-dispatch occupancy stays bounded by the per-worker execution-token
// capacity, every consumer's ticket follows its producer's closed supply,
// producers complete under full consumer pressure, and the whole graph drains
// exactly once inside an explicit turn budget.
TEST(HbgEarlyDispatchGate, WideFanoutBoundedOccupancyProducerProgressAndExactDrain) {
    constexpr uint64_t kGates = 48;
    std::vector<Shape> shapes(2 * kGates, {1, LaneMask::aic()});
    for (uint64_t gate = 0; gate < kGates; ++gate)
        shapes[kGates + gate] = {1, LaneMask::aiv0()};
    Fixture fixture(shapes);
    for (uint64_t gate = 0; gate < kGates; ++gate)
        fixture.dependency(kGates + gate, {static_cast<int32_t>(gate)});
    ASSERT_TRUE(fixture.prepare());

    std::set<uint64_t> completed;
    uint64_t max_waiting_total = 0;
    const auto check_invariants = [&]() {
        uint64_t waiting_total = 0;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            EXPECT_LE(fixture.held_tokens(worker), SCHEDULER_EXECUTION_TOKEN_COUNT);
            EXPECT_LE(fixture.waiting_tokens(worker), SCHEDULER_EXECUTION_TOKEN_COUNT);
            waiting_total += fixture.waiting_tokens(worker);
        }
        // A consumer ticket can only exist after its producer's supply closed.
        for (uint64_t gate = 0; gate < kGates; ++gate)
            if (fixture.control(kGates + gate)->next_block != 0)
                EXPECT_EQ(fixture.control(gate)->claim_seal, 1u) << "consumer " << gate << " claimed early";
        max_waiting_total = std::max(max_waiting_total, waiting_total);
        EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
    };

    // Stage without completing anything: consumers pre-dispatch behind
    // claimed-but-running producers until every worker's window is full.
    for (uint64_t round = 0; round < 4 * shapes.size(); ++round) {
        bool any = false;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            bool progress = false;
            ASSERT_TRUE(fixture.refill(worker, &progress));
            any = any || progress;
        }
        check_invariants();
        if (!any) break;
    }
    // Occupancy pressure was actually exercised: consumers sit in WAIT_FANIN
    // while no producer has completed.
    EXPECT_GE(max_waiting_total, 4u);
    EXPECT_LE(max_waiting_total, 4 * SCHEDULER_EXECUTION_TOKEN_COUNT);

    const uint64_t total = fixture.expected_entries();
    ASSERT_EQ(total, 2 * kGates);
    uint64_t drained = 0;
    for (uint64_t turn = 0; turn < total * 8 && drained < total; ++turn) {
        for (uint32_t worker = 0; worker < 6 && drained < total; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            Execution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.task >= static_cast<int64_t>(kGates)) {
                EXPECT_TRUE(completed.count(static_cast<uint64_t>(execution.task) - kGates) != 0)
                    << "consumer entered before its producer completed";
            }
            ASSERT_TRUE(fixture.complete(execution));
            EXPECT_TRUE(completed.insert(static_cast<uint64_t>(execution.task)).second)
                << "duplicate entry for task " << execution.task;
            ++drained;
        }
        check_invariants();
    }
    ASSERT_EQ(drained, total);
    EXPECT_EQ(fixture.entries.size(), total);
    EXPECT_EQ(scheduler_test::early_dispatch::kernel_run_count(), total);
    for (uint64_t task = 0; task < 2 * kGates; ++task) {
        EXPECT_TRUE(fixture.done(task));
        EXPECT_EQ(fixture.completions(task), 1u);
    }
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

// Independent ready work is never starved by a full window of waiting
// consumers: roots and producers drain first, consumers follow their
// producers, and every block still executes exactly once.
TEST(HbgEarlyDispatchGate, ReadyWorkProgressesWhileConsumersHoldFullWaitingWindows) {
    constexpr uint64_t kGates = 8;
    std::vector<Shape> shapes;
    for (uint64_t i = 0; i < kGates; ++i)
        shapes.push_back({1, LaneMask::aic()});
    for (uint64_t i = 0; i < kGates; ++i)
        shapes.push_back({1, LaneMask::aiv0()});
    for (uint64_t i = 0; i < 2; ++i)
        shapes.push_back({1, LaneMask::aiv0()});
    Fixture fixture(shapes);
    for (uint64_t gate = 0; gate < kGates; ++gate)
        fixture.dependency(kGates + gate, {static_cast<int32_t>(gate)});
    ASSERT_TRUE(fixture.prepare());
    std::set<uint64_t> completed;
    const uint64_t total = fixture.expected_entries();
    ASSERT_EQ(total, 2 * kGates + 2);
    uint64_t drained = 0;
    for (uint64_t turn = 0; turn < total * 8 && drained < total; ++turn) {
        for (uint32_t worker = 0; worker < 6 && drained < total; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            Execution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            const auto task = static_cast<uint64_t>(execution.task);
            if (task >= kGates && task < 2 * kGates) EXPECT_TRUE(completed.count(task - kGates) != 0);
            ASSERT_TRUE(fixture.complete(execution));
            EXPECT_TRUE(completed.insert(task).second);
            ++drained;
        }
        EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
    }
    ASSERT_EQ(drained, total);
    for (uint64_t task = 0; task < shapes.size(); ++task)
        EXPECT_TRUE(fixture.done(task));
}

// Over-claims, forged supply seals, foreign-owner cancellation and double
// retirement are all rejected closed instead of releasing extra work.
TEST(HbgEarlyDispatchGate, OverclaimForgedSupplyAndStaleOwnershipAreRejected) {
    Fixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    fixture.dependency(1, {0});
    ASSERT_TRUE(fixture.prefetch(0, 0));
    ASSERT_TRUE(fixture.prefetch(1, 1));
    // The losing worker cannot take the same block: the ticket is irrevocable
    // and exhaustion suppresses the retry without an error.
    const bool progress_before = scheduler_execution_token_has_task(fixture.locals[2], 1);
    EXPECT_FALSE(progress_before);
    ASSERT_TRUE(fixture.refill(2));
    EXPECT_FALSE(scheduler_execution_token_has_task(fixture.locals[2], 1));
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);

    auto *consumer = fixture.locals[1];
    SchedulerPreparedHandle handle{};
    for (const auto &owned : consumer->execution_tokens)
        if (owned.task_ref == 1) handle = owned.prepared;
    ASSERT_NE(handle.execution_index, UINT64_MAX);
    // A foreign worker cannot cancel or return the owned handle.
    SchedulerPreparedHandle foreign = handle;
    EXPECT_FALSE(scheduler_cancel_unstarted_handle(fixture.base(), fixture.locals[2], &foreign));
    EXPECT_EQ(scheduler_prepared_phase(handle.claimed_state), SchedulerPreparedPhase::CLAIMED);

    // A forged seal that disagrees with the block count fails closed. The
    // check runs on a worker that has never issued this task, so no cached
    // supply proof masks the read.
    const uint64_t live_seal = fixture.control(1)->claim_seal;
    fixture.control(1)->claim_seal = 7;
    bool closed = false;
    EXPECT_FALSE(scheduler_task_supply_closed(
        fixture.graph.graph(), fixture.base(), fixture.locals[4], fixture.run(), 1, &closed
    ));
    EXPECT_FALSE(closed);
    fixture.control(1)->claim_seal = live_seal;
    scheduler_gm_store(fixture.run()->scheduler_error, UINT64_C(0));
    for (auto &worker_error : fixture.run()->worker_errors)
        scheduler_gm_store(worker_error.status, UINT64_C(0));
    scheduler_gm_store(fixture.run()->error_claimed, UINT64_C(0));

    // The drain retires the block exactly once; the retired generation is
    // recognizable from the completion accounting.
    ASSERT_TRUE(fixture.drain(64));
    EXPECT_TRUE(scheduler_prepared_already_returned(fixture.base(), fixture.locals[1]->config.prepared, handle));
    EXPECT_EQ(fixture.completions(1), 1u);
    EXPECT_TRUE(fixture.done(1));
    EXPECT_EQ(scheduler_gm_query(fixture.run()->scheduler_error), 0u);
}

}  // namespace
