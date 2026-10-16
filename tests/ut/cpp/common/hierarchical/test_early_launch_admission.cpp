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

/**
 * Which staged successor may launch its device work while its predecessor is
 * still executing.
 *
 * The question is *when no further dispatch of the predecessor can be issued*,
 * and the answer is three facts together: its submission is closed, its
 * acceptance count has reached zero, and it is not terminal. These tests pin
 * each of the three as load-bearing, the wake that announces a new
 * authorization, and the sustained N/S/T/R refill the whole thing exists for.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "call_config.h"
#include "orchestrator.h"
#include "ring.h"
#include "scope.h"
#include "task_args.h"
#include "tensormap.h"
#include "types.h"

namespace {

// One orchestrator wired to real ready queues, an allocator and a scope, with
// all three budgets configured independently and the scheduler wake counted.
struct EarlyLaunchHarness {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    std::atomic<int> wakes{0};

    EarlyLaunchHarness(uint32_t depth, uint32_t launch_depth) {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, nullptr, [this] {
            ++wakes;
        });
        orch.configure_pipeline_depth(depth, depth, launch_depth);
    }

    ~EarlyLaunchHarness() { allocator.shutdown(); }

    // One OUTPUT-tensor NEXT_LEVEL single, which is the dispatch shape the
    // staging path admits and therefore the only shape a joined run has.
    SubmitResult submit_one(uint64_t buffer_id) {
        CallConfig cfg;
        TaskArgs args;
        Tensor t{};
        t.buffer.backend_kind = static_cast<uint8_t>(BackendKind::POSIX_SHM);
        t.buffer.access = static_cast<uint8_t>(AccessMode::READWRITE);
        t.buffer.nbytes = 1;
        t.buffer.identity.buffer_id = buffer_id;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.strides[0] = 1;
        t.dtype = DataType::UINT8;
        args.add_tensor(t, TensorArgType::OUTPUT);
        CallableIdentity callable;
        callable.digest.fill(53);
        return orch.submit_next_level(callable, args, cfg, 0);
    }

    struct Run {
        RunId id{INVALID_RUN_ID};
        SubmitResult task{};
    };

    /** A run whose one dispatch is published and whose submission is closed. */
    Run build_closed_run(uint64_t buffer_id) {
        Run run;
        run.id = orch.begin_run();
        run.task = submit_one(buffer_id);
        orch.close_run_submission(run.id);
        return run;
    }

    /** That run's dispatch reaching an endpoint outcome. */
    void accept(const Run &run) { orch.mark_task_accepted(run.task.task_slot); }

    const TaskSlotState &state(const Run &run) { return *allocator.slot_state(run.task.task_slot); }

    /** Drive a run's only task to completion so the run retires. */
    void complete(const Run &run) {
        allocator.slot_state(run.task.task_slot)->state.store(TaskState::COMPLETED, std::memory_order_release);
        EXPECT_TRUE(orch.on_consumed(run.task.task_slot));
    }
};

TEST(EarlyLaunchAdmission, DepthOneAuthorizesNothingHoweverReadyThePairIs) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/1);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    h.accept(first);

    ASSERT_EQ(h.orch.active_run_id(), first.id);
    ASSERT_EQ(h.orch.preparable_run_id(), second.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
}

TEST(EarlyLaunchAdmission, DistinctSubmissionsOwnIdentityAndLease) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(0xA001);
    const auto second = h.build_closed_run(0xA002);

    const Tensor &first_tensor = h.state(first).task_args.tensor(0);
    const Tensor &second_tensor = h.state(second).task_args.tensor(0);
    EXPECT_NE(first_tensor.buffer.identity.buffer_id, second_tensor.buffer.identity.buffer_id);

    EXPECT_NE(first.id, second.id);
    EXPECT_NE(first.task.task_slot, second.task.task_slot);
    const PipelineSlotLease first_lease = h.state(first).pipeline_lease;
    const PipelineSlotLease second_lease = h.state(second).pipeline_lease;
    ASSERT_NE(first_lease.generation, 0u);
    ASSERT_NE(second_lease.generation, 0u);
    EXPECT_NE(first_lease.slot_id, second_lease.slot_id);

    h.accept(first);
    EXPECT_EQ(h.orch.early_launch_run_id(), second.id);
    h.accept(second);

    h.complete(first);
    h.complete(second);
    h.orch.release_run(first.id);
    h.orch.release_run(second.id);
}

TEST(EarlyLaunchAdmission, ClosedSubmissionAloneIsNotTheBoundary) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);

    // The close ends the build declaration only. This run's published dispatch
    // has not reached an endpoint outcome, so another dispatch of it can still
    // be issued and a successor must not be ordered behind it.
    ASSERT_EQ(h.orch.preparable_run_id(), second.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);

    h.accept(first);
    EXPECT_EQ(h.orch.early_launch_run_id(), second.id);
}

TEST(EarlyLaunchAdmission, AZeroAcceptanceCountAloneIsNotTheBoundary) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const RunId first = h.orch.begin_run();

    // Before anything is published the count reads zero, which says nothing
    // about what this run is still going to declare.
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
    const SubmitResult task = h.submit_one(1);
    h.orch.close_run_submission(first);
    const auto second = h.build_closed_run(2);
    ASSERT_EQ(h.orch.preparable_run_id(), second.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);

    h.orch.mark_task_accepted(task.task_slot);
    EXPECT_EQ(h.orch.early_launch_run_id(), second.id);
}

TEST(EarlyLaunchAdmission, AFailedHeadDoesNotOpenTheGate) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);

    // An error recorded against the head says its remaining work will not
    // produce the completion a successor would be ordered behind. This is the
    // path an endpoint failure takes, which is how a head acquires one while
    // its dispatch still reports an outcome.
    h.orch.report_task_error(first.task.task_slot, "head failed");
    h.accept(first);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
}

TEST(EarlyLaunchAdmission, AuthorizationWakesTheSchedulerExactlyOncePerNewCandidate) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    const int wakes_before = h.wakes;

    h.accept(first);
    ASSERT_EQ(h.orch.early_launch_run_id(), second.id);
    const int wakes_after_authorizing = h.wakes;
    EXPECT_GT(wakes_after_authorizing, wakes_before);

    // Re-deriving the same answer is not a new authorization, so it owes no
    // wake: a spurious one would have the scheduler re-run a dispatch round
    // for a decision it has already acted on.
    EXPECT_EQ(h.orch.early_launch_run_id(), second.id);
    EXPECT_EQ(h.wakes, wakes_after_authorizing);
}

TEST(EarlyLaunchAdmission, RetirementWithdrawsTheAuthorizationTheRetiredHeadGranted) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);
    const auto first = h.build_closed_run(1);
    const auto second = h.build_closed_run(2);
    h.accept(first);
    ASSERT_EQ(h.orch.early_launch_run_id(), second.id);

    h.complete(first);
    // The authorized run is now the head itself. Naming it still would be a
    // stale identity: it is no longer a successor of anything, and the run it
    // was ordered behind is gone.
    ASSERT_EQ(h.orch.active_run_id(), second.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
}

TEST(EarlyLaunchAdmission, SustainedRefillAuthorizesEachSuccessorInTurn) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);

    // N and S: N active, S staged behind it, S authorized once N's dispatch
    // reaches an endpoint outcome.
    const auto n = h.build_closed_run(1);
    const auto s = h.build_closed_run(2);
    h.accept(n);
    ASSERT_EQ(h.orch.active_run_id(), n.id);
    ASSERT_EQ(h.orch.early_launch_run_id(), s.id);

    // N retires: S is promoted, and the freed admission takes T.
    h.complete(n);
    ASSERT_EQ(h.orch.active_run_id(), s.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
    const auto t = h.build_closed_run(3);
    h.accept(s);
    ASSERT_EQ(h.orch.preparable_run_id(), t.id);
    EXPECT_EQ(h.orch.early_launch_run_id(), t.id);

    // And again, which is what "sustained" means: every retirement frees
    // exactly one admission and authorizes exactly one successor.
    h.complete(s);
    ASSERT_EQ(h.orch.active_run_id(), t.id);
    const auto r = h.build_closed_run(4);
    h.accept(t);
    EXPECT_EQ(h.orch.early_launch_run_id(), r.id);

    h.complete(t);
    ASSERT_EQ(h.orch.active_run_id(), r.id);
    h.accept(r);
    // Nothing is staged behind R, so there is nothing to authorize — the gate
    // closes rather than naming a run that does not exist.
    EXPECT_EQ(h.orch.early_launch_run_id(), INVALID_RUN_ID);
    h.complete(r);
}

TEST(EarlyLaunchAdmission, LaunchDepthIsBoundedByTheAdmissionDepth) {
    EarlyLaunchHarness h(/*depth=*/2, /*launch_depth=*/2);

    // A launched run holds its pipeline slot for its whole lifetime, so the
    // slot budget is the real ceiling and exceeding it is refused rather than
    // silently clamped.
    EXPECT_THROW(h.orch.configure_pipeline_depth(1, 1, 2), std::invalid_argument);
    EXPECT_THROW(h.orch.configure_pipeline_depth(2, 2, 3), std::invalid_argument);
    EXPECT_THROW(h.orch.configure_pipeline_depth(2, 2, 0), std::invalid_argument);
    EXPECT_NO_THROW(h.orch.configure_pipeline_depth(1, 1, 1));
    EXPECT_NO_THROW(h.orch.configure_pipeline_depth(2, 2, 2));
}

}  // namespace

TEST(EarlyLaunchAdmission, AutomaticStartupWaitsForProofThenEnqueuesACompatibleSuccessor) {
    EarlyLaunchHarness h(8, 8);
    h.orch.configure_pipeline_depth(8, 0, 8, 1);
    EXPECT_EQ(h.orch.admissible_depth(), 1u);
    const auto first = h.build_closed_run(101);
    h.accept(first);
    auto next = std::async(std::launch::async, [&h] {
        return h.orch.begin_run();
    });
    EXPECT_EQ(next.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    h.orch.note_endpoint_capacity(0, 2);
    EXPECT_EQ(h.orch.admissible_depth(), 1u) << "a sample alone cannot widen a live run";
    h.complete(first);
    const auto ready = next.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) h.orch.update_admissible_depth(8);
    const EarlyLaunchHarness::Run second{next.get(), h.submit_one(102)};
    h.orch.close_run_submission(second.id);
    EXPECT_EQ(h.orch.admissible_depth(), 2u);
    const auto third = h.build_closed_run(103);
    h.accept(second);
    EXPECT_EQ(h.orch.early_launch_run_id(), third.id);
    const auto lease = h.state(third).pipeline_lease;
    auto fourth_id = std::async(std::launch::async, [&h] {
        return h.orch.begin_run();
    });
    EXPECT_EQ(fourth_id.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    h.orch.note_endpoint_capacity(0, 1);
    h.complete(second);
    EXPECT_EQ(h.orch.admissible_depth(), 1u);
    EXPECT_EQ(fourth_id.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout)
        << "retirement applies the lower capacity before waking an admission waiter";
    EXPECT_EQ(h.state(third).pipeline_lease.slot_id, lease.slot_id);
    EXPECT_EQ(h.state(third).pipeline_lease.generation, lease.generation);
    h.accept(third);
    h.complete(third);
    const auto fourth_ready = fourth_id.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(fourth_ready, std::future_status::ready);
    if (fourth_ready != std::future_status::ready) h.orch.update_admissible_depth(8);
    const EarlyLaunchHarness::Run fourth{fourth_id.get(), h.submit_one(104)};
    h.orch.close_run_submission(fourth.id);
    h.accept(fourth);
    h.complete(fourth);
    h.orch.release_run(first.id);
    h.orch.release_run(second.id);
    h.orch.release_run(third.id);
    h.orch.release_run(fourth.id);
}
