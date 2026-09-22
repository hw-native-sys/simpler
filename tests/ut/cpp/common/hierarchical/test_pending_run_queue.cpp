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
 * Logical pending queue versus native pipeline-slot lease.
 *
 * `begin_run` admits a logical run without taking a native lease. The lease
 * belongs to the two roles that can use one — the FIFO head and the first
 * eligible preparable successor — and is handed out when a run reaches one of
 * them. These tests pin the separation at the predicates production consults
 * (`can_dispatch_run`, `preparable_run_id`) and at the field the mailbox
 * dispatch validates before choosing TASK_READY or PREPARE_READY
 * (`TaskSlotState::pipeline_lease`).
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <functional>
#include <stdexcept>
#include <utility>

#include "call_config.h"
#include "orchestrator.h"
#include "ring.h"
#include "scope.h"
#include "task_args.h"
#include "tensormap.h"
#include "types.h"

namespace {

// One orchestrator wired to real ready queues, an allocator and a scope, with
// the two budgets configured independently.
struct PendingQueueHarness {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;

    PendingQueueHarness(uint32_t depth, uint32_t pending_depth, std::function<void()> ready_notify_cb = {}) {
        allocator.init(/*heap_bytes=*/1ULL << 20);
        rq_next_level.reset({0});
        orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level, nullptr, std::move(ready_notify_cb));
        orch.configure_pipeline_depth(depth, pending_depth);
    }

    ~PendingQueueHarness() { allocator.shutdown(); }

    // One OUTPUT-tensor task, so the run owns a slot the dispatch path would
    // read its lease from.
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
        callable.digest.fill(91);
        return orch.submit_next_level(callable, args, cfg, 0);
    }

    std::pair<RunId, SubmitResult> build_closed_run(uint64_t buffer_id) {
        RunId id = orch.begin_run();
        SubmitResult task = submit_one(buffer_id);
        orch.close_run_submission(id);
        return {id, task};
    }

    // Drive a run's only task to completion so the run retires.
    void complete(const SubmitResult &task) {
        allocator.slot_state(task.task_slot)->state.store(TaskState::COMPLETED, std::memory_order_release);
        EXPECT_TRUE(orch.on_consumed(task.task_slot));
    }

    PipelineSlotLease slot_lease(const SubmitResult &task) {
        return allocator.slot_state(task.task_slot)->pipeline_lease;
    }
};

// Releases a test hook parked on `gate` exactly once, from whichever comes
// first: the deliberate release, or destruction.
//
// Declaring one of these *after* the builder future is what makes a failed
// assertion report rather than hang: the future's destructor joins the parked
// task, so the gate has to be opened before that destructor runs, and reverse
// declaration order is the only thing that guarantees it on an early return.
class HookGate {
public:
    explicit HookGate(std::promise<void> &gate) :
        gate_(&gate) {}
    HookGate(const HookGate &) = delete;
    HookGate &operator=(const HookGate &) = delete;
    ~HookGate() { release(); }

    void release() {
        if (gate_ == nullptr) return;
        std::promise<void> *gate = gate_;
        gate_ = nullptr;
        gate->set_value();
    }

private:
    std::promise<void> *gate_;
};

// Upper bound on a hook's park. The happy path opens the gate in
// milliseconds; this only bounds a path that never reaches the release, so
// the builder cannot be stranded even if the gate is never opened.
constexpr std::chrono::seconds kHookParkBound{30};

}  // namespace

// The separation itself: with a deeper logical queue than the native pool,
// runs three and four are admitted and built while run one executes and run
// two is prepared.
//
// That they hold no native lease is not read out of the pool; it follows from
// the pool having two slots and both being demonstrably held elsewhere — run
// one is dispatchable and run two preparable, and neither predicate is
// satisfiable without a lease.
TEST(PendingQueue, ThirdAndFourthRunsBuildWithoutTakingNativeLeases) {
    static_assert(PTO_PIPELINE_MAX_DEPTH == 2, "this test reasons from a two-slot native pool");
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4);

    auto first = h.build_closed_run(0xE001);
    auto second = h.build_closed_run(0xE002);
    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));
    ASSERT_EQ(h.orch.preparable_run_id(), second.first);

    auto third = h.build_closed_run(0xE003);
    auto fourth = h.build_closed_run(0xE004);
    EXPECT_NE(third.first, INVALID_RUN_ID);
    EXPECT_NE(fourth.first, INVALID_RUN_ID);
    EXPECT_FALSE(h.orch.can_dispatch_run(third.first));
    EXPECT_FALSE(h.orch.can_dispatch_run(fourth.first));
    // The successor role still names run two, not a later pending run.
    EXPECT_EQ(h.orch.preparable_run_id(), second.first);
    // A pending run carries the invalid encoding, which is what the mailbox
    // dispatch identity check rejects.
    EXPECT_EQ(h.slot_lease(third.second).generation, 0u);
    EXPECT_EQ(h.slot_lease(fourth.second).generation, 0u);

    h.complete(first.second);
    h.complete(second.second);
    h.complete(third.second);
    h.complete(fourth.second);
    h.orch.release_run(first.first);
    h.orch.release_run(second.first);
    h.orch.release_run(third.first);
    h.orch.release_run(fourth.first);
}

// The successor preparation this change must not regress: `preparable_run_id`
// names the closed successor while the predecessor still executes, which is
// what drives prepare_only / PREPARE_READY.
TEST(PendingQueue, SuccessorIsPreparableWhileThePredecessorStillExecutes) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4);

    auto first = h.build_closed_run(0xE101);
    auto second = h.build_closed_run(0xE102);

    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));
    EXPECT_FALSE(h.orch.run_done(first.first));
    EXPECT_EQ(h.orch.preparable_run_id(), second.first);

    h.complete(first.second);
    h.complete(second.second);
    h.orch.release_run(first.first);
    h.orch.release_run(second.first);
}

// Both dispatch shapes read `TaskSlotState::pipeline_lease` before choosing
// TASK_READY or PREPARE_READY, so a run built without a lease must have its
// already-registered slots restamped when it acquires one. A zero generation
// here is exactly what `LocalMailboxEndpoint::submit_progress` rejects.
TEST(PendingQueue, BothDispatchIdentitiesResolveAfterALeaselessBuild) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4);

    auto first = h.build_closed_run(0xE201);
    auto second = h.build_closed_run(0xE202);

    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));
    ASSERT_EQ(h.orch.preparable_run_id(), second.first);

    const PipelineSlotLease active = h.slot_lease(first.second);
    const PipelineSlotLease prepared = h.slot_lease(second.second);
    EXPECT_NE(active.generation, 0u) << "the active run's slot would fail the dispatch identity check";
    EXPECT_NE(prepared.generation, 0u) << "the prepared successor's slot would fail the same check";
    EXPECT_EQ(active.reserved, 0u);
    EXPECT_EQ(prepared.reserved, 0u);
    EXPECT_LT(active.slot_id, static_cast<uint32_t>(PTO_PIPELINE_MAX_DEPTH));
    EXPECT_LT(prepared.slot_id, static_cast<uint32_t>(PTO_PIPELINE_MAX_DEPTH));
    EXPECT_NE(active.slot_id, prepared.slot_id);

    h.complete(first.second);
    h.complete(second.second);
    h.orch.release_run(first.first);
    h.orch.release_run(second.first);
}

// The cap is the FIFO's, and a terminal run leaves the FIFO at retirement
// rather than at `release_run`. A run the caller has not released still
// answers queries but no longer occupies admission budget.
TEST(PendingQueue, TerminalButUnreleasedRunDoesNotHoldAdmissionBudget) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/2);

    auto first = h.build_closed_run(0xE301);
    auto second = h.build_closed_run(0xE302);

    h.complete(first.second);
    ASSERT_TRUE(h.orch.run_done(first.first));

    // Deliberately not released: its RunState is still registered.
    RunId third = h.orch.begin_run();
    EXPECT_NE(third, INVALID_RUN_ID);
    SubmitResult third_task = h.submit_one(0xE303);
    h.orch.close_run_submission(third);
    EXPECT_TRUE(h.orch.run_done(first.first));

    h.orch.release_run(first.first);
    h.complete(second.second);
    h.complete(third_task);
    h.orch.release_run(second.first);
    h.orch.release_run(third);
}

// A parked admission waiter is released by the FIFO erase in
// `retire_terminal_run`, not by a returned lease — the pending run it admits
// takes no lease at all.
TEST(PendingQueue, RetirementWakesAnAdmissionWaiterBlockedOnTheCap) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/2);

    auto first = h.build_closed_run(0xE401);
    auto second = h.build_closed_run(0xE402);

    auto waiter = std::async(std::launch::async, [&h] {
        return h.orch.begin_run();
    });
    EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "a full admission FIFO must not admit another run";

    h.complete(first.second);

    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "retiring a run never woke the waiter its freed FIFO entry admits";
    RunId third = waiter.get();
    SubmitResult third_task = h.submit_one(0xE403);
    h.orch.close_run_submission(third);

    h.orch.release_run(first.first);
    h.complete(second.second);
    h.complete(third_task);
    h.orch.release_run(second.first);
    h.orch.release_run(third);
}

// Cancelling a run that never reached an active or preparable role frees its
// admission budget and leaves the two live leases alone. The pool already
// refuses a generation-zero release, so the guard in `retire_terminal_run` is
// defensive; what this pins is the budget release and the untouched roles.
TEST(PendingQueue, CancellingALeaselessRunFreesBudgetAndKeepsLiveLeases) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/3);

    auto first = h.build_closed_run(0xE501);
    auto second = h.build_closed_run(0xE502);
    RunId third = h.orch.begin_run();
    ASSERT_NE(h.slot_lease(first.second).generation, 0u);

    h.orch.fail_run_submission(third, std::make_exception_ptr(std::runtime_error("graph build failed")));
    EXPECT_TRUE(h.orch.run_done(third));
    EXPECT_TRUE(h.orch.run_failed(third));

    // The two roles are untouched by the cancellation.
    EXPECT_TRUE(h.orch.can_dispatch_run(first.first));
    EXPECT_EQ(h.orch.preparable_run_id(), second.first);

    // And the budget the cancelled run held is back.
    RunId fourth = h.orch.begin_run();
    EXPECT_NE(fourth, INVALID_RUN_ID);
    SubmitResult fourth_task = h.submit_one(0xE504);
    h.orch.close_run_submission(fourth);

    h.orch.release_run(third);
    h.complete(first.second);
    h.complete(second.second);
    h.complete(fourth_task);
    h.orch.release_run(first.first);
    h.orch.release_run(second.first);
    h.orch.release_run(fourth);
}

// A failed publication inside `begin_run` must give back the FIFO entry it
// published, because that entry is what the admission predicate counts. No
// lease is returned, because none was taken.
TEST(PendingQueue, FailedBeginGivesBackAdmissionBudgetWithoutReleasingALease) {
    for (OrchestratorTestPoint failure_point :
         {OrchestratorTestPoint::BEGIN_RUN_MAP_PUBLISHED, OrchestratorTestPoint::BEGIN_RUN_FIFO_PUBLISHED}) {
        PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/1);

        h.orch.set_test_hook([failure_point](OrchestratorTestPoint point) {
            if (point == failure_point) throw std::runtime_error("injected begin_run publication failure");
        });
        EXPECT_THROW(h.orch.begin_run(), std::runtime_error);
        h.orch.set_test_hook(nullptr);

        // With a cap of one, a leaked FIFO entry would block this outright.
        RunId recovered = h.orch.begin_run();
        EXPECT_NE(recovered, INVALID_RUN_ID);
        SubmitResult task = h.submit_one(0xE601);
        h.orch.close_run_submission(recovered);
        // The native lease is still available to whichever run reaches the head.
        EXPECT_TRUE(h.orch.can_dispatch_run(recovered));
        EXPECT_NE(h.slot_lease(task).generation, 0u);

        h.complete(task);
        h.orch.release_run(recovered);
    }
}

// The pre-wait exclusivity check is evaluated before the wait releases the
// mutex, so it says nothing about the state a woken waiter returns to. When
// several retirements free more than one FIFO entry, a second waiter can pass
// the capacity predicate after a first has already published itself as the
// builder. Only one may succeed.
//
// The window is a race, so this runs in rounds, like the depth-one wakeup test
// in test_orchestrator.cpp. Without the post-wake re-check both waiters become
// the building run.
TEST(PendingQueue, ConcurrentAdmissionNeverPublishesTwoBuilders) {
    for (int round = 0; round < 100; ++round) {
        PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/3);

        auto first = h.build_closed_run(0xE700 + static_cast<uint64_t>(round));
        auto second = h.build_closed_run(0xE800 + static_cast<uint64_t>(round));
        auto third = h.build_closed_run(0xE900 + static_cast<uint64_t>(round));

        std::atomic<int> succeeded{0};
        std::atomic<int> rejected{0};
        auto admit = [&h, &succeeded, &rejected]() -> RunId {
            try {
                RunId id = h.orch.begin_run();
                succeeded.fetch_add(1, std::memory_order_acq_rel);
                return id;
            } catch (const std::logic_error &) {
                rejected.fetch_add(1, std::memory_order_acq_rel);
                return INVALID_RUN_ID;
            }
        };
        auto a = std::async(std::launch::async, admit);
        auto b = std::async(std::launch::async, admit);

        // Two retirements back to back: enough freed budget that a second
        // waiter can pass the capacity predicate behind the first.
        h.complete(first.second);
        h.complete(second.second);

        ASSERT_EQ(a.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "round " << round;
        ASSERT_EQ(b.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "round " << round;
        RunId admitted_a = a.get();
        RunId admitted_b = b.get();

        EXPECT_LE(succeeded.load(std::memory_order_acquire), 1)
            << "round " << round << ": two callers became the building run";
        EXPECT_EQ(succeeded.load(std::memory_order_acquire) + rejected.load(std::memory_order_acquire), 2);

        // Drain in FIFO order: a run only terminalizes once it has reached the
        // head, so an admitted run cannot be released before its predecessors.
        h.complete(third.second);
        h.orch.release_run(first.first);
        h.orch.release_run(second.first);
        h.orch.release_run(third.first);
        for (RunId id : {admitted_a, admitted_b}) {
            if (id == INVALID_RUN_ID) continue;
            SubmitResult task = h.submit_one(0xEA00 + static_cast<uint64_t>(round));
            h.orch.close_run_submission(id);
            h.complete(task);
            h.orch.release_run(id);
        }
    }
}

// A pending run behind the successor inherits the lease when the successor is
// cancelled, while the head keeps executing. `activate_fifo_head` cannot do
// this — it returns early whenever a run is active, which is precisely this
// case — so the refresh in `retire_terminal_run` is what re-lends the slot.
TEST(PendingQueue, CancellingThePreparedSuccessorHandsItsLeaseToTheNextPendingRun) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4);

    auto first = h.build_closed_run(0xED01);
    auto second = h.build_closed_run(0xED02);
    auto third = h.build_closed_run(0xED03);

    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));
    ASSERT_EQ(h.orch.preparable_run_id(), second.first);
    // The third run is behind the successor and holds no lease.
    ASSERT_EQ(h.slot_lease(third.second).generation, 0u);

    h.orch.fail_run_submission(second.first, std::make_exception_ptr(std::runtime_error("successor cancelled")));
    ASSERT_TRUE(h.orch.run_done(second.first));

    // The head is untouched, and the freed lease moved to the next pending run
    // rather than waiting for the head to retire.
    EXPECT_TRUE(h.orch.can_dispatch_run(first.first));
    EXPECT_FALSE(h.orch.run_done(first.first));
    EXPECT_EQ(h.orch.preparable_run_id(), third.first);
    EXPECT_NE(h.slot_lease(third.second).generation, 0u);

    h.orch.release_run(second.first);
    h.complete(first.second);
    h.complete(third.second);
    h.orch.release_run(first.first);
    h.orch.release_run(third.first);
}

// A builder registering a slot races promotion: the run can gain its lease
// between the slot's allocation and its registration, so the restamp scan
// cannot see that slot. Registration and lease publication therefore happen
// under one mutex, and this drives exactly that interleaving — the submitting
// thread is parked at the pre-registration hook while the predecessor retires
// and promotes its run.
TEST(PendingQueue, SlotRegisteredAfterPromotionStillGetsAValidIdentity) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4);

    auto first = h.build_closed_run(0xEE01);
    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));

    RunId second = h.orch.begin_run();
    ASSERT_NE(second, INVALID_RUN_ID);

    std::promise<void> at_register;
    std::promise<void> may_register;
    std::future<void> at_register_f = at_register.get_future();
    std::shared_future<void> may_register_f = may_register.get_future().share();
    std::atomic<bool> armed{true};
    h.orch.set_test_hook([&](OrchestratorTestPoint point) {
        if (point != OrchestratorTestPoint::SUBMIT_RUN_SLOT_REGISTERING) return;
        if (!armed.exchange(false, std::memory_order_acq_rel)) return;
        at_register.set_value();
        (void)may_register_f.wait_for(kHookParkBound);
    });

    auto builder = std::async(std::launch::async, [&h] {
        return h.submit_one(0xEE02);
    });
    // After `builder`, so it is destroyed before the future joins.
    HookGate gate(may_register);
    ASSERT_EQ(at_register_f.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // The slot exists but is not registered, so promotion's restamp cannot
    // reach it.
    h.complete(first.second);

    gate.release();
    ASSERT_EQ(builder.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    SubmitResult task = builder.get();
    h.orch.set_test_hook(nullptr);

    const PipelineSlotLease stamped = h.slot_lease(task);
    EXPECT_NE(stamped.generation, 0u) << "a slot registered after promotion kept the invalid encoding";
    EXPECT_EQ(stamped.reserved, 0u);
    EXPECT_LT(stamped.slot_id, static_cast<uint32_t>(PTO_PIPELINE_MAX_DEPTH));

    h.orch.close_run_submission(second);
    EXPECT_TRUE(h.orch.can_dispatch_run(second));

    h.orch.release_run(first.first);
    h.complete(task);
    h.orch.release_run(second);
}

// The scheduler's wake is a generation counter, so a notify consumed while the
// successor is still lease-less is spent: `preparable_run_id` answers nothing
// and no later event follows while the predecessor is active. The lease must
// therefore be visible before the wake is issued.
TEST(PendingQueue, TheReadyWakeFollowsTheSuccessorLease) {
    std::atomic<int> wakes{0};
    std::atomic<int> wakes_with_preparable{0};
    Orchestrator *observed = nullptr;

    auto notify = [&] {
        wakes.fetch_add(1, std::memory_order_acq_rel);
        if (observed != nullptr && observed->preparable_run_id() != INVALID_RUN_ID) {
            wakes_with_preparable.fetch_add(1, std::memory_order_acq_rel);
        }
    };

    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/4, notify);
    observed = &h.orch;

    auto first = h.build_closed_run(0xEF01);
    ASSERT_TRUE(h.orch.can_dispatch_run(first.first));

    // Closing the successor is the transition that makes it preparable. The
    // wake it emits must already see the lease in place.
    const int before = wakes.load(std::memory_order_acquire);
    auto second = h.build_closed_run(0xEF02);
    EXPECT_GT(wakes.load(std::memory_order_acquire), before) << "closing a successor issued no wake at all";
    EXPECT_GT(wakes_with_preparable.load(std::memory_order_acquire), 0)
        << "every wake was issued while preparable_run_id was still empty";
    EXPECT_EQ(h.orch.preparable_run_id(), second.first);

    h.complete(first.second);
    h.complete(second.second);
    h.orch.release_run(first.first);
    h.orch.release_run(second.first);
}

// Depth and pending depth are separate budgets. Zero derives the FIFO bound
// from the native depth, which is what every caller had before they split.
TEST(PendingQueue, PendingDepthZeroDerivesTheCapFromTheNativeDepth) {
    PendingQueueHarness h(/*depth=*/1, /*pending_depth=*/0);

    RunId only = h.orch.begin_run();
    SubmitResult task = h.submit_one(0xEB01);
    h.orch.close_run_submission(only);

    auto waiter = std::async(std::launch::async, [&h] {
        return h.orch.begin_run();
    });
    EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout)
        << "pending_depth 0 must derive the cap from depth 1";

    h.complete(task);
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    RunId next = waiter.get();
    SubmitResult next_task = h.submit_one(0xEB02);
    h.orch.close_run_submission(next);
    h.complete(next_task);
    h.orch.release_run(only);
    h.orch.release_run(next);
}

TEST(PendingQueue, ConfigurationRejectsOnlyTheNativeDepthRange) {
    TensorMap tm;
    Ring allocator;
    Scope scope;
    NextLevelReadyQueues rq_next_level;
    ReadyQueue rq_sub;
    Orchestrator orch;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    rq_next_level.reset({0});
    orch.init(&tm, &allocator, &scope, &rq_sub, &rq_next_level);

    EXPECT_THROW(orch.configure_pipeline_depth(0, 1), std::invalid_argument);
    EXPECT_THROW(orch.configure_pipeline_depth(PTO_PIPELINE_MAX_DEPTH + 1, 1), std::invalid_argument);
    // A pending depth above the native depth is the point of the knob, and one
    // below it is legal too — it leaves the successor role unfillable rather
    // than rejecting the configuration.
    EXPECT_NO_THROW(orch.configure_pipeline_depth(2, 64));
    EXPECT_NO_THROW(orch.configure_pipeline_depth(2, 1));

    allocator.shutdown();
}

// A pending depth below the native depth serializes: the successor role needs
// a second FIFO entry, and the cap denies it one.
TEST(PendingQueue, PendingDepthBelowNativeDepthDisablesSuccessorPreparation) {
    PendingQueueHarness h(/*depth=*/2, /*pending_depth=*/1);

    auto first = h.build_closed_run(0xEC01);
    EXPECT_TRUE(h.orch.can_dispatch_run(first.first));
    EXPECT_EQ(h.orch.preparable_run_id(), INVALID_RUN_ID);

    h.complete(first.second);
    h.orch.release_run(first.first);
}
