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
 * What a blocked admission waits on, and what proves every waiter has left.
 *
 * Two pieces with one job between them. `DiagnosticCreditSignal` is the
 * runtime-side counter a collector's writer advances when it frees capacity;
 * `CreditCallGate` is the extension-side permission to be inside the entry
 * that reads it, and the only thing that can prove a caller has returned
 * before the runtime is unloaded under it.
 *
 * The interleavings below are driven rather than described: each case holds a
 * thread at the exact point the contract is about, using a latch the
 * production code knows nothing about, and then asserts what the other thread
 * observes. What they cannot cover, stated rather than simulated: an actual
 * `dlclose`, which no unit test can stage safely.
 */

#include "credit_call_gate.h"

#include <gtest/gtest.h>

#include "host/diagnostic_credit_signal.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace {

using simpler::dfx::credit::CreditWait;
using simpler::dfx::credit::DiagnosticCreditSignal;

/** Spin until `pred` holds or the budget runs out; false means it never did. */
template <typename Pred>
bool wait_until_true(Pred pred, int budget_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::yield();
    }
    return pred();
}

}  // namespace

// --- The signal ---------------------------------------------------------

TEST(DiagnosticCreditSignalTest, AReleaseWakesAWaiterParkedOnTheOldGeneration) {
    DiagnosticCreditSignal signal;
    const uint64_t seen = signal.generation();
    std::atomic<int> result{-99};
    std::atomic<bool> entered{false};

    std::thread waiter([&] {
        entered.store(true, std::memory_order_release);
        result.store(static_cast<int>(signal.wait_for(seen, 5000)), std::memory_order_release);
    });
    ASSERT_TRUE(wait_until_true([&] {
        return entered.load(std::memory_order_acquire);
    }));
    // What a collector's writer does when it publishes and frees a slot.
    signal.advance();
    waiter.join();

    EXPECT_EQ(result.load(), static_cast<int>(CreditWait::Advanced))
        << "a release did not wake the admission parked on it";
}

TEST(DiagnosticCreditSignalTest, AReleaseBeforeTheWaitIsNotLost) {
    // The whole of the lost-wakeup argument: the generation is read before the
    // admission attempt, so a release landing in the window between that read
    // and the wait has already moved it, and the wait must not park.
    DiagnosticCreditSignal signal;
    const uint64_t seen = signal.generation();
    signal.advance();

    // A budget long enough that parking would be visible as a hang rather than
    // as a timeout, so a pass here is "did not park" and not "woke in time".
    EXPECT_EQ(signal.wait_for(seen, 60000), CreditWait::Advanced);
}

TEST(DiagnosticCreditSignalTest, ABudgetThatElapsesIsATimeoutAndNotProgress) {
    DiagnosticCreditSignal signal;
    EXPECT_EQ(signal.wait_for(signal.generation(), 1), CreditWait::TimedOut);
    // A non-positive budget evaluates the predicate once and does not park.
    EXPECT_EQ(signal.wait_for(signal.generation(), 0), CreditWait::TimedOut);
}

TEST(DiagnosticCreditSignalTest, CancellationIsStickyForWaitsEnteredAfterIt) {
    DiagnosticCreditSignal signal;
    signal.cancel();
    EXPECT_TRUE(signal.cancelled());
    // Not a timeout: a wait entered after cancellation must not park against a
    // signal nobody will raise again, and its caller must be able to tell the
    // two apart.
    EXPECT_EQ(signal.wait_for(signal.generation(), 60000), CreditWait::Cancelled);
}

TEST(DiagnosticCreditSignalTest, CancellationReleasesAParkedWaiter) {
    DiagnosticCreditSignal signal;
    const uint64_t seen = signal.generation();
    std::atomic<int> result{-99};
    std::atomic<bool> entered{false};

    std::thread waiter([&] {
        entered.store(true, std::memory_order_release);
        result.store(static_cast<int>(signal.wait_for(seen, 60000)), std::memory_order_release);
    });
    ASSERT_TRUE(wait_until_true([&] {
        return entered.load(std::memory_order_acquire);
    }));
    signal.cancel();
    waiter.join();

    EXPECT_EQ(result.load(), static_cast<int>(CreditWait::Cancelled))
        << "a cancelled wait reported progress or waited out its budget";
}

// --- The gate -----------------------------------------------------------

namespace {

/** A gate whose targets are the counters below, so a call is observable. */
struct GateFixture {
    CreditCallGate gate;
    static std::atomic<int> generation_calls;
    static std::atomic<int> wait_calls;
    static std::atomic<int> cancel_calls;
    /** Held inside the wait target, so a test can stage an in-flight call. */
    static std::atomic<bool> hold_wait;

    GateFixture() {
        generation_calls.store(0);
        wait_calls.store(0);
        cancel_calls.store(0);
        hold_wait.store(false);
        gate.entries.ctx = reinterpret_cast<void *>(0x1);
        gate.entries.generation = [](void *) -> uint64_t {
            generation_calls.fetch_add(1);
            return 7;
        };
        gate.entries.wait = [](void *, uint64_t, int32_t) -> int {
            wait_calls.fetch_add(1);
            while (hold_wait.load(std::memory_order_acquire))
                std::this_thread::yield();
            return 1;
        };
        gate.entries.cancel = [](void *) {
            cancel_calls.fetch_add(1);
            // What the real cancel does: make every parked waiter's predicate
            // true. Here that is the latch the wait target spins on.
            hold_wait.store(false, std::memory_order_release);
        };
    }
};

std::atomic<int> GateFixture::generation_calls{0};
std::atomic<int> GateFixture::wait_calls{0};
std::atomic<int> GateFixture::cancel_calls{0};
std::atomic<bool> GateFixture::hold_wait{false};

}  // namespace

TEST(CreditCallGateTest, AnOpenGateAdmitsAndHandsOutItsTargets) {
    GateFixture fx;
    {
        CreditCall call(fx.gate, Admit::Ordinary);
        ASSERT_TRUE(static_cast<bool>(call));
        EXPECT_EQ(call.entries().generation(call.entries().ctx), 7u);
    }
    EXPECT_EQ(GateFixture::generation_calls.load(), 1);
    EXPECT_EQ(fx.gate.calls_in_flight, 0u) << "an admitted call did not release its lease";
}

TEST(CreditCallGateTest, ADetachedGateRefusesAndHandsOutNothing) {
    GateFixture fx;
    ASSERT_EQ(close_credit_gate(fx.gate), CloseOutcome::Owner);
    EXPECT_EQ(fx.gate.state, GateState::Detached);

    CreditCall call(fx.gate, Admit::Ordinary);
    EXPECT_FALSE(static_cast<bool>(call)) << "a detached gate admitted a call";
    // The targets are gone, so there is nothing a refused caller could read
    // even by mistake: this is what makes a later unload have no reader.
    EXPECT_EQ(fx.gate.entries.ctx, nullptr);
    EXPECT_EQ(fx.gate.entries.generation, nullptr);
    EXPECT_EQ(fx.gate.entries.wait, nullptr);
    EXPECT_EQ(fx.gate.entries.cancel, nullptr);
}

TEST(CreditCallGateTest, CloseCancelsBeforeItDrainsAndWaitsForTheCallInFlight) {
    // The ordering the whole close rests on. A waiter is held inside the wait
    // target with its lease taken; close must cancel — which is what releases
    // it — before draining, and must not detach until it has actually
    // returned. Reversing the two would wait out the waiter's whole budget.
    GateFixture fx;
    GateFixture::hold_wait.store(true, std::memory_order_release);
    std::atomic<bool> inside{false};
    std::atomic<bool> returned{false};

    std::thread waiter([&] {
        CreditCall call(fx.gate, Admit::Ordinary);
        ASSERT_TRUE(static_cast<bool>(call));
        inside.store(true, std::memory_order_release);
        (void)call.entries().wait(call.entries().ctx, 0, 60000);
        returned.store(true, std::memory_order_release);
    });
    ASSERT_TRUE(wait_until_true([&] {
        return inside.load(std::memory_order_acquire);
    }));

    EXPECT_EQ(close_credit_gate(fx.gate), CloseOutcome::Owner);
    // Close returned, so the drain completed, so the waiter's destructor ran —
    // in extension code, strictly after the target returned.
    EXPECT_TRUE(returned.load(std::memory_order_acquire)) << "close detached while a call was still in flight";
    EXPECT_EQ(fx.gate.calls_in_flight, 0u);
    EXPECT_EQ(GateFixture::cancel_calls.load(), 1) << "close drained without cancelling first";
    waiter.join();
}

TEST(CreditCallGateTest, OnlyOneThreadOwnsTeardownAndTheRestJoinIt) {
    GateFixture fx;
    constexpr int kClosers = 4;
    std::atomic<int> owners{0};
    std::atomic<int> already{0};
    std::vector<std::thread> threads;
    threads.reserve(kClosers);
    for (int i = 0; i < kClosers; i++) {
        threads.emplace_back([&] {
            if (close_credit_gate(fx.gate) == CloseOutcome::Owner) {
                owners.fetch_add(1);
            } else {
                already.fetch_add(1);
            }
        });
    }
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(owners.load(), 1) << "two threads both claimed teardown";
    EXPECT_EQ(already.load(), kClosers - 1);
    // Every follower returned only once detaching was a fact, so "close
    // returned" means "detached" for all of them.
    EXPECT_EQ(fx.gate.state, GateState::Detached);
    EXPECT_EQ(GateFixture::cancel_calls.load(), 1) << "the privileged cancel ran more than once";
}

TEST(CreditCallGateTest, ARepeatedCloseTouchesNoTarget) {
    GateFixture fx;
    ASSERT_EQ(close_credit_gate(fx.gate), CloseOutcome::Owner);
    const int cancels = GateFixture::cancel_calls.load();
    EXPECT_EQ(close_credit_gate(fx.gate), CloseOutcome::Already);
    EXPECT_EQ(GateFixture::cancel_calls.load(), cancels) << "a repeated close read a detached target";
}

TEST(CreditCallGateTest, AGateWithNoTargetsRefusesWithoutDereferencingThem) {
    // What a runtime that exports none of the three leaves behind. The call is
    // admitted — the gate is open — and the caller is the one that must check,
    // which is why every caller reads the targets out of the call rather than
    // assuming them.
    CreditCallGate gate;
    CreditCall call(gate, Admit::Ordinary);
    EXPECT_TRUE(static_cast<bool>(call));
    EXPECT_EQ(call.entries().generation, nullptr);
    EXPECT_EQ(call.entries().wait, nullptr);
    EXPECT_EQ(call.entries().ctx, nullptr);
}
