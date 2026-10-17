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
 * What the lane does with a run the backend will not admit for want of
 * diagnostic capacity.
 *
 * The backend is a stub, because what is under test is the lane's half of the
 * contract: that a capacity refusal is not a failure, that it is not waited
 * for under the lane mutex, that the deadline it starts is fixed once and
 * fails only the run it belongs to, and that a release wakes the run that was
 * waiting. The collectors' half — which owner refuses and when — is covered
 * where those collectors live.
 *
 * `g_launch_rc` returning `SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL` is exactly
 * what a backend whose owner has no free slot returns, and the credit entries
 * are stubbed so a case can say when capacity "came back".
 */

// Every standard and project header the case needs comes first: the
// `private`-to-`public` redefinition below must not be in force while one is
// parsed, or a class that declares an access specifier inside itself — the
// standard library has several — is redeclared with the wrong one.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "call_config.h"
#include "run_resource_arena.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "tensor.h"
#include "types.h"

#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"
#include "credit_call_gate.h"

namespace {

std::unordered_map<void *, uint32_t> g_slots;
std::array<bool, 2> g_complete{};
std::array<int, 2> g_launch_rc{};
std::vector<std::string> g_events;

// The stub credit signal. `g_generation` is what the lane reads, and a case
// advances it to stand for a collector's writer publishing and freeing a slot.
std::atomic<uint64_t> g_generation{1};
std::atomic<bool> g_cancelled{false};
std::atomic<int> g_credit_waits{0};

uint32_t slot_of(void *runtime) { return g_slots.at(runtime); }

int prepare_run(
    void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    (void)runtime;
    g_complete[descriptor->pipeline_slot] = false;
    g_events.push_back("prepare" + std::to_string(descriptor->pipeline_slot));
    return 0;
}

int launch_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("launch" + std::to_string(slot));
    return g_launch_rc[slot];
}

int poll_run(void *, void *runtime) {
    return g_complete[slot_of(runtime)] ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int wait_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("wait" + std::to_string(slot));
    g_complete[slot] = true;
    return 0;
}

int g_finalize_rc{0};
bool g_finalize_throws{false};

int finalize_run(void *, void *runtime) {
    g_events.push_back("finalize" + std::to_string(slot_of(runtime)));
    if (g_finalize_throws) return PTO_RUNTIME_ERR_INTERNAL;
    return g_finalize_rc;
}

// The current-context preflight the worker runs before it mutates anything.
int g_validate_rc{0};

int validate_execution_context(void *) { return g_validate_rc; }

int supports_successor(void *) { return 1; }

/**
 * Joined launch, declined.
 *
 * A worker whose contract and launch depth both exceed one is asked this on
 * every scheduling round, through a call with no null guard. Zero is this
 * fixture's answer: the stubs here implement the ordinary launch path only, and
 * `launch_run_joined_fn_` is reached only once this answers non-zero.
 */
int declines_joined_launch(void *) { return 0; }

uint64_t credit_generation(void *) { return g_generation.load(std::memory_order_acquire); }

int credit_wait(void *, uint64_t seen, int32_t timeout_ms) {
    g_credit_waits.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (g_cancelled.load(std::memory_order_acquire)) return -1;
        if (g_generation.load(std::memory_order_acquire) != seen) return 1;
        std::this_thread::yield();
    }
    return 0;
}

void credit_cancel(void *) { g_cancelled.store(true, std::memory_order_release); }

/**
 * The device context the lane's credit entries carry.
 *
 * `wait_for_credit` refuses a null one before it reaches the wait callback, so
 * a lane whose context is null never parks. The stubs below ignore the
 * pointer's value; what matters is that it names something outliving every
 * wait.
 */
int g_device_ctx = 0;

void prime_worker(ChipWorker &worker, unsigned launch_depth = 2) {
    g_slots.clear();
    g_complete = {};
    g_launch_rc = {};
    g_events.clear();
    g_generation.store(1);
    g_cancelled.store(false);
    g_credit_waits.store(0);
    g_finalize_rc = 0;
    g_finalize_throws = false;
    g_validate_rc = 0;
    worker.launch_depth_ = launch_depth;
    worker.initialized_ = true;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 2, {}};
    // Sized the way `init` sizes it from the resolved contract. Unsized, every
    // lease names a slot the filter has no row for and is refused as stale, so
    // no run in this case would reach the backend at all.
    worker.pipeline_generations_.resize(worker.pipeline_contract_.pipeline_depth);
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t));
    for (uint32_t slot = 0; slot < worker.runtime_bufs_.size(); ++slot) {
        g_slots.emplace(worker.runtime_bufs_[slot].data(), slot);
    }
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.finalize_run_fn_ = finalize_run;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
    worker.supports_joined_native_launch_fn_ = declines_joined_launch;
    worker.credit_generation_fn_ = credit_generation;
    worker.credit_wait_fn_ = credit_wait;
    worker.credit_cancel_fn_ = credit_cancel;
    worker.validate_execution_context_fn_ = validate_execution_context;
    worker.device_ctx_ = &g_device_ctx;
}

/** One host-space tensor, so a run stays inside the joinable scope. */
ChipStorageTaskArgs host_args() {
    ChipStorageTaskArgs args{};
    ChipTensor t{};
    const uint32_t shape = 1;
    t.init_external(reinterpret_cast<void *>(0x1000), 1, &shape, 1, DataType::UINT8, AddressSpace::HOST);
    args.add_tensor(t);
    return args;
}

/** A run with one diagnostic owner on, which is what makes credit a term. */
CallConfig diagnostic_config() {
    CallConfig config;
    config.enable_scope_stats = 1;
    config.output_prefix[0] = 'x';
    return config;
}

/**
 * A thread joined on every exit.
 *
 * A `std::thread` destroyed while joinable calls `std::terminate`, so a body
 * that returns early past its own join takes the whole binary with it.
 */
struct Joined {
    std::thread thread;

    explicit Joined(std::thread &&t) :
        thread(std::move(t)) {}
    Joined(const Joined &) = delete;
    Joined &operator=(const Joined &) = delete;
    ~Joined() {
        if (thread.joinable()) thread.join();
    }
};

/**
 * Released on every exit, before the thread waiting on it is joined.
 *
 * Declared after the `Joined` that holds the thread, so destruction releases
 * the loop first and joins second — a spin for a condition that never arrives
 * would otherwise hang the join.
 */
struct Released {
    std::atomic<bool> &flag;

    ~Released() { flag.store(true, std::memory_order_release); }
};

ChipRun submit(ChipRunLane &lane, uint64_t run_id, uint32_t slot, bool activate, const CallConfig &config) {
    const ChipStorageTaskArgs args = host_args();
    return lane.submit(1, args, config, PipelineSlotLease{slot, 0, run_id}, run_id, run_id, nullptr, 0, activate);
}

}  // namespace

TEST(ChipRunLaneCreditBlockTest, ACapacityRefusalLeavesTheRunPreparedRatherThanFailed) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 301, 0, /*activate=*/true, diagnostic_config());

    // Attempted and refused: the backend was asked, nothing was submitted, and
    // the run is neither launched nor failed. A failure here would be the
    // defect — the capacity is merely in use.
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0"}));
    EXPECT_FALSE(run.launched());
    EXPECT_FALSE(run.done());

    // And it is still launchable: capacity came back, nothing else changed.
    g_launch_rc[0] = 0;
    g_generation.fetch_add(1);
    EXPECT_TRUE(run.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(5)) || run.launched());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, AReleaseWakesTheWaiterAndTheRunLaunches) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 302, 0, /*activate=*/true, diagnostic_config());
    ASSERT_FALSE(run.launched());

    // A waiter parks outside the lane mutex, which is what lets this thread —
    // standing in for a collector's writer — make progress at all. Under the
    // old behaviour the waiter held the lane and nothing here could be
    // observed by it.
    std::atomic<bool> give_up{false};
    std::atomic<bool> saw_waiter{false};
    Joined writer(std::thread([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (g_credit_waits.load(std::memory_order_acquire) == 0) {
            // Bounded, and released by the guard below: a wait that never
            // arrives is this case's failure to report, not a thread for the
            // join to hang on.
            if (give_up.load(std::memory_order_acquire) || std::chrono::steady_clock::now() >= deadline) return;
            std::this_thread::yield();
        }
        saw_waiter.store(true, std::memory_order_release);
        // Both plain stores before the generation bump: the lane wakes on that
        // generation and relaunches, and its acquire read of `g_generation` is
        // what orders these for it. Storing after the bump is a race the lane
        // can lose by reading a stale `g_complete`.
        g_launch_rc[0] = 0;
        g_complete[0] = true;
        g_generation.fetch_add(1);
    }));
    Released release_writer{give_up};
    const bool finished = run.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(10));

    EXPECT_GT(g_credit_waits.load(), 0) << "the waiter never parked on the credit signal";
    EXPECT_TRUE(saw_waiter.load()) << "the release was published without a waiter to observe it";
    EXPECT_TRUE(finished) << "a release did not let the blocked run through";
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, TheDeadlineIsFixedOnceAndNotRenewedByPolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 303, 0, /*activate=*/true, diagnostic_config());
    ASSERT_FALSE(run.launched());
    const auto first = run.credit_deadline_for_test();
    ASSERT_NE(first, std::chrono::steady_clock::time_point{}) << "a front refusal started no deadline";

    // Several more rounds, each of which refuses again. A deadline renewed by
    // a poll would never expire, which is the failure this guards.
    for (int i = 0; i < 3; i++) {
        (void)run.wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(5));
        EXPECT_EQ(run.credit_deadline_for_test(), first) << "round " << i << " moved the deadline";
    }
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, AnExpiredDeadlineFailsOnlyThatRun) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    // The first run launches and stays executing; the second is refused for
    // credit. Only the second may be failed by its own deadline.
    g_launch_rc[0] = 0;
    g_launch_rc[1] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun first = submit(lane, 304, 0, /*activate=*/true, diagnostic_config());
    ASSERT_TRUE(first.launched());
    ChipRun second = submit(lane, 305, 1, /*activate=*/true, diagnostic_config());
    EXPECT_FALSE(second.launched());

    // Retire the front so the second reaches it and takes its own deadline,
    // then bring that deadline forward rather than waiting out 30 s.
    g_complete[0] = true;
    EXPECT_TRUE(first.done()) << "the launched run was affected by its successor's block";
    (void)second.wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(5));
    second.expire_credit_deadline_for_test();

    // Its own failure, naming credit, and nothing poisoned: the lane is still
    // usable and the run ahead kept its own outcome.
    EXPECT_THROW(second.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)), std::runtime_error);
    EXPECT_NO_THROW(lane.close());
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, CloseCancelsTheWaitRatherThanWaitingItOut) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 306, 0, /*activate=*/true, diagnostic_config());
    ASSERT_FALSE(run.launched());

    // Close publishes cancellation before it drains. Without that ordering the
    // drain would wait for a waiter that is itself waiting for capacity that
    // is never coming, which is the cycle this ordering exists to break.
    //
    // What is measured is the shutdown, not the run's outcome. This run is
    // deliberately expired below, so it *does* fail with its admission error
    // and must keep reporting it: a cancellation that swallowed that would
    // leave a caller believing an unlaunched run had run.
    std::exception_ptr waiter_error;
    std::chrono::steady_clock::duration waited{};
    {
        Joined waiter(std::thread([&] {
            const auto started = std::chrono::steady_clock::now();
            try {
                (void)run.wait_until(started + std::chrono::seconds(30));
            } catch (...) {
                waiter_error = std::current_exception();
            }
            waited = std::chrono::steady_clock::now() - started;
        }));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (g_credit_waits.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        EXPECT_GT(g_credit_waits.load(), 0) << "the waiter never parked on the credit signal";
        run.expire_credit_deadline_for_test();
        EXPECT_NO_THROW(lane.close());
        EXPECT_TRUE(g_cancelled.load()) << "close drained without cancelling the waits first";
    }

    // Bounded: the waiter came back on the cancellation rather than on its own
    // 30 s budget. The margin is wide because this asserts "woken, not waited
    // out", and the stub's wait quantum is the lane's own.
    EXPECT_LT(waited, std::chrono::seconds(20)) << "the waiter was left to time out instead of being cancelled";

    // And it came back with this run's own cause, which the shutdown neither
    // replaced nor suppressed.
    ASSERT_TRUE(waiter_error) << "an expired run reported no failure at all";
    std::string message;
    try {
        std::rethrow_exception(waiter_error);
    } catch (const std::exception &e) {
        message = e.what();
    }
    EXPECT_NE(message.find("not admitted"), std::string::npos)
        << "the cancellation replaced this run's admission failure: " << message;

    // Retired exactly once, and nothing took its slot before it was: a
    // cancelled wait must leave no carrier for a later run to prepare over.
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), std::string("finalize0")), 1)
        << "the cancelled run's native carrier was retired " << "more or less than once";
    worker.finalize();
}

// --- What an expired deadline retires -----------------------------------

TEST(ChipRunLaneCreditBlockTest, AnExpiredDeadlineRetiresThePreparedCarrier) {
    // A run that reached PREPARED owns a native run on the worker: the
    // runtime's prepared state, its bindings and its entry in `native_runs_`.
    // Only `finalize_native_run` destroys that. Marking the run terminal
    // without it leaves the carrier live for the worker's whole life, and the
    // slot's reuse becomes a claim rather than a fact — which is invisible to
    // an assertion that only reads the thrown error.
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 401, 0, /*activate=*/true, diagnostic_config());
    ASSERT_FALSE(run.launched());
    run.expire_credit_deadline_for_test();
    EXPECT_THROW(run.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(1)), std::runtime_error);

    EXPECT_NE(std::find(g_events.begin(), g_events.end(), "finalize0"), g_events.end())
        << "the expired run's native carrier was never finalized";

    // And the slot really is reusable: a later run takes it and launches.
    g_launch_rc[0] = 0;
    ChipRun next = submit(lane, 402, 0, /*activate=*/true, diagnostic_config());
    EXPECT_TRUE(next.launched()) << "the slot an expired run gave up could not be reused";
    g_complete[0] = true;
    EXPECT_TRUE(next.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, ARepeatedExpiryFinalizesTheCarrierOnce) {
    // The deadline check runs on every progress round. A run already retired
    // must not be finalized again — a second destruction of the same carrier
    // is the defect this guards, and the count is the only thing showing it.
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;

    ChipRun run = submit(lane, 403, 0, /*activate=*/true, diagnostic_config());
    run.expire_credit_deadline_for_test();
    for (int round = 0; round < 3; round++) {
        try {
            (void)run.wait_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(5));
        } catch (const std::runtime_error &) {
            // Its own failure, re-reported each round, which is expected.
        }
    }
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), std::string("finalize0")), 1)
        << "a retired carrier was finalized more than once";
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneCreditBlockTest, AFailedCleanupKeepsTheAdmissionFailureAsTheFirstCause) {
    // Two facts, and the order between them matters: the run did not launch
    // because capacity never came back, and the cleanup that followed also
    // failed. The first is this run's own error and must survive; the second
    // poisons the lane, because an un-retired carrier is exactly what a later
    // run must not prepare over.
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
    g_finalize_throws = true;

    ChipRun run = submit(lane, 404, 0, /*activate=*/true, diagnostic_config());
    run.expire_credit_deadline_for_test();
    std::string message;
    try {
        (void)run.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(1));
        ADD_FAILURE() << "an expired run reported no failure at all";
    } catch (const std::runtime_error &e) {
        message = e.what();
    }
    EXPECT_NE(message.find("not admitted"), std::string::npos)
        << "the cleanup failure replaced the admission failure as this run's cause: " << message;
    EXPECT_TRUE(run.lane_poisoned()) << "a carrier that could not be retired left the lane usable";
    g_finalize_throws = false;
    worker.finalize();
}

// --- Close refuses before it mutates ------------------------------------

TEST(ChipRunLaneCreditBlockTest, AWrongContextCloseChangesNothing) {
    // A close on the wrong device context is not a close: it throws, and the
    // worker it was called on is still open. Cancelling first would detach
    // that open worker's credit entries on the way to throwing, leaving a lane
    // that can never wait for capacity again and no record of why.
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    g_validate_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    EXPECT_THROW(lane.close(), std::runtime_error);
    EXPECT_FALSE(g_cancelled.load()) << "a refused close cancelled the still-open worker's credit waits";

    // And the lane is unchanged: with the context restored it closes normally.
    g_validate_rc = 0;
    EXPECT_NO_THROW(lane.close());
    EXPECT_TRUE(g_cancelled.load()) << "the accepted close did not cancel";
    worker.finalize();
}
