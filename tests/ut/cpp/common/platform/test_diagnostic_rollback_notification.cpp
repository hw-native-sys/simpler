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
 * Which of a refused admission's own notifications a waiter must still see.
 *
 * An all-or-nothing attempt that takes some owners' capacity and gives it back
 * when a later one reports full runs its undo through the same release paths a
 * real publication takes. Notifying there wakes the attempting thread onto its
 * own generation bump, and a repeated attempt then manufactures its own wake
 * every round while the blocking owner frees nothing — so the undo of a
 * *restoration* is suppressed, thread-locally, for exactly the thread undoing
 * its own attempt.
 *
 * That suppression is the whole risk this case exists for. A withdrawal does
 * not always restore: an owner that cannot prove its references released
 * quarantines the run, and a fatal one defers its storage to teardown. Then
 * the capacity every waiter last observed is gone for good, which is an event
 * and not the absence of one, and a suppression wide enough to swallow it
 * leaves the lane parked until its deadline instead of re-attempting and being
 * told. So the suppression covers a restoration and nothing else:
 * `undo_admission_attempt` holds it across the withdrawal and publishes an
 * advance itself when the withdrawal reports that some owner's slot did not
 * come back.
 *
 * The first five cases drive the production runner — a real admission, the real
 * `undo_admission_attempt`, and the release notifier the arming path installs.
 * One is the restoration, one is its positive control (because "the generation
 * did not move" means nothing from a runner whose notifier was never wired),
 * and two are the non-restorations the suppression used to swallow: a
 * withdrawal that could not prove its references released, and one naming a run
 * the background writer already owns. The last of the five is the restoring
 * control for those, differing from them by exactly one call. The remaining
 * cases pin the signal's own half: what a suppression does not reach.
 */

#include "runtime_c_api.h"

#include <gtest/gtest.h>

#include "device_runner_base.h"
#include "host/diagnostic_credit_signal.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace credit = simpler::dfx::credit;

namespace {

/** The simulated device this runner's collector threads bind to. */
constexpr int kSimDeviceId = 0;

void *rollback_alloc(size_t size) { return std::calloc(1, size); }

int rollback_free(void *ptr) {
    std::free(ptr);
    return 0;
}

/** A destination tree this case owns, removed with it. */
struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() / ("dfx_rollback_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }
};

/**
 * A sim runner that never launches anything.
 *
 * Admission, withdrawal, the credit signal and the release notifiers are the
 * production base's; what this adds is the ability to stand one owner up and
 * to read the generation, which is what the lane reads through the C entry.
 */
class RollbackRunner : public SimDeviceRunnerBase {
public:
    /** Retention on, ScopeStats initialized, and the notifiers wired. */
    bool stand_up_scope_stats() {
        set_retain_runs(true);
        if (scope_stats_collector_.init(/*num_threads=*/1, rollback_alloc, nullptr, rollback_free, 0) != 0) {
            return false;
        }
        // The arming path's first call, made here because this case arms
        // nothing: until it runs, every collector's release notifier is empty
        // and no release could move the generation at all.
        install_credit_release_notifiers();
        return true;
    }

    /** Take a slot directly, the way a run ahead of this one holds one. */
    bool admit_directly(uint64_t run_epoch, const std::string &prefix) {
        scope_stats_collector_.start([](std::function<void()> fn) {
            return std::thread(std::move(fn));
        });
        return scope_stats_collector_.run_begin(run_epoch, prefix);
    }

    /** Give a slot back the way a writer does: outside any suppression. */
    void release_directly(uint64_t run_epoch) { scope_stats_collector_.abandon_run(run_epoch); }

    bool owner_holds_nothing() { return scope_stats_collector_.retained_runs_idle(); }

    void shut_down() { scope_stats_collector_.finalize(nullptr, rollback_free); }

    /**
     * Retention on, ArgsDump initialized, and the notifiers wired.
     *
     * This owner rather than ScopeStats for the non-restoring cases, because it
     * is one whose withdrawal can report that a slot did not come back:
     * `abandon_run` refuses a run whose boundary has already been closed, and
     * ScopeStats frees its slot under its own lock with nothing to refuse.
     */
    bool stand_up_args_dump() {
        if (attach_current_thread(kSimDeviceId) != 0) return false;
        set_retain_runs(true);
        if (dump_collector_.initialize(
                /*num_dump_threads=*/1, kSimDeviceId, DumpArgsLevel::FULL, rollback_alloc, nullptr, rollback_free
            ) != 0) {
            return false;
        }
        install_credit_release_notifiers();
        return true;
    }

    /** Join the reader shards, so no one is left to acknowledge a release. */
    void join_args_dump_shards() { dump_collector_.stop(); }

    void shut_down_args_dump() {
        // Released first: the writer has to be able to drain what it was held
        // off, and `finish_retained_runs` is what waits for it.
        dump_collector_.hold_retained_writer_for_test(false);
        dump_collector_.finish_retained_runs();
        dump_collector_.stop();
        (void)dump_collector_.finalize(nullptr, rollback_free);
    }

    /**
     * Hold the background writer where it would take a closed run.
     *
     * The collector's own test seam, and what makes the non-restoring case
     * deterministic: a closed run is the writer's, and the writer moves it out
     * of `Admitting` as soon as it is woken, which is the state the withdrawal
     * reads. Held, the state the production undo decides on stays put.
     */
    void hold_writer(bool held) { dump_collector_.hold_retained_writer_for_test(held); }

    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override {
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution>, LaunchPermit) override {
        return LaunchOutcome{};
    }
    void abandon_prepared_execution(PreparedExecution &) noexcept override {}
    int poll_execution(const ActiveExecution &) override { return SIMPLER_NATIVE_RUN_POLL_ERROR; }
    DrainOutcome drain_execution(ActiveExecution &) override { return DrainOutcome::device_error(-1); }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs &) override { return 0; }
};

/** One owner on, writing under `prefix`. */
DfxRunConfig scope_stats_only(const fs::path &prefix) {
    CallConfig config;
    config.enable_scope_stats = 1;
    std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s", prefix.c_str());
    return DfxRunConfig::from(config);
}

/** The other owner this case drives, on the same terms. */
DfxRunConfig args_dump_only(const fs::path &prefix) {
    CallConfig config;
    config.enable_dump_args = true;
    std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s", prefix.c_str());
    return DfxRunConfig::from(config);
}

/** What the lane's own wait entry answers for a generation taken before an undo. */
int credit_wait(RollbackRunner &runner, uint64_t seen) {
    return simpler_wait_diagnostic_credit_ctx(static_cast<DeviceContextHandle>(&runner), seen, /*timeout_ms=*/0);
}

}  // namespace

TEST(DiagnosticRollbackNotificationTest, AReleaseOutsideAnUndoIsAlwaysPublished) {
    // The positive control for the case below: with the notifier installed, a
    // slot going back on an ordinary path *does* move the generation. Without
    // this, "the generation did not move" would be satisfied by a runner that
    // never wired its notifiers at all.
    OutputRoot root("positive-control");
    RollbackRunner runner;
    ASSERT_TRUE(runner.stand_up_scope_stats());
    ASSERT_TRUE(runner.admit_directly(5100, (root.base / "ahead").string()));

    const uint64_t before = runner.credit_generation();
    runner.release_directly(5100);
    EXPECT_GT(runner.credit_generation(), before) << "an ordinary release raised nothing a waiter could observe";
    EXPECT_TRUE(runner.owner_holds_nothing());

    runner.shut_down();
}

TEST(DiagnosticRollbackNotificationTest, ARestoringUndoWakesNobody) {
    // The admission is real, the undo is the production helper every decline
    // goes through, and the owner it withdraws is the one it admitted. The
    // state afterwards is exactly the pre-attempt state, so there is nothing
    // for any waiter to learn — and a bump here is what a retry would wake
    // itself on, round after round, until its deadline.
    OutputRoot root("restoring-undo");
    RollbackRunner runner;
    ASSERT_TRUE(runner.stand_up_scope_stats());

    const DfxRunConfig dfx = scope_stats_only(root.base / "attempt");
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, 5200, metadata), 0) << "the attempt was not admitted at all";
    ASSERT_FALSE(runner.owner_holds_nothing()) << "an admitted run left the owner reading as idle";

    const uint64_t before = runner.credit_generation();
    runner.undo_admission_attempt(dfx, 5200);

    EXPECT_EQ(runner.credit_generation(), before) << "undoing this attempt woke the thread that made it";
    EXPECT_TRUE(runner.owner_holds_nothing()) << "the undo did not give the admitted slot back";
    // And the slot really is re-admissible, which is the property the waiter
    // would have been told about if there were anything to tell.
    EXPECT_TRUE(runner.admit_directly(5201, (root.base / "after").string()));
    runner.release_directly(5201);

    runner.shut_down();
}

TEST(DiagnosticRollbackNotificationTest, AnUndoThatQuarantinedThisRunPublishesIt) {
    // The path the suppression used to swallow, end to end and through the
    // production undo: the withdrawal cannot prove this run's references were
    // released, so it quarantines the run and publishes a fatal — from inside
    // the suppressed span, where that publication is dropped. The undo's own
    // advance is what is left to tell a waiter, and without it the lane parks
    // on capacity that is gone until its 30 s deadline.
    OutputRoot root("quarantining-undo");
    RollbackRunner runner;
    ASSERT_TRUE(runner.stand_up_args_dump());

    const DfxRunConfig dfx = args_dump_only(root.base / "attempt");
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, 5320, metadata), 0) << "the attempt was not admitted at all";
    // Joining the reader shards is what makes the withdrawal unprovable: no
    // shard is left to acknowledge that it dropped its reference, exactly as
    // `AnUnprovableWithdrawalQuarantinesAndPublishesItsFailure` arranges it for
    // the swimlane collector.
    runner.join_args_dump_shards();

    const uint64_t before = runner.credit_generation();
    runner.undo_admission_attempt(dfx, 5320);

    EXPECT_NE(runner.credit_generation(), before) << "a quarantining withdrawal woke nobody";
    EXPECT_EQ(credit_wait(runner, before), static_cast<int>(credit::CreditWait::Advanced))
        << "the fatal this withdrawal published never reached a waiter";

    runner.shut_down_args_dump();
}

TEST(DiagnosticRollbackNotificationTest, AnUndoTheWriterAlreadyOwnsThisRunPublishesIt) {
    // The production decision, through the production undo. Closing the run's
    // boundary hands it to the background writer, which is a state a withdrawal
    // cannot take back: `abandon_run` reports that the slot did not come back,
    // and the capacity every waiter last observed is gone for good. The undo
    // must publish that, and a waiter holding a generation taken before it must
    // be told to re-attempt rather than park to its deadline.
    OutputRoot root("non-restoring-undo");
    RollbackRunner runner;
    ASSERT_TRUE(runner.stand_up_args_dump());
    // Held before the run is closed, so the writer cannot move the epoch out of
    // the state the withdrawal is about between the close and the undo.
    runner.hold_writer(true);

    const DfxRunConfig dfx = args_dump_only(root.base / "attempt");
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, 5300, metadata), 0) << "the attempt was not admitted at all";
    // `device_execution_complete=false` is the cheap close: it reads nothing
    // device-side, and it installs the writer's target exactly as the proved
    // one does, which is the state under test here. The stop proof is withheld
    // so the close settles nothing about the bank — this case is about the
    // withdrawal's own decision, not the boundary's.
    ASSERT_EQ(
        runner.close_args_dump_run_boundary(
            dfx, 5300, /*device_execution_complete=*/false, /*device_producers_stopped=*/false
        ),
        0
    );

    // After the close, which publishes its own progress: what is measured here
    // is the undo alone.
    const uint64_t before = runner.credit_generation();
    runner.undo_admission_attempt(dfx, 5300);

    EXPECT_NE(runner.credit_generation(), before) << "a withdrawal that restored nothing woke nobody";
    EXPECT_EQ(credit_wait(runner, before), static_cast<int>(credit::CreditWait::Advanced))
        << "a waiter that read the generation before the undo was left parked on capacity that is gone";

    runner.shut_down_args_dump();
}

TEST(DiagnosticRollbackNotificationTest, TheSameUndoBeforeTheWriterTakesTheRunWakesNobody) {
    // The control for the case above, differing by exactly one call: without
    // the boundary close the slot is still this thread's, the withdrawal gives
    // it back, and the pre-attempt state is restored — so there is nothing to
    // tell, and telling would be the self-wake this suppression exists for.
    OutputRoot root("restoring-undo-args-dump");
    RollbackRunner runner;
    ASSERT_TRUE(runner.stand_up_args_dump());

    const DfxRunConfig dfx = args_dump_only(root.base / "attempt");
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, 5310, metadata), 0);

    const uint64_t before = runner.credit_generation();
    runner.undo_admission_attempt(dfx, 5310);

    EXPECT_EQ(runner.credit_generation(), before) << "undoing this attempt woke the thread that made it";
    EXPECT_EQ(credit_wait(runner, before), static_cast<int>(credit::CreditWait::TimedOut));
    // The slot really did come back, which is what makes the generation's
    // stillness a restoration rather than a dropped notification. Under its own
    // destination: a withdrawn run's reserved names stay on disk as the names it
    // owned, and this collector refuses a second run under them — which is a
    // different refusal from the one being measured here.
    const DfxRunConfig next = args_dump_only(root.base / "next-attempt");
    ASSERT_EQ(runner.start_shared_collectors_for_run(next, 5311, metadata), 0)
        << "the withdrawn slot was not available to the next attempt";
    runner.undo_admission_attempt(next, 5311);

    runner.shut_down_args_dump();
}

TEST(DiagnosticRollbackNotificationTest, AnAdvanceAfterASuppressedSpanReachesItsWaiters) {
    // The signal's own half of the contract the two cases above exercise: what a
    // suppression covers while it is held, and that an advance raised after it —
    // which is what the undo does when it restored nothing — reaches a waiter
    // parked on the pre-attempt generation.
    credit::DiagnosticCreditSignal signal;
    const uint64_t before = signal.generation();
    {
        credit::DiagnosticCreditSignal::SuppressSelfNotify quiet;
        signal.advance();
        signal.advance();
    }
    ASSERT_EQ(signal.generation(), before) << "a suppressed undo moved the generation";
    EXPECT_EQ(signal.wait_for(before, 0), credit::CreditWait::TimedOut);

    signal.advance();
    EXPECT_NE(signal.generation(), before);
    EXPECT_EQ(signal.wait_for(before, 0), credit::CreditWait::Advanced)
        << "a waiter parked on the pre-attempt generation was not told the undo restored nothing";
}

TEST(DiagnosticRollbackNotificationTest, AReleaseOnAnotherThreadIsNeverSuppressed) {
    // What makes the suppression safe to hold at all: it is this thread's, and
    // the owner that frees capacity publishes on its own writer thread. A
    // collector that released while an attempt was being undone must still
    // wake it, or the attempt parks against a signal that was already raised.
    credit::DiagnosticCreditSignal signal;
    const uint64_t before = signal.generation();

    credit::DiagnosticCreditSignal::SuppressSelfNotify quiet;
    signal.advance();
    ASSERT_EQ(signal.generation(), before);

    std::thread writer([&signal] {
        signal.advance();
    });
    writer.join();

    EXPECT_NE(signal.generation(), before) << "a release on another thread was suppressed by this thread's undo";
    EXPECT_EQ(signal.wait_for(before, 0), credit::CreditWait::Advanced);
}

TEST(DiagnosticRollbackNotificationTest, ACancellationIsNeverSuppressed) {
    // Teardown is the one event a parked waiter cannot afford to miss, and it
    // can be published by a thread that is itself inside a suppressed span —
    // a close reached from a failed admission's own path. It does not go
    // through `advance` and is therefore outside the suppression by
    // construction; this is what holds that apart.
    credit::DiagnosticCreditSignal signal;
    const uint64_t before = signal.generation();

    credit::DiagnosticCreditSignal::SuppressSelfNotify quiet;
    signal.advance();
    ASSERT_EQ(signal.generation(), before);

    signal.cancel();
    EXPECT_TRUE(signal.cancelled());
    EXPECT_EQ(signal.wait_for(before, 0), credit::CreditWait::Cancelled)
        << "a cancellation raised inside an undo left waiters parked";
}

TEST(DiagnosticRollbackNotificationTest, ASuppressionEndsWithItsScope) {
    // Thread-local depth, not a latch: a nested span must not leave the thread
    // permanently unable to publish. Every later release on this thread is a
    // real one.
    credit::DiagnosticCreditSignal signal;
    {
        credit::DiagnosticCreditSignal::SuppressSelfNotify outer;
        {
            credit::DiagnosticCreditSignal::SuppressSelfNotify inner;
            signal.advance();
        }
        signal.advance();
    }
    const uint64_t after_undo = signal.generation();
    signal.advance();
    EXPECT_EQ(signal.generation(), after_undo + 1) << "a suppression outlived the undo that took it";
}
