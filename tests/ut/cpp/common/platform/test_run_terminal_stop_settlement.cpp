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
 * What a failed run's terminal bank does, per owner, on the producer-stop proof.
 *
 * Two facts decide it and they are independent. *This run completed* is the
 * caller's fence observation, and without it the close reads nothing and seals
 * nothing. *No producer can still write into this run's bank* is a separate
 * observation only a runner whose producers are host threads can make. The
 * first decides whether a snapshot is read; the second decides whether the
 * device storage comes back or is withdrawn.
 *
 * Each owner is driven through its own production stand-up — the same
 * `configure_retained_runs` / `init` / `open_run_terminal_banks` recipe
 * `test_run_terminal_release_policy.cpp` uses, which covers the *finalize*
 * allocator choice and nothing per-run — and then through a real admission, a
 * real failed close and the production
 * `simpler::dfx::runs::settle_failed_run_terminal_bank`. The observable is each
 * collector's own bank bookkeeping, by run identity.
 *
 * A second suite covers the moment the two facts are easiest to confuse: the
 * *host* writer finishing with a failed run. For the two owners whose writer
 * retires such a run's slot, that is where a bank can be handed back without
 * any statement about the device, so the writer is let go before those cases
 * assert. The withdrawal of a run that never launched is the control beside
 * them: that bank comes back with no proof at all, because no producer was
 * ever given it.
 *
 * The last suite drives the stop itself. `stop_run_producers` performs the
 * stop and is the only thing that raises the proof, and
 * `settle_run_after_producer_stop` reads it, so these cases execute the
 * ordering between the two, what a stop that threw leaves behind, and whether
 * the bank a stopped run returned is the next run's. What they stand in for is
 * the join those entries carry on a real simulated run: reaching that needs a
 * device, so the stop here is the seam's own invocation of a functor rather
 * than threads.
 */

#include "runtime_c_api.h"

#include <gtest/gtest.h>

#include "device_runner_base.h"
#include "host/args_dump_collector.h"
#include "host/dep_gen_collector.h"
#include "host/pmu_collector.h"
#include "host/run_boundary.h"
#include "host/scope_stats_collector.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace runs = simpler::dfx::runs;

namespace {

constexpr int kSimDeviceId = 0;
/** Two runs open at once, which is what makes "only this run's bank" a claim. */
constexpr uint64_t kFirstRun = 7100;
constexpr uint64_t kSecondRun = 7200;
/** An identity no collector here was ever asked about. */
constexpr uint64_t kForeignRun = 7999;

void *settle_alloc(size_t size) { return std::calloc(1, size); }

int settle_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread settle_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

/** A destination tree this case owns, removed with it. */
struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() / ("dfx_settle_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    /** One run's destination, created: a collector names a file inside it. */
    std::string prefix(uint64_t run_epoch) const {
        const fs::path dir = base / ("run-" + std::to_string(run_epoch));
        std::error_code ec;
        fs::create_directories(dir, ec);
        return dir.string();
    }
};

/**
 * One owner, stood up and driven the way its runner drives it.
 *
 * Specialized per collector because the stand-up differs — PMU counts cores
 * where the others count threads, ArgsDump carries a level — while everything
 * the cases assert is the same question asked of each.
 */
template <typename Collector>
struct Owner;

template <>
struct Owner<ScopeStatsCollector> {
    static constexpr const char *kName = "scope_stats";
    ScopeStatsCollector collector;

    bool stand_up() {
        collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
        if (collector.init(/*num_threads=*/1, settle_alloc, nullptr, settle_free, kSimDeviceId) != 0) return false;
        if (!collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS)) return false;
        collector.start(settle_thread_factory);
        return true;
    }
    bool admit(uint64_t run_epoch, const std::string &prefix) {
        const simpler::dfx::scope_stats_runs::CapacityMetadata caps{
            true, simpler::dfx::scope_stats_runs::Capacities{{8, 8, 8, 8}, {4, 4, 4, 4}, {1024, 1024, 1024, 1024}, 16}
        };
        return collector.run_begin(run_epoch, prefix, nullptr, &caps);
    }
    void close_failed(uint64_t run_epoch) { collector.run_close(run_epoch, /*device_execution_complete=*/false); }
    uint64_t bank_of(uint64_t run_epoch) const { return collector.run_terminal_bank_address(run_epoch); }
    size_t banks_in_use() const { return collector.run_terminal_banks_in_use(); }
    bool host_state_released() const { return collector.retained_runs_idle(); }
    void shut_down() { collector.finalize(nullptr, settle_free); }
};

template <>
struct Owner<DepGenCollector> {
    static constexpr const char *kName = "dep_gen";
    DepGenCollector collector;

    bool stand_up() {
        collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
        if (collector.init(/*num_threads=*/1, settle_alloc, nullptr, settle_free, kSimDeviceId) != 0) return false;
        if (!collector.open_run_terminal_banks(DEP_GEN_TERMINAL_PRODUCERS)) return false;
        collector.start(settle_thread_factory);
        return true;
    }
    bool admit(uint64_t run_epoch, const std::string &prefix) { return collector.run_begin(run_epoch, prefix); }
    void close_failed(uint64_t run_epoch) { collector.run_close(run_epoch, /*device_execution_complete=*/false); }
    uint64_t bank_of(uint64_t run_epoch) const { return collector.run_terminal_bank_address(run_epoch); }
    size_t banks_in_use() const { return collector.run_terminal_banks_in_use(); }
    bool host_state_released() const { return collector.retained_runs_idle(); }
    void shut_down() { collector.finalize(nullptr, settle_free); }
};

template <>
struct Owner<ArgsDumpCollector> {
    static constexpr const char *kName = "dump_args";
    ArgsDumpCollector collector;

    bool stand_up() {
        collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
        if (collector.initialize(
                /*num_dump_threads=*/1, kSimDeviceId, DumpArgsLevel::FULL, settle_alloc, nullptr, settle_free
            ) != 0) {
            return false;
        }
        if (!collector.open_run_terminal_banks(/*expected_producers=*/1)) return false;
        collector.start(settle_thread_factory);
        // This owner's failed close installs the writer's target, and the
        // writer then frees the epoch — including the reservation these cases
        // are about. Held at the collector's own seam so the state the
        // settlement decides on stays put.
        collector.hold_retained_writer_for_test(true);
        return true;
    }
    bool admit(uint64_t run_epoch, const std::string &prefix) {
        return collector.run_begin(run_epoch, prefix, DumpArgsLevel::FULL);
    }
    void close_failed(uint64_t run_epoch) { (void)collector.run_close(run_epoch, /*device_execution_complete=*/false); }
    uint64_t bank_of(uint64_t run_epoch) const { return collector.run_terminal_bank_address(run_epoch); }
    size_t banks_in_use() const { return collector.run_terminal_banks_in_use(); }

    /**
     * Let the background writer finish the runs it holds.
     *
     * The hold is released and the wait is for the writer's own retirement, not
     * for a fixed interval: these cases are about what the collector owns once
     * that retirement has actually happened, so a case that asserted while the
     * writer was still held would be asserting about the hold.
     */
    bool let_writer_finish() {
        collector.hold_retained_writer_for_test(false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            if (collector.retained_runs_idle()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return collector.retained_runs_idle();
    }

    bool host_state_released() const { return collector.retained_runs_idle(); }
    void shut_down() {
        collector.hold_retained_writer_for_test(false);
        collector.finish_retained_runs();
        collector.stop();
        (void)collector.finalize(nullptr, settle_free);
    }
};

template <>
struct Owner<PmuCollector> {
    static constexpr const char *kName = "pmu";
    PmuCollector collector;

    bool stand_up() {
        collector.configure_retained_runs(true);
        if (collector.init(
                /*num_aicore=*/1, /*aicpu_threads=*/1, settle_alloc, nullptr, settle_free, kSimDeviceId
            ) != 0) {
            return false;
        }
        if (!collector.open_run_terminal_banks(/*expected_producers=*/1)) return false;
        collector.start(settle_thread_factory);
        // Held for the reason ArgsDump's is: this owner's close installs the
        // writer's target on both paths.
        collector.hold_retained_writer_for_test(true);
        return true;
    }
    bool admit(uint64_t run_epoch, const std::string &prefix) {
        return collector.run_begin(run_epoch, prefix + "/pmu.csv", PmuEventType::PIPE_UTILIZATION);
    }
    void close_failed(uint64_t run_epoch) { collector.run_close(run_epoch, /*device_execution_complete=*/false); }
    uint64_t bank_of(uint64_t run_epoch) const { return collector.run_terminal_bank_address(run_epoch); }
    size_t banks_in_use() const { return collector.run_terminal_banks_in_use(); }

    /**
     * Let the background writer finish the runs it holds.
     *
     * The hold is released and the wait is for the writer's own retirement, not
     * for a fixed interval: these cases are about what the collector owns once
     * that retirement has actually happened, so a case that asserted while the
     * writer was still held would be asserting about the hold.
     */
    bool let_writer_finish() {
        collector.hold_retained_writer_for_test(false);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            if (collector.retained_runs_idle()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return collector.retained_runs_idle();
    }

    bool host_state_released() const { return collector.retained_runs_idle(); }
    void shut_down() {
        collector.hold_retained_writer_for_test(false);
        collector.finish_retained_runs();
        collector.stop();
        (void)collector.finalize(nullptr, settle_free);
    }
};

template <typename Collector>
class RunTerminalStopSettlement : public ::testing::Test {};

using SettlingOwners = ::testing::Types<ScopeStatsCollector, DepGenCollector, ArgsDumpCollector, PmuCollector>;
TYPED_TEST_SUITE(RunTerminalStopSettlement, SettlingOwners);

}  // namespace

TYPED_TEST(RunTerminalStopSettlement, AnUnprovedStopKeepsTheFailedRunsBank) {
    // The conservative half, and the one a wrong fix would break: without the
    // proof a producer may still be writing into that bank, so the run's close
    // leaves it withdrawn and the settlement declines to hand it on.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    const uint64_t held = owner.bank_of(kFirstRun);
    ASSERT_NE(held, 0u) << "an admitted run was given no terminal bank";

    owner.close_failed(kFirstRun);
    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/false
    );

    EXPECT_EQ(owner.bank_of(kFirstRun), held) << "a bank came back with no proof that its producers had stopped";

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopSettlement, AProvedStopReturnsThisRunsBankAndNoOther) {
    // Two runs hold two banks. The settlement names one epoch, so the other
    // run's storage must be untouched — a bank is held by the epoch that
    // reserved it, and releasing by anything coarser would hand a live
    // producer's address to the next run.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_TRUE(owner.admit(kSecondRun, root.prefix(kSecondRun)));
    const uint64_t first = owner.bank_of(kFirstRun);
    const uint64_t second = owner.bank_of(kSecondRun);
    ASSERT_NE(first, 0u);
    ASSERT_NE(second, 0u);
    ASSERT_NE(first, second) << "two runs open at once were given the same bank";

    owner.close_failed(kFirstRun);
    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );

    EXPECT_EQ(owner.bank_of(kFirstRun), 0u) << "a proved-stopped failed run kept its bank";
    EXPECT_EQ(owner.bank_of(kSecondRun), second) << "settling one run's bank moved another run's";

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopSettlement, AProvedStopFreesNoHostStateWithIt) {
    // The proof is about device producers and about nothing else. This run's
    // host records, files and slot stay exactly where a failed run leaves them:
    // their readers are this collector's own threads, which the proof does not
    // address, and freeing them here is the premature release the quarantine
    // exists to prevent.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_NE(owner.bank_of(kFirstRun), 0u);

    owner.close_failed(kFirstRun);
    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );

    EXPECT_EQ(owner.bank_of(kFirstRun), 0u) << "the device storage did not come back";
    EXPECT_FALSE(owner.host_state_released())
        << "the host state this failed run holds was released on a proof that only covers its producers";

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopSettlement, ARepeatedOrForeignSettlementReleasesNothing) {
    // Idempotence and identity, which are the two ways a settlement could reach
    // a bank that is not its own: the same epoch twice, and an epoch this
    // collector never admitted.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_TRUE(owner.admit(kSecondRun, root.prefix(kSecondRun)));
    const uint64_t second = owner.bank_of(kSecondRun);
    ASSERT_NE(second, 0u);

    owner.close_failed(kFirstRun);
    for (int round = 0; round < 3; round++) {
        runs::settle_failed_run_terminal_bank(
            owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
        );
    }
    runs::settle_failed_run_terminal_bank(
        owner.collector, kForeignRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );

    EXPECT_EQ(owner.bank_of(kFirstRun), 0u);
    EXPECT_EQ(owner.bank_of(kSecondRun), second)
        << "a repeated or foreign settlement released a bank belonging to another run";

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopSettlement, TheCompletionPathSettlesNothingHere) {
    // On the completion path the close consumed this run's snapshot and
    // released the bank with it, so a second settlement would be returning a
    // bank the run no longer holds. The helper declines whatever the stop
    // proof says, which is what keeps one decision in one place.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    const uint64_t held = owner.bank_of(kFirstRun);
    ASSERT_NE(held, 0u);

    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/true, /*device_producers_stopped=*/true
    );

    EXPECT_EQ(owner.bank_of(kFirstRun), held) << "the completion path's bank was settled by the failure path's rule";

    owner.shut_down();
}

// ---------------------------------------------------------------------------
// What the host writer finishing does, and does not, say about the device
// ---------------------------------------------------------------------------

namespace {

template <typename Collector>
class RunTerminalStopAfterTheWriter : public ::testing::Test {};

// The two owners whose background writer retires a failed run's slot. The other
// two quarantine such a run instead and never free its slot, so there is no
// writer completion for a bank of theirs to be returned by; `AnUnprovedStopKeeps
// TheFailedRunsBank` is where their half is held.
using WriterRetiringOwners = ::testing::Types<ArgsDumpCollector, PmuCollector>;
TYPED_TEST_SUITE(RunTerminalStopAfterTheWriter, WriterRetiringOwners);

}  // namespace

TYPED_TEST(RunTerminalStopAfterTheWriter, AFinishedWriterIsNoProofAboutTheProducers) {
    // The whole point of the two proofs being separate, at the one moment they
    // are easiest to confuse. The run failed, no stop was ever proved, and the
    // *host* writer has now finished with it — which says the host is done
    // reading, and nothing at all about whether this run's device producers
    // stopped writing their bank. Returning it here is the premature free, and
    // the writer's own slot release is where it used to happen.
    //
    // Measured against the pool rather than against the run's slot, because the
    // slot is legitimately gone by then: `bank_of` can only answer while a run
    // still has one.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_NE(owner.bank_of(kFirstRun), 0u) << "an admitted run was given no terminal bank";
    ASSERT_EQ(owner.banks_in_use(), 1u);

    owner.close_failed(kFirstRun);
    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/false
    );
    ASSERT_TRUE(owner.let_writer_finish()) << "the writer never retired the failed run";

    EXPECT_EQ(
        owner.banks_in_use(), 1u
    ) << "the host writer finishing returned a bank no producer was proved to have stopped writing";

    // And the proof still reaches exactly this run's bank afterwards: the
    // capacity is withheld, not abandoned.
    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );
    EXPECT_EQ(owner.banks_in_use(), 0u) << "a stop proof arriving after the writer could not reach that run's bank";

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopAfterTheWriter, AProofAfterTheWriterNamesOneRunOnly) {
    // Two failed runs, both retired by the writer, one proof. Identity has to
    // survive the slot that carried it, or the second run's producers are handed
    // their bank away on the first run's evidence.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_TRUE(owner.admit(kSecondRun, root.prefix(kSecondRun)));
    ASSERT_NE(owner.bank_of(kFirstRun), 0u);
    ASSERT_NE(owner.bank_of(kSecondRun), 0u);

    owner.close_failed(kFirstRun);
    owner.close_failed(kSecondRun);
    ASSERT_TRUE(owner.let_writer_finish()) << "the writer never retired both failed runs";
    ASSERT_EQ(owner.banks_in_use(), 2u) << "the writer returned banks it had no proof about";

    runs::settle_failed_run_terminal_bank(
        owner.collector, kFirstRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );
    EXPECT_EQ(owner.banks_in_use(), 1u) << "one run's proof returned more than one run's bank";

    // A foreign identity reaches none of what is left.
    runs::settle_failed_run_terminal_bank(
        owner.collector, kForeignRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );
    EXPECT_EQ(owner.banks_in_use(), 1u) << "an epoch this collector never admitted released a parked bank";

    runs::settle_failed_run_terminal_bank(
        owner.collector, kSecondRun, /*device_execution_complete=*/false, /*device_producers_stopped=*/true
    );
    EXPECT_EQ(owner.banks_in_use(), 0u);

    owner.shut_down();
}

TYPED_TEST(RunTerminalStopAfterTheWriter, AWithdrawnRunGivesItsBankBackWithNoProofAtAll) {
    // The case the parking must not swallow: a run that was admitted and never
    // launched. No producer was ever handed that bank, so its return needs no
    // statement about the device — and a withdrawal that parked it instead would
    // starve the pool one bank per rolled-back admission.
    OutputRoot root(Owner<TypeParam>::kName);
    Owner<TypeParam> owner;
    ASSERT_TRUE(owner.stand_up());
    ASSERT_TRUE(owner.admit(kFirstRun, root.prefix(kFirstRun)));
    ASSERT_EQ(owner.banks_in_use(), 1u);

    EXPECT_TRUE(owner.collector.abandon_run(kFirstRun)) << "the withdrawal did not complete";
    EXPECT_EQ(owner.banks_in_use(), 0u) << "a run that never launched kept its bank out of the pool";

    // And the pool really is usable again, which a parked token would deny.
    EXPECT_TRUE(owner.admit(kSecondRun, root.prefix(kSecondRun)));
    EXPECT_NE(owner.bank_of(kSecondRun), 0u);

    owner.shut_down();
}

namespace {

/**
 * A sim runner that launches nothing, standing its ArgsDump owner up the way
 * the arch runner does.
 *
 * The teardown under test is the production base's, and the proved case reaches
 * it the way the arch runner does: through `settle_run_with_joined_producers`,
 * which is where the simulated platform's stop proof is written. So what a case
 * supplies is the run's own result, and the proof comes from production code.
 */
class SettlementRunner : public SimDeviceRunnerBase {
public:
    bool stand_up() {
        if (attach_current_thread(kSimDeviceId) != 0) return false;
        if (!set_run_resource_grant(1)) return false;
        set_retain_runs(true);
        if (dump_collector_.initialize(
                /*num_dump_threads=*/1, kSimDeviceId, DumpArgsLevel::FULL, settle_alloc, nullptr, settle_free
            ) != 0) {
            return false;
        }
        if (!dump_collector_.open_run_terminal_banks(/*expected_producers=*/1)) return false;
        install_credit_release_notifiers();
        // The boundary under test installs this owner's writer target, and the
        // writer then frees the epoch. Held at the collector's own seam so the
        // state the teardown decides on is the state these cases read.
        dump_collector_.hold_retained_writer_for_test(true);
        return true;
    }

    uint64_t dump_bank_of(uint64_t run_epoch) const { return dump_collector_.run_terminal_bank_address(run_epoch); }
    size_t dump_banks_in_use() const { return dump_collector_.run_terminal_banks_in_use(); }
    bool dump_host_state_released() const { return dump_collector_.retained_runs_idle(); }

    void shut_down() {
        dump_collector_.hold_retained_writer_for_test(false);
        dump_collector_.finish_retained_runs();
        dump_collector_.stop();
        (void)dump_collector_.finalize(nullptr, settle_free);
    }

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
DfxRunConfig args_dump_only(const std::string &prefix) {
    CallConfig config;
    config.enable_dump_args = true;
    std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s", prefix.c_str());
    return DfxRunConfig::from(config);
}

}  // namespace

TEST(RunTerminalStopSettlementThroughTheRunner, AFailedTeardownThatProvedTheStopReturnsTheBank) {
    // The whole chain, in production code: the base's teardown, this owner's
    // own run boundary, the shared settlement and the collector's bookkeeping.
    // This is the path a simulated run's error exit takes after
    // `active_run_->join()`, which is where that `true` comes from.
    OutputRoot root("runner-proved");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());

    const DfxRunConfig dfx = args_dump_only(root.prefix(kFirstRun));
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, kFirstRun, metadata), 0);
    ASSERT_NE(runner.dump_bank_of(kFirstRun), 0u) << "the admitted run was given no terminal bank";

    // The stop, performed through the production seam, and then the production
    // settlement reading what that stop left. No boolean is handed in.
    runner.stop_run_producers(kFirstRun, [] {});
    const int rc =
        runner.settle_run_after_producer_stop(dfx, /*pipeline_slot=*/0, kFirstRun, /*device_execution_complete=*/false);
    EXPECT_EQ(rc, 0) << "the failed teardown reported an ownership failure of its own";

    EXPECT_EQ(runner.dump_bank_of(kFirstRun), 0u)
        << "a teardown that proved every producer stopped still withdrew this run's bank";
    EXPECT_FALSE(runner.dump_host_state_released()) << "the failed run's host state went with the device storage";

    runner.shut_down();
}

// --- The stop itself: what performs it, and in what order -----------------

TEST(RunTerminalStopSettlementThroughTheRunner, TheStopRunsBeforeTheSettlementAndIsWhatItReads) {
    // Ordering, observed rather than assumed: the functor the seam invokes
    // checks what the collector holds while it runs, so a settlement that ran
    // first — or a proof raised before the stop — is visible here. The bank is
    // still this run's at that instant, and only afterwards does the
    // settlement return it.
    OutputRoot root("runner-order");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());

    const DfxRunConfig dfx = args_dump_only(root.prefix(kFirstRun));
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, kFirstRun, metadata), 0);
    const uint64_t held = runner.dump_bank_of(kFirstRun);
    ASSERT_NE(held, 0u);

    bool stop_ran = false;
    bool bank_still_held_during_stop = false;
    bool proof_raised_during_stop = true;
    runner.stop_run_producers(kFirstRun, [&] {
        stop_ran = true;
        bank_still_held_during_stop = runner.dump_bank_of(kFirstRun) == held;
        proof_raised_during_stop = runner.run_producers_stopped(kFirstRun);
    });

    EXPECT_TRUE(stop_ran) << "the seam did not perform the stop it owns";
    EXPECT_TRUE(bank_still_held_during_stop) << "the bank was settled before its producers were stopped";
    EXPECT_FALSE(proof_raised_during_stop) << "the proof was raised before the stop had returned";
    EXPECT_TRUE(runner.run_producers_stopped(kFirstRun)) << "a completed stop left no proof";
    EXPECT_FALSE(runner.run_producers_stopped(kSecondRun)) << "one run's stop answered for another";

    EXPECT_EQ(
        runner.settle_run_after_producer_stop(dfx, /*pipeline_slot=*/0, kFirstRun, /*device_execution_complete=*/false),
        0
    );
    EXPECT_EQ(runner.dump_bank_of(kFirstRun), 0u) << "the settlement did not act on the stop it was given";

    runner.shut_down();
}

TEST(RunTerminalStopSettlementThroughTheRunner, AStopThatThrewLeavesNoProofAndWithdrawsTheBank) {
    // A stop that failed part-way stopped nothing it can speak for. The failure
    // travels — the caller owns it — and what it must not leave behind is a
    // proof: the run settles as an unstopped one and keeps its bank withdrawn.
    OutputRoot root("runner-throwing-stop");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());

    const DfxRunConfig dfx = args_dump_only(root.prefix(kSecondRun));
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, kSecondRun, metadata), 0);
    const uint64_t held = runner.dump_bank_of(kSecondRun);
    ASSERT_NE(held, 0u);

    EXPECT_THROW(
        runner.stop_run_producers(
            kSecondRun,
            [] {
                throw std::runtime_error("a producer thread could not be joined");
            }
        ),
        std::runtime_error
    );
    EXPECT_FALSE(runner.run_producers_stopped(kSecondRun))
        << "a stop that threw still claimed its producers had stopped";

    EXPECT_EQ(
        runner.settle_run_after_producer_stop(
            dfx, /*pipeline_slot=*/0, kSecondRun, /*device_execution_complete=*/false
        ),
        0
    );
    EXPECT_EQ(runner.dump_bank_of(kSecondRun), held) << "a run whose stop failed handed its bank to the next one";

    runner.shut_down();
}

TEST(RunTerminalStopSettlementThroughTheRunner, AReturnedBankIsTheNextRunsAndAWithheldOneIsNot) {
    // What the two outcomes cost. A stopped failed run gives its own storage
    // back and the next run is handed that same bank; a run whose producers
    // were never stopped keeps its one, and the pool stays one short.
    //
    // The pool is what this reads, not a later admission: whether a further
    // run is admitted also depends on this collector's output slots and on
    // whether a quarantine has closed it, which are separate contracts with
    // their own cases below.
    OutputRoot root("runner-reuse");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());
    const RunLocalDfxMetadata metadata{};

    const DfxRunConfig stopped = args_dump_only(root.prefix(kFirstRun));
    ASSERT_EQ(runner.start_shared_collectors_for_run(stopped, kFirstRun, metadata), 0);
    const uint64_t returned = runner.dump_bank_of(kFirstRun);
    ASSERT_NE(returned, 0u) << "the admitted run was given no terminal bank";
    runner.stop_run_producers(kFirstRun, [] {});
    ASSERT_EQ(
        runner.settle_run_after_producer_stop(
            stopped, /*pipeline_slot=*/0, kFirstRun, /*device_execution_complete=*/false
        ),
        0
    );
    EXPECT_EQ(runner.dump_banks_in_use(), 0u) << "a stopped failed run kept its bank";

    const DfxRunConfig unstopped = args_dump_only(root.prefix(kSecondRun));
    ASSERT_EQ(runner.start_shared_collectors_for_run(unstopped, kSecondRun, metadata), 0);
    // The storage really came back: this run holds the very bank the first one
    // returned, which a pool that merely stopped counting it would not give.
    EXPECT_EQ(runner.dump_bank_of(kSecondRun), returned) << "the bank a proved stop returned was not re-issued";
    ASSERT_FALSE(runner.run_producers_stopped(kSecondRun)) << "the previous run's proof carried into this one";
    ASSERT_TRUE(runner.run_producers_stopped(kFirstRun))
        << "admitting a successor erased a predecessor's own stop proof";
    ASSERT_EQ(
        runner.settle_run_after_producer_stop(
            unstopped, /*pipeline_slot=*/0, kSecondRun, /*device_execution_complete=*/false
        ),
        0
    );
    EXPECT_EQ(runner.dump_banks_in_use(), 1u) << "a run whose producers were never stopped gave its bank back";
    EXPECT_EQ(runner.dump_bank_of(kSecondRun), returned) << "a withheld run's bank moved while it was withheld";

    runner.shut_down();
}

TEST(RunTerminalStopSettlementThroughTheRunner, AdmissionAfterTheseTwoRunsIsRefusedByTheOutputSlotsNotTheBanks) {
    // The two contracts are independent, and this is what separates them. The
    // bank pool has two of its three banks free after the pair above, so a
    // refusal here cannot be about banks — it is this collector's unpublished
    // output slots, which the fixture holds by holding the writer. Reading an
    // admission's success as proof that a bank came back would conflate the
    // two, which is why the reuse case asserts the bank address instead.
    OutputRoot root("runner-admission-reason");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());
    const RunLocalDfxMetadata metadata{};

    const DfxRunConfig stopped = args_dump_only(root.prefix(kFirstRun));
    ASSERT_EQ(runner.start_shared_collectors_for_run(stopped, kFirstRun, metadata), 0);
    runner.stop_run_producers(kFirstRun, [] {});
    ASSERT_EQ(
        runner.settle_run_after_producer_stop(
            stopped, /*pipeline_slot=*/0, kFirstRun, /*device_execution_complete=*/false
        ),
        0
    );

    const DfxRunConfig unstopped = args_dump_only(root.prefix(kSecondRun));
    ASSERT_EQ(runner.start_shared_collectors_for_run(unstopped, kSecondRun, metadata), 0);
    ASSERT_EQ(
        runner.settle_run_after_producer_stop(
            unstopped, /*pipeline_slot=*/0, kSecondRun, /*device_execution_complete=*/false
        ),
        0
    );
    ASSERT_EQ(runner.dump_banks_in_use(), 1u);

    SettlementRunner::DiagnosticAdmission admission;
    const DfxRunConfig third = args_dump_only(root.prefix(kForeignRun));
    EXPECT_NE(runner.start_shared_collectors_for_run(third, kForeignRun, metadata, &admission), 0);
    EXPECT_EQ(admission.refusal, simpler::dfx::runs::AdmissionRefusal::Capacity)
        << "a refusal for want of an output slot was reported as this owner being unusable";
    EXPECT_EQ(runner.dump_banks_in_use(), 1u) << "the refused admission took a bank, or the withheld run's bank moved";

    runner.shut_down();
}

TEST(RunTerminalStopSettlementThroughTheRunner, AFailedTeardownWithoutTheProofWithdrawsIt) {
    // The onboard shape, reached through the same base call the onboard runner
    // reaches — that platform's base does not even offer the proof as an
    // argument, so this states the `false` the onboard path always carries. A
    // real device's producers do not become observable from a host teardown, so
    // the bank stays withdrawn.
    OutputRoot root("runner-unproved");
    SettlementRunner runner;
    ASSERT_TRUE(runner.stand_up());

    const DfxRunConfig dfx = args_dump_only(root.prefix(kSecondRun));
    const RunLocalDfxMetadata metadata{};
    ASSERT_EQ(runner.start_shared_collectors_for_run(dfx, kSecondRun, metadata), 0);
    const uint64_t held = runner.dump_bank_of(kSecondRun);
    ASSERT_NE(held, 0u);

    // No stop performed, which is the onboard platform's permanent state: that
    // base does not even offer the proof as an argument.
    ASSERT_FALSE(runner.run_producers_stopped(kSecondRun));
    const int rc = runner.settle_run_after_producer_stop(
        dfx, /*pipeline_slot=*/0, kSecondRun, /*device_execution_complete=*/false
    );
    EXPECT_EQ(rc, 0);

    EXPECT_EQ(runner.dump_bank_of(kSecondRun), held)
        << "a teardown with no stop proof handed this run's bank to the next one";

    runner.shut_down();
}
