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
 * The three things an all-or-nothing admission needs from each owner.
 *
 * Which refusal is re-attemptable, whether withdrawing a run an owner never
 * admitted is exact, and whether that owner still has a writer reading its
 * pools. The runner composes these into one decision; what is driven here is
 * each owner's own answer, because that is where a wrong one would be
 * invisible — the composition would simply decline or rebuild for a reason
 * nobody can see.
 */

#include <gtest/gtest.h>

#include "host/args_dump_collector.h"
#include "host/dep_gen_collector.h"
#include "host/scope_stats_collector.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;
namespace runs = simpler::dfx::runs;

namespace {

void *admission_alloc(size_t size) { return std::calloc(1, size); }

int admission_free(void *ptr) {
    std::free(ptr);
    return 0;
}

std::thread admission_thread_factory(std::function<void()> fn) { return std::thread(std::move(fn)); }

/** A destination tree this case owns, removed with it. */
struct OutputRoot {
    fs::path base;

    explicit OutputRoot(const char *name) {
        base = fs::temp_directory_path() / ("dfx_admission_" + std::string(name) + "_" + std::to_string(::getpid()));
        fs::remove_all(base);
        EXPECT_TRUE(fs::create_directories(base));
    }
    ~OutputRoot() { fs::remove_all(base); }

    fs::path prefix(const char *run) const { return base / run; }
};

}  // namespace

// --- Which refusal the caller may re-attempt ----------------------------

TEST(DiagnosticAdmissionTest, AFullSlotTableRefusesAsCapacityAndNotAsFatal) {
    // The distinction the whole wait rests on. Capacity says "right now", and
    // the caller parks on the credit signal; anything else says "this owner",
    // and parking would only burn the run's deadline.
    OutputRoot root("scope-stats-capacity");
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);
    collector.start(admission_thread_factory);

    for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
        const std::string leaf = "run-" + std::to_string(i);
        ASSERT_TRUE(collector.run_begin(9100 + i, root.prefix(leaf.c_str()).string()))
            << "admission " << i << " of a free table was refused";
    }

    runs::AdmissionRefusal why = runs::AdmissionRefusal::None;
    EXPECT_FALSE(collector.run_begin(9199, root.prefix("run-over").string(), &why));
    EXPECT_EQ(why, runs::AdmissionRefusal::Capacity)
        << "a full slot table refused as something the caller must not re-attempt";

    collector.finalize(nullptr, admission_free);
}

TEST(DiagnosticAdmissionTest, AnUnclassifiedRefusalReportsFatalRatherThanInvitingARetry) {
    // A collector that retains nothing has no admission to give and never
    // will. Reporting capacity here would park a run until its deadline for
    // an answer that cannot change — so the conservative default is the one
    // that fails early.
    OutputRoot root("scope-stats-not-retaining");
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);

    runs::AdmissionRefusal why = runs::AdmissionRefusal::None;
    EXPECT_FALSE(collector.run_begin(9201, root.prefix("run-a").string(), &why));
    EXPECT_EQ(why, runs::AdmissionRefusal::Fatal);

    collector.finalize(nullptr, admission_free);
}

// --- Withdrawing an owner the run never reached -------------------------

TEST(DiagnosticAdmissionTest, WithdrawingARunAnOwnerNeverAdmittedLeavesTheOtherRunAlone) {
    // What makes the rollback sweep exact rather than merely convenient: the
    // guard withdraws every enabled owner, including ones the attempt never
    // got to, so withdrawing by an identity this collector does not hold must
    // be a no-op — not a release of whichever run it does hold. The whole slot
    // table is held while that is checked, because one leaked slot is invisible
    // to a sequence of admissions that each close before the next one opens.
    OutputRoot root("scope-stats-rollback");
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);
    collector.start(admission_thread_factory);

    ASSERT_TRUE(collector.run_begin(9301, root.prefix("run-a").string()));
    ASSERT_TRUE(collector.run_begin(9302, root.prefix("run-b").string()));
    ASSERT_FALSE(collector.retained_runs_idle()) << "two admitted runs left the collector reading as idle";

    // The identity of a run this collector was never asked about.
    collector.abandon_run(9399);

    // Both admitted ones are untouched. Holding the whole table at once is what
    // makes that observable: a release would leave a slot free, and a free slot
    // is exactly what a further admission would take.
    runs::AdmissionRefusal why = runs::AdmissionRefusal::None;
    EXPECT_FALSE(collector.run_begin(9303, root.prefix("run-c").string(), &why))
        << "withdrawing an unrelated identity released an admitted run's slot";
    EXPECT_EQ(why, runs::AdmissionRefusal::Capacity);

    collector.abandon_run(9301);
    collector.abandon_run(9302);
    EXPECT_TRUE(collector.retained_runs_idle()) << "withdrawing the admitted runs did not free them";

    collector.finalize(nullptr, admission_free);
}

// --- Whether a writer may still be reading this owner's pools -----------

TEST(DiagnosticAdmissionTest, AnOwnerWithRetentionOffIsAlwaysIdle) {
    // The rebuild guard asks every collector, not only the enabled ones. A
    // collector that retains nothing has no writer and must not block a
    // rebuild the run ahead of it has no stake in.
    ScopeStatsCollector collector;
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);
    EXPECT_TRUE(collector.retained_runs_idle());
    collector.finalize(nullptr, admission_free);
}

TEST(DiagnosticAdmissionTest, AnAdmittedRunKeepsTheCollectorNonIdleUntilItIsPublished) {
    // This is the question a stale-shape rebuild must ask. "No run is
    // executing" is not it: an admitted run's writer holds the pools past its
    // own native retirement, and the rebuild frees exactly those.
    OutputRoot root("scope-stats-idle");
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);
    collector.start(admission_thread_factory);
    EXPECT_TRUE(collector.retained_runs_idle()) << "a collector with nothing admitted read as busy";

    ASSERT_TRUE(collector.run_begin(9401, root.prefix("run-a").string()));
    EXPECT_FALSE(collector.retained_runs_idle()) << "an unpublished run let a rebuild free its writer's pools";

    // This host-only fixture models an initialized, empty producer. An
    // admitted identity alone is insufficient evidence that device init ran.
    auto *state = get_scope_stats_buffer_state(collector.get_scope_stats_shm_device_ptr(), 0);
    state->run_epoch = 9401;
    collector.run_close(9401, /*device_execution_complete=*/true);
    std::string error;
    (void)collector.flush_retained_runs(5000, &error);
    EXPECT_TRUE(collector.retained_runs_idle()) << "a published run still blocked the rebuild";

    collector.finalize(nullptr, admission_free);
}

// --- A bank array that could not be mapped ------------------------------

namespace {

/** Refuses the bank-sized allocation only, so the shm region still comes up. */
size_t g_refuse_alloc_at_least = 0;

void *failing_alloc(size_t size) {
    if (g_refuse_alloc_at_least != 0 && size >= g_refuse_alloc_at_least) return nullptr;
    return std::calloc(1, size);
}

}  // namespace

TEST(DiagnosticAdmissionTest, AnUnmappableBankArrayLeavesTheOwnerOffTheJoinedPath) {
    // The failure must not become a silent address of 0. The owner stays on
    // its default path — which is a complete, if unjoined, diagnostic — and
    // every joined question it is asked answers no, rather than a run
    // publishing into an address nobody mapped.
    ScopeStatsCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    g_refuse_alloc_at_least = 0;
    ASSERT_EQ(collector.init(/*num_threads=*/1, failing_alloc, nullptr, admission_free, 0), 0);

    // From here the only allocation left is the bank array itself.
    g_refuse_alloc_at_least = 1;
    EXPECT_FALSE(collector.open_run_terminal_banks(SCOPE_STATS_TERMINAL_PRODUCERS))
        << "a refused allocation reported a mapped bank array";
    g_refuse_alloc_at_least = 0;

    EXPECT_FALSE(collector.device_publishes_run_terminal());
    EXPECT_EQ(collector.run_terminal_device_address(), 0u);
    EXPECT_FALSE(collector.can_take_run_terminal_bank(9501))
        << "an owner with no mapped array offered a bank to a joining run";

    collector.finalize(nullptr, admission_free);
}

// --- The same three answers from a second owner -------------------------

TEST(DiagnosticAdmissionTest, DepGenReportsCapacityAndIdlenessOnTheSameTerms) {
    // One owner's answers are not the contract; every owner's are. DepGen is
    // the one whose collector lives on the arch runner, so the runner reaches
    // it through a hook rather than directly — which is exactly where a
    // divergence would hide.
    OutputRoot root("dep-gen-admission");
    DepGenCollector collector;
    collector.configure_retained_runs(true, runs::kDefaultBudgetBytes);
    ASSERT_EQ(collector.init(/*num_threads=*/1, admission_alloc, nullptr, admission_free, 0), 0);
    collector.start(admission_thread_factory);
    EXPECT_TRUE(collector.retained_runs_idle());

    for (size_t i = 0; i < runs::kMaxOpenEpochs; i++) {
        const std::string leaf = "run-" + std::to_string(i);
        ASSERT_TRUE(collector.run_begin(9600 + i, root.prefix(leaf.c_str()).string()));
    }
    EXPECT_FALSE(collector.retained_runs_idle());

    runs::AdmissionRefusal why = runs::AdmissionRefusal::None;
    EXPECT_FALSE(collector.run_begin(9699, root.prefix("run-over").string(), &why));
    EXPECT_EQ(why, runs::AdmissionRefusal::Capacity);

    collector.finalize(nullptr, admission_free);
}
