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
 * Snapshot-vs-live consistency verdicts for one run's retained terminal
 * accounting.
 *
 * Each case drives the production consumer — `reconcile_counters` for the live
 * side, then `run_terminal_consistency` — over a region the real AICPU flush
 * closed. No case reimplements the verdict policy.
 *
 * Scope of what these establish: that the consumer distinguishes transport
 * failure, producer identity, coverage and sum agreement for one run under the
 * current per-run reset and exclusivity. Host and device share process memory
 * here and `cache_flush_range` is a no-op, so nothing here is evidence about
 * device cache visibility.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"

using Verdict = ChipSwimlaneCollector::RunTerminalVerdict;

namespace {

void *consistency_test_alloc(size_t size) { return std::calloc(1, size); }

int consistency_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

int init_collector(ChipSwimlaneCollector &collector, int num_aicore) {
    return collector.initialize(
        num_aicore, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHEDULE_TIMING, consistency_test_alloc,
        nullptr, consistency_test_free
    );
}

// Open a run's window, arm its bank, and bring the AICPU side up against it —
// the order arm_collectors_for_run uses.
void arm_run(ChipSwimlaneCollector &collector, int num_aicore, uint32_t slot, uint64_t epoch, const char *prefix) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.begin_run(prefix, ChipSwimlaneLevel::SCHEDULE_TIMING);

    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(slot, epoch))
    );
    chip_swimlane_aicpu_init(num_aicore);
}

// The production order at teardown: reconcile captures the live side, then the
// snapshot is read and compared.
ChipSwimlaneCollector::RunTerminalConsistency consume(ChipSwimlaneCollector &collector, uint32_t slot, uint64_t epoch) {
    collector.reconcile_counters();
    return collector.run_terminal_consistency(collector.read_run_terminal_snapshot(slot, epoch));
}

}  // namespace

// Every expected core reported and the retained sums match the live ones.
TEST(ChipSwimlaneRunTerminalConsistencyTest, FullCoverageWithMatchingSumsAgrees) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2), 0);

    constexpr uint64_t kEpoch = 1101;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kEpoch, "agree");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/1, 0, 1, 30, 40), 0);
    const int cores[] = {0, 1};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/2);

    auto c = consume(collector, /*slot=*/0, kEpoch);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Agree);
    EXPECT_EQ(c.aicpu_task.expected_count, 2);
    EXPECT_EQ(c.aicpu_task.reported_count, 2);
    EXPECT_EQ(c.aicpu_task.missing_count, 0);
    EXPECT_EQ(c.aicpu_task.unexpected_count, 0);
    // AICore has no live counterpart in reconcile, so coverage is checkable but
    // sums are not: it can never reach Agree.
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unknown);
    EXPECT_EQ(c.aicore_task.missing_count, 0);

    collector.finalize(nullptr, consistency_test_free);
}

// An entry at an index outside [0, num_aicore_) standing in for a missing
// expected one. Cardinality matches, so a count-only check would call this
// complete; the verdict must be Unexpected, and must outrank Partial.
TEST(ChipSwimlaneRunTerminalConsistencyTest, UnexpectedIndexIsNotAgreementAtEqualCardinality) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2), 0);
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    constexpr uint64_t kEpoch = 1102;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kEpoch, "substituted");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    // Core 0 closes; core 1 does not.
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    // An entry appears at core 5, which no producer of this 2-core run owns.
    ChipSwimlaneRunTerminal *bank = get_run_terminal_bank(shm, /*bank_index=*/0);
    ChipSwimlaneRunTerminal *stray = get_run_terminal(bank, PLATFORM_RUN_TERMINAL_AICPU_TASK_BASE + 5);
    stray->total = 1;
    stray->dropped = 0;
    stray->run_epoch = kEpoch;

    auto c = consume(collector, /*slot=*/0, kEpoch);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Unexpected) << "a substituted index was accepted as coverage";
    EXPECT_EQ(c.aicpu_task.reported_count, 2) << "the cardinalities do match — that is the point";
    EXPECT_EQ(c.aicpu_task.unexpected_count, 1);
    EXPECT_EQ(c.aicpu_task.missing_count, 1);

    collector.finalize(nullptr, consistency_test_free);
}

// An expected index published nothing, and no stray entry hides it.
TEST(ChipSwimlaneRunTerminalConsistencyTest, MissingExpectedIndexIsPartial) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2), 0);

    constexpr uint64_t kEpoch = 1103;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kEpoch, "partial");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    auto c = consume(collector, /*slot=*/0, kEpoch);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Partial);
    EXPECT_EQ(c.aicpu_task.missing_count, 1);
    EXPECT_EQ(c.aicpu_task.unexpected_count, 0);

    collector.finalize(nullptr, consistency_test_free);
}

// Full coverage, but the retained sums and the live counters disagree. The
// live head is perturbed after its close, which is what a snapshot that failed
// to capture the settled value would look like.
TEST(ChipSwimlaneRunTerminalConsistencyTest, SumMismatchIsDisagreeNotSilence) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    constexpr uint64_t kEpoch = 1104;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "mismatch");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    // The live head moves after the snapshot was taken from it.
    get_perf_buffer_state(shm, 0)->head.total_record_count = 99;

    auto c = consume(collector, /*slot=*/0, kEpoch);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Disagree);
    EXPECT_EQ(c.aicpu_task.missing_count, 0);

    collector.finalize(nullptr, consistency_test_free);
}

// A readable bank holding nothing for this run is not a transport failure, and
// the two must not collapse into one verdict. All-zero reads as Partial (every
// expected index missing); an unreadable bank reads as Unknown.
TEST(ChipSwimlaneRunTerminalConsistencyTest, AllZeroReadableDiffersFromFailedTransport) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 1105;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "all-zero");
    // No task, no flush: the bank stays at its initialized zero state.

    auto readable = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    EXPECT_TRUE(readable.transport_ok) << "an all-zero bank was read successfully and must say so";
    EXPECT_FALSE(readable.valid) << "no entry carried this run's epoch";
    collector.reconcile_counters();
    EXPECT_EQ(collector.run_terminal_consistency(readable).aicpu_task.verdict, Verdict::Partial);

    // A snapshot whose transport never succeeded carries no verdict at all.
    ChipSwimlaneCollector::RunTerminalSnapshot unreadable;
    unreadable.run_epoch = kEpoch;
    EXPECT_FALSE(unreadable.transport_ok);
    EXPECT_EQ(collector.run_terminal_consistency(unreadable).aicpu_task.verdict, Verdict::Unknown)
        << "a failed transport must not be reported as missing coverage";

    collector.finalize(nullptr, consistency_test_free);
}

// Excess with nothing missing: every expected index reported AND a stray entry
// carries this run's epoch. Neither Partial nor Agree describes that; the
// intrusion has to be reported on its own.
//
// An exactly-empty expected set is not constructible for the task classes —
// `initialize()` validates `num_aicore >= 1` — so this is the reachable form of
// "entries present where none belong". The empty-set case exists only for the
// phase classes, which this consumer deliberately reports as Unknown.
TEST(ChipSwimlaneRunTerminalConsistencyTest, ExcessEntryIsReportedEvenWithFullCoverage) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    constexpr uint64_t kEpoch = 1106;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "excess");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    // Core 0 (the whole expected set) reported; core 3 also carries this epoch.
    ChipSwimlaneRunTerminal *bank = get_run_terminal_bank(shm, /*bank_index=*/0);
    ChipSwimlaneRunTerminal *stray = get_run_terminal(bank, PLATFORM_RUN_TERMINAL_AICPU_TASK_BASE + 3);
    stray->total = 0;
    stray->dropped = 0;
    stray->run_epoch = kEpoch;

    auto c = consume(collector, /*slot=*/0, kEpoch);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Unexpected) << "an excess entry was hidden by full coverage";
    EXPECT_EQ(c.aicpu_task.missing_count, 0) << "nothing is missing — excess alone drives the verdict";
    EXPECT_EQ(c.aicpu_task.unexpected_count, 1);

    collector.finalize(nullptr, consistency_test_free);
}

// begin_run drops the previous run's live figures, so a successor cannot be
// judged against them.
TEST(ChipSwimlaneRunTerminalConsistencyTest, BeginRunInvalidatesThePriorRunsLiveSide) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    const int cores[] = {0};

    constexpr uint64_t kFirst = 1107;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kFirst, "first");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    ASSERT_EQ(consume(collector, /*slot=*/0, kFirst).aicpu_task.verdict, Verdict::Agree);

    // The successor's window opens and its producers close into bank 1. Its own
    // reconcile has not run yet, so the sums cannot be judged — but coverage
    // still can, and the first run's live figures must not be reused.
    constexpr uint64_t kSecond = 1108;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/1, kSecond, "second");
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 30, 40), 0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    auto before_reconcile =
        collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/1, kSecond));
    EXPECT_EQ(before_reconcile.aicpu_task.verdict, Verdict::Unknown)
        << "the successor was judged against the previous run's live counters";
    EXPECT_EQ(before_reconcile.aicpu_task.missing_count, 0) << "coverage is still established";

    collector.finalize(nullptr, consistency_test_free);
}

// The phase classes are deliberately not compared. Their expected producer
// counts exist only as untagged device observations in the shared header, which
// no per-run reset clears, so a successful read cannot separate this run's
// counts from a previous run's. The consumer reports Unknown rather than
// inferring coverage from them — including on a run that produced phase records
// and whose header counts are populated.
TEST(ChipSwimlaneRunTerminalConsistencyTest, PhaseClassesReportUnknownRatherThanInferredCoverage) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(
        collector.initialize(
            /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHED_PHASES,
            consistency_test_alloc, nullptr, consistency_test_free
        ),
        0
    );
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    constexpr uint64_t kEpoch = 1109;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "phase-unknown");
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/0);
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    // The device did publish a sched-phase producer count into the header.
    ASSERT_EQ(get_chip_swimlane_header(shm)->num_sched_phase_threads, 1u);

    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    ASSERT_TRUE(snapshot.transport_ok);
    ASSERT_GT(snapshot.sched_phase.producers, 0) << "the sched-phase producer did close an entry";

    collector.reconcile_counters();
    auto c = collector.run_terminal_consistency(snapshot);
    EXPECT_EQ(c.sched_phase.verdict, Verdict::Unknown) << "a header count was promoted to an independent expected set";
    EXPECT_EQ(c.orch_phase.verdict, Verdict::Unknown);

    collector.finalize(nullptr, consistency_test_free);
}
