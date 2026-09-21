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
 * Retained per-run terminal snapshots of the chip-swimlane record accounting.
 *
 * Every case drives the production sequence — the host arms this run's bank, the
 * real AICPU flush closes it, the host reads it back — with the two sides sharing
 * process memory. That shape covers the addressing, the epoch gate, the class
 * attribution, the retention and the lifetime. It says nothing about device cache
 * visibility: `cache_flush_range` is a no-op on the host, so a passing test here
 * is not evidence that a producer's entry becomes visible to the host on silicon.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"

namespace {

void *terminal_test_alloc(size_t size) { return std::calloc(1, size); }

int terminal_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

int init_collector(ChipSwimlaneCollector &collector, int num_aicore, ChipSwimlaneLevel level) {
    return collector.initialize(
        num_aicore, /*aicpu_thread_num=*/1, /*device_id=*/0, level, terminal_test_alloc, nullptr, terminal_test_free
    );
}

// Open a run's window, arm its bank, and bring the AICPU side up against it —
// the same order arm_collectors_for_run uses, with the arm between the region
// publication and the device-side init.
void arm_run(
    ChipSwimlaneCollector &collector, int num_aicore, uint32_t slot, uint64_t epoch, const char *prefix,
    ChipSwimlaneLevel level
) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.begin_run(prefix, level);

    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(slot, epoch))
    );
    chip_swimlane_aicpu_init(num_aicore);
}

}  // namespace

// The point of the whole mechanism: the successor's counter reset does not take
// the previous run's totals with it.
//
// The reset is the production one — `begin_run` calls `publish_run_config` — and
// the assertions bracket it, so a regression that re-clears the bank, or one that
// stops clearing the live counters, both show up here.
TEST(ChipSwimlaneRunTerminalTest, SnapshotSurvivesTheSuccessorsCounterReset) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    constexpr uint64_t kEpoch = 77;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "retained", ChipSwimlaneLevel::SCHEDULE_TIMING);
    for (uint32_t task = 1; task <= 3; task++) {
        ASSERT_EQ(
            chip_swimlane_aicpu_complete_task(
                /*core_id=*/0, /*thread_idx=*/0, task, /*dispatch_time=*/100 * task, /*finish_time=*/100 * task + 50
            ),
            0
        );
    }
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    ASSERT_EQ(get_perf_buffer_state(shm, 0)->head.total_record_count, 3u) << "the run did not take three records";

    ChipSwimlaneCollector::RunTerminalSnapshot before = collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch);
    ASSERT_TRUE(before.valid);
    EXPECT_EQ(before.aicpu_task.producers, 1);
    EXPECT_EQ(before.aicpu_task.total, 3u);
    EXPECT_EQ(before.aicpu_task.dropped, 0u);

    // The successor's window opens, which is what clears the live counters.
    collector.begin_run("successor", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(get_perf_buffer_state(shm, 0)->head.total_record_count, 0u)
        << "publish_run_config no longer clears the live counters — this test's premise is gone";

    ChipSwimlaneCollector::RunTerminalSnapshot after = collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch);
    ASSERT_TRUE(after.valid) << "the successor's reset destroyed the retained snapshot";
    EXPECT_EQ(after.aicpu_task.total, 3u);
    EXPECT_EQ(after.aicpu_task.dropped, 0u);

    collector.finalize(nullptr, terminal_test_free);
}

// An enabled producer that took no records closes with zeros, which is a
// different fact from a producer that never closed. Core 1 records nothing; both
// cores still report.
TEST(ChipSwimlaneRunTerminalTest, EnabledButIdlePoolsCloseWithZeroTotals) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);

    constexpr uint64_t kEpoch = 91;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kEpoch, "idle-pool", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(
        chip_swimlane_aicpu_complete_task(
            /*core_id=*/0, /*thread_idx=*/0, /*reg_task_id=*/1, /*dispatch_time=*/10, /*finish_time=*/20
        ),
        0
    );
    const int cores[] = {0, 1};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/2);

    ChipSwimlaneCollector::RunTerminalSnapshot snapshot =
        collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch);
    ASSERT_TRUE(snapshot.valid);
    EXPECT_EQ(snapshot.aicpu_task.producers, 2) << "an idle core's pool did not close";
    EXPECT_EQ(snapshot.aicpu_task.total, 1u);
    // The AICore pools exist for both cores and took nothing: closed, zero.
    EXPECT_EQ(snapshot.aicore_task.producers, 2);
    EXPECT_EQ(snapshot.aicore_task.total, 0u);
    EXPECT_EQ(snapshot.foreign_entries, 0);

    collector.finalize(nullptr, terminal_test_free);
}

// A phase pool that recorded nothing still closes. This is the behaviour that
// puts the close in the public flush entry point rather than in
// `flush_phase_pool`, which returns early for a pool with no active buffer or no
// records — a close inside it would leave an enabled-but-idle pool absent.
TEST(ChipSwimlaneRunTerminalTest, AnIdleSchedPhasePoolStillCloses) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHED_PHASES), 0);

    constexpr uint64_t kEpoch = 103;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "idle-phase", ChipSwimlaneLevel::SCHED_PHASES);
    chip_swimlane_aicpu_init_phase(/*worker_count=*/1, /*num_sched_phase_threads=*/1, /*num_orch_phase_threads=*/0);
    // No record_sched_phase call at all, then the run-end flush.
    chip_swimlane_aicpu_flush_sched_phase_buffer(/*thread_idx=*/0);

    ChipSwimlaneCollector::RunTerminalSnapshot snapshot =
        collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch);
    ASSERT_TRUE(snapshot.valid) << "an idle sched-phase pool reported nothing at all";
    EXPECT_EQ(snapshot.sched_phase.producers, 1);
    EXPECT_EQ(snapshot.sched_phase.total, 0u);
    EXPECT_EQ(snapshot.orch_phase.producers, 0) << "a pool that was never initialized must not report";

    collector.finalize(nullptr, terminal_test_free);
}

// Two concurrently retained runs, one per bank, keep their own totals.
TEST(ChipSwimlaneRunTerminalTest, TwoBanksRetainDistinctRuns) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);
    const int cores[] = {0};

    constexpr uint64_t kFirst = 201;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kFirst, "slot0", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    constexpr uint64_t kSecond = 202;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/1, kSecond, "slot1", ChipSwimlaneLevel::SCHEDULE_TIMING);
    for (uint32_t task = 1; task <= 4; task++) {
        ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, task, 10 * task, 10 * task + 5), 0);
    }
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    ChipSwimlaneCollector::RunTerminalSnapshot first = collector.read_run_terminal_snapshot(/*bank_index=*/0, kFirst);
    ASSERT_TRUE(first.valid) << "the second run's bank overwrote the first's";
    EXPECT_EQ(first.aicpu_task.total, 1u);

    ChipSwimlaneCollector::RunTerminalSnapshot second = collector.read_run_terminal_snapshot(/*bank_index=*/1, kSecond);
    ASSERT_TRUE(second.valid);
    EXPECT_EQ(second.aicpu_task.total, 4u);

    collector.finalize(nullptr, terminal_test_free);
}

// Reusing a slot replaces only the entries this run's producers closed. Core 1
// belongs to the first run alone, so the second run's read counts it as foreign
// rather than folding its records into its own total.
TEST(ChipSwimlaneRunTerminalTest, SameSlotReuseReplacesOnlyTheProducersThatClosed) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/2, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);

    constexpr uint64_t kWide = 301;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kWide, "wide", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/1, 0, 1, 10, 20), 0);
    const int two_cores[] = {0, 1};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, two_cores, /*core_num=*/2);

    ChipSwimlaneCollector::RunTerminalSnapshot wide = collector.read_run_terminal_snapshot(/*bank_index=*/0, kWide);
    ASSERT_TRUE(wide.valid);
    ASSERT_EQ(wide.aicpu_task.producers, 2);
    ASSERT_EQ(wide.aicpu_task.total, 1u);

    // Same slot, one core.
    constexpr uint64_t kNarrow = 302;
    arm_run(collector, /*num_aicore=*/2, /*slot=*/0, kNarrow, "narrow", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 30, 40), 0);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 2, 50, 60), 0);
    const int one_core[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, one_core, /*core_num=*/1);

    ChipSwimlaneCollector::RunTerminalSnapshot narrow = collector.read_run_terminal_snapshot(/*bank_index=*/0, kNarrow);
    ASSERT_TRUE(narrow.valid);
    EXPECT_EQ(narrow.aicpu_task.producers, 1) << "core 1 never ran for this run and must not report";
    EXPECT_EQ(narrow.aicpu_task.total, 2u) << "the predecessor's records leaked into this run's total";
    EXPECT_GT(narrow.foreign_entries, 0) << "the predecessor's un-overwritten entries went unnoticed";

    collector.finalize(nullptr, terminal_test_free);
}

// Negative control: a matching non-zero epoch is the whole validity test, so
// reading a bank under any other epoch reports unknown rather than the
// occupant's numbers.
TEST(ChipSwimlaneRunTerminalTest, AForeignEpochReadsAsUnknown) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);

    constexpr uint64_t kEpoch = 401;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "epoch-gate", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    ASSERT_TRUE(collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch).valid);

    ChipSwimlaneCollector::RunTerminalSnapshot wrong =
        collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch + 1);
    EXPECT_FALSE(wrong.valid);
    EXPECT_EQ(wrong.aicpu_task.total, 0u) << "an unknown snapshot must carry no numbers";

    // Zero is the entries' no-snapshot state, so it is never a readable identity.
    EXPECT_FALSE(collector.read_run_terminal_snapshot(/*bank_index=*/0, 0).valid);

    collector.finalize(nullptr, terminal_test_free);
}

// Nothing to arm, nothing to publish: the device gets 0, every close is a no-op,
// and no entry claims the bank.
TEST(ChipSwimlaneRunTerminalTest, NoBankIsArmedWithoutARegionASlotOrAnEpoch) {
    ChipSwimlaneCollector uninitialized;
    EXPECT_EQ(uninitialized.arm_run_terminal_bank(/*bank_index=*/0, /*run_epoch=*/1), nullptr)
        << "a bank was handed out before the region existed";

    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);
    EXPECT_EQ(collector.arm_run_terminal_bank(/*bank_index=*/0, /*run_epoch=*/0), nullptr) << "epoch 0 armed a bank";
    EXPECT_EQ(collector.arm_run_terminal_bank(PLATFORM_RUN_TERMINAL_BANKS, /*run_epoch=*/1), nullptr)
        << "a slot outside the bank array armed one anyway";

    // A run that armed nothing: the device-side bank pointer stays 0, so the
    // flush's closes write nowhere and the read reports unknown.
    constexpr uint64_t kEpoch = 501;
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    collector.begin_run("unarmed", ChipSwimlaneLevel::SCHEDULE_TIMING);
    set_platform_run_result(/*region_base=*/0, kEpoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(0);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    EXPECT_FALSE(collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch).valid)
        << "a close wrote into a bank the host never armed";

    collector.finalize(nullptr, terminal_test_free);
}

// The bank lives in the collector's allocation, so a finalize invalidates every
// outstanding snapshot. Reading one armed against the old region must report
// unknown rather than dereference the new one's addresses.
TEST(ChipSwimlaneRunTerminalTest, ASnapshotDoesNotSurviveItsAllocation) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);

    constexpr uint64_t kEpoch = 601;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "generation", ChipSwimlaneLevel::SCHEDULE_TIMING);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    ASSERT_TRUE(collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch).valid);

    collector.finalize(nullptr, terminal_test_free);
    EXPECT_FALSE(collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch).valid)
        << "a snapshot was read out of a freed region";

    // A rebuilt region is a different allocation, and the old epoch addresses
    // nothing in it.
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);
    EXPECT_FALSE(collector.read_run_terminal_snapshot(/*bank_index=*/0, kEpoch).valid)
        << "an epoch armed against the previous allocation read through the new one";

    collector.finalize(nullptr, terminal_test_free);
}

// Entries are one cache line each and the array starts on a line boundary,
// because a producer publishes with a flush that writes back whole lines: two
// entries sharing a line would let one producer's flush restore a stale copy of
// the other's.
TEST(ChipSwimlaneRunTerminalTest, EveryEntryOwnsItsOwnCacheLine) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1, ChipSwimlaneLevel::SCHEDULE_TIMING), 0);
    void *shm = collector.get_chip_swimlane_setup_device_ptr();

    for (int bank = 0; bank < PLATFORM_RUN_TERMINAL_BANKS; bank++) {
        ChipSwimlaneRunTerminal *base = get_run_terminal_bank(shm, bank);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(base) % 64, 0u) << "bank " << bank << " does not start on a line";
        for (int producer = 1; producer < PLATFORM_RUN_TERMINAL_PRODUCERS; producer++) {
            const uintptr_t prev = reinterpret_cast<uintptr_t>(get_run_terminal(base, producer - 1));
            const uintptr_t cur = reinterpret_cast<uintptr_t>(get_run_terminal(base, producer));
            ASSERT_EQ(cur - prev, 64u) << "producers " << producer - 1 << " and " << producer << " share a line";
        }
    }

    // And the whole array fits inside the region the collector allocated.
    const ChipSwimlaneRunTerminal *last = get_run_terminal(
        get_run_terminal_bank(shm, PLATFORM_RUN_TERMINAL_BANKS - 1), PLATFORM_RUN_TERMINAL_PRODUCERS - 1
    );
    const uintptr_t end = reinterpret_cast<uintptr_t>(last) + sizeof(ChipSwimlaneRunTerminal);
    EXPECT_LE(end - reinterpret_cast<uintptr_t>(shm), calc_perf_data_size_with_phases())
        << "the bank array runs past the allocated region";

    collector.finalize(nullptr, terminal_test_free);
}
