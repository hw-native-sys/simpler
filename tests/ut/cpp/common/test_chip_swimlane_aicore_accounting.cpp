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
 * AICore per-run accounting: what the device produced, what the host accepted,
 * and what the host declined — kept apart.
 *
 * Every case drives the production path: the real AICPU dispatch/flush produces
 * a buffer, the collector's own `on_buffer_collected` copies it, and
 * `reconcile_counters` (which calls `reconcile_aicore_counters`) produces the
 * figures asserted. No case reimplements the accounting.
 *
 * Host and device share process memory here, so nothing below is evidence about
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

void *aicore_test_alloc(size_t size) { return std::calloc(1, size); }

int aicore_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

int init_collector(ChipSwimlaneCollector &collector, int num_aicore) {
    return collector.initialize(
        num_aicore, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::TASK_TIMING, aicore_test_alloc, nullptr,
        aicore_test_free
    );
}

void arm_run(ChipSwimlaneCollector &collector, int num_aicore, uint32_t slot, uint64_t epoch, const char *prefix) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(shm, nullptr);
    collector.begin_run(prefix, ChipSwimlaneLevel::TASK_TIMING);

    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(slot, epoch))
    );
    chip_swimlane_aicpu_init(num_aicore);
}

// Dispatch `records` AICore tasks on `core_id` and fill the active buffer's
// slots, leaving the last `unwritten` of them at start_time == 0 — the shape a
// race-window write or a recycled tail leaves behind.
void dispatch_aicore(ChipSwimlaneCollector &collector, int core_id, int records, int unwritten) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *ac_state = get_aicore_buffer_state(shm, core_id);
    const uint64_t held = ac_state->head.current_buf_ptr;
    ASSERT_NE(held, 0u) << "core " << core_id << " acquired no AICore buffer";
    auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(held);

    for (int i = 0; i < records; i++) {
        chip_swimlane_aicpu_on_aicore_dispatch(core_id, /*thread_idx=*/0, static_cast<uint32_t>(i + 1));
        if (i < records - unwritten) {
            buf->records[i].start_time = 1000 + static_cast<uint64_t>(i);
            buf->records[i].end_time = 2000 + static_cast<uint64_t>(i);
        }
        buf->records[i].reg_task_id = static_cast<uint32_t>(i + 1);
    }
}

// Hand every AICore buffer the flush published to the collector, exactly as the
// mgmt thread would.
void collect_published(ChipSwimlaneCollector &collector, uint32_t from_tail) {
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    for (uint32_t i = from_tail; i < header->queue_tails[0]; i++) {
        const ReadyQueueEntry &entry = header->queues[0][i];
        if (entry.kind != ChipSwimlaneBufferKind::AicoreTask) continue;
        ReadyBufferInfo info{};
        info.type = ProfBufferType::AICORE_TASK;
        info.index = entry.core_index;
        info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        info.buffer_seq = entry.buffer_seq;
        collector.on_buffer_collected(info, /*collector_shard=*/0);
    }
}

}  // namespace

// The balanced case: every dispatched record was written, accepted, and the
// device dropped none. This is the arm the mismatch cases below differ from.
TEST(ChipSwimlaneAicoreAccountingTest, EveryAcceptedRecordBalancesAgainstTheDeviceTotal) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3101;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "balanced");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/3, /*unwritten=*/0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known);
    EXPECT_EQ(a.device_total, 3u);
    EXPECT_EQ(a.device_dropped, 0u);
    EXPECT_EQ(a.host_collected, 3u) << "the host accepted fewer records than it was handed";
    EXPECT_EQ(a.host_skipped, 0u);
    EXPECT_EQ(a.host_collected + a.device_dropped + a.host_skipped, a.device_total);

    collector.finalize(nullptr, aicore_test_free);
}

// A slot the device never wrote is declined by the host. It must be counted as
// a host-side skip, and must NOT appear as a device drop — the device's own
// dropped counter stays zero while the identity still balances.
TEST(ChipSwimlaneAicoreAccountingTest, UnwrittenSlotIsAHostSkipNotADeviceDrop) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3102;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "unwritten");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/3, /*unwritten=*/1);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known);
    EXPECT_EQ(a.device_total, 3u);
    EXPECT_EQ(a.host_collected, 2u) << "the unwritten slot was accepted as a record";
    EXPECT_EQ(a.skipped_unwritten, 1u);
    EXPECT_EQ(a.device_dropped, 0u) << "a host-side skip was relabelled as a device drop";
    EXPECT_EQ(a.host_collected + a.device_dropped + a.host_skipped, a.device_total)
        << "the identity does not balance once host skips are counted";

    collector.finalize(nullptr, aicore_test_free);
}

// A buffer arriving for a core outside this run's set is declined wholesale.
// Its records are neither collected nor charged to the device.
TEST(ChipSwimlaneAicoreAccountingTest, OutOfRangeCoreIsCountedNotSilentlyDiscarded) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3103;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "bad-core");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    // Re-present the published buffer under a core index this 1-core run does
    // not own.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *header = get_chip_swimlane_header(shm);
    const ReadyQueueEntry &entry = header->queues[0][0];
    ASSERT_EQ(entry.kind, ChipSwimlaneBufferKind::AicoreTask);
    ReadyBufferInfo info{};
    info.type = ProfBufferType::AICORE_TASK;
    info.index = 7;  // outside [0, num_aicore_)
    info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
    info.host_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
    info.buffer_seq = entry.buffer_seq;
    collector.on_buffer_collected(info, /*collector_shard=*/0);

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known);
    EXPECT_EQ(a.skipped_bad_core, 2u) << "a buffer for an unowned core vanished without being counted";
    EXPECT_EQ(a.host_collected, 0u);
    EXPECT_EQ(a.device_dropped, 0u) << "a host-side rejection was relabelled as a device drop";

    collector.finalize(nullptr, aicore_test_free);
}

// The same buffer presented twice inflates what the host accepted past what the
// device produced. The mismatch must surface rather than balance.
TEST(ChipSwimlaneAicoreAccountingTest, DuplicateDeliveryDoesNotBalance) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3104;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "duplicate");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);
    collect_published(collector, /*from_tail=*/0);  // delivered twice

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known);
    EXPECT_EQ(a.device_total, 2u);
    EXPECT_EQ(a.host_collected, 4u) << "the second delivery was not counted";
    EXPECT_NE(a.host_collected + a.device_dropped + a.host_skipped, a.device_total)
        << "accepting a record twice must not balance against the device total";

    collector.finalize(nullptr, aicore_test_free);
}

// The regression this class exists for: the run's own records are replaced
// one-for-one by another run's, so every count still lines up. Collected == 2,
// device total == 2, dropped == 0 — a conservation check that ignores identity
// prints a clean balance for a run whose records never arrived.
//
// Reaches production reconcile, and asserts it cannot report that balance.
TEST(ChipSwimlaneAicoreAccountingTest, EqualCountForeignReplacementCannotReportABalance) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3105;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "foreign-replacement");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);

    // Every record this run produced is re-stamped as another run's before the
    // host sees it. The device's own totals are untouched.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    auto *ac_state = get_aicore_buffer_state(shm, 0);
    auto *buf = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(ac_state->head.current_buf_ptr);
    buf->run_epoch = kEpoch + 1;

    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known);
    EXPECT_EQ(a.device_total, 2u) << "the device still produced two records";
    EXPECT_FALSE(a.identity_ok) << "foreign records satisfied this run's accounting";
    EXPECT_EQ(a.foreign_identity, 2u);
    EXPECT_EQ(a.host_collected, 0u) << "another run's records were counted as this run's collected";
    EXPECT_EQ(a.host_skipped, 0u) << "a foreign record was balanced away as a host skip";
    EXPECT_EQ(a.device_dropped, 0u) << "a foreign record was relabelled as a device drop";
    // The naive identity would hold if foreign records were folded into either
    // term; it must not.
    EXPECT_NE(a.host_collected + a.device_dropped + a.host_skipped, a.device_total)
        << "an identity-blind conservation check reported a clean balance";

    // And the snapshot comparison reaches no verdict from it.
    auto c = collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch));
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unknown);

    // The artifact still holds every record, with its own identity intact.
    const auto &collected = collector.collected_aicore_records_for_test();
    ASSERT_FALSE(collected.empty());
    ASSERT_EQ(collected[0].size(), 2u) << "records were dropped from the artifact";
    for (const auto &rec : collected[0]) {
        EXPECT_EQ(rec.run_epoch, kEpoch + 1) << "a record was restamped";
    }

    collector.finalize(nullptr, aicore_test_free);
}

// begin_run clears the accounting, and only a reconcile pass for this run
// produces it. Until one runs there is nothing behind the AICore class, and the
// snapshot comparison must not reach a verdict from an empty one.
//
// This covers the absence of a pass, not a failed mirror — the mirror guard is
// covered where a copy can actually fail, in the transport test.
TEST(ChipSwimlaneAicoreAccountingTest, NoReconcilePassLeavesAccountingUnknown) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    const int cores[] = {0};

    // A first run that DOES reconcile, so there is a prior accounting for the
    // successor's begin_run to clear. Without this the case would pass on an
    // empty collector and prove nothing about the clear.
    constexpr uint64_t kFirst = 3106;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kFirst, "first");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);
    collector.reconcile_counters();
    ASSERT_TRUE(collector.aicore_accounting_for_test().known);
    ASSERT_EQ(collector.aicore_accounting_for_test().host_collected, 2u);

    // The successor runs but does not reconcile. Its accounting must be its
    // own — which is to say, absent — not the predecessor's.
    constexpr uint64_t kSecond = 3116;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/1, kSecond, "no-reconcile");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/3, /*unwritten=*/0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);

    auto a = collector.aicore_accounting_for_test();
    EXPECT_FALSE(a.known) << "the successor inherited the previous run's AICore accounting";
    EXPECT_EQ(a.host_collected, 0u);

    auto c = collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/1, kSecond));
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unknown)
        << "the AICore class reached a verdict with no accounting behind it";

    collector.finalize(nullptr, aicore_test_free);
}

// With the accounting present, the AICore class can now reach a real verdict
// against the retained snapshot — the gap this change closes. Coverage and sums
// both come from the same run.
TEST(ChipSwimlaneAicoreAccountingTest, AicoreClassNowReachesAVerdict) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3107;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "verdict");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);

    collector.reconcile_counters();
    auto c = collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch));
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Agree)
        << "the retained AICore snapshot did not match the device figures reconcile summed";
    EXPECT_EQ(c.aicore_task.missing_count, 0);

    // The AICPU class is unaffected by this change.
    EXPECT_NE(c.aicpu_task.verdict, Verdict::Unexpected);
    // Phase classes stay unknown.
    EXPECT_EQ(c.sched_phase.verdict, Verdict::Unknown);
    EXPECT_EQ(c.orch_phase.verdict, Verdict::Unknown);

    collector.finalize(nullptr, aicore_test_free);
}

// A snapshot entry at an index no producer of this run owns is still reported
// as a wrong identity, not smoothed by the new sum comparison.
TEST(ChipSwimlaneAicoreAccountingTest, SubstitutedAicoreIndexStaysUnexpected) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    constexpr uint64_t kEpoch = 3108;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kEpoch, "substituted");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);

    // An AICore terminal entry appears for a core this run does not own.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    ChipSwimlaneRunTerminal *bank = get_run_terminal_bank(shm, /*bank_index=*/0);
    ChipSwimlaneRunTerminal *stray = get_run_terminal(bank, PLATFORM_RUN_TERMINAL_AICORE_TASK_BASE + 4);
    stray->total = 0;
    stray->dropped = 0;
    stray->run_epoch = kEpoch;

    collector.reconcile_counters();
    auto c = collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch));
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unexpected)
        << "a substituted AICore index was accepted because the sums happened to match";
    EXPECT_EQ(c.aicore_task.unexpected_count, 1);

    collector.finalize(nullptr, aicore_test_free);
}
// Zero records under no expected identity is unknown, not a clean run. An
// unarmed collector has nothing to attribute a record to, so an empty
// population cannot be reported as this run having balanced — and the snapshot
// comparison must not reach a verdict either, even for a caller that supplies a
// well-formed snapshot.
TEST(ChipSwimlaneAicoreAccountingTest, UnarmedCollectorWithNoRecordsIsUnknownNotClean) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);

    // begin_run without a preceding successful arm: no expected identity.
    collector.begin_run("unarmed", ChipSwimlaneLevel::TASK_TIMING);
    collector.reconcile_counters();

    auto a = collector.aicore_accounting_for_test();
    ASSERT_TRUE(a.known) << "the mirror refreshed, so the figures were produced";
    EXPECT_FALSE(a.identity_ok) << "an empty population with no expected identity was called clean";
    EXPECT_EQ(a.host_collected, 0u);
    EXPECT_EQ(a.foreign_identity, 0u) << "nothing arrived, so nothing is foreign either";

    // Even a snapshot that looks valid cannot produce a verdict without an
    // expected identity behind the live side.
    auto c = collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/0, /*run_epoch=*/9001));
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unknown);

    collector.finalize(nullptr, aicore_test_free);
}

// An arm that fails must not leave the previous run's identity in place. A
// successor armed at an out-of-range slot gets no bank, and its records must
// not be matched against the identity the earlier run established.
TEST(ChipSwimlaneAicoreAccountingTest, FailedArmDoesNotLeaveThePriorIdentityCurrent) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector, /*num_aicore=*/1), 0);
    const int cores[] = {0};

    // A first run that arms successfully and balances, establishing an identity.
    constexpr uint64_t kFirst = 3201;
    arm_run(collector, /*num_aicore=*/1, /*slot=*/0, kFirst, "armed");
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    collect_published(collector, /*from_tail=*/0);
    collector.reconcile_counters();
    ASSERT_TRUE(collector.aicore_accounting_for_test().identity_ok);

    // The successor's arm fails: the slot is outside the retained banks.
    collector.begin_run("failed-arm", ChipSwimlaneLevel::TASK_TIMING);
    ASSERT_EQ(collector.arm_run_terminal_bank(PLATFORM_RUN_TERMINAL_BANKS, kFirst), nullptr)
        << "an out-of-range slot must not arm";

    // Records stamped with the FIRST run's epoch now arrive. If the failed arm
    // had left that epoch current, they would match and balance.
    void *shm = collector.get_chip_swimlane_setup_device_ptr();
    set_platform_run_result(/*region_base=*/0, kFirst);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(0);
    chip_swimlane_aicpu_init(/*worker_count=*/1);
    dispatch_aicore(collector, /*core_id=*/0, /*records=*/2, /*unwritten=*/0);
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
    const uint32_t tail_before = 0;
    collect_published(collector, tail_before);

    collector.reconcile_counters();
    auto a = collector.aicore_accounting_for_test();
    EXPECT_FALSE(a.identity_ok) << "a failed arm left the previous run's identity current";
    EXPECT_EQ(a.host_collected, 0u) << "records were attributed to an identity this collector no longer holds";

    collector.finalize(nullptr, aicore_test_free);
}
