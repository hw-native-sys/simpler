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
 * Device-to-host transfer failures on the two reads the consistency consumer
 * depends on.
 *
 * Links a `profiling_copy` implementation with separate host and device
 * storage, so the collector takes the non-SVM path: `alloc_paired_buffer`
 * mallocs a host shadow distinct from the device allocation, and each host read
 * of device state is a real copy that a test can make fail. Under the aliasing
 * arch stubs both failure branches are unreachable, so these are the only cases
 * that reach them.
 *
 * Each failure case first performs the same sequence successfully, leaving the
 * host shadow holding plausible values. That is what makes ignoring a return
 * code dangerous rather than merely untidy: the stale bytes read back as a
 * well-formed answer.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"
#include "profiling_copy_fault.h"

using Verdict = ChipSwimlaneCollector::RunTerminalVerdict;

namespace {

void *fault_test_alloc(size_t size) { return std::calloc(1, size); }

int fault_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

// One core, one AICPU thread: the smallest shape with a non-empty expected
// producer set.
int init_collector(ChipSwimlaneCollector &collector) {
    return collector.initialize(
        /*num_aicore=*/1, /*aicpu_thread_num=*/1, /*device_id=*/0, ChipSwimlaneLevel::SCHEDULE_TIMING, fault_test_alloc,
        nullptr, fault_test_free
    );
}

// Open a run's window, arm its bank, bring the AICPU side up against the
// DEVICE allocation, run one task and close it. The device side writes device
// memory; the host sees it only through a copy.
void run_one_task(ChipSwimlaneCollector &collector, uint32_t slot, uint64_t epoch, const char *prefix) {
    void *dev = collector.get_chip_swimlane_setup_device_ptr();
    ASSERT_NE(dev, nullptr);
    collector.begin_run(prefix, ChipSwimlaneLevel::SCHEDULE_TIMING);

    set_platform_run_result(/*region_base=*/0, epoch);
    set_chip_swimlane_enabled(true);
    set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(dev));
    set_platform_chip_swimlane_aicore_rotation_table(0);
    set_platform_chip_swimlane_run_terminal_bank(
        reinterpret_cast<uint64_t>(collector.arm_run_terminal_bank(slot, epoch))
    );
    chip_swimlane_aicpu_init(/*worker_count=*/1);

    ASSERT_EQ(chip_swimlane_aicpu_complete_task(/*core_id=*/0, 0, 1, 10, 20), 0);
    const int cores[] = {0};
    chip_swimlane_aicpu_flush(/*thread_idx=*/0, cores, /*core_num=*/1);
}

class RunTerminalTransportFault : public ::testing::Test {
protected:
    void SetUp() override { copy_fault::reset(); }
    void TearDown() override { copy_fault::reset(); }
};

}  // namespace

// The storage really is separate — a device-side write is invisible to the host
// until a copy runs. Without this the two failure cases below would prove
// nothing, because an aliasing allocator makes every read succeed by accident.
TEST_F(RunTerminalTransportFault, HostAndDeviceStorageAreDistinct) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);
    EXPECT_NE(collector.get_chip_swimlane_setup_device_ptr(), nullptr);

    constexpr uint64_t kEpoch = 2101;
    run_one_task(collector, /*slot=*/0, kEpoch, "distinct");

    // The device entry exists; the host shadow has not been refreshed yet.
    const ChipSwimlaneRunTerminal *dev_entry = get_run_terminal(
        get_run_terminal_bank(collector.get_chip_swimlane_setup_device_ptr(), 0), PLATFORM_RUN_TERMINAL_AICPU_TASK_BASE
    );
    ASSERT_EQ(dev_entry->run_epoch, kEpoch) << "the device side did not close its entry";

    // A read now performs an actual transfer.
    const int before = copy_fault::counts().from_device_calls;
    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    EXPECT_GT(copy_fault::counts().from_device_calls, before) << "no copy ran — the pointers must be aliased";
    EXPECT_TRUE(snapshot.transport_ok);
    EXPECT_TRUE(snapshot.valid);

    collector.finalize(nullptr, fault_test_free);
}

// Positive arm: with both transfers succeeding, this exact sequence agrees.
// The two failure cases differ from it only by which copy fails.
TEST_F(RunTerminalTransportFault, BothTransfersSucceedingAgrees) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);

    constexpr uint64_t kEpoch = 2102;
    run_one_task(collector, /*slot=*/0, kEpoch, "success");

    collector.reconcile_counters();
    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    ASSERT_TRUE(snapshot.transport_ok);
    EXPECT_EQ(collector.run_terminal_consistency(snapshot).aicpu_task.verdict, Verdict::Agree);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 0);

    collector.finalize(nullptr, fault_test_free);
}

// Case A: the bank's copy fails while the host shadow holds a plausible copy of
// it from an earlier successful read. Reading the shadow would produce a
// complete, agreeing answer for a run whose bytes never arrived.
TEST_F(RunTerminalTransportFault, FailedBankCopyIsNotTheStaleShadow) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);

    constexpr uint64_t kEpoch = 2103;
    run_one_task(collector, /*slot=*/0, kEpoch, "stale-bank");

    // Populate the host shadow with this run's real entries.
    collector.reconcile_counters();
    auto warm = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    ASSERT_TRUE(warm.transport_ok);
    ASSERT_EQ(collector.run_terminal_consistency(warm).aicpu_task.verdict, Verdict::Agree)
        << "the shadow must hold a plausible, agreeing answer for the failure to be dangerous";

    // Now the bank transfer fails. The shadow still holds those same bytes.
    copy_fault::arm({/*fail_from_device_size=*/calc_run_terminal_bank_size(), /*fail_rc=*/-7});
    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    EXPECT_EQ(copy_fault::counts().from_device_failures, 1) << "the bank copy was never attempted";
    EXPECT_FALSE(snapshot.transport_ok) << "a failed bank transfer was reported as a successful read";
    EXPECT_FALSE(snapshot.valid);
    EXPECT_EQ(snapshot.aicpu_task.producers, 0) << "the stale shadow was summed as this run's entries";

    auto c = collector.run_terminal_consistency(snapshot);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Unknown);
    EXPECT_EQ(c.aicore_task.verdict, Verdict::Unknown);

    collector.finalize(nullptr, fault_test_free);
}

// Case B: the bulk live mirror fails while the terminal read succeeds. The
// shadow's stale live counters are exactly equal to the retained ones, so a
// comparison that ignores the mirror's return code reports agreement it never
// established.
TEST_F(RunTerminalTransportFault, FailedLiveMirrorIsNotAgreementOnStaleEqualSums) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);

    constexpr uint64_t kEpoch = 2104;
    run_one_task(collector, /*slot=*/0, kEpoch, "stale-mirror");

    // A successful pass leaves the shadow's live counters equal to the retained
    // ones, which is the agreeing state.
    collector.reconcile_counters();
    ASSERT_EQ(
        collector.run_terminal_consistency(collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch)).aicpu_task.verdict,
        Verdict::Agree
    );

    // The region-sized mirror refresh now fails; the bank-sized read still
    // succeeds, so the snapshot side is intact and only the live side is stale.
    copy_fault::arm({/*fail_from_device_size=*/calc_perf_data_size_with_phases(), /*fail_rc=*/-9});
    collector.reconcile_counters();
    EXPECT_GE(copy_fault::counts().from_device_failures, 1) << "the mirror refresh was never attempted";

    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    ASSERT_TRUE(snapshot.transport_ok) << "the bank read should be unaffected";
    ASSERT_EQ(snapshot.aicpu_task.producers, 1) << "coverage is still established";

    auto c = collector.run_terminal_consistency(snapshot);
    EXPECT_EQ(c.aicpu_task.verdict, Verdict::Unknown)
        << "stale live counters that happen to match were reported as agreement";
    EXPECT_EQ(c.aicpu_task.missing_count, 0) << "coverage is unaffected by the live side failing";

    collector.finalize(nullptr, fault_test_free);
}
// The AICore accounting shares the mirror the AICPU comparison uses, so a failed
// refresh leaves it unknown too — no verdict is reached from device figures that
// never arrived.
TEST_F(RunTerminalTransportFault, FailedMirrorLeavesAicoreAccountingUnknown) {
    ChipSwimlaneCollector collector;
    ASSERT_EQ(init_collector(collector), 0);

    constexpr uint64_t kEpoch = 2105;
    run_one_task(collector, /*slot=*/0, kEpoch, "aicore-mirror");

    // A successful pass first: the accounting is produced and known.
    collector.reconcile_counters();
    ASSERT_TRUE(collector.aicore_accounting_for_test().known);

    // Now the region-sized mirror refresh fails. The shadow still holds the
    // same plausible figures.
    copy_fault::arm({/*fail_from_device_size=*/calc_perf_data_size_with_phases(), /*fail_rc=*/-11});
    collector.reconcile_counters();
    EXPECT_FALSE(collector.aicore_accounting_for_test().known)
        << "AICore accounting was produced from a mirror that did not refresh";

    auto snapshot = collector.read_run_terminal_snapshot(/*slot=*/0, kEpoch);
    ASSERT_TRUE(snapshot.transport_ok);
    EXPECT_EQ(collector.run_terminal_consistency(snapshot).aicore_task.verdict, Verdict::Unknown);

    collector.finalize(nullptr, fault_test_free);
}
