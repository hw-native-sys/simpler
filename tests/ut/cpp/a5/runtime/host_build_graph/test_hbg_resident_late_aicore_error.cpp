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
// What a5 host_build_graph's resident executor publishes when the AICore claims
// its error late.
//
// The AICore's exit-wait watchdog claims `scheduler_error` after the
// supervisor's last look at it, then acknowledges exit normally — so the
// AICPU's shutdown succeeds, every participant's rc is zero, every participant
// claims the audited path, and the shared header is the only place that error
// can still land.
//
// These cases call `AicpuExecutor::snapshot_run_terminal` and
// `AicpuExecutor::publish_run_terminal` **themselves**, on a real executor, and
// read the record out of a host region. So they cover the call site, not only
// the rule it calls: reverting the finalizer to a bare header read fails the
// first case.
//
// The executor's struct is defined in its own translation unit, so this test
// compiles that unit rather than duplicating anything from it. Nothing here
// runs `run()`, touches a register, or starts a thread: the finalizer is
// entered directly with the state a finished run leaves behind. That the
// production call sits inside the completion gate's finalizer — the
// all-participant boundary — is reviewed statically and is not what these cases
// establish.
#include "aicpu_executor.cpp"  // NOLINT(bugprone-suspicious-include): drives the executor in its own TU

#include <gtest/gtest.h>

#include <cstdint>

namespace {

constexpr uint64_t kEpoch = 0x5100'0000'0000'0007ULL;

/** A finished resident run: one participant, the audited path claimed, nothing failed. */
class ResidentFinalizer : public ::testing::Test {
protected:
    void SetUp() override {
        context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&run_control);
        context.run_control_offset = 0;
        runtime.set_worker_count(8);
        runtime.set_gm_sm_ptr(&header_storage);
        runtime.publish_scheduler_bootstrap(
            SCHEDULER_RUNTIME_MODE_RESIDENT_READY, reinterpret_cast<uint64_t>(&context)
        );
        executor.aicpu_thread_num_ = 1;
        executor.normal_path_claims_.store(1);
        set_platform_run_result(reinterpret_cast<uint64_t>(&region), kEpoch);
    }

    std::atomic<int32_t> &header() { return header_storage.sched_error_code; }

    /** What the AICore's exit-wait watchdog does after the supervisor stopped looking. */
    bool exit_wait_watchdog_fires() {
        return record_aicore_scheduler_runtime_error(
            &run_control, SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::EXIT_WAIT_TIMEOUT
        );
    }

    /** The finalizer, then the cleanup owner's commit — both the executor's own. */
    DeviceRunTerminal finalize_and_publish() {
        executor.snapshot_run_terminal(&runtime);
        executor.publish_run_terminal();
        return device_run_result_terminal(region, kEpoch);
    }

    SharedMemoryHeader header_storage{};
    SchedulerWorkerContext context{};
    SchedulerRunControl run_control{};
    DeviceRunResultRegion region{};
    Runtime runtime{};
    AicpuExecutor executor{};
};

// The case the fold exists for. Without it the finalizer reads a clean header
// and publishes Ok for a run the AICore recorded a timeout against.
TEST_F(ResidentFinalizer, AnErrorClaimedAfterTheSupervisorsReadStillDecidesTheRun) {
    ASSERT_TRUE(exit_wait_watchdog_fires());
    ASSERT_EQ(header().load(), SIMPLER_ERROR_NONE) << "nothing has folded it yet";

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
}

// Settling is not a licence to invent: a control the AICore left clean still
// publishes the success every participant claimed.
TEST_F(ResidentFinalizer, ACleanControlPublishesTheSuccessItClaimed) {
    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(terminal.code, 0);
    EXPECT_EQ(header().load(), SIMPLER_ERROR_NONE) << "no write on a clean path";
}

// First-wins: an error already latched keeps its own code, so a late timeout
// cannot relabel the failure the run actually had.
TEST_F(ResidentFinalizer, AnEarlierHeaderErrorKeepsPrecedence) {
    header().store(SIMPLER_ERROR_INVALID_ARGS);
    ASSERT_TRUE(exit_wait_watchdog_fires());

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(header().load(), SIMPLER_ERROR_INVALID_ARGS) << "the first error owns the slot";
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
}

// A participant's own failure stays the run's business result; settling a clean
// control gives the header no new precedence over it.
TEST_F(ResidentFinalizer, AParticipantFailureIsNotReplacedBySettling) {
    executor.terminal_.record_participant(/*run_rc=*/-1, /*shutdown_rc=*/0);

    const DeviceRunTerminal terminal = finalize_and_publish();

    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -1);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ThreadRc);
}

// A participant that never claimed the audited path withholds the success, and
// settling does not supply one.
TEST_F(ResidentFinalizer, AnUnclaimedAuditedPathPublishesNothing) {
    executor.normal_path_claims_.store(0);

    executor.snapshot_run_terminal(&runtime);
    executor.publish_run_terminal();

    EXPECT_EQ(region.published, 0u) << "no record to publish";
    EXPECT_EQ(device_run_result_terminal(region, kEpoch).state, DeviceRunTerminalState::Undecided);
}

}  // namespace
