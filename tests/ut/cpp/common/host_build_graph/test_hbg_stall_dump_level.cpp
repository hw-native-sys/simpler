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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "scheduler/scheduler_context.h"

uint64_t __attribute__((weak)) read_reg(uint64_t, RegId) { return 0; }

void __attribute__((weak)) reg_store_release(volatile uint32_t *, uint32_t) {}

extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }

extern "C" uint64_t get_platform_regs() { return 0; }

int SchedulerContext::prepare_block_for_dispatch(
    int32_t, int32_t, ChipTaskSlotState &, ResourceShape, bool, int32_t, PublishHandle *, bool
) {
    return 0;
}

// The cold path also holds the emergency-shutdown half, which reaches the platform's
// register entries and the graph-execution lookup. Nothing below is called by these
// tests — only the link needs them. Weak, so a target that does link a real
// implementation keeps it. The retirement entry differs per architecture and is
// stubbed per target instead.
GraphExecution *__attribute__((weak)) graph_execution_localize(ChipTaskSlotState &) { return nullptr; }

void __attribute__((weak)) platform_init_aicore_regs(uint64_t) {}

uint32_t __attribute__((weak)) platform_get_physical_cores_count() { return 0; }

// The SUMMARY line is the one part of a dump that needs neither a task table nor an
// owned cluster, so a context with an empty task view and no clusters emits exactly
// it — which keeps this test about the level a dump leaves at, and bounds the wiring
// it has to stand up. Named for the friend declaration in SchedulerContext, the
// sanctioned reach into its private state.
class SchedulerContextTestPeer {
public:
    static void emit(SchedulerContext &context, SchedulerState &state, StallDumpReport report) {
        context.sched_ = &state;
        context.aicpu_thread_num_ = 1;
        context.active_sched_threads_ = 1;
        context.completed_tasks_.store(0, std::memory_order_relaxed);
        context.log_stall_diagnostics(
            /*thread_idx=*/0, /*task_count=*/0, /*idle_iterations=*/480000, /*last_progress_count=*/0, report
        );
    }
};

namespace {

std::string capture_dump(StallDumpReport report) {
    auto state = std::make_unique<SchedulerState>();
    SharedMemoryTaskHeader tasks{};
    state->task_view.tasks = &tasks;

    SchedulerContext context;
    testing::internal::CaptureStderr();
    SchedulerContextTestPeer::emit(context, *state, report);
    return testing::internal::GetCapturedStderr();
}

}  // namespace

// A shutdown snapshot is the post-mortem of a run the scheduler is about to kill,
// and whoever reads that log did not know in advance to raise the device log level.
// Its lines therefore leave at the level of the SHUTDOWN_SNAPSHOT line that
// announces them. At INFO the default device log level drops them
// (`CheckLogLevel(AICPU, DLOG_INFO)` is false there), leaving a log that announces
// a dump and carries none of it.
TEST(StallDumpLevelTest, AShutdownSnapshotEmitsAboveTheDefaultLogLevel) {
    const std::string dump = capture_dump(StallDumpReport::Shutdown);

    ASSERT_NE(dump.find("SUMMARY"), std::string::npos) << "the shutdown dump emitted no summary at all";
    EXPECT_NE(dump.find("[WARN]"), std::string::npos)
        << "the shutdown snapshot's lines sit below the default device log level, so a stall reports nothing";
    EXPECT_EQ(dump.find("[INFO]"), std::string::npos) << "a shutdown snapshot line was left at INFO";
}

// The periodic round is the opposite case: it fires every STALL_LOG_INTERVAL idle
// iterations, from every scheduler thread, on a run that may still be progressing
// elsewhere. It stays at INFO, so raising the device log level is what turns it on —
// a cadence that survived the default level would flood device_log on any run that
// idles, and an AICPU log flood is slow enough to trip the op-execute timeout it was
// meant to diagnose.
TEST(StallDumpLevelTest, APeriodicRoundStaysBehindTheLogLevel) {
    const std::string dump = capture_dump(StallDumpReport::Periodic);

    ASSERT_NE(dump.find("SUMMARY"), std::string::npos) << "the periodic round emitted no summary at all";
    EXPECT_NE(dump.find("[INFO]"), std::string::npos) << "the periodic round is not at INFO";
    EXPECT_EQ(dump.find("[WARN]"), std::string::npos)
        << "the periodic round would reach the default log level, where its cadence floods device_log";
}
