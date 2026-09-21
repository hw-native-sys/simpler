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
// The protocol that binds an AICore's handshake report to the run that asked
// for it: what a native producer stages, what the AICPU accepts, and when the
// AICPU may reply.
//
// Every case drives the production functions the producers and consumers call —
// `aicore_stage_native_report`, `aicore_report_accepted`,
// `aicore_context_reply_permitted`, `fill_shared_launch_args`. Deleting a
// production reset or guard therefore fails a case here rather than leaving it
// green.
//
// Limits, stated so no case is read for more than it shows: these run on the
// host with ordinary loads and stores. They check *what* each step publishes and
// in what program order, never when a write becomes visible to another
// processor. Nothing here is evidence about cache behaviour, the write-back, or
// any barrier's runtime effect.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "aicore_scheduler_state.h"
#include "common/kernel_args.h"
#include "runtime.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace {

constexpr uint64_t kRunN = 0x0000'0000'0000'2a01ULL;
constexpr uint64_t kRunNPlus1 = 0x0000'0000'0000'2a02ULL;
constexpr uint64_t kRunNPlus2 = 0x0000'0000'0000'2a03ULL;
constexpr uint32_t kBlockIdx = 5;
constexpr uint32_t kPhysicalCoreId = 11;
constexpr uint64_t kStaleContext = 0xdead'beefULL;
constexpr uint64_t kSomeRegAddr = 0x7f00'1000ULL;

// What a predecessor resident run leaves behind: its own accepted report, and
// the reply its AICPU published to this worker.
Handshake left_by_previous_resident_run() {
    Handshake handshake{};
    handshake.physical_core_id = kPhysicalCoreId;
    handshake.core_type = CoreType::AIC;
    handshake.aicore_done = kBlockIdx + 1;
    handshake.report_epoch = kRunN;
    handshake.aicpu_ready = SCHEDULER_RUNTIME_MODE_RESIDENT_READY;
    handshake.task = kStaleContext;
    return handshake;
}

// The producer's native sequence, with the barrier and write-back left out —
// those are architecture intrinsics and their effect is not what a host test
// can observe. `stage` and the marker store are the two production steps whose
// order this file can check.
void publish_native_report(Handshake *handshake, uint64_t epoch) {
    aicore_stage_native_report(
        handshake, kPhysicalCoreId, CoreType::AIC, kBlockIdx + 1, SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING
    );
    handshake->report_epoch = epoch;
}

// ---------------------------------------------------------------------------
// Acceptance
// ---------------------------------------------------------------------------

TEST(HandshakeReportEpoch, AZeroedLineIsNotAReport) {
    Handshake handshake{};
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1));
    EXPECT_FALSE(aicore_report_accepted(&handshake, 0));
}

TEST(HandshakeReportEpoch, APriorRunsReportIsRejected) {
    const Handshake stale = left_by_previous_resident_run();
    EXPECT_TRUE(aicore_report_accepted(&stale, kRunN)) << "its own run must still accept it";
    EXPECT_FALSE(aicore_report_accepted(&stale, kRunNPlus1));
}

TEST(HandshakeReportEpoch, AMarkerWithoutThisRunsStampIsRejected) {
    Handshake handshake{};
    handshake.aicore_done = kBlockIdx + 1;
    handshake.report_epoch = 0;
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1));
}

// A producer commits the stamp last, so a stamp without a marker is a line no
// producer ever published.
TEST(HandshakeReportEpoch, AStampWithoutTheMarkerIsRejected) {
    Handshake handshake{};
    handshake.report_epoch = kRunNPlus1;
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1));
}

// ---------------------------------------------------------------------------
// The native producer's own staging — the reset lives here
// ---------------------------------------------------------------------------

// The reverse hand-off words are cleared by the production staging function, so
// removing that reset breaks this case.
TEST(HandshakeReportEpoch, NativeStagingClearsAPredecessorsReply) {
    Handshake handshake = left_by_previous_resident_run();
    ASSERT_EQ(handshake.aicpu_ready, SCHEDULER_RUNTIME_MODE_RESIDENT_READY) << "seed did not take";
    ASSERT_EQ(handshake.task, kStaleContext);

    aicore_stage_native_report(
        &handshake, kPhysicalCoreId, CoreType::AIC, kBlockIdx + 1, SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING
    );

    EXPECT_EQ(handshake.aicpu_ready, SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING)
        << "a predecessor's READY survived this run's staging";
    EXPECT_EQ(handshake.task, 0u) << "a predecessor's context address survived this run's staging";
    EXPECT_EQ(handshake.physical_core_id, kPhysicalCoreId);
    EXPECT_EQ(handshake.aicore_done, kBlockIdx + 1);
}

// Staging is payload only. Until the marker store the line still carries the
// predecessor's epoch, so the AICPU cannot accept a half-published report —
// which is what the barrier between the two steps exists to keep true.
TEST(HandshakeReportEpoch, StagedPayloadIsNotAcceptableUntilTheMarkerIsWritten) {
    Handshake handshake = left_by_previous_resident_run();

    aicore_stage_native_report(
        &handshake, kPhysicalCoreId, CoreType::AIC, kBlockIdx + 1, SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING
    );
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1)) << "staging alone must not be acceptable";
    EXPECT_EQ(handshake.report_epoch, kRunN) << "staging must not touch the marker";

    handshake.report_epoch = kRunNPlus1;
    EXPECT_TRUE(aicore_report_accepted(&handshake, kRunNPlus1));
}

// An accepted native report always carries the cleared reply, because one
// production function publishes both.
TEST(HandshakeReportEpoch, AnAcceptedNativeReportCarriesNoPredecessorReply) {
    Handshake handshake = left_by_previous_resident_run();
    publish_native_report(&handshake, kRunNPlus1);

    ASSERT_TRUE(aicore_report_accepted(&handshake, kRunNPlus1));
    EXPECT_NE(handshake.aicpu_ready, SCHEDULER_RUNTIME_MODE_RESIDENT_READY);
    EXPECT_EQ(handshake.task, 0u);
}

// ---------------------------------------------------------------------------
// When the AICPU may reply
// ---------------------------------------------------------------------------

TEST(HandshakeReportEpoch, NoReplyToAWorkerThatDidNotReport) {
    SchedulerWorkerContext context{};
    // `cores_[i].reg_addr` is zero for a worker this run never accepted.
    EXPECT_FALSE(aicore_context_reply_permitted(&context, 0));
    EXPECT_TRUE(aicore_context_reply_permitted(&context, kSomeRegAddr));
}

// The other half of the predicate. A null bootstrap context is the
// mode-and-base verdict of `aicore_scheduler_bootstrap_context` — this run
// carries no resident scheduler state. Whether configuration *succeeded* is an
// outer gate in `AicpuExecutor::init`, which this predicate never sees.
TEST(HandshakeReportEpoch, NoReplyWithoutResidentSchedulerState) {
    EXPECT_FALSE(aicore_context_reply_permitted(nullptr, kSomeRegAddr))
        << "a run with no resident scheduler state has no context to hand over";
    EXPECT_FALSE(aicore_context_reply_permitted(nullptr, 0));
}

// Re-entry: a worker skipped by run N+1 must not be replied to on that run, and
// must be replied to again once its own report is accepted on N+2.
TEST(HandshakeReportEpoch, ASkippedWorkerIsRepliedToAgainOnItsNextAcceptedRun) {
    SchedulerWorkerContext context{};
    Handshake handshake = left_by_previous_resident_run();

    // Run N+1: this worker never reported, so it is neither accepted nor
    // replied to, and its line still holds run N's report.
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1));
    EXPECT_FALSE(aicore_context_reply_permitted(&context, /*worker_reg_addr=*/0));
    EXPECT_EQ(handshake.report_epoch, kRunN) << "a skipped worker's line is untouched";

    // Run N+2: the core runs, stages over two-runs-old state, and is accepted.
    publish_native_report(&handshake, kRunNPlus2);
    EXPECT_FALSE(aicore_report_accepted(&handshake, kRunNPlus1)) << "the skipped run must still not match";
    EXPECT_TRUE(aicore_report_accepted(&handshake, kRunNPlus2));
    EXPECT_TRUE(aicore_context_reply_permitted(&context, kSomeRegAddr));
    EXPECT_EQ(handshake.task, 0u) << "run N's context address must not survive into N+2";
}

// ---------------------------------------------------------------------------
// Mode projection: what each launch path puts on the wire
// ---------------------------------------------------------------------------

// A native program launch projects its run identity onto the AICore's block, so
// both processors test the same number.
TEST(HandshakeReportEpoch, ANativeLaunchProjectsThisRunsIdentity) {
    KernelArgs k_args{};
    k_args.run_result_epoch = kRunNPlus1;

    AicoreLaunchArgs args{};
    fill_shared_launch_args(args, k_args);
    EXPECT_EQ(args.report_epoch, kRunNPlus1);
    EXPECT_EQ(args.enable_profiling_flag, k_args.enable_profiling_flag);
}

// A kernel/persistent launch projects 0, which is what selects the protocol
// that predates the stamp on both sides.
TEST(HandshakeReportEpoch, AKernelLaunchProjectsZeroAndKeepsTheOldPredicate) {
    KernelArgs k_args{};
    ASSERT_EQ(k_args.run_result_epoch, 0u) << "kernel mode leaves the epoch unset";

    AicoreLaunchArgs args{};
    fill_shared_launch_args(args, k_args);
    ASSERT_EQ(args.report_epoch, 0u);

    // With that input a producer writes no stamp, and the consumer accepts on
    // the marker alone — including on a line still carrying some older native
    // run's stamp.
    Handshake handshake{};
    handshake.aicore_done = kBlockIdx + 1;
    EXPECT_TRUE(aicore_report_accepted(&handshake, args.report_epoch));

    Handshake carries_old_stamp = left_by_previous_resident_run();
    EXPECT_TRUE(aicore_report_accepted(&carries_old_stamp, args.report_epoch))
        << "epoch 0 must not start rejecting on the stamp";
}

// ---------------------------------------------------------------------------
// Layout the protocol depends on
// ---------------------------------------------------------------------------

// The staged payload, the cleared reply and the marker share one 64-byte line;
// the ordering argument is only meaningful because they travel together.
TEST(HandshakeReportEpoch, TheWholeReportFitsOneCacheLine) {
    EXPECT_EQ(sizeof(Handshake), 64u);
    EXPECT_EQ(alignof(Handshake), 64u);
    EXPECT_EQ(offsetof(Handshake, aicpu_ready), 0u);
    EXPECT_EQ(offsetof(Handshake, aicore_done), 4u);
    EXPECT_EQ(offsetof(Handshake, task), 8u);
    EXPECT_EQ(offsetof(Handshake, core_type), 16u);
    EXPECT_EQ(offsetof(Handshake, physical_core_id), 20u);
    EXPECT_EQ(offsetof(Handshake, report_epoch), 24u);
    EXPECT_LE(offsetof(Handshake, report_epoch) + sizeof(Handshake::report_epoch), 64u);
}

}  // namespace
