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
// What a fenced drain does once a run's boundaries have settled: whether it
// returns on the run's own record, and what happens to a status the record
// transfer itself reported.
//
// The production pieces are driven, not imitated. Boundaries come from a real
// `RunCompletionFence` through `wait_and_retain_run_boundaries`; the record
// read goes through `RunRecordReadLedgerT::read_with_status` against a
// scripted transport; the terminal comes from the shared
// `device_run_result_terminal` validator; and the branch is chosen by
// `decide_run_drain`, which is the rule `DeviceRunnerBase::wait_run_fence`
// switches on. The only thing a case supplies is what the device and the
// transport did.
//
// What this cannot cover, and what the PR says instead of simulating: the
// `rtMemcpy` inside `read_device_run_result` and the `sync_stream_pair` call
// the `Synchronize` and `ReportTransferError` branches make, both of which are
// CANN-bound members of an abstract runner.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "common/device_run_result.h"
#include "host/run_completion_fence.h"
#include "host/run_evidence_retention.h"
#include "host/run_outcome_decision.h"
#include "worker/native_run_execution.h"

namespace {

using Completion = RunCompletionFence::Completion;
using StreamRole = RunCompletionFence::StreamRole;

constexpr size_t kSlots = 4;
constexpr int kTimeoutMs = 2000;
// The code the a2a3 measurement saw a stream synchronize produce. Used here as
// a transport status so a case can show the exact value surviving, not that
// this test reproduces that measurement.
constexpr int kTransportError = 507018;
constexpr int kOtherTransportError = 507053;
constexpr int32_t kRuntimeCode = -507;

NativeRunIdentity run_at(uint32_t slot, uint64_t epoch) {
    NativeRunIdentity identity;
    identity.pipeline_slot = slot;
    identity.run_epoch = epoch;
    identity.generation = epoch;
    identity.dispatch_id = epoch;
    return identity;
}

// Boundary events with no device behind them. Both boundaries complete unless
// a case says otherwise.
class ScriptedEvents {
public:
    RunCompletionFence::DeviceEventOps ops() {
        RunCompletionFence::DeviceEventOps ops;
        ops.create = [this](void **out) {
            *out = &handles_[created_ % handles_.size()];
            ++created_;
            return 0;
        };
        ops.record = [](void *, void *) {
            return 0;
        };
        ops.query = [this](void *, bool *complete) {
            *complete = complete_;
            return 0;
        };
        ops.wait = [this](void *, int) {
            return wait_rc_;
        };
        ops.destroy = [](void *) {
            return 0;
        };
        return ops;
    }

    void device_never_completes() { complete_ = false; }

private:
    std::array<uint8_t, RunCompletionFence::kRoleCount> handles_{};
    size_t created_{0};
    bool complete_{true};
    int wait_rc_{0};
};

char kAicoreStream = 0;
char kAicpuStream = 0;

void fence_the_run(RunCompletionFence &fence, const NativeRunIdentity &run) {
    ASSERT_EQ(fence.arm(run), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicore);
    ASSERT_EQ(fence.record(run, StreamRole::Aicore, &kAicoreStream), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicpu);
    ASSERT_EQ(fence.record(run, StreamRole::Aicpu, &kAicpuStream), 0);
}

DeviceRunResultRegion published_success(uint64_t run_epoch) {
    DeviceRunResultRegion region{};
    region.published = run_epoch;
    region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    return region;
}

DeviceRunResultRegion published_failure(uint64_t run_epoch) {
    DeviceRunResultRegion region{};
    region.published = run_epoch;
    region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    region.completion_code = kRuntimeCode;
    region.code_source = static_cast<uint32_t>(DeviceRunCodeSource::Header);
    return region;
}

// A success record that also carries a failure code: the shape the validator
// refuses rather than reading as either verdict.
DeviceRunResultRegion self_inconsistent(uint64_t run_epoch) {
    DeviceRunResultRegion region = published_success(run_epoch);
    region.completion_code = kRuntimeCode;
    return region;
}

// A transport that lands `source`'s bytes and reports its own status code,
// which is what the production copy now returns instead of a bool.
class ScriptedTransport {
public:
    explicit ScriptedTransport(const DeviceRunResultRegion *source) :
        source_(source) {}

    int operator()(void *dst, const void *src) const {
        ++calls_;
        if (lands_bytes_) {
            EXPECT_EQ(src, static_cast<const void *>(source_));
            std::memcpy(dst, source_, sizeof(DeviceRunResultRegion));
        }
        return status_;
    }

    // Reports `status` after the bytes landed: the case where discarding the
    // code would leave a record that looks perfectly readable.
    void reports(int status) { status_ = status; }
    void reports_without_landing(int status) {
        status_ = status;
        lands_bytes_ = false;
    }
    unsigned calls() const { return calls_; }

private:
    const DeviceRunResultRegion *source_;
    mutable unsigned calls_{0};
    int status_{0};
    bool lands_bytes_{true};
};

// One drained run's evidence, assembled the way `wait_run_fence` assembles it.
struct DrainedRun {
    int transfer_rc{0};
    RunOutcomeEvidence evidence;
};

DrainedRun drain(
    RunCompletionFence &fence, RunBoundaryLedgerT<kSlots> &boundaries, RunRecordReadLedgerT<kSlots> &reads,
    DeviceRunResultRegion &cached, const NativeRunIdentity &run, const DeviceRunResultRegion *device_region,
    const ScriptedTransport &transport
) {
    DrainedRun out;
    out.evidence.boundaries = wait_and_retain_run_boundaries(fence, boundaries, run, kTimeoutMs).completion;
    out.transfer_rc = reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, device_region, transport);
    out.evidence.record_read = reads.state(run.pipeline_slot, run.run_epoch);
    out.evidence.terminal = out.evidence.record_read == RunRecordRead::Ok ?
                                device_run_result_terminal(cached, run.run_epoch) :
                                DeviceRunTerminal{};
    return out;
}

// ===== The fast branch =====

TEST(RunDrainDecision, BothBoundariesAndAValidOkDecideSuccessWithoutSynchronizing) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(1, 11);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    ASSERT_EQ(drained.evidence.boundaries, Completion::Complete);
    ASSERT_EQ(drained.transfer_rc, 0);
    ASSERT_EQ(drained.evidence.record_read, RunRecordRead::Ok);
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::AcceptRecordedSuccess);
}

TEST(RunDrainDecision, AnOkRecordWhoseBoundariesHaveNotCompletedStillSynchronizes) {
    ScriptedEvents events;
    events.device_never_completes();
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(1, 12);
    // Submitted but never recorded: the fence cannot decide it.
    ASSERT_EQ(fence.arm(run), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicore);
    fence.note_kernel_submitted(run, StreamRole::Aicpu);

    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    ASSERT_EQ(drained.evidence.boundaries, Completion::Unfenced);
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::Synchronize)
        << "a success record is not the missing completion proof";
}

// ===== An observed transport error is never replaced =====

TEST(RunDrainDecision, ATransportStatusOutranksARecordThatWouldOtherwiseDecideSuccess) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(2, 21);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    // The bytes land and they say Ok; the transport still reports a code.
    transport.reports(kTransportError);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    EXPECT_EQ(drained.transfer_rc, kTransportError) << "the transport's own code, not a substitute";
    EXPECT_EQ(drained.evidence.boundaries, Completion::Complete);
    EXPECT_EQ(drained.evidence.record_read, RunRecordRead::Failed)
        << "a transfer that reported a status did not produce a record to trust";
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::ReportTransferError);

    // The discriminator: the same evidence with no observed status does not
    // reach that branch, so it is the retained code that decides here.
    EXPECT_EQ(decide_run_drain(0, drained.evidence), RunDrainAction::Synchronize);
}

TEST(RunDrainDecision, AFailedTransferLeavesNoBytesForAnyBranchToRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(0, 31);
    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached = published_failure(run.run_epoch);
    ScriptedTransport transport(&device_region);
    transport.reports_without_landing(kTransportError);

    const int rc = reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, &device_region, transport);

    EXPECT_EQ(rc, kTransportError);
    EXPECT_EQ(reads.state(run.pipeline_slot, run.run_epoch), RunRecordRead::Failed);
    EXPECT_EQ(cached.published, 0u) << "the previous occupant's bytes are gone, not left to be misread";
}

// ===== The retained status, and the once-per-run read =====

TEST(RunRecordTransferStatus, TheRawCodeIsRetainedVerbatimAndRepeatedForTheSameRun) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(3, 41);
    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    transport.reports(kTransportError);

    EXPECT_EQ(
        reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, &device_region, transport), kTransportError
    );
    EXPECT_EQ(reads.transfer_status(run.pipeline_slot, run.run_epoch), kTransportError);
    EXPECT_EQ(transport.calls(), 1u);

    // What finalize's later call for the same run does.
    EXPECT_EQ(
        reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, &device_region, transport), kTransportError
    ) << "a deduplicated call answers with the retained code, not with a zero that looks like success";
    EXPECT_EQ(transport.calls(), 1u) << "and costs no second transfer";
    EXPECT_EQ(reads.state(run.pipeline_slot, run.run_epoch), RunRecordRead::Failed);
}

TEST(RunRecordTransferStatus, ASucceedingTransferRetainsNoStatus) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(3, 42);
    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);

    EXPECT_EQ(reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, &device_region, transport), 0);
    EXPECT_EQ(reads.transfer_status(run.pipeline_slot, run.run_epoch), 0);
    EXPECT_EQ(reads.state(run.pipeline_slot, run.run_epoch), RunRecordRead::Ok);
}

TEST(RunRecordTransferStatus, ASlotWithNoRegionReportsNoStatusAndNoAttempt) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(0, 43);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(nullptr);

    EXPECT_EQ(reads.read_with_status(run.pipeline_slot, run.run_epoch, cached, /*region=*/nullptr, transport), 0);
    EXPECT_EQ(transport.calls(), 0u) << "an absent region is not a transfer that failed";
    EXPECT_EQ(reads.state(run.pipeline_slot, run.run_epoch), RunRecordRead::NotAttempted);
    EXPECT_EQ(reads.transfer_status(run.pipeline_slot, run.run_epoch), 0);
}

// The bool-reporting entry stays for callers with no transport code to give.
TEST(RunRecordTransferStatus, TheSuccessOnlyEntryReportsAFailureWithoutClaimingATransportCode) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(1, 44);
    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};

    reads.read(run.pipeline_slot, run.run_epoch, cached, &device_region, [](void *, const void *) {
        return false;
    });

    EXPECT_EQ(reads.state(run.pipeline_slot, run.run_epoch), RunRecordRead::Failed);
    const int status = reads.transfer_status(run.pipeline_slot, run.run_epoch);
    EXPECT_NE(status, 0) << "a failed copy still reaches the drain as an observed error";
    EXPECT_NE(status, kTransportError) << "and does not invent a transport code it was never given";
}

// ===== Identity =====

TEST(RunRecordTransferStatus, AStatusBelongsToTheRunThatOwnsTheRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity predecessor = run_at(1, 51);
    const NativeRunIdentity successor = run_at(1, 52);
    const NativeRunIdentity other_slot = run_at(2, 51);
    const DeviceRunResultRegion device_region = published_success(predecessor.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    transport.reports(kOtherTransportError);

    ASSERT_EQ(
        reads.read_with_status(predecessor.pipeline_slot, predecessor.run_epoch, cached, &device_region, transport),
        kOtherTransportError
    );

    EXPECT_EQ(reads.transfer_status(successor.pipeline_slot, successor.run_epoch), 0)
        << "the next run on this slot owns no read here";
    EXPECT_EQ(reads.state(successor.pipeline_slot, successor.run_epoch), RunRecordRead::NotAttempted);
    EXPECT_EQ(reads.transfer_status(other_slot.pipeline_slot, other_slot.run_epoch), 0)
        << "and neither does the same epoch on another slot";
    EXPECT_EQ(reads.transfer_status(predecessor.pipeline_slot, predecessor.run_epoch), kOtherTransportError);
}

TEST(RunDrainDecision, ARecordPublishedUnderAnEarlierEpochSynchronizes) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity predecessor = run_at(1, 61);
    const NativeRunIdentity successor = run_at(1, 62);

    // The slot's region still holds what the predecessor published.
    const DeviceRunResultRegion device_region = published_success(predecessor.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);

    fence_the_run(fence, successor);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, successor, &device_region, transport);

    ASSERT_EQ(drained.evidence.boundaries, Completion::Complete);
    ASSERT_EQ(drained.evidence.record_read, RunRecordRead::Ok) << "the bytes did arrive";
    EXPECT_EQ(drained.evidence.terminal.state, DeviceRunTerminalState::Undecided)
        << "but they were published under another run's epoch";
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::Synchronize);
}

TEST(RunDrainDecision, AReadForgottenByAGenerationResetSynchronizes) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(2, 71);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);
    ASSERT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::AcceptRecordedSuccess);

    // What a runner starting a fresh device generation does.
    reads.reset();

    drained.evidence.record_read = reads.state(run.pipeline_slot, run.run_epoch);
    drained.evidence.terminal = DeviceRunTerminal{};
    EXPECT_EQ(drained.evidence.record_read, RunRecordRead::NotAttempted);
    EXPECT_EQ(reads.transfer_status(run.pipeline_slot, run.run_epoch), 0);
    EXPECT_EQ(decide_run_drain(/*record_transfer_rc=*/0, drained.evidence), RunDrainAction::Synchronize);
}

// ===== Every remaining shape keeps the synchronize =====

TEST(RunDrainDecision, APublishedFailureSynchronizesAndKeepsItsAttribution) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(0, 81);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region = published_failure(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    ASSERT_EQ(drained.evidence.terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(drained.evidence.terminal.code, kRuntimeCode);
    EXPECT_EQ(drained.evidence.terminal.source, DeviceRunCodeSource::Header);
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::Synchronize);
}

TEST(RunDrainDecision, ARecordNeverPublishedSynchronizes) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(1, 91);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region{};
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    ASSERT_EQ(drained.evidence.record_read, RunRecordRead::Ok);
    EXPECT_EQ(drained.evidence.terminal.state, DeviceRunTerminalState::Undecided);
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::Synchronize);
}

TEST(RunDrainDecision, ASelfInconsistentRecordSynchronizes) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(2, 101);
    fence_the_run(fence, run);

    const DeviceRunResultRegion device_region = self_inconsistent(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedTransport transport(&device_region);
    const DrainedRun drained = drain(fence, boundaries, reads, cached, run, &device_region, transport);

    EXPECT_EQ(drained.evidence.terminal.state, DeviceRunTerminalState::Undecided);
    EXPECT_EQ(decide_run_drain(drained.transfer_rc, drained.evidence), RunDrainAction::Synchronize);
}

}  // namespace
