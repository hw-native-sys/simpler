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
// How the shadow consumer's evidence is captured, and what the shared rule
// makes of it.
//
// The capture seams are driven, not imitated: `wait_and_retain_run_boundaries`
// and `poll_and_retain_run_boundaries` are called against a real
// `RunCompletionFence` with scripted event operations, and
// `RunRecordReadLedgerT::read` against a scripted copy. Each seam performs the
// observation and retains it in one call, so a case that reads the retained
// answer has driven the production capture that produced it.
//
// `DeviceRunnerBase` owns the two ledgers and calls these seams; it is abstract
// and CANN-bound, so what it cannot cover is stated in the PR rather than
// simulated here.
//
// Nothing here touches a device: every device operation the fence and the read
// would make is injected by the case.
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
constexpr int kWaitFailure = -117;
constexpr int32_t kRuntimeCode = -507;

NativeRunIdentity run_at(uint32_t slot, uint64_t epoch) {
    NativeRunIdentity identity;
    identity.pipeline_slot = slot;
    identity.run_epoch = epoch;
    identity.generation = epoch;
    identity.dispatch_id = epoch;
    return identity;
}

// Event operations with no device behind them: `complete` is what a query
// reports, `wait_rc` what a bounded wait returns, and the counters say whether
// the fence was asked at all.
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
            ++queries_;
            *complete = complete_;
            return 0;
        };
        ops.wait = [this](void *, int) {
            ++waits_;
            return wait_rc_;
        };
        ops.destroy = [](void *) {
            return 0;
        };
        return ops;
    }

    void device_completed() { complete_ = true; }
    void wait_fails_with(int rc) { wait_rc_ = rc; }
    unsigned waits() const { return waits_; }
    unsigned queries() const { return queries_; }

private:
    std::array<uint8_t, RunCompletionFence::kRoleCount> handles_{};
    size_t created_{0};
    bool complete_{false};
    int wait_rc_{0};
    unsigned waits_{0};
    unsigned queries_{0};
};

char kAicoreStream = 0;
char kAicpuStream = 0;

// A run whose two kernels were submitted and whose two boundaries were
// recorded — the state the launch path leaves a fenced run in.
void fence_the_run(RunCompletionFence &fence, const NativeRunIdentity &run) {
    ASSERT_EQ(fence.arm(run), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicore);
    ASSERT_EQ(fence.record(run, StreamRole::Aicore, &kAicoreStream), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicpu);
    ASSERT_EQ(fence.record(run, StreamRole::Aicpu, &kAicpuStream), 0);
}

// A run whose kernels were submitted but whose boundaries were never recorded:
// submitted work the fence cannot decide.
void arm_without_boundaries(RunCompletionFence &fence, const NativeRunIdentity &run) {
    ASSERT_EQ(fence.arm(run), 0);
    fence.note_kernel_submitted(run, StreamRole::Aicore);
    fence.note_kernel_submitted(run, StreamRole::Aicpu);
}

DeviceRunResultRegion published_failure(uint64_t run_epoch) {
    DeviceRunResultRegion region{};
    region.published = run_epoch;
    region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    region.completion_code = kRuntimeCode;
    region.code_source = static_cast<uint32_t>(DeviceRunCodeSource::Header);
    return region;
}

DeviceRunResultRegion published_success(uint64_t run_epoch) {
    DeviceRunResultRegion region{};
    region.published = run_epoch;
    region.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    return region;
}

// A copy that lands `source`'s bytes, counting its invocations so a case can
// tell "the copy failed" from "no copy was attempted".
class ScriptedCopy {
public:
    explicit ScriptedCopy(const DeviceRunResultRegion *source) :
        source_(source) {}

    bool operator()(void *dst, const void *src) const {
        ++calls_;
        if (!succeeds_) return false;
        EXPECT_EQ(src, static_cast<const void *>(source_));
        std::memcpy(dst, source_, sizeof(DeviceRunResultRegion));
        return true;
    }

    void fails() { succeeds_ = false; }
    unsigned calls() const { return calls_; }

private:
    const DeviceRunResultRegion *source_;
    bool succeeds_{true};
    mutable unsigned calls_{0};
};

// What `collect_run_evidence` assembles in c_api_shared.cpp: the retained
// boundary observation, the read state, and the cached bytes interpreted by
// the record's own reader. The runner reports a reason instead of those bytes
// for a read that is not Ok, which the rule does not consult — it decides the
// record channel from `record_read` alone unless that read is Ok.
RunOutcomeEvidence evidence_for(
    const RunBoundaryLedgerT<kSlots> &boundaries, const RunRecordReadLedgerT<kSlots> &reads,
    const DeviceRunResultRegion &cached, const NativeRunIdentity &run
) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = boundaries.observed(run);
    evidence.record_read = reads.state(run.pipeline_slot, run.run_epoch);
    evidence.terminal = device_run_result_terminal(cached, run.run_epoch);
    return evidence;
}

}  // namespace

// ===== Capturing what the drain observed =====

TEST(RunBoundaryCapture, AFencedDrainRetainsCompleteAndReportsTheWaitsRc) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity run = run_at(0, 11);
    fence_the_run(fence, run);

    const RunBoundaryWait observed = wait_and_retain_run_boundaries(fence, ledger, run, kTimeoutMs);

    EXPECT_EQ(observed.completion, Completion::Complete);
    EXPECT_EQ(observed.rc, 0);
    EXPECT_EQ(events.waits(), 2u) << "one bounded wait per boundary";
    EXPECT_EQ(ledger.observed(run), Completion::Complete);
}

TEST(RunBoundaryCapture, AFailedBoundaryWaitIsRetainedAsErrorWithItsRcReturned) {
    ScriptedEvents events;
    events.wait_fails_with(kWaitFailure);
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity run = run_at(1, 21);
    fence_the_run(fence, run);

    const RunBoundaryWait observed = wait_and_retain_run_boundaries(fence, ledger, run, kTimeoutMs);

    EXPECT_EQ(observed.rc, kWaitFailure) << "the caller's rc is the fence's own, unchanged";
    EXPECT_EQ(observed.completion, Completion::Error);
    EXPECT_EQ(ledger.observed(run), Completion::Error);
}

// The observation is retained before the caller's fallback runs, so whatever
// the whole-stream wait then does to the streams cannot erase it.
TEST(RunBoundaryCapture, AnUnfencedRunIsRetainedWithoutAskingTheFenceToWait) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity run = run_at(2, 31);
    arm_without_boundaries(fence, run);

    const RunBoundaryWait observed = wait_and_retain_run_boundaries(fence, ledger, run, kTimeoutMs);

    EXPECT_EQ(observed.completion, Completion::Unfenced);
    EXPECT_EQ(observed.rc, 0) << "the fence decides nothing here; the caller owes its own bounded proof";
    EXPECT_EQ(events.waits(), 0u);
    EXPECT_EQ(ledger.observed(run), Completion::Unfenced);
}

TEST(RunBoundaryCapture, APollRetainsEachAnswerItGets) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity run = run_at(3, 41);
    fence_the_run(fence, run);

    EXPECT_EQ(poll_and_retain_run_boundaries(fence, ledger, run), Completion::Pending);
    EXPECT_EQ(ledger.observed(run), Completion::Pending);

    events.device_completed();

    EXPECT_EQ(poll_and_retain_run_boundaries(fence, ledger, run), Completion::Complete);
    EXPECT_EQ(ledger.observed(run), Completion::Complete) << "the later observation replaces the earlier one";
}

TEST(RunBoundaryCapture, AnUnfencedPollIsRetainedAsUnfenced) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity run = run_at(0, 51);
    arm_without_boundaries(fence, run);

    EXPECT_EQ(poll_and_retain_run_boundaries(fence, ledger, run), Completion::Unfenced);
    EXPECT_EQ(ledger.observed(run), Completion::Unfenced);
}

// A slot is reused, so the identity is what keeps a successor's drain from
// overwriting the answer its predecessor is still owed.
TEST(RunBoundaryCapture, ASuccessorOnTheSameSlotInheritsNothing) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> ledger;
    const NativeRunIdentity predecessor = run_at(1, 61);
    const NativeRunIdentity successor = run_at(1, 62);

    fence_the_run(fence, predecessor);
    EXPECT_EQ(wait_and_retain_run_boundaries(fence, ledger, predecessor, kTimeoutMs).completion, Completion::Complete);
    EXPECT_EQ(ledger.observed(successor), Completion::Pending) << "the successor has observed nothing yet";

    ASSERT_EQ(fence.retire(predecessor), 0);
    arm_without_boundaries(fence, successor);
    EXPECT_EQ(wait_and_retain_run_boundaries(fence, ledger, successor, kTimeoutMs).completion, Completion::Unfenced);

    EXPECT_EQ(ledger.observed(successor), Completion::Unfenced);
    EXPECT_EQ(ledger.observed(predecessor), Completion::Pending) << "the slot holds one observation";
}

// ===== The capture-before-retire integration =====
//
// The case the retention exists for: the drain's cleanup retires the fence, so
// the consumer that runs at finalize cannot ask it — the observation has to
// have survived that retirement, keyed to the run that made it.
TEST(RunBoundaryRetirement, TheDrainsObservationOutlivesTheCleanupThatRetiresTheFence) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(2, 71);
    const uint64_t epoch = run.run_epoch;

    fence_the_run(fence, run);
    ASSERT_EQ(wait_and_retain_run_boundaries(fence, boundaries, run, kTimeoutMs).completion, Completion::Complete);

    const DeviceRunResultRegion device_region = published_success(epoch);
    DeviceRunResultRegion cached{};
    reads.read(run.pipeline_slot, epoch, cached, &device_region, ScriptedCopy(&device_region));
    ASSERT_EQ(reads.state(run.pipeline_slot, epoch), RunRecordRead::Ok);

    // What `cleanup_execution` does before finalize ever runs.
    ASSERT_EQ(fence.retire(run), 0);

    const unsigned queries_before = events.queries();
    EXPECT_EQ(fence.poll(run), Completion::Error)
        << "a retired fence answers about itself, not about the run that armed it";
    EXPECT_EQ(events.queries(), queries_before) << "and answers without asking the device";
    EXPECT_EQ(boundaries.observed(run), Completion::Complete) << "the drain's own observation is still the run's";

    const RunExecutionOutcome outcome = decide_run_execution(evidence_for(boundaries, reads, cached, run));
    EXPECT_EQ(outcome.state, RunExecutionState::Succeeded);

    // Had the consumer re-polled the retired fence instead, this is the
    // evidence it would have assembled, and it is not the run's.
    RunOutcomeEvidence refetched = evidence_for(boundaries, reads, cached, run);
    refetched.boundaries = fence.poll(run);
    EXPECT_EQ(decide_run_execution(refetched).state, RunExecutionState::Undecided);
}

// ===== Capturing what the record read left behind =====

// An absent region is not a copy that failed: a run that never launched has no
// region and lost no read.
TEST(RunRecordReadCapture, ASlotWithNoRegionAttemptsNoCopy) {
    RunRecordReadLedgerT<kSlots> reads;
    DeviceRunResultRegion cached = published_success(9);
    ScriptedCopy copy(nullptr);

    reads.read(/*slot=*/0, /*run_epoch=*/9, cached, /*region=*/nullptr, copy);

    EXPECT_EQ(copy.calls(), 0u);
    EXPECT_EQ(reads.state(0, 9), RunRecordRead::NotAttempted);
    EXPECT_EQ(cached.published, 0u) << "the read is taken, so no predecessor's bytes are left behind";
}

TEST(RunRecordReadCapture, AFailedCopyIsRecordedAsAFailedRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion device_region = published_success(9);
    DeviceRunResultRegion cached{};
    ScriptedCopy copy(&device_region);
    copy.fails();

    reads.read(/*slot=*/0, /*run_epoch=*/9, cached, &device_region, copy);

    EXPECT_EQ(copy.calls(), 1u);
    EXPECT_EQ(reads.state(0, 9), RunRecordRead::Failed);
    EXPECT_EQ(cached.published, 0u) << "a failed copy leaves nothing partial behind";
}

// The third state: the copy succeeded onto a region no run published into.
TEST(RunRecordReadCapture, ASuccessfulCopyOfUnpublishedBytesIsStillAnOkRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion device_region{};
    DeviceRunResultRegion cached{};

    reads.read(/*slot=*/1, /*run_epoch=*/9, cached, &device_region, ScriptedCopy(&device_region));

    EXPECT_EQ(reads.state(1, 9), RunRecordRead::Ok) << "the read state is about the copy, not about the bytes";
    EXPECT_EQ(device_run_result_terminal(cached, 9).state, DeviceRunTerminalState::Undecided)
        << "the bytes are the record's own business";
}

TEST(RunRecordReadCapture, ARunThatNeverReadIsNotAttempted) {
    const RunRecordReadLedgerT<kSlots> reads;
    EXPECT_EQ(reads.state(0, 9), RunRecordRead::NotAttempted);
}

TEST(RunRecordReadCapture, AReadTakenForAnotherRunIsNotThisRunsRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion device_region = published_success(8);
    DeviceRunResultRegion cached{};

    reads.read(/*slot=*/0, /*run_epoch=*/8, cached, &device_region, ScriptedCopy(&device_region));

    EXPECT_EQ(reads.state(0, 8), RunRecordRead::Ok);
    EXPECT_EQ(reads.state(0, 9), RunRecordRead::NotAttempted) << "the successor on this slot has read nothing";
    EXPECT_EQ(reads.state(0, 0), RunRecordRead::NotAttempted) << "a run with no epoch owns no read";
}

TEST(RunRecordReadCapture, ARunsSecondReadIsNotTaken) {
    RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion device_region = published_success(9);
    DeviceRunResultRegion cached{};
    ScriptedCopy copy(&device_region);

    reads.read(/*slot=*/0, /*run_epoch=*/9, cached, &device_region, copy);
    copy.fails();
    reads.read(/*slot=*/0, /*run_epoch=*/9, cached, &device_region, copy);

    EXPECT_EQ(copy.calls(), 1u) << "later consumers share the first read's bytes";
    EXPECT_EQ(reads.state(0, 9), RunRecordRead::Ok);
    EXPECT_EQ(cached.published, 9u);
}

TEST(RunRecordReadCapture, ASlotPastCapacityIsIgnoredRatherThanRead) {
    RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion device_region = published_success(9);
    DeviceRunResultRegion cached{};
    ScriptedCopy copy(&device_region);

    reads.read(/*slot=*/static_cast<uint32_t>(kSlots), /*run_epoch=*/9, cached, &device_region, copy);

    EXPECT_EQ(copy.calls(), 0u);
    EXPECT_EQ(reads.state(static_cast<uint32_t>(kSlots), 9), RunRecordRead::NotAttempted);
}

// ===== What the rule makes of the assembled evidence =====

// Failed precedence: the rule answers Failed before it looks at the boundaries,
// because a failing run is precisely the one whose boundaries may never
// complete. Each boundary state below is the one its own drain produced.
TEST(ShadowConsumerEvidence, AFailedRecordDecidesWhateverTheBoundariesObserved) {
    ScriptedEvents complete_events;
    ScriptedEvents pending_events;
    ScriptedEvents error_events;
    ScriptedEvents unfenced_events;
    error_events.wait_fails_with(kWaitFailure);
    RunCompletionFence complete_fence(complete_events.ops());
    RunCompletionFence pending_fence(pending_events.ops());
    RunCompletionFence error_fence(error_events.ops());
    RunCompletionFence unfenced_fence(unfenced_events.ops());

    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity completed = run_at(0, 81);
    const NativeRunIdentity pending = run_at(1, 82);
    const NativeRunIdentity errored = run_at(2, 83);
    const NativeRunIdentity unfenced = run_at(3, 84);

    fence_the_run(complete_fence, completed);
    ASSERT_EQ(
        wait_and_retain_run_boundaries(complete_fence, boundaries, completed, kTimeoutMs).completion,
        Completion::Complete
    );
    fence_the_run(pending_fence, pending);
    ASSERT_EQ(poll_and_retain_run_boundaries(pending_fence, boundaries, pending), Completion::Pending);
    fence_the_run(error_fence, errored);
    ASSERT_EQ(
        wait_and_retain_run_boundaries(error_fence, boundaries, errored, kTimeoutMs).completion, Completion::Error
    );
    arm_without_boundaries(unfenced_fence, unfenced);
    ASSERT_EQ(
        wait_and_retain_run_boundaries(unfenced_fence, boundaries, unfenced, kTimeoutMs).completion,
        Completion::Unfenced
    );

    for (const NativeRunIdentity &run : {completed, pending, errored, unfenced}) {
        const DeviceRunResultRegion device_region = published_failure(run.run_epoch);
        DeviceRunResultRegion cached{};
        reads.read(run.pipeline_slot, run.run_epoch, cached, &device_region, ScriptedCopy(&device_region));

        const RunExecutionOutcome outcome = decide_run_execution(evidence_for(boundaries, reads, cached, run));
        EXPECT_EQ(outcome.state, RunExecutionState::Failed) << "slot " << run.pipeline_slot;
        EXPECT_EQ(outcome.code, kRuntimeCode);
        EXPECT_EQ(outcome.source, DeviceRunCodeSource::Header);
    }
}

// Both silences leave the region empty; only the read's bookkeeping separates
// them, and the rule names each one differently.
TEST(ShadowConsumerEvidence, ThePossibleRecordSilencesGiveDifferentReasons) {
    ScriptedEvents events;
    RunCompletionFence fence(events.ops());
    RunBoundaryLedgerT<kSlots> boundaries;
    RunRecordReadLedgerT<kSlots> reads;
    const NativeRunIdentity run = run_at(0, 91);
    fence_the_run(fence, run);
    ASSERT_EQ(wait_and_retain_run_boundaries(fence, boundaries, run, kTimeoutMs).completion, Completion::Complete);

    const DeviceRunResultRegion device_region = published_success(run.run_epoch);
    DeviceRunResultRegion cached{};
    ScriptedCopy failing_copy(&device_region);
    failing_copy.fails();
    reads.read(run.pipeline_slot, run.run_epoch, cached, &device_region, failing_copy);
    const RunExecutionOutcome failed_read = decide_run_execution(evidence_for(boundaries, reads, cached, run));

    RunRecordReadLedgerT<kSlots> no_region_reads;
    DeviceRunResultRegion no_region_cached{};
    no_region_reads.read(run.pipeline_slot, run.run_epoch, no_region_cached, nullptr, ScriptedCopy(nullptr));
    const RunExecutionOutcome no_region =
        decide_run_execution(evidence_for(boundaries, no_region_reads, no_region_cached, run));

    EXPECT_EQ(failed_read.state, RunExecutionState::Undecided);
    EXPECT_EQ(no_region.state, RunExecutionState::Undecided);
    ASSERT_NE(failed_read.reason, nullptr);
    ASSERT_NE(no_region.reason, nullptr);
    EXPECT_STRNE(failed_read.reason, no_region.reason)
        << "reporting a region that was never there as a lost copy is what the read state exists to prevent";
}

// A run whose evidence was never observed reads Pending, not a verdict — the
// shape a run that never launched leaves behind.
TEST(ShadowConsumerEvidence, AnUnobservedRunIsPendingRatherThanDecided) {
    const RunBoundaryLedgerT<kSlots> boundaries;
    const RunRecordReadLedgerT<kSlots> reads;
    const DeviceRunResultRegion cached{};

    const RunExecutionOutcome outcome = decide_run_execution(evidence_for(boundaries, reads, cached, run_at(0, 101)));

    EXPECT_EQ(outcome.state, RunExecutionState::Pending);
}
