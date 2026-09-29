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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "host/queued_stream_waits.h"
#include "host/queued_wait_discharge.h"

namespace {

using Shape = QueuedStreamWaits::Shape;
using StreamRole = RunCompletionFence::StreamRole;

constexpr NativeRunIdentity kPredecessor{9, 1, 41, 0};
constexpr NativeRunIdentity kSuccessor{10, 1, 42, 1};
constexpr int kTimeoutMs = 1500;

// Ordered fake device queues, so an event completes only once everything ahead
// of it in its own queue has — which is the property both wait shapes rest on.
// Per-operation fault counters make each failure rung reachable.
class FakeDevice {
public:
    RunCompletionFence::DeviceEventOps ops() {
        RunCompletionFence::DeviceEventOps ops;
        ops.create = [this](void **out) {
            return create(out);
        };
        ops.record = [this](void *event, void *stream) {
            return record(event, stream);
        };
        ops.query = [this](void *event, bool *complete) {
            return query(event, complete);
        };
        ops.wait = [this](void *event, int timeout_ms) {
            return wait(event, timeout_ms);
        };
        ops.destroy = [this](void *event) {
            return destroy(event);
        };
        return ops;
    }

    /** Let a stream's next `n` queued items finish, in order. */
    void complete_next(void *stream, size_t n) {
        std::deque<void *> &queue = queues_[stream];
        for (size_t i = 0; i < n && !queue.empty(); ++i) {
            completed_.push_back(queue.front());
            queue.pop_front();
        }
    }

    void fail_next_creates(int n) { create_failures_ = n; }
    void fail_next_records(int n) { record_failures_ = n; }
    void fail_next_queries(int n) { query_failures_ = n; }
    void fail_next_waits(int n) { wait_failures_ = n; }
    void fail_next_destroys(int n) { destroy_failures_ = n; }

    size_t live_events() const { return live_events_.size(); }
    int waits() const { return waits_; }
    int last_wait_timeout_ms() const { return last_wait_timeout_ms_; }

    void *predecessor_aicpu_stream() { return &pred_aicpu_marker_; }
    void *predecessor_aicore_stream() { return &pred_aicore_marker_; }
    void *successor_aicore_stream() { return &succ_aicore_marker_; }

private:
    int create(void **out) {
        if (create_failures_ > 0) {
            --create_failures_;
            return -201;
        }
        *out = reinterpret_cast<void *>(++next_event_);
        live_events_.push_back(*out);
        return 0;
    }

    int record(void *event, void *stream) {
        if (record_failures_ > 0) {
            --record_failures_;
            return -202;
        }
        completed_.erase(std::remove(completed_.begin(), completed_.end(), event), completed_.end());
        for (auto &entry : queues_) {
            std::deque<void *> &queue = entry.second;
            queue.erase(std::remove(queue.begin(), queue.end(), event), queue.end());
        }
        queues_[stream].push_back(event);
        return 0;
    }

    int query(void *event, bool *complete) {
        if (query_failures_ > 0) {
            --query_failures_;
            return -203;
        }
        *complete = completed(event);
        return 0;
    }

    int wait(void *event, int timeout_ms) {
        ++waits_;
        last_wait_timeout_ms_ = timeout_ms;
        if (wait_failures_ > 0) {
            --wait_failures_;
            return 507047;  // ACL_ERROR_RT_EVENT_SYNC_TIMEOUT
        }
        for (auto &entry : queues_) {
            const std::deque<void *> &queue = entry.second;
            for (size_t i = 0; i < queue.size(); ++i) {
                if (queue[i] != event) continue;
                complete_next(entry.first, i + 1);
                break;
            }
        }
        return completed(event) ? 0 : -204;
    }

    int destroy(void *event) {
        if (destroy_failures_ > 0) {
            --destroy_failures_;
            return -205;
        }
        live_events_.erase(std::remove(live_events_.begin(), live_events_.end(), event), live_events_.end());
        return 0;
    }

    bool completed(void *event) const {
        return std::find(completed_.begin(), completed_.end(), event) != completed_.end();
    }

    std::map<void *, std::deque<void *>> queues_;
    std::vector<void *> live_events_;
    std::vector<void *> completed_;
    uintptr_t next_event_{0};
    int create_failures_{0};
    int record_failures_{0};
    int query_failures_{0};
    int wait_failures_{0};
    int destroy_failures_{0};
    int waits_{0};
    int last_wait_timeout_ms_{0};
    char pred_aicpu_marker_{0};
    char pred_aicore_marker_{0};
    char succ_aicore_marker_{0};
};

/**
 * A predecessor fence with both boundaries recorded, which is the state any
 * wait on it starts from: the table refuses a boundary nothing recorded, and
 * that refusal is what stops a queued wait naming an event that never fires.
 */
class Fixture : public ::testing::Test {
protected:
    Fixture() :
        fence_(device_.ops()),
        waits_(device_.ops()) {}

    void SetUp() override {
        ASSERT_EQ(fence_.arm(kPredecessor), 0);
        fence_.note_kernel_submitted(kPredecessor, StreamRole::Aicore);
        ASSERT_EQ(fence_.record(kPredecessor, StreamRole::Aicore, device_.predecessor_aicore_stream()), 0);
        fence_.note_kernel_submitted(kPredecessor, StreamRole::Aicpu);
        ASSERT_EQ(fence_.record(kPredecessor, StreamRole::Aicpu, device_.predecessor_aicpu_stream()), 0);
    }

    /** The predecessor's own intra-run wait, queued and committed. */
    void open_own_wait() {
        void *boundary = nullptr;
        ASSERT_EQ(
            waits_.open(
                Shape::Own, kPredecessor, kPredecessor, fence_, StreamRole::Aicore, StreamRole::Aicpu, &boundary
            ),
            0
        );
        ASSERT_NE(boundary, nullptr);
        ASSERT_EQ(waits_.commit(Shape::Own, kPredecessor), 0);
    }

    /** A successor's cross-run wait on the predecessor's whole-operator boundary. */
    void open_cross_run_wait(bool record_proof = true) {
        void *boundary = nullptr;
        ASSERT_EQ(
            waits_.open(
                Shape::CrossRun, kSuccessor, kPredecessor, fence_, StreamRole::Aicpu, StreamRole::Aicore, &boundary
            ),
            0
        );
        ASSERT_NE(boundary, nullptr);
        ASSERT_EQ(waits_.commit(Shape::CrossRun, kSuccessor), 0);
        if (record_proof) {
            ASSERT_EQ(waits_.record_proof(kSuccessor, device_.successor_aicore_stream()), 0);
        }
    }

    /** The device running the predecessor's own AICPU queue to its boundary. */
    void complete_predecessor_boundaries() {
        device_.complete_next(device_.predecessor_aicore_stream(), 8);
        device_.complete_next(device_.predecessor_aicpu_stream(), 8);
    }

    FakeDevice device_;
    RunCompletionFence fence_;
    QueuedStreamWaits waits_;
};

TEST_F(Fixture, OwnWaitRetiresOnlyAgainstTheRunsOwnBoundaries) {
    open_own_wait();
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    // The fence cannot retire while a wait names one of its boundaries, which
    // is what stops its events being reused under a queued wait.
    EXPECT_NE(fence_.retire(kPredecessor), 0);

    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), 0);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));

    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(waits_.live_count(), 0u);
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, CrossRunWaitRetiresWhenItsProofEventCompletes) {
    open_cross_run_wait();
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));

    // The proof event is behind the wait in the successor's own stream, so
    // until that stream reaches it the reference stays live — and asking costs
    // a query, never a wait, while the boundary has not fired.
    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), 0);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(device_.waits(), 0);

    device_.complete_next(device_.successor_aicore_stream(), 1);
    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(device_.waits(), 0);
}

TEST_F(Fixture, ProvenBoundariesLicenseABoundedWaitOnTheProofEvent) {
    open_cross_run_wait();
    complete_predecessor_boundaries();

    // With the boundary fired, the successor's stream is already past the wait,
    // so the proof can only be pending for as long as the host takes to see it.
    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(device_.waits(), 1);
    EXPECT_EQ(device_.last_wait_timeout_ms(), kTimeoutMs);
}

TEST_F(Fixture, ATimingOutProofWaitReportsItsErrorAndKeepsTheReference) {
    open_cross_run_wait();
    complete_predecessor_boundaries();
    device_.fail_next_waits(1);

    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 507047);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    EXPECT_NE(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AFailingProofQueryReportsItsErrorAndKeepsTheReference) {
    open_cross_run_wait();
    device_.fail_next_queries(1);

    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), -203);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
}

TEST_F(Fixture, AWaitWhoseProofWasNeverRecordedRetiresOnlyOnQuiescence) {
    open_cross_run_wait(/*record_proof=*/false);

    // No proof was recorded, so no per-stream evidence exists at any rung, even
    // with the predecessor's own boundaries complete.
    complete_predecessor_boundaries();
    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 0);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(device_.waits(), 0);

    EXPECT_EQ(waits_.discharge_on_quiescence(kPredecessor), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AProofRecordFailureLeavesTheWaitCommittedRatherThanRevoked) {
    void *boundary = nullptr;
    ASSERT_EQ(
        waits_.open(
            Shape::CrossRun, kSuccessor, kPredecessor, fence_, StreamRole::Aicpu, StreamRole::Aicore, &boundary
        ),
        0
    );
    ASSERT_EQ(waits_.commit(Shape::CrossRun, kSuccessor), 0);
    device_.fail_next_records(1);

    EXPECT_EQ(waits_.record_proof(kSuccessor, device_.successor_aicore_stream()), -202);
    // The wait is queued whether or not the proof reached the device, so the
    // reference is what keeps the boundary event alive.
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 0);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
}

TEST_F(Fixture, AWaitNeverQueuedIsRevokedWithoutConsumingACount) {
    void *boundary = nullptr;
    ASSERT_EQ(
        waits_.open(Shape::Own, kPredecessor, kPredecessor, fence_, StreamRole::Aicore, StreamRole::Aicpu, &boundary), 0
    );
    EXPECT_EQ(waits_.revoke(Shape::Own, kPredecessor), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(waits_.live_count(), 0u);
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
    // Revoking twice would decrement a count this table no longer holds.
    EXPECT_NE(waits_.revoke(Shape::Own, kPredecessor), 0);
}

TEST_F(Fixture, OneRunHoldsAtMostOneWaitOfEachShape) {
    open_own_wait();
    void *boundary = nullptr;
    // A second token against one queued wait is the hazard the whole reference
    // model exists to prevent, so it is refused rather than given an entry.
    EXPECT_NE(
        waits_.open(Shape::Own, kPredecessor, kPredecessor, fence_, StreamRole::Aicore, StreamRole::Aicpu, &boundary), 0
    );
    EXPECT_EQ(boundary, nullptr);
    EXPECT_EQ(waits_.live_count(), 1u);

    // The other shape is a different wait in a different stream, so it is not
    // refused by that rule.
    open_cross_run_wait();
    EXPECT_EQ(waits_.live_count(), 2u);
}

TEST_F(Fixture, AnUnrecordedBoundaryCannotBeWaitedOn) {
    RunCompletionFence unrecorded(device_.ops());
    ASSERT_EQ(unrecorded.arm(kSuccessor), 0);
    void *boundary = nullptr;

    EXPECT_NE(
        waits_.open(
            Shape::CrossRun, kSuccessor, kSuccessor, unrecorded, StreamRole::Aicpu, StreamRole::Aicore, &boundary
        ),
        0
    );
    EXPECT_EQ(boundary, nullptr);
    EXPECT_EQ(waits_.live_count(), 0u);
}

TEST_F(Fixture, TokensAreDroppedOnlyAfterTheirFenceIsAbandoned) {
    open_own_wait();
    open_cross_run_wait();

    // A fence that still recognises its token refuses: the count is live over
    // there, and emptying the token here would strand it.
    EXPECT_NE(waits_.abandon(), 0);
    EXPECT_EQ(waits_.live_count(), 2u);

    fence_.abandon();
    EXPECT_EQ(waits_.abandon(), 0);
    EXPECT_EQ(waits_.live_count(), 0u);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
}

TEST_F(Fixture, ProofEventsSurviveOneWaitAndAreDestroyedOnlyWhenNoneIsLive) {
    open_cross_run_wait();
    const size_t events_with_one_wait = device_.live_events();
    // Refused rather than destroying an event a queued wait still names.
    EXPECT_NE(waits_.release_events(), 0);
    EXPECT_EQ(device_.live_events(), events_with_one_wait);

    device_.complete_next(device_.successor_aicore_stream(), 1);
    ASSERT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), 0);
    ASSERT_FALSE(waits_.holds_reference_to(kPredecessor));

    // The retired entry keeps its event, so a second join re-records rather
    // than creating one.
    open_cross_run_wait();
    EXPECT_EQ(device_.live_events(), events_with_one_wait);

    device_.complete_next(device_.successor_aicore_stream(), 1);
    ASSERT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/false, kTimeoutMs), 0);
    EXPECT_EQ(waits_.release_events(), 0);
    EXPECT_LT(device_.live_events(), events_with_one_wait);
}

TEST_F(Fixture, DischargingOneBoundaryOwnerLeavesAnotherOwnersWaitAlone) {
    open_own_wait();

    // A second predecessor on its own fence, with its own successor's wait.
    constexpr NativeRunIdentity kOtherPredecessor{11, 1, 43, 1};
    constexpr NativeRunIdentity kOtherSuccessor{12, 1, 44, 0};
    RunCompletionFence other(device_.ops());
    ASSERT_EQ(other.arm(kOtherPredecessor), 0);
    other.note_kernel_submitted(kOtherPredecessor, StreamRole::Aicpu);
    ASSERT_EQ(other.record(kOtherPredecessor, StreamRole::Aicpu, device_.predecessor_aicpu_stream()), 0);
    void *boundary = nullptr;
    ASSERT_EQ(
        waits_.open(
            Shape::CrossRun, kOtherSuccessor, kOtherPredecessor, other, StreamRole::Aicpu, StreamRole::Aicore, &boundary
        ),
        0
    );
    ASSERT_EQ(waits_.commit(Shape::CrossRun, kOtherSuccessor), 0);

    EXPECT_EQ(waits_.discharge(kPredecessor, /*boundaries_complete=*/true, kTimeoutMs), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_TRUE(waits_.holds_reference_to(kOtherPredecessor));
    EXPECT_NE(other.retire(kOtherPredecessor), 0);
}

// ---------------------------------------------------------------------------
// The ladder a runner drives over this table: which rung proves consumption,
// which return code survives, and when a reference has to stay retained. The
// two device-facing steps are counted rather than performed, which is the only
// way the rungs below the first are reachable at all — the poison step is
// recorded as having been asked for, and nothing here resets a device.
// ---------------------------------------------------------------------------

/** The runner's two steps, recorded so a case can say which rungs were used. */
class LadderSteps {
public:
    QueuedWaitDischargeOps ops() {
        QueuedWaitDischargeOps ops;
        ops.synchronize = [this]() {
            ++synchronizes_;
            return synchronize_rc_;
        };
        ops.poison = [this](int rc) {
            ++poisons_;
            poisoned_with_ = rc;
        };
        return ops;
    }

    void synchronize_fails(int rc) { synchronize_rc_ = rc; }
    int synchronizes() const { return synchronizes_; }
    int poisons() const { return poisons_; }
    int poisoned_with() const { return poisoned_with_; }

private:
    int synchronize_rc_{0};
    int synchronizes_{0};
    int poisons_{0};
    int poisoned_with_{0};
};

TEST_F(Fixture, AnUnjoinedRunClimbsNoRungAtAll) {
    LadderSteps steps;
    // Nothing named this run's boundaries, so there is nothing to prove and the
    // ordinary path pays neither a synchronize nor a poison.
    EXPECT_EQ(discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), 0);
    EXPECT_EQ(steps.synchronizes(), 0);
    EXPECT_EQ(steps.poisons(), 0);
}

TEST_F(Fixture, AProvedProofRetiresOnTheFirstRungWithoutWaitingForTheSuccessor) {
    open_own_wait();
    open_cross_run_wait();
    LadderSteps steps;

    // The predecessor's own boundaries, and the successor's proof — but *not*
    // the successor's kernels, which stay queued behind that proof. That is the
    // property the proof event exists for: the predecessor retires without
    // waiting for the run ordered behind it.
    complete_predecessor_boundaries();
    device_.complete_next(device_.successor_aicore_stream(), 1);

    EXPECT_EQ(discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(steps.synchronizes(), 0);
    EXPECT_EQ(steps.poisons(), 0);
    // And only now may the fence be retired, which is what the reference held.
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AnUnprovableProofIsRetiredByTheSynchronizeAndQuiescenceRung) {
    open_cross_run_wait(/*record_proof=*/false);
    LadderSteps steps;
    complete_predecessor_boundaries();

    EXPECT_EQ(discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), 0);
    // The first rung had no evidence, so the second was used and proved it.
    EXPECT_EQ(steps.synchronizes(), 1);
    EXPECT_EQ(steps.poisons(), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AFailingSynchronizeStopsAdmissionAndKeepsTheReference) {
    open_cross_run_wait(/*record_proof=*/false);
    LadderSteps steps;
    steps.synchronize_fails(507047);
    complete_predecessor_boundaries();

    // Nothing proved the wait consumed, so the reported code is the
    // synchronize's own, admission is asked to stop, and the events stay
    // retained rather than being released.
    EXPECT_EQ(
        discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), 507047
    );
    EXPECT_EQ(steps.poisons(), 1);
    EXPECT_EQ(steps.poisoned_with(), 507047);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    // The retention this is all for: the fence keeps its events while a stream
    // may still name one. Whether they ever become releasable is settled by a
    // later consumption proof or a confirmed reset, neither of which is here.
    EXPECT_NE(fence_.retire(kPredecessor), 0);
    EXPECT_NE(fence_.release(), 0);
}

TEST_F(Fixture, TheFirstFailureIsReportedRatherThanALaterRungs) {
    open_cross_run_wait();
    LadderSteps steps;
    // The first rung queries the proof and that query fails, so the run's
    // original failure is -203; the rungs below it then fail too.
    complete_predecessor_boundaries();
    device_.fail_next_queries(1);
    steps.synchronize_fails(507047);

    EXPECT_EQ(
        discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), -203
    );
    EXPECT_EQ(steps.synchronizes(), 1);
    EXPECT_EQ(steps.poisons(), 1);
    // Reported with the original code, not the consequence.
    EXPECT_EQ(steps.poisoned_with(), -203);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
}

TEST_F(Fixture, AFirstRungFailureSurvivesASucceedingFallback) {
    open_cross_run_wait();
    LadderSteps steps;
    complete_predecessor_boundaries();
    device_.fail_next_queries(1);

    // The fallback proves consumption, so nothing is quarantined — but the error
    // the first rung reported is still the run's, because a later rung
    // succeeding does not unmake it.
    EXPECT_EQ(
        discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, steps.ops()), -203
    );
    EXPECT_EQ(steps.synchronizes(), 1);
    EXPECT_EQ(steps.poisons(), 0);
    EXPECT_FALSE(waits_.holds_reference_to(kPredecessor));
    EXPECT_EQ(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AnUnprovenOwnWaitIsAlsoRetainedRatherThanReleased) {
    open_own_wait();
    LadderSteps steps;
    steps.synchronize_fails(507047);

    // `boundaries_complete=false` is the cleanup path: this run's completion was
    // never established, so the table has no first-rung evidence either.
    EXPECT_EQ(
        discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/false, kTimeoutMs, steps.ops()), 507047
    );
    EXPECT_EQ(steps.poisons(), 1);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
    EXPECT_NE(fence_.retire(kPredecessor), 0);
}

TEST_F(Fixture, AMissingSynchronizeStepIsAReportedFailureRatherThanASilentPass) {
    open_cross_run_wait(/*record_proof=*/false);
    complete_predecessor_boundaries();
    QueuedWaitDischargeOps none;

    // A caller that supplied no steps cannot have proved anything, so the
    // absence reports rather than reads as a retired reference.
    EXPECT_NE(discharge_queued_waits(waits_, kPredecessor, /*boundaries_complete=*/true, kTimeoutMs, none), 0);
    EXPECT_TRUE(waits_.holds_reference_to(kPredecessor));
}

}  // namespace
