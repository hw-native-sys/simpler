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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "host/run_completion_fence.h"

namespace {

using Completion = RunCompletionFence::Completion;
using StreamRole = RunCompletionFence::StreamRole;
using WaitReference = RunCompletionFence::WaitReference;

// The token is the count it holds, so a second copy of it would be a second
// releaser, and a move that left the source live would be the same thing.
static_assert(!std::is_copy_constructible_v<WaitReference>);
static_assert(!std::is_copy_assignable_v<WaitReference>);
static_assert(std::is_nothrow_move_constructible_v<WaitReference>);
static_assert(!std::is_move_assignable_v<WaitReference>);

constexpr NativeRunIdentity kRunA{7, 3, 11, 0};
constexpr NativeRunIdentity kRunB{7, 3, 12, 0};
constexpr int kTimeoutMs = 2000;

// Fake device streams carrying an ordered queue of items, so a boundary means
// what it means on real hardware: it completes once everything ahead of it in
// its own queue has. `complete_next` is the device making progress, and
// per-operation fault counters make every failure path reachable.
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

    /** Push a non-boundary item onto a stream, i.e. ordinary queued work. */
    void submit_work(void *stream, const std::string &label) { queues_[stream].push_back(Item{label, nullptr}); }

    /** Let a stream's next `n` queued items finish, in order. */
    void complete_next(void *stream, size_t n) {
        std::deque<Item> &queue = queues_[stream];
        for (size_t i = 0; i < n && !queue.empty(); ++i) {
            if (queue.front().event != nullptr) completed_events_.push_back(queue.front().event);
            queue.pop_front();
        }
    }

    size_t queued(void *stream) const {
        auto it = queues_.find(stream);
        return it == queues_.end() ? 0 : it->second.size();
    }

    void fail_next_creates(int n) { create_failures_ = n; }
    void fail_next_records(int n) { record_failures_ = n; }
    void fail_next_queries(int n) { query_failures_ = n; }
    void fail_next_waits(int n) { wait_failures_ = n; }
    void fail_next_destroys(int n) { destroy_failures_ = n; }

    size_t live_events() const { return live_events_.size(); }
    int last_wait_timeout_ms() const { return last_wait_timeout_ms_; }
    int waits() const { return waits_; }

    // Distinct handles standing in for the two run streams.
    void *aicore_stream() { return &aicore_marker_; }
    void *aicpu_stream() { return &aicpu_marker_; }

private:
    struct Item {
        std::string label;
        void *event;
    };

    int create(void **out) {
        if (create_failures_ > 0) {
            --create_failures_;
            return -101;
        }
        *out = reinterpret_cast<void *>(++next_event_);
        live_events_.push_back(*out);
        return 0;
    }

    int record(void *event, void *stream) {
        if (record_failures_ > 0) {
            --record_failures_;
            return -102;
        }
        // A re-record moves the boundary to the queue's current tail, which is
        // what makes a reused event answer for the run that recorded it last.
        completed_events_.erase(
            std::remove(completed_events_.begin(), completed_events_.end(), event), completed_events_.end()
        );
        for (auto &entry : queues_) {
            std::deque<Item> &queue = entry.second;
            queue.erase(
                std::remove_if(
                    queue.begin(), queue.end(),
                    [event](const Item &item) {
                        return item.event == event;
                    }
                ),
                queue.end()
            );
        }
        queues_[stream].push_back(Item{"boundary", event});
        return 0;
    }

    int query(void *event, bool *complete) {
        if (query_failures_ > 0) {
            --query_failures_;
            return -103;
        }
        *complete = is_completed(event);
        return 0;
    }

    int wait(void *event, int timeout_ms) {
        ++waits_;
        last_wait_timeout_ms_ = timeout_ms;
        if (wait_failures_ > 0) {
            --wait_failures_;
            return 507047;  // ACL_ERROR_RT_EVENT_SYNC_TIMEOUT
        }
        // A blocking wait runs the queue the boundary sits in up to and
        // including that boundary, the way the device would.
        for (auto &entry : queues_) {
            const std::deque<Item> &queue = entry.second;
            for (size_t i = 0; i < queue.size(); ++i) {
                if (queue[i].event != event) continue;
                complete_next(entry.first, i + 1);
                break;
            }
        }
        return is_completed(event) ? 0 : -104;
    }

    int destroy(void *event) {
        if (destroy_failures_ > 0) {
            --destroy_failures_;
            return -105;
        }
        live_events_.erase(std::remove(live_events_.begin(), live_events_.end(), event), live_events_.end());
        return 0;
    }

    bool is_completed(void *event) const {
        for (void *e : completed_events_) {
            if (e == event) return true;
        }
        return false;
    }

    std::map<void *, std::deque<Item>> queues_;
    std::vector<void *> live_events_;
    std::vector<void *> completed_events_;
    uintptr_t next_event_{0};
    int create_failures_{0};
    int record_failures_{0};
    int query_failures_{0};
    int wait_failures_{0};
    int destroy_failures_{0};
    int waits_{0};
    int last_wait_timeout_ms_{0};
    int aicore_marker_{1};
    int aicpu_marker_{2};
};

/** The launch shape the runners use: submit each kernel, then fence it. */
void launch(RunCompletionFence &fence, FakeDevice &device, const NativeRunIdentity &identity) {
    ASSERT_EQ(fence.arm(identity), 0);
    device.submit_work(device.aicore_stream(), "aicore_kernel");
    fence.note_kernel_submitted(identity, StreamRole::Aicore);
    ASSERT_EQ(fence.record(identity, StreamRole::Aicore, device.aicore_stream()), 0);
    device.submit_work(device.aicpu_stream(), "aicpu_kernel");
    fence.note_kernel_submitted(identity, StreamRole::Aicpu);
    ASSERT_EQ(fence.record(identity, StreamRole::Aicpu, device.aicpu_stream()), 0);
}

TEST(RunCompletionFenceTest, OneBoundaryCompleteIsNotCompletion) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    ASSERT_TRUE(fence.fenced(kRunA));

    EXPECT_EQ(fence.poll(kRunA), Completion::Pending);

    // AICore alone: the two kernels handshake, so the AICPU one may still be
    // running. Reporting completion here would free memory it is reading.
    device.complete_next(device.aicore_stream(), 2);
    EXPECT_EQ(fence.poll(kRunA), Completion::Pending);

    device.complete_next(device.aicpu_stream(), 2);
    EXPECT_EQ(fence.poll(kRunA), Completion::Complete);
}

TEST(RunCompletionFenceTest, CompletionIsStickyAcrossPolls) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    device.complete_next(device.aicore_stream(), 2);
    device.complete_next(device.aicpu_stream(), 2);
    ASSERT_EQ(fence.poll(kRunA), Completion::Complete);

    // A query that could no longer answer must not un-complete a proven run.
    device.fail_next_queries(4);
    EXPECT_EQ(fence.poll(kRunA), Completion::Complete);
}

// The property the whole change exists for: a successor queued behind this run
// must not hold its completion open.
TEST(RunCompletionFenceTest, WorkQueuedAfterTheBoundaryDoesNotDelayCompletion) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    device.submit_work(device.aicore_stream(), "successor_aicore_kernel");
    device.submit_work(device.aicpu_stream(), "successor_aicpu_kernel");

    device.complete_next(device.aicore_stream(), 2);
    device.complete_next(device.aicpu_stream(), 2);

    EXPECT_EQ(fence.poll(kRunA), Completion::Complete);
    // Both queues still hold the successor, which a whole-stream query would
    // have waited for.
    EXPECT_EQ(device.queued(device.aicore_stream()), 1u);
    EXPECT_EQ(device.queued(device.aicpu_stream()), 1u);
}

TEST(RunCompletionFenceTest, WaitRunsBothBoundariesWithTheGivenTimeout) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    EXPECT_EQ(fence.wait(kRunA, kTimeoutMs), 0);
    EXPECT_EQ(device.last_wait_timeout_ms(), kTimeoutMs);
    EXPECT_EQ(device.waits(), 2);
    EXPECT_EQ(fence.poll(kRunA), Completion::Complete);
}

TEST(RunCompletionFenceTest, WaitReportsTheDeviceErrorAndKeepsIt) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    device.fail_next_waits(1);
    EXPECT_EQ(fence.wait(kRunA, kTimeoutMs), 507047);
    // AICPU is waited first, so that is where the failure is attributed, and a
    // later poll keeps reporting the error rather than re-deciding.
    EXPECT_EQ(fence.boundary_error(kRunA, StreamRole::Aicpu), 507047);
    EXPECT_EQ(fence.boundary_error(kRunA, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.poll(kRunA), Completion::Error);
}

TEST(RunCompletionFenceTest, QueryFailureIsAnErrorNotPending) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    device.fail_next_queries(1);
    EXPECT_EQ(fence.poll(kRunA), Completion::Error);
}

TEST(RunCompletionFenceTest, EventCreationFailureLeavesTheRunUnarmed) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());

    device.fail_next_creates(1);
    EXPECT_EQ(fence.arm(kRunA), -101);
    EXPECT_FALSE(fence.armed());
    EXPECT_EQ(fence.poll(kRunA), Completion::Error);
    EXPECT_EQ(device.live_events(), 0u);

    // Nothing was submitted, so a retry is the whole recovery.
    EXPECT_EQ(fence.arm(kRunA), 0);
    EXPECT_TRUE(fence.armed());
    EXPECT_EQ(device.live_events(), 2u);
}

TEST(RunCompletionFenceTest, SecondEventCreationFailureLeavesTheRunUnarmed) {
    FakeDevice device;
    RunCompletionFence::DeviceEventOps ops = device.ops();
    RunCompletionFence::CreateEventFn create_event = ops.create;
    int creates = 0;
    ops.create = [&creates, create_event](void **out) {
        // The pair's second event is the one that fails.
        if (++creates == 2) return -101;
        return create_event(out);
    };
    RunCompletionFence fence(std::move(ops));

    EXPECT_EQ(fence.arm(kRunA), -101);
    EXPECT_FALSE(fence.armed());
    // The handle that was obtained stays owned, so the retry only creates the
    // one that is missing.
    EXPECT_EQ(device.live_events(), 1u);
    EXPECT_EQ(fence.arm(kRunA), 0);
    EXPECT_TRUE(fence.armed());
    EXPECT_EQ(device.live_events(), 2u);
    EXPECT_EQ(creates, 3);
}

// A record that fails after its kernel is submitted must leave the run holding
// device work — never looking unlaunched.
TEST(RunCompletionFenceTest, SubmittedKernelWithoutABoundaryIsUnfenced) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    ASSERT_EQ(fence.arm(kRunA), 0);

    device.submit_work(device.aicore_stream(), "aicore_kernel");
    fence.note_kernel_submitted(kRunA, StreamRole::Aicore);
    device.fail_next_records(1);
    EXPECT_EQ(fence.record(kRunA, StreamRole::Aicore, device.aicore_stream()), -102);

    EXPECT_TRUE(fence.kernel_submitted(kRunA, StreamRole::Aicore));
    EXPECT_FALSE(fence.boundary_recorded(kRunA, StreamRole::Aicore));
    EXPECT_TRUE(fence.has_device_references(kRunA));
    EXPECT_FALSE(fence.fenced(kRunA));
    // The record error is retained, so the run's own state says why it cannot
    // be decided from events.
    EXPECT_EQ(fence.boundary_error(kRunA, StreamRole::Aicore), -102);
}

// A failed record is a fact about this run, so poll reports it rather than
// falling through to `Unfenced`. Drain asks the separate question — whether a
// boundary proof exists at all — and takes its own fallback.
TEST(RunCompletionFenceTest, ARecordFailureMakesPollReportError) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    ASSERT_EQ(fence.arm(kRunA), 0);

    device.submit_work(device.aicore_stream(), "aicore_kernel");
    fence.note_kernel_submitted(kRunA, StreamRole::Aicore);
    device.fail_next_records(1);
    ASSERT_EQ(fence.record(kRunA, StreamRole::Aicore, device.aicore_stream()), -102);

    EXPECT_EQ(fence.poll(kRunA), Completion::Error);
    EXPECT_FALSE(fence.fenced(kRunA));
}

// Drain holds the state for its whole bounded wait. A progress thread polling
// then must answer "ask again" rather than block for that timeout — on a2a3 it
// would be holding the stream pair while it did. The poll runs on its own
// thread so a regression here times out and fails instead of deadlocking the
// suite against the drain it would be waiting for.
TEST(RunCompletionFenceTest, PollDoesNotWaitBehindABlockingDrain) {
    FakeDevice device;
    RunCompletionFence::DeviceEventOps ops = device.ops();
    RunCompletionFence::WaitEventFn wait_event = ops.wait;
    std::promise<void> wait_entered;
    std::promise<void> release_wait;
    std::shared_future<void> release_signal = release_wait.get_future().share();
    std::atomic<bool> entered_published{false};
    ops.wait = [&](void *event, int timeout_ms) {
        if (!entered_published.exchange(true)) wait_entered.set_value();
        release_signal.wait();
        return wait_event(event, timeout_ms);
    };
    RunCompletionFence fence(std::move(ops));
    launch(fence, device, kRunA);

    std::future<int> drain = std::async(std::launch::async, [&fence]() {
        return fence.wait(kRunA, kTimeoutMs);
    });
    wait_entered.get_future().wait();

    std::future<Completion> polled = std::async(std::launch::async, [&fence]() {
        return fence.poll(kRunA);
    });
    const bool answered = polled.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    EXPECT_TRUE(answered) << "poll blocked behind the drain's bounded wait";
    if (answered) EXPECT_EQ(polled.get(), Completion::Pending);

    release_wait.set_value();
    EXPECT_EQ(drain.get(), 0);
    if (!answered) polled.wait();
    EXPECT_EQ(fence.poll(kRunA), Completion::Complete);
}

TEST(RunCompletionFenceTest, PartialLaunchIsUnfencedAndNotWaitable) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    ASSERT_EQ(fence.arm(kRunA), 0);

    // AICore submitted and fenced, AICPU never submitted: the AICore kernel may
    // be spinning on a handshake the missing AICPU kernel will never answer, so
    // its boundary cannot complete and proves nothing on its own.
    device.submit_work(device.aicore_stream(), "aicore_kernel");
    fence.note_kernel_submitted(kRunA, StreamRole::Aicore);
    ASSERT_EQ(fence.record(kRunA, StreamRole::Aicore, device.aicore_stream()), 0);

    EXPECT_FALSE(fence.fenced(kRunA));
    EXPECT_EQ(fence.poll(kRunA), Completion::Unfenced);
    EXPECT_EQ(fence.wait(kRunA, kTimeoutMs), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(device.waits(), 0);
}

TEST(RunCompletionFenceTest, AnotherRunsIdentityReadsNothing) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    device.complete_next(device.aicore_stream(), 2);
    device.complete_next(device.aicpu_stream(), 2);
    ASSERT_EQ(fence.poll(kRunA), Completion::Complete);

    EXPECT_EQ(fence.poll(kRunB), Completion::Error);
    EXPECT_FALSE(fence.fenced(kRunB));
    EXPECT_FALSE(fence.kernel_submitted(kRunB, StreamRole::Aicore));
    EXPECT_EQ(fence.wait(kRunB, kTimeoutMs), PTO_RUNTIME_ERR_INTERNAL);
    // Retiring on behalf of a run that does not own the arming changes nothing.
    EXPECT_EQ(fence.retire(kRunB), 0);
    EXPECT_TRUE(fence.armed());
}

TEST(RunCompletionFenceTest, SlotReuseKeepsTheEventsAndDropsThePreviousFacts) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    ASSERT_EQ(fence.wait(kRunA, kTimeoutMs), 0);
    ASSERT_EQ(fence.retire(kRunA), 0);
    const size_t events_after_first_run = device.live_events();
    ASSERT_EQ(events_after_first_run, 2u);

    launch(fence, device, kRunB);
    EXPECT_EQ(device.live_events(), events_after_first_run);
    // The previous run's completion must not carry into this one.
    EXPECT_EQ(fence.poll(kRunB), Completion::Pending);
    device.complete_next(device.aicore_stream(), 2);
    device.complete_next(device.aicpu_stream(), 2);
    EXPECT_EQ(fence.poll(kRunB), Completion::Complete);
}

TEST(RunCompletionFenceTest, RetireAndReleaseAreRepeatable) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    ASSERT_EQ(fence.wait(kRunA, kTimeoutMs), 0);

    EXPECT_EQ(fence.retire(kRunA), 0);
    EXPECT_EQ(fence.retire(kRunA), 0);
    EXPECT_FALSE(fence.armed());

    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(device.live_events(), 0u);
    EXPECT_EQ(fence.release(), 0);
}

TEST(RunCompletionFenceTest, ReleaseKeepsAHandleWhoseDestroyFailed) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    device.fail_next_destroys(1);
    EXPECT_EQ(fence.release(), -105);
    EXPECT_EQ(device.live_events(), 1u);
    // The surviving handle is still owned, so a retry reaches it.
    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(device.live_events(), 0u);
}

TEST(RunCompletionFenceTest, AbandonForgetsHandlesWithoutDestroying) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    fence.abandon();
    EXPECT_FALSE(fence.armed());
    // A reset invalidated the whole generation; nothing was handed to destroy.
    EXPECT_EQ(device.live_events(), 2u);
    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(device.live_events(), 2u);
}

// ---- Cross-run join contract ---------------------------------------------

TEST(RunCompletionFenceTest, ReserveCommitReleaseIsExactlyOnce) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &ref), 0);
    ASSERT_TRUE(ref.valid());
    EXPECT_FALSE(ref.committed());
    EXPECT_NE(fence.boundary_event(kRunA, StreamRole::Aicpu), nullptr);

    ASSERT_EQ(fence.commit_wait_reference(ref), 0);
    EXPECT_TRUE(ref.committed());
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);

    ASSERT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), 0);
    EXPECT_FALSE(ref.valid());
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 0u);
    // A second release has nothing to release.
    EXPECT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(RunCompletionFenceTest, RevokeDropsAReservationWhoseWaitWasNeverQueued) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &ref), 0);
    ASSERT_EQ(fence.revoke_wait_reference(ref), 0);
    EXPECT_FALSE(ref.valid());
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicore), 0u);
    // Nothing is left holding the fence, so the slot admits its next run.
    EXPECT_EQ(fence.retire(kRunA), 0);
    EXPECT_EQ(fence.arm(kRunB), 0);
}

// The join crosses the streams, so the successor stream that completes is not
// the one naming the predecessor boundary. A proof about the wrong stream
// leaves the wait uncovered.
TEST(RunCompletionFenceTest, ReleaseNeedsTheStreamThatHoldsTheWait) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &ref), 0);
    ASSERT_EQ(fence.commit_wait_reference(ref), 0);

    EXPECT_EQ(fence.release_wait_reference(ref, StreamRole::Aicpu), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(ref.committed());
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);

    EXPECT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 0u);
}

TEST(RunCompletionFenceTest, QuiescenceReleasesAWaitNoStreamProofCovers) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &ref), 0);
    ASSERT_EQ(fence.commit_wait_reference(ref), 0);

    EXPECT_EQ(fence.release_wait_reference_on_quiescence(ref), 0);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicore), 0u);
    EXPECT_EQ(fence.release_wait_reference_on_quiescence(ref), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(RunCompletionFenceTest, AnOutstandingReferenceBlocksRetireAndRearm) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    ASSERT_EQ(fence.wait(kRunA, kTimeoutMs), 0);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &ref), 0);
    ASSERT_EQ(fence.commit_wait_reference(ref), 0);

    // This run completed, but a queued wait still names its boundary. Admitting
    // the slot's next run would re-record the event under that wait.
    EXPECT_EQ(fence.retire(kRunA), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(fence.armed());
    EXPECT_EQ(fence.arm(kRunB), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.record(kRunA, StreamRole::Aicpu, device.aicpu_stream()), PTO_RUNTIME_ERR_INTERNAL);

    ASSERT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.retire(kRunA), 0);
    EXPECT_EQ(fence.arm(kRunB), 0);
}

// A pending reservation is as blocking as a committed one: the caller may be
// between reserving and learning whether its stream wait was queued.
TEST(RunCompletionFenceTest, APendingReservationAlsoBlocksRearm) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &ref), 0);
    EXPECT_EQ(fence.retire(kRunA), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.arm(kRunB), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(RunCompletionFenceTest, AnUnrecordedBoundaryCannotBeReservedOrNamed) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    ASSERT_EQ(fence.arm(kRunA), 0);
    device.submit_work(device.aicore_stream(), "aicore_kernel");
    fence.note_kernel_submitted(kRunA, StreamRole::Aicore);

    WaitReference ref;
    EXPECT_EQ(
        fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &ref), PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_FALSE(ref.valid());
    EXPECT_EQ(fence.boundary_event(kRunA, StreamRole::Aicore), nullptr);
}

// NotStarted kernels do not imply that nothing was queued against this run: a
// successor's wait can be live while neither of this run's kernels is.
TEST(RunCompletionFenceTest, UnsubmittedKernelsStillLeaveACommittedWaitLive) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &ref), 0);
    ASSERT_EQ(fence.commit_wait_reference(ref), 0);

    RunCompletionFence successor(device.ops());
    ASSERT_EQ(successor.arm(kRunB), 0);
    EXPECT_FALSE(successor.kernel_submitted(kRunB, StreamRole::Aicore));
    EXPECT_FALSE(successor.kernel_submitted(kRunB, StreamRole::Aicpu));
    // Its own kernels never started, yet the reference it queued on the
    // predecessor is still live and must be released, not assumed away.
    EXPECT_FALSE(successor.has_device_references(kRunB));
    EXPECT_TRUE(fence.has_device_references(kRunA));
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);
    EXPECT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), 0);
}

TEST(RunCompletionFenceTest, CommitAndRevokeRejectAnEmptyReference) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference ref;
    EXPECT_EQ(fence.commit_wait_reference(ref), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.revoke_wait_reference(ref), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.release_wait_reference(ref, StreamRole::Aicore), PTO_RUNTIME_ERR_INTERNAL);

    // A reservation is not reusable storage either.
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &ref), 0);
    EXPECT_EQ(
        fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &ref), PTO_RUNTIME_ERR_INTERNAL
    );
    ASSERT_EQ(fence.revoke_wait_reference(ref), 0);
}

// A token names a count on one fence. Offered to another, it must not decrement
// whichever of that fence's counters happens to share its two roles — the other
// fence could then retire while its own wait is still queued.
TEST(RunCompletionFenceTest, AForeignFenceRejectsTheToken) {
    FakeDevice device_a;
    FakeDevice device_b;
    RunCompletionFence fence_a(device_a.ops());
    RunCompletionFence fence_b(device_b.ops());
    launch(fence_a, device_a, kRunA);
    launch(fence_b, device_b, kRunB);

    WaitReference on_a;
    ASSERT_EQ(fence_a.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &on_a), 0);
    ASSERT_EQ(fence_a.commit_wait_reference(on_a), 0);
    WaitReference on_b;
    ASSERT_EQ(fence_b.reserve_wait_reference(kRunB, StreamRole::Aicpu, StreamRole::Aicore, &on_b), 0);
    ASSERT_EQ(fence_b.commit_wait_reference(on_b), 0);

    EXPECT_EQ(fence_b.release_wait_reference(on_a, StreamRole::Aicore), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence_b.release_wait_reference_on_quiescence(on_a), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence_b.commit_wait_reference(on_a), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence_b.revoke_wait_reference(on_a), PTO_RUNTIME_ERR_INTERNAL);

    // B still holds its own reference, so it cannot retire.
    EXPECT_EQ(fence_b.committed_wait_count(StreamRole::Aicpu), 1u);
    EXPECT_EQ(fence_b.retire(kRunB), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(on_a.committed());

    ASSERT_EQ(fence_a.release_wait_reference(on_a, StreamRole::Aicore), 0);
    ASSERT_EQ(fence_b.release_wait_reference(on_b, StreamRole::Aicore), 0);
}

// "Not mine" is not "no longer live". A foreign fence emptying a live token
// would strand the count it holds on its own fence, which would then be blocked
// from retiring with nothing left to release it.
TEST(RunCompletionFenceTest, AForeignFenceCannotDiscardALiveToken) {
    FakeDevice device_a;
    FakeDevice device_b;
    RunCompletionFence fence_a(device_a.ops());
    RunCompletionFence fence_b(device_b.ops());
    launch(fence_a, device_a, kRunA);
    launch(fence_b, device_b, kRunB);

    WaitReference committed_on_a;
    ASSERT_EQ(fence_a.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &committed_on_a), 0);
    ASSERT_EQ(fence_a.commit_wait_reference(committed_on_a), 0);
    WaitReference reserved_on_a;
    ASSERT_EQ(fence_a.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &reserved_on_a), 0);

    EXPECT_EQ(fence_b.discard_stale_wait_reference(committed_on_a), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence_b.discard_stale_wait_reference(reserved_on_a), PTO_RUNTIME_ERR_INTERNAL);
    // Both tokens survive, so A keeps its route to releasing what it holds.
    EXPECT_TRUE(committed_on_a.committed());
    EXPECT_TRUE(reserved_on_a.valid());
    EXPECT_EQ(fence_a.committed_wait_count(StreamRole::Aicpu), 1u);

    ASSERT_EQ(fence_a.release_wait_reference(committed_on_a, StreamRole::Aicore), 0);
    ASSERT_EQ(fence_a.revoke_wait_reference(reserved_on_a), 0);
    EXPECT_EQ(fence_a.retire(kRunA), 0);
    // B was never holding anything of its own, and still is not.
    EXPECT_EQ(fence_b.committed_wait_count(StreamRole::Aicpu), 0u);
    EXPECT_EQ(fence_b.committed_wait_count(StreamRole::Aicore), 0u);
    EXPECT_EQ(fence_b.retire(kRunB), 0);
}

// The same fence, a later run: the token's generation is gone, so charging its
// release here would consume the current run's count.
TEST(RunCompletionFenceTest, AStaleGenerationTokenCannotChargeTheCurrentRun) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference stale;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &stale), 0);
    ASSERT_EQ(fence.commit_wait_reference(stale), 0);

    // A verified reset invalidates the whole generation, counts included.
    fence.abandon();
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 0u);

    launch(fence, device, kRunB);
    WaitReference current;
    ASSERT_EQ(fence.reserve_wait_reference(kRunB, StreamRole::Aicpu, StreamRole::Aicore, &current), 0);
    ASSERT_EQ(fence.commit_wait_reference(current), 0);
    ASSERT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);

    EXPECT_EQ(fence.release_wait_reference_on_quiescence(stale), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.release_wait_reference(stale, StreamRole::Aicore), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);

    // The reset is the proof that covers the stale token, so it is discardable —
    // and the live one is not, because emptying it would leak its count.
    EXPECT_EQ(fence.discard_stale_wait_reference(stale), 0);
    EXPECT_FALSE(stale.valid());
    EXPECT_EQ(fence.discard_stale_wait_reference(current), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);
    ASSERT_EQ(fence.release_wait_reference(current, StreamRole::Aicore), 0);
}

// Two genuine waits on the same boundary and waiter roles: moving one token and
// releasing both halves must consume one count, not two.
TEST(RunCompletionFenceTest, AMovedTokenLeavesNoSecondReleaser) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference first;
    WaitReference second;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &first), 0);
    ASSERT_EQ(fence.commit_wait_reference(first), 0);
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &second), 0);
    ASSERT_EQ(fence.commit_wait_reference(second), 0);
    ASSERT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 2u);

    WaitReference moved = std::move(first);
    EXPECT_TRUE(moved.committed());
    EXPECT_FALSE(first.valid());

    ASSERT_EQ(fence.release_wait_reference(moved, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);
    // The moved-from half must not be able to consume `second`'s count.
    EXPECT_EQ(fence.release_wait_reference(first, StreamRole::Aicore), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);

    ASSERT_EQ(fence.release_wait_reference(second, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 0u);
}

// release() is the last place an event could be destroyed under a queued wait,
// so it is guarded like arm/record/retire rather than trusted.
TEST(RunCompletionFenceTest, ReleaseRefusesWhileAReferenceIsOutstanding) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);
    ASSERT_EQ(fence.wait(kRunA, kTimeoutMs), 0);

    WaitReference committed;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &committed), 0);
    ASSERT_EQ(fence.commit_wait_reference(committed), 0);

    EXPECT_EQ(fence.release(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(device.live_events(), 2u);
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 1u);
    EXPECT_TRUE(committed.committed());

    ASSERT_EQ(fence.release_wait_reference(committed, StreamRole::Aicore), 0);
    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(device.live_events(), 0u);
}

// A reservation is as blocking as a committed reference: its holder may be
// between reserving and learning whether its stream wait was queued.
TEST(RunCompletionFenceTest, ReleaseRefusesWhileAReservationIsOutstanding) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference reserved;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicore, StreamRole::Aicpu, &reserved), 0);

    EXPECT_EQ(fence.release(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(device.live_events(), 2u);

    ASSERT_EQ(fence.revoke_wait_reference(reserved), 0);
    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(device.live_events(), 0u);
}

// The reset path is the one that may drop a count without a per-stream proof,
// and it must still hand back the events it never destroyed.
TEST(RunCompletionFenceTest, AbandonDropsReferencesWhereReleaseWillNot) {
    FakeDevice device;
    RunCompletionFence fence(device.ops());
    launch(fence, device, kRunA);

    WaitReference committed;
    ASSERT_EQ(fence.reserve_wait_reference(kRunA, StreamRole::Aicpu, StreamRole::Aicore, &committed), 0);
    ASSERT_EQ(fence.commit_wait_reference(committed), 0);

    fence.abandon();
    EXPECT_EQ(fence.committed_wait_count(StreamRole::Aicpu), 0u);
    EXPECT_FALSE(fence.armed());
    EXPECT_EQ(device.live_events(), 2u);
    // Its arming is gone, so the slot admits its next run.
    EXPECT_EQ(fence.arm(kRunB), 0);
    ASSERT_EQ(fence.discard_stale_wait_reference(committed), 0);
}

}  // namespace
