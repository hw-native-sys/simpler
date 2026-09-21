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

#include <chrono>
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

#include "host/run_stream_pair.h"

namespace {

using CompletionStatus = RunStreamPair::CompletionStatus;

// Stand-ins for the PreparedExecution addresses the runner passes as owners.
// Distinct values so no linker's identical-data folding can give the two
// constants one address — the tests compare owners by address.
const int kRunA = 1;
const int kRunB = 2;
const void *const kOwnerA = &kRunA;
const void *const kOwnerB = &kRunB;

// Hands out distinct fake handles and can be told to fail the next N destroys,
// which is what makes the destroy-failure path reachable without a device.
class FakeStreams {
public:
    int create(void **out) {
        if (create_failures_ > 0) {
            --create_failures_;
            return -7;
        }
        *out = reinterpret_cast<void *>(++next_handle_);
        live_.push_back(*out);
        return 0;
    }

    int destroy(void *stream) {
        if (destroy_failures_ > 0) {
            --destroy_failures_;
            return -13;
        }
        for (auto it = live_.begin(); it != live_.end(); ++it) {
            if (*it == stream) {
                live_.erase(it);
                return 0;
            }
        }
        ADD_FAILURE() << "destroyed a handle that was not live";
        return -1;
    }

    void fail_next_destroys(int n) { destroy_failures_ = n; }
    void fail_next_creates(int n) { create_failures_ = n; }
    size_t live_count() const { return live_.size(); }

private:
    std::vector<void *> live_;
    uintptr_t next_handle_{0};
    int destroy_failures_{0};
    int create_failures_{0};
};

RunStreamPair make_pair(FakeStreams &fake) {
    return RunStreamPair(
        [&fake](void **out) {
            return fake.create(out);
        },
        [&fake](void *s) {
            return fake.destroy(s);
        }
    );
}

// Ready, submit and complete one run, the shape every reuse test starts from.
void run_once(RunStreamPair &pair, const void *owner) {
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(owner), 0);
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, owner), 0);
}

TEST(RunStreamPair, RetiredPairIsReusedWithoutCodePublication) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *aicpu = pair.aicpu();
    void *first_aicore = pair.aicore();
    ASSERT_NE(aicpu, nullptr);
    ASSERT_NE(first_aicore, nullptr);
    EXPECT_EQ(pair.created_count(), 1u);

    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);
    EXPECT_EQ(pair.aicore(), first_aicore);

    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicpu(), aicpu) << "the AICPU stream must not be recreated";
    EXPECT_EQ(pair.aicore(), first_aicore) << "a retired stream stays warm until code publication";
    EXPECT_EQ(pair.created_count(), 1u);
}

// Successive runs share one pair: the count is per publication, not per slot.
TEST(RunStreamPair, ConsecutiveRunsShareOneAicoreStream) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    run_once(pair, kOwnerA);
    void *aicore = pair.aicore();
    run_once(pair, kOwnerB);
    run_once(pair, kOwnerA);

    EXPECT_EQ(pair.aicore(), aicore);
    EXPECT_EQ(pair.created_count(), 1u);
    EXPECT_EQ(fake.live_count(), 2u) << "one AICPU and one AICore stream, no per-run churn";
}

TEST(RunStreamPair, CodePublicationInvalidatesTheAicoreStreamOnly) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    run_once(pair, kOwnerA);
    void *aicpu = pair.aicpu();
    void *first_aicore = pair.aicore();

    pair.mark_stale();
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicpu(), aicpu) << "the AICPU stream carries no instruction state";
    EXPECT_NE(pair.aicore(), first_aicore);
    EXPECT_EQ(pair.created_count(), 2u);
    EXPECT_EQ(fake.live_count(), 2u) << "the replaced stream is destroyed, not leaked";

    // One publication invalidates once, not every run after it.
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.created_count(), 2u);
}

TEST(RunStreamPair, CodePublicationDuringARunRebuildsAfterRetirement) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *active_aicore = pair.aicore();
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    // A registration lands while the run is on the device. Its own code was
    // never overwritten, so it finishes on the stream it launched with.
    pair.mark_stale();
    EXPECT_EQ(pair.aicore(), active_aicore);
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);

    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_NE(pair.aicore(), active_aicore);
    EXPECT_EQ(pair.created_count(), 2u);
}

// The invariant a shared pair needs that per-slot streams did not: a prepared
// successor overlaps its predecessor's execution, so its cleanup must not
// destroy the stream the predecessor is still running on.
TEST(RunStreamPair, ANonSubmittingOwnerRetiresNothing) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *active_aicore = pair.aicore();
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    // The successor prepared, failed after preparation, and cleans up.
    EXPECT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerB), 0);
    EXPECT_EQ(pair.aicore(), active_aicore) << "the predecessor's live stream must survive";
    EXPECT_EQ(fake.live_count(), 2u);

    // The predecessor's own poll and retirement still work.
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_COMPLETE
    );
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);
    EXPECT_EQ(pair.aicore(), active_aicore);
}

TEST(RunStreamPair, RetirementAfterCompletionIsIgnored) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    run_once(pair, kOwnerA);
    void *aicore = pair.aicore();

    // Cleanup runs after drain already published the terminal state.
    EXPECT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    EXPECT_EQ(pair.aicore(), aicore) << "a second retirement must not destroy the warm stream";
    EXPECT_EQ(fake.live_count(), 2u);
}

TEST(RunStreamPair, CompleteRetirementWithoutAHandleIsBenign) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    EXPECT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);
    EXPECT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    EXPECT_EQ(pair.created_count(), 0u);
}

TEST(RunStreamPair, PollCompletionDoesNotPermitReuseBeforeRetirement) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *aicore = pair.aicore();
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    ASSERT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_COMPLETE
    );

    EXPECT_NE(pair.ensure(), 0) << "a device-complete run is not a finalized one";
    ASSERT_EQ(pair.retire(CompletionStatus::Complete, kOwnerA), 0);
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicore(), aicore);
}

TEST(RunStreamPair, UnprovenRunDestroysTheAicoreStreamAndKeepsTheAicpuOne) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *aicpu = pair.aicpu();
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    ASSERT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    EXPECT_EQ(pair.aicore(), nullptr);
    EXPECT_EQ(pair.aicpu(), aicpu);

    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicpu(), aicpu);
    EXPECT_EQ(pair.created_count(), 2u);
}

// A launch that never reached submission owns nothing, so its cleanup is a
// no-op and the next launch reuses what is already there.
TEST(RunStreamPair, AnUnsubmittedReadyingIsReusedByTheNextLaunch) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    ASSERT_EQ(pair.ensure(), 0);
    void *aicore = pair.aicore();
    EXPECT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    EXPECT_EQ(pair.aicore(), aicore);

    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicore(), aicore);
    EXPECT_EQ(pair.created_count(), 1u);
}

TEST(RunStreamPair, AFailedStaleDestroyReportsAndKeepsTheHandle) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    run_once(pair, kOwnerA);
    void *stranded = pair.aicore();

    // 1. the image transition reports the destroy error
    pair.mark_stale();
    fake.fail_next_destroys(1);
    EXPECT_EQ(pair.ensure(), -13);

    // 2. the handle survives, so teardown still has something to reclaim
    EXPECT_EQ(pair.aicore(), stranded);
    EXPECT_EQ(pair.created_count(), 1u) << "a failed readying must not create anything";

    // 3. the stream stays stale, so the next launch retries the destroy rather
    //    than running on a stream that may hold the previous image
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_NE(pair.aicore(), stranded);
    EXPECT_EQ(pair.created_count(), 2u);
    EXPECT_EQ(fake.live_count(), 2u);

    EXPECT_EQ(pair.destroy(), 0);
    EXPECT_EQ(pair.aicore(), nullptr);
    EXPECT_EQ(fake.live_count(), 0u);
}

TEST(RunStreamPair, AHandleThatSurvivedTeardownStaysStale) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    run_once(pair, kOwnerA);
    void *stranded_aicpu = pair.aicpu();
    void *stranded_aicore = pair.aicore();

    // Teardown fails on both handles and keeps them for a later retry.
    fake.fail_next_destroys(2);
    EXPECT_EQ(pair.destroy(), -13);
    EXPECT_EQ(pair.aicpu(), stranded_aicpu);
    EXPECT_EQ(pair.aicore(), stranded_aicore);
    EXPECT_EQ(fake.live_count(), 2u);

    // A reused runner must not launch on the surviving AICore stream: readying
    // replaces it first. The AICPU stream carries no instruction state, so it
    // is kept.
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_EQ(pair.aicpu(), stranded_aicpu);
    EXPECT_NE(pair.aicore(), stranded_aicore);
    EXPECT_EQ(pair.destroy(), 0);
    EXPECT_EQ(fake.live_count(), 0u);
}

TEST(RunStreamPair, UnprovenRetirementClearsAnEarlyCompletion) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    EXPECT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_COMPLETE
    );

    // A later stream-sync error is authoritative even if a concurrent query
    // reached COMPLETE first. Error-path cleanup must not preserve that fence.
    ASSERT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            ADD_FAILURE() << "an unproven retired stream must not be queried";
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_ERROR
    );
}

TEST(RunStreamPair, PollRequiresSubmissionAndPropagatesQueryErrors) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);
    ASSERT_EQ(pair.ensure(), 0);

    EXPECT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_ERROR
    );
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_ERROR;
        }),
        SIMPLER_NATIVE_RUN_POLL_ERROR
    );
}

TEST(RunStreamPair, SubmissionRequiresBothStreamsAndAnOwner) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    EXPECT_NE(pair.mark_submitted(kOwnerA), 0) << "nothing is created yet";
    ASSERT_EQ(pair.ensure(), 0);
    EXPECT_NE(pair.mark_submitted(nullptr), 0) << "an anonymous run could never retire";
    EXPECT_EQ(pair.mark_submitted(kOwnerA), 0);
}

TEST(RunStreamPair, RetireWaitsForAnInFlightPoll) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    std::promise<void> query_entered;
    std::promise<void> release_query;
    std::shared_future<void> release = release_query.get_future().share();
    auto poll = std::async(std::launch::async, [&]() {
        return pair.poll([&](void *, void *) {
            query_entered.set_value();
            release.wait();
            return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        });
    });
    query_entered.get_future().wait();

    std::promise<void> retire_started;
    auto retire = std::async(std::launch::async, [&]() {
        retire_started.set_value();
        return pair.retire(CompletionStatus::Unproven, kOwnerA);
    });
    retire_started.get_future().wait();
    EXPECT_EQ(retire.wait_for(std::chrono::seconds(0)), std::future_status::timeout);

    release_query.set_value();
    EXPECT_EQ(poll.get(), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(retire.get(), 0);
    EXPECT_EQ(fake.live_count(), 1u);
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            return SIMPLER_NATIVE_RUN_POLL_ERROR;
        }),
        SIMPLER_NATIVE_RUN_POLL_ERROR
    );
}

TEST(RunStreamPair, PollDoesNotWaitBehindRetirement) {
    FakeStreams fake;
    std::promise<void> destroy_entered;
    std::promise<void> release_destroy;
    std::shared_future<void> release = release_destroy.get_future().share();
    RunStreamPair pair(
        [&fake](void **out) {
            return fake.create(out);
        },
        [&](void *stream) {
            destroy_entered.set_value();
            release.wait();
            return fake.destroy(stream);
        }
    );
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    auto retire = std::async(std::launch::async, [&]() {
        return pair.retire(CompletionStatus::Unproven, kOwnerA);
    });
    destroy_entered.get_future().wait();

    // The destroy holds the pair's mutex. Poll reports NOT_READY rather than
    // blocking a progress thread behind a driver call.
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            ADD_FAILURE() << "poll must not run its query while retirement holds the pair";
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_NOT_READY
    );

    release_destroy.set_value();
    EXPECT_EQ(retire.get(), 0);
}

TEST(RunStreamPair, DestroyReportsFailureAndRetriesWhatSurvived) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);
    ASSERT_EQ(pair.ensure(), 0);

    fake.fail_next_destroys(1);
    EXPECT_EQ(pair.destroy(), -13);
    EXPECT_EQ(fake.live_count(), 1u);

    EXPECT_EQ(pair.destroy(), 0);
    EXPECT_EQ(fake.live_count(), 0u);
    EXPECT_FALSE(pair.ready());
}

TEST(RunStreamPair, AbandonClearsHandlesWithoutDestroyingThem) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);

    pair.abandon();

    EXPECT_EQ(pair.aicpu(), nullptr);
    EXPECT_EQ(pair.aicore(), nullptr);
    EXPECT_FALSE(pair.ready());
    EXPECT_EQ(fake.live_count(), 2u) << "a reset device invalidated them; destroy must not be called";
    EXPECT_EQ(
        pair.poll([](void *, void *) {
            ADD_FAILURE() << "an abandoned pair has nothing to query";
            return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        }),
        SIMPLER_NATIVE_RUN_POLL_ERROR
    );
    EXPECT_EQ(pair.destroy(), 0);
}

TEST(RunStreamPair, AFailedCreateLeavesThePairEmpty) {
    FakeStreams fake;
    RunStreamPair pair = make_pair(fake);

    fake.fail_next_creates(1);
    EXPECT_NE(pair.ensure(), 0);
    EXPECT_EQ(pair.aicpu(), nullptr);
    EXPECT_EQ(pair.aicore(), nullptr);
    EXPECT_EQ(pair.created_count(), 0u);

    // The AICPU stream lands, the AICore one fails: the pair stays unusable
    // rather than half-ready.
    fake.fail_next_creates(0);
    ASSERT_EQ(pair.ensure(), 0);
    ASSERT_EQ(pair.mark_submitted(kOwnerA), 0);
    ASSERT_EQ(pair.retire(CompletionStatus::Unproven, kOwnerA), 0);
    fake.fail_next_creates(1);
    EXPECT_NE(pair.ensure(), 0);
    EXPECT_FALSE(pair.ready());
}

}  // namespace
