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

#include <atomic>
#include <future>
#include <memory>
#include <thread>

#include "native_submission.h"
#include "queued_submission.h"

namespace {
struct FakeRun {
    using Deadline = std::chrono::steady_clock::time_point;
    std::shared_ptr<std::atomic<bool>> complete;
    bool done() { return *complete; }
    bool wait_until(Deadline) { return done(); }
};
using Submission = QueuedSubmission<FakeRun>;

TEST(QueuedSubmissionTest, PreparationDoesNotExecuteAndSubmitIsOneShot) {
    int calls = 0;
    auto complete = std::make_shared<std::atomic<bool>>(false);
    Submission submission([&] {
        ++calls;
        return FakeRun{complete};
    });
    EXPECT_EQ(calls, 0);
    EXPECT_FALSE(submission.done());
    EXPECT_FALSE(submission.wait_until(std::chrono::steady_clock::now()));
    EXPECT_THROW(submission.retire(), std::logic_error);
    submission.submit();
    EXPECT_EQ(calls, 1);
    EXPECT_THROW(submission.submit(), std::logic_error);
    EXPECT_FALSE(submission.done());
    *complete = true;
    EXPECT_TRUE(submission.wait_until(FakeRun::Deadline::max()));
    submission.retire();
    EXPECT_TRUE(submission.done());
    EXPECT_TRUE(submission.wait_until(FakeRun::Deadline::max()));
}

TEST(QueuedSubmissionTest, CancelBeforeSubmitNeverCallsExecutorAndWakesWaiter) {
    bool called = false;
    Submission submission([&] {
        called = true;
        return FakeRun{};
    });
    auto result = std::async(std::launch::async, [&] {
        EXPECT_THROW(submission.wait_until(FakeRun::Deadline::max()), std::runtime_error);
    });
    EXPECT_TRUE(submission.cancel(std::make_exception_ptr(std::runtime_error("cancelled"))));
    result.get();
    EXPECT_THROW(submission.submit(), std::logic_error);
    EXPECT_FALSE(called);
    EXPECT_TRUE(submission.done());
    submission.retire();
    EXPECT_THROW(submission.wait_until(FakeRun::Deadline::max()), std::runtime_error);
}

TEST(QueuedSubmissionTest, CancelCannotRevokeSubmissionInProgress) {
    std::promise<void> entered, release;
    auto ready = release.get_future();
    auto complete = std::make_shared<std::atomic<bool>>(true);
    Submission submission([&] {
        entered.set_value();
        ready.wait();
        return FakeRun{complete};
    });
    auto submit = std::async(std::launch::async, [&] {
        submission.submit();
    });
    entered.get_future().wait();
    EXPECT_FALSE(submission.cancel(std::make_exception_ptr(std::runtime_error("cancelled"))));
    EXPECT_FALSE(submission.done());
    EXPECT_FALSE(submission.wait_until(std::chrono::steady_clock::now()));
    EXPECT_THROW(submission.retire(), std::logic_error);
    release.set_value();
    submit.get();
    EXPECT_TRUE(submission.wait_until(FakeRun::Deadline::max()));
}

TEST(QueuedSubmissionTest, SubmissionErrorRemainsObservableToEveryWaiter) {
    Submission submission([]() -> FakeRun {
        throw std::runtime_error("native admission failed");
    });
    EXPECT_THROW(submission.submit(), std::runtime_error);
    EXPECT_TRUE(submission.done());
    EXPECT_THROW(submission.wait_until(FakeRun::Deadline::max()), std::runtime_error);
    EXPECT_THROW(submission.wait_until(FakeRun::Deadline::max()), std::runtime_error);
    EXPECT_FALSE(submission.cancel(std::make_exception_ptr(std::runtime_error("cancelled"))));
}

TEST(QueuedSubmissionTest, CapturedNativeArgumentsAreReleasedAfterConsumption) {
    auto owner = std::make_shared<int>(1);
    std::weak_ptr<int> weak = owner;
    auto complete = std::make_shared<std::atomic<bool>>(true);
    Submission submission([owner, complete] {
        return FakeRun{complete};
    });
    owner.reset();
    EXPECT_FALSE(weak.expired());
    submission.submit();
    EXPECT_TRUE(weak.expired());
}

struct RefCounter {
    std::atomic<int> references{0};
    std::atomic<int> calls{0};
    SimplerNativeSubmissionV1 api{
        1, this,
        [](void *p) noexcept {
            ++static_cast<RefCounter *>(p)->references;
        },
        [](void *p) noexcept {
            --static_cast<RefCounter *>(p)->references;
        },
        [](void *p, char *, size_t) noexcept {
            ++static_cast<RefCounter *>(p)->calls;
            return 0;
        }
    };
};

TEST(NativeSubmissionRefTest, CopiesMovesAndQueueDestructionBalanceReferences) {
    RefCounter counter;
    {
        NativeSubmissionRef ref(counter.api);
        NativeSubmissionRef moved(std::move(ref));
        EXPECT_THROW(ref.submit(), std::logic_error);
        std::thread queue([copy = moved] {
            copy.submit();
        });
        queue.join();
        EXPECT_EQ(counter.calls, 1);
        EXPECT_EQ(counter.references, 1);
    }
    EXPECT_EQ(counter.references, 0);
}

TEST(NativeSubmissionRefTest, RejectsUnknownAbiBeforeRetaining) {
    RefCounter counter;
    counter.api.version = 2;
    EXPECT_THROW(NativeSubmissionRef ref(counter.api), std::invalid_argument);
    EXPECT_EQ(counter.references, 0);
}
}  // namespace
