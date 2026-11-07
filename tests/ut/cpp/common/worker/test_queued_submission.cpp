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

#include "../task_interface/call_argument_snapshot.h"
#include "native_submission.h"
#include "queued_submission.h"

namespace {
struct FakeRun {
    using Deadline = std::chrono::steady_clock::time_point;
    std::shared_ptr<std::atomic<bool>> complete;
    bool done() { return *complete; }
    bool owner_recovery_required() const { return !*complete; }
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

TEST(QueuedSubmissionTest, CancelBeforeSubmitNeverCallsExecutorAndFailsWait) {
    bool called = false;
    Submission submission([&] {
        called = true;
        return FakeRun{};
    });
    EXPECT_FALSE(submission.wait_until(std::chrono::steady_clock::now()));
    EXPECT_TRUE(submission.cancel(std::make_exception_ptr(std::runtime_error("cancelled"))));
    EXPECT_THROW(submission.wait_until(FakeRun::Deadline::max()), std::runtime_error);
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
    SimplerNativeSubmission api{
        this,
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

TEST(NativeSubmissionRefTest, RejectsMissingCallbackBeforeRetaining) {
    RefCounter counter;
    counter.api.submit = nullptr;
    EXPECT_THROW(NativeSubmissionRef ref(counter.api), std::invalid_argument);
    EXPECT_EQ(counter.references, 0);
}

TEST(QueuedSubmissionTest, NativeHostSnapshotSurvivesProducerMutationAndOwnerDestruction) {
    auto complete = std::make_shared<std::atomic<bool>>(true);
    std::unique_ptr<Submission> queued;
    {
        int32_t values[] = {99, 7, 11, 13};
        ChipStorageTaskArgs args;
        ChipTensor view{};
        view.buffer.addr = reinterpret_cast<uintptr_t>(values);
        view.shapes[0] = 2;
        view.strides[0] = 2;
        view.ndims = 1;
        view.dtype = DataType::INT32;
        view.address_space = AddressSpace::HOST;
        view.transfer = TensorTransfer::NONE;
        view.start_offset = 1;
        view.buffer.size = sizeof(values);
        args.add_tensor(view);
        args.add_scalar(41);
        CallArgumentSnapshot snapshot(args, {ArgDirection::IN});
        queued = std::make_unique<Submission>([snapshot, complete] {
            const auto &tensor = snapshot.args().tensor(0);
            const auto *data = tensor.data_as<int32_t>();
            EXPECT_EQ(data[tensor.start_offset], 7);
            EXPECT_EQ(data[tensor.start_offset + tensor.strides[0]], 13);
            EXPECT_EQ(snapshot.args().scalar(0), 41u);
            return FakeRun{complete};
        });
        values[1] = 19;
    }
    std::thread consumer([&] {
        queued->submit();
    });
    consumer.join();
    EXPECT_TRUE(queued->wait_until(FakeRun::Deadline::max()));
    std::thread destroy([submission = std::move(queued)] {});
    destroy.join();
}

TEST(QueuedSubmissionTest, HostSnapshotRejectsOutOfBoundsViewBeforeReading) {
    int32_t value = 7;
    ChipStorageTaskArgs args;
    ChipTensor tensor{};
    tensor.buffer = {reinterpret_cast<uintptr_t>(&value), sizeof(value)};
    tensor.shapes[0] = tensor.strides[0] = tensor.ndims = 1;
    tensor.dtype = DataType::INT32;
    tensor.address_space = AddressSpace::HOST;
    tensor.transfer = TensorTransfer::NONE;
    tensor.start_offset = UINT64_MAX;
    args.add_tensor(tensor);
    EXPECT_THROW(CallArgumentSnapshot(args, {ArgDirection::IN}), std::invalid_argument);
}

TEST(QueuedSubmissionTest, HostSnapshotBorrowsDeviceAndWritableHostData) {
    int32_t value = 7;
    ChipTensor tensor{};
    tensor.buffer = {reinterpret_cast<uintptr_t>(&value), sizeof(value)};
    tensor.shapes[0] = tensor.strides[0] = tensor.ndims = 1;
    tensor.dtype = DataType::INT32;
    tensor.transfer = TensorTransfer::NONE;
    for (auto space : {AddressSpace::DEVICE, AddressSpace::HOST}) {
        tensor.address_space = space;
        ChipStorageTaskArgs args;
        args.add_tensor(tensor);
        CallArgumentSnapshot snapshot(args, {space == AddressSpace::DEVICE ? ArgDirection::IN : ArgDirection::INOUT});
        EXPECT_EQ(snapshot.args().tensor(0).buffer.addr, tensor.buffer.addr);
    }
}

}  // namespace

TEST(QueuedSubmissionTest, RecoveryProofComesFromTheRun) {
    auto complete = std::make_shared<std::atomic<bool>>(false);
    Submission submission([complete] {
        return FakeRun{complete};
    });
    EXPECT_TRUE(submission.owner_recovery_required());
    submission.submit();
    EXPECT_TRUE(submission.owner_recovery_required());
    *complete = true;
    EXPECT_FALSE(submission.owner_recovery_required());
}

TEST(QueuedSubmissionTest, SubmitFailureWithoutRunHasNoCleanupProof) {
    Submission submission([]() -> FakeRun {
        throw std::runtime_error("unknown submission");
    });
    EXPECT_THROW(submission.submit(), std::runtime_error);
    EXPECT_TRUE(submission.owner_recovery_required());
}
