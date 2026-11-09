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

#include "platform_comm/ccu/ccu_instance_run_lease.h"

namespace {
using namespace simpler::ccu;
constexpr int HOST_ERROR = -1000;

struct Backend {
    PollResult poll_result = PollResult::PENDING;
    StopResult drain_result = StopResult::UNCONFIRMED;
    int polls = 0;
    int drains = 0;
    int closes = 0;
};

PollResult poll(void *handle) {
    auto &backend = *static_cast<Backend *>(handle);
    ++backend.polls;
    return backend.poll_result;
}
StopResult drain(void *handle) {
    auto &backend = *static_cast<Backend *>(handle);
    ++backend.drains;
    return backend.drain_result;
}
StopResult close_program(void *handle) {
    ++static_cast<Backend *>(handle)->closes;
    return StopResult::CONFIRMED;
}
int32_t build(const BuildContext *context, Program *program) {
    *program = {ABI_VERSION, sizeof(Program), context->platform_context, 42, drain, close_program, poll};
    return 0;
}

std::shared_ptr<Instance> instance(const std::shared_ptr<Backend> &backend) {
    auto module = std::make_shared<int>(1);
    auto result = std::make_shared<Instance>(
        std::vector<FunctionDefinition>{{module, build}},
        Domain{{1, 2, 3}, 0, {{0x1000, 256, 0}}, backend, backend.get()}
    );
    result->prepare();
    return result;
}

TEST(CcuRunLease, OrdinaryStreamFenceDoesNotProveCcuCompletion) {
    auto backend = std::make_shared<Backend>();
    auto program = instance(backend);
    InstanceRunLease lease({program}, HOST_ERROR);
    EXPECT_EQ(lease.poll_completion(false), PollResult::PENDING);
    EXPECT_FALSE(lease.drain_completion(false));
    EXPECT_EQ(backend->polls, 0);
    EXPECT_EQ(backend->drains, 0);
    EXPECT_EQ(lease.poll_completion(true), PollResult::PENDING);
    EXPECT_FALSE(lease.release_after_readers(true));
    EXPECT_EQ(backend->drains, 0);
    EXPECT_THROW(program->release_run(true), std::runtime_error);
    EXPECT_THROW(program->retain_run(), std::runtime_error);
    backend->poll_result = PollResult::COMPLETE;
    backend->drain_result = StopResult::CONFIRMED;
    EXPECT_EQ(lease.poll_completion(true), PollResult::COMPLETE);
    EXPECT_TRUE(lease.release_after_readers(true));
}

TEST(CcuRunLease, NonblockingPendingPollDoesNotDrainOrPoisonReuse) {
    auto backend = std::make_shared<Backend>();
    auto program = instance(backend);
    InstanceRunLease lease({program}, HOST_ERROR);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(lease.poll_completion(true), PollResult::PENDING);
    EXPECT_EQ(backend->drains, 0);
    EXPECT_FALSE(lease.failed());
    backend->poll_result = PollResult::COMPLETE;
    backend->drain_result = StopResult::CONFIRMED;
    EXPECT_EQ(lease.poll_completion(true), PollResult::COMPLETE);
    EXPECT_TRUE(lease.release_after_readers(true));
    EXPECT_TRUE(lease.release_after_readers(false));
    InstanceRunLease next({program}, HOST_ERROR);
    EXPECT_EQ(next.poll_completion(true), PollResult::COMPLETE);
    EXPECT_TRUE(next.release_after_readers(true));
    EXPECT_EQ(backend->polls, 5);
}

TEST(CcuRunLease, UnconfirmedDrainKeepsOwnershipAndPoisonsTheInstance) {
    auto backend = std::make_shared<Backend>();
    std::weak_ptr<Backend> owner = backend;
    auto program = instance(backend);
    InstanceRunLease lease({program}, HOST_ERROR);
    EXPECT_FALSE(lease.drain_completion(true));
    EXPECT_TRUE(lease.failed());
    EXPECT_FALSE(lease.release_after_readers(false));
    EXPECT_EQ(backend->closes, 0);
    backend.reset();
    ASSERT_FALSE(owner.expired());
    owner.lock()->drain_result = StopResult::CONFIRMED;
    EXPECT_TRUE(lease.drain_completion(true));
    EXPECT_TRUE(lease.release_after_readers(true));
    EXPECT_THROW(program->retain_run(), std::runtime_error);
    EXPECT_TRUE(program->close());
    EXPECT_TRUE(owner.expired());
}

TEST(CcuRunLease, PartialRetainRollsBackOnlyThisConstructor) {
    auto first_backend = std::make_shared<Backend>();
    auto second_backend = std::make_shared<Backend>();
    first_backend->drain_result = second_backend->drain_result = StopResult::CONFIRMED;
    auto first = instance(first_backend);
    auto second = instance(second_backend);
    InstanceRunLease busy({second}, HOST_ERROR);
    EXPECT_THROW((InstanceRunLease({first, second}, HOST_ERROR)), std::runtime_error);
    InstanceRunLease recovered({first}, HOST_ERROR);
    EXPECT_TRUE(recovered.drain_completion(true));
    EXPECT_TRUE(recovered.release_after_readers(true));
    EXPECT_THROW(second->retain_run(), std::runtime_error);
    EXPECT_TRUE(busy.drain_completion(true));
    EXPECT_TRUE(busy.release_after_readers(true));
}

TEST(CcuRunLease, PollFailureRetainsOwnershipUntilDrainIsConfirmed) {
    auto backend = std::make_shared<Backend>();
    backend->poll_result = PollResult::FAILED;
    auto program = instance(backend);
    InstanceRunLease lease({program}, HOST_ERROR);
    EXPECT_EQ(lease.poll_completion(true), PollResult::FAILED);
    EXPECT_TRUE(lease.failed());
    EXPECT_FALSE(lease.release_after_readers(false));
    backend->drain_result = StopResult::CONFIRMED;
    EXPECT_TRUE(lease.drain_completion(true));
    EXPECT_TRUE(lease.release_after_readers(true));
    EXPECT_THROW(program->retain_run(), std::runtime_error);
}

TEST(CcuRunLease, LeaseDestructionDoesNotReleaseUnconfirmedOwners) {
    auto backend = std::make_shared<Backend>();
    std::weak_ptr<Backend> owner = backend;
    {
        auto program = instance(backend);
        InstanceRunLease lease({program}, HOST_ERROR);
        EXPECT_EQ(lease.poll_completion(true), PollResult::PENDING);
    }
    backend.reset();
    EXPECT_FALSE(owner.expired());
}

TEST(CcuRunLease, RunReadersReleaseOnlyAfterTheFinalReaderOwnerEnds) {
    auto backend = std::make_shared<Backend>();
    auto program = instance(backend);
    auto lease = std::make_shared<InstanceRunLease>(std::vector<std::shared_ptr<Instance>>{program}, HOST_ERROR);
    RunReaders readers;
    readers.lease = lease;
    EXPECT_EQ(readers.poll().state, ReaderState::PENDING);
    EXPECT_FALSE(readers.stopped());
    EXPECT_EQ(backend->drains, 0);
    backend->poll_result = PollResult::COMPLETE;
    backend->drain_result = StopResult::CONFIRMED;
    EXPECT_EQ(readers.poll().state, ReaderState::STOPPED);
    EXPECT_TRUE(readers.stopped());
    lease.reset();
    EXPECT_THROW(program->retain_run(), std::runtime_error);
    readers.lease.reset();
    InstanceRunLease next({program}, HOST_ERROR);
    EXPECT_EQ(next.poll().state, ReaderState::STOPPED);
    EXPECT_TRUE(next.release_after_readers(true));
}

TEST(CcuRunLease, LaterStopProofDoesNotEraseAnUnconfirmedFailure) {
    auto backend = std::make_shared<Backend>();
    auto program = instance(backend);
    RunReaders readers;
    readers.lease = std::make_shared<InstanceRunLease>(std::vector<std::shared_ptr<Instance>>{program}, HOST_ERROR);
    const auto failure = readers.drain();
    EXPECT_EQ(failure.state, ReaderState::UNCONFIRMED);
    EXPECT_EQ(failure.error_code, HOST_ERROR);
    EXPECT_FALSE(readers.stopped());
    backend->drain_result = StopResult::CONFIRMED;
    const auto stopped = readers.drain();
    EXPECT_EQ(stopped.state, ReaderState::STOPPED);
    EXPECT_EQ(stopped.error_code, HOST_ERROR);
    EXPECT_TRUE(readers.stopped());
    EXPECT_EQ(readers.poll().error_code, HOST_ERROR);
    readers.lease.reset();
    EXPECT_THROW(program->retain_run(), std::runtime_error);
    EXPECT_TRUE(program->close());
}

TEST(CcuRunLease, ConfirmedDrainLeavesProgramReusable) {
    auto backend = std::make_shared<Backend>();
    backend->drain_result = StopResult::CONFIRMED;
    auto program = instance(backend);
    {
        InstanceRunLease first({program}, HOST_ERROR);
        const auto result = first.drain();
        EXPECT_EQ(result.state, ReaderState::STOPPED);
        EXPECT_EQ(result.error_code, 0);
    }
    EXPECT_EQ(backend->closes, 0);
    InstanceRunLease next({program}, HOST_ERROR);
    EXPECT_EQ(next.poll().state, ReaderState::PENDING);
    EXPECT_EQ(next.drain().state, ReaderState::STOPPED);
    EXPECT_EQ(backend->drains, 2);
    EXPECT_EQ(backend->closes, 0);
    EXPECT_TRUE(next.release_after_readers(true));
}

TEST(CcuRunLease, ZeroErrorCodeIsRejectedBeforeRetainingInstances) {
    auto backend = std::make_shared<Backend>();
    backend->drain_result = StopResult::CONFIRMED;
    auto program = instance(backend);
    EXPECT_THROW((InstanceRunLease({program}, 0)), std::invalid_argument);
    InstanceRunLease accepted({program}, HOST_ERROR);
    EXPECT_EQ(accepted.drain().state, ReaderState::STOPPED);
    EXPECT_TRUE(accepted.release_after_readers(true));
}
}  // namespace
