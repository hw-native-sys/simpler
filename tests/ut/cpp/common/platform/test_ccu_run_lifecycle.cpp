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
#include <map>
#include "platform_comm/ccu/ccu_run_lifecycle.h"

namespace {
using namespace simpler::ccu;

class FakeSdk final : public RunLifecycleSdk {
public:
    int stops{};
    int stop_error{};
    int launches{};
    int aborts{};
    int abort_error{};
    bool abort_throws{};
    bool abort_completes{};
    int ready_error{};
    int stream_error{};
    int fail_launch{-1};
    std::map<uint64_t, bool> events;
    std::map<uint64_t, StreamState> streams;
    std::vector<uint64_t> aborted_streams;
    std::vector<uint64_t> queried_streams;
    int launch(uint64_t, const uint64_t *, uint32_t count, uint64_t) override {
        EXPECT_EQ(count, SIMPLER_CCU_TASK_ARGUMENT_COUNT);
        return launches++ == fail_launch ? 19 : 0;
    }
    int poll_ready_event(uint64_t address, uint64_t mask, bool *ready) override {
        EXPECT_EQ(mask, 1U);
        *ready = events[address];
        events[address] = false;
        return ready_error;
    }
    int query_stream(uint64_t stream, StreamState *state) override {
        queried_streams.push_back(stream);
        *state = streams[stream];
        return stream_error;
    }
    int stop_completed_stream(uint64_t stream) override {
        ++stops;
        if (stop_error) return stop_error;
        streams[stream] = StreamState::COMPLETE;
        return 0;
    }
    int abort_stream(uint64_t stream) override {
        ++aborts;
        aborted_streams.push_back(stream);
        if (abort_throws) throw std::runtime_error("SDK abort exception");
        if (abort_completes) streams[stream] = StreamState::COMPLETE;
        return abort_error;
    }
};
MissionRunResources mission(uint32_t index) {
    return {index, 101 + index, 11 + index, {0x1000 + index * 64, 0x2000}, {0x3000 + index * 8, 1}};
}
std::vector<RegisteredKernel> programs() { return {{1, 0, 101}, {2, 1, 102}}; }

TEST(CcuRunLifecycle, ClearsReadyOnceAndWaitsForEveryMission) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(7, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    EXPECT_FALSE(run->poll_ready());
    sdk.events[0x3000] = true;
    EXPECT_FALSE(run->poll_ready());
    EXPECT_FALSE(sdk.events[0x3000]);
    sdk.events[0x3008] = true;
    EXPECT_TRUE(run->poll_ready());
    EXPECT_FALSE(sdk.events[0x3008]);
}
TEST(CcuRunLifecycle, RequiresStreamTerminationWithoutStopGateOrFinalWake) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(8, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.streams[11] = StreamState::COMPLETE;
    sdk.streams[12] = StreamState::RUNNING;
    EXPECT_FALSE(run->stop());
    EXPECT_TRUE(owner.has_active_run());
    sdk.streams[12] = StreamState::COMPLETE;
    EXPECT_TRUE(run->stop());
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_TRUE(owner.admission_open());
    EXPECT_FALSE(owner.has_active_run());
}
TEST(CcuRunLifecycle, ReadyFailureClosesAdmissionAndRetainsOwners) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(9, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.ready_error = 71;
    EXPECT_FALSE(run->poll_ready());
    EXPECT_EQ(run->primary_failure()->native_error, 71);
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(run->confirmed_stopped());
}
TEST(CcuRunLifecycle, PartialLaunchAbortsEveryAttemptedMission) {
    FakeSdk sdk;
    sdk.fail_launch = 1;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(10, programs(), {mission(0), mission(1)}, sdk);
    EXPECT_THROW(run->launch_all(), std::runtime_error);
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_FALSE(run->abort());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11, 12}));
    EXPECT_TRUE(owner.resources_retained());
}
TEST(CcuRunLifecycle, RejectsOverlapAndPreservesStreamFailure) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(11, programs(), {mission(0)}, sdk);
    EXPECT_THROW(owner.begin_run(12, programs(), {mission(0)}, sdk), std::runtime_error);
    run->launch_all();
    sdk.stream_error = 81;
    EXPECT_FALSE(run->poll_terminal());
    EXPECT_EQ(run->primary_failure()->native_error, 81);
    EXPECT_TRUE(owner.resources_retained());
}
TEST(CcuRunLifecycle, ResidentCompletionConfirmsEveryStreamAndAllowsReuse) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(12, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.streams[11] = StreamState::COMPLETE;
    sdk.streams[12] = StreamState::RUNNING;
    ASSERT_TRUE(run->finish_completed_requests());
    EXPECT_EQ(sdk.stops, 1);
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_FALSE(run->failed());
    EXPECT_TRUE(owner.admission_open());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_NO_THROW(owner.begin_run(13, programs(), {mission(0)}, sdk));
}
TEST(CcuRunLifecycle, ResidentStopFailureCannotReleaseOrReopenAdmission) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(14, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.stop_error = 91;
    EXPECT_FALSE(run->finish_completed_requests());
    EXPECT_EQ(sdk.stops, 1);
    EXPECT_EQ(run->primary_failure()->native_error, 91);
    EXPECT_FALSE(run->confirmed_stopped());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_THROW(owner.begin_run(15, programs(), {mission(0)}, sdk), std::runtime_error);
}
TEST(CcuRunLifecycle, UnknownStreamStateCannotBeReclassifiedAsExpectedStop) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(16, programs(), {mission(0)}, sdk);
    run->launch_all();
    sdk.streams[11] = StreamState::UNKNOWN;
    EXPECT_FALSE(run->finish_completed_requests());
    EXPECT_EQ(sdk.stops, 0);
    EXPECT_TRUE(owner.resources_retained());
}

TEST(CcuRunLifecycle, ConfirmedFailureAbortAllowsReclamationWithoutReopeningAdmission) {
    FakeSdk sdk;
    sdk.abort_completes = true;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(17, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.ready_error = 71;
    EXPECT_FALSE(run->poll_ready());
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(run->abort());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 71);
    EXPECT_TRUE(run->failed());
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_TRUE(run->terminal());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.has_active_run());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_THROW(owner.begin_run(18, programs(), {mission(0)}, sdk), std::runtime_error);
    EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11, 12}));
    run->abort();
    EXPECT_EQ(sdk.aborts, 2);
    EXPECT_TRUE(run->stop());
    EXPECT_EQ(run->primary_failure()->native_error, 71);
}

TEST(CcuRunLifecycle, FailedFirstLaunchIsStoppedBeforeReclamation) {
    FakeSdk sdk;
    sdk.abort_completes = true;
    sdk.fail_launch = 0;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(19, programs(), {mission(0), mission(1)}, sdk);
    EXPECT_THROW(run->launch_all(), std::runtime_error);
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(run->abort());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11}));
    EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11}));
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_THROW(run->launch_all(), std::runtime_error);
    EXPECT_TRUE(run->stop());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 19);
}

TEST(CcuRunLifecycle, FailedAbortCannotBeOverriddenByCompleteStreamQuery) {
    FakeSdk sdk;
    sdk.abort_error = 92;
    sdk.abort_completes = true;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(20, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    run->abort();
    EXPECT_EQ(sdk.aborts, 2);
    EXPECT_FALSE(run->confirmed_stopped());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(run->poll_terminal());
    EXPECT_FALSE(owner.admission_open());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 92);
}

TEST(CcuRunLifecycle, FailedAbortQueryRetainsEveryMissionAndPrimaryError) {
    FakeSdk sdk;
    sdk.abort_completes = true;
    sdk.stream_error = 93;
    sdk.ready_error = 71;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(21, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    EXPECT_FALSE(run->poll_ready());
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_FALSE(run->abort());
    EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11, 12}));
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(run->confirmed_stopped());
    ASSERT_EQ(run->failures().size(), 3U);
    EXPECT_EQ(run->primary_failure()->native_error, 71);
    EXPECT_EQ(run->failures()[1].native_error, 93);
    EXPECT_EQ(run->failures()[2].native_error, 93);
}

TEST(CcuRunLifecycle, LateAbortOfCompletedRunCannotAbortAReusedStream) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto completed = owner.begin_run(22, programs(), {mission(0)}, sdk);
    completed->launch_all();
    sdk.streams[11] = StreamState::COMPLETE;
    ASSERT_TRUE(completed->stop());
    auto active = owner.begin_run(23, programs(), {mission(0)}, sdk);
    active->launch_all();
    completed->abort();
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(owner.has_active_run());
    EXPECT_FALSE(owner.resources_retained());
}

TEST(CcuRunLifecycle, AbortBeforeLaunchClosesAdmissionWithoutCallingSdk) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(24, programs(), {mission(0)}, sdk);
    EXPECT_TRUE(run->abort());
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_TRUE(run->stop());
    EXPECT_FALSE(owner.has_active_run());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(sdk.queried_streams.empty());
    EXPECT_THROW(run->launch_all(), std::runtime_error);
}

TEST(CcuRunLifecycle, SuccessfulFailureAbortPreservesCompletedRequestStopError) {
    FakeSdk sdk;
    sdk.abort_completes = true;
    sdk.stop_error = 91;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(25, programs(), {mission(0)}, sdk);
    run->launch_all();
    EXPECT_FALSE(run->finish_completed_requests());
    EXPECT_EQ(sdk.aborts, 0);
    EXPECT_TRUE(run->abort());
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_TRUE(run->failed());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 91);
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(run->finish_completed_requests());
}

TEST(CcuRunLifecycle, SuccessfulAbortRequeriesOnlyRunningMissionsUntilComplete) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(26, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    sdk.streams[11] = StreamState::COMPLETE;
    sdk.streams[12] = StreamState::RUNNING;
    EXPECT_FALSE(run->abort());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_TRUE(run->abort_pending());
    EXPECT_FALSE(run->failed());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11, 12}));

    // Later confirmation probes must touch only unfinished streams.
    sdk.streams[11] = StreamState::ERROR;
    sdk.streams[12] = StreamState::COMPLETE;
    EXPECT_TRUE(run->abort());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11, 12}));
    EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11, 12, 12}));
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_FALSE(run->abort_pending());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_THROW(owner.begin_run(27, programs(), {mission(0)}, sdk), std::runtime_error);
    EXPECT_TRUE(run->abort());
    EXPECT_EQ(sdk.queried_streams.size(), 3U);
}

TEST(CcuRunLifecycle, AbortRetryCannotOverrideHardErrors) {
    for (int failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        FakeSdk sdk;
        DomainRunLifecycle owner;
        auto run = owner.begin_run(28, programs(), {mission(0), mission(1)}, sdk);
        run->launch_all();
        sdk.ready_error = 71;
        ASSERT_FALSE(run->poll_ready());
        sdk.streams[11] = StreamState::COMPLETE;
        sdk.streams[12] = StreamState::RUNNING;
        if (failure == 0) sdk.abort_error = 92;
        if (failure == 1) sdk.stream_error = 93;
        if (failure == 2) sdk.streams[12] = StreamState::ERROR;
        if (failure == 3) sdk.streams[12] = StreamState::UNKNOWN;
        EXPECT_FALSE(run->abort());
        EXPECT_FALSE(run->abort_pending());
        const size_t failures = run->failures().size();
        EXPECT_GT(failures, 1U);

        sdk.abort_error = 0;
        sdk.stream_error = 0;
        sdk.streams[11] = StreamState::COMPLETE;
        sdk.streams[12] = StreamState::COMPLETE;
        EXPECT_FALSE(run->abort());
        EXPECT_FALSE(run->confirmed_stopped());
        EXPECT_FALSE(run->poll_terminal());
        EXPECT_FALSE(owner.admission_open());
        EXPECT_TRUE(owner.resources_retained());
        EXPECT_EQ(run->primary_failure()->native_error, 71);
        EXPECT_EQ(run->failures().size(), failures);
        EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11, 12}));
        EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11, 12}));
    }
}

TEST(CcuRunLifecycle, PartialLaunchAbortCanFinishAfterRunningQuery) {
    FakeSdk sdk;
    sdk.fail_launch = 1;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(29, programs(), {mission(0), mission(1)}, sdk);
    ASSERT_THROW(run->launch_all(), std::runtime_error);
    sdk.streams[11] = StreamState::RUNNING;
    sdk.streams[12] = StreamState::COMPLETE;
    EXPECT_FALSE(run->abort());
    EXPECT_TRUE(run->abort_pending());
    EXPECT_FALSE(run->abort());
    EXPECT_EQ(run->failures().size(), 1U);
    sdk.streams[11] = StreamState::COMPLETE;
    EXPECT_TRUE(run->abort());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11, 12}));
    EXPECT_EQ(sdk.queried_streams, (std::vector<uint64_t>{11, 12, 11, 11}));
    EXPECT_EQ(run->primary_failure()->native_error, 19);
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_FALSE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
}

TEST(CcuRunLifecycle, AbortExceptionCannotTurnIntoAQueryOnlyRetry) {
    FakeSdk sdk;
    sdk.abort_throws = true;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(30, programs(), {mission(0), mission(1)}, sdk);
    run->launch_all();
    EXPECT_THROW(run->abort(), std::runtime_error);
    EXPECT_FALSE(run->abort_pending());
    sdk.abort_throws = false;
    sdk.streams[11] = StreamState::COMPLETE;
    sdk.streams[12] = StreamState::COMPLETE;
    EXPECT_FALSE(run->abort());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(owner.admission_open());
    EXPECT_FALSE(run->confirmed_stopped());
    EXPECT_EQ(sdk.aborted_streams, (std::vector<uint64_t>{11}));
    EXPECT_TRUE(sdk.queried_streams.empty());
}
}  // namespace
