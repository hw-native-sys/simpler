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
    int launches{};
    int aborts{};
    int ready_error{};
    int stream_error{};
    int fail_launch{-1};
    std::map<uint64_t, bool> events;
    std::map<uint64_t, StreamState> streams;
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
        *state = streams[stream];
        return stream_error;
    }
    int abort_stream(uint64_t) override {
        ++aborts;
        return 0;
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
    EXPECT_EQ(sdk.aborts, 2);
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_FALSE(run->confirmed_stopped());
}
TEST(CcuRunLifecycle, PartialLaunchAbortsOnlyAcceptedMissions) {
    FakeSdk sdk;
    sdk.fail_launch = 1;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(10, programs(), {mission(0), mission(1)}, sdk);
    EXPECT_THROW(run->launch_all(), std::runtime_error);
    EXPECT_EQ(sdk.aborts, 1);
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
}  // namespace
