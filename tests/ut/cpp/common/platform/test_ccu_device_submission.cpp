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
#include "platform_comm/ccu/ccu_device_submission.h"
#include "platform_comm/ccu/ccu_ready.h"
namespace {
TEST(CcuDeviceSubmission, ResolvesOwnerWindowCredentialsAndPositionalScalars) {
    CommCcuWindow windows[]{{0x1000, 1024, 17, 4, 0}, {0x8000, 2048, 29, 2, 0}};
    SimplerCcuWorkspace workspace{};
    workspace.struct_size = sizeof(workspace);
    workspace.windows_addr = reinterpret_cast<uint64_t>(windows);
    workspace.window_count = 2;
    workspace.window_size = sizeof(windows[0]);
    CommContext domain{};
    domain.rankNum = 2;
    domain.windowsIn[1] = 0xdead0000;
    set_comm_workspace(domain, CommWorkspaceKind::CCU, reinterpret_cast<uint64_t>(&workspace), sizeof(workspace));
    using Arg = simpler::ccu::CcuCallArg;
    const Arg args[]{{Arg::Kind::Window, 0, 64, 128}, {Arg::Kind::Window, 1, 256, 128}, {Arg::Kind::Scalar, 0, 128, 0}};
    uint64_t words[6]{}, bytes = 0;
    ASSERT_TRUE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
    EXPECT_EQ(bytes, 40u);
    EXPECT_EQ(words[0], 0x1040u);
    EXPECT_EQ(words[1], 17u);
    EXPECT_EQ(words[2], 0x8100u);
    EXPECT_EQ(words[3], 29u);
    EXPECT_EQ(words[4], 128u);
}
TEST(CcuDeviceSubmission, RejectsInvalidWindowRangesAndAddressOverflow) {
    CommCcuWindow window{0x1000, 1024, 17, 5, 0};
    SimplerCcuWorkspace workspace{};
    workspace.struct_size = sizeof(workspace);
    workspace.windows_addr = reinterpret_cast<uint64_t>(&window);
    workspace.window_count = 1;
    workspace.window_size = sizeof(window);
    CommContext domain{};
    domain.rankNum = 1;
    set_comm_workspace(domain, CommWorkspaceKind::CCU, reinterpret_cast<uint64_t>(&workspace), sizeof(workspace));
    using Arg = simpler::ccu::CcuCallArg;
    Arg args[]{{Arg::Kind::Window, 0, 960, 64}};
    uint64_t words[2]{}, bytes = 0;
    ASSERT_TRUE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
    for (const auto &invalid :
         {Arg{Arg::Kind::Window, 1, 0, 64}, Arg{Arg::Kind::Window, 0, 1024, 1},
          Arg{Arg::Kind::Window, 0, UINT64_MAX, 64}, Arg{Arg::Kind::Window, 0, 0, 0}}) {
        args[0] = invalid;
        EXPECT_FALSE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
    }
    window.address = UINT64_MAX - 32;
    args[0] = {Arg::Kind::Window, 0, 0, 64};
    EXPECT_FALSE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
    window.address = 0x1000;
    comm_workspace<CommWorkspaceKind::CCU>(domain).size = 0;
    EXPECT_FALSE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
}
struct Fixture {
    alignas(64) uint64_t parameters[2]{};
    uint64_t ready{}, trigger{}, done{};
    SimplerCcuControl mission{};
    Fixture() {
        mission.struct_size = sizeof(mission);
        mission.run_generation = 7;
        mission.parameter_alignment = 64;
        mission.parameter_addr = reinterpret_cast<uint64_t>(parameters);
        mission.parameter_size = sizeof(parameters);
        mission.ready = {reinterpret_cast<uint64_t>(&ready), 1};
        mission.trigger = {reinterpret_cast<uint64_t>(&trigger), 1};
        mission.done = {reinterpret_cast<uint64_t>(&done), 1};
    }
};
TEST(CcuDeviceSubmission, RegistersDoneBeforeWritingParametersOrTrigger) {
    Fixture f;
    const uint64_t input[]{19, 23};
    ASSERT_TRUE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [&](uint64_t address, uint64_t mask) {
        EXPECT_EQ(address, f.mission.done.address);
        EXPECT_EQ(mask, 1U);
        EXPECT_EQ(f.parameters[0], 0U);
        EXPECT_EQ(f.trigger, 0U);
        return true;
    }));
    EXPECT_EQ(f.parameters[0], 19U);
    EXPECT_EQ(f.parameters[1], 23U);
    EXPECT_EQ(f.trigger, (uint64_t{1} << 63) | 1);
}
TEST(CcuDeviceSubmission, RegistrationFailureDoesNotPublishParametersOrTrigger) {
    Fixture f;
    const uint64_t input[]{19, 23};
    EXPECT_FALSE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [](uint64_t, uint64_t) {
        return false;
    }));
    EXPECT_EQ(f.parameters[0], 0U);
    EXPECT_EQ(f.trigger, 0U);
}
TEST(CcuDeviceSubmission, DirectCallableIndexChecksBoundsAndGeneration) {
    Fixture f;
    SimplerCcuControl missions[]{f.mission, f.mission};
    missions[1].ccu_id = 1;
    SimplerCcuWorkspace workspace{
        sizeof(SimplerCcuWorkspace), 2, 7, reinterpret_cast<uint64_t>(missions), sizeof(missions[0]), 0, 0, 0, 0
    };
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 0), &missions[0]);
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 1), &missions[1]);
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 2), nullptr);
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, UINT32_MAX), nullptr);
    missions[1].run_generation = 6;
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 1), nullptr);
    missions[1].run_generation = 7;
    missions[0].run_generation = 0;
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 1), &missions[1]);
    missions[1].ccu_id = 0;
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 1), nullptr);
    workspace.struct_size = 2;
    EXPECT_EQ(simpler::ccu::ccu_control(&workspace, 0), nullptr);
}
TEST(CcuDeviceSubmission, RejectsAliasedEventsBeforeRegisteringCompletion) {
    Fixture f;
    f.mission.done.address = f.mission.trigger.address;
    const uint64_t input[]{19, 23};
    EXPECT_FALSE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [](uint64_t, uint64_t) {
        ADD_FAILURE();
        return true;
    }));
    EXPECT_EQ(f.trigger, 0U);
}
TEST(CcuDeviceSubmission, MapsSubsetRankToCommunicationRankForChannelTable) {
    CommCcuWindow windows[]{{0x1000, 1024, 17, 4, 0}, {0x8000, 1024, 29, 2, 0}};
    SimplerCcuWorkspace workspace{};
    workspace.struct_size = sizeof(workspace);
    workspace.windows_addr = reinterpret_cast<uint64_t>(windows);
    workspace.window_count = 2;
    workspace.window_size = sizeof(windows[0]);
    CommContext domain{};
    domain.rankNum = 2;
    set_comm_workspace(domain, CommWorkspaceKind::CCU, reinterpret_cast<uint64_t>(&workspace), sizeof(workspace));
    using Arg = simpler::ccu::CcuCallArg;
    const Arg args[]{{Arg::Kind::PeerRank, 1, 0, 0}};
    uint64_t words[2]{}, bytes{};
    ASSERT_TRUE(simpler::ccu::resolve_call_args(&domain, args, words, bytes));
    EXPECT_EQ(words[0], 2u);
    EXPECT_EQ(bytes, 8u);
}
TEST(CcuDeviceSubmission, AicpuConsumesEachLocalReadyBeforeAdmission) {
    Fixture first, second;
    first.ready = 1;
    SimplerCcuControl controls[]{first.mission, second.mission};
    controls[1].ccu_id = 1;
    SimplerCcuWorkspace workspace{
        sizeof(SimplerCcuWorkspace), 2, 7, reinterpret_cast<uint64_t>(controls), sizeof(controls[0]), 0, 0, 0, 0
    };
    int spins = 0;
    const int result = simpler::ccu::wait_local_ready(
        reinterpret_cast<uint64_t>(&workspace),
        [&] {
            return ++spins > 3;
        },
        [&] {
            EXPECT_EQ(first.ready, 0u);
            EXPECT_EQ(first.trigger, 0u);
            second.ready = 1;
        }
    );
    EXPECT_EQ(result, 0);
    EXPECT_EQ(second.ready, 0u);
    EXPECT_GT(spins, 0);
}
TEST(CcuDeviceSubmission, AicpuReadyTimeoutDoesNotTriggerOrReleasePendingMission) {
    Fixture f;
    SimplerCcuWorkspace workspace{
        sizeof(SimplerCcuWorkspace), 1, 7, reinterpret_cast<uint64_t>(&f.mission), sizeof(f.mission), 0, 0, 0, 0
    };
    EXPECT_EQ(
        simpler::ccu::wait_local_ready(
            reinterpret_cast<uint64_t>(&workspace),
            [] {
                return true;
            },
            [] {}
        ),
        -2
    );
    EXPECT_EQ(f.trigger, 0u);
    EXPECT_EQ(f.done, 0u);
}
}  // namespace
