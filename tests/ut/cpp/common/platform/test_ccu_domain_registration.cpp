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

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "platform_comm/ccu/ccu_domain_registration.h"

namespace {
using simpler::ccu::DomainRegistrationError;
using simpler::ccu::DomainRegistrationRequest;
using simpler::ccu::DomainRegistrationSdk;
using simpler::ccu::MissionRegistrationResources;
using simpler::ccu::NativeKernelDefinition;

class FakeSdk final : public DomainRegistrationSdk {
public:
    struct Observation {
        std::string operation;
        const void *entry{};
        SimplerCcuRegistrationBinding binding{};
        std::string name;
        uint64_t program_handle{};
    };

    int query_resources(const void *entry, const void *const *arguments, uint32_t argument_count) override {
        EXPECT_EQ(argument_count, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT);
        observe("query", entry, arguments, {});
        return query_error;
    }
    int validate_capacity() override {
        observations.push_back({"capacity"});
        return capacity_error;
    }
    int register_start() override {
        observations.push_back({"start"});
        return start_error;
    }
    int register_kernel(
        const std::string &name, const void *entry, const void *const *arguments, uint32_t argument_count,
        uint64_t *program_handle
    ) override {
        EXPECT_EQ(argument_count, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT);
        observe("register", entry, arguments, name);
        if (throw_on_register) throw std::runtime_error("register adapter failure");
        if (register_error_at >= 0 && register_calls++ == register_error_at) return register_error;
        *program_handle = next_program++;
        observations.back().program_handle = *program_handle;
        return 0;
    }
    int register_end() override {
        observations.push_back({"end"});
        return end_error;
    }
    int get_task_argument_count(uint64_t program_handle, uint32_t *argument_count) override {
        observations.push_back({"task_args", nullptr, {}, {}, program_handle});
        *argument_count = task_argument_count;
        return task_argument_error;
    }

    int query_error{};
    int capacity_error{};
    int start_error{};
    int register_error_at{-1};
    int register_error{44};
    bool throw_on_register{};
    int end_error{};
    int task_argument_error{};
    uint32_t task_argument_count{SIMPLER_CCU_TASK_ARGUMENT_COUNT};
    std::vector<Observation> observations;

private:
    void observe(std::string operation, const void *entry, const void *const *arguments, const std::string &name) {
        ASSERT_NE(arguments, nullptr);
        ASSERT_NE(arguments[0], nullptr);
        observations.push_back(
            {std::move(operation), entry, *static_cast<const SimplerCcuRegistrationBinding *>(arguments[0]), name}
        );
    }

    int register_calls{};
    uint64_t next_program{100};
};

NativeKernelDefinition kernel(uint32_t kernel_id, uint32_t, uintptr_t entry) {
    NativeKernelDefinition definition;
    definition.record.struct_size = sizeof(SimplerCcuKernelRecord);
    definition.record.kernel_id = kernel_id;
    definition.record.capability_flags = SIMPLER_CCU_KERNEL_CAP_MASK;
    definition.record.registration_binding_size = sizeof(SimplerCcuRegistrationBinding);
    definition.record.registration_argument_count = SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT;
    definition.record.task_argument_count = SIMPLER_CCU_TASK_ARGUMENT_COUNT;
    definition.entry = reinterpret_cast<const void *>(entry);
    return definition;
}

DomainRegistrationRequest request() {
    return {91, 1, 2, {0xabc}, "test_ccu", {{0x300, 0x301, 0x302}, {0x400, 0x401, 0x402}}};
}

TEST(CcuDomainRegistration, QueriesBeforeOneRegistrationTransactionAndBindsEachMission) {
    const std::vector<NativeKernelDefinition> kernels{kernel(3, 7, 0x103), kernel(4, 9, 0x104)};
    FakeSdk sdk;

    const auto registered = simpler::ccu::register_domain_kernels(kernels, request(), sdk);

    ASSERT_EQ(registered.size(), 2U);
    EXPECT_EQ(registered[0].kernel_id, 3U);
    EXPECT_EQ(registered[0].program_handle, 100U);
    EXPECT_EQ(registered[1].kernel_id, 4U);
    EXPECT_EQ(registered[1].program_handle, 101U);
    ASSERT_EQ(sdk.observations.size(), 9U);
    EXPECT_EQ(sdk.observations[0].operation, "query");
    EXPECT_EQ(sdk.observations[1].operation, "query");
    EXPECT_EQ(sdk.observations[2].operation, "capacity");
    EXPECT_EQ(sdk.observations[3].operation, "start");
    EXPECT_EQ(sdk.observations[4].operation, "register");
    EXPECT_EQ(sdk.observations[5].operation, "register");
    EXPECT_EQ(sdk.observations[6].operation, "end");
    EXPECT_EQ(sdk.observations[7].operation, "task_args");
    EXPECT_EQ(sdk.observations[8].operation, "task_args");
    EXPECT_EQ(sdk.observations[0].binding.ccu_id, 0U);
    EXPECT_EQ(sdk.observations[1].binding.ccu_id, 1U);
    EXPECT_EQ(sdk.observations[0].binding.rank_id, 1U);
    EXPECT_EQ(sdk.observations[0].binding.rank_count, 2U);
    EXPECT_EQ(sdk.observations[0].binding.channels[0], 0xabcU);
    EXPECT_EQ(sdk.observations[0].binding.trigger_event_handle, 0x301U);
    EXPECT_EQ(sdk.observations[0].binding.done_event_handle, 0x302U);
    EXPECT_EQ(sdk.observations[4].name, "test_ccu_d91_k3_r1_m0");
}

TEST(CcuDomainRegistration, QueryFailureDoesNotStartRegistration) {
    FakeSdk sdk;
    sdk.query_error = 31;
    auto single_mission = request();
    single_mission.missions.pop_back();

    try {
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103)}, single_mission, sdk);
        FAIL() << "expected query failure";
    } catch (const DomainRegistrationError &error) {
        EXPECT_EQ(error.operation(), "HcommCcuKernelQueryResReq");
        EXPECT_EQ(error.native_error(), 31);
        EXPECT_EQ(error.cleanup_error(), 0);
    }
    ASSERT_EQ(sdk.observations.size(), 1U);
    EXPECT_EQ(sdk.observations[0].operation, "query");
}

TEST(CcuDomainRegistration, RequiresEveryPeerAndPublishesRankIndexedChannels) {
    auto topology = request();
    topology.rank_count = 3;
    topology.channels[2] = 0xdef;
    FakeSdk sdk;
    (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 9, 0x104)}, topology, sdk);
    EXPECT_EQ(sdk.observations.front().binding.channels[0], 0xabcU);
    EXPECT_EQ(sdk.observations.front().binding.channels[1], 0U);
    EXPECT_EQ(sdk.observations.front().binding.channels[2], 0xdefU);

    topology.channels[2] = 0;
    FakeSdk missing;
    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 9, 0x104)}, topology, missing),
        DomainRegistrationError
    );
    EXPECT_TRUE(missing.observations.empty());
}

TEST(CcuDomainRegistration, RegisterFailureEndsTheRoundAndPreservesTheFirstError) {
    FakeSdk sdk;
    sdk.register_error_at = 1;
    sdk.register_error = 41;
    sdk.end_error = 42;

    try {
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 7, 0x104)}, request(), sdk);
        FAIL() << "expected register failure";
    } catch (const DomainRegistrationError &error) {
        EXPECT_EQ(error.operation(), "HcommCcuKernelRegister");
        EXPECT_EQ(error.native_error(), 41);
        EXPECT_EQ(error.cleanup_error(), 42);
    }
    ASSERT_FALSE(sdk.observations.empty());
    EXPECT_EQ(sdk.observations.back().operation, "end");
}

TEST(CcuDomainRegistration, RegisterExceptionStillEndsTheRound) {
    FakeSdk sdk;
    sdk.throw_on_register = true;
    auto single_mission = request();
    single_mission.missions.pop_back();

    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103)}, single_mission, sdk), std::runtime_error
    );
    ASSERT_FALSE(sdk.observations.empty());
    EXPECT_EQ(sdk.observations.back().operation, "end");
}

TEST(CcuDomainRegistration, TaskArgumentMismatchFailsAfterTheRegistrationRoundCloses) {
    FakeSdk sdk;
    sdk.task_argument_count = SIMPLER_CCU_TASK_ARGUMENT_COUNT - 1;

    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 7, 0x104)}, request(), sdk),
        DomainRegistrationError
    );
    ASSERT_GE(sdk.observations.size(), 2U);
    EXPECT_EQ(sdk.observations[sdk.observations.size() - 2].operation, "end");
    EXPECT_EQ(sdk.observations.back().operation, "task_args");
}

TEST(CcuDomainRegistration, MissingOrDuplicateMissionBindingsFailBeforeSdkCalls) {
    auto missing = request();
    missing.missions.pop_back();
    FakeSdk missing_sdk;
    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 7, 0x104)}, missing, missing_sdk),
        DomainRegistrationError
    );
    EXPECT_TRUE(missing_sdk.observations.empty());

    auto duplicate = request();
    duplicate.missions.push_back(duplicate.missions.front());
    FakeSdk duplicate_sdk;
    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103)}, duplicate, duplicate_sdk),
        DomainRegistrationError
    );
    EXPECT_TRUE(duplicate_sdk.observations.empty());

    auto unknown = request();
    unknown.missions.push_back({0x990, 0x991, 0x992});
    FakeSdk unknown_sdk;
    EXPECT_THROW(
        (void)simpler::ccu::register_domain_kernels({kernel(3, 7, 0x103), kernel(4, 7, 0x104)}, unknown, unknown_sdk),
        DomainRegistrationError
    );
    EXPECT_TRUE(unknown_sdk.observations.empty());
}

TEST(CcuDomainRegistration, ListPositionDefinesIdEvenWhenModuleKernelIdsMatch) {
    FakeSdk sdk;
    const auto registered =
        simpler::ccu::register_domain_kernels({kernel(99, 0, 0x103), kernel(99, 0, 0x104)}, request(), sdk);
    ASSERT_EQ(registered.size(), 2U);
    EXPECT_EQ(registered[0].ccu_id, 0U);
    EXPECT_EQ(registered[1].ccu_id, 1U);
    EXPECT_EQ(sdk.observations[0].entry, reinterpret_cast<const void *>(0x103));
    EXPECT_EQ(sdk.observations[1].entry, reinterpret_cast<const void *>(0x104));
}

TEST(CcuDomainRegistration, SelectedKernelKeepsCallableIdAndLogicalChannelOrder) {
    auto topology = request();
    topology.missions.resize(1);
    topology.ccu_ids = {7};
    FakeSdk sdk;
    const auto registered = simpler::ccu::register_domain_kernels({kernel(99, 0, 0x103)}, topology, sdk);
    ASSERT_EQ(registered.size(), 1U);
    EXPECT_EQ(registered[0].ccu_id, 7U);
    EXPECT_EQ(sdk.observations.front().binding.ccu_id, 7U);
    EXPECT_EQ(sdk.observations.front().binding.rank_id, 1U);
    EXPECT_EQ(sdk.observations.front().binding.channels[0], 0xabcU);
}
TEST(CcuDomainRegistration, RankSpecializationMismatchFailsBeforeSdkCalls) {
    auto definition = kernel(99, 0, 0x103);
    definition.record.required_rank_count = 4;
    auto topology = request();
    topology.missions.resize(1);
    FakeSdk sdk;
    EXPECT_THROW(simpler::ccu::register_domain_kernels({definition}, topology, sdk), DomainRegistrationError);
    EXPECT_TRUE(sdk.observations.empty());
}
TEST(CcuDomainRegistration, DuplicateSelectedIdsFailBeforeSdkCalls) {
    auto topology = request();
    topology.ccu_ids = {7, 7};
    FakeSdk sdk;
    EXPECT_THROW(
        simpler::ccu::register_domain_kernels({kernel(1, 0, 0x103), kernel(2, 0, 0x104)}, topology, sdk),
        DomainRegistrationError
    );
    EXPECT_TRUE(sdk.observations.empty());
}
}  // namespace
