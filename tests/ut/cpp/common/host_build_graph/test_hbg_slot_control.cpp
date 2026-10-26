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

#include <cstring>

#include "host_build_graph/kernel_graph_slot_control.h"

namespace {
class HbgSlotControlTest : public ::testing::Test {
protected:
    alignas(1024) hbg::GraphSlotRegistry registry{};
    alignas(1024) hbg::GraphSlotRegistry other{};
    hbg::GraphSlotRegistration registration{};

    void SetUp() override {
        registration.magic = hbg::GRAPH_SLOT_MAGIC;
        registration.version = hbg::GRAPH_SLOT_VERSION;
        registration.bytes = sizeof(registration);
        registration.flags = hbg::GRAPH_SLOT_FROZEN_SERIAL;
        registration.device_id = 0;
        registration.slot_generation = 8;
        registration.runtime_binary_id = 123;
        registration.destinations[0] = {0x100000, 4096};
        registration.destinations[1] = {0x200000, 4096};
        registration.registry = {reinterpret_cast<uintptr_t>(&registry), sizeof(registry)};
        seal(registration);
        ASSERT_TRUE(hbg::is_valid_graph_slot_registration(registration));
    }
    void TearDown() override {
        if (auto *current = hbg::current_graph_slot_registry()) hbg::detach_graph_slot_registry(current);
    }
    static void seal(hbg::GraphSlotRegistration &value) {
        ASSERT_TRUE(hbg::calculate_graph_slot_packet_capacity(value.destinations, value.max_packet_bytes));
        value.checksum = hbg::calculate_graph_slot_registration_checksum(value);
    }
    hbg::GraphSlotStatus install() { return hbg::install_graph_execution_slot(&registration, sizeof(registration)); }
};

TEST_F(HbgSlotControlTest, DuplicateInstallPreservesExecutionState) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    registry.execution.attempt = 19;
    const auto before = registry;
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
}

TEST_F(HbgSlotControlTest, DetachedInstallPreservesExecutionState) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    registry.execution.attempt = 19;
    ASSERT_TRUE(hbg::detach_graph_slot_registry(&registry));
    const auto before = registry;
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
}

TEST_F(HbgSlotControlTest, DetachedPoisonedSlotCannotBeRevived) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    ASSERT_EQ(hbg::poison_graph_execution_slot(&registry), hbg::GraphSlotStatus::Ok);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(&registry));
    const auto before = registry;
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
    EXPECT_EQ(hbg::current_graph_slot_registry(), nullptr);
}

TEST_F(HbgSlotControlTest, ConflictingRegistrationDoesNotModifyBoundSlot) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    const auto before = registry;
    registration.destinations[0].capacity *= 2;
    seal(registration);
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
}

TEST_F(HbgSlotControlTest, AnotherSlotIsRejectedBeforeInitialization) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    const auto before = registry;
    const auto other_before = other;
    registration.registry.address = reinterpret_cast<uintptr_t>(&other);
    seal(registration);
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
    EXPECT_EQ(std::memcmp(&other_before, &other, sizeof(other)), 0);
}

TEST_F(HbgSlotControlTest, StaleDetachKeepsCurrentBinding) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    hbg::GraphSlotDetachRequest request{reinterpret_cast<uintptr_t>(&registry), 7, 123, 0, 0};
    EXPECT_EQ(simpler_aicpu_l1_hbg_detach_execution_slot(&request), -1);
    EXPECT_EQ(hbg::current_graph_slot_registry(), &registry);
    request.context_generation = 8;
    EXPECT_EQ(simpler_aicpu_l1_hbg_detach_execution_slot(&request), 0);
    EXPECT_EQ(hbg::current_graph_slot_registry(), nullptr);
}

TEST_F(HbgSlotControlTest, DirectInitializationCannotClearPoisonForSameGeneration) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    ASSERT_EQ(hbg::poison_graph_execution_slot(&registry), hbg::GraphSlotStatus::Ok);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(&registry));
    const auto before = registry;
    EXPECT_EQ(hbg::initialize_graph_slot_registry(&registry, 0, 8, 123), hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
}

TEST_F(HbgSlotControlTest, FreshContextGenerationCanReuseReleasedAddress) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(&registry));
    // The owner has retired the old context and grants this allocation to a new one.
    registration.slot_generation = 9;
    seal(registration);
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(registry.context_generation, 9U);
    EXPECT_EQ(registry.execution.attempt, 0U);
}

TEST_F(HbgSlotControlTest, DetachedConflictingRegistrationDoesNotOverwriteState) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    ASSERT_TRUE(hbg::detach_graph_slot_registry(&registry));
    const auto before = registry;
    registration.destinations[0].capacity *= 2;
    seal(registration);
    EXPECT_EQ(install(), hbg::GraphSlotStatus::Conflict);
    EXPECT_EQ(std::memcmp(&before, &registry, sizeof(registry)), 0);
    EXPECT_EQ(hbg::current_graph_slot_registry(), nullptr);
}
}  // namespace
