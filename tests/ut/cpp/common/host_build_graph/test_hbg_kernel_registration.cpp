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
#include <array>
#include <cstddef>

#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_entry_args.h"
#include "callable_protocol.h"
#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_graph_execution_entry.h"
#include "host_build_graph/kernel_graph_tensor_validation.h"
#include "host_build_graph/device/graph_image_source.h"
#include "host_build_graph/kernel_graph_slot_control.h"

namespace {
class HbgKernelRegistrationTest : public ::testing::Test {
protected:
    alignas(1024) std::array<std::byte, 16384> storage{};
    hbg::GraphSlotRegistry *registry{nullptr};
    simpler::kernel::HbgCallableBinding callable{};

    void SetUp() override {
        hbg::GraphSlotRegistration registration{};
        registration.magic = hbg::GRAPH_SLOT_MAGIC;
        registration.version = hbg::GRAPH_SLOT_VERSION;
        registration.bytes = sizeof(registration);
        registration.flags = hbg::GRAPH_SLOT_FROZEN_SERIAL;
        registration.device_id = 0;
        registration.slot_generation = 7;
        registration.runtime_binary_id = 123;
        const auto address = reinterpret_cast<uint64_t>(storage.data());
        registration.destinations[0] = {address, 4096};
        registration.destinations[1] = {address + 4096, 4096};
        registration.destinations[2] = {address + 8192, 1024};
        registration.registry = {address + 12288, sizeof(hbg::GraphSlotRegistry)};
        ASSERT_TRUE(
            hbg::calculate_graph_slot_packet_capacity(registration.destinations, registration.max_packet_bytes)
        );
        registration.checksum = hbg::calculate_graph_slot_registration_checksum(registration);
        ASSERT_EQ(hbg::install_graph_execution_slot(&registration, sizeof(registration)), hbg::GraphSlotStatus::Ok);
        registry = hbg::current_graph_slot_registry();
        callable = {address + 4096, 7, 11, 0, 1, 1, 0};
        platform_aicpu_affinity_set_thread_idx(0);
        set_platform_entry_args(0, {});
    }
    void TearDown() override {
        if (registry != nullptr) {
            (void)simpler_aicpu_l1_hbg_revoke_callable(&callable);
            EXPECT_TRUE(hbg::detach_graph_slot_registry(registry));
        }
        set_platform_entry_args(0, {});
    }
};

TEST_F(HbgKernelRegistrationTest, RejectsNullExternalDeviceBacking) {
    uint32_t shape[] = {16};
    auto tensor = simpler::hbg::make_tensor_external(reinterpret_cast<void *>(4096), shape, 1, DataType::FLOAT32);
    tensor.address_space = AddressSpace::DEVICE;
    ASSERT_TRUE(hbg::is_valid_graph_tensor_descriptor(tensor));
    tensor.buffer.addr = 0;
    EXPECT_FALSE(hbg::is_valid_graph_tensor_descriptor(tensor));
    tensor.owner_task_id = TaskId::make_sub_task(0, 0);
    EXPECT_TRUE(hbg::is_valid_graph_tensor_descriptor(tensor));
}

TEST_F(HbgKernelRegistrationTest, SlotControlAbiRejectsMalformedAndForeignDetach) {
    EXPECT_NE(simpler_aicpu_l1_hbg_register_execution_slot(nullptr), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_detach_execution_slot(nullptr), 0);
    auto registration = registry->registration;
    EXPECT_EQ(simpler_aicpu_l1_hbg_register_execution_slot(&registration), 0);
    ++registration.checksum;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_execution_slot(&registration), 0);
    hbg::GraphSlotDetachRequest detach{reinterpret_cast<uint64_t>(registry), 7, 123, 0, 0};
    auto foreign = detach;
    ++foreign.context_generation;
    EXPECT_NE(simpler_aicpu_l1_hbg_detach_execution_slot(&foreign), 0);
    foreign = detach;
    foreign.reserved = 1;
    EXPECT_NE(simpler_aicpu_l1_hbg_detach_execution_slot(&foreign), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_detach_execution_slot(&detach), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_detach_execution_slot(&detach), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_detach_execution_slot(&foreign), 0);
    registry = nullptr;
}

TEST_F(HbgKernelRegistrationTest, RegistrationIsIdempotentAndRejectsReplacementWithoutRevoke) {
    EXPECT_EQ(simpler_aicpu_l1_hbg_register_callable(&callable), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_register_callable(&callable), 0);
    auto replacement = callable;
    ++replacement.callable_generation;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&replacement), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_revoke_callable(&replacement), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_revoke_callable(&callable), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_revoke_callable(&callable), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_register_callable(&replacement), 0);
    EXPECT_EQ(simpler_aicpu_l1_hbg_revoke_callable(&replacement), 0);
}

TEST_F(HbgKernelRegistrationTest, RejectsWrongContextAndMalformedIdentity) {
    auto invalid = callable;
    ++invalid.context_generation;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&invalid), 0);
    invalid = callable;
    invalid.callable_generation = 0;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&invalid), 0);
    invalid = callable;
    invalid.reserved = 1;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&invalid), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(nullptr), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_revoke_callable(nullptr), 0);
}

TEST_F(HbgKernelRegistrationTest, ControlAbiRejectsEveryMalformedBindingField) {
    for (int field = 0; field < 8; ++field) {
        SCOPED_TRACE(field);
        auto invalid = callable;
        switch (field) {
        case 0:
            invalid.runtime_address = 0;
            break;
        case 1:
            invalid.context_generation = 0;
            break;
        case 2:
            invalid.callable_generation = 0;
            break;
        case 3:
            invalid.callable_id = -1;
            break;
        case 4:
            invalid.callable_id = MAX_REGISTERED_CALLABLE_IDS;
            break;
        case 5:
            invalid.tensor_count = -1;
            break;
        case 6:
            invalid.scalar_count = CHIP_MAX_SCALAR_ARGS + 1;
            break;
        default:
            invalid.reserved = 1;
            break;
        }
        EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&invalid), 0);
        EXPECT_NE(simpler_aicpu_l1_hbg_revoke_callable(&invalid), 0);
    }
}

TEST_F(HbgKernelRegistrationTest, RejectsControlRecordsWithoutRegisteredContext) {
    ASSERT_TRUE(hbg::detach_graph_slot_registry(registry));
    registry = nullptr;
    EXPECT_NE(simpler_aicpu_l1_hbg_register_callable(&callable), 0);
    EXPECT_NE(simpler_aicpu_l1_hbg_revoke_callable(&callable), 0);
}

TEST_F(HbgKernelRegistrationTest, RestoredDefinitionsRequireTheKernelEntryRoute) {
    Runtime runtime;
    runtime.publish_graph_section(GraphSectionSource::RestoredSlot, reinterpret_cast<uint64_t>(storage.data()), 64);
    EXPECT_EQ(graph_image_view_for_reader(runtime).section, nullptr);
    PlatformEntryArgs args{};
    args.source = static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation);
    set_platform_entry_args(0, args);
    EXPECT_EQ(graph_image_view_for_reader(runtime).section, nullptr);
    args.graph_source = static_cast<uint32_t>(GraphSectionSource::RestoredSlot);
    set_platform_entry_args(0, args);
    EXPECT_EQ(graph_image_view_for_reader(runtime).section, storage.data());
}

TEST_F(HbgKernelRegistrationTest, OrdinaryRunDoesNotRestoreAndMalformedKernelRunIsRejected) {
    PlatformEntryArgs args{};
    for (auto source : {EntryArgsSource::Descriptor, EntryArgsSource::LaunchEnvelope}) {
        args.source = static_cast<uint32_t>(source);
        set_platform_entry_args(0, args);
        EXPECT_EQ(hbg::restore_kernel_graph(nullptr), 0);
        EXPECT_EQ(hbg::retire_kernel_graph(nullptr, -5), 0);
    }
    args = {};
    args.source = static_cast<uint32_t>(EntryArgsSource::HbgKernelInvocation);
    set_platform_entry_args(0, args);
    EXPECT_NE(hbg::restore_kernel_graph(nullptr), 0);
}
}  // namespace
