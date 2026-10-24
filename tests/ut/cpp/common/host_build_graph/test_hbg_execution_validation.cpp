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
#include <cstring>

#include "host_build_graph/kernel_graph_execution_validation.h"
#include "host_build_graph/kernel_graph_slot_control.h"
#include "host_build_graph/kernel_graph_packet_reader.h"
#include "host_build_graph/kernel_graph_packet_parser.h"

namespace {
class HbgExecutionValidationTest : public ::testing::Test {
protected:
    alignas(1024) hbg::GraphSlotRegistry registry{};
    hbg::GraphSlotRegistration registration{};

    static constexpr size_t payload_offset = 256;
    static constexpr size_t payload_bytes = 128;
    std::array<std::byte, sizeof(SimplerKernelInvocationHeader) + payload_offset + payload_bytes> packet{};
    simpler::kernel::PreparedInvocationView trusted{1, 0, 0, 19};

    hbg::GraphPacketHeader packet_header() const {
        hbg::GraphPacketHeader header{};
        std::memcpy(&header, packet.data() + sizeof(SimplerKernelInvocationHeader), sizeof(header));
        return header;
    }
    void write_header(hbg::GraphPacketHeader header) {
        auto *destination = packet.data() + sizeof(SimplerKernelInvocationHeader);
        std::memcpy(destination, &header, sizeof(header));
        header.checksum = hbg::graph_packet_checksum(packet.data(), packet.size());
        std::memcpy(destination, &header, sizeof(header));
    }
    void make_packet() {
        packet.fill(std::byte{0});
        SimplerKernelInvocationHeader invocation{};
        invocation.mode = SIMPLER_MODE_KERNEL;
        invocation.callable_id = trusted.callable_id;
        invocation.generation = trusted.generation;
        invocation.payload_bytes = packet.size() - sizeof(invocation);
        std::memcpy(packet.data(), &invocation, sizeof(invocation));
        hbg::GraphPacketHeader header{};
        header.magic = hbg::GRAPH_PACKET_MAGIC;
        header.version = hbg::GRAPH_PACKET_VERSION;
        header.header_bytes = sizeof(header);
        header.total_bytes = invocation.payload_bytes;
        header.payload_offset = payload_offset;
        header.payload_bytes = payload_bytes;
        header.slot_generation = registration.slot_generation;
        header.callable_hash = header.argument_hash = header.function_hash = 1;
        header.task_window = 16;
        header.sm_offset = 64;
        header.region_count = 1;
        std::memcpy(header.destinations, registration.destinations, sizeof(header.destinations));
        header.device_id = registration.device_id;
        header.runtime_binary_id = registration.runtime_binary_id;
        header.heap_bytes = 64;
        const hbg::GraphImageRegion region{hbg::GraphImageKind::Runtime, 0, 0, payload_bytes, 0};
        std::memcpy(packet.data() + sizeof(invocation) + sizeof(header), &region, sizeof(region));
        write_header(header);
    }
    hbg::GraphPacketStatus parse(hbg::ParsedGraphPacket &out) const {
        return hbg::parse_graph_packet(packet.data(), packet.size(), hbg::GraphPacketAddress::HostTemplate, out);
    }

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

TEST_F(HbgExecutionValidationTest, OverlappingPacketIsRejectedBeforeReadingAndLeavesOutputUntouched) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    hbg::ParsedGraphPacket output{};
    hbg::GraphSlotRegistration snapshot{};
    output.graph.device_id = 55;
    const auto before = output;
    bool read = false;
    hbg::GraphPacketReadOps ops{&read, [](void *context, const void *, size_t) {
                                    *static_cast<bool *>(context) = true;
                                    return true;
                                }};

    EXPECT_EQ(
        hbg::read_graph_execution_packet(&registry, sizeof(registry), 0, 123, snapshot, output, ops),
        hbg::GraphSlotStatus::SourceOverlap
    );
    EXPECT_FALSE(read);
    EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
}

TEST_F(HbgExecutionValidationTest, MissingRegistryLeavesOutputUntouched) {
    hbg::ParsedGraphPacket output{};
    hbg::GraphSlotRegistration snapshot{};
    output.graph.device_id = 55;
    const auto before = output;

    EXPECT_EQ(hbg::read_graph_execution_packet(nullptr, 0, 0, 123, snapshot, output), hbg::GraphSlotStatus::NotReady);
    EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
}

TEST_F(HbgExecutionValidationTest, PoisonedRegistryRejectsValidationWithoutChangingOutput) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    ASSERT_EQ(hbg::poison_graph_execution_slot(&registry), hbg::GraphSlotStatus::Ok);
    hbg::ParsedGraphPacket output{};
    hbg::GraphSlotRegistration snapshot{};
    output.graph.device_id = 55;
    const auto before = output;

    EXPECT_EQ(hbg::read_graph_execution_packet(nullptr, 0, 0, 123, snapshot, output), hbg::GraphSlotStatus::Poisoned);
    EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
}

TEST_F(HbgExecutionValidationTest, UnavailablePacketLeavesRegistryAndOutputUntouched) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    const auto registry_before = registry;
    std::byte packet[sizeof(SimplerKernelInvocationHeader) + sizeof(hbg::GraphPacketHeader)]{};
    hbg::ParsedGraphPacket output{};
    hbg::GraphSlotRegistration snapshot{};
    output.graph.device_id = 55;
    const auto before = output;

    bool read = false;
    hbg::GraphPacketReadOps ops{&read, [](void *context, const void *, size_t) {
                                    *static_cast<bool *>(context) = true;
                                    return false;
                                }};
    EXPECT_EQ(
        hbg::read_graph_execution_packet(packet, sizeof(packet), 0, 123, snapshot, output, ops),
        hbg::GraphSlotStatus::SourceUnavailable
    );
    EXPECT_TRUE(read);
    EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
    EXPECT_EQ(std::memcmp(&registry_before, &registry, sizeof(registry)), 0);
}

TEST_F(HbgExecutionValidationTest, ParsedStructureDoesNotAuthorizeResourceIdentity) {
    make_packet();
    const auto original = packet_header();
    for (int field = 0; field < 4; ++field) {
        auto header = original;
        hbg::GraphSlotStatus expected;
        switch (field) {
        case 0:
            ++header.device_id;
            expected = hbg::GraphSlotStatus::DeviceMismatch;
            break;
        case 1:
            ++header.slot_generation;
            expected = hbg::GraphSlotStatus::GenerationMismatch;
            break;
        case 2:
            ++header.runtime_binary_id;
            expected = hbg::GraphSlotStatus::BinaryMismatch;
            break;
        default:
            header.destinations[0].address += 1024;
            expected = hbg::GraphSlotStatus::BindingMismatch;
            break;
        }
        write_header(header);
        hbg::ParsedGraphPacket parsed;
        ASSERT_EQ(parse(parsed), hbg::GraphPacketStatus::Ok);
        hbg::ValidatedGraphExecutionBinding output{};
        output.slot.device_id = 55;
        const auto before = output;
        EXPECT_EQ(hbg::validate_graph_execution_binding(parsed, registration, trusted, output), expected);
        EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
    }
}

TEST_F(HbgExecutionValidationTest, BindingUsesParsedHeadersAndIndependentCallableIdentity) {
    make_packet();
    hbg::ParsedGraphPacket parsed;
    ASSERT_EQ(parse(parsed), hbg::GraphPacketStatus::Ok);
    hbg::ValidatedGraphExecutionBinding output;
    ASSERT_EQ(hbg::validate_graph_execution_binding(parsed, registration, trusted, output), hbg::GraphSlotStatus::Ok);
    EXPECT_EQ(output.packet.payload, parsed.payload);
    const auto before = output;
    auto stale = trusted;
    ++stale.generation;
    EXPECT_EQ(
        hbg::validate_graph_execution_binding(parsed, registration, stale, output),
        hbg::GraphSlotStatus::CallableMismatch
    );
    EXPECT_EQ(std::memcmp(&before, &output, sizeof(output)), 0);
    auto counts = trusted;
    ++counts.tensor_count;
    EXPECT_EQ(
        hbg::validate_graph_execution_binding(parsed, registration, counts, output), hbg::GraphSlotStatus::InvalidPacket
    );
}

TEST_F(HbgExecutionValidationTest, ReadBoundaryPublishesSnapshotAndParsedPacketTogether) {
    ASSERT_EQ(install(), hbg::GraphSlotStatus::Ok);
    make_packet();
    hbg::ParsedGraphPacket parsed;
    hbg::GraphSlotRegistration snapshot{};
    ASSERT_EQ(
        hbg::read_graph_execution_packet(
            packet.data(), packet.size(), 0, 123, snapshot, parsed, {}, hbg::GraphPacketAddress::HostTemplate
        ),
        hbg::GraphSlotStatus::Ok
    );
    EXPECT_EQ(snapshot.slot_generation, 8U);
    hbg::ValidatedGraphExecutionBinding binding;
    EXPECT_EQ(hbg::validate_graph_execution_binding(parsed, snapshot, trusted, binding), hbg::GraphSlotStatus::Ok);
    const auto before = snapshot;
    const auto parsed_before = parsed;
    packet.back() = std::byte{1};
    EXPECT_EQ(
        hbg::read_graph_execution_packet(
            packet.data(), packet.size(), 0, 123, snapshot, parsed, {}, hbg::GraphPacketAddress::HostTemplate
        ),
        hbg::GraphSlotStatus::InvalidPacket
    );
    EXPECT_EQ(std::memcmp(&before, &snapshot, sizeof(snapshot)), 0);
    EXPECT_EQ(std::memcmp(&parsed_before, &parsed, sizeof(parsed)), 0);
}

}  // namespace
