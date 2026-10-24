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

#include "host_build_graph/kernel_graph_packet_parser.h"

namespace {
class HbgPacketParserTest : public ::testing::Test {
protected:
    static constexpr size_t payload_offset = 256;
    static constexpr size_t payload_bytes = 128;
    std::array<std::byte, sizeof(SimplerKernelInvocationHeader) + payload_offset + payload_bytes> packet{};

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
        invocation.callable_id = 1;
        invocation.generation = 19;
        invocation.payload_bytes = packet.size() - sizeof(invocation);
        std::memcpy(packet.data(), &invocation, sizeof(invocation));
        hbg::GraphPacketHeader header{};
        header.magic = hbg::GRAPH_PACKET_MAGIC;
        header.version = hbg::GRAPH_PACKET_VERSION;
        header.header_bytes = sizeof(header);
        header.total_bytes = invocation.payload_bytes;
        header.payload_offset = payload_offset;
        header.payload_bytes = payload_bytes;
        header.slot_generation = 8;
        header.callable_hash = header.argument_hash = header.function_hash = 1;
        header.task_window = 16;
        header.sm_offset = 64;
        header.region_count = 1;
        header.destinations[0] = {0x100000, 4096};
        header.destinations[1] = {0x200000, 4096};
        header.device_id = 0;
        header.runtime_binary_id = 123;
        header.heap_bytes = 64;
        const hbg::GraphImageRegion region{hbg::GraphImageKind::Runtime, 0, 0, payload_bytes, 0};
        std::memcpy(packet.data() + sizeof(invocation) + sizeof(header), &region, sizeof(region));
        write_header(header);
    }
    hbg::GraphPacketStatus parse(hbg::ParsedGraphPacket &out) const {
        return hbg::parse_graph_packet(packet.data(), packet.size(), hbg::GraphPacketAddress::HostTemplate, out);
    }
};

TEST_F(HbgPacketParserTest, ParserProducesOneCompleteBorrowedView) {
    make_packet();
    hbg::ParsedGraphPacket parsed;
    ASSERT_EQ(parse(parsed), hbg::GraphPacketStatus::Ok);
    EXPECT_EQ(parsed.invocation.callable_id, 1);
    EXPECT_EQ(parsed.graph.slot_generation, 8U);
    EXPECT_EQ(parsed.payload, packet.data() + sizeof(SimplerKernelInvocationHeader) + payload_offset);
    EXPECT_EQ(parsed.regions[1].bytes, payload_bytes);
    EXPECT_EQ(parsed.regions[2].bytes, 0U);
}

TEST_F(HbgPacketParserTest, ParsingFailureDoesNotPublishPartialResult) {
    make_packet();
    hbg::ParsedGraphPacket parsed{};
    parsed.graph.device_id = 55;
    const auto before = parsed;
    packet.back() = std::byte{1};
    EXPECT_EQ(parse(parsed), hbg::GraphPacketStatus::InvalidChecksum);
    EXPECT_EQ(std::memcmp(&before, &parsed, sizeof(parsed)), 0);
    EXPECT_EQ(
        hbg::parse_graph_packet(packet.data(), 1, hbg::GraphPacketAddress::HostTemplate, parsed),
        hbg::GraphPacketStatus::InvalidEnvelope
    );
    EXPECT_EQ(std::memcmp(&before, &parsed, sizeof(parsed)), 0);
}

TEST_F(HbgPacketParserTest, AddressModesShareParserButRequireDifferentPatchState) {
    make_packet();
    hbg::ParsedGraphPacket parsed;
    ASSERT_EQ(parse(parsed), hbg::GraphPacketStatus::Ok);
    EXPECT_EQ(
        hbg::parse_graph_packet(packet.data(), packet.size(), hbg::GraphPacketAddress::DeviceCopy, parsed),
        hbg::GraphPacketStatus::InvalidHeader
    );
    auto header = packet_header();
    header.inline_payload_addr =
        reinterpret_cast<uintptr_t>(packet.data()) + sizeof(SimplerKernelInvocationHeader) + payload_offset;
    // CANN's address patch must not require recomputing the checksum.
    std::memcpy(packet.data() + sizeof(SimplerKernelInvocationHeader), &header, sizeof(header));
    EXPECT_EQ(
        hbg::parse_graph_packet(packet.data(), packet.size(), hbg::GraphPacketAddress::DeviceCopy, parsed),
        hbg::GraphPacketStatus::Ok
    );
    EXPECT_EQ(parse(parsed), hbg::GraphPacketStatus::InvalidHeader);
}

TEST_F(HbgPacketParserTest, RejectsUnsupportedVersionAndInvalidRegionBounds) {
    make_packet();
    hbg::ParsedGraphPacket parsed{};
    parsed.graph.device_id = 55;
    const auto before = parsed;
    auto header = packet_header();
    ++header.version;
    write_header(header);
    EXPECT_EQ(parse(parsed), hbg::GraphPacketStatus::InvalidHeader);
    EXPECT_EQ(std::memcmp(&before, &parsed, sizeof(parsed)), 0);
    make_packet();
    hbg::GraphImageRegion region{hbg::GraphImageKind::Runtime, 0, 0, payload_bytes + 64, 0};
    std::memcpy(
        packet.data() + sizeof(SimplerKernelInvocationHeader) + sizeof(hbg::GraphPacketHeader), &region, sizeof(region)
    );
    write_header(packet_header());
    EXPECT_EQ(parse(parsed), hbg::GraphPacketStatus::InvalidRegion);
    EXPECT_EQ(std::memcmp(&before, &parsed, sizeof(parsed)), 0);
}

TEST_F(HbgPacketParserTest, RejectsNonzeroPaddingEvenWithMatchingChecksum) {
    make_packet();
    const size_t padding =
        sizeof(SimplerKernelInvocationHeader) + sizeof(hbg::GraphPacketHeader) + sizeof(hbg::GraphImageRegion);
    packet[padding] = std::byte{1};
    write_header(packet_header());
    hbg::ParsedGraphPacket parsed;
    EXPECT_EQ(parse(parsed), hbg::GraphPacketStatus::InvalidHeader);
}

}  // namespace
