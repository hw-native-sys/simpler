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
#include <vector>

#include "platform_comm/comm_domain_use_wire.h"

namespace {
std::vector<CommDomainUse> examples() {
    return {
        {CommDomainKind::LOCAL, 0, UINT64_C(0x1020304050607080), UINT64_C(0x102030405000)},
        {CommDomainKind::GLOBAL, 0, UINT64_C(0x1020304050607080), UINT64_C(0x203040506000)}
    };
}

CommDomainUsesHeader header_of(const std::vector<uint8_t> &bytes) {
    CommDomainUsesHeader header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    return header;
}

void replace_header(std::vector<uint8_t> &bytes, const CommDomainUsesHeader &header) {
    std::memcpy(bytes.data(), &header, sizeof(header));
}

TEST(CommDomainUseWire, EmptySidecarStillHasAnExplicitVersionedHeader) {
    std::array<uint8_t, sizeof(CommDomainUsesHeader)> bytes{};
    EXPECT_EQ(comm_domain_uses_wire_size({}), bytes.size());
    write_comm_domain_uses(bytes.data(), bytes.size(), {});
    EXPECT_TRUE(read_comm_domain_uses(bytes.data(), bytes.size()).empty());
    EXPECT_THROW(read_comm_domain_uses(nullptr, 0), std::length_error);
    EXPECT_THROW(read_comm_domain_uses(bytes.data(), bytes.size() - 1), std::length_error);
    std::array<uint8_t, sizeof(CommDomainUsesHeader)> unversioned{};
    EXPECT_THROW(read_comm_domain_uses(unversioned.data(), unversioned.size()), std::invalid_argument);
}

TEST(CommDomainUseWire, UnalignedRoundTripPreservesNamespacesAndFullWidthIdentities) {
    const auto uses = examples();
    const size_t length = comm_domain_uses_wire_size(uses);
    std::vector<uint8_t> bytes(length + 2, 0xa5);
    write_comm_domain_uses(bytes.data() + 1, length, uses);
    EXPECT_EQ(bytes.front(), 0xa5);
    EXPECT_EQ(bytes.back(), 0xa5);
    const auto decoded = read_comm_domain_uses(bytes.data() + 1, length);
    ASSERT_EQ(decoded.size(), uses.size());
    for (size_t i = 0; i < uses.size(); ++i) {
        EXPECT_EQ(decoded[i].kind, uses[i].kind);
        EXPECT_EQ(decoded[i].id, uses[i].id);
        EXPECT_EQ(decoded[i].device_context, uses[i].device_context);
        EXPECT_EQ(decoded[i].reserved, 0u);
    }
}

TEST(CommDomainUseWire, InvalidFormatAndCountAreRejectedBeforeReadingRows) {
    const auto uses = examples();
    std::vector<uint8_t> bytes(comm_domain_uses_wire_size(uses));
    write_comm_domain_uses(bytes.data(), bytes.size(), uses);
    const auto original = header_of(bytes);
    for (uint32_t field = 0; field < 3; ++field) {
        auto header = original;
        if (field == 0) header.magic ^= 1u;
        if (field == 1) ++header.version;
        if (field == 2) header.row_bytes = UINT32_MAX;
        replace_header(bytes, header);
        EXPECT_THROW(read_comm_domain_uses(bytes.data(), bytes.size()), std::invalid_argument);
    }
    auto oversized = original;
    oversized.count = UINT32_MAX;
    replace_header(bytes, oversized);
    EXPECT_THROW(read_comm_domain_uses(bytes.data(), bytes.size()), std::length_error);
    replace_header(bytes, original);
    for (size_t length = 0; length < bytes.size(); ++length)
        EXPECT_THROW(read_comm_domain_uses(bytes.data(), length), std::length_error) << length;
}

TEST(CommDomainUseWire, InvalidAndDuplicateRowsCannotBePublishedOrDecoded) {
    const auto valid = examples();
    for (uint32_t kind = 0; kind < 6; ++kind) {
        auto invalid = valid;
        if (kind == 0) invalid[0].kind = static_cast<CommDomainKind>(0);
        if (kind == 1) invalid[0].reserved = 1;
        if (kind == 2) invalid[0].id = 0;
        if (kind == 3) invalid[0].device_context = 0;
        if (kind == 4) {
            invalid[1].kind = invalid[0].kind;
            invalid[1].id = invalid[0].id;
        }
        if (kind == 5) invalid[1].device_context = invalid[0].device_context;
        std::vector<uint8_t> bytes(comm_domain_uses_wire_size(invalid), 0x5a);
        const auto original = bytes;
        EXPECT_THROW(write_comm_domain_uses(bytes.data(), bytes.size(), invalid), std::invalid_argument);
        EXPECT_EQ(bytes, original);
        const CommDomainUsesHeader header{COMM_DOMAIN_USES_MAGIC, COMM_DOMAIN_USES_VERSION, 2, sizeof(CommDomainUse)};
        replace_header(bytes, header);
        std::memcpy(bytes.data() + sizeof(header), invalid.data(), invalid.size() * sizeof(CommDomainUse));
        EXPECT_THROW(read_comm_domain_uses(bytes.data(), bytes.size()), std::invalid_argument);
    }
}

TEST(CommDomainUseWire, ShortDestinationCannotPartiallyOverwriteTheTaskTrailer) {
    const auto uses = examples();
    std::vector<uint8_t> bytes(comm_domain_uses_wire_size(uses), 0x71);
    const auto original = bytes;
    EXPECT_THROW(write_comm_domain_uses(bytes.data(), bytes.size() - 1, uses), std::length_error);
    EXPECT_EQ(bytes, original);
    EXPECT_THROW(write_comm_domain_uses(nullptr, bytes.size(), uses), std::length_error);
}

TEST(CommDomainUseWire, DecoderConsumesDeclaredRowsWithinTheAvailableFrame) {
    const auto uses = examples();
    const size_t length = comm_domain_uses_wire_size(uses);
    std::vector<uint8_t> frame(length + 32, 0xcd);
    write_comm_domain_uses(frame.data(), frame.size(), uses);
    const auto decoded = read_comm_domain_uses(frame.data(), frame.size());
    EXPECT_EQ(decoded.size(), uses.size());
    for (size_t i = length; i < frame.size(); ++i)
        EXPECT_EQ(frame[i], 0xcd);
}
}  // namespace
