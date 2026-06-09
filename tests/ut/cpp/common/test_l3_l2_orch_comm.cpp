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

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "common/l3_l2_orch_comm.h"

namespace {

L3L2OrchRegionDesc valid_desc() {
    return L3L2OrchRegionDesc{
        l3_l2_orch_comm_magic_version(), 7, 0x1000, 4096, 0x3000, 0x3040,
    };
}

TEST(L3L2OrchCommTest, DescriptorRoundTripsThroughSixTaskArgScalars) {
    L3L2OrchRegionDesc desc = valid_desc();
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    EXPECT_TRUE(l3_l2_orch_comm_encode_desc(desc, scalars.data(), scalars.size()));
    EXPECT_EQ(scalars[0], desc.magic_version);
    EXPECT_EQ(scalars[1], desc.region_id);
    EXPECT_EQ(scalars[2], desc.payload_base);
    EXPECT_EQ(scalars[3], desc.payload_bytes);
    EXPECT_EQ(scalars[4], desc.l3_to_l2_signal_base);
    EXPECT_EQ(scalars[5], desc.l2_to_l3_signal_base);

    L3L2OrchRegionDesc decoded{};
    L3L2OrchCommValidationError error{};
    EXPECT_TRUE(l3_l2_orch_comm_decode_desc(scalars.data(), scalars.size(), &decoded, &error));
    EXPECT_EQ(error, L3L2OrchCommValidationError::OK);
    EXPECT_EQ(decoded.magic_version, desc.magic_version);
    EXPECT_EQ(decoded.region_id, desc.region_id);
    EXPECT_EQ(decoded.payload_base, desc.payload_base);
    EXPECT_EQ(decoded.payload_bytes, desc.payload_bytes);
    EXPECT_EQ(decoded.l3_to_l2_signal_base, desc.l3_to_l2_signal_base);
    EXPECT_EQ(decoded.l2_to_l3_signal_base, desc.l2_to_l3_signal_base);
}

TEST(L3L2OrchCommTest, DescriptorRejectsBadMajorVersion) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.magic_version = l3_l2_orch_comm_pack_magic_version(L3L2_ORCH_COMM_MAGIC, L3L2_ORCH_COMM_ABI_MAJOR + 1, 0);

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_MAGIC_VERSION);
}

TEST(L3L2OrchCommTest, DescriptorRejectsZeroPayloadBytes) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_bytes = 0;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsOverflowingPayloadRange) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_base = UINT64_MAX - 7;
    desc.payload_bytes = 16;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsUnalignedSignalBase) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.l2_to_l3_signal_base = 0x3041;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_SIGNAL_BASE);
}

TEST(L3L2OrchCommTest, PayloadBoundsRejectOverflowAndOutOfRange) {
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(16, 8, 32), L3L2OrchCommValidationError::OK);
    EXPECT_EQ(
        l3_l2_orch_comm_validate_payload_bounds(UINT64_MAX - 3, 8, UINT64_MAX),
        L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE
    );
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(24, 16, 32), L3L2OrchCommValidationError::OUT_OF_BOUNDS);
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(0, 0, 32), L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, SignalSlotValidationIsDirectional) {
    EXPECT_TRUE(l3_l2_orch_comm_valid_signal_slot(L3L2OrchCommSignalSlot::L3_TO_L2));
    EXPECT_TRUE(l3_l2_orch_comm_valid_signal_slot(L3L2OrchCommSignalSlot::L2_TO_L3));
    EXPECT_FALSE(l3_l2_orch_comm_valid_signal_slot(static_cast<L3L2OrchCommSignalSlot>(2)));
}

TEST(L3L2OrchCommTest, RequestAndResponseAreFixedSizePodDescriptorsOnly) {
    static_assert(std::is_standard_layout<L3L2OrchRegionDesc>::value, "descriptor must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchRegionDesc>::value, "descriptor must be fixed-size");
    static_assert(std::is_standard_layout<L3L2OrchCommRequest>::value, "request must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchCommRequest>::value, "request must be fixed-size");
    static_assert(std::is_standard_layout<L3L2OrchCommResponse>::value, "response must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchCommResponse>::value, "response must be fixed-size");

    EXPECT_EQ(offsetof(L3L2OrchCommRequest, cmd), 0u);
    EXPECT_EQ(sizeof(L3L2OrchCommResponse::message), 256u);
    EXPECT_EQ(sizeof(L3L2OrchCommRequest), sizeof(uint32_t) * 2 + sizeof(uint64_t) * 7)
        << "request carries descriptors only; payload bytes must not be embedded";
}

}  // namespace
