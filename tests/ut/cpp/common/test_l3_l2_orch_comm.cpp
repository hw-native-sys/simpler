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
#include "host/l3_l2_orch_region_access.h"

namespace {

L3L2OrchRegionDesc valid_desc() {
    return L3L2OrchRegionDesc{
        L3L2_ORCH_COMM_MAGIC_VERSION, 7, 0x1000, 4096, 0x3000, 128,
    };
}

TEST(L3L2OrchCommTest, MagicVersionConstantMatchesCompatibilityWrapper) {
    EXPECT_EQ(
        L3L2_ORCH_COMM_MAGIC_VERSION,
        l3_l2_orch_comm_pack_magic_version(L3L2_ORCH_COMM_MAGIC, L3L2_ORCH_COMM_ABI_MAJOR, L3L2_ORCH_COMM_ABI_MINOR)
    );
    EXPECT_EQ(l3_l2_orch_comm_magic_version(), L3L2_ORCH_COMM_MAGIC_VERSION);
    EXPECT_EQ(l3_l2_orch_comm::magic_version(), L3L2_ORCH_COMM_MAGIC_VERSION);
}

TEST(L3L2OrchCommTest, DescriptorRoundTripsThroughSixTaskArgScalars) {
    L3L2OrchRegionDesc desc = valid_desc();
    std::array<uint64_t, L3L2_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    EXPECT_TRUE(l3_l2_orch_comm_encode_desc(desc, scalars.data(), scalars.size()));
    EXPECT_EQ(scalars[0], desc.magic_version);
    EXPECT_EQ(scalars[1], desc.region_id);
    EXPECT_EQ(scalars[2], desc.payload_base);
    EXPECT_EQ(scalars[3], desc.payload_bytes);
    EXPECT_EQ(scalars[4], desc.counter_base);
    EXPECT_EQ(scalars[5], desc.counter_bytes);

    L3L2OrchRegionDesc decoded{};
    L3L2OrchCommValidationError error{};
    EXPECT_TRUE(l3_l2_orch_comm_decode_desc(scalars.data(), scalars.size(), &decoded, &error));
    EXPECT_EQ(error, L3L2OrchCommValidationError::OK);
    EXPECT_EQ(decoded.magic_version, desc.magic_version);
    EXPECT_EQ(decoded.region_id, desc.region_id);
    EXPECT_EQ(decoded.payload_base, desc.payload_base);
    EXPECT_EQ(decoded.payload_bytes, desc.payload_bytes);
    EXPECT_EQ(decoded.counter_base, desc.counter_base);
    EXPECT_EQ(decoded.counter_bytes, desc.counter_bytes);
}

TEST(L3L2OrchCommTest, DescriptorRejectsBadMajorVersion) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.magic_version = l3_l2_orch_comm_pack_magic_version(L3L2_ORCH_COMM_MAGIC, L3L2_ORCH_COMM_ABI_MAJOR + 1, 0);

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_MAGIC_VERSION);
}

TEST(L3L2OrchCommTest, DescriptorKeepsRegionIdZeroAsInvalidSentinel) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.region_id = 0;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_REGION_ID);
}

TEST(L3L2OrchCommTest, DescriptorAcceptsZeroBaseAddressesWhenRangesAreValid) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_base = 0;
    desc.payload_bytes = 128;
    desc.counter_base = 256;
    desc.counter_bytes = 64;
    EXPECT_EQ(l3_l2_orch_comm_validate_desc(desc), L3L2OrchCommValidationError::OK);

    desc = valid_desc();
    desc.payload_base = 128;
    desc.payload_bytes = 128;
    desc.counter_base = 0;
    desc.counter_bytes = 64;
    EXPECT_EQ(l3_l2_orch_comm_validate_desc(desc), L3L2OrchCommValidationError::OK);
}

TEST(L3L2OrchCommTest, DescriptorRejectsZeroPayloadBytes) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_bytes = 0;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsPayloadCounterOverlapWithZeroBase) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_base = 0;
    desc.payload_bytes = 128;
    desc.counter_base = 64;
    desc.counter_bytes = 64;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsOverflowingPayloadRange) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.payload_base = UINT64_MAX - 7;
    desc.payload_bytes = 16;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsUnalignedCounterBase) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.counter_base = 0x3041;

    L3L2OrchCommValidationError error = l3_l2_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, L3L2OrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(L3L2OrchCommTest, DescriptorRejectsInvalidCounterBytes) {
    L3L2OrchRegionDesc desc = valid_desc();
    desc.counter_bytes = 0;
    EXPECT_EQ(l3_l2_orch_comm_validate_desc(desc), L3L2OrchCommValidationError::BAD_COUNTER_RANGE);

    desc = valid_desc();
    desc.counter_bytes = 6;
    EXPECT_EQ(l3_l2_orch_comm_validate_desc(desc), L3L2OrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(L3L2OrchCommTest, CounterAddressValidationRejectsUnalignedAndOutOfRange) {
    const L3L2OrchRegionDesc desc = valid_desc();

    EXPECT_EQ(l3_l2_orch_comm_validate_counter_addr(desc, desc.counter_base), L3L2OrchCommValidationError::OK);
    EXPECT_EQ(
        l3_l2_orch_comm_validate_counter_addr(desc, desc.counter_base + desc.counter_bytes - sizeof(int32_t)),
        L3L2OrchCommValidationError::OK
    );
    EXPECT_EQ(
        l3_l2_orch_comm_validate_counter_addr(desc, desc.counter_base + 2),
        L3L2OrchCommValidationError::BAD_COUNTER_RANGE
    );
    EXPECT_EQ(
        l3_l2_orch_comm_validate_counter_addr(desc, desc.counter_base - sizeof(int32_t)),
        L3L2OrchCommValidationError::OUT_OF_BOUNDS
    );
    EXPECT_EQ(
        l3_l2_orch_comm_validate_counter_addr(desc, desc.counter_base + desc.counter_bytes),
        L3L2OrchCommValidationError::OUT_OF_BOUNDS
    );
}

TEST(L3L2OrchCommTest, PayloadBoundsRejectOverflowAndOutOfRange) {
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(16, 8, 32), L3L2OrchCommValidationError::OK);
    EXPECT_EQ(
        l3_l2_orch_comm_validate_payload_bounds(UINT64_MAX - 3, 8, UINT64_MAX),
        L3L2OrchCommValidationError::OUT_OF_BOUNDS
    );
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(24, 16, 32), L3L2OrchCommValidationError::OUT_OF_BOUNDS);
    EXPECT_EQ(l3_l2_orch_comm_validate_payload_bounds(0, 0, 32), L3L2OrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(L3L2OrchCommTest, AddSatAndOverflowHelpersHandleUint64Edges) {
    EXPECT_FALSE(l3_l2_orch_comm_add_overflows(7, 9));
    EXPECT_TRUE(l3_l2_orch_comm_add_overflows(UINT64_MAX - 1, 2));
    EXPECT_EQ(l3_l2_orch_comm_add_sat(7, 9), 16u);
    EXPECT_EQ(l3_l2_orch_comm_add_sat(UINT64_MAX - 1, 2), UINT64_MAX);
}

TEST(L3L2OrchCommTest, CompileTimeAlignmentRequiresPowerOfTwoAbiAlignment) {
    static_assert(l3_l2_orch_comm_is_aligned<L3L2_ORCH_COMM_COUNTER_BYTES>(16), "counter alignment must work");
    EXPECT_TRUE(l3_l2_orch_comm_is_aligned<L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT>(0x3000));
    EXPECT_FALSE(l3_l2_orch_comm_is_aligned<L3L2_ORCH_COMM_COUNTER_BASE_ALIGNMENT>(0x3041));
    EXPECT_TRUE(l3_l2_orch_comm_is_aligned_runtime(24, 8));
    EXPECT_FALSE(l3_l2_orch_comm_is_aligned_runtime(24, 3));
}

TEST(L3L2OrchCommTest, NotifyOpAndWaitCmpValidationRejectUnknownValues) {
    EXPECT_TRUE(l3_l2_orch_comm_valid_notify_op(L3L2OrchNotifyOp::Set));
    EXPECT_TRUE(l3_l2_orch_comm_valid_notify_op(L3L2OrchNotifyOp::Add));
    EXPECT_FALSE(l3_l2_orch_comm_valid_notify_op(static_cast<L3L2OrchNotifyOp>(2)));

    EXPECT_TRUE(l3_l2_orch_comm_valid_wait_cmp(L3L2OrchWaitCmp::EQ));
    EXPECT_TRUE(l3_l2_orch_comm_valid_wait_cmp(L3L2OrchWaitCmp::LE));
    EXPECT_FALSE(l3_l2_orch_comm_valid_wait_cmp(static_cast<L3L2OrchWaitCmp>(6)));
}

TEST(L3L2OrchCommTest, WaitCmpComparisonCoversAllPredicates) {
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(5, 5, L3L2OrchWaitCmp::EQ));
    EXPECT_FALSE(l3_l2_orch_comm_compare_counter(4, 5, L3L2OrchWaitCmp::EQ));
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(4, 5, L3L2OrchWaitCmp::NE));
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(6, 5, L3L2OrchWaitCmp::GT));
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(5, 5, L3L2OrchWaitCmp::GE));
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(4, 5, L3L2OrchWaitCmp::LT));
    EXPECT_TRUE(l3_l2_orch_comm_compare_counter(5, 5, L3L2OrchWaitCmp::LE));
    EXPECT_FALSE(l3_l2_orch_comm_compare_counter(5, 5, static_cast<L3L2OrchWaitCmp>(6)));
}

TEST(L3L2OrchCommTest, RequestAndResponseAreFixedSizePodDescriptorsOnly) {
    static_assert(std::is_standard_layout<L3L2OrchRegionDesc>::value, "descriptor must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchRegionDesc>::value, "descriptor must be fixed-size");
    static_assert(std::is_standard_layout<L3L2OrchSignalTestResult>::value, "test result must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchSignalTestResult>::value, "test result must be fixed-size");
    static_assert(std::is_standard_layout<L3L2OrchCommRequest>::value, "request must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchCommRequest>::value, "request must be fixed-size");
    static_assert(std::is_standard_layout<L3L2OrchCommResponse>::value, "response must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2OrchCommResponse>::value, "response must be fixed-size");

    EXPECT_EQ(offsetof(L3L2OrchCommRequest, cmd), 0u);
    EXPECT_EQ(offsetof(L3L2OrchCommRequest, op), sizeof(uint32_t));
    EXPECT_EQ(offsetof(L3L2OrchCommRequest, payload_offset), sizeof(uint32_t) * 2 + sizeof(uint64_t));
    EXPECT_EQ(offsetof(L3L2OrchCommRequest, counter_addr), sizeof(uint32_t) * 2 + sizeof(uint64_t) * 4);
    EXPECT_EQ(offsetof(L3L2OrchCommRequest, counter_operand), sizeof(uint32_t) * 2 + sizeof(uint64_t) * 6);
    EXPECT_EQ(offsetof(L3L2OrchCommResponse, observed_counter), sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint64_t));
    EXPECT_EQ(
        offsetof(L3L2OrchCommResponse, matched), sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(int32_t)
    );
    EXPECT_EQ(sizeof(L3L2OrchCommResponse::message), 256u);
    EXPECT_EQ(sizeof(L3L2OrchCommRequest), sizeof(uint32_t) * 4 + sizeof(uint64_t) * 7)
        << "request carries descriptors only; payload bytes must not be embedded";
}

TEST(L3L2OrchCommTest, LifecycleCreateWireStructsHaveFixedLayout) {
    static_assert(std::is_standard_layout<L3L2RegionCreateRequest>::value, "create request must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2RegionCreateRequest>::value, "create request must be fixed-size");
    static_assert(std::is_standard_layout<L3L2RegionCreateReply>::value, "create reply must be POD-like");
    static_assert(std::is_trivially_copyable<L3L2RegionCreateReply>::value, "create reply must be fixed-size");

    EXPECT_EQ(offsetof(L3L2RegionCreateRequest, magic_version), 0u);
    EXPECT_EQ(offsetof(L3L2RegionCreateRequest, payload_bytes), sizeof(uint64_t) * 2);
    EXPECT_EQ(sizeof(L3L2RegionCreateRequest), 40u);
    EXPECT_EQ(offsetof(L3L2RegionCreateReply, desc), 0u);
    EXPECT_EQ(offsetof(L3L2RegionCreateReply, access_profile), sizeof(uint64_t) * 6);
    EXPECT_EQ(offsetof(L3L2RegionCreateReply, device_id), sizeof(uint64_t) * 6 + sizeof(uint32_t) * 2);
    EXPECT_EQ(offsetof(L3L2RegionCreateReply, backing_shm), sizeof(uint64_t) * 6 + sizeof(uint32_t) * 2 + 4u + 65u);
    EXPECT_EQ(offsetof(L3L2RegionCreateReply, mapping_bytes), 160u);
    EXPECT_EQ(sizeof(L3L2RegionCreateReply), 168u);
}

}  // namespace
