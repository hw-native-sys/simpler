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
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include "common/worker_chip_orch_comm.h"

namespace {

WorkerChipOrchRegionDesc valid_desc() {
    return WorkerChipOrchRegionDesc{
        WORKER_CHIP_ORCH_COMM_MAGIC_VERSION, 7, 0x1000, 4096, 0x3000, 128,
    };
}

TEST(WorkerChipOrchCommTest, MagicVersionConstantMatchesCompatibilityWrapper) {
    EXPECT_EQ(
        WORKER_CHIP_ORCH_COMM_MAGIC_VERSION,
        worker_chip_orch_comm_pack_magic_version(
            WORKER_CHIP_ORCH_COMM_MAGIC, WORKER_CHIP_ORCH_COMM_ABI_MAJOR, WORKER_CHIP_ORCH_COMM_ABI_MINOR
        )
    );
    EXPECT_EQ(worker_chip_orch_comm_magic_version(), WORKER_CHIP_ORCH_COMM_MAGIC_VERSION);
    EXPECT_EQ(worker_chip_orch_comm::magic_version(), WORKER_CHIP_ORCH_COMM_MAGIC_VERSION);
}

TEST(WorkerChipOrchCommTest, DescriptorRoundTripsThroughSixTaskArgScalars) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    std::array<uint64_t, WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT> scalars{};

    EXPECT_TRUE(worker_chip_orch_comm_encode_desc(desc, scalars.data(), scalars.size()));
    EXPECT_EQ(scalars[0], desc.magic_version);
    EXPECT_EQ(scalars[1], desc.region_id);
    EXPECT_EQ(scalars[2], desc.payload_base);
    EXPECT_EQ(scalars[3], desc.payload_bytes);
    EXPECT_EQ(scalars[4], desc.counter_base);
    EXPECT_EQ(scalars[5], desc.counter_bytes);

    WorkerChipOrchRegionDesc decoded{};
    WorkerChipOrchCommValidationError error{};
    EXPECT_TRUE(worker_chip_orch_comm_decode_desc(scalars.data(), scalars.size(), &decoded, &error));
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::OK);
    EXPECT_EQ(decoded.magic_version, desc.magic_version);
    EXPECT_EQ(decoded.region_id, desc.region_id);
    EXPECT_EQ(decoded.payload_base, desc.payload_base);
    EXPECT_EQ(decoded.payload_bytes, desc.payload_bytes);
    EXPECT_EQ(decoded.counter_base, desc.counter_base);
    EXPECT_EQ(decoded.counter_bytes, desc.counter_bytes);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsBadMajorVersion) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.magic_version =
        worker_chip_orch_comm_pack_magic_version(WORKER_CHIP_ORCH_COMM_MAGIC, WORKER_CHIP_ORCH_COMM_ABI_MAJOR + 1, 0);

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_MAGIC_VERSION);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsAbiMajor2) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.magic_version = worker_chip_orch_comm_pack_magic_version(WORKER_CHIP_ORCH_COMM_MAGIC, 2, 0);

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_MAGIC_VERSION);
}

TEST(WorkerChipOrchCommTest, DescriptorKeepsRegionIdZeroAsInvalidSentinel) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.region_id = 0;

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_REGION_ID);
}

TEST(WorkerChipOrchCommTest, DescriptorAcceptsZeroBaseAddressesWhenRangesAreValid) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.payload_base = 0;
    desc.payload_bytes = 128;
    desc.counter_base = 256;
    desc.counter_bytes = 64;
    EXPECT_EQ(worker_chip_orch_comm_validate_desc(desc), WorkerChipOrchCommValidationError::OK);

    desc = valid_desc();
    desc.payload_base = 128;
    desc.payload_bytes = 128;
    desc.counter_base = 0;
    desc.counter_bytes = 64;
    EXPECT_EQ(worker_chip_orch_comm_validate_desc(desc), WorkerChipOrchCommValidationError::OK);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsZeroPayloadBytes) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.payload_bytes = 0;

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsPayloadCounterOverlapWithZeroBase) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.payload_base = 0;
    desc.payload_bytes = 128;
    desc.counter_base = 64;
    desc.counter_bytes = 64;

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsOverflowingPayloadRange) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.payload_base = UINT64_MAX - 7;
    desc.payload_bytes = 16;

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_PAYLOAD_RANGE);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsUnalignedCounterBase) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.counter_base = 0x3041;

    WorkerChipOrchCommValidationError error = worker_chip_orch_comm_validate_desc(desc);
    EXPECT_EQ(error, WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(WorkerChipOrchCommTest, DescriptorRejectsInvalidCounterBytes) {
    WorkerChipOrchRegionDesc desc = valid_desc();
    desc.counter_bytes = 0;
    EXPECT_EQ(worker_chip_orch_comm_validate_desc(desc), WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE);

    desc = valid_desc();
    desc.counter_bytes = 6;
    EXPECT_EQ(worker_chip_orch_comm_validate_desc(desc), WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE);
}

TEST(WorkerChipOrchCommTest, CounterAddressValidationRejectsUnalignedAndOutOfRange) {
    const WorkerChipOrchRegionDesc desc = valid_desc();

    EXPECT_EQ(
        worker_chip_orch_comm_validate_counter_addr(desc, desc.counter_base), WorkerChipOrchCommValidationError::OK
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_counter_addr(desc, desc.counter_base + desc.counter_bytes - sizeof(int32_t)),
        WorkerChipOrchCommValidationError::OK
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_counter_addr(desc, desc.counter_base + 2),
        WorkerChipOrchCommValidationError::BAD_COUNTER_RANGE
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_counter_addr(desc, desc.counter_base - sizeof(int32_t)),
        WorkerChipOrchCommValidationError::OUT_OF_BOUNDS
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_counter_addr(desc, desc.counter_base + desc.counter_bytes),
        WorkerChipOrchCommValidationError::OUT_OF_BOUNDS
    );
}

TEST(WorkerChipOrchCommTest, PayloadBoundsRejectOverflowAndOutOfRange) {
    EXPECT_EQ(worker_chip_orch_comm_validate_payload_bounds(16, 8, 32), WorkerChipOrchCommValidationError::OK);
    EXPECT_EQ(
        worker_chip_orch_comm_validate_payload_bounds(UINT64_MAX - 3, 8, UINT64_MAX),
        WorkerChipOrchCommValidationError::OUT_OF_BOUNDS
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_payload_bounds(24, 16, 32), WorkerChipOrchCommValidationError::OUT_OF_BOUNDS
    );
    EXPECT_EQ(
        worker_chip_orch_comm_validate_payload_bounds(0, 0, 32), WorkerChipOrchCommValidationError::BAD_PAYLOAD_RANGE
    );
}

TEST(WorkerChipOrchCommTest, AddSatAndOverflowHelpersHandleUint64Edges) {
    EXPECT_FALSE(worker_chip_orch_comm_add_overflows(7, 9));
    EXPECT_TRUE(worker_chip_orch_comm_add_overflows(UINT64_MAX - 1, 2));
    EXPECT_EQ(worker_chip_orch_comm_add_sat(7, 9), 16u);
    EXPECT_EQ(worker_chip_orch_comm_add_sat(UINT64_MAX - 1, 2), UINT64_MAX);
}

TEST(WorkerChipOrchCommTest, CompileTimeAlignmentRequiresPowerOfTwoAbiAlignment) {
    static_assert(
        worker_chip_orch_comm_is_aligned<WORKER_CHIP_ORCH_COMM_COUNTER_BYTES>(16), "counter alignment must work"
    );
    EXPECT_TRUE(worker_chip_orch_comm_is_aligned<WORKER_CHIP_ORCH_COMM_COUNTER_BASE_ALIGNMENT>(0x3000));
    EXPECT_FALSE(worker_chip_orch_comm_is_aligned<WORKER_CHIP_ORCH_COMM_COUNTER_BASE_ALIGNMENT>(0x3041));
    EXPECT_TRUE(worker_chip_orch_comm_is_aligned_runtime(24, 8));
    EXPECT_FALSE(worker_chip_orch_comm_is_aligned_runtime(24, 3));
}

TEST(WorkerChipOrchCommTest, NeutralHelpersMatchCompatibilityDelegation) {
    EXPECT_EQ(worker_chip_orch_comm_add_overflows(7, 9), region_add_overflows(7, 9));
    EXPECT_EQ(worker_chip_orch_comm_add_overflows(UINT64_MAX - 1, 2), region_add_overflows(UINT64_MAX - 1, 2));
    EXPECT_EQ(
        worker_chip_orch_comm_ranges_overlap(0, 64, 32, 64),
        region_spans_overlap(RegionPartLocalSpan{0, 64}, RegionPartLocalSpan{32, 64})
    );
    EXPECT_EQ(
        worker_chip_orch_comm_ranges_overlap(0, 64, 256, 64),
        region_spans_overlap(RegionPartLocalSpan{0, 64}, RegionPartLocalSpan{256, 64})
    );

    RegionNotifyOp notify{};
    ASSERT_TRUE(worker_chip_orch_comm_as_region_notify_op(WorkerChipOrchNotifyOp::Set, notify));
    EXPECT_TRUE(region_valid_notify_op(notify));
    EXPECT_TRUE(worker_chip_orch_comm_valid_notify_op(WorkerChipOrchNotifyOp::Set));
    EXPECT_FALSE(worker_chip_orch_comm_as_region_notify_op(static_cast<WorkerChipOrchNotifyOp>(2), notify));

    RegionWaitCmp cmp{};
    ASSERT_TRUE(worker_chip_orch_comm_as_region_wait_cmp(WorkerChipOrchWaitCmp::GE, cmp));
    EXPECT_EQ(
        worker_chip_orch_comm_compare_counter(5, 5, WorkerChipOrchWaitCmp::GE), region_compare_counter(5, 5, cmp)
    );
    EXPECT_FALSE(worker_chip_orch_comm_as_region_wait_cmp(static_cast<WorkerChipOrchWaitCmp>(6), cmp));
}

TEST(WorkerChipOrchCommTest, NotifyOpAndWaitCmpValidationRejectUnknownValues) {
    EXPECT_TRUE(worker_chip_orch_comm_valid_notify_op(WorkerChipOrchNotifyOp::Set));
    EXPECT_TRUE(worker_chip_orch_comm_valid_notify_op(WorkerChipOrchNotifyOp::Add));
    EXPECT_FALSE(worker_chip_orch_comm_valid_notify_op(static_cast<WorkerChipOrchNotifyOp>(2)));

    EXPECT_TRUE(worker_chip_orch_comm_valid_wait_cmp(WorkerChipOrchWaitCmp::EQ));
    EXPECT_TRUE(worker_chip_orch_comm_valid_wait_cmp(WorkerChipOrchWaitCmp::LE));
    EXPECT_FALSE(worker_chip_orch_comm_valid_wait_cmp(static_cast<WorkerChipOrchWaitCmp>(6)));
}

TEST(WorkerChipOrchCommTest, WaitCmpComparisonCoversAllPredicates) {
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(5, 5, WorkerChipOrchWaitCmp::EQ));
    EXPECT_FALSE(worker_chip_orch_comm_compare_counter(4, 5, WorkerChipOrchWaitCmp::EQ));
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(4, 5, WorkerChipOrchWaitCmp::NE));
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(6, 5, WorkerChipOrchWaitCmp::GT));
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(5, 5, WorkerChipOrchWaitCmp::GE));
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(4, 5, WorkerChipOrchWaitCmp::LT));
    EXPECT_TRUE(worker_chip_orch_comm_compare_counter(5, 5, WorkerChipOrchWaitCmp::LE));
    EXPECT_FALSE(worker_chip_orch_comm_compare_counter(5, 5, static_cast<WorkerChipOrchWaitCmp>(6)));
}

TEST(WorkerChipOrchCommTest, DescriptorAndSignalResultAreFixedSizePodTypes) {
    static_assert(std::is_standard_layout<WorkerChipOrchRegionDesc>::value, "descriptor must be POD-like");
    static_assert(std::is_trivially_copyable<WorkerChipOrchRegionDesc>::value, "descriptor must be fixed-size");
    static_assert(std::is_standard_layout<WorkerChipOrchSignalTestResult>::value, "test result must be POD-like");
    static_assert(std::is_trivially_copyable<WorkerChipOrchSignalTestResult>::value, "test result must be fixed-size");
}

}  // namespace
