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
#include <limits>
#include <vector>

#include "kernel_invocation_validation.h"

namespace {
using namespace simpler::kernel;

TEST(KernelInvocationValidation, EffectiveScalarCompatibilityAndFailureTransaction) {
    const ArgDirection signature[] = {ArgDirection::IN, ArgDirection::OUT, ArgDirection::SCALAR};
    int32_t tensors = -1, scalars = -1;
    EXPECT_EQ(derive_invocation_counts(signature, 3, 0, &tensors, &scalars), InvocationStatus::Ok);
    EXPECT_EQ(tensors, 2);
    EXPECT_EQ(scalars, 1);
    EXPECT_EQ(derive_invocation_counts(signature, 3, 1, &tensors, &scalars), InvocationStatus::Ok);
    EXPECT_EQ(derive_invocation_counts(signature, 3, 2, &tensors, &scalars), InvocationStatus::InvalidSignature);
    EXPECT_EQ(tensors, 2);
    EXPECT_EQ(scalars, 1);
    const ArgDirection interleaved[] = {ArgDirection::SCALAR, ArgDirection::IN};
    EXPECT_EQ(derive_invocation_counts(interleaved, 2, 0, &tensors, &scalars), InvocationStatus::InvalidSignature);
    const ArgDirection unknown[] = {static_cast<ArgDirection>(42)};
    EXPECT_EQ(derive_invocation_counts(unknown, 1, 0, &tensors, &scalars), InvocationStatus::InvalidSignature);
    EXPECT_EQ(derive_invocation_counts(nullptr, 0, 0, &tensors, &scalars), InvocationStatus::Ok);
    EXPECT_EQ(tensors, 0);
    EXPECT_EQ(scalars, 0);
    EXPECT_EQ(derive_invocation_counts(signature, -1, 0, &tensors, &scalars), InvocationStatus::InvalidCounts);
    EXPECT_EQ(derive_invocation_counts(signature, 257, 0, &tensors, &scalars), InvocationStatus::InvalidCounts);
    EXPECT_EQ(derive_invocation_counts(signature, 3, -1, &tensors, &scalars), InvocationStatus::InvalidCounts);
    EXPECT_EQ(derive_invocation_counts(nullptr, 1, 0, &tensors, &scalars), InvocationStatus::InvalidArgument);
}

TEST(KernelInvocationValidation, FramingRejectsBeforePayloadReadsAndPreservesOutput) {
    const PreparedInvocationView prepared{2, 1, 1, 7};
    SimplerKernelInvocationHeader header{};
    header.mode = SIMPLER_MODE_KERNEL;
    header.callable_id = prepared.callable_id;
    header.generation = prepared.slot_generation;
    header.tensor_count = 1;
    header.scalar_count = 1;
    std::vector<uint8_t> bytes(sizeof(header) + 1);
    auto check = [&](InvocationStatus expected, size_t size = sizeof(SimplerKernelInvocationHeader)) {
        std::memcpy(bytes.data() + 1, &header, sizeof(header));
        SimplerKernelInvocationHeader out{};
        out.callable_id = 31;
        EXPECT_EQ(validate_invocation_header({bytes.data() + 1, size}, prepared, &out), expected);
        if (expected != InvocationStatus::Ok) EXPECT_EQ(out.callable_id, 31);
    };
    check(InvocationStatus::Ok);
    check(InvocationStatus::InvalidSize, sizeof(header) - 1);
    header.payload_bytes = std::numeric_limits<uint64_t>::max();
    check(InvocationStatus::InvalidSize);
    header.payload_bytes = 0;
    header.mode = SIMPLER_MODE_PROGRAM;
    check(InvocationStatus::InvalidHeader);
    header.mode = 99;
    check(InvocationStatus::InvalidHeader);
    header.mode = SIMPLER_MODE_KERNEL;
    header.tensor_count = -1;
    check(InvocationStatus::InvalidCounts);
    header.tensor_count = 256;
    check(InvocationStatus::InvalidCounts);
    header.tensor_count = 1;
    header.scalar_count = -1;
    check(InvocationStatus::InvalidCounts);
    header.scalar_count = 129;
    check(InvocationStatus::InvalidCounts);
    header.scalar_count = 0;
    check(InvocationStatus::InvalidCounts);
    header.scalar_count = 1;
    header.host_copy_tensor_count = 1;
    check(InvocationStatus::InvalidCounts);
    header.host_copy_tensor_count = -1;
    check(InvocationStatus::InvalidCounts);
    header.host_copy_tensor_count = 0;
    header.generation = 8;
    check(InvocationStatus::StaleCallable);
    header.generation = 0;
    check(InvocationStatus::InvalidHeader);
    header.generation = 7;
    header.callable_id = 64;
    check(InvocationStatus::InvalidHeader);
    header.callable_id = 3;
    check(InvocationStatus::StaleCallable);
}
TEST(KernelInvocationValidation, RuntimePayloadIsOpaqueAndTrustedCountsAreRequired) {
    const PreparedInvocationView prepared{5, 2, 1, 9};
    SimplerKernelInvocationHeader header{};
    header.mode = SIMPLER_MODE_KERNEL;
    header.callable_id = prepared.callable_id;
    header.generation = prepared.slot_generation;
    header.tensor_count = prepared.tensor_count;
    header.scalar_count = prepared.scalar_count;
    header.payload_bytes = 13;
    std::vector<uint8_t> packet(sizeof(header) + header.payload_bytes, 0xa5);
    std::memcpy(packet.data(), &header, sizeof(header));
    SimplerKernelInvocationHeader out{};
    ASSERT_EQ(validate_invocation_header({packet.data(), packet.size()}, prepared, &out), InvocationStatus::Ok);
    EXPECT_EQ(out.payload_bytes, 13u);

    auto different = prepared;
    different.tensor_count = 1;
    EXPECT_EQ(
        validate_invocation_header({packet.data(), packet.size()}, different, &out), InvocationStatus::InvalidCounts
    );
    EXPECT_EQ(out.tensor_count, prepared.tensor_count);
    different = prepared;
    different.slot_generation = 0;
    EXPECT_EQ(
        validate_invocation_header({packet.data(), packet.size()}, different, &out), InvocationStatus::InvalidArgument
    );
    EXPECT_EQ(out.generation, prepared.slot_generation);
}
}  // namespace
