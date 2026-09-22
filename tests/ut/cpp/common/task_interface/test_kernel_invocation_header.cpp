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

#include "kernel_invocation_header.h"

namespace {

TEST(KernelInvocationHeaderWire, ModeValuesArePinned) {
    EXPECT_EQ(SIMPLER_MODE_PROGRAM, 0);
    EXPECT_EQ(SIMPLER_MODE_KERNEL, 1);
}

TEST(KernelInvocationHeaderWire, MatchesWireLayoutAndSurvivesMemcpy) {
    SimplerKernelInvocationHeader header{};
    header.mode = SIMPLER_MODE_KERNEL;
    header.callable_id = 17;
    header.generation = 0x1122334455667788ull;
    header.payload_bytes = 4096;
    header.tensor_count = 12;
    header.scalar_count = 5;

    unsigned char wire[sizeof(SimplerKernelInvocationHeader)];
    std::memcpy(wire, &header, sizeof(header));
    SimplerKernelInvocationHeader restored{};
    std::memcpy(&restored, wire, sizeof(restored));

    EXPECT_EQ(restored.mode, static_cast<uint32_t>(SIMPLER_MODE_KERNEL));
    EXPECT_EQ(restored.callable_id, 17);
    EXPECT_EQ(restored.generation, 0x1122334455667788ull);
    EXPECT_EQ(restored.payload_bytes, 4096u);
    EXPECT_EQ(restored.tensor_count, 12);
    EXPECT_EQ(restored.scalar_count, 5);
    EXPECT_EQ(restored.host_copy_tensor_count, 0);
    EXPECT_EQ(restored.reserved_, 0u);

    const unsigned char expected[40] = {
        1, 0, 0, 0, 17, 0, 0, 0, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0, 0x10, 0, 0,
        0, 0, 0, 0, 12, 0, 0, 0, 5,    0,    0,    0,    0,    0,    0,    0,    0, 0,    0, 0,
    };
    EXPECT_EQ(std::memcmp(wire, expected, sizeof(expected)), 0);
}

TEST(KernelInvocationHeaderWire, ZeroInitializedBlobReadsAsEmpty) {
    unsigned char wire[sizeof(SimplerKernelInvocationHeader)] = {};
    SimplerKernelInvocationHeader header;
    std::memcpy(&header, wire, sizeof(header));
    EXPECT_EQ(header.mode, static_cast<uint32_t>(SIMPLER_MODE_PROGRAM));
    EXPECT_EQ(header.payload_bytes, 0u);
    EXPECT_EQ(header.tensor_count, 0);
    EXPECT_EQ(header.scalar_count, 0);
    EXPECT_EQ(header.host_copy_tensor_count, 0);
    EXPECT_EQ(header.reserved_, 0u);
}

}  // namespace
