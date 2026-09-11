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

TEST(KernelInvocationHeaderWire, LayoutIsPinned) {
    EXPECT_EQ(sizeof(SimplerKernelInvocationHeader), 64u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, abi_version), 0u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, header_bytes), 4u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, mode), 8u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, callable_id), 12u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, generation), 16u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, tensor_count), 24u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, scalar_count), 28u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, host_copy_tensor_count), 32u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, reserved0), 36u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, payload_bytes), 40u);
    EXPECT_EQ(offsetof(SimplerKernelInvocationHeader, reserved), 48u);
    EXPECT_EQ(SIMPLER_KERNEL_INVOCATION_ABI_VERSION, 1);
}

TEST(KernelInvocationHeaderWire, SurvivesMemcpyRoundtrip) {
    SimplerKernelInvocationHeader header{};
    header.abi_version = SIMPLER_KERNEL_INVOCATION_ABI_VERSION;
    header.header_bytes = sizeof(SimplerKernelInvocationHeader);
    header.mode = 1;
    header.callable_id = 17;
    header.generation = 0x1122334455667788ull;
    header.tensor_count = 12;
    header.scalar_count = 5;
    header.payload_bytes = 4096;

    unsigned char wire[sizeof(SimplerKernelInvocationHeader)];
    std::memcpy(wire, &header, sizeof(header));
    SimplerKernelInvocationHeader restored{};
    std::memcpy(&restored, wire, sizeof(restored));

    EXPECT_EQ(restored.abi_version, header.abi_version);
    EXPECT_EQ(restored.header_bytes, sizeof(SimplerKernelInvocationHeader));
    EXPECT_EQ(restored.mode, 1u);
    EXPECT_EQ(restored.callable_id, 17);
    EXPECT_EQ(restored.generation, 0x1122334455667788ull);
    EXPECT_EQ(restored.tensor_count, 12);
    EXPECT_EQ(restored.scalar_count, 5);
    EXPECT_EQ(restored.host_copy_tensor_count, 0);
    EXPECT_EQ(restored.payload_bytes, 4096u);
}

TEST(KernelInvocationHeaderWire, ZeroInitializedBlobReadsAsEmpty) {
    unsigned char wire[sizeof(SimplerKernelInvocationHeader)] = {};
    SimplerKernelInvocationHeader header;
    std::memcpy(&header, wire, sizeof(header));
    EXPECT_EQ(header.abi_version, 0u);
    EXPECT_EQ(header.payload_bytes, 0u);
    EXPECT_EQ(header.reserved0, 0u);
    EXPECT_EQ(header.reserved[0], 0u);
    EXPECT_EQ(header.reserved[1], 0u);
}

}  // namespace
