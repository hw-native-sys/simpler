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
#include <cstdint>

#include "intrinsic.h"

TEST(HbgDmaWorkspace, UnavailableEnginesReturnNull) {
    GlobalContext context{};
    std::array<int64_t, SPMD_GLOBAL_CONTEXT_INDEX + 1> args{};
    args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
    EXPECT_EQ(get_dma_workspace(args.data(), DMA_WORKSPACE_SDMA), nullptr);
    EXPECT_EQ(get_dma_workspace(args.data(), DMA_WORKSPACE_URMA), nullptr);
}

TEST(HbgDmaWorkspace, EachEngineReturnsItsInjectedAddress) {
    std::array<uint8_t, 64> sdma{};
    std::array<uint8_t, 64> urma{};
    GlobalContext context{};
    context.dma_workspace[DMA_WORKSPACE_SDMA] = reinterpret_cast<uint64_t>(sdma.data());
    context.dma_workspace[DMA_WORKSPACE_URMA] = reinterpret_cast<uint64_t>(urma.data());
    std::array<int64_t, SPMD_GLOBAL_CONTEXT_INDEX + 1> args{};
    args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
    EXPECT_EQ(get_dma_workspace(args.data(), DMA_WORKSPACE_SDMA), sdma.data());
    EXPECT_EQ(get_dma_workspace(args.data(), DMA_WORKSPACE_URMA), urma.data());
}
TEST(HbgDmaWorkspace, InvalidKindsDoNotReadTheContext) {
    EXPECT_EQ(get_dma_workspace(nullptr, -1), nullptr);
    EXPECT_EQ(get_dma_workspace(nullptr, DMA_WORKSPACE_KIND_COUNT), nullptr);
}

TEST(HbgDmaWorkspace, AnUnprovisionedEngineDoesNotAliasAnotherEngine) {
    std::array<uint8_t, 64> sdma{};
    GlobalContext context{};
    context.dma_workspace[DMA_WORKSPACE_SDMA] = reinterpret_cast<uint64_t>(sdma.data());
    std::array<int64_t, SPMD_GLOBAL_CONTEXT_INDEX + 1> args{};
    args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
    EXPECT_EQ(get_dma_workspace(args.data(), DMA_WORKSPACE_URMA), nullptr);
}
