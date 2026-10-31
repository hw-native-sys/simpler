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

#include <cstddef>
#include <cstring>

#include "platform_comm/comm_context.h"

TEST(CommWorkspaces, BackendPublicationCannotOverwriteAnotherEngine) {
    CommContext ctx{};
    set_comm_workspace(ctx, CommWorkspaceKind::SDMA, 0x1000, 16384);
    set_comm_workspace(ctx, CommWorkspaceKind::URMA, 0x2000, 32768);
    set_comm_workspace(ctx, CommWorkspaceKind::RDMA, 0x3000, 65536);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::SDMA>(ctx).addr, 0x1000u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::SDMA>(ctx).size, 16384u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::URMA>(ctx).addr, 0x2000u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::URMA>(ctx).size, 32768u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::RDMA>(ctx).addr, 0x3000u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::RDMA>(ctx).size, 65536u);
}

TEST(CommWorkspaces, MissingBackendHasNoFallbackToAnotherWorkspace) {
    CommContext ctx{};
    set_comm_workspace(ctx, CommWorkspaceKind::URMA, 0x2000, 32768);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::SDMA>(ctx).addr, 0u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::RDMA>(ctx).addr, 0u);
}

TEST(CommWorkspaces, ContextCopyKeepsDomainWorkspaceReferencesIndependent) {
    CommContext first{};
    set_comm_workspace(first, CommWorkspaceKind::SDMA, 0x1000, 16384);
    set_comm_workspace(first, CommWorkspaceKind::URMA, 0x2000, 32768);
    CommContext second = first;
    set_comm_workspace(second, CommWorkspaceKind::URMA, 0x3000, 65536);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::URMA>(first).addr, 0x2000u);
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::URMA>(second).addr, 0x3000u);
    EXPECT_EQ(
        comm_workspace<CommWorkspaceKind::SDMA>(first).addr, comm_workspace<CommWorkspaceKind::SDMA>(second).addr
    );
}

TEST(CommWorkspaces, InvalidDynamicKindCannotWritePastArray) {
    CommContext ctx{};
    set_comm_workspace(ctx, CommWorkspaceKind::URMA, 0x2000, 32768);
    const CommContext before = ctx;
    for (auto kind : {CommWorkspaceKind::COUNT, static_cast<CommWorkspaceKind>(UINT32_MAX)}) {
        EXPECT_FALSE(set_comm_workspace(ctx, kind, 0xdeadbeef, 65536));
        EXPECT_EQ(std::memcmp(&ctx, &before, sizeof(ctx)), 0);
        EXPECT_EQ(comm_workspace_bit(kind), 0u);
    }
}

TEST(CommWorkspaces, EnumIndicesMatchSerializedDescriptorOrder) {
    CommContext ctx{};
    ctx.rankId = 3;
    ctx.windowsIn[3] = 0xabcdef;
    set_comm_workspace(ctx, CommWorkspaceKind::RDMA, 0x12345678, 65536);
    const CommContext &view = ctx;
    EXPECT_EQ(comm_workspace<CommWorkspaceKind::RDMA>(view).addr, 0x12345678u);
    EXPECT_EQ(COMM_WORKSPACE_COUNT, 3u);
    ASSERT_EQ(sizeof(ctx), 1088u);
    unsigned char image[sizeof(ctx)];
    std::memcpy(image, &ctx, sizeof(ctx));
    uint32_t rank = 0;
    std::memcpy(&rank, image, sizeof(rank));
    EXPECT_EQ(rank, 3u);
    uint64_t address = 0;
    std::memcpy(&address, image + 16 + 3 * sizeof(uint64_t), sizeof(address));
    EXPECT_EQ(address, ctx.windowsIn[3]);
    std::memcpy(&address, image + 1040 + 2 * sizeof(CommWorkspace), sizeof(address));
    EXPECT_EQ(address, 0x12345678u);
}
