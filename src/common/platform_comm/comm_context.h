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
/**
 * CommContext is simpler's host/device communication ABI. Host allocation
 * publishes rank metadata, mapped windows and independent backend descriptors.
 * Kernels resolve addresses and pass the selected workspace explicitly to PTO
 * session builders. PTO instructions do not consume this structure directly.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

static constexpr uint32_t COMM_MAX_RANK_NUM = 64;

enum class CommWorkspaceKind : uint32_t { SDMA = 0, URMA = 1, RDMA = 2, COUNT = 3 };
static constexpr size_t COMM_WORKSPACE_COUNT = static_cast<size_t>(CommWorkspaceKind::COUNT);

constexpr uint32_t comm_workspace_bit(CommWorkspaceKind kind) {
    const auto index = static_cast<uint32_t>(kind);
    return index < COMM_WORKSPACE_COUNT ? uint32_t{1} << index : 0;
}

struct CommWorkspace {
    uint64_t addr;
    uint64_t size;
};

struct CommContext {
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[COMM_MAX_RANK_NUM];
    uint64_t windowsOut[COMM_MAX_RANK_NUM];
    CommWorkspace workspaces[COMM_WORKSPACE_COUNT];
};

#if defined(__CCE_AICORE__)
#define COMM_CONTEXT_DEVICE [aicore]
#define COMM_CONTEXT_GM __gm__
#else
#define COMM_CONTEXT_DEVICE
#define COMM_CONTEXT_GM
#endif

template <CommWorkspaceKind Kind>
COMM_CONTEXT_DEVICE inline COMM_CONTEXT_GM CommWorkspace &comm_workspace(COMM_CONTEXT_GM CommContext &ctx) {
    static_assert(static_cast<size_t>(Kind) < COMM_WORKSPACE_COUNT, "invalid communication workspace kind");
    return ctx.workspaces[static_cast<size_t>(Kind)];
}

template <CommWorkspaceKind Kind>
COMM_CONTEXT_DEVICE inline const COMM_CONTEXT_GM CommWorkspace &comm_workspace(const COMM_CONTEXT_GM CommContext &ctx) {
    static_assert(static_cast<size_t>(Kind) < COMM_WORKSPACE_COUNT, "invalid communication workspace kind");
    return ctx.workspaces[static_cast<size_t>(Kind)];
}

#undef COMM_CONTEXT_GM
#undef COMM_CONTEXT_DEVICE

inline bool set_comm_workspace(CommContext &ctx, CommWorkspaceKind kind, uint64_t addr, uint64_t size) {
    const auto index = static_cast<size_t>(kind);
    if (index >= COMM_WORKSPACE_COUNT) return false;
    ctx.workspaces[index] = {addr, size};
    return true;
}

static_assert(std::is_trivially_copyable_v<CommWorkspace> && std::is_standard_layout_v<CommWorkspace>);
static_assert(std::is_trivially_copyable_v<CommContext> && std::is_standard_layout_v<CommContext>);
static_assert(sizeof(CommWorkspace) == 16, "CommWorkspace size shifted");
static_assert(sizeof(CommContext) == 1088, "CommContext size shifted");
static_assert(offsetof(CommContext, rankId) == 0, "CommContext layout drift");
static_assert(offsetof(CommContext, rankNum) == 4, "CommContext layout drift");
static_assert(offsetof(CommContext, winSize) == 8, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsIn) == 16, "CommContext layout drift");
static_assert(offsetof(CommContext, windowsOut) == 528, "CommContext layout drift");
static_assert(offsetof(CommContext, workspaces) == 1040, "CommContext layout drift");
