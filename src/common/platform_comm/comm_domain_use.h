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

#pragma once

#include <cstdint>
#include <type_traits>

#include "comm_context.h"

enum class CommDomainKind : uint32_t { LOCAL = 1, GLOBAL = 2 };
enum class CommDomainExecution : uint32_t { UNKNOWN = 0, EXCLUSIVE_RUN = 1 };

// Host submission metadata. A device context is a device VA, never an owner
// pointer. The caller must select this identity from its actual retained domain.
struct CommDomainUse {
    CommDomainKind kind{};
    uint32_t reserved{0};
    uint64_t id{0};  // LOCAL allocations begin at zero; GLOBAL reserves zero.
    uint64_t device_context{0};
};

// Read-only copy from an existing backend allocation owner, not a new lifetime
// reference. The caller serializes query with domain release and retains that
// owner through its run fence. No cache, DMA, or completion-record access rights
// are conferred: a whole SDMA workspace is not a set of legal post-done records.
struct CommDomainSnapshot {
    CommDomainUse use{};
    CommContext context{};
};

inline bool comm_domain_snapshot_valid(const CommDomainSnapshot &snapshot) {
    const auto &use = snapshot.use;
    const auto &context = snapshot.context;
    const auto valid_span = [](uint64_t base, uint64_t bytes) {
        return base != 0 && bytes != 0 && bytes <= UINT64_MAX - base;
    };
    if ((use.kind != CommDomainKind::LOCAL && use.kind != CommDomainKind::GLOBAL) || use.reserved != 0 ||
        (use.kind == CommDomainKind::GLOBAL && use.id == 0) || !valid_span(use.device_context, sizeof(CommContext)) ||
        context.rankNum == 0 || context.rankNum > COMM_MAX_RANK_NUM || context.rankId >= context.rankNum ||
        !valid_span(context.windowsIn[context.rankId], context.winSize))
        return false;
    for (uint32_t i = 0; i < context.rankNum; ++i) {
        if ((context.windowsIn[i] != 0 && !valid_span(context.windowsIn[i], context.winSize)) ||
            (context.windowsOut[i] != 0 && !valid_span(context.windowsOut[i], context.winSize)))
            return false;
    }
    for (const auto &workspace : context.workspaces) {
        if ((workspace.addr != 0 || workspace.size != 0) && !valid_span(workspace.addr, workspace.size)) return false;
    }
    return true;
}

static_assert(std::is_trivially_copyable_v<CommDomainUse>);
static_assert(std::is_trivially_copyable_v<CommDomainSnapshot>);
