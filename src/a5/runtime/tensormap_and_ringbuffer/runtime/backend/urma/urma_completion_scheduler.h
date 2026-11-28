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

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "completion_token.h"
#include "runtime_status.h"
#include "../../../../urma/urma_cq.h"
#include <pto/comm/async/urma/urma_types.hpp>

namespace pto2::urma_backend {

using pto::comm::urma::UrmaCqCtx;
using pto::comm::urma::UrmaInfo;
using pto::comm::urma::UrmaLayout;
using pto::comm::urma::UrmaWQCtx;

struct CpuCqAccess {
    static uint32_t load(uint64_t address) {
        auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(address));
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    static void store(uint64_t address, uint32_t value) {
        auto *ptr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(address));
        __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#if defined(__aarch64__)
        __asm__ __volatile__("dsb sy" ::: "memory");
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }
};

// Only the AICPU consumer publishes tails; accepted AIV posts never enter ISA's CQ drain.
inline CompletionPollResult
poll_urma_event_handle(uint64_t event_handle, uint64_t workspace_addr, uint32_t target_cqe) {
    const CompletionPollResult invalid{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (workspace_addr == 0) return invalid;

    uint32_t queue_row = 0, target_bb = 0;
    queue_row = static_cast<uint32_t>(event_handle >> 32);
    target_bb = static_cast<uint32_t>(event_handle);
    auto *info = reinterpret_cast<volatile UrmaInfo *>(static_cast<uintptr_t>(workspace_addr));
    const bool per_peer = info->layout == UrmaLayout::PER_PEER;
    if ((!per_peer && info->layout != UrmaLayout::SHARED_POOL) || info->rankCount == 0 || info->rowsPerPeer == 0 ||
        info->jettyCount == 0 || queue_row >= info->jettyCount || info->jettiesPerCore == 0 ||
        info->jettiesPerCore > pto::comm::urma::kUrmaMaxJettiesPerCore ||
        (per_peer && static_cast<uint64_t>(info->rankCount) * info->rowsPerPeer != info->jettyCount) ||
        info->sqPtr == 0 || info->scqPtr == 0) {
        return invalid;
    }
    auto *cq = reinterpret_cast<volatile UrmaCqCtx *>(static_cast<uintptr_t>(info->scqPtr)) + queue_row;
    auto *wq = reinterpret_cast<volatile UrmaWQCtx *>(static_cast<uintptr_t>(info->sqPtr)) + queue_row;
    const auto result =
        simpler::urma::poll_completion_row<CpuCqAccess, CompletionPollState>(wq, cq, target_bb, target_cqe);
    return {
        result, result == CompletionPollState::FAILED ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID : SIMPLER_ERROR_NONE
    };
}

inline void retire_urma_event_handle(uint64_t /*event_handle*/, uint64_t /*workspace_addr*/) {}

}  // namespace pto2::urma_backend
