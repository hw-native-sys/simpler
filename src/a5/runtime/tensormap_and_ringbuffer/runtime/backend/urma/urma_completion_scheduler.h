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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_

#include <cstdint>

#include "aicpu/platform_regs.h"
#include "aicore_completion_mailbox.h"
#include "completion_token.h"
#include "runtime_status.h"

namespace pto2::urma_backend {

inline constexpr uint32_t kUrmaCqeOwnerBit = 2U;
inline constexpr uint32_t kUrmaCqeSubstatusShift = 16U;
inline constexpr uint32_t kUrmaCqeStatusShift = 24U;
inline constexpr uint32_t kUrmaCqeStatusMask = 0xFFU;

inline uintptr_t cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1U);
}

inline CompletionPollResult poll_urma_cqe_record(uint64_t cqe_addr, uint32_t expected_owner) {
    if (cqe_addr == 0U || expected_owner > 1U) {
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    auto *dw0_addr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(cqe_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(cache_line(dw0_addr)), CHIP_ALIGN_SIZE);
    const uint32_t dw0 = __atomic_load_n(dw0_addr, __ATOMIC_ACQUIRE);
    if (((dw0 >> kUrmaCqeOwnerBit) & 1U) != expected_owner) {
        return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    }
    if (((dw0 >> kUrmaCqeStatusShift) & kUrmaCqeStatusMask) != 0U ||
        ((dw0 >> kUrmaCqeSubstatusShift) & kUrmaCqeStatusMask) != 0U) {
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }
    return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
}

inline void retire_urma_cqe_record(uint64_t /*cqe_addr*/, uint32_t /*expected_owner*/) {}

}  // namespace pto2::urma_backend

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_URMA_URMA_COMPLETION_SCHEDULER_H_
