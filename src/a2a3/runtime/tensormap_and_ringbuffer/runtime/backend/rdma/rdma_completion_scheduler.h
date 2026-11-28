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

#include "aicpu/platform_regs.h"
#include "completion_token.h"
#include "rdma/in_die_completion.h"
#include "runtime_status.h"

namespace simpler::rdma {

struct InDieDeviceReader {
    uint32_t read32(uint64_t address) {
        const auto line = static_cast<uintptr_t>(address) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1);
        cache_invalidate_range(reinterpret_cast<const void *>(line), CHIP_ALIGN_SIZE);
        return __atomic_load_n(reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(address)), __ATOMIC_ACQUIRE);
    }
    uint64_t read64(uint64_t address) {
        const auto line = static_cast<uintptr_t>(address) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1);
        cache_invalidate_range(reinterpret_cast<const void *>(line), CHIP_ALIGN_SIZE);
        return __atomic_load_n(reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(address)), __ATOMIC_ACQUIRE);
    }
};

inline CompletionPollResult
poll_rdma_in_die_event(uint64_t handle, uint64_t workspace, uint64_t owner_epoch, InDieCompletionCursor &cursor) {
    InDieDeviceReader reader;
    const auto result = poll_in_die_completion(handle, workspace, owner_epoch, cursor, reader);
    if (result.state == InDiePollState::Ready) return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
    if (result.state == InDiePollState::Pending) return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
}

inline void retire_rdma_in_die_event(InDieCompletionCursor &cursor) { cursor = {}; }

}  // namespace simpler::rdma
