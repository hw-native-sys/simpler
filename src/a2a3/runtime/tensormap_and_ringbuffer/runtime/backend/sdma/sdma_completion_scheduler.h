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
#include "aicore_completion_mailbox.h"
#include "pto_completion_token.h"
#include "pto_runtime_status.h"

inline uintptr_t sdma_completion_cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

// PTO-ISA stores the latest completed post ID as a monotonic uint64_t in one
// cache-line-sized record per queue. The completion type keeps its historical
// name, but the scheduler must compare the record with the post ID carried in
// backend_cookie and must not clear or otherwise retire the shared record.
inline CompletionPollResult poll_sdma_post_done_record(uint64_t record_addr, uint64_t expected_post_id) {
    if (record_addr == 0 || expected_post_id == 0) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }
    volatile uint64_t *record = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(record_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(sdma_completion_cache_line(record)), PTO2_ALIGN_SIZE);
    uint64_t completed_post_id = __atomic_load_n(record, __ATOMIC_ACQUIRE);
    return {
        completed_post_id >= expected_post_id ? CompletionPollState::READY : CompletionPollState::PENDING,
        PTO2_ERROR_NONE
    };
}

inline void retire_sdma_post_done_record(uint64_t /*record_addr*/, uint64_t /*expected_post_id*/) {}
