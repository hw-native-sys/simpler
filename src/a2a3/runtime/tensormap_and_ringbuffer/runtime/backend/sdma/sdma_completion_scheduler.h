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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "pto_completion_ingress.h"
#include "pto_completion_token.h"
#include "pto_runtime_status.h"

// runtime-side mirror of the PTO-ISA SdmaEventRecord. SDMA backend is the only
// allowed holder of this ABI knowledge; the generic scheduler dispatches into
// the helpers below through the completion ops table.
struct SdmaEventRecord {
    uint32_t flag;
    uint32_t sq_tail;
    uint64_t channel_info;
};

static_assert(sizeof(SdmaEventRecord) == 16, "SDMA event record ABI drift");
static_assert(offsetof(SdmaEventRecord, sq_tail) == 4, "SDMA event record ABI drift");

inline uintptr_t sdma_completion_cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

inline CompletionPollResult poll_sdma_event_record(uint64_t record_addr) {
    if (record_addr == 0) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }
    volatile SdmaEventRecord *record =
        reinterpret_cast<volatile SdmaEventRecord *>(static_cast<uintptr_t>(record_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(sdma_completion_cache_line(record)), PTO2_ALIGN_SIZE);
    uint32_t flag = __atomic_load_n(&record->flag, __ATOMIC_ACQUIRE);
    return {flag != 0 ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE};
}

inline void retire_sdma_event_record(uint64_t record_addr) {
    if (record_addr == 0) return;
    volatile SdmaEventRecord *record =
        reinterpret_cast<volatile SdmaEventRecord *>(static_cast<uintptr_t>(record_addr));
    cache_invalidate_range(reinterpret_cast<const void *>(sdma_completion_cache_line(record)), PTO2_ALIGN_SIZE);
    uint32_t completed_tail = __atomic_load_n(&record->sq_tail, __ATOMIC_ACQUIRE);
    uint64_t channel_info_addr = __atomic_load_n(&record->channel_info, __ATOMIC_ACQUIRE);

    volatile uint64_t *record_head = reinterpret_cast<volatile uint64_t *>(record);
    __atomic_store_n(record_head, 0ULL, __ATOMIC_RELEASE);
    cache_flush_range(const_cast<const void *>(reinterpret_cast<volatile void *>(record_head)), sizeof(uint64_t));

    if (channel_info_addr == 0) return;
    uint64_t packed = (static_cast<uint64_t>(completed_tail) << 32) | static_cast<uint64_t>(completed_tail);
    volatile uint64_t *channel_info = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(channel_info_addr));
    __atomic_store_n(channel_info, packed, __ATOMIC_RELEASE);
    cache_flush_range(const_cast<const void *>(reinterpret_cast<volatile void *>(channel_info)), sizeof(uint64_t));
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_
