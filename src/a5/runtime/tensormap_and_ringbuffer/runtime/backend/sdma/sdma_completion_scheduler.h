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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "aicore_completion_mailbox.h"
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

inline CompletionPollResult poll_sdma_event_record(uint64_t record_addr) {
    if (record_addr == 0) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }
    volatile SdmaEventRecord *record =
        reinterpret_cast<volatile SdmaEventRecord *>(static_cast<uintptr_t>(record_addr));
    uint32_t flag = __atomic_load_n(&record->flag, __ATOMIC_ACQUIRE);
    return {flag != 0 ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE};
}

inline void retire_sdma_event_record(uint64_t record_addr) {
    if (record_addr == 0) return;
    volatile SdmaEventRecord *record =
        reinterpret_cast<volatile SdmaEventRecord *>(static_cast<uintptr_t>(record_addr));
    uint32_t completed_tail = __atomic_load_n(&record->sq_tail, __ATOMIC_ACQUIRE);
    uint64_t channel_info_addr = __atomic_load_n(&record->channel_info, __ATOMIC_ACQUIRE);

    volatile uint64_t *record_head = reinterpret_cast<volatile uint64_t *>(record);
    __atomic_store_n(record_head, 0ULL, __ATOMIC_RELEASE);

    if (channel_info_addr == 0) return;
    uint64_t packed = (static_cast<uint64_t>(completed_tail) << 32) | static_cast<uint64_t>(completed_tail);
    volatile uint64_t *channel_info = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(channel_info_addr));
    __atomic_store_n(channel_info, packed, __ATOMIC_RELEASE);
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_SCHEDULER_H_
