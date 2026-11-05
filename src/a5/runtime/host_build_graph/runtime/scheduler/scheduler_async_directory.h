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

#include "scheduler_memory.h"

struct alignas(128) SchedulerAsyncDirectoryWord {
    volatile uint64_t bits{0};
};

static_assert(sizeof(SchedulerAsyncDirectoryWord) == 128);
static_assert(std::is_standard_layout_v<SchedulerAsyncDirectoryWord>);
static_assert(std::is_trivially_copyable_v<SchedulerAsyncDirectoryWord>);

enum class SchedulerAsyncDirectoryResult : uint8_t { SUCCESS, INVALID_BIT, PROTOCOL_ERROR };

// Every bit has exactly one publisher and one terminal owner per run. Legal
// additions/subtractions cannot carry/borrow into a neighbour, including bit 63.
// A duplicate may already have corrupted neighbours when it is detected: the
// caller must fail the run, never compensate or retry the contribution.
inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_publish_once(__gm__ SchedulerAsyncDirectoryWord &directory, uint32_t bit) {
    if (bit >= 64) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t mask = UINT64_C(1) << bit;
    const uint64_t previous = scheduler_gm_fetch_add(directory.bits, mask);
    return (previous & mask) == 0 ? SchedulerAsyncDirectoryResult::SUCCESS :
                                    SchedulerAsyncDirectoryResult::PROTOCOL_ERROR;
}

inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_clear_once(__gm__ SchedulerAsyncDirectoryWord &directory, uint32_t bit) {
    if (bit >= 64) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t mask = UINT64_C(1) << bit;
    const uint64_t previous = scheduler_gm_fetch_add(directory.bits, UINT64_C(0) - mask);
    return (previous & mask) != 0 ? SchedulerAsyncDirectoryResult::SUCCESS :
                                    SchedulerAsyncDirectoryResult::PROTOCOL_ERROR;
}

// The directory is runtime-owned atomic storage, never a DMA completion word.
// Contents/lease publication precedes publish_once; terminal precedes clear_once.
// Pending retries leave the activity bit set and only transfer lease ownership.
inline __aicore__ uint64_t scheduler_async_directory_load(__gm__ SchedulerAsyncDirectoryWord &directory) {
    return scheduler_gm_fetch_add(directory.bits, UINT64_C(0));
}
