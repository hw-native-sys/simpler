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

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#include "intrinsic.h"

namespace pto2::resident {

inline constexpr uint32_t ASYNC_RECORD_ALIGNMENT = 64;
inline constexpr uint32_t ASYNC_INVALID_INDEX = UINT32_MAX;
enum class AsyncPublicationState : uint32_t { OPEN = 0, SEALED = 1, FAILED = 2 };

// One physical execution owns these lines until its return publishes them.
// Scheduler leases and progress occupy separate cache lines.
struct alignas(ASYNC_RECORD_ALIGNMENT) AsyncGroupData {
    // The invocation updates registration state in its first cache line.
    volatile uint32_t count;
    volatile int32_t error_code;
    volatile uint32_t reservation_serial;
    volatile uint32_t reservation_limit;
    volatile uint32_t reservation_begin;
    volatile uint32_t reservation_written;
    volatile AsyncPublicationState state;
    uint32_t group_index;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t entries_offset;
    uint64_t sdma_guards_offset;

    uint32_t block_index;
    uint32_t subtask_slot;
    uint32_t entries_bytes;
    uint32_t sdma_guard_count;
    uint16_t capacity;
    uint16_t urma_single_aiv;
};

// SDK session creation and SQ/post-id updates share one writer per physical
// cluster. Both AIV lanes use the same default channel group.
struct alignas(ASYNC_RECORD_ALIGNMENT) AsyncSdmaSubmissionGuard {
    volatile uint64_t word;
};

struct AsyncReservation {
    uint64_t run_epoch{0};
    uint32_t group_index{ASYNC_INVALID_INDEX};
    uint32_t serial{0};
    uint32_t capacity{0};
};

constexpr inline __aicore__ uint32_t async_group_entry_bytes(uint32_t capacity) {
    if (capacity == 0 || capacity > MAX_COMPLETIONS_PER_TASK) return 0;
    const uint32_t bytes = capacity * sizeof(DeferredCompletionEntry);
    return (bytes + ASYNC_RECORD_ALIGNMENT - 1) & ~(ASYNC_RECORD_ALIGNMENT - 1);
}

static_assert(sizeof(AsyncGroupData) == 128 && alignof(AsyncGroupData) == ASYNC_RECORD_ALIGNMENT);
static_assert(offsetof(AsyncGroupData, count) == 0 && offsetof(AsyncGroupData, state) == 24);
static_assert(offsetof(AsyncGroupData, sdma_guards_offset) == 56 && offsetof(AsyncGroupData, entries_offset) == 48);
static_assert(std::is_standard_layout_v<AsyncGroupData> && std::is_trivially_copyable_v<AsyncGroupData>);
static_assert(sizeof(AsyncReservation) == 24);
static_assert(sizeof(AsyncSdmaSubmissionGuard) == 64);
static_assert(
    std::is_standard_layout_v<AsyncSdmaSubmissionGuard> && std::is_trivially_copyable_v<AsyncSdmaSubmissionGuard>
);
static_assert(async_group_entry_bytes(1) == 64 && async_group_entry_bytes(64) == 2048);

}  // namespace pto2::resident
