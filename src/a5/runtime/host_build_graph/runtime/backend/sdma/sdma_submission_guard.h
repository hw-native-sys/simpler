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

#include "async_condition_writer.h"
#include "scheduler/scheduler_memory.h"

namespace pto2::resident {

enum class SdmaGuardResult : uint8_t { ACQUIRED, BUSY, FAILED };

// One attempt only; the submission adapter owns the bounded contention wait.
static inline __aicore__ SdmaGuardResult sdma_try_acquire(__gm__ AsyncSdmaSubmissionGuard &guard) {
    const uint64_t previous = scheduler_gm_compare_exchange(guard.word, UINT64_C(0), UINT64_C(1));
    if (previous == 0) {
        scheduler_cache_barrier();
        return SdmaGuardResult::ACQUIRED;
    }
    return previous == 1 ? SdmaGuardResult::BUSY : SdmaGuardResult::FAILED;
}

static inline __aicore__ bool sdma_release(__gm__ AsyncSdmaSubmissionGuard &guard, bool success) {
    // SDK SQ tails and the persisted post ID must be visible before a successor
    // rebuilds its session. Failed submissions poison this run's channel group.
    scheduler_cache_barrier();
    const uint64_t result = scheduler_gm_compare_exchange(guard.word, UINT64_C(1), success ? UINT64_C(0) : UINT64_C(2));
    scheduler_cache_barrier();
    return result == 1;
}

inline __aicore__ __gm__ AsyncSdmaSubmissionGuard *
sdma_submission_guard(const AsyncWriter &writer, uint32_t channel_group) {
    if (!detail::valid(writer)) return nullptr;
    auto *data = detail::group(writer);
    const uint64_t bytes = static_cast<uint64_t>(data->sdma_guard_count) * sizeof(AsyncSdmaSubmissionGuard);
    if (data->sdma_guards_offset == 0 || channel_group >= data->sdma_guard_count ||
        !detail::span_valid(writer, data->sdma_guards_offset, bytes, ASYNC_RECORD_ALIGNMENT)) {
        detail::fail(writer, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        return nullptr;
    }
    return reinterpret_cast<__gm__ AsyncSdmaSubmissionGuard *>(writer.image + data->sdma_guards_offset) + channel_group;
}

}  // namespace pto2::resident
