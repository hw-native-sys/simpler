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

enum class SchedulerAsyncDirectoryResult : uint8_t {
    SUCCESS,
    INVALID_BIT,
    UNEXPECTED_STATE,
    PROTOCOL_CORRUPTION,
};

namespace scheduler_async_directory_detail {

inline constexpr uint32_t kWordBits = 64;
// Each other bit changes at most twice per run. Every failed strong CAS
// observes a value change; the target bit has one writer for each transition.
inline constexpr uint32_t kMaxCasAttempts = 2 * (kWordBits - 1) + 1;

struct GmAtomics {
    __gm__ volatile uint64_t &word;

    inline __aicore__ uint64_t load() { return scheduler_gm_fetch_add(word, UINT64_C(0)); }

    inline __aicore__ uint64_t compare_exchange(uint64_t expected, uint64_t desired) {
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

// Atomics supplies a linearizable load and a strong CAS returning the old word.
template <bool Publish, typename Atomics>
inline __aicore__ SchedulerAsyncDirectoryResult update(Atomics &atomics, uint32_t bit_index) {
    if (bit_index >= kWordBits) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t bit = UINT64_C(1) << bit_index;
    uint64_t observed = atomics.load();
    for (uint32_t attempt = 0; attempt < kMaxCasAttempts; ++attempt) {
        if (((observed & bit) != 0) == Publish) return SchedulerAsyncDirectoryResult::UNEXPECTED_STATE;
        const uint64_t desired = Publish ? observed | bit : observed & ~bit;
        const uint64_t actual = atomics.compare_exchange(observed, desired);
        if (actual == observed) return SchedulerAsyncDirectoryResult::SUCCESS;
        observed = actual;
    }
    return SchedulerAsyncDirectoryResult::PROTOCOL_CORRUPTION;
}

}  // namespace scheduler_async_directory_detail

// The caller owns the bit's unique 0 -> 1 -> 0 lifecycle; a cleared bit is
// not reused until run retirement. Its entire line admits only atomic accesses
// while the run is live. Payload/terminal publication barriers belong to the
// caller. Any non-success result requires run failure, never a normal retry.
inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_publish(__gm__ volatile uint64_t &word, uint32_t bit_index) {
    scheduler_async_directory_detail::GmAtomics atomics{word};
    return scheduler_async_directory_detail::update<true>(atomics, bit_index);
}

inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_clear(__gm__ volatile uint64_t &word, uint32_t bit_index) {
    scheduler_async_directory_detail::GmAtomics atomics{word};
    return scheduler_async_directory_detail::update<false>(atomics, bit_index);
}
