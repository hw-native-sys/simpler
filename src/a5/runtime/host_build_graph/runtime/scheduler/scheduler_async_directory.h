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
#include "scheduler_async_layout.h"

enum class SchedulerAsyncDirectoryResult : uint8_t {
    SUCCESS,
    INVALID_BIT,
    UNEXPECTED_STATE,
    PROTOCOL_CORRUPTION,
};

namespace scheduler_async_directory_detail {

inline constexpr uint32_t kWordBits = 64;
// Each other bit changes at most twice per run. The weak initial snapshot
// allows one extra mismatch; later failures observe changes after a strong CAS.
inline constexpr uint32_t kMaxCasAttempts = 2 * (kWordBits - 1) + 2;

namespace {

struct GmAtomics {
    __gm__ volatile uint64_t &word;

    inline __aicore__ uint64_t query() { return scheduler_gm_query(word); }

    inline __aicore__ uint64_t compare_exchange(uint64_t expected, uint64_t desired) {
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

}  // namespace

// Atomics supplies a weak query and a strong CAS returning the old word.
template <bool Publish, typename Atomics>
inline __aicore__ SchedulerAsyncDirectoryResult update(Atomics &atomics, uint32_t bit_index) {
    if (bit_index >= kWordBits) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t bit = UINT64_C(1) << bit_index;
    uint64_t observed = atomics.query();
    for (uint32_t attempt = 0; attempt < kMaxCasAttempts; ++attempt) {
        const uint64_t desired = Publish ? observed | bit : observed & ~bit;
        const uint64_t actual = atomics.compare_exchange(observed, desired);
        // An apparent duplicate uses an identity CAS; only its returned word
        // can reject the transition, even when the initial hint is stale.
        if (((actual & bit) != 0) == Publish) return SchedulerAsyncDirectoryResult::UNEXPECTED_STATE;
        if (actual == observed) return SchedulerAsyncDirectoryResult::SUCCESS;
        observed = actual;
    }
    return SchedulerAsyncDirectoryResult::PROTOCOL_CORRUPTION;
}

}  // namespace scheduler_async_directory_detail

// The caller owns the bit's unique 0 -> 1 -> 0 lifecycle; a cleared bit is
// not reused until run retirement. The line admits atomic writes and weak
// bypass queries, never cached writes. Payload/terminal barriers belong to the
// caller. Any non-success result requires run failure, never a normal retry.
static inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_publish(__gm__ volatile uint64_t &word, uint32_t bit_index) {
    scheduler_async_directory_detail::GmAtomics atomics{word};
    return scheduler_async_directory_detail::update<true>(atomics, bit_index);
}

static inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_clear(__gm__ volatile uint64_t &word, uint32_t bit_index) {
    scheduler_async_directory_detail::GmAtomics atomics{word};
    return scheduler_async_directory_detail::update<false>(atomics, bit_index);
}

// A scan still covers 64 groups. Only halves in the remaining scan range are
// read; the last word never touches a notification beyond the group's extent.
inline __aicore__ uint64_t
scheduler_async_directory_query(__gm__ SchedulerAsyncDirectoryWord &word, uint32_t first_bit, uint32_t group_count) {
    uint64_t pending = 0;
    uint64_t fallback = 0;
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t shift = half * 32;
        if (group_count <= shift || first_bit >= shift + 32) continue;
        const uint64_t address = word.notification_addresses[half];
        if (address == 0) fallback |= UINT64_C(0xffffffff) << shift;
        else
            pending |=
                static_cast<uint64_t>(scheduler_notify_query(*reinterpret_cast<__gm__ volatile uint32_t *>(address)))
                << shift;
    }
    if (fallback != 0) pending |= scheduler_gm_query(word.bits) & fallback;
    return pending;
}

// Mapped notifications do not return an old bit. The successful first seal
// proves publication ownership; a successful lease finish proves clear ownership.
// The GM fallback retains its checked CAS transition and error contract.
inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_publish(__gm__ SchedulerAsyncDirectoryWord &word, uint32_t bit_index) {
    if (bit_index >= 64) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t address = word.notification_addresses[bit_index / 32];
    if (address == 0) return scheduler_async_directory_publish(word.bits, bit_index);
    scheduler_notify_set_bits(*reinterpret_cast<__gm__ volatile uint32_t *>(address), UINT32_C(1) << (bit_index % 32));
    return SchedulerAsyncDirectoryResult::SUCCESS;
}

inline __aicore__ SchedulerAsyncDirectoryResult
scheduler_async_directory_clear(__gm__ SchedulerAsyncDirectoryWord &word, uint32_t bit_index) {
    if (bit_index >= 64) return SchedulerAsyncDirectoryResult::INVALID_BIT;
    const uint64_t address = word.notification_addresses[bit_index / 32];
    if (address == 0) return scheduler_async_directory_clear(word.bits, bit_index);
    scheduler_notify_clear_bits(
        *reinterpret_cast<__gm__ volatile uint32_t *>(address), UINT32_C(1) << (bit_index % 32)
    );
    return SchedulerAsyncDirectoryResult::SUCCESS;
}
