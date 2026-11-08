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

#include <stdint.h>

#include "async_completion_format.h"
#include "host_build_graph/completion_token.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// Reads only the external condition word. The registering kernel owns the
// live-address contract; the runtime does not infer ownership from an address.
// The entire condition cache line must be read-only/clean on this observer:
// invalidation cannot repair an old dirty line owned by some other use.
//
// DCCI + a scalar read follows PTO's TWAIT signal-observation pattern. Barriers
// explicitly complete maintenance and the load. No CACHELINE_OUT, cached store,
// atomic RMW, old kernel UB, or wait-for-ready loop is used. Compiling this path
// for AIC and AIV does not establish DMA/cache visibility on real hardware.
struct AsyncCompletionScalarReader {
    template <typename Word>
    inline __aicore__ Word load(uint64_t address) const {
        static_assert(sizeof(Word) == 4 || sizeof(Word) == 8);
        auto *word = reinterpret_cast<volatile __gm__ const Word *>(static_cast<uintptr_t>(address));
#if defined(__CCE_AICORE__)
        constexpr uint64_t cache_line_bytes = 64;
        auto *line = reinterpret_cast<__gm__ void *>(static_cast<uintptr_t>(address & ~(cache_line_bytes - 1)));
        dcci(line, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        __asm__ __volatile__("" ::: "memory");
        const Word value = *word;
        dsb(DSB_DDR);
        __asm__ __volatile__("" ::: "memory");
        return value;
#else
        // CPU test/simulation reader: an acquire load, never fetch-add/CAS.
        return __atomic_load_n(word, __ATOMIC_ACQUIRE);
#endif
    }

    inline __aicore__ int32_t counter(uint64_t address) const { return load<int32_t>(address); }
    inline __aicore__ uint64_t sdma_post_done(uint64_t address) const { return load<uint64_t>(address); }
};

inline __aicore__ bool async_completion_entry_valid(const DeferredCompletionEntry &entry) {
    return entry._pad == 0 && async_completion_format_valid(entry);
}

// Exactly one scalar observation for a valid entry; malformed metadata is
// rejected without dereferencing addr. Resource validation belongs to the
// caller's existing run/domain binding and cannot be inferred from this POD.
// READY proves the condition, not remote consumer completion or arbitrary
// payload cache acquisition. SDMA records are monotonic and are never cleared.
template <typename Reader>
inline __aicore__ CompletionPollResult async_observe_once(const DeferredCompletionEntry &entry, Reader &reader) {
    constexpr CompletionPollResult invalid{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (!async_completion_entry_valid(entry)) return invalid;
    if (entry.completion_type == COMPLETION_TYPE_COUNTER) {
        const int32_t value = reader.counter(entry.addr);
        // A negative sampled value remains below every supported threshold.
        // Converting it to uint32_t would incorrectly satisfy nonnegative waits.
        return {
            value >= static_cast<int32_t>(entry.expected_value) ? CompletionPollState::READY :
                                                                  CompletionPollState::PENDING,
            SIMPLER_ERROR_NONE
        };
    }
    if (entry.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) {
        const uint64_t value = reader.sdma_post_done(entry.addr);
        return {
            value >= entry.backend_cookie ? CompletionPollState::READY : CompletionPollState::PENDING,
            SIMPLER_ERROR_NONE
        };
    }
    return invalid;
}

inline __aicore__ CompletionPollResult async_observe_once(const DeferredCompletionEntry &entry) {
    AsyncCompletionScalarReader reader;
    return async_observe_once(entry, reader);
}
