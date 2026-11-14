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
/**
 * Reading one deferred completion condition from a lane.
 *
 * The condition formats are not this runtime's to define: `CompletionToken`,
 * the `COMPLETION_TYPE_*` constants and the record layouts are the ABI a
 * backend writes and either runtime reads, so they are included here rather
 * than restated. The poll is kept here for the reason `gm.h` gives for the
 * accesses it holds -- a lane reaches this from every checkpoint, where the
 * sibling runtime reaches it from a scheduler lane that is already carrying
 * phase instrumentation, so what is one load for one of them is a traced
 * call for the other.
 *
 * Three properties of the formats are load-bearing here:
 *
 *   - COUNTER is an `int32_t` on the wire. A negative sample is below every
 *     supported threshold, so it reads as pending rather than as a large
 *     unsigned value.
 *   - An SDMA event record counts upward and is never cleared, so its test
 *     is a threshold and the read leaves it alone.
 *   - A CCU event is consumed by whoever observes it. The clear must not
 *     issue before the sample that justified it, and the line must read back
 *     clear afterwards: one that does not means a second observer holds the
 *     same event, which is malformed rather than merely pending.
 */

#pragma once

#include <cstdint>

#include "host_build_graph/completion_types.h"
#include "host_build_graph/completion_token.h"
#include "platform_comm/ccu/ccu_native_abi.h"
#include "scheduler/gm.h"

/** A CCU event sample, ordered against the clear that may follow it. */
inline __aicore__ uint64_t percounter_ccu_event_read(__gm__ uint64_t *event) {
#if defined(__CCE_AICORE__)
    const auto value = static_cast<uint64_t>(__builtin_cce_ld_dev(event, 0));
    dsb(DSB_DDR);
    return value;
#else
    return __atomic_load_n(event, __ATOMIC_ACQUIRE);
#endif
}

/** Consume the event, ordered against the read that confirms the clear. */
inline __aicore__ void percounter_ccu_event_clear(__gm__ uint64_t *event) {
#if defined(__CCE_AICORE__)
    st_dev(UINT64_C(0), event, 0);
    dsb(DSB_DDR);
#else
    __atomic_store_n(event, UINT64_C(0), __ATOMIC_RELEASE);
#endif
}

/** Whether a token names a condition this runtime knows how to read. */
inline __aicore__ bool percounter_completion_token_valid(const CompletionToken &token) {
    if (token.addr == 0) return false;
    if (token.engine == COMPLETION_ENGINE_CCU && token.completion_type == COMPLETION_TYPE_CCU)
        return token.addr % alignof(uint64_t) == 0 && token.expected_value == 0 &&
               token.backend_cookie == SIMPLER_CCU_EVENT_MASK;
    if (token.engine != COMPLETION_ENGINE_SDMA) return false;
    if (token.completion_type == COMPLETION_TYPE_COUNTER)
        return token.addr % sizeof(uint32_t) == 0 && token.backend_cookie == 0 && token.expected_value <= INT32_MAX;
    if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD)
        return token.addr % sizeof(uint64_t) == 0 && token.expected_value == 0 && token.backend_cookie != 0;
    return false;
}

/** One sample of one condition; the caller re-reads a pending one later. */
inline __aicore__ CompletionPollResult percounter_poll_valid_completion(const CompletionToken &token) {
    if (token.completion_type == COMPLETION_TYPE_CCU) {
        auto *event = reinterpret_cast<__gm__ uint64_t *>(token.addr);
        if ((percounter_ccu_event_read(event) & token.backend_cookie) != token.backend_cookie)
            return {CompletionPollState::PENDING, 0};
        percounter_ccu_event_clear(event);
        return percounter_ccu_event_read(event) == 0 ?
                   CompletionPollResult{CompletionPollState::READY, 0} :
                   CompletionPollResult{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    }

    if (token.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD) {
        auto *record = reinterpret_cast<__gm__ uint64_t *>(token.addr);
        const uint64_t completed = percounter_gm_read(*record);
        return {completed >= token.backend_cookie ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
    }

    auto *counter = reinterpret_cast<__gm__ uint32_t *>(token.addr);
    const uint32_t value = percounter_gm_read(*counter);
    if (value > INT32_MAX) return {CompletionPollState::PENDING, 0};
    return {value >= token.expected_value ? CompletionPollState::READY : CompletionPollState::PENDING, 0};
}
