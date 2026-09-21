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

#include "scheduler_types.h"

inline __aicore__ void scheduler_writeback_cache_line(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    (void)address;
#endif
}

inline __aicore__ void scheduler_writeback_cache_line(volatile __gm__ void *address) {
    scheduler_writeback_cache_line(const_cast<__gm__ void *>(address));
}

inline __aicore__ void scheduler_cache_barrier() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void scheduler_publish_cache_line(__gm__ void *address) {
    scheduler_writeback_cache_line(address);
    scheduler_cache_barrier();
}

inline __aicore__ void scheduler_publish_cache_line(volatile __gm__ void *address) {
    scheduler_publish_cache_line(const_cast<__gm__ void *>(address));
}

inline __aicore__ void scheduler_invalidate_cache_line(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE);
#else
    (void)address;
#endif
}

inline __aicore__ void scheduler_invalidate_cache_line(volatile __gm__ void *address) {
    scheduler_invalidate_cache_line(const_cast<__gm__ void *>(address));
}

inline __aicore__ void scheduler_observe_cache_line(__gm__ void *address) {
    scheduler_invalidate_cache_line(address);
    scheduler_cache_barrier();
}

inline __aicore__ void scheduler_observe_cache_line(volatile __gm__ void *address) {
    scheduler_observe_cache_line(const_cast<__gm__ void *>(address));
}

inline __aicore__ void scheduler_observe_data_cache(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void scheduler_writeback_dispatch_payload(__gm__ DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 0; offset < sizeof(DispatchPayload); offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(payload) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#else
    (void)payload;
#endif
}

inline __aicore__ void scheduler_observe_dispatch_payload_control(__gm__ DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    dcci(payload, SINGLE_CACHE_LINE);
#else
    (void)payload;
#endif
}

inline __aicore__ void scheduler_observe_dispatch_payload_arguments(__gm__ DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 64; offset < sizeof(DispatchPayload); offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(payload) + offset, SINGLE_CACHE_LINE);
    }
#else
    (void)payload;
#endif
}

inline __aicore__ void scheduler_observe_dispatch_payload_barrier() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// A5 ld_dev bypasses the scalar DCache. It is only an observation/prefilter
// primitive; ownership changes use the CAS/exchange/fetch wrappers below.
// Publication-protected metadata is invalidated separately before consumption.
inline __aicore__ int64_t scheduler_gm_query(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
    __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
    return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint64_t scheduler_gm_query(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint64_t scheduler_gm_query_u32_pair(__gm__ volatile uint32_t *values, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    __gm__ uint32_t *first = const_cast<__gm__ uint32_t *>(values);
    __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(first);
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
#else
    const uint64_t low = __atomic_load_n(&values[0], order);
    const uint64_t high = __atomic_load_n(&values[1], order);
    return low | (high << 32);
#endif
}

inline __aicore__ void
scheduler_gm_store(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    st_dev(
        static_cast<uint64_t>(desired), reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&value)), 0
    );
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
scheduler_gm_store(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    st_dev(desired, const_cast<__gm__ uint64_t *>(&value), 0);
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
scheduler_gm_store(__gm__ volatile uint32_t &value, uint32_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    st_dev(desired, const_cast<__gm__ uint32_t *>(&value), 0);
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
scheduler_gm_publish(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    (void)atomicExch(const_cast<__gm__ uint64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ int64_t
scheduler_gm_exchange(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    return atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
    return __atomic_exchange_n(&value, desired, order);
#endif
}

inline __aicore__ uint64_t
scheduler_gm_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    return atomicAdd(const_cast<__gm__ uint64_t *>(&value), delta);
#else
    return __atomic_fetch_add(&value, delta, order);
#endif
}

inline __aicore__ int64_t scheduler_gm_compare_exchange(
    __gm__ volatile int64_t &value, int64_t expected, int64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
#if defined(__CCE_AICORE__)
    (void)success_order;
    (void)failure_order;
    return atomicCAS(const_cast<__gm__ int64_t *>(&value), expected, desired);
#else
    int64_t observed = expected;
    (void)__atomic_compare_exchange_n(&value, &observed, desired, false, success_order, failure_order);
    return observed;
#endif
}

inline __aicore__ uint64_t scheduler_gm_compare_exchange(
    __gm__ volatile uint64_t &value, uint64_t expected, uint64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
#if defined(__CCE_AICORE__)
    (void)success_order;
    (void)failure_order;
    return atomicCAS(const_cast<__gm__ uint64_t *>(&value), expected, desired);
#else
    uint64_t observed = expected;
    (void)__atomic_compare_exchange_n(&value, &observed, desired, false, success_order, failure_order);
    return observed;
#endif
}

inline __aicore__ uint64_t
scheduler_gm_fetch_or(__gm__ volatile uint64_t &value, uint64_t bits, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    uint64_t observed = scheduler_gm_query(value);
    while ((observed & bits) != bits) {
        uint64_t actual = scheduler_gm_compare_exchange(value, observed, observed | bits);
        if (actual == observed) break;
        observed = actual;
    }
    return observed;
#else
    return __atomic_fetch_or(&value, bits, order);
#endif
}

inline __aicore__ uint64_t
scheduler_gm_fetch_and(__gm__ volatile uint64_t &value, uint64_t bits, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    uint64_t observed = scheduler_gm_query(value);
    while ((observed & bits) != observed) {
        uint64_t actual = scheduler_gm_compare_exchange(value, observed, observed & bits);
        if (actual == observed) break;
        observed = actual;
    }
    return observed;
#else
    return __atomic_fetch_and(&value, bits, order);
#endif
}
