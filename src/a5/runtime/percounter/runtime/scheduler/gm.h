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
 * The GM accesses this runtime's protocols are built out of.
 *
 * These are host_build_graph's `scheduler_memory.h` primitives, kept here
 * because the two runtimes read GM for different reasons and at different
 * rates. A lane here is self-scheduling: it reaches GM at every checkpoint,
 * where the sibling's AICore side mostly waits to be told. An access that is
 * one instruction for one of them and a traced call for the other is not the
 * same primitive, and sharing the declaration hides that.
 *
 * **Each of these is a plain access, and an AICore caller depends on it being
 * one.** They are small enough to inline into the loops that call them, and
 * the loops are tight enough that it matters.
 *
 * What the barriers are for is measured, not assumed:
 *
 *   - `percounter_gm_read` is a bypass load. It does observe the writing
 *     core's own atomic write, but across cores it can lag.
 *   - `percounter_gm_fetch_add` adding zero is a read-modify-write, which
 *     does not lag. `percounter_gate_read` in the lane loop is that read
 *     followed by the fence.
 *   - Neither orders a *pair* of reads. Two GM reads are two samples, not one
 *     snapshot, and only a barrier between them makes the second no older
 *     than the first -- see `percounter_observe_fence` in the lane loop.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "dispatch_payload.h"
#include "scheduler/scheduler_types.h"

// pto-isa 8fb67fa7e routes urma_gather_async.hpp into every TU that includes
// <pto/pto-inst.hpp>; on A5 that pulls in <simt_api/device_functions.h> and its
// SIMT atomicAdd template wins overload resolution for the classic 64-bit add.
// The self-contained incore link (ld.lld over the kernel object only) has no
// SIMT device library to resolve it, so perform the add with the classic 64-bit
// CAS, which is still inlined. Remove once pto-isa keeps the SIMT header out of
// the common pto-inst.hpp chain.
#if defined(__CCE_AICORE__)
static inline __aicore__ uint64_t percounter_gm_atomic_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta) {
    auto *word = const_cast<__gm__ int64_t *>(reinterpret_cast<__gm__ volatile int64_t *>(&value));
    int64_t observed = atomicCAS(word, 0, 0);
    while (true) {
        const int64_t desired = static_cast<int64_t>(static_cast<uint64_t>(observed) + delta);
        const int64_t actual = atomicCAS(word, observed, desired);
        if (actual == observed) return static_cast<uint64_t>(observed);
        observed = actual;
    }
}
#endif

/** Invalidate one line, then wait for it, so the next read sees publication. */
inline __aicore__ void percounter_observe_cache_line(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline __aicore__ void percounter_observe_cache_line(volatile __gm__ void *address) {
    percounter_observe_cache_line(const_cast<__gm__ void *>(address));
}

/** The same for the whole data cache, before reading a producer's output. */
inline __aicore__ void percounter_observe_data_cache(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    dcci(address, ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/** A bypass load: cheap, and behind by at most one publication. */
inline __aicore__ int64_t percounter_gm_read(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
    __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
    return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint64_t percounter_gm_read(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
    return __atomic_load_n(&value, order);
#endif
}

inline __aicore__ uint32_t percounter_gm_read(__gm__ volatile uint32_t &value, int order = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)order;
    return static_cast<uint32_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint32_t *>(&value), 0));
#else
    return __atomic_load_n(&value, order);
#endif
}

/** Two adjacent 32-bit fields as one word, so neither can tear from the other. */
inline __aicore__ uint64_t percounter_gm_read_u32_pair(__gm__ volatile uint32_t *values, int order = __ATOMIC_ACQUIRE) {
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

/** An exchange, which is how every shared word here is published. */
inline __aicore__ void
percounter_gm_store(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    (void)atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
percounter_gm_store(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    (void)atomicExch(const_cast<__gm__ uint64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
}

inline __aicore__ void
percounter_gm_store(__gm__ volatile uint32_t &value, uint32_t desired, int order = __ATOMIC_RELEASE) {
#if defined(__CCE_AICORE__)
    (void)order;
    (void)atomicExch(const_cast<__gm__ uint32_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
}

/** Adding zero reads a word without leaving it changed, and does not lag. */
inline __aicore__ uint64_t
percounter_gm_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)order;
    return percounter_gm_atomic_fetch_add(value, delta);
#else
    return __atomic_fetch_add(&value, delta, order);
#endif
}

inline __aicore__ int64_t percounter_gm_compare_exchange(
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

inline __aicore__ uint64_t percounter_gm_compare_exchange(
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

/**
 * Push a whole dispatch payload out, line by line.
 *
 * A lane fills and runs its own payload, but the kernel reads its arguments
 * from GM with ordinary loads, so what this lane wrote has to leave the
 * core's cache first. The payload spans several lines, hence the loop.
 */
inline __aicore__ void percounter_writeback_dispatch_payload(__gm__ DispatchPayload *payload) {
#if defined(__CCE_AICORE__)
    for (uint64_t offset = 0; offset < sizeof(DispatchPayload); offset += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(payload) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    // Preparation can yield to another task, whose input acquisition drops
    // this cache. Finish every slot writeback before that invalidation.
    dsb((mem_dsb_t)0);
#else
    (void)payload;
#endif
}
