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
#include <type_traits>

#include "scheduler_types.h"
#include "scheduler_scalar_record.h"

// pto-isa 8fb67fa7e routes urma_gather_async.hpp into every TU that includes
// <pto/pto-inst.hpp>; on A5 that pulls in <simt_api/device_functions.h> and its
// SIMT atomicAdd template wins overload resolution for the classic 64-bit add.
// The self-contained incore link (ld.lld over the kernel object only) has no
// SIMT device library to resolve it, so perform the add with the classic 64-bit
// CAS, which is still inlined. Remove once pto-isa keeps the SIMT header out of
// the common pto-inst.hpp chain.
#if defined(__CCE_AICORE__)
static inline __aicore__ uint64_t scheduler_gm_atomic_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta) {
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

// Capture ordinary operands and address values by value at observer calls.
// GM words retain reference captures so their accesses stay inside the timed
// primitive, while caller temporaries do not escape through the callback.
static inline __aicore__ void scheduler_writeback_cache_line(__gm__ void *address) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDcciOut, 1, [address]() __aicore__ {
#if defined(__CCE_AICORE__)
        dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
    (void)address;
#endif
    });
}

static inline __aicore__ void scheduler_writeback_cache_line(volatile __gm__ void *address) {
    scheduler_writeback_cache_line(const_cast<__gm__ void *>(address));
}

static inline __aicore__ void scheduler_cache_barrier() {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDsbAll, 1, []() __aicore__ {
#if defined(__CCE_AICORE__)
        dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    });
}

static inline __aicore__ void scheduler_publish_cache_line(__gm__ void *address) {
    scheduler_writeback_cache_line(address);
    scheduler_cache_barrier();
}

static inline __aicore__ void scheduler_publish_cache_line(volatile __gm__ void *address) {
    scheduler_publish_cache_line(const_cast<__gm__ void *>(address));
}

static inline __aicore__ void scheduler_invalidate_cache_line(__gm__ void *address) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDcciSingle, 1, [address]() __aicore__ {
#if defined(__CCE_AICORE__)
        dcci(address, SINGLE_CACHE_LINE);
#else
    (void)address;
#endif
    });
}

static inline __aicore__ void scheduler_invalidate_cache_line(volatile __gm__ void *address) {
    scheduler_invalidate_cache_line(const_cast<__gm__ void *>(address));
}

static inline __aicore__ void scheduler_observe_cache_line(__gm__ void *address) {
    scheduler_invalidate_cache_line(address);
    scheduler_cache_barrier();
}

static inline __aicore__ void scheduler_observe_cache_line(volatile __gm__ void *address) {
    scheduler_observe_cache_line(const_cast<__gm__ void *>(address));
}

static inline __aicore__ void scheduler_invalidate_startup_cache() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
    dci();
    __asm__ volatile("" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static inline __aicore__ void scheduler_observe_data_cache(__gm__ void *address) {
#if defined(__CCE_AICORE__)
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDcciEntire, 1, [address]() __aicore__ {
        dcci(address, ENTIRE_DATA_CACHE);
    });
    scheduler_cache_barrier();
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

// A5 ld_dev bypasses the scalar DCache but does not observe a value an atomic
// wrote until that write has drained -- the writer's own core included -- so a
// word any core publishes by atomic must be read by an atomic too. This is a
// read: adding zero returns the current value and stores it back unchanged.
// Reserve it for words whose staleness breaks a protocol -- it is a
// read-modify-write, so a hot poll shared by every core stays on
// scheduler_gm_query.
static inline __aicore__ int64_t scheduler_gm_load(__gm__ volatile int64_t &value) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarAtomicLoad, [&value]() __aicore__ {
#if defined(__CCE_AICORE__)
        __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&value));
        return static_cast<int64_t>(scheduler_gm_atomic_fetch_add(*address, UINT64_C(0)));
#else
    return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
#endif
    });
}

// A5 ld_dev bypasses the scalar DCache. Atomic-published words require
// atomic confirmation; bypass-published signals use this read directly.
// Ownership changes use the CAS/exchange/fetch wrappers below.
// Publication-protected metadata is invalidated separately before consumption.
// The memory order every wrapper below takes is honored by the host and sim
// build, whose implementation is a C++ atomic. An AICore reaches ordering
// through explicit barriers instead, so the parameter is inert there.
static inline __aicore__ int64_t scheduler_gm_query(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [&value, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
        __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
        return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
    return __atomic_load_n(&value, order);
#endif
    });
}

static inline __aicore__ uint64_t scheduler_gm_query(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [&value, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
    return __atomic_load_n(&value, order);
#endif
    });
}

static inline __aicore__ uint64_t
scheduler_gm_query_u32_pair(__gm__ volatile uint32_t *values, int order = __ATOMIC_ACQUIRE) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarBypassLoad, [values, order]() __aicore__ {
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
    });
}

static inline __aicore__ void scheduler_store_barrier() {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDsbDdr, 1, []() __aicore__ {
#if defined(__CCE_AICORE__)
        dsb(DSB_DDR);
#elif defined(OUT_OF_ORDER_STORE_BARRIER)
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    });
}

static inline __aicore__ void scheduler_ddr_barrier() {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarDsbDdr, 1, []() __aicore__ {
#if defined(__CCE_AICORE__)
        dsb(DSB_DDR);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    });
}

// Each word has one active writer. Publication completes before ownership
// handoff. While workers run, its whole cache line excludes AICore RMW,
// cached stores and dirty writeback. Payload ordering belongs to the caller.
template <typename Word>
static inline __aicore__ void scheduler_publish_signal(__gm__ volatile Word &value, Word desired) {
    static_assert(std::is_same_v<Word, int64_t> || std::is_same_v<Word, uint64_t>);
#if defined(__CCE_AICORE__)
    auto *address = reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ Word *>(&value));
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarBypassStore, 1, [address, desired]() __aicore__ {
        st_dev(static_cast<uint64_t>(desired), address, 0);
    });
    scheduler_ddr_barrier();
#else
    __atomic_store_n(&value, desired, __ATOMIC_RELEASE);
#endif
}

// Shared ownership words and atomic-published data retain atomic exchange.
// Their lines exclude cached writes that could restore old values.
static inline __aicore__ void
scheduler_gm_store(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_RELEASE) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarAtomicStoreIssue, 1, [&value, desired, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        (void)atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
    });
}

static inline __aicore__ void
scheduler_gm_store(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarAtomicStoreIssue, 1, [&value, desired, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        (void)atomicExch(const_cast<__gm__ uint64_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
    });
}

static inline __aicore__ void
scheduler_gm_store(__gm__ volatile uint32_t &value, uint32_t desired, int order = __ATOMIC_RELEASE) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarAtomicStoreIssue, 1, [&value, desired, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        (void)atomicExch(const_cast<__gm__ uint32_t *>(&value), desired);
#else
    __atomic_store_n(&value, desired, order);
#endif
    });
}

static inline __aicore__ int64_t
scheduler_gm_exchange(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_ACQ_REL) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarAtomicExchange, [&value, desired, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        return atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
    return __atomic_exchange_n(&value, desired, order);
#endif
    });
}

static inline __aicore__ uint64_t
scheduler_gm_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
    return scheduler_scalar_memory_result(SchedPhaseKind::ScalarAtomicFetchAdd, [&value, delta, order]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)order;
        return scheduler_gm_atomic_fetch_add(value, delta);
#else
    return __atomic_fetch_add(&value, delta, order);
#endif
    });
}

static inline __aicore__ int64_t scheduler_gm_compare_exchange(
    __gm__ volatile int64_t &value, int64_t expected, int64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
    return scheduler_scalar_memory_result(
        SchedPhaseKind::ScalarAtomicCas, [&value, expected, desired, success_order, failure_order]() __aicore__ {
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
    );
}

static inline __aicore__ uint64_t scheduler_gm_compare_exchange(
    __gm__ volatile uint64_t &value, uint64_t expected, uint64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
    return scheduler_scalar_memory_result(
        SchedPhaseKind::ScalarAtomicCas, [&value, expected, desired, success_order, failure_order]() __aicore__ {
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
    );
}

static inline __aicore__ uint64_t
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

static inline __aicore__ uint64_t
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

static inline __aicore__ void scheduler_gm_fetch_add_discard(__gm__ volatile uint64_t &value, uint64_t delta) {
    scheduler_scalar_memory_issue(SchedPhaseKind::ScalarAtomicFetchAddIssue, 1, [&value, delta]() __aicore__ {
#if defined(__CCE_AICORE__)
        (void)scheduler_gm_atomic_fetch_add(value, delta);
#else
    (void)__atomic_fetch_add(&value, delta, __ATOMIC_ACQ_REL);
#endif
    });
}
