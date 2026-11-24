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
 * The uninstrumented lane compiles these to plain accesses in its loop.
 * The profiled translation unit uses percounter's own per-core binding;
 * no HBG observer state participates in either path.
 *
 * What the barriers are for is measured, not assumed:
 *
 *   - `percounter_gm_read` is a bypass load. It does observe the writing
 *     core's own atomic write, but across cores it can lag.
 *   - `percounter_gm_fetch_add` adding zero is a read-modify-write, which
 *     does not lag; the atomic operation still needs an ordering fence when
 *     its result is combined with another publication.
 *   - Neither orders a *pair* of reads. Two GM reads are two samples, not one
 *     snapshot, and only a barrier between them makes the second no older
 *     than the first. Callers use publication or cache-observation barriers
 *     before consuming dependent metadata or tensor values.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "dispatch_payload.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/profiling_memory.h"

static inline __aicore__ void percounter_memory_barrier() {
    simpler::percounter::percounter_scalar_memory_issue(simpler::percounter::ProfilePhase::DsbAll, 1, []() __aicore__ {
#if defined(__CCE_AICORE__)
        dsb((mem_dsb_t)0);
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    });
}

/** Invalidate one line, then wait for it, so the next read sees publication. */
static inline __aicore__ void percounter_observe_cache_line(__gm__ void *address) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::DcciSingle, 1, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
            dcci(address, SINGLE_CACHE_LINE);
#else
            (void)address;
#endif
        }
    );
    percounter_memory_barrier();
}

static inline __aicore__ void percounter_observe_cache_line(volatile __gm__ void *address) {
    percounter_observe_cache_line(const_cast<__gm__ void *>(address));
}

/** Invalidate the whole data cache at a run boundary. */
static inline __aicore__ void percounter_observe_data_cache(__gm__ void *address) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::DcciEntire, 1, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
            dcci(address, ENTIRE_DATA_CACHE);
#else
            (void)address;
#endif
        }
    );
    percounter_memory_barrier();
}

/** A bypass load: cheap, and behind by at most one publication. */
static inline __aicore__ int64_t percounter_gm_read(__gm__ volatile int64_t &value, int order = __ATOMIC_ACQUIRE) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::BypassLoad, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
            __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
            return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
            return __atomic_load_n(&value, order);
#endif
        }
    );
}

static inline __aicore__ uint64_t percounter_gm_read(__gm__ volatile uint64_t &value, int order = __ATOMIC_ACQUIRE) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::BypassLoad, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(&value), 0));
#else
            return __atomic_load_n(&value, order);
#endif
        }
    );
}

static inline __aicore__ uint32_t percounter_gm_read(__gm__ volatile uint32_t &value, int order = __ATOMIC_ACQUIRE) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::BypassLoad, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            return static_cast<uint32_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint32_t *>(&value), 0));
#else
            return __atomic_load_n(&value, order);
#endif
        }
    );
}

/** Eight immutable metadata bytes; the device observes the Host-published word through GM. */
static inline __aicore__ uint64_t percounter_gm_read_immutable_word(const __gm__ void *address) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::BypassLoad, [=]() __aicore__ {
#if defined(__CCE_AICORE__)
            auto *word = reinterpret_cast<const __gm__ uint64_t *>(address);
            return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(word), 0));
#else
            uint64_t value;
            __builtin_memcpy(&value, address, sizeof(value));
            return value;
#endif
        }
    );
}

/**
 * A STARS CntNotify slot is 32 bytes: the 32-bit SET register doubles as the
 * read address, and the ADD alias sits at SET + 8. Reads are non-consuming, so
 * several observers see the same value. Neither access may be widened into a
 * neighbouring register, and the mapping never takes cache maintenance.
 *
 * These commands access CntNotify SRAM through device MMIO.
 */
static inline __aicore__ uint32_t percounter_notify_query(__gm__ volatile uint32_t &value) {
    return simpler::percounter::percounter_scalar_notify_read(
        reinterpret_cast<uint64_t>(&value), [&value]() __aicore__ {
#if defined(__CCE_AICORE__)
            return static_cast<uint32_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint32_t *>(&value), 0));
#else
            return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
#endif
        }
    );
}

/** Command one increments once: a payload of two does not add two. */
static inline __aicore__ void percounter_notify_add(__gm__ volatile uint32_t &value, uint32_t credits) {
    for (uint32_t credit = 0; credit < credits; ++credit) {
        simpler::percounter::percounter_scalar_notify_issue(
            simpler::percounter::ProfilePhase::CntNotifyAddIssue, reinterpret_cast<uint64_t>(&value), 1,
            [&value]() __aicore__ {
#if defined(__CCE_AICORE__)
                __builtin_cce_st_dev(UINT32_C(1), const_cast<__gm__ uint32_t *>(&value), 0);
#else
                (void)__atomic_fetch_add(&value, UINT32_C(1), __ATOMIC_RELEASE);
#endif
            }
        );
    }
}

/** An exchange, which is how every shared word here is published. */
static inline __aicore__ void
percounter_gm_store(__gm__ volatile int64_t &value, int64_t desired, int order = __ATOMIC_RELEASE) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::AtomicStoreIssue, 1, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            (void)atomicExch(const_cast<__gm__ int64_t *>(&value), desired);
#else
            __atomic_store_n(&value, desired, order);
#endif
        }
    );
}

static inline __aicore__ void
percounter_gm_store(__gm__ volatile uint64_t &value, uint64_t desired, int order = __ATOMIC_RELEASE) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::AtomicStoreIssue, 1, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            (void)atomicExch(const_cast<__gm__ uint64_t *>(&value), desired);
#else
            __atomic_store_n(&value, desired, order);
#endif
        }
    );
}

static inline __aicore__ void
percounter_gm_store(__gm__ volatile uint32_t &value, uint32_t desired, int order = __ATOMIC_RELEASE) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::AtomicStoreIssue, 1, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            (void)atomicExch(const_cast<__gm__ uint32_t *>(&value), desired);
#else
            __atomic_store_n(&value, desired, order);
#endif
        }
    );
}

/** Adding zero reads a word without leaving it changed, and does not lag. */
static inline __aicore__ uint64_t
percounter_gm_fetch_add(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::AtomicFetchAdd, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            return atomicAdd(const_cast<__gm__ uint64_t *>(&value), delta);
#else
            return __atomic_fetch_add(&value, delta, order);
#endif
        }
    );
}

static inline __aicore__ void
percounter_gm_fetch_add_issue(__gm__ volatile uint64_t &value, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
    simpler::percounter::percounter_scalar_memory_issue(
        simpler::percounter::ProfilePhase::AtomicFetchAddIssue, 1, [=, &value]() __aicore__ {
#if defined(__CCE_AICORE__)
            (void)order;
            (void)atomicAdd(const_cast<__gm__ uint64_t *>(&value), delta);
#else
            (void)__atomic_fetch_add(&value, delta, order);
#endif
        }
    );
}

static inline __aicore__ int64_t percounter_gm_compare_exchange(
    __gm__ volatile int64_t &value, int64_t expected, int64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::AtomicCas, [=, &value]() __aicore__ {
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

static inline __aicore__ uint64_t percounter_gm_compare_exchange(
    __gm__ volatile uint64_t &value, uint64_t expected, uint64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
    return simpler::percounter::percounter_scalar_memory_result(
        simpler::percounter::ProfilePhase::AtomicCas, [=, &value]() __aicore__ {
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
