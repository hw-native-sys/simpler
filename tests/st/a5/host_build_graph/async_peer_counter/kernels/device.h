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

#include <pto/pto-inst.hpp>
#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#endif

#include "common.h"
#include "intrinsic.h"
#include "tensor.h"
#include "common/platform_config.h"
#include "platform_comm/comm_context.h"

namespace peer_counter {

template <typename T>
inline __aicore__ __gm__ T *data(int64_t argument) {
    const auto *tensor = reinterpret_cast<__gm__ const Tensor *>(argument);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

inline __aicore__ bool parameters(int64_t epoch, int64_t tag) {
    return epoch > 0 && epoch <= INT32_MAX && tag > 0 && tag <= kMaxTag;
}

inline __aicore__ uint64_t atomic_add(__gm__ uint64_t *address, uint64_t value) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
#else
    // pto-isa 8fb67fa7e's SIMT atomicAdd template is undefined at the
    // self-contained incore link; the classic 64-bit CAS still links.
    auto *word = const_cast<__gm__ int64_t *>(reinterpret_cast<__gm__ volatile int64_t *>(address));
    int64_t observed = atomicCAS(word, 0, 0);
    while (true) {
        const int64_t desired = static_cast<int64_t>(static_cast<uint64_t>(observed) + value);
        const int64_t actual = atomicCAS(word, observed, desired);
        if (actual == observed) return static_cast<uint64_t>(observed);
        observed = actual;
    }
#endif
}
inline __aicore__ uint64_t observe(__gm__ uint64_t *address) { return atomic_add(address, 0); }

// Diagnostic fields start at zero; counters accumulate, while scalar observations
// are recorded once by their owning kernel.
// Atomic accesses avoid a dirty scalar cache alias of the same diagnostic line.
inline __aicore__ void record(__gm__ uint64_t *diagnostic, Field field, int64_t value = 1) {
    if (value != 0) atomic_add(diagnostic + static_cast<uint32_t>(field), static_cast<uint64_t>(value));
}
inline __aicore__ void error(__gm__ uint64_t *diagnostic, int64_t value) {
    if (value != 0 && observe(diagnostic + static_cast<uint32_t>(Field::ERROR)) == 0)
        record(diagnostic, Field::ERROR, value);
}
inline __aicore__ void error(__gm__ uint64_t *diagnostic, Error value) {
    error(diagnostic, static_cast<int64_t>(value));
}
inline __aicore__ void begin(__gm__ uint64_t *diagnostic, Role role, int64_t epoch, int64_t tag) {
    record(diagnostic, Field::VISITS);
    record(diagnostic, Field::ROLE, static_cast<int64_t>(role));
    record(diagnostic, Field::EPOCH, epoch);
    record(diagnostic, Field::TAG, tag);
}
inline __aicore__ void finish(__gm__ uint64_t *diagnostic) { record(diagnostic, Field::BODY_DONE); }

inline __aicore__ uint64_t now() {
#ifdef PTO_CPUSTUB_HPP
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#else
    return get_sys_cnt();
#endif
}
inline __aicore__ uint64_t ticks_per_ms() {
#ifdef PTO_CPUSTUB_HPP
    return 1000000;
#else
    return PLATFORM_PROF_SYS_CNT_FREQ / 1000;
#endif
}

inline __aicore__ void flush_line(__gm__ void *address) {
#ifdef PTO_CPUSTUB_HPP
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
#endif
}
inline __aicore__ int32_t signal_value(__gm__ int32_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    dcci(address, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *reinterpret_cast<volatile __gm__ int32_t *>(address);
#endif
}

inline __aicore__ int32_t expected_value(int64_t tag, uint32_t source_rank, uint32_t row, uint32_t element) {
    return static_cast<int32_t>(tag * 100000 + source_rank * 1000 + row * kDataWords + element + 1);
}

inline __aicore__ bool peer_region(
    __gm__ const CommContext *context, uint32_t rank, uint64_t local_address, uint64_t bytes, uint64_t *remote_address
) {
    if (context == nullptr || remote_address == nullptr || context->rankNum != 2 || context->rankId != rank ||
        rank > 1 || context->winSize == 0 || bytes == 0)
        return false;
    const uint64_t local_base = context->windowsIn[rank];
    const uint64_t peer_base = context->windowsIn[1 - rank];
    if (local_base == 0 || peer_base == 0 || (local_base & 63u) != 0 || (peer_base & 63u) != 0 ||
        local_address < local_base || context->winSize > UINT64_MAX - local_base ||
        context->winSize > UINT64_MAX - peer_base)
        return false;
    const uint64_t offset = local_address - local_base;
    if ((local_address & 63u) != 0 || offset > context->winSize || bytes > context->winSize - offset ||
        offset > UINT64_MAX - peer_base)
        return false;
    const uint64_t remote = peer_base + offset;
    if (bytes > UINT64_MAX - 63u) return false;
    const uint64_t rounded_bytes = (bytes + 63u) & ~UINT64_C(63);
    if (rounded_bytes > context->winSize - offset) return false;
    *remote_address = remote;
    return true;
}

}  // namespace peer_counter
