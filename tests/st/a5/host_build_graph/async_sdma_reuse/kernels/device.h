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
#include "intrinsic.h"
#include "tensor.h"
#include "common.h"
namespace sdma_series {
template <typename T>
inline __aicore__ __gm__ T *data(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}
inline __aicore__ uint64_t add(__gm__ uint64_t *address, uint64_t value) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
#else
    return atomicAdd(address, value);
#endif
}
inline __aicore__ uint64_t observe(__gm__ uint64_t *address) { return add(address, 0); }
inline __aicore__ void flush(__gm__ void *address) {
#ifdef PTO_CPUSTUB_HPP
    (void)address;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
    dcci(address, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
#endif
}
inline __aicore__ uint64_t read_record_word(__gm__ uint64_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    dcci(address, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *reinterpret_cast<volatile __gm__ uint64_t *>(address);
#endif
}
inline __aicore__ uint64_t read_completion(__gm__ uint64_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    auto *scratch = reinterpret_cast<__ubuf__ uint64_t *>(0);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ uint32_t *>(scratch), reinterpret_cast<__gm__ uint32_t *>(address), 0, 1,
        sizeof(uint64_t), 0, 0, 0, 0, 0, 0
    );
    pipe_barrier(PIPE_ALL);
    return *scratch;
#endif
}
inline __aicore__ bool records_ready(__gm__ uint64_t *meta) {
    const uint64_t count = observe(meta + RECEIPT_COUNT);
    if (count == 0 || count > kQueueNum) return false;
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t address = read_record_word(meta + kReceiptBase + 2 * i);
        const uint64_t expected = read_record_word(meta + kReceiptBase + 2 * i + 1);
        if (address == 0 || expected == 0 || read_completion(reinterpret_cast<__gm__ uint64_t *>(address)) < expected)
            return false;
    }
    return true;
}
// Each already-published receipt is sampled at most once; an absent receipt never blocks a core.
inline __aicore__ void sample_prior(__gm__ uint64_t *all, int64_t index, int64_t posts) {
    auto *mine = all + index * kMetaWords;
    for (int64_t prior = 0; prior < posts; ++prior) {
        if (prior == index) continue;
        auto *meta = all + prior * kMetaWords;
        const uint64_t count = observe(meta + RECEIPT_COUNT);
        if (count == 0 || count > kQueueNum) continue;
        for (uint64_t record = 0; record < count; ++record) {
            const uint64_t address = read_record_word(meta + kReceiptBase + 2 * record);
            const uint64_t expected = read_record_word(meta + kReceiptBase + 2 * record + 1);
            if (address == 0 || expected == 0) continue;
            const uint64_t value = read_completion(reinterpret_cast<__gm__ uint64_t *>(address));
            if (value >= expected) continue;
            mine[PRIOR_INDEX] = prior + 1;
            mine[PRIOR_RECORD] = record + 1;
            mine[PRIOR_VALUE] = value;
            mine[PRIOR_EXPECTED] = expected;
            flush(mine + PRIOR_INDEX);
            add(mine + PENDING_SAMPLE, 1);
            return;
        }
    }
}
}  // namespace sdma_series
