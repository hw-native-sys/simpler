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

#include "intrinsic.h"
#include "tensor.h"
#include "common/platform_config.h"
#include "layout.h"

namespace async_case {

inline __aicore__ int32_t payload_value(int64_t epoch, int64_t execution, int64_t element) {
    return static_cast<int32_t>(epoch * 100000 + execution * kPayloadWords + element + 1);
}

template <typename T>
inline __aicore__ __gm__ T *data(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(arg);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

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

inline __aicore__ void delay(int64_t milliseconds) {
    const uint64_t started = now();
    while (now() - started < static_cast<uint64_t>(milliseconds) * ticks_per_ms()) {}
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

inline __aicore__ int32_t signal_value(__gm__ int32_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    dcci(address, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *reinterpret_cast<volatile __gm__ int32_t *>(address);
#endif
}

inline __aicore__ void publish_signal(__gm__ int32_t *address, int32_t value) {
#ifdef PTO_CPUSTUB_HPP
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#else
    *address = value;
    flush(address);
#endif
}

inline __aicore__ __gm__ uint64_t *cell(__gm__ uint64_t *control, Control index, int64_t instance) {
    return control + (instance * kControlCells + static_cast<int64_t>(index)) * kControlStride;
}

inline __aicore__ bool await_at_least(__gm__ uint64_t *address, uint64_t expected, int64_t timeout_ms = 2000) {
    const uint64_t start = now();
    while (observe(address) < expected) {
        if (now() - start >= static_cast<uint64_t>(timeout_ms) * ticks_per_ms()) return false;
    }
    return true;
}

inline __aicore__ void stamp(__gm__ uint64_t *record, Observation field, uint64_t value = 1) {
    add(record + static_cast<int64_t>(field), value);
}

inline __aicore__ __gm__ int32_t *condition(__gm__ int32_t *signals, int64_t execution, int64_t index) {
    return signals + (execution * kConditionsPerExecution + index) * kSignalStride;
}

}  // namespace async_case
