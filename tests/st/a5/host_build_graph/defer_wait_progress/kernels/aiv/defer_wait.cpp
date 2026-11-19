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
#include <cstdint>
#include <pto/pto-inst.hpp>
#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#endif

#include "async_kernel_api.h"
#include "common/platform_config.h"
#include "tensor.h"

namespace {
constexpr uint32_t kLineWords = 32;

inline __aicore__ uint64_t now_ticks() {
#ifdef PTO_CPUSTUB_HPP
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#else
    return get_sys_cnt();
#endif
}

inline __aicore__ uint64_t millisecond_ticks() {
#ifdef PTO_CPUSTUB_HPP
    return 1000000;
#else
    return static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) / 1000;
#endif
}

inline __aicore__ int32_t read_word(__gm__ int32_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    return static_cast<int32_t>(__builtin_cce_ld_dev(reinterpret_cast<__gm__ uint32_t *>(address), 0));
#endif
}

inline __aicore__ void write_word(__gm__ int32_t *address, int32_t value) {
#ifdef PTO_CPUSTUB_HPP
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#else
    *address = value;
    dcci(address, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#endif
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *state_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *state = reinterpret_cast<__gm__ int32_t *>(state_tensor->buffer.addr) + state_tensor->start_offset;
    auto *out = reinterpret_cast<__gm__ int32_t *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t role = static_cast<int32_t>(args[2]);
    const int32_t index = static_cast<int32_t>(args[3]);
    const int32_t waiters = static_cast<int32_t>(args[4]);
    const int32_t chunks = (waiters + 63) / 64;
    if (role == 0) {
        for (int32_t i = 0; i < 2 * waiters + 3 + chunks; ++i)
            write_word(state + i * kLineWords, 0);
        write_word(state, index);
    } else if (role == 1) {
        auto ctx = get_async_ctx(args);
        write_word(state + (index + 1) * kLineWords, 1);
        // The existing registration API must return without waiting for signal.
        save_expected_notification_counter(ctx, state, 1);
    } else if (role == 2) {
        write_word(state + (waiters + 1) * kLineWords, 1);
    } else if (role == 3) {
        // The publisher cannot wait for other tasks: a pending probe or waiter
        // may already occupy another dispatch slot on this same execution core.
        // Expose a premature consumer while another AIV executes this publisher.
        const uint64_t delay = now_ticks();
        while (now_ticks() - delay < 5 * millisecond_ticks()) {}
        write_word(state + (waiters + 2) * kLineWords, 1);
        write_word(state, 1);
    } else if (role == 4) {
        write_word(state + (waiters + 3 + index) * kLineWords, read_word(state));
    } else if (role == 5) {
        bool ready = true;
        for (int32_t i = index; i < waiters && i < index + 64; ++i)
            ready = ready && read_word(state + (waiters + 3 + i) * kLineWords) == 1;
        write_word(state + (2 * waiters + 3 + index / 64) * kLineWords, ready ? 1 : -1);
    } else if (role == 6) {
        bool ready = true;
        for (int32_t i = 0; i < chunks; ++i)
            ready = ready && read_word(state + (2 * waiters + 3 + i) * kLineWords) == 1;
        for (int32_t i = 0; i < waiters; ++i)
            ready = ready && read_word(state + (i + 1) * kLineWords) == 1;
        write_word(out, read_word(state + (waiters + 2) * kLineWords));
        write_word(out + kLineWords, read_word(state + (waiters + 1) * kLineWords));
        write_word(out + 2 * kLineWords, ready ? 1 : -1);
    }
}
