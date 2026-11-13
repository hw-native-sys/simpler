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

#include <cstdint>
#include <pto/pto-inst.hpp>
#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#endif

#include "async_kernel_api.h"
#include "common/platform_config.h"
#include "tensor.h"

namespace {
constexpr int32_t kLineWords = 32;
constexpr int32_t kExecutionLines = 65;

inline __aicore__ uint64_t shape_ticks() {
#ifdef PTO_CPUSTUB_HPP
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
#else
    return get_sys_cnt();
#endif
}

inline __aicore__ void shape_delay(int32_t milliseconds) {
#ifdef PTO_CPUSTUB_HPP
    const uint64_t duration = static_cast<uint64_t>(milliseconds) * 1000000;
#else
    const uint64_t duration = static_cast<uint64_t>(milliseconds) * PLATFORM_PROF_SYS_CNT_FREQ / 1000;
#endif
    const uint64_t start = shape_ticks();
    while (shape_ticks() - start < duration) {}
}

inline __aicore__ int32_t shape_read(__gm__ int32_t *address) {
#ifdef PTO_CPUSTUB_HPP
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#else
    return static_cast<int32_t>(__builtin_cce_ld_dev(reinterpret_cast<__gm__ uint32_t *>(address), 0));
#endif
}

inline __aicore__ void shape_write(__gm__ int32_t *address, int32_t value) {
#ifdef PTO_CPUSTUB_HPP
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
#else
    *address = value;
    dcci(address, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
#endif
}

inline __aicore__ int32_t shape_conditions(int32_t block, int32_t lane, int32_t capacity) {
    const int32_t kind = (block + lane + 1) % 3;
    return kind == 0 ? 0 : kind == 1 ? 1 : capacity;
}

inline __aicore__ void run_shape(__gm__ int64_t *args, bool vector) {
    const auto *state_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *state = reinterpret_cast<__gm__ int32_t *>(state_tensor->buffer.addr) + state_tensor->start_offset;
    auto *out = reinterpret_cast<__gm__ int32_t *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t role = static_cast<int32_t>(args[2]);
    const int32_t mask = static_cast<int32_t>(args[3]);
    const int32_t blocks = static_cast<int32_t>(args[4]);
    const bool skip = args[5] != 0;
    const int32_t capacity = static_cast<int32_t>(args[6]);
    if (role == 0) {
        for (int32_t i = 0; i < 1 + 3 * blocks * kExecutionLines; ++i)
            shape_write(state + i * kLineWords, 0);
        shape_write(state, skip ? 0 : 1);
        return;
    }
    if (role == 3) {
        bool returned = true;
        bool satisfied = true;
        for (int32_t block = 0; block < blocks; ++block) {
            for (int32_t lane = 0; lane < 3; ++lane) {
                if ((mask & (1 << lane)) == 0) continue;
                auto *row = state + (1 + (block * 3 + lane) * kExecutionLines) * kLineWords;
                returned = returned && shape_read(row) == (skip ? 0 : 2);
                for (int32_t condition = 0; !skip && condition < shape_conditions(block, lane, capacity); ++condition)
                    satisfied = satisfied && shape_read(row + (condition + 1) * kLineWords) == 1;
            }
        }
        shape_write(out, returned ? 1 : -1);
        shape_write(out + kLineWords, satisfied ? 1 : -1);
        return;
    }
    const int32_t block = get_block_idx(args);
    const int32_t lane = vector ? (mask == 2 ? 1 : 1 + get_sub_block_id(args)) : 0;
    auto *row = state + (1 + (block * 3 + lane) * kExecutionLines) * kLineWords;
    const int32_t count = shape_conditions(block, lane, capacity);
    if (role == 1) {
        auto ctx = get_async_ctx(args);
        if (ctx.resident_enabled != 1) {
            shape_write(row, -1);
            return;
        }
        shape_write(row, 1);
        for (int32_t condition = 0; condition < count; ++condition)
            save_expected_notification_counter(ctx, row + (condition + 1) * kLineWords, 1);
        // A zero-condition member still owes its execution-return contribution.
        if (count == 0 && block == blocks - 1) shape_delay(5);
        shape_write(row, 2);
    } else if (role == 2) {
        shape_delay(2);
        for (int32_t condition = 0; condition < count; ++condition) {
            if (count == 64 && condition == 63) shape_delay(2);
            shape_write(row + (condition + 1) * kLineWords, 1);
        }
    }
}
}  // namespace
