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

#include "common/platform_config.h"
#include "tensor.h"

#include "pipe_sync.h"

static __aicore__ uint64_t delay_ticks() {
#ifdef PTO_CPUSTUB_HPP
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count() /
        (1000000000 / PLATFORM_PROF_SYS_CNT_FREQ)
    );
#else
    return get_sys_cnt();
#endif
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *src1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    // Keep the synthetic observation window below the CI scheduler's two-second
    // no-progress budget even when a scalar loop runs slowly. The iteration
    // argument remains an upper work bound; no result depends on completing it.
    const int32_t spin_iters = static_cast<int32_t>(args[3]);
    const uint64_t started = delay_ticks();
    constexpr uint64_t kDelayBudgetTicks = PLATFORM_PROF_SYS_CNT_FREQ;
    volatile int32_t spin_accumulator = 0;
    for (int32_t i = 0; i < spin_iters; ++i) {
        if ((i & 1023) == 0 && delay_ticks() - started >= kDelayBudgetTicks) break;
        ++spin_accumulator;
    }
    (void)spin_accumulator;

    __gm__ float *src0 = reinterpret_cast<__gm__ float *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float *src1 = reinterpret_cast<__gm__ float *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kTRows = 128;
    constexpr int kTCols = 128;
    using DynShapeDim5 = pto::Shape<1, 1, 1, kTRows, kTCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols, 1>;
    using GlobalData = pto::GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = pto::Tile<pto::TileType::Vec, float, kTRows, kTCols, pto::BLayout::RowMajor, -1, -1>;

    TileData src0_tile(kTRows, kTCols);
    TileData src1_tile(kTRows, kTCols);
    TileData dst_tile(kTRows, kTCols);
    TASSIGN(src0_tile, 0x0);
    TASSIGN(src1_tile, 0x10000);
    TASSIGN(dst_tile, 0x20000);

    GlobalData src0_global(src0);
    GlobalData src1_global(src1);
    GlobalData dst_global(out);

    TLOAD(src0_tile, src0_global);
    TLOAD(src1_tile, src1_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dst_tile, src0_tile, src1_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst_global, dst_tile);
    pipe_sync();
}
