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
 * Demo kernel:
 *   mapped_host_buffer[i] = mapped_host_buffer[i] + 1.0f
 *   out[i] = mapped_host_buffer[i] + 1.0f
 *
 * The input pointer comes from mallocHostDeviceShareMem(), so a successful result
 * shows that the kernel was able to read and write the mapped host buffer
 * directly while also producing a regular copy-back output.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"  // NOLINT(build/include_subdir)

using namespace pto;  // NOLINT(build/namespaces)

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *mapped_host_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *mapped_host = reinterpret_cast<__gm__ float *>(mapped_host_tensor->buffer.addr) + mapped_host_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr float kAddValue = 1.0f;
    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src_tile(vRows, vCols);
    TileData dst_tile(vRows, vCols);
    TASSIGN(src_tile, 0x0);
    TASSIGN(dst_tile, 0x10000);

    GlobalData mapped_host_global(mapped_host);
    GlobalData dst_global(out);

    TLOAD(src_tile, mapped_host_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(dst_tile, src_tile, kAddValue);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mapped_host_global, dst_tile);
    TSTORE(dst_global, dst_tile);

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}
