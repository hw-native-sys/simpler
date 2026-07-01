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

#include "pipe_sync.h"
#include "tensor.h"  // NOLINT(build/include_subdir)

// NOLINTNEXTLINE(build/namespaces)
using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

enum InputWindowOp : uint64_t {
    ADD_SCALAR = 1,
    ADD_TILES = 2,
};

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *first_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *second_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    uint64_t op = static_cast<uint64_t>(args[3]);
    float scalar = from_u64<float>(static_cast<uint64_t>(args[4]));

    __gm__ float *first = reinterpret_cast<__gm__ float *>(first_tensor->buffer.addr) + first_tensor->start_offset;
    __gm__ float *second = reinterpret_cast<__gm__ float *>(second_tensor->buffer.addr) + second_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kRows = 128;
    constexpr int kCols = 128;
    using TileShape = Shape<1, 1, 1, kRows, kCols>;
    using TileStride = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, TileShape, TileStride>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileData first_tile(kRows, kCols);
    TileData second_tile(kRows, kCols);
    TileData out_tile(kRows, kCols);
    TASSIGN(first_tile, 0x0);
    TASSIGN(second_tile, 0x10000);
    TASSIGN(out_tile, 0x20000);

    GlobalData first_global(first);
    GlobalData second_global(second);
    GlobalData out_global(out);

    TLOAD(first_tile, first_global);
    if (op == ADD_TILES) {
        TLOAD(second_tile, second_global);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (op == ADD_TILES) {
        TADD(out_tile, first_tile, second_tile);
    } else {
        TADDS(out_tile, first_tile, scalar);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(out_global, out_tile);

    pipe_sync();
}
