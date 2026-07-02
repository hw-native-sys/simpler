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
 * MIX co-ownership test — AIV0 subtask: V0 = A + B (single tile, element-wise).
 *
 * AIV0 lane of a 1C+2V MIX task. Shared argument list (see kernel_mm.cpp);
 * this lane writes the V0 output at args[3].
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int TILE>
static __aicore__ void add_tile_impl(__gm__ float *a_ptr, __gm__ float *b_ptr, __gm__ float *dst_ptr) {
    using DynShapeDim5 = Shape<1, 1, 1, TILE, TILE>;
    using DynStridDim5 = Stride<1, 1, 1, TILE, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, TILE, TILE, BLayout::RowMajor, -1, -1>;

    TileData aTile(TILE, TILE);
    TileData bTile(TILE, TILE);
    TileData outTile(TILE, TILE);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(outTile, 0x20000);

    GlobalData aGlobal(a_ptr);
    GlobalData bGlobal(b_ptr);
    GlobalData outGlobal(dst_ptr);

    TLOAD(aTile, aGlobal);
    TLOAD(bTile, bGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(outTile, aTile, bTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);  // V0
    __gm__ Tensor *config = reinterpret_cast<__gm__ Tensor *>(args[5]);

    __gm__ int64_t *cfg = reinterpret_cast<__gm__ int64_t *>(config->buffer.addr);
    uint64_t tile_size = static_cast<uint64_t>(cfg[0]);
    uint64_t tile_elems = tile_size * tile_size;
    int num_tiles = static_cast<int>(cfg[3]);

    __gm__ float *base_a = reinterpret_cast<__gm__ float *>(a_tensor->buffer.addr) + a_tensor->start_offset;
    __gm__ float *base_b = reinterpret_cast<__gm__ float *>(b_tensor->buffer.addr) + b_tensor->start_offset;
    __gm__ float *base_out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float *a = base_a + (tile_idx * tile_elems);
        __gm__ float *b = base_b + (tile_idx * tile_elems);
        __gm__ float *o = base_out + (tile_idx * tile_elems);
        switch (tile_size) {
        case 16:
            add_tile_impl<16>(a, b, o);
            break;
        case 32:
            add_tile_impl<32>(a, b, o);
            break;
        case 64:
            add_tile_impl<64>(a, b, o);
            break;
        case 128:
            add_tile_impl<128>(a, b, o);
            break;
        default:
            break;
        }
    }
}
