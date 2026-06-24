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
 * MIX co-ownership test — consumer (AIV): Vfinal = V0 + V1.
 *
 * Reads the two heap outputs produced by the MIX task's AIV0/AIV1 lanes and
 * writes the external Vfinal. Its fan-in is the single MIX task id, so it can
 * only run once the joint completion flag is set (i.e. after BOTH co-owned
 * AIV subtasks finished), validating the block.won remaining-counter logic.
 *
 *   args[0] = V0     (INPUT, heap)
 *   args[1] = V1     (INPUT, heap)
 *   args[2] = Vfinal (INOUT, external)
 *   args[3] = config (INPUT) int64_t[4]: [tile_size, grid_k, num_groups, num_tiles]
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
static __aicore__ void sum_tile_impl(__gm__ float *v0_ptr, __gm__ float *v1_ptr, __gm__ float *dst_ptr) {
    using DynShapeDim5 = Shape<1, 1, 1, TILE, TILE>;
    using DynStridDim5 = Stride<1, 1, 1, TILE, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, TILE, TILE, BLayout::RowMajor, -1, -1>;

    TileData v0Tile(TILE, TILE);
    TileData v1Tile(TILE, TILE);
    TileData outTile(TILE, TILE);
    TASSIGN(v0Tile, 0x0);
    TASSIGN(v1Tile, 0x10000);
    TASSIGN(outTile, 0x20000);

    GlobalData v0Global(v0_ptr);
    GlobalData v1Global(v1_ptr);
    GlobalData outGlobal(dst_ptr);

    TLOAD(v0Tile, v0Global);
    TLOAD(v1Tile, v1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(outTile, v0Tile, v1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *config = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ int64_t *cfg = reinterpret_cast<__gm__ int64_t *>(config->buffer.addr);
    uint64_t tile_size = static_cast<uint64_t>(cfg[0]);
    uint64_t tile_elems = tile_size * tile_size;
    int num_tiles = static_cast<int>(cfg[3]);

    __gm__ float *base_v0 = reinterpret_cast<__gm__ float *>(v0_tensor->buffer.addr) + v0_tensor->start_offset;
    __gm__ float *base_v1 = reinterpret_cast<__gm__ float *>(v1_tensor->buffer.addr) + v1_tensor->start_offset;
    __gm__ float *base_out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float *v0 = base_v0 + (tile_idx * tile_elems);
        __gm__ float *v1 = base_v1 + (tile_idx * tile_elems);
        __gm__ float *o = base_out + (tile_idx * tile_elems);
        switch (tile_size) {
        case 16:
            sum_tile_impl<16>(v0, v1, o);
            break;
        case 32:
            sum_tile_impl<32>(v0, v1, o);
            break;
        case 64:
            sum_tile_impl<64>(v0, v1, o);
            break;
        case 128:
            sum_tile_impl<128>(v0, v1, o);
            break;
        default:
            break;
        }
    }
}
