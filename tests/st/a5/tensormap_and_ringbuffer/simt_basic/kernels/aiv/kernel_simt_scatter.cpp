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

// Minimal SIMT element-scatter kernel (AIV).
//
// Distilled from the ptoas-generated mscatter reference. Drops cosmetic
// noise (v1..v30 names, dummy v4/v5/v6/v7 constants, explicit
// Layout::ND, the verbose Tile template tail, GM-offset arithmetic that
// always reduces to zero, the ptoas_auto_sync_tail wrapper).
//
// Kept on purpose:
//   - per-data 3-tile alias pattern (TLOAD binds one tile, MSCATTER
//     reads from another aliased to the same UB address; a single-tile
//     form has reproduced golden mismatches on hw)
//   - `set_mask_norm` / `set_vector_mask` SIMT mask init
//   - `MTE2 → V` sync before MSCATTER (the ptoas default `MTE2 → MTE3`
//     silently drops the scatter on a5 hw)
//   - `__DAV_VEC__` guard so the AIC variant compiles to a no-op
//
// Operation: out[idx[r, c]] = src[r, c] for an 8x32 source and 256-slot
// destination.

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "pipe_sync.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr int TILE_ROWS = 8;
static constexpr int TILE_COLS = 32;
static constexpr int DST_LEN = TILE_ROWS * TILE_COLS;  // 256
static constexpr int SRC_TILE_BYTES = TILE_ROWS * TILE_COLS * sizeof(float);

static __aicore__ void simt_scatter_impl(__gm__ float *src, __gm__ int32_t *idx, __gm__ float *out) {
    using SrcTile = Tile<TileType::Vec, float, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, int32_t, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;

    using TileShape = Shape<1, 1, 1, TILE_ROWS, TILE_COLS>;
    using TileStride = pto::Stride<DST_LEN, DST_LEN, DST_LEN, TILE_COLS, 1>;
    using SrcGT = GlobalTensor<float, TileShape, TileStride>;
    using IdxGT = GlobalTensor<int32_t, TileShape, TileStride>;

    using DstShape = Shape<1, 1, 1, 1, DST_LEN>;
    using DstStride = pto::Stride<DST_LEN, DST_LEN, DST_LEN, DST_LEN, 1>;
    using DstGT = GlobalTensor<float, DstShape, DstStride>;

    // Per-data 3-tile alias pattern:
    //   *_loader  — bound directly at the UB offset; consumed by TLOAD
    //   *_scatter — bound via the loader's data() pointer; consumed by MSCATTER
    //   *_anchor  — bound to the same offset literal; preserves the
    //               original ptoas binding sequence
    constexpr int SRC_UB = 0;
    constexpr int IDX_UB = SRC_TILE_BYTES;

    SrcTile src_loader(TILE_ROWS, TILE_COLS);
    TASSIGN(src_loader, SRC_UB);
    SrcTile src_scatter(TILE_ROWS, TILE_COLS);
    TASSIGN(src_scatter, reinterpret_cast<uint64_t>(src_loader.data()));
    SrcTile src_anchor(TILE_ROWS, TILE_COLS);
    TASSIGN(src_anchor, static_cast<uint64_t>(SRC_UB));

    IdxTile idx_loader(TILE_ROWS, TILE_COLS);
    TASSIGN(idx_loader, IDX_UB);
    IdxTile idx_scatter(TILE_ROWS, TILE_COLS);
    TASSIGN(idx_scatter, reinterpret_cast<uint64_t>(idx_loader.data()));
    IdxTile idx_anchor(TILE_ROWS, TILE_COLS);
    TASSIGN(idx_anchor, static_cast<uint64_t>(IDX_UB));

    SrcGT srcGlobal(src);
    IdxGT idxGlobal(idx);
    DstGT dstGlobal(out);

    TLOAD(src_anchor, srcGlobal);
    TLOAD(idx_anchor, idxGlobal);

    // MTE2 → V before MSCATTER (critical: MTE2 → MTE3 silently drops the
    // scatter on a5 hw).
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MSCATTER(dstGlobal, src_scatter, idx_scatter);

    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;

    __gm__ Tensor *idx_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *idx = reinterpret_cast<__gm__ int32_t *>(idx_tensor->buffer.addr) + idx_tensor->start_offset;

    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    simt_scatter_impl(src, idx, out);
}
