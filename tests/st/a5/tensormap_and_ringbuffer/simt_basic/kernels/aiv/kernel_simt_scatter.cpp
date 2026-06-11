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

// Minimal SIMT element-scatter kernel (AIV), structurally mirroring
// pto-isa's runElem2D template (tests/npu/a5/src/st/testcase/mscatter/
// mscatter_kernel.cpp) so the MSCATTER call sees the same Shape /
// Stride / Tile / sync surroundings that the upstream ST validates
// against.
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
    // No explicit set_mask_norm / set_vector_mask: pto-isa's mscatter ST does
    // not call them either. MSCATTER is `__simt_callee__`; bisheng inlines
    // the SIMT-side mask setup at every call site, so the vec mask state
    // is owned by MSCATTER itself rather than by the host kernel.

    // Follow pto-isa's runElem2D structure: Shape/Stride descriptors,
    // static-Valid-extents Tile, single-tile binding, idx-then-src TASSIGN
    // and TLOAD order, and the MTE2->V sync before MSCATTER. The post-MSCATTER
    // tail uses the repo's pipe_sync() helper (drains MTE3->S on AIV) rather
    // than runElem2D's pipe_barrier(PIPE_ALL) + V->MTE3.
    using SrcShape = pto::Shape<1, 1, 1, TILE_ROWS, TILE_COLS>;
    using SrcStride = pto::Stride<1, 1, 1, TILE_COLS, 1>;
    using IdxShape = pto::Shape<1, 1, 1, TILE_ROWS, TILE_COLS>;
    using IdxStride = pto::Stride<1, 1, 1, TILE_COLS, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, DST_LEN>;
    using OutStride = pto::Stride<1, 1, 1, DST_LEN, 1>;

    GlobalTensor<float, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<int32_t, IdxShape, IdxStride> idxGlobal(idx);
    GlobalTensor<float, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, float, TILE_ROWS, TILE_COLS, BLayout::RowMajor, TILE_ROWS, TILE_COLS>;
    using IdxTile = Tile<TileType::Vec, int32_t, TILE_ROWS, TILE_COLS, BLayout::RowMajor, TILE_ROWS, TILE_COLS>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((TILE_ROWS * TILE_COLS * static_cast<int>(sizeof(int32_t)) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Element-scatter on both sim and onboard. The CPU sim backend exposes
    // only a non-templated MSCATTER whose impl is already per-element, while
    // the a5 onboard backend defaults the non-templated form to Coalesce::Row
    // and gates the templated overloads behind PTO_NPU_ARCH_A5, so onboard
    // must select Coalesce::Elem explicitly. See pto-isa#164.
#ifdef __CPU_SIM
    MSCATTER(outGlobal, srcTile, idxTile);
#else
    MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(outGlobal, srcTile, idxTile);
#endif

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
