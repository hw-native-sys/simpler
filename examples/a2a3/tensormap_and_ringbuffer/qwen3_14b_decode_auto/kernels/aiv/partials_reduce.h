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

// Sums the private split-K partials of one output band.
//
// `part` addresses row 0 of a [splits*16, RowStride] fp32 partial buffer, already
// offset to the band's first column; partial j occupies rows [j*16, j*16+16), so
// its element offset from `part` is j * 16 * RowStride. `out` addresses the same
// band of the [16, RowStride] accumulator. Both carry row stride RowStride
// because each is a column slice of its parent, so a row step is a full parent row.
//
// `Atomic` is AtomicNone when the reducer is the band's only writer, and
// AtomicAdd when another task still accumulates into the same band and the sum
// has to join it rather than replace it.
//
// `band_width` must be a multiple of 256; `splits` must be at least 1.
template <int64_t RowStride, pto::AtomicType Atomic>
static __aicore__ inline void
reduce_partials_into_band(__gm__ float *part, __gm__ float *out, int64_t splits, int64_t band_width) {
    constexpr int64_t kRows = 16;
    constexpr int64_t kChunk = 256;
    constexpr int64_t kPartialStride = kRows * RowStride;
    constexpr uint64_t kAccUb = 0;
    constexpr uint64_t kTmpUb = kRows * kChunk * sizeof(float);

    using VecTile = pto::Tile<
        pto::TileType::Vec, float, kRows, kChunk, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
        pto::PadValue::Null, pto::CompactMode::Null>;
    using ChunkShape = pto::Shape<1, 1, 1, kRows, kChunk>;
    using ChunkStride = pto::Stride<kPartialStride, kPartialStride, kPartialStride, RowStride, 1>;
    using ChunkTensor = pto::GlobalTensor<float, ChunkShape, ChunkStride, pto::Layout::ND>;

    ChunkShape shape = ChunkShape();
    ChunkStride stride = ChunkStride();

    for (int64_t c = 0; c < band_width; c += kChunk) {
        VecTile acc = VecTile(kRows, kChunk);
        TASSIGN(acc, kAccUb);
        ChunkTensor first = ChunkTensor(part + c, shape, stride);
        TLOAD(acc, first);
        pipe_barrier(PIPE_ALL);

        for (int64_t j = 1; j < splits; ++j) {
            VecTile tmp = VecTile(kRows, kChunk);
            TASSIGN(tmp, kTmpUb);
            ChunkTensor src = ChunkTensor(part + j * kPartialStride + c, shape, stride);
            TLOAD(tmp, src);
            pipe_barrier(PIPE_ALL);
            TADD(acc, acc, tmp);
            pipe_barrier(PIPE_ALL);
        }

        ChunkTensor dst = ChunkTensor(out + c, shape, stride);
        TSTORE<VecTile, ChunkTensor, Atomic>(dst, acc);
        pipe_barrier(PIPE_ALL);
    }
}
