// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// FP32 HC coefficients with compile-time row tiling.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
template <int ROWS>
using Input = Tile<TileType::Vec, float, ROWS, 32, BLayout::RowMajor, DYNAMIC, 32>;
using Small = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
template <int ROWS>
using Output = Tile<TileType::Vec, float, ROWS, 8, BLayout::RowMajor, DYNAMIC, 4>;
template <int ROWS>
using Comb = Tile<TileType::Vec, float, ROWS, 16, BLayout::RowMajor, DYNAMIC, 16>;

__aicore__ inline __gm__ float* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ float*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
template <int ROWS>
__tf__ __aicore__ inline void coefficients(
    typename Output<ROWS>::TileDType __out__ pre, typename Output<ROWS>::TileDType __out__ post, typename Comb<ROWS>::TileDType __out__ comb,
    typename Input<ROWS>::TileDType __in__ p0, typename Input<ROWS>::TileDType __in__ p1,
    typename Input<ROWS>::TileDType __in__ p2, typename Input<ROWS>::TileDType __in__ p3,
    Small::TileDType __in__ inverse, Small::TileDType __in__ bias,
    Small::TileDType __in__ scales, uint16_t rows, float eps, uint16_t iterations) {
    auto a0 = (__ubuf__ float*)__cce_get_tile_ptr(p0);
    auto a1 = (__ubuf__ float*)__cce_get_tile_ptr(p1);
    auto a2 = (__ubuf__ float*)__cce_get_tile_ptr(p2);
    auto a3 = (__ubuf__ float*)__cce_get_tile_ptr(p3);
    auto inv = (__ubuf__ float*)__cce_get_tile_ptr(inverse);
    auto base = (__ubuf__ float*)__cce_get_tile_ptr(bias);
    auto scale = (__ubuf__ float*)__cce_get_tile_ptr(scales);
    auto out_pre = (__ubuf__ float*)__cce_get_tile_ptr(pre);
    auto out_post = (__ubuf__ float*)__cce_get_tile_ptr(post);
    auto out_comb = (__ubuf__ float*)__cce_get_tile_ptr(comb);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float values[24];
        for (int col = 0; col < 24; ++col) {
            int offset = row * 32 + col;
            float total = ((a0[offset] + a1[offset]) + a2[offset]) + a3[offset];
            values[col] = total * inv[row] * scale[col < 4 ? 0 : col < 8 ? 1 : 2] + base[col];
        }
        for (int col = 0; col < 4; ++col) {
            out_pre[row * 8 + col] = 1.0f / (1.0f + std::exp(-values[col])) + eps;
            out_post[row * 8 + col] = 2.0f / (1.0f + std::exp(-values[col + 4]));
        }
        float matrix[4][4];
        for (int r = 0; r < 4; ++r) {
            float maximum = *std::max_element(values + 8 + r * 4, values + 12 + r * 4);
            for (int c = 0; c < 4; ++c) matrix[r][c] = std::exp(values[8 + r * 4 + c] - maximum);
            float sum = (matrix[r][0] + matrix[r][1]) + (matrix[r][2] + matrix[r][3]);
            for (int c = 0; c < 4; ++c) matrix[r][c] = matrix[r][c] / sum + eps;
        }
        for (int iteration = 0; iteration < iterations; ++iteration) {
            if (iteration > 0) {
                for (int r = 0; r < 4; ++r) {
                    float sum = (matrix[r][0] + matrix[r][1]) + (matrix[r][2] + matrix[r][3]) + eps;
                    for (int c = 0; c < 4; ++c) matrix[r][c] /= sum;
                }
            }
            for (int c = 0; c < 4; ++c) {
                float sum = ((matrix[0][c] + matrix[1][c]) + matrix[2][c]) + matrix[3][c] + eps;
                for (int r = 0; r < 4; ++r) matrix[r][c] /= sum;
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) out_comb[row * 16 + r * 4 + c] = matrix[r][c];
        }
    }
#else
    __VEC_SCOPE__ {
        uint32_t count = rows * 4;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        MaskReg full = pset_b32(PAT_ALL);
        RegTensor<uint32_t> lane, token, column, index, pre_index, comb_index;
        vci((vector_s32&)lane, 0, INC_ORDER);
        vshrs(token, lane, 2u, full, MODE_ZEROING);
        RegTensor<uint32_t> low_bits;
        vbr(low_bits, 3u);
        vand(column, lane, low_bits, full, MODE_ZEROING);
        vmuls(index, token, 32u, full, MODE_ZEROING);
        vadd(index, index, column, full, MODE_ZEROING);
        vmuls(pre_index, token, 8u, full, MODE_ZEROING);
        vadd(pre_index, pre_index, column, full, MODE_ZEROING);
        vmuls(comb_index, token, 16u, full, MODE_ZEROING);
        vadd(comb_index, comb_index, column, full, MODE_ZEROING);
        RegTensor<float> s0, s1, s2, rms, one;
        vlds(s0, scale, 0, BRC_B32);
        vlds(s1, scale, 1, BRC_B32);
        vlds(s2, scale, 2, BRC_B32);
        vgather2(rms, inv, token, mask);
        vbr(one, 1.0f);
        RegTensor<float> pre_value;
        {
            RegTensor<float> part, b;
            vgather2(pre_value, a0 + 0, index, mask);
            vgather2(part, a1 + 0, index, mask);
            vadd(pre_value, pre_value, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 0, index, mask);
            vadd(pre_value, pre_value, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 0, index, mask);
            vadd(pre_value, pre_value, part, mask, MODE_ZEROING);
            vmul(pre_value, pre_value, rms, mask, MODE_ZEROING);
            vmul(pre_value, pre_value, s0, mask, MODE_ZEROING);
            vgather2(b, base + 0, column, mask);
            vadd(pre_value, pre_value, b, mask, MODE_ZEROING);
        }
        RegTensor<float> post_value;
        {
            RegTensor<float> part, b;
            vgather2(post_value, a0 + 4, index, mask);
            vgather2(part, a1 + 4, index, mask);
            vadd(post_value, post_value, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 4, index, mask);
            vadd(post_value, post_value, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 4, index, mask);
            vadd(post_value, post_value, part, mask, MODE_ZEROING);
            vmul(post_value, post_value, rms, mask, MODE_ZEROING);
            vmul(post_value, post_value, s1, mask, MODE_ZEROING);
            vgather2(b, base + 4, column, mask);
            vadd(post_value, post_value, b, mask, MODE_ZEROING);
        }
        RegTensor<float> m0;
        {
            RegTensor<float> part, b;
            vgather2(m0, a0 + 8, index, mask);
            vgather2(part, a1 + 8, index, mask);
            vadd(m0, m0, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 8, index, mask);
            vadd(m0, m0, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 8, index, mask);
            vadd(m0, m0, part, mask, MODE_ZEROING);
            vmul(m0, m0, rms, mask, MODE_ZEROING);
            vmul(m0, m0, s2, mask, MODE_ZEROING);
            vgather2(b, base + 8, column, mask);
            vadd(m0, m0, b, mask, MODE_ZEROING);
        }
        RegTensor<float> m1;
        {
            RegTensor<float> part, b;
            vgather2(m1, a0 + 12, index, mask);
            vgather2(part, a1 + 12, index, mask);
            vadd(m1, m1, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 12, index, mask);
            vadd(m1, m1, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 12, index, mask);
            vadd(m1, m1, part, mask, MODE_ZEROING);
            vmul(m1, m1, rms, mask, MODE_ZEROING);
            vmul(m1, m1, s2, mask, MODE_ZEROING);
            vgather2(b, base + 12, column, mask);
            vadd(m1, m1, b, mask, MODE_ZEROING);
        }
        RegTensor<float> m2;
        {
            RegTensor<float> part, b;
            vgather2(m2, a0 + 16, index, mask);
            vgather2(part, a1 + 16, index, mask);
            vadd(m2, m2, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 16, index, mask);
            vadd(m2, m2, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 16, index, mask);
            vadd(m2, m2, part, mask, MODE_ZEROING);
            vmul(m2, m2, rms, mask, MODE_ZEROING);
            vmul(m2, m2, s2, mask, MODE_ZEROING);
            vgather2(b, base + 16, column, mask);
            vadd(m2, m2, b, mask, MODE_ZEROING);
        }
        RegTensor<float> m3;
        {
            RegTensor<float> part, b;
            vgather2(m3, a0 + 20, index, mask);
            vgather2(part, a1 + 20, index, mask);
            vadd(m3, m3, part, mask, MODE_ZEROING);
            vgather2(part, a2 + 20, index, mask);
            vadd(m3, m3, part, mask, MODE_ZEROING);
            vgather2(part, a3 + 20, index, mask);
            vadd(m3, m3, part, mask, MODE_ZEROING);
            vmul(m3, m3, rms, mask, MODE_ZEROING);
            vmul(m3, m3, s2, mask, MODE_ZEROING);
            vgather2(b, base + 20, column, mask);
            vadd(m3, m3, b, mask, MODE_ZEROING);
        }
        vmuls(pre_value, pre_value, -1.0f, mask, MODE_ZEROING);
        vexp(pre_value, pre_value, mask, MODE_ZEROING);
        vadds(pre_value, pre_value, 1.0f, mask, MODE_ZEROING);
        vdiv(pre_value, one, pre_value, mask, MODE_ZEROING);
        vadds(pre_value, pre_value, eps, mask, MODE_ZEROING);
        vscatter(pre_value, out_pre, pre_index, mask);
        vmuls(post_value, post_value, -1.0f, mask, MODE_ZEROING);
        vexp(post_value, post_value, mask, MODE_ZEROING);
        vadds(post_value, post_value, 1.0f, mask, MODE_ZEROING);
        vdiv(post_value, one, post_value, mask, MODE_ZEROING);
        vmuls(post_value, post_value, 2.0f, mask, MODE_ZEROING);
        vscatter(post_value, out_post, pre_index, mask);
        RegTensor<float> red, t0, t1, t2, t3, sum;
        vdintlv(t0, t1, m0, m0);
        vmax(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vmax(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vsub(m0, m0, red, mask);
        vexp(m0, m0, mask, MODE_ZEROING);
        vdintlv(t0, t1, m0, m0);
        vadd(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vadd(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vdiv(m0, m0, red, mask, MODE_ZEROING);
        vadds(m0, m0, eps, mask, MODE_ZEROING);
        vdintlv(t0, t1, m1, m1);
        vmax(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vmax(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vsub(m1, m1, red, mask);
        vexp(m1, m1, mask, MODE_ZEROING);
        vdintlv(t0, t1, m1, m1);
        vadd(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vadd(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vdiv(m1, m1, red, mask, MODE_ZEROING);
        vadds(m1, m1, eps, mask, MODE_ZEROING);
        vdintlv(t0, t1, m2, m2);
        vmax(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vmax(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vsub(m2, m2, red, mask);
        vexp(m2, m2, mask, MODE_ZEROING);
        vdintlv(t0, t1, m2, m2);
        vadd(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vadd(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vdiv(m2, m2, red, mask, MODE_ZEROING);
        vadds(m2, m2, eps, mask, MODE_ZEROING);
        vdintlv(t0, t1, m3, m3);
        vmax(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vmax(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vsub(m3, m3, red, mask);
        vexp(m3, m3, mask, MODE_ZEROING);
        vdintlv(t0, t1, m3, m3);
        vadd(t0, t0, t1, full, MODE_ZEROING);
        vdintlv(t2, t3, t0, t0);
        vadd(t2, t2, t3, full, MODE_ZEROING);
        vintlv(t0, t1, t2, t2);
        vintlv(red, t1, t0, t0);
        vdiv(m3, m3, red, mask, MODE_ZEROING);
        vadds(m3, m3, eps, mask, MODE_ZEROING);
        vadd(sum, m0, m1, mask, MODE_ZEROING);
        vadd(sum, sum, m2, mask, MODE_ZEROING);
        vadd(sum, sum, m3, mask, MODE_ZEROING);
        vadds(sum, sum, eps, mask, MODE_ZEROING);
        vdiv(m0, m0, sum, mask, MODE_ZEROING);
        vdiv(m1, m1, sum, mask, MODE_ZEROING);
        vdiv(m2, m2, sum, mask, MODE_ZEROING);
        vdiv(m3, m3, sum, mask, MODE_ZEROING);
        for (uint16_t iteration = 1; iteration < iterations; ++iteration) {
            vdintlv(t0, t1, m0, m0);
            vadd(t0, t0, t1, full, MODE_ZEROING);
            vdintlv(t2, t3, t0, t0);
            vadd(t2, t2, t3, full, MODE_ZEROING);
            vintlv(t0, t1, t2, t2);
            vintlv(red, t1, t0, t0);
            vadds(red, red, eps, mask, MODE_ZEROING);
            vdiv(m0, m0, red, mask, MODE_ZEROING);
            vdintlv(t0, t1, m1, m1);
            vadd(t0, t0, t1, full, MODE_ZEROING);
            vdintlv(t2, t3, t0, t0);
            vadd(t2, t2, t3, full, MODE_ZEROING);
            vintlv(t0, t1, t2, t2);
            vintlv(red, t1, t0, t0);
            vadds(red, red, eps, mask, MODE_ZEROING);
            vdiv(m1, m1, red, mask, MODE_ZEROING);
            vdintlv(t0, t1, m2, m2);
            vadd(t0, t0, t1, full, MODE_ZEROING);
            vdintlv(t2, t3, t0, t0);
            vadd(t2, t2, t3, full, MODE_ZEROING);
            vintlv(t0, t1, t2, t2);
            vintlv(red, t1, t0, t0);
            vadds(red, red, eps, mask, MODE_ZEROING);
            vdiv(m2, m2, red, mask, MODE_ZEROING);
            vdintlv(t0, t1, m3, m3);
            vadd(t0, t0, t1, full, MODE_ZEROING);
            vdintlv(t2, t3, t0, t0);
            vadd(t2, t2, t3, full, MODE_ZEROING);
            vintlv(t0, t1, t2, t2);
            vintlv(red, t1, t0, t0);
            vadds(red, red, eps, mask, MODE_ZEROING);
            vdiv(m3, m3, red, mask, MODE_ZEROING);
            vadd(sum, m0, m1, mask, MODE_ZEROING);
            vadd(sum, sum, m2, mask, MODE_ZEROING);
            vadd(sum, sum, m3, mask, MODE_ZEROING);
            vadds(sum, sum, eps, mask, MODE_ZEROING);
            vdiv(m0, m0, sum, mask, MODE_ZEROING);
            vdiv(m1, m1, sum, mask, MODE_ZEROING);
            vdiv(m2, m2, sum, mask, MODE_ZEROING);
            vdiv(m3, m3, sum, mask, MODE_ZEROING);
        }
        vscatter(m0, out_comb + 0, comb_index, mask);
        vscatter(m1, out_comb + 4, comb_index, mask);
        vscatter(m2, out_comb + 8, comb_index, mask);
        vscatter(m3, out_comb + 12, comb_index, mask);
    }
#endif
}

template <int ROWS>
__aicore__ inline void sinkhorn(__gm__ int64_t* args) {
    const int tokens = args[7];
    const int padded = args[8];
    const float eps = *reinterpret_cast<__gm__ float*>(&args[9]);
    const int iterations = args[10];
    Small bias(24), scales(3);
    TASSIGN(bias, ROWS == 6 ? 4224 : 8320);
    TASSIGN(scales, ROWS == 6 ? 4352 : 8448);
    using SShape = TileShape2D<float, 1, DYNAMIC>;
    using SStride = BaseShape2D<float, 1, DYNAMIC>;
    GlobalTensor<float, SShape, SStride> bg(data(args, 3), SShape(1, 24), SStride(1, 24));
    GlobalTensor<float, SShape, SStride> sg(data(args, 2), SShape(1, 3), SStride(1, 3));
    TLOAD(bias, bg);
    TLOAD(scales, sg);
    for (int start = get_block_idx(args) * ROWS; start < tokens; start += get_block_num(args) * ROWS) {
        int rows = tokens - start < ROWS ? tokens - start : ROWS;
        Small inverse(rows);
        TASSIGN(inverse, ROWS == 6 ? 4096 : 8192);
        Input<ROWS> p0(rows), p1(rows), p2(rows), p3(rows);
        Output<ROWS> pre(rows), post(rows);
        Comb<ROWS> comb(rows);
        TASSIGN(p0, 0);
        TASSIGN(p1, ROWS == 6 ? 768 : 2048);
        TASSIGN(p2, ROWS == 6 ? 1536 : 4096);
        TASSIGN(p3, ROWS == 6 ? 2304 : 6144);
        TASSIGN(pre, ROWS == 6 ? 4608 : 8576);
        TASSIGN(post, ROWS == 6 ? 4864 : 9088);
        TASSIGN(comb, ROWS == 6 ? 5120 : 9600);
        using Shape = TileShape2D<float, DYNAMIC, 32>;
        using Stride = BaseShape2D<float, DYNAMIC, 32>;
        GlobalTensor<float, Shape, Stride> g0(data(args, 0) + (0 * padded + start) * 32,
                                               Shape(rows, 32), Stride(rows, 32));
        TLOAD(p0, g0);
        GlobalTensor<float, Shape, Stride> g1(data(args, 0) + (1 * padded + start) * 32,
                                               Shape(rows, 32), Stride(rows, 32));
        TLOAD(p1, g1);
        GlobalTensor<float, Shape, Stride> g2(data(args, 0) + (2 * padded + start) * 32,
                                               Shape(rows, 32), Stride(rows, 32));
        TLOAD(p2, g2);
        GlobalTensor<float, Shape, Stride> g3(data(args, 0) + (3 * padded + start) * 32,
                                               Shape(rows, 32), Stride(rows, 32));
        TLOAD(p3, g3);
        GlobalTensor<float, SShape, SStride> ig(data(args, 1) + start, SShape(1, rows), SStride(1, rows));
        TLOAD(inverse, ig);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        coefficients<ROWS>(pre.data(), post.data(), comb.data(), p0.data(), p1.data(), p2.data(), p3.data(),
                     inverse.data(), bias.data(), scales.data(), rows, eps, iterations);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        using OShape = TileShape2D<float, DYNAMIC, 4>;
        using OStride = BaseShape2D<float, DYNAMIC, 4>;
        GlobalTensor<float, OShape, OStride> pg(data(args, 4) + start * 4, OShape(rows, 4), OStride(rows, 4));
        GlobalTensor<float, OShape, OStride> qg(data(args, 5) + start * 4, OShape(rows, 4), OStride(rows, 4));
        using CShape = TileShape2D<float, DYNAMIC, 16>;
        using CStride = BaseShape2D<float, DYNAMIC, 16>;
        GlobalTensor<float, CShape, CStride> cg(data(args, 6) + start * 16, CShape(rows, 16), CStride(rows, 16));
        TSTORE(pg, pre);
        TSTORE(qg, post);
        TSTORE(cg, comb);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    if (args[11] == 6) {
        sinkhorn<6>(args);
    } else {
        sinkhorn<16>(args);
    }
#endif
}
