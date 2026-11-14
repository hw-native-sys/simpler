// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Merge adjacent sparse-attention tile intervals with one sink contribution.
#pragma once
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace sparse_merge {
constexpr int HEADS = 64;
constexpr int HALF = 32;
constexpr int DIM = 512;
constexpr int PARTIAL_STRIDE = HEADS * (DIM + 2);
using State = Tile<TileType::Vec, float, 1, HALF>;
using Values = Tile<TileType::Vec, float, HALF, DIM>;
using Result = Tile<TileType::Vec, bfloat16_t, HALF, DIM>;

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
__tf__ __aicore__ inline void merge_factors(
    State::TileDType __out__ weight_left, State::TileDType __out__ weight_right,
    State::TileDType __in__ max_left, State::TileDType __in__ sum_left,
    State::TileDType __in__ max_right, State::TileDType __in__ sum_right,
    State::TileDType __in__ sinks) {
    auto wl = (__ubuf__ float*)__cce_get_tile_ptr(weight_left);
    auto wr = (__ubuf__ float*)__cce_get_tile_ptr(weight_right);
    auto ml = (__ubuf__ float*)__cce_get_tile_ptr(max_left);
    auto ll = (__ubuf__ float*)__cce_get_tile_ptr(sum_left);
    auto mr = (__ubuf__ float*)__cce_get_tile_ptr(max_right);
    auto lr = (__ubuf__ float*)__cce_get_tile_ptr(sum_right);
    auto sk = (__ubuf__ float*)__cce_get_tile_ptr(sinks);
    __VEC_SCOPE__ {
        uint32_t count = HALF;
        MaskReg heads = CreatePredicate<float>(count);
        RegTensor<float> m0, m1, l0, l1, sink, maximum, a, b, den;
        vlds(m0, ml, 0, NORM);
        vlds(m1, mr, 0, NORM);
        vlds(l0, ll, 0, NORM);
        vlds(l1, lr, 0, NORM);
        vlds(sink, sk, 0, NORM);
        vmax(maximum, m0, m1, heads, MODE_ZEROING);
        vmax(maximum, maximum, sink, heads, MODE_ZEROING);
        vsub(a, m0, maximum, heads, MODE_ZEROING);
        vsub(b, m1, maximum, heads, MODE_ZEROING);
        vexp(a, a, heads, MODE_ZEROING);
        vexp(b, b, heads, MODE_ZEROING);
        vmul(l0, l0, a, heads, MODE_ZEROING);
        vmul(l1, l1, b, heads, MODE_ZEROING);
        vadd(den, l0, l1, heads, MODE_ZEROING);
        vsub(sink, sink, maximum, heads, MODE_ZEROING);
        vexp(sink, sink, heads, MODE_ZEROING);
        vadd(den, den, sink, heads, MODE_ZEROING);
        vdiv(a, a, den, heads, MODE_ZEROING);
        vdiv(b, b, den, heads, MODE_ZEROING);
        vsts(a, wl, 0, NORM_B32, heads);
        vsts(b, wr, 0, NORM_B32, heads);
    }
}

__tf__ __aicore__ inline void merge_values(
    Result::TileDType __out__ result, Values::TileDType __in__ left, Values::TileDType __in__ right,
    State::TileDType __in__ left_weight, State::TileDType __in__ right_weight) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(result);
    auto a = (__ubuf__ float*)__cce_get_tile_ptr(left);
    auto b = (__ubuf__ float*)__cce_get_tile_ptr(right);
    auto wa = (__ubuf__ float*)__cce_get_tile_ptr(left_weight);
    auto wb = (__ubuf__ float*)__cce_get_tile_ptr(right_weight);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < HALF; ++row) {
            RegTensor<float> alpha, beta;
            vlds(alpha, wa, row, BRC_B32);
            vlds(beta, wb, row, BRC_B32);
            for (uint16_t col = 0; col < DIM; col += 64) {
                RegTensor<float> x, y;
                RegTensor<bfloat16_t> out;
                vlds(x, a, row * DIM + col, NORM);
                vlds(y, b, row * DIM + col, NORM);
                vmul(x, x, alpha, all, MODE_ZEROING);
                vmul(y, y, beta, all, MODE_ZEROING);
                vadd(x, x, y, all, MODE_ZEROING);
                vcvt(out, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(out, dst, row * DIM + col, PK_B32, all);
            }
        }
    }
}

__aicore__ inline void reduce(
    __gm__ float* partials, __gm__ bfloat16_t* output, __gm__ float* sink_values,
    int64_t output_stride, int64_t sink_stride, int heads, int boundary, int head, int token) {
    const int valid = heads - head < HALF ? heads - head : HALF;
    auto left_base = partials + (int64_t)(2 * boundary - 1) * PARTIAL_STRIDE;
    auto right_base = partials + (int64_t)(2 * boundary) * PARTIAL_STRIDE;
    State ml, ll, mr, lr, sink, wl, wr;
    TASSIGN(ml, 0);
    TASSIGN(ll, 128);
    TASSIGN(mr, 256);
    TASSIGN(lr, 384);
    TASSIGN(sink, 512);
    TASSIGN(wl, 640);
    TASSIGN(wr, 768);
    using GState = GlobalTensor<float, TileShape2D<float, 1, HALF>, BaseShape2D<float, 1, HALF>>;
    GState ml_gm(left_base + HEADS * DIM + head);
    GState ll_gm(left_base + HEADS * (DIM + 1) + head);
    GState mr_gm(right_base + HEADS * DIM + head);
    GState lr_gm(right_base + HEADS * (DIM + 1) + head);
    TLOAD(ml, ml_gm);
    TLOAD(ll, ll_gm);
    TLOAD(mr, mr_gm);
    TLOAD(lr, lr_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (sink_stride == 1) {
        Tile<TileType::Vec, float, 1, HALF, BLayout::RowMajor, 1, DYNAMIC> sink_load(valid);
        TASSIGN(sink_load, 512);
        using SinkShape = TileShape2D<float, 1, DYNAMIC>;
        GlobalTensor<float, SinkShape, BaseShape2D<float, 1, HEADS>> sink_gm(
            sink_values + head, SinkShape(1, valid));
        TLOAD(sink_load, sink_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        for (int i = 0; i < HALF; ++i) {
            sink.SetValue(i, i < valid ? sink_values[(head + i) * sink_stride] : 0.0f);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }
    merge_factors(wl.data(), wr.data(), ml.data(), ll.data(), mr.data(), lr.data(), sink.data());
    Values left, right;
    Result result;
    TASSIGN(left, 8192);
    TASSIGN(right, 73728);
    TASSIGN(result, 139264);
    for (int channel = 0; channel < 1; ++channel) {
        using InputShape = TileShape2D<float, HALF, DIM>;
        using InputBase = BaseShape2D<float, HEADS, DIM>;
        GlobalTensor<float, InputShape, InputBase> left_gm(left_base + head * DIM + channel * DIM);
        GlobalTensor<float, InputShape, InputBase> right_gm(right_base + head * DIM + channel * DIM);
        TLOAD(left, left_gm);
        TLOAD(right, right_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        merge_values(result.data(), left.data(), right.data(), wl.data(), wr.data());
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        Tile<TileType::Vec, bfloat16_t, HALF, DIM, BLayout::RowMajor, DYNAMIC, DIM> active(valid);
        TASSIGN(active, 139264);
        using OutShape = TileShape2D<bfloat16_t, DYNAMIC, DIM>;
        using OutBase = BaseShape2D<bfloat16_t, HEADS, DIM>;
        GlobalTensor<bfloat16_t, OutShape, OutBase> dst(
            output + token * output_stride + head * DIM + channel * DIM, OutShape(valid, DIM));
        TSTORE(dst, active);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
}
