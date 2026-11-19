// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// MXFP8 projections: four K256 L1 slots, two K128 L0 slots, and resident scales.
#pragma once

#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "intrinsic.h"

namespace mxfp8_projection {

using namespace pto;

constexpr int L1_BUFFERS = 4;
constexpr int L0_BUFFERS = 2;
using F = float8_e4m3_t;
using S = float8_e8m0_t;
using AM = Tile<TileType::Mat, F, 80, 256, BLayout::ColMajor, 80, 256, SLayout::RowMajor, 512>;
template <int NT>
using BM = Tile<TileType::Mat, F, 256, NT, BLayout::ColMajor, 256, NT, SLayout::RowMajor, 512>;
template <int K>
using AS = Tile<TileType::Mat, S, 80, K / 32, BLayout::RowMajor, 80, K / 32, SLayout::RowMajor, 32>;
template <int K, int NT>
using BS = Tile<TileType::Mat, S, K / 32, NT, BLayout::ColMajor, K / 32, NT, SLayout::ColMajor, 32>;
template <int NT>
using AC = TileAcc<float, 80, NT, DYNAMIC, NT>;
template <typename T, int R, int C, int BR, int BC, Layout L>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}

// Two L1 halves, each containing two payload slots.
__aicore__ inline int a_addr(int slot) { return (slot % 2) * 262144 + (slot / 2) * 20480; }
template <int NT>
__aicore__ inline int b_addr(int slot) { return (slot % 2) * 262144 + 40960 + (slot / 2) * (256 * NT); }
template <int K, int N, int NT = 256>
__aicore__ inline void load_payload(__gm__ F* a, __gm__ F* b, int k, int n, int slot) {
    AM am;
    BM<NT> bm;
    TASSIGN(am, a_addr(slot));
    TASSIGN(bm, b_addr<NT>(slot));
    GM<F, 80, 256, 80, K, Layout::ND> ag(a + k);
    GM<F, 256, NT, K, N, Layout::NZ> bg(b + n * K + k * 32);
    TLOAD(am, ag);
    TLOAD(bm, bg);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int K, int INNER, int NT>
__aicore__ inline void compute(AM& am, BM<NT>& bm, AS<K>& sa, BS<K, NT>& sb, AC<NT>& acc, int k) {
    constexpr int slot = INNER % L0_BUFFERS;
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    TileLeft<F, 80, 128, 80, 128> a;
    TileRight<F, 128, NT, 128, NT> b;
    TileLeftScale<S, 80, 4, 80, 4> as;
    TileRightScale<S, 4, NT, 4, NT> bs;
    TASSIGN(a, slot * 32768);
    TASSIGN(b, slot * 32768);
    TASSIGN(as, GetScaleAddr(a.data()));
    TASSIGN(bs, GetScaleAddr(b.data()));
    TEXTRACT(a, am, 0, INNER * 128);
    TEXTRACT(b, bm, INNER * 128, 0);
    TEXTRACT(as, sa, 0, k / 32 + INNER * 4);
    TEXTRACT(bs, sb, k / 32 + INNER * 4, 0);
    set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    if (k == 0 && INNER == 0) {
        TMATMUL_MX(acc, a, as, b, bs);
    } else {
        TMATMUL_MX(acc, acc, a, as, b, bs);
    }
    set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int K, int N, int NT = 256>
__aicore__ inline void project(
    __gm__ F* a, __gm__ F* b, __gm__ S* a_scale, __gm__ S* b_scale,
    __gm__ bfloat16_t* output, int rows, int output_width, int n) {
    AS<K> sa;
    BS<K, NT> sb;
    AC<NT> acc;
    TASSIGN(sa, (40960 + 512 * NT));
    TASSIGN(sb, (40960 + 512 * NT) + 80 * (K / 32));
    TASSIGN(acc, 0);
    acc.SetValidRow(80);
    GM<S, 80, K / 32, 80, K / 32, Layout::MX_A_ZZ> sag(a_scale);
    GM<S, K / 32, NT, K / 32, N, Layout::MX_B_NN> sbg(b_scale + n * (K / 32));
    TLOAD(sa, sag);
    TLOAD(sb, sbg);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    for (int s = 0; s < L0_BUFFERS; ++s) {
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(s));
    }
    for (int s = 0; s < L1_BUFFERS; ++s) {
        load_payload<K, N, NT>(a, b, s * 256, n, s);
    }
    for (int i = 0; i < K / 256; ++i) {
        int slot = i % L1_BUFFERS;
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
        AM am;
        BM<NT> bm;
        TASSIGN(am, a_addr(slot));
        TASSIGN(bm, b_addr<NT>(slot));
        compute<K, 0, NT>(am, bm, sa, sb, acc, i * 256);
        compute<K, 1, NT>(am, bm, sa, sb, acc, i * 256);
        if (i + L1_BUFFERS < K / 256) {
            set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            load_payload<K, N, NT>(a, b, (i + L1_BUFFERS) * 256, n, slot);
        }
    }
    for (int s = 0; s < L0_BUFFERS; ++s) {
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(s));
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    acc.SetValidRow(rows);
    using OutShape = TileShape2D<bfloat16_t, DYNAMIC, NT, Layout::ND>;
    using OutStride = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC, Layout::ND>;
    GlobalTensor<bfloat16_t, OutShape, OutStride, Layout::ND> cg(
        output + n, OutShape(rows, NT), OutStride(rows, output_width));
    TSTORE(cg, acc);
    pipe_barrier(PIPE_ALL);
}

}  // namespace mxfp8_projection
