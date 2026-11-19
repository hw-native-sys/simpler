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
template <int K, int NT, int MT>
constexpr int L1Slots = (MT + NT) * (512 + K / 32) <= 262144 ? 4 : 2;
using F = float8_e4m3_t;
using S = float8_e8m0_t;
template <int MT = 80>
using AM = Tile<TileType::Mat, F, MT, 256, BLayout::ColMajor, MT, 256, SLayout::RowMajor, 512>;
template <int NT>
using BM = Tile<TileType::Mat, F, 256, NT, BLayout::ColMajor, 256, NT, SLayout::RowMajor, 512>;
template <int K, int MT = 80>
using AS = Tile<TileType::Mat, S, MT, K / 32, BLayout::RowMajor, MT, K / 32, SLayout::RowMajor, 32>;
template <int K, int NT>
using BS = Tile<TileType::Mat, S, K / 32, NT, BLayout::ColMajor, K / 32, NT, SLayout::ColMajor, 32>;
template <int NT, int MT = 80>
using AC = TileAcc<float, MT, NT, DYNAMIC, NT>;
template <typename T, int R, int C, int BR, int BC, Layout L>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}

// Two L1 halves, each containing two payload slots.
template <int MT = 80>
__aicore__ inline int a_addr(int slot) { return (slot % 2) * 262144 + (slot / 2) * (MT * 256); }
template <int NT, int MT = 80, int BUFFERS = 4>
__aicore__ inline int b_addr(int slot) { return (slot % 2) * 262144 + (BUFFERS / 2) * MT * 256 + (slot / 2) * (256 * NT); }
template <int K, int N, int NT = 256, int MT = 80>
__aicore__ inline void load_payload(__gm__ F* a, __gm__ F* b, int k, int n, int slot) {
    AM<MT> am;
    BM<NT> bm;
    TASSIGN(am, a_addr<MT>(slot));
    TASSIGN(bm, b_addr<NT, MT, L1Slots<K, NT, MT>>(slot));
    GM<F, MT, 256, MT, K, Layout::ND> ag(a + k);
    GM<F, 256, NT, K, N, Layout::NZ> bg(b + n * K + k * 32);
    TLOAD(am, ag);
    TLOAD(bm, bg);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int K, int INNER, int NT, int MT>
__aicore__ inline void compute(AM<MT>& am, BM<NT>& bm, AS<K, MT>& sa, BS<K, NT>& sb, AC<NT, MT>& acc, int k) {
    constexpr int slot = INNER % L0_BUFFERS;
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    TileLeft<F, MT, 128, MT, 128> a;
    TileRight<F, 128, NT, 128, NT> b;
    TileLeftScale<S, MT, 4, MT, 4> as;
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

template <int K, int N, int NT, int MT = 80, typename Emit>
__aicore__ inline void project_emit(
    __gm__ F* a, __gm__ F* b, __gm__ S* a_scale, __gm__ S* b_scale,
    int n, Emit& emit) {
    AS<K, MT> sa;
    BS<K, NT> sb;
    AC<NT, MT> acc;
    TASSIGN(sa, ((L1Slots<K, NT, MT> / 2) * (MT + NT) * 256));
    TASSIGN(sb, ((L1Slots<K, NT, MT> / 2) * (MT + NT) * 256) + MT * (K / 32));
    TASSIGN(acc, 0);
    acc.SetValidRow(MT);
    GM<S, MT, K / 32, MT, K / 32, Layout::MX_A_ZZ> sag(a_scale);
    GM<S, K / 32, NT, K / 32, N, Layout::MX_B_NN> sbg(b_scale + n * (K / 32));
    TLOAD(sa, sag);
    TLOAD(sb, sbg);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    for (int s = 0; s < L0_BUFFERS; ++s) {
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(s));
    }
    for (int s = 0; s < L1Slots<K, NT, MT>; ++s) {
        load_payload<K, N, NT, MT>(a, b, s * 256, n, s);
    }
    for (int i = 0; i < K / 256; ++i) {
        int slot = i % L1Slots<K, NT, MT>;
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
        AM<MT> am;
        BM<NT> bm;
        TASSIGN(am, a_addr<MT>(slot));
        TASSIGN(bm, b_addr<NT, MT, L1Slots<K, NT, MT>>(slot));
        compute<K, 0, NT, MT>(am, bm, sa, sb, acc, i * 256);
        compute<K, 1, NT, MT>(am, bm, sa, sb, acc, i * 256);
        if (i + L1Slots<K, NT, MT> < K / 256) {
            set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            load_payload<K, N, NT, MT>(a, b, (i + L1Slots<K, NT, MT>) * 256, n, slot);
        }
    }
    for (int s = 0; s < L0_BUFFERS; ++s) {
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(s));
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    emit(acc);
    pipe_barrier(PIPE_ALL);
}

template <int NT, int MT = 80>
struct StoreOutput {
    __gm__ bfloat16_t* output;
    int rows, output_width, n;

    __aicore__ inline void operator()(AC<NT, MT>& acc) {
        acc.SetValidRow(rows);
        using OutShape = TileShape2D<bfloat16_t, DYNAMIC, NT, Layout::ND>;
        using OutStride = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC, Layout::ND>;
        GlobalTensor<bfloat16_t, OutShape, OutStride, Layout::ND> cg(
            output + n, OutShape(rows, NT), OutStride(rows, output_width));
        TSTORE(cg, acc);
    }
};

template <int K, int N, int NT = 256, int MT = 80>
__aicore__ inline void project(
    __gm__ F* a, __gm__ F* b, __gm__ S* a_scale, __gm__ S* b_scale,
    __gm__ bfloat16_t* output, int rows, int output_width, int n) {
    StoreOutput<NT, MT> emit{output, rows, output_width, n};
    project_emit<K, N, NT, MT>(a, b, a_scale, b_scale, n, emit);
}

}  // namespace mxfp8_projection
