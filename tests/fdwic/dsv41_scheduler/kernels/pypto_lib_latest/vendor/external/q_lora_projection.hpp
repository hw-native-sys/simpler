// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Q-latent projections with resident A/scales and buffered K/N transfers.
#pragma once
#include "mxfp8_projection.hpp"
using namespace mxfp8_projection;
using MatB = mxfp8_projection::BM<256>;
template <int MT>
using Acc = mxfp8_projection::AC<256, MT>;
using ScaleB = BS<1280, 256>;
template <int MT>
using FullA = Tile<TileType::Mat, F, MT, 1280, BLayout::ColMajor, MT, 1280, SLayout::RowMajor, 512>;

#if defined(__CPU_SIM)
template <int MT>
using QBRopePipe = TPipe<1, Direction::DIR_C2V, MT * 256, 2, 2>;
#endif
template <int MT>
using QBRopeAcc = TileAcc<float, MT, 64>;
template <int MT>
using QBRopeRaw = Tile<TileType::Vec, float, MT, 64>;
template <int MT>
using QBRopeResult = Tile<TileType::Vec, bfloat16_t, MT, 64>;
template <int MT>
using QBRopeAngles = Tile<TileType::Vec, float, MT, 32>;

#if defined(__DAV_CUBE__)
template <bool ENABLED, int MT>
struct QBRopeTransfer {
    __aicore__ inline void finish() {}
};

template <int MT>
struct QBRopeTransfer<true, MT> {
    int transferred = 0;
#if defined(__CPU_SIM)
    QBRopePipe<MT> pipe;
    __aicore__ QBRopeTransfer() : pipe(nullptr, 0, 0) {}
#endif
    __aicore__ inline void push(QBRopeAcc<MT>& tail) {
#if defined(__CPU_SIM)
        TPUSH<QBRopePipe<MT>, QBRopeAcc<MT>, TileSplitAxis::TILE_NO_SPLIT>(pipe, tail);
#else
        int slot = transferred % 2;
        // The vector lane owns both slots before the first transfer.
        wait_intra_block(PIPE_FIX, slot * 2 + 1);
        QBRopeRaw<MT> raw;
        TASSIGN(raw, slot * (MT * 256));
        TMOV<QBRopeRaw<MT>, QBRopeAcc<MT>, AccToVecMode::SingleModeVec0>(raw, tail);
        set_intra_block(PIPE_FIX, slot * 2);
#endif
        ++transferred;
    }
    __aicore__ inline void finish() {
#if !defined(__CPU_SIM)
        if (transferred > 0) {
            // Drain both credits, including an unused second startup slot.
            // Vector may return only after this final Cube acknowledgement.
            wait_intra_block(PIPE_FIX, 1);
            wait_intra_block(PIPE_FIX, 3);
            set_intra_block(PIPE_FIX, 4);
        }
#endif
    }
};

template <int N, int MT>
__aicore__ inline void load_b(__gm__ F* b, int step, int block, int workers, int slot) {
    int n = (step / 5 * workers + block) * 256;
    int k = step % 5 * 256;
    MatB tile;
    TASSIGN(tile, (MT == 80 ? 196608 : 262144) + slot * 65536);
    GM<F, 256, 256, 1280, N, Layout::NZ> gm(b + n * 1280 + k * 32);
    TLOAD(tile, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int INNER, int MT>
__aicore__ inline void compute_qb(FullA<MT>& am, MatB& bm, AS<1280, MT>& sa, ScaleB& sb, Acc<MT>& acc, int k, int wave) {
    constexpr int slot = INNER;
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    TileLeft<F, MT, 128, MT, 128> a;
    TileRight<F, 128, 256, 128, 256> b;
    TileLeftScale<S, MT, 4, MT, 4> as;
    TileRightScale<S, 4, 256, 4, 256> bs;
    TASSIGN(a, slot * 32768);
    TASSIGN(b, slot * 32768);
    TASSIGN(as, GetScaleAddr(a.data()));
    TASSIGN(bs, GetScaleAddr(b.data()));
    TEXTRACT(a, am, 0, k + INNER * 128);
    TEXTRACT(b, bm, INNER * 128, 0);
    TEXTRACT(as, sa, 0, k / 32 + INNER * 4);
    TEXTRACT(bs, sb, k / 32 + INNER * 4, 0);
    set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    if (k == 0 && INNER == 0) {
        wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + wave % 2));
        TMATMUL_MX(acc, a, as, b, bs);
    } else {
        TMATMUL_MX(acc, acc, a, as, b, bs);
    }
    set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int N, int WORKERS = (N / 256 < 32 ? N / 256 : 32), bool OFFSET_INPUT = false, bool FUSED_ROPE = false, int MT = 80>
__aicore__ inline void project_qb(__gm__ int64_t* args, int row_offset = 0, int tile_rows = -1, int lane = -1, int lane_count = 0) {
    // Preserve constant loop bounds for full-width launches.
    if constexpr (WORKERS > 0) {
        if (get_block_num(args) != WORKERS) {
            project_qb<N, 0, OFFSET_INPUT, FUSED_ROPE, MT>(args);
            return;
        }
    }
    // Runtime ABI groups Tensor descriptors before scalar arguments.
    const int rows = tile_rows < 0 ? args[FUSED_ROPE ? 7 : 5] : tile_rows;
    const int row_start = args[FUSED_ROPE ? 9 : 7] + row_offset;
    QBRopeTransfer<FUSED_ROPE, MT> rope_transfer;
    int block = lane < 0 ? get_block_idx(args) : lane;
    int workers = WORKERS;
    int waves;
    if constexpr (WORKERS > 0) {
        waves = N / (256 * WORKERS);
    } else {
        workers = lane_count > 0 ? lane_count : get_block_num(args);
        waves = (N / 256 - block + workers - 1) / workers;
    }
    int input_row = row_offset;
    if constexpr (OFFSET_INPUT) input_row += args[8];
    FullA<MT> am;
    AS<1280, MT> sa;
    TASSIGN(am, 0);
    TASSIGN(sa, MT * 1280);
    GM<F, MT, 1280, MT, 1280, Layout::ND> ag(data<F>(args, 0) + input_row * 1280);
    GM<S, MT, 40, MT, 40, Layout::MX_A_ZZ> sag(data<S>(args, 2) + input_row * 40);
    TLOAD(am, ag);
    TLOAD(sa, sag);
    // Eight scale tiles precede the buffered weights.
    for (int first_wave = 0; first_wave < waves; first_wave += 8) {
        int batch_waves = waves - first_wave < 8 ? waves - first_wave : 8;
        for (int wave = 0; wave < batch_waves; ++wave) {
            ScaleB sb;
            TASSIGN(sb, MT * 1320 + wave * 10240);
            int n = ((first_wave + wave) * workers + block) * 256;
            GM<S, 40, 256, 40, N, Layout::MX_B_NN> sbg(data<S>(args, 3) + n * 40);
            TLOAD(sb, sbg);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        for (int slot = 0; slot < 2; ++slot) {
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        for (int slot = 0; slot < 4; ++slot) {
            load_b<N, MT>(data<F>(args, 1), first_wave * 5 + slot, block, workers, slot);
        }
        for (int step = 0; step < 5 * batch_waves; ++step) {
            int slot = step % 4;
            int wave = step / 5;
            int k = step % 5 * 256;
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
            MatB bm;
            ScaleB sb;
            Acc<MT> acc;
            TASSIGN(bm, (MT == 80 ? 196608 : 262144) + slot * 65536);
            TASSIGN(sb, MT * 1320 + wave * 10240);
            TASSIGN(acc, wave % 2 * (MT * 1024));
            acc.SetValidRow(MT);
            compute_qb<0, MT>(am, bm, sa, sb, acc, k, wave);
            compute_qb<1, MT>(am, bm, sa, sb, acc, k, wave);
            if (step + 4 < 5 * batch_waves) {
                set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                load_b<N, MT>(data<F>(args, 1), first_wave * 5 + step + 4, block, workers, slot);
            }
            if (step % 5 == 4) {
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                acc.SetValidRow(rows);
                using OS = TileShape2D<bfloat16_t, DYNAMIC, 256, Layout::ND>;
                using ST = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC, Layout::ND>;
                int n = ((first_wave + wave) * workers + block) * 256;
                GlobalTensor<bfloat16_t, OS, ST, Layout::ND> cg(
                    data<bfloat16_t>(args, 4) + row_start * N + n, OS(rows, 256), ST(rows, N));
                if constexpr (FUSED_ROPE) {
                    if (n % 512 == 256) {
                        TileAcc<float, MT, 256, DYNAMIC, 192> prefix(rows);
                        TASSIGN(prefix, wave % 2 * (MT * 1024));
                        using PS = TileShape2D<bfloat16_t, DYNAMIC, 192, Layout::ND>;
                        GlobalTensor<bfloat16_t, PS, ST, Layout::ND> prefix_gm(
                            data<bfloat16_t>(args, 4) + row_start * N + n, PS(rows, 192), ST(rows, N));
                        TSTORE(prefix_gm, prefix);
                        QBRopeAcc<MT> tail;
                        TASSIGN(tail, wave % 2 * (MT * 1024) + 192 * MT * sizeof(float));
                        rope_transfer.push(tail);
                    } else {
                        TSTORE(cg, acc);
                    }
                } else {
                    TSTORE(cg, acc);
                }
                set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + wave % 2));
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        pipe_barrier(PIPE_ALL);
    }
    rope_transfer.finish();
}
#endif

#if defined(__DAV_VEC__)
template <int MT>
__tf__ __aicore__ inline void qb_rotate_tail(
    typename QBRopeResult<MT>::TileDType __out__ output, typename QBRopeRaw<MT>::TileDType __in__ input,
    typename QBRopeAngles<MT>::TileDType __in__ cosine, typename QBRopeAngles<MT>::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        for (int pair = 0; pair < 32; ++pair) {
            float even = static_cast<float>(static_cast<bfloat16_t>(src[row * 64 + pair * 2]));
            float odd = static_cast<float>(static_cast<bfloat16_t>(src[row * 64 + pair * 2 + 1]));
            float ec = even * cp[row * 32 + pair];
            float os = odd * sp[row * 32 + pair];
            float es = even * sp[row * 32 + pair];
            float oc = odd * cp[row * 32 + pair];
            dst[row * 64 + pair * 2] = static_cast<bfloat16_t>(ec - os);
            dst[row * 64 + pair * 2 + 1] = static_cast<bfloat16_t>(es + oc);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg half = plt_b32(count, POST_UPDATE);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> values, even, odd, cosv, sinv, a, b, real, imag, rotated, unused;
            vlds(values, src, row * 64, NORM);
            vcvt(bf, values, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(values, bf, full, PART_EVEN);
            vlds(cosv, cp, row * 32, NORM);
            vlds(sinv, sp, row * 32, NORM);
            vdintlv(even, odd, values, values);
            vmul(a, even, cosv, half, MODE_ZEROING);
            vmul(b, odd, sinv, half, MODE_ZEROING);
            vsub(real, a, b, half);
            vmul(a, even, sinv, half, MODE_ZEROING);
            vmul(b, odd, cosv, half, MODE_ZEROING);
            vadd(imag, a, b, half, MODE_ZEROING);
            vintlv(rotated, unused, real, imag);
            vcvt(bf, rotated, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row * 64, PK_B32, full);
        }
    }
#endif
}

template <int MT = 80>
__aicore__ inline void project_qb_rope(__gm__ int64_t* args, int row_offset = 0, int tile_rows = -1, int lane = -1, int lane_count = 0) {
    int block = lane < 0 ? get_block_idx(args) : lane;
    int workers = lane_count > 0 ? lane_count : get_block_num(args);
    int width = args[8];
    int waves = (width / 256 - block + workers - 1) / workers;
    int rows = tile_rows < 0 ? args[7] : tile_rows;
    bool have_tail = false;
    for (int wave = 0; wave < waves; ++wave) {
        if (((wave * workers + block) & 1) != 0) have_tail = true;
    }
    if (!have_tail) return;
#if defined(__CPU_SIM)
    QBRopePipe<MT> pipe(nullptr, 0, 0);
#endif
    QBRopeAngles<MT> cosine, sine;
    QBRopeResult<MT> result;
    TASSIGN(cosine, (MT * 512));
    TASSIGN(sine, (MT * 640));
    TASSIGN(result, (MT * 768));
    if (rows > 0) {
        using Shape = TileShape2D<float, DYNAMIC, 32>;
        using Stride = BaseShape2D<float, DYNAMIC, DYNAMIC>;
        auto cos_tensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
        auto sin_tensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
        Tile<TileType::Vec, float, MT, 32, BLayout::RowMajor, DYNAMIC, 32> c(rows), s(rows);
        TASSIGN(c, (MT * 512));
        TASSIGN(s, (MT * 640));
        GlobalTensor<float, Shape, Stride> cg(data<float>(args, 5) + (args[9] + row_offset) * cos_tensor->strides[0],
            Shape(rows, 32), Stride(rows, cos_tensor->strides[0]));
        GlobalTensor<float, Shape, Stride> sg(data<float>(args, 6) + (args[9] + row_offset) * sin_tensor->strides[0],
            Shape(rows, 32), Stride(rows, sin_tensor->strides[0]));
        TLOAD(c, cg);
        TLOAD(s, sg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }
    int received = 0;
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#if !defined(__CPU_SIM)
    // No-tail lanes returned above. Active lanes lend two row-scaled slots;
    // the Cube consumes every credit and acknowledges their final return.
    set_intra_block(PIPE_V, 1);
    set_intra_block(PIPE_V, 3);
#endif
    for (int wave = 0; wave < waves; ++wave) {
        int n = (wave * workers + block) * 256;
        if (n % 512 != 256) continue;
        int slot = received % 2;
        QBRopeRaw<MT> raw;
#if defined(__CPU_SIM)
        TPOP<QBRopePipe<MT>, QBRopeRaw<MT>, TileSplitAxis::TILE_NO_SPLIT>(pipe, raw);
#else
        TASSIGN(raw, slot * (MT * 256));
        wait_intra_block(PIPE_V, slot * 2);
#endif
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        qb_rotate_tail<MT>(result.data(), raw.data(), cosine.data(), sine.data(), rows);
#if defined(__CPU_SIM)
        TFREE<QBRopePipe<MT>, TileSplitAxis::TILE_NO_SPLIT>(pipe);
#else
        set_intra_block(PIPE_V, slot * 2 + 1);
#endif
        if (rows > 0) {
            using Shape = TileShape2D<bfloat16_t, DYNAMIC, 64>;
            using Stride = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC>;
            Tile<TileType::Vec, bfloat16_t, MT, 64, BLayout::RowMajor, DYNAMIC, 64> active_result(rows);
            TASSIGN(active_result, (MT * 768));
            GlobalTensor<bfloat16_t, Shape, Stride> output(data<bfloat16_t>(args, 4) + (args[9] + row_offset) * width + n + 192,
                Shape(rows, 64), Stride(rows, width));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(output, active_result);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        ++received;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#if !defined(__CPU_SIM)
    wait_intra_block(PIPE_V, 4);
#endif
    pipe_barrier(PIPE_ALL);
}
#endif
