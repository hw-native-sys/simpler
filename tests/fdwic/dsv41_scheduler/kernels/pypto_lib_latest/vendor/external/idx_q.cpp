// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// MXFP8 query projection with two head-wise Cube-to-Vector transfers.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <bit>
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
#if defined(__CPU_SIM)
using SimPipe = TPipe<0, Direction::DIR_C2V, 20480, 2, 2>;
#endif

namespace {
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int C>
using GM = GlobalTensor<T, TileShape2D<T, 1, C>, BaseShape2D<T, 1, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void quantize(
    Vec<uint8_t, 40, 64>::TileDType __out__ packed,
    Vec<int32_t, 40, 32>::TileDType __out__ scales,
    Vec<float, 40, 128>::TileDType __in__ raw,
    Vec<float, 40, 32>::TileDType __in__ cosine,
    Vec<float, 40, 32>::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(raw);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(packed);
    auto exp = (__ubuf__ int32_t*)__cce_get_tile_ptr(scales);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int head = 0; head < rows; ++head) {
        float values[128];
        for (int lane = 0; lane < 128; ++lane) values[lane] = static_cast<float>(static_cast<bfloat16_t>(src[head * 128 + lane]));
        for (int pair = 0; pair < 32; ++pair) {
            float even = values[64 + pair * 2];
            float odd = values[65 + pair * 2];
            values[64 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(even * cp[head * 32 + pair] - odd * sp[head * 32 + pair]));
            values[65 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(even * sp[head * 32 + pair] + odd * cp[head * 32 + pair]));
        }
        for (int group = 0; group < 4; ++group) {
            float maximum = 7.052966104933725e-38f;
            for (int lane = 0; lane < 32; ++lane) maximum = std::max(maximum, std::abs(values[group * 32 + lane]));
            int exponent = (std::bit_cast<int32_t>(maximum * (1.0f / 6.0f)) + 8388607) >> 23;
            exp[head * 32 + group] = exponent;
            float scale = std::bit_cast<float>(exponent << 23);
            for (int pair = 0; pair < 16; ++pair) {
                uint8_t packed_pair = 0;
                for (int half = 0; half < 2; ++half) {
                    float value = values[group * 32 + pair * 2 + half] / scale;
                    float magnitude = std::abs(value);
                    int code = (magnitude > 0.25f) + (magnitude >= 0.75f) + (magnitude > 1.25f) +
                               (magnitude >= 1.75f) + (magnitude > 2.5f) + (magnitude >= 3.5f) + (magnitude > 5.0f);
                    code |= (std::bit_cast<int32_t>(value) >> 28) & 8;
                    packed_pair |= code << (half * 4);
                }
                dst[head * 64 + group * 16 + pair] = packed_pair;
            }
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t half_count = 32;
        MaskReg lower = plt_b32(half_count, POST_UPDATE);
        MaskReg upper;
        pnot(upper, lower, full);
        RegTensor<float> cosv, sinv;
        for (uint16_t head = 0; head < rows; ++head) {
            vlds(cosv, cp, head * 32, NORM);
            vlds(sinv, sp, head * 32, NORM);
            for (uint16_t segment = 0; segment < 2; ++segment) {
                RegTensor<bfloat16_t> bf;
                RegTensor<float> values;
                vlds(values, src, head * 128 + segment * 64, NORM);
                vcvt(bf, values, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vcvt(values, bf, full, PART_EVEN);
                if (segment == 1) {
                    RegTensor<float> even, odd, a, b, real, imag, unused;
                    vdintlv(even, odd, values, values);
                    vmul(a, even, cosv, lower, MODE_ZEROING);
                    vmul(b, odd, sinv, lower, MODE_ZEROING);
                    vsub(real, a, b, lower);
                    vmul(a, even, sinv, lower, MODE_ZEROING);
                    vmul(b, odd, cosv, lower, MODE_ZEROING);
                    vadd(imag, a, b, lower, MODE_ZEROING);
                    vintlv(values, unused, real, imag);
                    vcvt(bf, values, full, ROUND_R, RS_DISABLE, PART_EVEN);
                    vcvt(values, bf, full, PART_EVEN);
                }
                RegTensor<float> magnitude, lo, hi, maximum, scale, normalized;
                vabs(magnitude, values, full);
                vcmax(lo, magnitude, lower);
                vcmax(hi, magnitude, upper);
                vdup(lo, lo, full, POS_LOWEST, MODE_ZEROING);
                vdup(hi, hi, full, POS_LOWEST, MODE_ZEROING);
                vsel(maximum, lo, hi, lower);
                vmaxs(maximum, maximum, 7.052966104933725e-38f, full, MODE_ZEROING);
                vmuls(maximum, maximum, 1.0f / 6.0f, full, MODE_ZEROING);
                RegTensor<int32_t> exponent, bits;
                vadds(bits, (RegTensor<int32_t>&)maximum, 8388607, full, MODE_ZEROING);
                vshrs(exponent, bits, 23, full, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)scale, exponent, 23, full, MODE_ZEROING);
                vdiv(normalized, values, scale, full, MODE_ZEROING);
                vabs(magnitude, normalized, full);
                RegTensor<int32_t> codes, level, sign, sign_mask;
                vdup(codes, 0, full, MODE_ZEROING);
                MaskReg selected;
                vcmps_gt(selected, magnitude, 0.25f, full);
                vdup(level, 1, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_ge(selected, magnitude, 0.75f, full);
                vdup(level, 2, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_gt(selected, magnitude, 1.25f, full);
                vdup(level, 3, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_ge(selected, magnitude, 1.75f, full);
                vdup(level, 4, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_gt(selected, magnitude, 2.5f, full);
                vdup(level, 5, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_ge(selected, magnitude, 3.5f, full);
                vdup(level, 6, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_gt(selected, magnitude, 5.0f, full);
                vdup(level, 7, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vshrs(sign, (RegTensor<int32_t>&)normalized, 28, full, MODE_ZEROING);
                vdup(sign_mask, 8, full, MODE_ZEROING);
                vand(sign, sign, sign_mask, full, MODE_ZEROING);
                vor(codes, codes, sign, full, MODE_ZEROING);
                RegTensor<int32_t> even, odd, bytes;
                vdintlv(even, odd, codes, codes);
                vshls(odd, odd, 4, lower, MODE_ZEROING);
                vor(bytes, even, odd, lower, MODE_ZEROING);
                vsts(bytes, (__ubuf__ int32_t*)(dst + head * 64 + segment * 32), 0, PK4_B32, lower);
                vsts(exponent, exp, head * 32 + segment * 2, ONEPT_B32, full);
                // The second group occupies the upper 32 lanes.
                RegTensor<int32_t> second;
                vdup(second, exponent, full, POS_HIGHEST, MODE_ZEROING);
                vsts(second, exp, head * 32 + segment * 2 + 1, ONEPT_B32, full);
            }
        }
    }
#endif
}
#endif
} // namespace


#if defined(__DAV_CUBE__)
#include "mxfp8_projection.hpp"
namespace {
using F = float8_e4m3_t;
using S = float8_e8m0_t;
using FullA = Tile<TileType::Mat, F, 80, 1280, BLayout::ColMajor, 80, 1280, SLayout::RowMajor, 512>;
using FullB = mxfp8_projection::BM<256>;
using ScaleA = mxfp8_projection::AS<1280>;
using ScaleB = mxfp8_projection::BS<1280, 256>;
using Acc = TileAcc<float, 80, 128>;

template <int INNER>
__aicore__ inline void multiply(FullA& am, FullB& bm, ScaleA& sa, ScaleB& sb, Acc& acc, int k, int head) {
    constexpr int slot = INNER;
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    TileLeft<F, 80, 128> a;
    TileRight<F, 128, 128> b;
    TileLeftScale<S, 80, 4> as;
    TileRightScale<S, 4, 128> bs;
    TASSIGN(a, slot * 32768);
    TASSIGN(b, slot * 32768);
    TASSIGN(as, GetScaleAddr(a.data()));
    TASSIGN(bs, GetScaleAddr(b.data()));
    TEXTRACT(a, am, 0, k + INNER * 128);
    TEXTRACT(b, bm, INNER * 128, head * 128);
    TEXTRACT(as, sa, 0, k / 32 + INNER * 4);
    TEXTRACT(bs, sb, k / 32 + INNER * 4, head * 128);
    set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    if (k == 0 && INNER == 0) TMATMUL_MX(acc, a, as, b, bs);
    else TMATMUL_MX(acc, acc, a, as, b, bs);
    set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
}

__aicore__ inline void project(__gm__ int64_t* args, int block) {
    using namespace mxfp8_projection;
#if defined(__CPU_SIM)
    SimPipe pipe(nullptr, 0, 0);
#endif
    FullA am;
    ScaleA sa;
    ScaleB sb;
    TASSIGN(am, 0);
    TASSIGN(sa, 102400);
    TASSIGN(sb, 105600);
    mxfp8_projection::GM<S, 40, 256, 40, 4096, Layout::MX_B_NN> sbg(::data<S>(args, 3) + block * 256 * 40);
    TLOAD(sb, sbg);
    for (int k = 0; k < 5; ++k) {
        FullB bm;
        TASSIGN(bm, 131072 + k * 65536);
        mxfp8_projection::GM<F, 256, 256, 1280, 4096, Layout::NZ> bg(
            ::data<F>(args, 1) + block * 256 * 1280 + k * 256 * 32);
        TLOAD(bm, bg);
    }
    for (int row = 0; row < args[8]; row += 80) {
        mxfp8_projection::GM<F, 80, 1280, 80, 1280, Layout::ND> ag(::data<F>(args, 0) + row * 1280);
        mxfp8_projection::GM<S, 80, 40, 80, 40, Layout::MX_A_ZZ> sag(::data<S>(args, 2) + row * 40);
        TLOAD(am, ag);
        TLOAD(sa, sag);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        for (int head = 0; head < 2; ++head) {
            Acc acc;
            TASSIGN(acc, head * 40960);
            for (int k = 0; k < 5; ++k) {
                FullB bm;
                TASSIGN(bm, 131072 + k * 65536);
                multiply<0>(am, bm, sa, sb, acc, k * 256, head);
                multiply<1>(am, bm, sa, sb, acc, k * 256, head);
            }
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#if defined(__CPU_SIM)
            TPUSH<SimPipe, Acc, TileSplitAxis::TILE_UP_DOWN>(pipe, acc);
#else
            wait_intra_block(PIPE_FIX, head * 2 + 1);
            wait_intra_block(PIPE_FIX, head * 2 + 17);
            Vec<float, 40, 128> received;
            TASSIGN(received, head * 20480);
            TMOV<decltype(received), Acc, AccToVecMode::DualModeSplitM>(received, acc);
            set_intra_block(PIPE_FIX, head * 2);
            set_intra_block(PIPE_FIX, head * 2 + 16);
#endif
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }
#if !defined(__CPU_SIM)
    if (args[8] > 0) {
        wait_intra_block(PIPE_FIX, 1);
        wait_intra_block(PIPE_FIX, 17);
        wait_intra_block(PIPE_FIX, 3);
        wait_intra_block(PIPE_FIX, 19);
    }
#endif
}
} // namespace
#endif

#if defined(__DAV_VEC__)
namespace {
__aicore__ inline void finish(__gm__ int64_t* args, int block) {
    int sub = get_subblockid();
#if defined(__CPU_SIM)
    SimPipe pipe(nullptr, 0, 0);
#endif
    Vec<float, 40, 32> cosine, sine;
    Vec<uint8_t, 40, 64> packed;
    Vec<int32_t, 40, 32> exponents;
    Vec<uint8_t, 40, 32> scales;
    TASSIGN(cosine, 40960);
    TASSIGN(sine, 46080);
    TASSIGN(packed, 51200);
    TASSIGN(exponents, 53760);
    TASSIGN(scales, 58880);
#if !defined(__CPU_SIM)
    // Both AIV receivers release their UB slots before the first C2V transfer.
    if (args[8] > 0) {
        set_intra_block(PIPE_V, 1);
        set_intra_block(PIPE_V, 3);
    }
#endif
    for (int begin = 0; begin < args[8]; begin += 80) {
        int row = begin + sub * 40;
        int rows = args[8] - row;
        rows = rows < 0 ? 0 : (rows < 40 ? rows : 40);
        if (rows > 0) {
            using Shape = TileShape2D<float, DYNAMIC, 32>;
            using Stride = BaseShape2D<float, DYNAMIC, 32>;
            Tile<TileType::Vec, float, 40, 32, BLayout::RowMajor, DYNAMIC, 32> c(rows), s(rows);
            TASSIGN(c, 40960);
            TASSIGN(s, 46080);
            GlobalTensor<float, Shape, Stride> cg(data<float>(args, 4) + row * 32, Shape(rows, 32), Stride(rows, 32));
            GlobalTensor<float, Shape, Stride> sg(data<float>(args, 5) + row * 32, Shape(rows, 32), Stride(rows, 32));
            TLOAD(c, cg);
            TLOAD(s, sg);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
        for (int head = 0; head < 2; ++head) {
            Vec<float, 40, 128> raw;
            TASSIGN(raw, head * 20480);
#if defined(__CPU_SIM)
            TPOP<SimPipe, decltype(raw), TileSplitAxis::TILE_UP_DOWN>(pipe, raw);
#else
            wait_intra_block(PIPE_V, head * 2);
#endif
            if (rows > 0) {
                quantize(packed.data(), exponents.data(), raw.data(), cosine.data(), sine.data(), rows);
                TCVT(scales, exponents, RoundMode::CAST_RINT);
                Tile<TileType::Vec, uint8_t, 40, 64, BLayout::RowMajor, DYNAMIC, 64> output(rows);
                Tile<TileType::Vec, uint8_t, 40, 32, BLayout::RowMajor, DYNAMIC, 4> scale(rows);
                TASSIGN(output, 51200);
                TASSIGN(scale, 58880);
                using OS = TileShape2D<uint8_t, DYNAMIC, 64>;
                using SS = TileShape2D<uint8_t, DYNAMIC, 4>;
                using ST = BaseShape2D<uint8_t, DYNAMIC, DYNAMIC>;
                GlobalTensor<uint8_t, OS, ST> og(data<uint8_t>(args, 6) + row * 2048 + (block * 2 + head) * 64,
                    OS(rows, 64), ST(rows, 2048));
                GlobalTensor<uint8_t, SS, ST> sg(data<uint8_t>(args, 7) + row * 128 + (block * 2 + head) * 4,
                    SS(rows, 4), ST(rows, 128));
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(og, output);
                TSTORE(sg, scale);
                pipe_barrier(PIPE_ALL);
            }
#if defined(__CPU_SIM)
            TFREE<SimPipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
#else
            set_intra_block(PIPE_V, head * 2 + 1);
#endif
        }
    }
    pipe_barrier(PIPE_ALL);
}
} // namespace
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_atomic_none();
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
#else
    set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
#endif
#endif
#if defined(__DAV_CUBE__)
    for (int block = get_block_idx(args); block < 16; block += get_block_num(args)) {
        project(args, block);
    }
#else
    for (int block = get_block_idx(args); block < 16; block += get_block_num(args)) {
        finish(args, block);
    }
#endif
}
