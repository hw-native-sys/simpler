// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// O-A with BF16 rounding and OCP MXFP8 quantization on the paired Vector cores.
#include "mxfp8_projection.hpp"
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <cstring>
#endif

using namespace mxfp8_projection;
namespace {
template <int MT>
using Raw = Tile<TileType::Vec, float, MT, 128>;
template <int MT>
using Payload = Tile<TileType::Vec, F, MT, 128>;
template <int MT>
using Scales = Tile<TileType::Vec, uint8_t, MT / 16, 64>;
constexpr int RAW_UB = 0;
constexpr int READY = 0, RELEASE = 1, FINISH = 2, STORE_DONE = 3, VEC_OFFSET = 16;
#if defined(__CPU_SIM)
template <int MT>
using SimPipe = TPipe<0, Direction::DIR_C2V, MT * 128 * 4, 1, 2>;
#endif

#if defined(__DAV_CUBE__)
template <int MT>
struct QuantizeOutput {
#if defined(__CPU_SIM)
    SimPipe<MT> pipe;
    __aicore__ QuantizeOutput() : pipe(nullptr, 0, 0) {}
#endif
    __aicore__ inline void operator()(AC<256, MT>& acc) {
#if defined(__CPU_SIM)
        TPUSH<SimPipe<MT>, AC<256, MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, acc);
#else
        wait_intra_block(PIPE_FIX, RELEASE);
        wait_intra_block(PIPE_FIX, RELEASE + VEC_OFFSET);
        Raw<MT> raw;
        TASSIGN(raw, RAW_UB);
        TMOV<Raw<MT>, AC<256, MT>, AccToVecMode::DualModeSplitN>(raw, acc);
        set_intra_block(PIPE_FIX, READY);
        set_intra_block(PIPE_FIX, READY + VEC_OFFSET);
#endif
    }
    __aicore__ inline void finish() {
#if !defined(__CPU_SIM)
        wait_intra_block(PIPE_FIX, RELEASE);
        wait_intra_block(PIPE_FIX, RELEASE + VEC_OFFSET);
        wait_intra_block(PIPE_FIX, STORE_DONE);
        wait_intra_block(PIPE_FIX, STORE_DONE + VEC_OFFSET);
        set_intra_block(PIPE_FIX, FINISH);
        set_intra_block(PIPE_FIX, FINISH + VEC_OFFSET);
#endif
    }
};
#endif

#if defined(__DAV_VEC__)
template <int MT>
__tf__ __aicore__ inline void quantize(
    typename Payload<MT>::TileDType __out__ output, typename Scales<MT>::TileDType __out__ scales,
    typename Raw<MT>::TileDType __in__ input, uint16_t active_rows) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto codes = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
#if defined(__CPU_SIM)
    for (int row = 0; row < MT; ++row) {
        for (int group = 0; group < 4; ++group) {
            float values[32];
            float maximum = 0;
            for (int i = 0; i < 32; ++i) {
                values[i] = row < active_rows ? static_cast<float>(static_cast<bfloat16_t>(src[row * 128 + group * 32 + i])) : 0.0f;
                maximum = std::max(maximum, std::abs(values[i]));
            }
            uint32_t bits;
            std::memcpy(&bits, &maximum, 4);
            int exponent = std::max(0, static_cast<int>(bits >> 23) - 8);
            float reciprocal = exponent ? std::ldexp(1.0f, 127 - exponent) : 0.0f;
            for (int i = 0; i < 32; ++i) {
                F value(std::max(-448.0f, std::min(448.0f, values[i] * reciprocal)), RoundMode::CAST_RINT);
                std::memcpy(dst + row * 128 + group * 32 + i, &value, 1);
            }
            reinterpret_cast<__ubuf__ uint8_t*>(codes)[(row / 16) * 64 + (group / 2) * 32 + row % 16 * 2 + group % 2] = exponent;
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t count64 = 64, count1 = 1;
        MaskReg bytes64 = CreatePredicate<uint8_t>(count64);
        MaskReg half1 = CreatePredicate<uint16_t>(count1);
        RegTensor<uint32_t> lane, lo, hi, tmp, ix1, ix2, ix4, ix8, ix16;
        RegTensor<int32_t> izero, ione;
        RegTensor<float> low, high;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vdup(lo, 0u, all, MODE_ZEROING); vdup(hi, 32u, all, MODE_ZEROING);
        vdup(izero, 0, all, MODE_ZEROING); vdup(ione, 1, all, MODE_ZEROING);
        vdup(low, -448.0f, all, MODE_ZEROING); vdup(high, 448.0f, all, MODE_ZEROING);
        vshrs(tmp, lane, 1u, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vsub(tmp, lane, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix1, lane, 1u, all, MODE_ZEROING); vsub(ix1, ix1, tmp, all, MODE_ZEROING);
        vshrs(ix2, lane, 1u, all, MODE_ZEROING); vshls(ix2, ix2, 1u, all, MODE_ZEROING);
        vshrs(tmp, lane, 2u, all, MODE_ZEROING); vshls(tmp, tmp, 2u, all, MODE_ZEROING);
        vsub(tmp, ix2, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix2, lane, 2u, all, MODE_ZEROING); vsub(ix2, ix2, tmp, all, MODE_ZEROING);
        vshrs(ix4, lane, 2u, all, MODE_ZEROING); vshls(ix4, ix4, 2u, all, MODE_ZEROING);
        vshrs(tmp, lane, 3u, all, MODE_ZEROING); vshls(tmp, tmp, 3u, all, MODE_ZEROING);
        vsub(tmp, ix4, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix4, lane, 4u, all, MODE_ZEROING); vsub(ix4, ix4, tmp, all, MODE_ZEROING);
        vshrs(ix8, lane, 3u, all, MODE_ZEROING); vshls(ix8, ix8, 3u, all, MODE_ZEROING);
        vshrs(tmp, lane, 4u, all, MODE_ZEROING); vshls(tmp, tmp, 4u, all, MODE_ZEROING);
        vsub(tmp, ix8, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix8, lane, 8u, all, MODE_ZEROING); vsub(ix8, ix8, tmp, all, MODE_ZEROING);
        vshrs(ix16, lane, 4u, all, MODE_ZEROING); vshls(ix16, ix16, 4u, all, MODE_ZEROING);
        vshrs(tmp, lane, 5u, all, MODE_ZEROING); vshls(tmp, tmp, 5u, all, MODE_ZEROING);
        vsub(tmp, ix16, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix16, lane, 16u, all, MODE_ZEROING); vsub(ix16, ix16, tmp, all, MODE_ZEROING);
        RegTensor<uint8_t> byte_index;
        vci((RegTensor<int8_t>&)byte_index, (int8_t)0, INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_index, (RegTensor<int16_t>&)byte_index, (int16_t)4, all8);
        for (uint16_t row = 0; row < MT; ++row) {
            for (uint16_t chunk = 0; chunk < 2; ++chunk) {
                RegTensor<float> x, amax, shuffled, reciprocal;
                RegTensor<int32_t> exponent, nonzero, reciprocal_bits;
                if (row < active_rows) {
                    RegTensor<bfloat16_t> rounded;
                    vlds(x, src, row * 128 + chunk * 64, NORM);
                    vcvt(rounded, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
                    vcvt(x, rounded, all16, PART_EVEN);
                } else {
                    vdup(x, 0.0f, all, MODE_ZEROING);
                }
                vabs(amax, x, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix16); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix8); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix4); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix2); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix1); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vshrs(exponent, (RegTensor<int32_t>&)amax, 23, all, MODE_ZEROING);
                vadds(exponent, exponent, -8, all, MODE_ZEROING);
                vmax(exponent, exponent, izero, all, MODE_ZEROING);
                vmin(nonzero, exponent, ione, all, MODE_ZEROING);
                vmuls(reciprocal_bits, exponent, -1, all, MODE_ZEROING);
                vadds(reciprocal_bits, reciprocal_bits, 254, all, MODE_ZEROING);
                vmul(reciprocal_bits, reciprocal_bits, nonzero, all, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)reciprocal, reciprocal_bits, 23, all, MODE_ZEROING);
                vmul(x, x, reciprocal, all, MODE_ZEROING);
                vmax(x, x, low, all, MODE_ZEROING); vmin(x, x, high, all, MODE_ZEROING);
                vector_f8e4m3 fp8;
                RegTensor<uint8_t> packed;
                vcvt(fp8, x, all, ROUND_R, RS_DISABLE, PART_P0);
                vselr(packed, (RegTensor<uint8_t>&)fp8, byte_index);
                vsts(packed, dst, row * 128 + chunk * 64, NORM_B8, bytes64);
                RegTensor<uint32_t> e0, e1;
                vselr(e0, (RegTensor<uint32_t>&)exponent, lo); vselr(e1, (RegTensor<uint32_t>&)exponent, hi);
                vshls(e1, e1, 8u, all, MODE_ZEROING); vor(e0, e0, e1, all);
                vsts((RegTensor<uint16_t>&)e0, codes, (row / 16) * 32 + chunk * 16 + row % 16, ONEPT_B16, half1);
            }
        }
    }
#endif
}
#endif
} // namespace

template <int MT>
__aicore__ inline void project_oa(__gm__ int64_t* args) {
    const int width = args[6];
    const int active = args[7];
    const int workers = get_block_num(args);
    const int padded_rows = args[8];
    const int columns = width / 256;
#if defined(__DAV_CUBE__)
    QuantizeOutput<MT> emit;
    for (int tile = get_block_idx(args); tile < (padded_rows / MT) * columns; tile += workers) {
        const int row = tile / columns * MT;
        const int column = tile % columns;
        int group = column / 4;
        project_emit<4096, 1024, 256, MT>(
            data<F>(args, 0) + (group * padded_rows + row) * 4096,
            data<F>(args, 1) + group * 4096 * 1024,
            data<S>(args, 2) + (group * padded_rows + row) * 128,
            data<S>(args, 3) + group * 128 * 1024, column % 4 * 256, emit);
    }
    emit.finish();
#elif defined(__DAV_VEC__)
    Raw<MT> raw;
    Payload<MT> payload;
    Scales<MT> scales;
    TASSIGN(raw, RAW_UB);
    TASSIGN(payload, MT * 128 * 4);
    TASSIGN(scales, MT * 128 * 5);
#if defined(__CPU_SIM)
    SimPipe<MT> pipe(nullptr, 0, 0);
#else
    set_intra_block(PIPE_V, RELEASE);
#endif
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int tile = get_block_idx(args); tile < (padded_rows / MT) * columns; tile += workers) {
#if defined(__CPU_SIM)
        TPOP<SimPipe<MT>, Raw<MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, raw);
#else
        wait_intra_block(PIPE_V, READY);
#endif
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        const int row = tile / columns * MT;
        const int column = tile % columns;
        const int valid = active - row < 0 ? 0 : (active - row > MT ? MT : active - row);
        quantize<MT>(payload.data(), scales.data(), raw.data(), valid);
#if defined(__CPU_SIM)
        TFREE<SimPipe<MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe);
#else
        set_intra_block(PIPE_V, RELEASE);
#endif
        const int n = column * 256 + get_sub_block_id(args) * 128;
        using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        using PayloadShape = TileShape2D<F, MT, 128>;
        using ScaleShape = TileShape2D<uint8_t, MT / 16, 64>;
        const int64_t payload_stride = reinterpret_cast<__gm__ Tensor*>(args[4])->strides[0];
        GlobalTensor<F, PayloadShape, Stride> pg(data<F>(args, 4) + row * payload_stride + n,
            PayloadShape(), Stride(1, 1, 1, payload_stride, 1));
        GlobalTensor<uint8_t, ScaleShape, Stride> sg(data<uint8_t>(args, 5) + row * (width / 32) + n / 64 * 32,
            ScaleShape(), Stride(1, 1, 1, width / 2, 1));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(pg, payload); TSTORE(sg, scales);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#if !defined(__CPU_SIM)
    set_intra_block(PIPE_MTE3, STORE_DONE);
    wait_intra_block(PIPE_V, FINISH);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    if (args[9] == 128) {
        project_oa<128>(args);
    } else {
        project_oa<80>(args);
    }
}
