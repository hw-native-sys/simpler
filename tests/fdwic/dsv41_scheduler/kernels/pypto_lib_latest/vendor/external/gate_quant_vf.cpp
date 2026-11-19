// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Group-32 MXFP8 activation quantization and MX_A_ZZ scale packing.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
constexpr int Rows = 16;
constexpr int Columns = 1024;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using Input = Vec<bfloat16_t, Rows, Columns>;
using Payload = Vec<uint8_t, Rows, Columns>;
using Scales = Vec<uint8_t, 1, Rows * Columns / 32>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__) && defined(PTO_NPU_ARCH_A5)
__tf__ __aicore__ inline void quantize(
    Payload::TileDType __out__ payload, Scales::TileDType __out__ scales,
    Input::TileDType __in__ input) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(payload);
    auto packed_scales = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t count64 = 64, count1 = 1;
        MaskReg bytes64 = CreatePredicate<uint8_t>(count64);
        MaskReg half1 = CreatePredicate<uint16_t>(count1);
        RegTensor<uint32_t> lane, bits, ix1, ix2, ix4, ix8, ix16, lo, hi;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vdup(bits, 1u, all, MODE_ZEROING); vxor(ix1, lane, bits, all);
        vdup(bits, 2u, all, MODE_ZEROING); vxor(ix2, lane, bits, all);
        vdup(bits, 4u, all, MODE_ZEROING); vxor(ix4, lane, bits, all);
        vdup(bits, 8u, all, MODE_ZEROING); vxor(ix8, lane, bits, all);
        vdup(bits, 16u, all, MODE_ZEROING); vxor(ix16, lane, bits, all);
        vdup(lo, 0u, all, MODE_ZEROING); vdup(hi, 32u, all, MODE_ZEROING);
        RegTensor<uint8_t> byte_index;
        vci((RegTensor<int8_t>&)byte_index, static_cast<int8_t>(0), INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_index, (RegTensor<int16_t>&)byte_index,
              static_cast<int16_t>(4), all8);
        RegTensor<float> floor, lower, upper;
        vdup(floor, 1e-4f, all, MODE_ZEROING);
        vdup(lower, -448.0f, all, MODE_ZEROING);
        vdup(upper, 448.0f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < Rows; ++row) {
            for (uint16_t segment = 0; segment < Columns / 64; ++segment) {
                RegTensor<float> x, maximum, shuffled, divisor;
                RegTensor<int32_t> exponent;
                RegTensor<bfloat16_t> raw;
                vlds(raw, src, row * Columns + segment * 64, UNPK_B16);
                vcvt(x, raw, all, PART_EVEN);
                vabs(maximum, x, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix16);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix8);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix4);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix2);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix1);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vmax(maximum, maximum, floor, all, MODE_ZEROING);
                vmuls(maximum, maximum, 1.0f / 448.0f, all, MODE_ZEROING);
                vadds(exponent, (RegTensor<int32_t>&)maximum, 8388607, all, MODE_ZEROING);
                vshrs(exponent, exponent, 23, all, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)divisor, exponent, 23, all, MODE_ZEROING);
                vdiv(x, x, divisor, all, MODE_ZEROING);
                vmax(x, x, lower, all, MODE_ZEROING); vmin(x, x, upper, all, MODE_ZEROING);
                vector_f8e4m3 converted;
                RegTensor<uint8_t> packed;
                vcvt(converted, x, all, ROUND_R, RS_DISABLE, PART_P0);
                vselr(packed, (RegTensor<uint8_t>&)converted, byte_index);
                vsts(packed, dst, row * Columns + segment * 64, NORM_B8, bytes64);
                RegTensor<uint32_t> e0, e1;
                vselr(e0, (RegTensor<uint32_t>&)exponent, lo);
                vselr(e1, (RegTensor<uint32_t>&)exponent, hi);
                vshls(e1, e1, 8u, all, MODE_ZEROING); vor(e0, e0, e1, all);
                vsts((RegTensor<uint16_t>&)e0, packed_scales, segment * Rows + row, ONEPT_B16, half1);
            }
        }
    }
}

__aicore__ inline void run_native(__gm__ int64_t* args, int block) {
    int width = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[1];
    int chunks = width / Columns;
    int first = block / chunks * Rows, column = block % chunks * Columns;
    Input input;
    Payload payload;
    Scales scales;
    TASSIGN(input, 0); TASSIGN(payload, 65536); TASSIGN(scales, 81920);
    using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, Rows, Columns>, Stride> input_gm(
        data<bfloat16_t>(args, 0) + first * width + column,
        TileShape2D<bfloat16_t, Rows, Columns>(), Stride(1, 1, 1, width, 1));
    TLOAD(input, input_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    quantize(payload.data(), scales.data(), input.data());
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GlobalTensor<uint8_t, TileShape2D<uint8_t, Rows, Columns>, Stride> payload_gm(
        data<uint8_t>(args, 1) + first * width + column,
        TileShape2D<uint8_t, Rows, Columns>(), Stride(1, 1, 1, width, 1));
    using ScaleShape = TileShape2D<uint8_t, 1, Rows * Columns / 32>;
    GlobalTensor<uint8_t, ScaleShape, BaseShape2D<uint8_t, 1, Rows * Columns / 32>> scale_gm(
        data<uint8_t>(args, 2) + first * (width / 32) + (column / 32) * Rows);
    TSTORE(payload_gm, payload); TSTORE(scale_gm, scales);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_VEC__) && !defined(PTO_NPU_ARCH_A5)
__aicore__ inline void run_tiles(__gm__ int64_t* args, int block) {
    int width = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[1];
    int chunks = width / Columns;
    int first = block / chunks * Rows, column = block % chunks * Columns;
    using Chunk = Vec<float, Rows, 256>;
    using Groups = Vec<float, 128, 32>;
    using Maximum = Tile<TileType::Vec, float, 128, 1, BLayout::ColMajor>;
    using Integers = Vec<int32_t, 1, 128>;
    Chunk input;
    Groups groups, absolute, normalized, max_scratch;
    Maximum maximum, scale;
    Vec<float, 1, 128> maximum_row;
    Vec<float, 1, 128> signed_float;
    Integers exponent, scale_bits, signed_code, shifted;
    Vec<half, 1, 128> half_code;
    Vec<int16_t, 1, 128> short_code;
    Vec<int8_t, 1, 128> byte_code;
    Vec<uint8_t, 1, 128> flat, packed_flat;
    Vec<uint8_t, 1, 96> pack_scratch;
    Tile<TileType::Vec, uint8_t, 16, 8, BLayout::RowMajor, 16, 8, SLayout::RowMajor, 32> packed;
    Vec<float8_e4m3_t, 128, 32> quantized;
    Vec<float8_e4m3_t, Rows, 256> payload;
    TASSIGN(input, 0); TASSIGN(groups, 0); TASSIGN(absolute, 16384); TASSIGN(normalized, 32768);
    TASSIGN(maximum, 49152); TASSIGN(maximum_row, 49152);
    TASSIGN(exponent, 49664); TASSIGN(scale, 50176); TASSIGN(scale_bits, 50176);
    TASSIGN(signed_code, 50688); TASSIGN(shifted, 51200); TASSIGN(signed_float, 51712);
    TASSIGN(half_code, 52224); TASSIGN(short_code, 52480); TASSIGN(byte_code, 52736); TASSIGN(flat, 52736);
    TASSIGN(pack_scratch, 52864); TASSIGN(packed, 52992); TASSIGN(packed_flat, 52992);
    TASSIGN(quantized, 53248); TASSIGN(payload, 53248); TASSIGN(max_scratch, 57600);
    for (int part = 0; part < 4; ++part) {
        using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, Rows, 256>, Stride> input_gm(
            data<bfloat16_t>(args, 0) + first * width + column + part * 256,
            TileShape2D<bfloat16_t, Rows, 256>(), Stride(1, 1, 1, width, 1));
        Vec<bfloat16_t, Rows, 256> raw;
        TASSIGN(raw, 65536);
        TLOAD(raw, input_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(input, raw, RoundMode::CAST_NONE);
        TABS(absolute, groups); TROWMAX(maximum, absolute, max_scratch);
        TMAXS(maximum_row, maximum_row, 1e-4f); TMULS(maximum_row, maximum_row, 1.0f / 448.0f);
        Integers bits;
        TASSIGN(bits, 49152);
        TADDS(exponent, bits, 8388607); TSHRS(exponent, exponent, 23); TSHLS(scale_bits, exponent, 23);
        TROWEXPANDDIV(normalized, groups, scale);
        TMAXS(normalized, normalized, -448.0f); TMINS(normalized, normalized, 448.0f);
        TCVT(quantized, normalized, RoundMode::CAST_RINT);
        TSHRS(shifted, exponent, 7); TMULS(shifted, shifted, 256); TSUB(signed_code, exponent, shifted);
        TCVT(signed_float, signed_code, RoundMode::CAST_RINT);
        TCVT(half_code, signed_float, RoundMode::CAST_RINT);
        TCVT(short_code, half_code, RoundMode::CAST_RINT);
        TCVT(byte_code, short_code, RoundMode::CAST_RINT);
        TMOV(packed, flat, pack_scratch);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GlobalTensor<float8_e4m3_t, TileShape2D<float8_e4m3_t, Rows, 256>, Stride> payload_gm(
            data<float8_e4m3_t>(args, 1) + first * width + column + part * 256,
            TileShape2D<float8_e4m3_t, Rows, 256>(), Stride(1, 1, 1, width, 1));
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 1, 128>, BaseShape2D<uint8_t, 1, 128>> scale_gm(
            data<uint8_t>(args, 2) + first * (width / 32) + (column / 32 + part * 8) * Rows);
        TSTORE(payload_gm, payload); TSTORE(scale_gm, packed_flat);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
#endif
    auto input = reinterpret_cast<__gm__ Tensor*>(args[0]);
    int tasks = input->shapes[0] / Rows * (input->shapes[1] / Columns);
    for (int block = get_block_idx(args); block < tasks; block += get_block_num(args)) {
#if defined(PTO_NPU_ARCH_A5)
        run_native(args, block);
#else
        run_tiles(args, block);
#endif
    }
#endif
}
