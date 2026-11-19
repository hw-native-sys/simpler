// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Rotary embedding and group-16 cache publication with selectable E2M1 rounding.
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
template <typename T, int C>
using Vec = Tile<TileType::Vec, T, 1, C>;
template <typename T, int C>
using Batch = Tile<TileType::Vec, T, 8, C>;
template <typename T, int C>
using GM = GlobalTensor<T, TileShape2D<T, 1, C>, BaseShape2D<T, 1, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
template <bool RoundEven>
__tf__ __aicore__ inline void quantize(
    Batch<uint8_t, 256>::TileDType __out__ packed,
    Batch<bfloat16_t, 32>::TileDType __out__ scales,
    Batch<bfloat16_t, 512>::TileDType __in__ raw,
    Vec<float, 32>::TileDType __out__ scratch, uint16_t rows) {
    auto src_base = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(raw);
    auto dst_base = (__ubuf__ uint8_t*)__cce_get_tile_ptr(packed);
    auto rounded_scales = (__ubuf__ float*)__cce_get_tile_ptr(scratch);
    auto factors_base = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(scales);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        auto src = src_base + row * 512;
        auto dst = dst_base + row * 256;
        auto factors = factors_base + row * 32;
        for (int group = 0; group < 32; ++group) {
            float maximum = 0.0f;
            for (int i = 0; i < 16; ++i) maximum = std::max(maximum, std::abs(static_cast<float>(src[group * 16 + i])));
            bfloat16_t stored = static_cast<bfloat16_t>(std::max(maximum * (1.0f / 6.0f), 0.001953125f));
            factors[group] = stored;
            for (int pair = 0; pair < 8; ++pair) {
                int byte = 0;
                for (int half = 0; half < 2; ++half) {
                    float value = static_cast<float>(src[group * 16 + pair * 2 + half]) / static_cast<float>(stored);
                    float mag = std::abs(value);
                    int code = (mag > 0.25f) + (RoundEven ? mag >= 0.75f : mag > 0.75f) + (mag > 1.25f) + (RoundEven ? mag >= 1.75f : mag > 1.75f) +
                               (mag > 2.5f) + (RoundEven ? mag >= 3.5f : mag > 3.5f) + (mag > 5.0f);
                    code |= (std::bit_cast<int32_t>(value) >> 28) & 8;
                    byte |= code << (half * 4);
                }
                dst[group * 8 + pair] = byte;
            }
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t half_count = 32;
        MaskReg lower = plt_b32(half_count, POST_UPDATE);
        MaskReg full16 = pset_b16(PAT_ALL);
        RegTensor<uint16_t> abs_mask;
        vdup(abs_mask, (uint16_t)0x7fff, full16, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            auto src = src_base + row * 512;
            auto dst = dst_base + row * 256;
            auto factors = factors_base + row * 32;
            vector_align max_store;
            for (uint16_t chunk = 0; chunk < 4; ++chunk) {
                RegTensor<bfloat16_t> values;
                RegTensor<uint16_t> magnitude, maximum;
                vlds(values, src, chunk * 128, NORM);
                vand(magnitude, (RegTensor<uint16_t>&)values, abs_mask, full16, MODE_ZEROING);
                vcgmax(maximum, magnitude, full16, MODE_ZEROING);
                vstus(max_store, 8, (RegTensor<bfloat16_t>&)maximum, factors + chunk * 8);
            }
            vstas(max_store, factors, 0);
            mem_bar(VST_VLD);
            RegTensor<bfloat16_t> bf_scales;
            RegTensor<float> maxima, rounded;
            vlds(bf_scales, factors, 0, UNPK_B16);
            vcvt(maxima, bf_scales, lower, PART_EVEN);
            vmuls(maxima, maxima, 1.0f / 6.0f, lower, MODE_ZEROING);
            vmaxs(maxima, maxima, 0.001953125f, lower, MODE_ZEROING);
            vcvt(bf_scales, maxima, lower, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(rounded, bf_scales, lower, PART_EVEN);
            vsts(bf_scales, factors, 0, PK_B32, lower);
            vsts(rounded, rounded_scales, 0, NORM_B32, lower);
            mem_bar(VST_VLD);
            RegTensor<uint32_t> indices;
            vci((RegTensor<int32_t>&)indices, 0, INC_ORDER);
            vshrs(indices, indices, (uint32_t)4, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<bfloat16_t> bf_values;
                RegTensor<float> values, magnitude, scale, normalized;
                vlds(bf_values, src, chunk * 64, UNPK_B16);
                vcvt(values, bf_values, full, PART_EVEN);
                vgather2(scale, rounded_scales + chunk * 4, indices, full);
                vdiv(normalized, values, scale, full, MODE_ZEROING);
                vabs(magnitude, normalized, full);
                RegTensor<int32_t> codes, level, sign, sign_mask;
                vdup(codes, 0, full, MODE_ZEROING);
                MaskReg selected;
                vcmps_gt(selected, magnitude, 0.25f, full);
                vdup(level, 1, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                if constexpr (RoundEven) { vcmps_ge(selected, magnitude, 0.75f, full); }
                else { vcmps_gt(selected, magnitude, 0.75f, full); }
                vdup(level, 2, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_gt(selected, magnitude, 1.25f, full);
                vdup(level, 3, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                if constexpr (RoundEven) { vcmps_ge(selected, magnitude, 1.75f, full); }
                else { vcmps_gt(selected, magnitude, 1.75f, full); }
                vdup(level, 4, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                vcmps_gt(selected, magnitude, 2.5f, full);
                vdup(level, 5, full, MODE_ZEROING);
                vsel(codes, level, codes, selected);
                if constexpr (RoundEven) { vcmps_ge(selected, magnitude, 3.5f, full); }
                else { vcmps_gt(selected, magnitude, 3.5f, full); }
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
                vsts(bytes, (__ubuf__ int32_t*)(dst + chunk * 32), 0, PK4_B32, lower);
            }
        }
    }
#endif
}

__tf__ __aicore__ inline void rotate(
    Batch<bfloat16_t, 512>::TileDType __out__ output,
    Batch<bfloat16_t, 512>::TileDType __in__ row,
    Batch<float, 32>::TileDType __in__ cosine,
    Batch<float, 32>::TileDType __in__ sine, uint16_t rows) {
    auto values = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(row);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row_index = 0; row_index < rows; ++row_index) {
        for (int pair = 0; pair < 32; ++pair) {
            float even = static_cast<float>(values[row_index * 512 + 448 + pair * 2]);
            float odd = static_cast<float>(values[row_index * 512 + 449 + pair * 2]);
            dst[row_index * 512 + 448 + pair * 2] = static_cast<bfloat16_t>(even * cp[row_index * 32 + pair] - odd * sp[row_index * 32 + pair]);
            dst[row_index * 512 + 449 + pair * 2] = static_cast<bfloat16_t>(even * sp[row_index * 32 + pair] + odd * cp[row_index * 32 + pair]);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg half = plt_b32(count, POST_UPDATE);
        for (uint16_t row_index = 0; row_index < rows; ++row_index) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> raw, even, odd, cosv, sinv, a, b, real, imag, result, unused;
            vlds(bf, values, row_index * 512 + 448, UNPK_B16);
            vcvt(raw, bf, full, PART_EVEN);
            vlds(cosv, cp, row_index * 32, NORM);
            vlds(sinv, sp, row_index * 32, NORM);
            vdintlv(even, odd, raw, raw);
            vmul(a, even, cosv, half, MODE_ZEROING);
            vmul(b, odd, sinv, half, MODE_ZEROING);
            vsub(real, a, b, half);
            vmul(a, even, sinv, half, MODE_ZEROING);
            vmul(b, odd, cosv, half, MODE_ZEROING);
            vadd(imag, a, b, half, MODE_ZEROING);
            vintlv(result, unused, real, imag);
            vcvt(bf, result, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row_index * 512 + 448, PK_B32, full);
        }
        mem_bar(VST_VLD);
    }
#endif
}

template <bool RoundEven>
__aicore__ inline void publish_rows(
    Batch<bfloat16_t, 512>& input, Batch<float, 32>& cosine, Batch<float, 32>& sine,
    Batch<uint8_t, 256>& packed, Batch<bfloat16_t, 32>& scales, Vec<float, 32>& scratch,
    Vec<int64_t, 8>& targets, Vec<int32_t, 8>& tokens, int rows,
    int input_address, int packed_address, int scales_address,
    __gm__ bfloat16_t* latent, __gm__ uint8_t* cache, __gm__ bfloat16_t* cache_scales,
    int64_t cache_stride, int64_t scale_stride) {
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int row = 0; row < rows; ++row) {
        Vec<bfloat16_t, 512> value;
        TASSIGN(value, input_address + row * 1024);
        GM<bfloat16_t, 512> latent_gm(latent + (int64_t)tokens.GetValue(row) * 512);
        TSTORE(latent_gm, value);
    }
    // Finish storing the unrotated latent before changing its UB buffer.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    rotate(input.data(), input.data(), cosine.data(), sine.data(), rows);
    quantize<RoundEven>(packed.data(), scales.data(), input.data(), scratch.data(), rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int row = 0; row < rows; ++row) {
        int64_t slot = targets.GetValue(row);
        if (slot < 0) continue;
        Vec<uint8_t, 256> payload;
        Vec<bfloat16_t, 32> factor;
        TASSIGN(payload, packed_address + row * 256);
        TASSIGN(factor, scales_address + row * 64);
        GM<uint8_t, 256> output_gm(cache + slot * cache_stride);
        GM<bfloat16_t, 32> scales_gm(cache_scales + slot * scale_stride);
        TSTORE(output_gm, payload);
        TSTORE(scales_gm, factor);
        pipe_barrier(PIPE_MTE3);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace
