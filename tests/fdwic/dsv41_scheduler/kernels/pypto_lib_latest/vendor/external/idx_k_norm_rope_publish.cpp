// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Index-key RMSNorm, BF16 RoPE, group-32 E2M1 quantization, and page-owned publication.
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
__tf__ __aicore__ inline void postprocess(
    Batch<uint8_t, 64>::TileDType __out__ packed,
    Batch<uint8_t, 32>::TileDType __out__ scales,
    Batch<bfloat16_t, 128>::TileDType __in__ raw,
    Vec<bfloat16_t, 128>::TileDType __in__ weight,
    Batch<float, 32>::TileDType __in__ cosine,
    Batch<float, 32>::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(raw);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(packed);
    auto exp = (__ubuf__ uint8_t*)__cce_get_tile_ptr(scales);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int head = 0; head < rows; ++head) {
        float values[128];
        float partial[2][64];
        for (int half = 0; half < 2; ++half) {
            for (int lane = 0; lane < 64; ++lane) {
                float value = static_cast<float>(src[head * 128 + half * 64 + lane]);
                partial[half][lane] = value * value;
            }
            for (int width = 64; width > 1; width /= 2) {
                for (int lane = 0; lane < width / 2; ++lane)
                    partial[half][lane] = partial[half][lane * 2] + partial[half][lane * 2 + 1];
            }
        }
        float square = partial[0][0] + partial[1][0];
        float inv = 1.0f / std::sqrt(square / 128.0f + 1.0e-6f);
        for (int lane = 0; lane < 128; ++lane)
            values[lane] = static_cast<float>(static_cast<bfloat16_t>(
                static_cast<float>(src[head * 128 + lane]) * inv * static_cast<float>(gamma[lane])));
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
        RegTensor<float> cosv, sinv, gamma0, gamma1;
        RegTensor<bfloat16_t> gamma_bf;
        vlds(gamma_bf, gamma, 0, UNPK_B16);
        vcvt(gamma0, gamma_bf, full, PART_EVEN);
        vlds(gamma_bf, gamma, 64, UNPK_B16);
        vcvt(gamma1, gamma_bf, full, PART_EVEN);
        for (uint16_t head = 0; head < rows; ++head) {
            RegTensor<bfloat16_t> source_bf;
            RegTensor<float> source0, source1, square0, square1, inverse, one;
            vlds(source_bf, src, head * 128, UNPK_B16);
            vcvt(source0, source_bf, full, PART_EVEN);
            vlds(source_bf, src, head * 128 + 64, UNPK_B16);
            vcvt(source1, source_bf, full, PART_EVEN);
            vmul(square0, source0, source0, full, MODE_ZEROING);
            vmul(square1, source1, source1, full, MODE_ZEROING);
            vcadd(square0, square0, full, MODE_ZEROING);
            vcadd(square1, square1, full, MODE_ZEROING);
            vadd(inverse, square0, square1, full, MODE_ZEROING);
            vmuls(inverse, inverse, 1.0f / 128.0f, full, MODE_ZEROING);
            vadds(inverse, inverse, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inverse, inverse, full, MODE_ZEROING);
            vdup(one, 1.0f, full, MODE_ZEROING);
            vdiv(inverse, one, inverse, full, MODE_ZEROING);
            vdup(inverse, inverse, full, POS_LOWEST, MODE_ZEROING);
            RegTensor<int32_t> scale_bytes;
            vlds(cosv, cp, head * 32, NORM);
            vlds(sinv, sp, head * 32, NORM);
            for (uint16_t segment = 0; segment < 2; ++segment) {
                RegTensor<bfloat16_t> bf;
                RegTensor<float> values;
                if (segment == 0) {
                    vmul(values, source0, inverse, full, MODE_ZEROING);
                    vmul(values, values, gamma0, full, MODE_ZEROING);
                } else {
                    vmul(values, source1, inverse, full, MODE_ZEROING);
                    vmul(values, values, gamma1, full, MODE_ZEROING);
                }
                // Preserve the RMSNorm BF16 boundary before the rotary arithmetic.
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
                RegTensor<int32_t> first, second, pair;
                vdup(first, exponent, full, POS_LOWEST, MODE_ZEROING);
                vdup(second, exponent, full, POS_HIGHEST, MODE_ZEROING);
                vshls(second, second, 8, full, MODE_ZEROING);
                vor(pair, first, second, full, MODE_ZEROING);
                if (segment == 0) {
                    vadds(scale_bytes, pair, 0, full, MODE_ZEROING);
                } else {
                    vshls(pair, pair, 16, full, MODE_ZEROING);
                    vor(scale_bytes, scale_bytes, pair, full, MODE_ZEROING);
                    vsts(scale_bytes, (__ubuf__ int32_t*)(exp + head * 32), 0, ONEPT_B32, full);
                }
            }
        }
    }
#endif
}

__aicore__ inline void flush_rows(__gm__ int64_t* args, int rows, Vec<int64_t, 8>& targets) {
    Batch<bfloat16_t, 128> input;
    Batch<float, 32> cosine, sine;
    Batch<uint8_t, 64> packed;
    Batch<uint8_t, 32> scales;
    Vec<bfloat16_t, 128> weight;
    TASSIGN(input, 0);
    TASSIGN(cosine, 2048);
    TASSIGN(sine, 3072);
    TASSIGN(packed, 4096);
    TASSIGN(scales, 4608);
    TASSIGN(weight, 4864);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    postprocess(packed.data(), scales.data(), input.data(), weight.data(), cosine.data(), sine.data(), rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int row = 0; row < rows; ++row) {
        int64_t slot = targets.GetValue(row);
        Vec<uint8_t, 64> payload;
        Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, 1, 4> codes;
        TASSIGN(payload, 4096 + row * 64);
        TASSIGN(codes, 4608 + row * 32);
        GM<uint8_t, 64> output_gm(data<uint8_t>(args, 5) + slot * 64);
        GM<uint8_t, 4> scale_gm(data<uint8_t>(args, 6) + slot * 4);
        TSTORE(output_gm, payload);
        TSTORE(scale_gm, codes);
        // Preserve input order for repeated slots and adjacent short scale writes.
        pipe_barrier(PIPE_MTE3);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    Tile<TileType::Vec, int64_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC> slots(32);
    Vec<int64_t, 8> targets;
    Vec<bfloat16_t, 128> weight;
    TASSIGN(slots, 5120);
    TASSIGN(targets, 5376);
    TASSIGN(weight, 4864);
    GM<bfloat16_t, 128> weight_gm(data<bfloat16_t>(args, 1));
    TLOAD(weight, weight_gm);
    int tokens = args[7];
    int worker = get_block_idx(args);
    int workers = get_block_num(args);
    int pending = 0;
    for (int base = 0; base < tokens; base += 32) {
        int count = tokens - base < 32 ? tokens - base : 32;
        slots.SetValidCol(count);
        using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 32>>;
        SlotGM slot_gm(data<int64_t>(args, 4) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
        TLOAD(slots, slot_gm);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int lane = 0; lane < count; ++lane) {
            int64_t slot = slots.GetValue(lane);
            if (slot < 0 || (slot / 128) % workers != worker) continue;
            targets.SetValue(pending, slot);
            Vec<bfloat16_t, 128> input;
            Vec<float, 32> cosine, sine;
            TASSIGN(input, pending * 256);
            TASSIGN(cosine, 2048 + pending * 128);
            TASSIGN(sine, 3072 + pending * 128);
            GM<bfloat16_t, 128> input_gm(data<bfloat16_t>(args, 0) + (int64_t)(base + lane) * 128);
            GM<float, 32> cos_gm(data<float>(args, 2) + (int64_t)(base + lane) * 32);
            GM<float, 32> sin_gm(data<float>(args, 3) + (int64_t)(base + lane) * 32);
            TLOAD(input, input_gm);
            TLOAD(cosine, cos_gm);
            TLOAD(sine, sin_gm);
            if (++pending == 8) {
                flush_rows(args, pending, targets);
                pending = 0;
            }
        }
    }
    if (pending > 0) flush_rows(args, pending, targets);
    pipe_barrier(PIPE_ALL);
#endif
}
