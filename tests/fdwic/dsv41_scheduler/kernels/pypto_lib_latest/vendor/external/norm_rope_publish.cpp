// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Window-key RMSNorm, BF16 RoPE, group-32 FP8 quantization, and page-owned publication.
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
    Batch<uint8_t, 512>::TileDType __out__ packed,
    Batch<uint8_t, 32>::TileDType __out__ scales,
    Batch<bfloat16_t, 512>::TileDType __in__ raw,
    Vec<bfloat16_t, 512>::TileDType __in__ weight,
    Batch<float, 32>::TileDType __in__ cosine,
    Batch<float, 32>::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(raw);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(packed);
    auto factors = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(scales);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float values[512];
        float partial[64] = {};
        for (int segment = 0; segment < 8; ++segment) {
            for (int lane = 0; lane < 64; ++lane) {
                float value = static_cast<float>(src[row * 512 + segment * 64 + lane]);
                partial[lane] += value * value;
            }
        }
        for (int width = 64; width > 1; width /= 2) {
            for (int lane = 0; lane < width / 2; ++lane) {
                partial[lane] = partial[lane * 2] + partial[lane * 2 + 1];
            }
        }
        float inv = 1.0f / std::sqrt(partial[0] / 512.0f + 1.0e-6f);
        for (int lane = 0; lane < 512; ++lane) {
            values[lane] = static_cast<float>(static_cast<bfloat16_t>(
                static_cast<float>(src[row * 512 + lane]) * inv * static_cast<float>(gamma[lane])));
        }
        for (int pair = 0; pair < 32; ++pair) {
            float even = values[448 + pair * 2];
            float odd = values[449 + pair * 2];
            values[448 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(
                even * cp[row * 32 + pair] - odd * sp[row * 32 + pair]));
            values[449 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(
                even * sp[row * 32 + pair] + odd * cp[row * 32 + pair]));
        }
        for (int group = 0; group < 16; ++group) {
            float maximum = 1.0e-4f;
            for (int lane = 0; lane < 32; ++lane) maximum = std::max(maximum, std::abs(values[group * 32 + lane]));
            int32_t exponent = (std::bit_cast<int32_t>(maximum * (1.0f / 448.0f)) + 8388607) >> 23;
            bfloat16_t scale = static_cast<bfloat16_t>(std::bit_cast<float>(exponent << 23));
            factors[row * 16 + group] = scale;
            auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst);
            for (int lane = 0; lane < 32; ++lane) {
                output[row * 512 + group * 32 + lane] = float8_e4m3_t(
                    values[group * 32 + lane] / static_cast<float>(scale), RoundMode::CAST_RINT);
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
        uint32_t byte_count = 64;
        MaskReg bytes_mask = plt_b8(byte_count, POST_UPDATE);
        RegTensor<uint8_t> byte_indices;
        vci((RegTensor<int8_t>&)byte_indices, (int8_t)0, INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_indices, (RegTensor<int16_t>&)byte_indices, (int16_t)4, pset_b8(PAT_ALL));
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> sum, source, square, inverse, one, cosv, sinv;
            RegTensor<bfloat16_t> bf;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t segment = 0; segment < 8; ++segment) {
                vlds(bf, src, row * 512 + segment * 64, UNPK_B16);
                vcvt(source, bf, full, PART_EVEN);
                vmul(square, source, source, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(inverse, sum, 1.0f / 512.0f, full, MODE_ZEROING);
            vadds(inverse, inverse, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inverse, inverse, full, MODE_ZEROING);
            vdup(one, 1.0f, full, MODE_ZEROING);
            vdiv(inverse, one, inverse, full, MODE_ZEROING);
            vdup(inverse, inverse, full, POS_LOWEST, MODE_ZEROING);
            vlds(cosv, cp, row * 32, NORM);
            vlds(sinv, sp, row * 32, NORM);
            for (uint16_t segment = 0; segment < 8; ++segment) {
                RegTensor<float> values, gain;
                vlds(bf, src, row * 512 + segment * 64, UNPK_B16);
                vcvt(values, bf, full, PART_EVEN);
                vlds(bf, gamma, segment * 64, UNPK_B16);
                vcvt(gain, bf, full, PART_EVEN);
                vmul(values, values, inverse, full, MODE_ZEROING);
                vmul(values, values, gain, full, MODE_ZEROING);
                vcvt(bf, values, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vcvt(values, bf, full, PART_EVEN);
                if (segment == 7) {
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
                RegTensor<float> magnitude, lo, hi, scale, normalized;
                vabs(magnitude, values, full);
                vcmax(lo, magnitude, lower);
                vcmax(hi, magnitude, upper);
                vdup(lo, lo, full, POS_LOWEST, MODE_ZEROING);
                vdup(hi, hi, full, POS_LOWEST, MODE_ZEROING);
                vsel(scale, lo, hi, lower);
                vmaxs(scale, scale, 1.0e-4f, full, MODE_ZEROING);
                vmuls(scale, scale, 1.0f / 448.0f, full, MODE_ZEROING);
                RegTensor<int32_t> exponent, bits;
                vadds(bits, (RegTensor<int32_t>&)scale, 8388607, full, MODE_ZEROING);
                vshrs(exponent, bits, 23, full, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)scale, exponent, 23, full, MODE_ZEROING);
                vdiv(normalized, values, scale, full, MODE_ZEROING);
                vector_f8e4m3 fp8;
                RegTensor<uint8_t> packed_bytes;
                vcvt(fp8, normalized, full, ROUND_R, RS_DISABLE, PART_P0);
                vselr(packed_bytes, (RegTensor<uint8_t>&)fp8, byte_indices);
                vsts(packed_bytes, dst, row * 512 + segment * 64, NORM_B8, bytes_mask);
                RegTensor<uint32_t> first, second, codes;
                vdup(first, (RegTensor<uint32_t>&)scale, full, POS_LOWEST, MODE_ZEROING);
                vdup(second, (RegTensor<uint32_t>&)scale, full, POS_HIGHEST, MODE_ZEROING);
                vshrs(first, first, 16, full, MODE_ZEROING);
                vor(codes, first, second, full, MODE_ZEROING);
                vsts(codes, (__ubuf__ uint32_t*)factors, row * 8 + segment, ONEPT_B32, full);
            }
        }
    }
#endif
}

__aicore__ inline void flush_rows(__gm__ int64_t* args, int rows, Vec<int64_t, 8>& targets) {
    Batch<bfloat16_t, 512> input;
    Batch<float, 32> cosine, sine;
    Batch<uint8_t, 512> packed;
    Batch<uint8_t, 32> scales;
    Vec<bfloat16_t, 512> weight;
    TASSIGN(input, 0);
    TASSIGN(cosine, 8192);
    TASSIGN(sine, 9216);
    TASSIGN(packed, 10240);
    TASSIGN(scales, 14336);
    TASSIGN(weight, 14592);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    postprocess(packed.data(), scales.data(), input.data(), weight.data(), cosine.data(), sine.data(), rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int row = 0; row < rows;) {
        int64_t slot = targets.GetValue(row);
        int count = 1;
        while (row + count < rows && targets.GetValue(row + count) == slot + count) ++count;
        using Payload = Tile<TileType::Vec, uint8_t, 8, 512, BLayout::RowMajor, DYNAMIC, 512>;
        using Codes = Tile<TileType::Vec, uint8_t, 8, 32, BLayout::RowMajor, DYNAMIC, 32>;
        Payload payload(count);
        Codes codes(count);
        TASSIGN(payload, 10240 + row * 512);
        TASSIGN(codes, 14336 + row * 32);
        using PayloadShape = TileShape2D<uint8_t, DYNAMIC, 512>;
        using CodeShape = TileShape2D<uint8_t, DYNAMIC, 32>;
        using RowStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        int64_t payload_stride = reinterpret_cast<__gm__ Tensor*>(args[5])->strides[1];
        int64_t scale_stride = payload_stride;
        auto scale_base = data<uint8_t>(args, 5) + 512;
        GlobalTensor<uint8_t, PayloadShape, RowStride> output_gm(
            data<uint8_t>(args, 5) + slot * payload_stride, PayloadShape(count, 512),
            RowStride(1, 1, 1, payload_stride, 1));
        GlobalTensor<uint8_t, CodeShape, RowStride> scale_gm(
            scale_base + slot * scale_stride, CodeShape(count, 32), RowStride(1, 1, 1, scale_stride, 1));
        TSTORE(output_gm, payload);
        TSTORE(scale_gm, codes);
        // Preserve input order between runs, including repeated cache slots.
        pipe_barrier(PIPE_MTE3);
        row += count;
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    Tile<TileType::Vec, int64_t, 1, 128, BLayout::RowMajor, 1, DYNAMIC> slots(128);
    Vec<int64_t, 8> targets;
    Vec<bfloat16_t, 512> weight;
    TASSIGN(slots, 15616);
    TASSIGN(targets, 16640);
    TASSIGN(weight, 14592);
    GM<bfloat16_t, 512> weight_gm(data<bfloat16_t>(args, 1));
    TLOAD(weight, weight_gm);
    int tokens = args[6];
    int worker = get_block_idx(args);
    int workers = get_block_num(args);
    int pending = 0;
    for (int base = 0; base < tokens; base += 128) {
        int count = tokens - base < 128 ? tokens - base : 128;
        slots.SetValidCol(count);
        using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 128>>;
        SlotGM slot_gm(data<int64_t>(args, 4) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
        TLOAD(slots, slot_gm);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int lane = 0; lane < count;) {
            int64_t slot = slots.GetValue(lane);
            if (slot < 0 || (slot / 128) % workers != worker) {
                ++lane;
                continue;
            }
            int run = 0;
            while (lane + run < count && pending + run < 8) {
                int64_t target = slots.GetValue(lane + run);
                if (target < 0 || (target / 128) % workers != worker) break;
                targets.SetValue(pending + run, target);
                ++run;
            }
            using Input = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, DYNAMIC, 512>;
            using Rope = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, DYNAMIC, 32>;
            Input input(run);
            Rope cosine(run), sine(run);
            TASSIGN(input, pending * 1024);
            TASSIGN(cosine, 8192 + pending * 128);
            TASSIGN(sine, 9216 + pending * 128);
            using InputShape = TileShape2D<bfloat16_t, DYNAMIC, 512>;
            using RopeShape = TileShape2D<float, DYNAMIC, 32>;
            GlobalTensor<bfloat16_t, InputShape, BaseShape2D<bfloat16_t, 8, 512>> input_gm(
                data<bfloat16_t>(args, 0) + (int64_t)(base + lane) * 512, InputShape(run, 512));
            GlobalTensor<float, RopeShape, BaseShape2D<float, 8, 32>> cos_gm(
                data<float>(args, 2) + (int64_t)(base + lane) * 32, RopeShape(run, 32));
            GlobalTensor<float, RopeShape, BaseShape2D<float, 8, 32>> sin_gm(
                data<float>(args, 3) + (int64_t)(base + lane) * 32, RopeShape(run, 32));
            TLOAD(input, input_gm);
            TLOAD(cosine, cos_gm);
            TLOAD(sine, sin_gm);
            lane += run;
            pending += run;
            if (pending == 8) {
                flush_rows(args, pending, targets);
                pending = 0;
            }
        }
    }
    if (pending > 0) flush_rows(args, pending, targets);
    pipe_barrier(PIPE_ALL);
#endif
}
