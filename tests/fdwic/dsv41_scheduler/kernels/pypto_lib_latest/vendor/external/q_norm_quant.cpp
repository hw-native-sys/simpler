// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Q-latent RMSNorm and OCP MXFP8 quantization.
#include "mx_quant.hpp"
using namespace mx_quant;
namespace {
#if defined(__DAV_VEC__)
using Input = Tile<TileType::Vec, bfloat16_t, 16, 1280, BLayout::RowMajor, DYNAMIC, 1280>;
__tf__ __aicore__ inline void normalize_quantize(
    Vec<bfloat16_t, 16, 1280>::TileDType __out__ output,
    Vec<uint8_t, 16, 1280>::TileDType __out__ payload,
    Vec<uint8_t, 1, 640>::TileDType __out__ scales,
    Input::TileDType __in__ input,
    Vec<bfloat16_t, 1, 1280>::TileDType __in__ weight, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto quant_dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(payload);
    auto codes = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int k = 0; k < 1280; ++k) {
            float x = static_cast<float>(src[row * 1280 + k]);
            sum += x * x;
        }
        float inverse = 1.0f / std::sqrt(sum / 1280.0f + 1.0e-6f);
        for (int k = 0; k < 1280; ++k) {
            dst[row * 1280 + k] = static_cast<bfloat16_t>(
                static_cast<float>(src[row * 1280 + k]) * inverse * static_cast<float>(gamma[k]));
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t half_count = 32, byte_count = 64, scalar_count = 1;
        MaskReg lower = plt_b32(half_count, POST_UPDATE), upper;
        pnot(upper, lower, full);
        MaskReg bytes = CreatePredicate<uint8_t>(byte_count);
        MaskReg scalar = CreatePredicate<uint16_t>(scalar_count);
        RegTensor<uint8_t> byte_indices;
        vci((RegTensor<int8_t>&)byte_indices, (int8_t)0, INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_indices, (RegTensor<int16_t>&)byte_indices,
              (int16_t)4, pset_b8(PAT_ALL));
        RegTensor<int32_t> zero, one;
        vdup(zero, 0, full, MODE_ZEROING);
        vdup(one, 1, full, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> sum, value, square, inverse, float_one;
            RegTensor<bfloat16_t> raw;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t k = 0; k < 20; ++k) {
                vlds(raw, src, row * 1280 + k * 64, UNPK_B16);
                vcvt(value, raw, full, PART_EVEN);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(inverse, sum, 1.0f / 1280.0f, full, MODE_ZEROING);
            vadds(inverse, inverse, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inverse, inverse, full, MODE_ZEROING);
            vdup(float_one, 1.0f, full, MODE_ZEROING);
            vdiv(inverse, float_one, inverse, full, MODE_ZEROING);
            vdup(inverse, inverse, full, POS_LOWEST, MODE_ZEROING);
            for (uint16_t k = 0; k < 20; ++k) {
                RegTensor<float> gain;
                vlds(raw, src, row * 1280 + k * 64, UNPK_B16);
                vcvt(value, raw, full, PART_EVEN);
                vlds(raw, gamma, k * 64, UNPK_B16);
                vcvt(gain, raw, full, PART_EVEN);
                vmul(value, value, inverse, full, MODE_ZEROING);
                vmul(value, value, gain, full, MODE_ZEROING);
                vcvt(raw, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(raw, dst, row * 1280 + k * 64, PK_B32, full);
                vcvt(value, raw, full, PART_EVEN);
                quantize_pair(value, quant_dst, codes, row * 1280 + k * 64, k * 16 + row,
                              full, lower, upper, bytes, scalar, byte_indices, zero, one);
            }
        }
        RegTensor<uint8_t> padding;
        MaskReg all_bytes = pset_b8(PAT_ALL);
        vdup(padding, uint8_t(0), all_bytes, MODE_ZEROING);
        for (uint16_t row = rows; row < 16; ++row) {
            for (uint16_t k = 0; k < 5; ++k) {
                vsts(padding, quant_dst, row * 1280 + k * 256, NORM_B8, all_bytes);
            }
            for (uint16_t k = 0; k < 20; ++k) {
                vsts((RegTensor<uint16_t>&)padding, codes, k * 16 + row, ONEPT_B16, scalar);
            }
        }
    }
#endif
}
#endif
}
__aicore__ inline void normalize_chunk(__gm__ int64_t* args, int first, int rows) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    Input input(rows);
    Vec<bfloat16_t, 16, 1280> normalized;
    Vec<bfloat16_t, 1, 1280> weight;
    Vec<uint8_t, 16, 1280> output;
    Vec<uint8_t, 1, 640> scales;
    TASSIGN(input, 0);
    TASSIGN(normalized, 40960);
    TASSIGN(weight, 81920);
    TASSIGN(output, 84480);
    TASSIGN(scales, 104960);
    using Shape = TileShape2D<bfloat16_t, DYNAMIC, 1280>;
    using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    if (rows > 0) {
        GlobalTensor<bfloat16_t, Shape, Stride> source(data<bfloat16_t>(args, 0) + first * stride(args, 0),
            Shape(rows, 1280), Stride(1, 1, 1, stride(args, 0), 1));
        GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, 1280>, BaseShape2D<bfloat16_t, 1, 1280>> gamma(
            data<bfloat16_t>(args, 1));
        TLOAD(input, source);
        TLOAD(weight, gamma);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    normalize_quantize(normalized.data(), output.data(), scales.data(), input.data(), weight.data(), rows);
#if defined(__CPU_SIM)
    quantize<1280, decltype(normalized)>(output.data(), scales.data(), normalized.data(), rows);
#endif
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (rows > 0) {
        Input active_normalized(rows);
        TASSIGN(active_normalized, 40960);
        GlobalTensor<bfloat16_t, Shape, Stride> latent(data<bfloat16_t>(args, 2) + first * stride(args, 2),
            Shape(rows, 1280), Stride(1, 1, 1, stride(args, 2), 1));
        TSTORE(latent, active_normalized);
    }
    GlobalTensor<uint8_t, TileShape2D<uint8_t, 16, 1280>, BaseShape2D<uint8_t, 16, 1280>> target(data<uint8_t>(args, 3) + first * 1280);
    GlobalTensor<uint8_t, TileShape2D<uint8_t, 1, 640>, BaseShape2D<uint8_t, 1, 640>> scale_target(data<uint8_t>(args, 4) + first * 40);
    TSTORE(target, output);
    TSTORE(scale_target, scales);
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    int active = args[5];
    const int capacity = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0];
    for (int first = get_block_idx(args) * 16; first < capacity; first += get_block_num(args) * 16) {
        int rows = active - first;
        rows = rows < 0 ? 0 : (rows > 16 ? 16 : rows);
        normalize_chunk(args, first, rows);
    }
#endif
}
