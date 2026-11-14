// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Group-32 OCP FP8 quantization with M16-packed E8M0 scales.
#pragma once
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <bit>
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

namespace mx_quant {
using namespace pto;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
__aicore__ inline int64_t stride(__gm__ int64_t* args, int i) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->strides[0];
}

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
__aicore__ inline void quantize_pair(
    RegTensor<float> value, __ubuf__ uint8_t* dst, __ubuf__ uint16_t* codes,
    uint32_t offset, uint32_t scale_offset,
    MaskReg full, MaskReg lower, MaskReg upper, MaskReg bytes, MaskReg scalar,
    RegTensor<uint8_t>& byte_indices, RegTensor<int32_t>& zero, RegTensor<int32_t>& one) {
    RegTensor<float> magnitude, lo, hi, maximum, reciprocal;
    vabs(magnitude, value, full);
    vcmax(lo, magnitude, lower);
    vcmax(hi, magnitude, upper);
    vdup(lo, lo, full, POS_LOWEST, MODE_ZEROING);
    vdup(hi, hi, full, POS_LOWEST, MODE_ZEROING);
    vsel(maximum, lo, hi, lower);
    RegTensor<int32_t> exponent, nonzero, reciprocal_code;
    vshrs(exponent, (RegTensor<int32_t>&)maximum, 23, full, MODE_ZEROING);
    vadds(exponent, exponent, -8, full, MODE_ZEROING);
    vmax(exponent, exponent, zero, full, MODE_ZEROING);
    vmin(nonzero, exponent, one, full, MODE_ZEROING);
    vmuls(reciprocal_code, exponent, -1, full, MODE_ZEROING);
    vadds(reciprocal_code, reciprocal_code, 254, full, MODE_ZEROING);
    vmul(reciprocal_code, reciprocal_code, nonzero, full, MODE_ZEROING);
    vshls((RegTensor<int32_t>&)reciprocal, reciprocal_code, 23, full, MODE_ZEROING);
    vmul(value, value, reciprocal, full, MODE_ZEROING);
    vmaxs(value, value, -448.0f, full, MODE_ZEROING);
    vmins(value, value, 448.0f, full, MODE_ZEROING);
    vector_f8e4m3 fp8;
    RegTensor<uint8_t> packed;
    vcvt(fp8, value, full, ROUND_R, RS_DISABLE, PART_P0);
    vselr(packed, (RegTensor<uint8_t>&)fp8, byte_indices);
    vsts(packed, dst, offset, NORM_B8, bytes);
    RegTensor<uint32_t> first, second;
    vdup(first, (RegTensor<uint32_t>&)exponent, full, POS_LOWEST, MODE_ZEROING);
    vdup(second, (RegTensor<uint32_t>&)exponent, full, POS_HIGHEST, MODE_ZEROING);
    vshls(second, second, 8u, full, MODE_ZEROING);
    vor(first, first, second, full);
    vsts((RegTensor<uint16_t>&)first, codes, scale_offset, ONEPT_B16, scalar);
}
#endif

#if defined(__DAV_VEC__)
template <int C, typename Input>
__tf__ __aicore__ inline void quantize(
    typename Vec<uint8_t, 16, C>::TileDType __out__ output,
    typename Vec<uint8_t, 1, 16 * C / 32>::TileDType __out__ scales,
    typename Input::TileDType __in__ input, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto codes = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
#if defined(__CPU_SIM)
    for (int row = 0; row < 16; ++row) {
        for (int group = 0; group < C / 32; ++group) {
            float maximum = 0.0f;
            for (int lane = 0; lane < 32; ++lane) {
                float value = row < rows ? static_cast<float>(src[row * C + group * 32 + lane]) : 0.0f;
                maximum = std::max(maximum, std::abs(value));
            }
            int exponent = std::max(int(std::bit_cast<uint32_t>(maximum) >> 23) - 8, 0);
            float reciprocal = exponent ? std::bit_cast<float>(uint32_t(254 - exponent) << 23) : 0.0f;
            reinterpret_cast<__ubuf__ uint8_t*>(codes)[group / 2 * 32 + row * 2 + group % 2] = exponent;
            for (int lane = 0; lane < 32; ++lane) {
                float value = row < rows ? static_cast<float>(src[row * C + group * 32 + lane]) : 0.0f;
                value = std::clamp(value * reciprocal, -448.0f, 448.0f);
                reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst)[row * C + group * 32 + lane] =
                    float8_e4m3_t(value, RoundMode::CAST_RINT);
            }
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
            for (uint16_t chunk = 0; chunk < C / 64; ++chunk) {
                RegTensor<bfloat16_t> raw;
                RegTensor<float> value;
                vlds(raw, src, row * C + chunk * 64, UNPK_B16);
                vcvt(value, raw, full, PART_EVEN);
                quantize_pair(value, dst, codes, row * C + chunk * 64, chunk * 16 + row,
                              full, lower, upper, bytes, scalar, byte_indices, zero, one);
            }
        }
        RegTensor<uint8_t> padding;
        MaskReg all_bytes = pset_b8(PAT_ALL);
        vdup(padding, uint8_t(0), all_bytes, MODE_ZEROING);
        for (uint16_t row = rows; row < 16; ++row) {
            for (uint16_t chunk = 0; chunk < C / 256; ++chunk) {
                vsts(padding, dst, row * C + chunk * 256, NORM_B8, all_bytes);
            }
            for (uint16_t chunk = 0; chunk < C / 64; ++chunk) {
                vsts((RegTensor<uint16_t>&)padding, codes, chunk * 16 + row, ONEPT_B16, scalar);
            }
        }
    }
#endif
}
#endif
} // namespace mx_quant
