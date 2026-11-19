// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#pragma once
#include <pto/pto-inst.hpp>
namespace o_quant {
using namespace pto;
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <typename Output, typename Scales, typename Input, typename Trig>
__tf__ __aicore__ inline void rotate_quant_vf(
    typename Output::TileDType __out__ output, typename Scales::TileDType __out__ scales,
    typename Input::TileDType __out__ input,
    typename Trig::TileDType __in__ cosine, typename Trig::TileDType __in__ sine, uint16_t active_rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto scale = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    auto cos = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sin = (__ubuf__ float*)__cce_get_tile_ptr(sine);
    __VEC_SCOPE__ {
        {
            MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL);
            RegTensor<uint32_t> lane, tmp, ix1, pair_index;
            vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
            vshrs(tmp, lane, 1u, all, MODE_ZEROING);
            vshls(tmp, tmp, 1u, all, MODE_ZEROING);
            vsub(tmp, lane, tmp, all, MODE_ZEROING);
            vshls(tmp, tmp, 1u, all, MODE_ZEROING);
            vadds(ix1, lane, 1u, all, MODE_ZEROING);
            vsub(ix1, ix1, tmp, all, MODE_ZEROING);
            vshrs(pair_index, lane, 1u, all, MODE_ZEROING);
            MaskReg even;
            vcmp_gt(even, ix1, lane, all);
            for (uint16_t row = 0; row < active_rows; ++row) {
                RegTensor<float> c, s;
                vgather2(c, cos + row * 32, pair_index, all);
                vgather2(s, sin + row * 32, pair_index, all);
                vmuls(s, s, -1.0f, all, MODE_ZEROING);
                RegTensor<bfloat16_t> raw;
                RegTensor<float> x, swapped, a, b, re, im;
                vlds(raw, src, row * 512 + 448, UNPK_B16);
                vcvt(x, raw, all16, PART_EVEN);
                vselr((RegTensor<uint32_t>&)swapped, (RegTensor<uint32_t>&)x, ix1);
                vmul(a, x, c, all, MODE_ZEROING);
                vmul(b, swapped, s, all, MODE_ZEROING);
                vsub(re, a, b, all, MODE_ZEROING);
                vadd(im, b, a, all, MODE_ZEROING);
                vsel(x, re, im, even);
                vcvt(raw, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(raw, src, row * 512 + 448, PK_B32, all);
            }
        }
        // Order rounded RoPE stores before the quantization loads in this VF.
        mem_bar(VST_VLD);
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t four_count = 4;
        MaskReg four = CreatePredicate<uint16_t>(four_count);
        RegTensor<uint16_t> half_lane, group_index;
        vci((RegTensor<int16_t>&)half_lane, 0, INC_ORDER);
        vshrs(group_index, half_lane, (uint16_t)4, all16, MODE_ZEROING);
        RegTensor<uint16_t> exponent_mask, exponent_floor, inverse_bias, zero16;
        vdup(exponent_mask, (uint16_t)0x7f80, all16, MODE_ZEROING);
        vdup(exponent_floor, (uint16_t)(8 << 7), all16, MODE_ZEROING);
        vdup(inverse_bias, (uint16_t)(254 << 7), all16, MODE_ZEROING);
        vdup(zero16, (uint16_t)0, all16, MODE_ZEROING);
        RegTensor<uint16_t> pair_lo, pair_hi, scatter_offset;
        vshls(pair_lo, half_lane, (uint16_t)1, all16, MODE_ZEROING);
        vadds(pair_hi, pair_lo, (uint16_t)1, all16, MODE_ZEROING);
        vshls(scatter_offset, half_lane, (uint16_t)4, all16, MODE_ZEROING);
        RegTensor<float> zero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < active_rows; ++row) {
            for (uint16_t chunk = 0; chunk < 2; ++chunk) {
                RegTensor<float> x0, x1, x2, x3;
                RegTensor<bfloat16_t> even, odd;
                vlds(even, odd, src, row * 512 + chunk * 256, DINTLV_B16);
                RegTensor<uint16_t> e0_bits, e1_bits, maximum, shared, codes;
                // OCP shared exponents use the maximum BF16 exponent in each group of 32.
                vand(e0_bits, (RegTensor<uint16_t>&)even, exponent_mask, all16);
                vand(e1_bits, (RegTensor<uint16_t>&)odd, exponent_mask, all16);
                vmax(maximum, e0_bits, e1_bits, all16, MODE_ZEROING);
                vcgmax(maximum, maximum, all16, MODE_ZEROING);
                vmax(maximum, maximum, exponent_floor, all16, MODE_ZEROING);
                vsub(shared, maximum, exponent_floor, all16, MODE_ZEROING);
                vshrs(codes, shared, (uint16_t)7, all16, MODE_ZEROING);
                RegTensor<uint16_t> inverse, expanded_inverse;
                vsub(inverse, inverse_bias, shared, all16, MODE_ZEROING);
                MaskReg nonzero;
                vcmps_gt(nonzero, shared, (uint16_t)0, all16);
                vsel(inverse, inverse, zero16, nonzero);
                vselr(expanded_inverse, inverse, group_index);
                vmul(even, even, (RegTensor<bfloat16_t>&)expanded_inverse, all16, MODE_ZEROING);
                vmul(odd, odd, (RegTensor<bfloat16_t>&)expanded_inverse, all16, MODE_ZEROING);
                // Clamp normalized BF16 values to the finite E4M3 range.
                vmaxs(even, even, bfloat16_t(-448), all16, MODE_ZEROING);
                vmins(even, even, bfloat16_t(448), all16, MODE_ZEROING);
                vmaxs(odd, odd, bfloat16_t(-448), all16, MODE_ZEROING);
                vmins(odd, odd, bfloat16_t(448), all16, MODE_ZEROING);
                vcvt(x0, even, all16, PART_EVEN);
                vcvt(x2, even, all16, PART_ODD);
                vcvt(x1, odd, all16, PART_EVEN);
                vcvt(x3, odd, all16, PART_ODD);
                vector_f8e4m3 p0, p1, p2, p3;
                vcvt(p0, x0, all, ROUND_R, RS_DISABLE, PART_P0);
                vcvt(p1, x1, all, ROUND_R, RS_DISABLE, PART_P1);
                vcvt(p2, x2, all, ROUND_R, RS_DISABLE, PART_P2);
                vcvt(p3, x3, all, ROUND_R, RS_DISABLE, PART_P3);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, all8);
                vor((RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, all8);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p2, all8);
                vsts((RegTensor<uint8_t>&)p0, dst, row * 512 + chunk * 256, NORM_B8, all8);
                RegTensor<uint16_t> e0, e1, pairs;
                vselr(e0, codes, pair_lo);
                vselr(e1, codes, pair_hi);
                vshls(e1, e1, (uint16_t)8, all16, MODE_ZEROING);
                vor(pairs, e0, e1, all16);
                auto scale_base = scale + (row / 16) * 128 + row % 16 + chunk * 64;
                vscatter(pairs, scale_base, scatter_offset, four);
            }
        }
        for (uint16_t row = active_rows; row < Input::Rows; ++row) {
            for (uint16_t chunk = 0; chunk < 2; ++chunk) {
                vsts((RegTensor<uint8_t>&)zero, dst, row * 512 + chunk * 256, NORM_B8, all8);
                auto scale_base = scale + (row / 16) * 128 + row % 16 + chunk * 64;
                vscatter((RegTensor<uint16_t>&)zero, scale_base, scatter_offset, four);
            }
        }
    }
}
#endif
} // namespace o_quant
