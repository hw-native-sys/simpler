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
template <typename Output, typename Scales, typename Input>
__tf__ __aicore__ inline void quant_vf(
    typename Output::TileDType __out__ output, typename Scales::TileDType __out__ scales,
    typename Input::TileDType __in__ input, uint16_t active_rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto scale = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t four_count = 4;
        MaskReg four = CreatePredicate<uint16_t>(four_count);
        RegTensor<uint32_t> lane, group_index;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(group_index, lane, 3u, all, MODE_ZEROING);
        RegTensor<uint32_t> pair_lo, pair_hi;
        RegTensor<uint16_t> half_index, scatter_offset;
        vshls(pair_lo, lane, 1u, all, MODE_ZEROING);
        vadds(pair_hi, pair_lo, 1u, all, MODE_ZEROING);
        vci((RegTensor<int16_t>&)half_index, 0, INC_ORDER);
        vshls(scatter_offset, half_index, (uint16_t)4, all16, MODE_ZEROING);
        vshls(half_index, half_index, (uint16_t)1, all16, MODE_ZEROING);
        RegTensor<int32_t> izero, ione;
        RegTensor<float> zero, onef, low, high;
        vdup(izero, 0, all, MODE_ZEROING);
        vdup(ione, 1, all, MODE_ZEROING);
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(onef, 1.0f, all, MODE_ZEROING);
        vdup(low, -448.0f, all, MODE_ZEROING);
        vdup(high, 448.0f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < active_rows; ++row) {
            for (uint16_t chunk = 0; chunk < 2; ++chunk) {
                RegTensor<float> x0, x1, x2, x3;
                RegTensor<bfloat16_t> even, odd;
                vlds(even, odd, src, row * 512 + chunk * 256, DINTLV_B16);
                vcvt(x0, even, all16, PART_EVEN);
                vcvt(x2, even, all16, PART_ODD);
                vcvt(x1, odd, all16, PART_EVEN);
                vcvt(x3, odd, all16, PART_ODD);
                RegTensor<float> m0, m1, m2, m3, maximum;
                vabs(m0, x0, all, MODE_ZEROING);
                vabs(m1, x1, all, MODE_ZEROING);
                vabs(m2, x2, all, MODE_ZEROING);
                vabs(m3, x3, all, MODE_ZEROING);
                vmax(m0, m0, m1, all, MODE_ZEROING);
                vmax(m2, m2, m3, all, MODE_ZEROING);
                vmax(m0, m0, m2, all, MODE_ZEROING);
                vcgmax(maximum, m0, all, MODE_ZEROING);
                RegTensor<int32_t> compact_exp, exponent, nonzero;
                vshrs(compact_exp, (RegTensor<int32_t>&)maximum, 23, all, MODE_ZEROING);
                vadds(compact_exp, compact_exp, -8, all, MODE_ZEROING);
                vmax(compact_exp, compact_exp, izero, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)exponent, (RegTensor<uint32_t>&)compact_exp, group_index);
                vmax(nonzero, exponent, ione, all, MODE_ZEROING);
                RegTensor<float> divisor, floor_mask;
                vshls((RegTensor<int32_t>&)divisor, nonzero, 23, all, MODE_ZEROING);
                MaskReg nonzero_mask;
                vcmps_gt(nonzero_mask, exponent, 0, all);
                vsel(floor_mask, onef, zero, nonzero_mask);
                vdiv(x0, x0, divisor, all, MODE_ZEROING);
                vmul(x0, x0, floor_mask, all, MODE_ZEROING);
                vmax(x0, x0, low, all, MODE_ZEROING);
                vmin(x0, x0, high, all, MODE_ZEROING);
                vdiv(x1, x1, divisor, all, MODE_ZEROING);
                vmul(x1, x1, floor_mask, all, MODE_ZEROING);
                vmax(x1, x1, low, all, MODE_ZEROING);
                vmin(x1, x1, high, all, MODE_ZEROING);
                vdiv(x2, x2, divisor, all, MODE_ZEROING);
                vmul(x2, x2, floor_mask, all, MODE_ZEROING);
                vmax(x2, x2, low, all, MODE_ZEROING);
                vmin(x2, x2, high, all, MODE_ZEROING);
                vdiv(x3, x3, divisor, all, MODE_ZEROING);
                vmul(x3, x3, floor_mask, all, MODE_ZEROING);
                vmax(x3, x3, low, all, MODE_ZEROING);
                vmin(x3, x3, high, all, MODE_ZEROING);
                vector_f8e4m3 p0, p1, p2, p3;
                vcvt(p0, x0, all, ROUND_R, RS_DISABLE, PART_P0);
                vcvt(p1, x1, all, ROUND_R, RS_DISABLE, PART_P1);
                vcvt(p2, x2, all, ROUND_R, RS_DISABLE, PART_P2);
                vcvt(p3, x3, all, ROUND_R, RS_DISABLE, PART_P3);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, all8);
                vor((RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, all8);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p2, all8);
                vsts((RegTensor<uint8_t>&)p0, dst, row * 512 + chunk * 256, NORM_B8, all8);
                RegTensor<uint32_t> e0, e1;
                RegTensor<uint16_t> pairs;
                vselr(e0, (RegTensor<uint32_t>&)compact_exp, pair_lo);
                vselr(e1, (RegTensor<uint32_t>&)compact_exp, pair_hi);
                vshls(e1, e1, 8u, all, MODE_ZEROING);
                vor(e0, e0, e1, all);
                vselr(pairs, (RegTensor<uint16_t>&)e0, half_index);
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
