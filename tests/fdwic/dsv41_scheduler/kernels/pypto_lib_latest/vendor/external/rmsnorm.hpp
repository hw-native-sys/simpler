// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// CANN 9.3 arch35 RMS: folded square reduction and Newton-corrected inverse root.
#pragma once
#include <type_traits>
#include <limits>
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
namespace recipe_rms {
using namespace pto;
__aicore__ constexpr int fold_point(int width) {
    int result = 1;
    while (result * 2 < width) result *= 2;
    return result;
}
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
__aicore__ inline void fma(RegTensor<float>& acc, RegTensor<float>& lhs,
                          RegTensor<float>& rhs, MaskReg& mask) {
    RegTensor<float> value = lhs;
    vmadd(value, rhs, acc, mask, MODE_ZEROING);
    acc = value;
}
__aicore__ inline void inverse_root(RegTensor<float>& result, RegTensor<float>& variance,
                                   MaskReg& mask) {
    RegTensor<float> r, y, s, t, one, half, t1, t3, t4;
    vdup(one, 1.0f, mask, MODE_ZEROING);
    vdup(half, 0.5f, mask, MODE_ZEROING);
    vdup(t1, 1.5f, mask, MODE_ZEROING);
    vdup(s, 1.0f, mask, MODE_ZEROING);
    vadds(variance, variance, 1.0e-6f, mask, MODE_ZEROING);
    vdiv(r, one, variance, mask, MODE_ZEROING);
    vsqrt(y, r, mask, MODE_ZEROING);
    vmuls(t, variance, -0.5f, mask, MODE_ZEROING);
    vmul(t, t, y, mask, MODE_ZEROING);
    fma(t1, t, y, mask);
    vmul(result, y, t1, mask, MODE_ZEROING);
    vmuls(t3, variance, -1.0f, mask, MODE_ZEROING);
    fma(s, t3, r, mask);
    vmuls(t4, result, -1.0f, mask, MODE_ZEROING);
    fma(r, t4, result, mask);
    fma(s, variance, r, mask);
    vmul(s, s, result, mask, MODE_ZEROING);
    fma(result, s, half, mask);
    RegTensor<float> zero, infinity;
    MaskReg is_infinite, is_zero;
    vdup(zero, 0.0f, mask, MODE_ZEROING);
    vdup(infinity, std::numeric_limits<float>::infinity(), mask, MODE_ZEROING);
    vcmps_eq(is_infinite, variance, std::numeric_limits<float>::infinity(), mask);
    vcmps_eq(is_zero, variance, 0.0f, mask);
    vsel(result, zero, result, is_infinite);
    vsel(result, infinity, result, is_zero);
}
template <typename T>
__aicore__ inline void load(RegTensor<float>& value, __ubuf__ T* src, int offset, MaskReg& mask) {
    if constexpr (std::is_same_v<T, float>) {
        vlds(value, src, offset, NORM);
    } else {
        RegTensor<bfloat16_t> raw;
        vlds(raw, src, offset, UNPK_B16);
        vcvt(value, raw, mask, PART_EVEN);
    }
}
template <int Width, typename T>
__aicore__ inline void row_variance(RegTensor<float>& sum, __ubuf__ T* src, int offset,
                                  MaskReg& full) {
    constexpr int Fold = fold_point(Width);
    constexpr int Blocks = Fold / 64;
    RegTensor<float> value, folded, partial, partials;
    RegTensor<uint32_t> lane;
    vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
    vdup(partials, 0.0f, full, MODE_ZEROING);
    for (uint16_t block = 0; block < Blocks; ++block) {
        load(value, src, offset + block * 64, full);
        vmul(value, value, value, full, MODE_ZEROING);
        if (block * 64 < Width - Fold) {
            load(folded, src, offset + Fold + block * 64, full);
            vmul(folded, folded, folded, full, MODE_ZEROING);
            vadd(value, value, folded, full, MODE_ZEROING);
        }
        vcadd(partial, value, full, MODE_ZEROING);
        vdup(partial, partial, full, POS_LOWEST, MODE_ZEROING);
        MaskReg slot;
        vcmps_eq(slot, lane, static_cast<uint32_t>(block), full);
        vsel(partials, partial, partials, slot);
    }
    uint32_t count = Blocks;
    MaskReg active = plt_b32(count, POST_UPDATE);
    vcadd(sum, partials, active, MODE_ZEROING);
    vmuls(sum, sum, 1.0f / static_cast<float>(Width), full, MODE_ZEROING);

}
template <int Width, typename T>
__aicore__ inline void row_inverse(RegTensor<float>& inverse, __ubuf__ T* src, int offset,
                                  MaskReg& full) {
    RegTensor<float> variance;
    row_variance<Width>(variance, src, offset, full);
    inverse_root(inverse, variance, full);
    vdup(inverse, inverse, full, POS_LOWEST, MODE_ZEROING);
}
#endif
} // namespace recipe_rms
