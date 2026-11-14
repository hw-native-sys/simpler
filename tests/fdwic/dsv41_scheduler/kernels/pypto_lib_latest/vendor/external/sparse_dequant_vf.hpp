// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once
#include <pto/pto-inst.hpp>

namespace sparse_dequant {
using namespace pto;

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <bool FP4, typename Output, typename Payload, typename Indices>
__tf__ __aicore__ inline void decode_nz(
    typename Output::TileDType __out__ dst, typename Payload::TileDType __in__ src,
    typename Indices::TileDType __in__ ids) {
    auto output = (__ubuf__ uint16_t*)__cce_get_tile_ptr(dst);
    auto input = (__ubuf__ uint8_t*)__cce_get_tile_ptr(src);
    auto scale = (__ubuf__ uint16_t*)(input + (FP4 ? 256 : 512));
    auto indices = (__ubuf__ int32_t*)__cce_get_tile_ptr(ids);
    constexpr int nz_rows = Output::Rows + (Output::Compact == CompactMode::RowPlusOne ? 1 : 0);
    __VEC_SCOPE__ {
        MaskReg all8 = pset_b8(PAT_ALL);
        MaskReg all16 = pset_b16(PAT_ALL);
        MaskReg all32 = pset_b32(PAT_ALL);
        RegTensor<uint16_t> group16;
        RegTensor<float> zero;
        vdup(zero, 0.0f, all32, MODE_ZEROING);
        if constexpr (!FP4) {
            vci((vector_s16&)group16, 0, INC_ORDER);
            vshrs(group16, group16, (uint16_t)5, all16, MODE_ZEROING);
        }
        for (uint16_t row = 0; row < Output::Rows; ++row) {
            RegTensor<int32_t> index;
            MaskReg valid;
            vlds(index, indices, row, BRC_B32);
            vcmps_ge(valid, index, 0, all32);
            if constexpr (FP4) {
#pragma unroll
                for (uint16_t chunk = 0; chunk < 4; ++chunk) {
                    RegTensor<bfloat16_t> packed, scale16;
                    vector_f4e2m1x2 raw;
                    vlds((RegTensor<uint8_t>&)raw, input, row * 544 + chunk * 64, UNPK4_B8);
                    vcvt(packed, raw, all8, PART_P0);
                    vlds((RegTensor<uint16_t>&)scale16,
                         scale + row * 272 + chunk * 8, 0, E2B_B16);
                    vmul((vector_bf16&)packed, (vector_bf16&)packed, (vector_bf16&)scale16,
                         all16, MODE_ZEROING);
                    // Each int32 predicate lane selects both adjacent BF16 values.
                    vsel((RegTensor<uint32_t>&)packed, (RegTensor<uint32_t>&)packed,
                         (RegTensor<uint32_t>&)zero, valid);
                    auto target = output + chunk * nz_rows * 128 + row * 16;
                    vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(nz_rows << 16), all16, POST_UPDATE);
                }
            } else {
#pragma unroll
                for (uint16_t chunk = 0; chunk < 4; ++chunk) {
                    RegTensor<bfloat16_t> b0, b1, packed, unused;
                    RegTensor<float> x0, x1;
                    vector_f8e4m3 raw0, raw1;
                    vlds((RegTensor<uint8_t>&)raw0, input, row * 544 + chunk * 128, UNPK4_B8);
                    vlds((RegTensor<uint8_t>&)raw1, input, row * 544 + chunk * 128 + 64, UNPK4_B8);
                    vcvt(x0, raw0, all8, PART_P0);
                    vcvt(x1, raw1, all8, PART_P0);
                    // Every finite E4M3 value is exactly representable in BF16.
                    // Match the FP4 path: multiply by BF16 scales in 128 lanes.
                    vcvt(b0, x0, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                    vcvt(b1, x1, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                    vdintlv(packed, unused, b0, b1);
                    RegTensor<bfloat16_t> scale16;
                    vgather2((RegTensor<uint16_t>&)scale16,
                             scale + row * 272 + chunk * 4, group16, all16);
                    vmul((vector_bf16&)packed, (vector_bf16&)packed, (vector_bf16&)scale16,
                         all16, MODE_ZEROING);
                    vsel((RegTensor<uint32_t>&)packed, (RegTensor<uint32_t>&)packed,
                         (RegTensor<uint32_t>&)zero, valid);
                    auto target = output + chunk * nz_rows * 128 + row * 16;
                    vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(nz_rows << 16), all16, POST_UPDATE);
                }
            }
        }
    }
}
#endif
} // namespace sparse_dequant
