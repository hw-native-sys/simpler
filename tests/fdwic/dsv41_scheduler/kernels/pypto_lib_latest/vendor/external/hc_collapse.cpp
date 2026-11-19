// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Four-stream HC collapse, optional RMSNorm, and optional raw BF16 output.
// Four-stream HC collapse and RMSNorm with optional collapsed output.
#include <cstdint>
#if defined(__CPU_SIM)
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
#if defined(__DAV_VEC__)
using Input = Tile<TileType::Vec, float, 1, 20480, BLayout::RowMajor, DYNAMIC, 20480>;
using Output = Tile<TileType::Vec, bfloat16_t, 1, 5120, BLayout::RowMajor, DYNAMIC, 5120>;
using Premix = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
template <bool StoreCollapsed>
__tf__ __aicore__ inline void collapse_norm(
    Output::TileDType __out__ output, Input::TileDType __in__ input,
    Vec<bfloat16_t, 1, 5120>::TileDType __in__ weight,
    Premix::TileDType __in__ premix, uint16_t rows, uint16_t row_offset,
    Output::TileDType __out__ collapsed) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto mix = (__ubuf__ float*)__cce_get_tile_ptr(premix);
    auto raw = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(collapsed);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float partial[64] = {};
        for (int k = 0; k < 5120; ++k) {
            volatile float y0 = src[row * 20480 + k] * mix[(row + row_offset) * 4];
            volatile float y1 = src[row * 20480 + 5120 + k] * mix[(row + row_offset) * 4 + 1];
            volatile float y2 = src[row * 20480 + 10240 + k] * mix[(row + row_offset) * 4 + 2];
            volatile float y3 = src[row * 20480 + 15360 + k] * mix[(row + row_offset) * 4 + 3];
            volatile float first_pair = y0 + y1;
            volatile float second_pair = y2 + y3;
            bfloat16_t rounded = static_cast<bfloat16_t>(first_pair + second_pair);
            dst[row * 5120 + k] = rounded;
            if constexpr (StoreCollapsed) {
                raw[row * 5120 + k] = rounded;
            }
            float value = static_cast<float>(rounded);
            partial[k % 64] += value * value;
        }
        for (int width = 64; width > 1; width /= 2) {
            for (int lane = 0; lane < width / 2; ++lane)
                partial[lane] = partial[lane * 2] + partial[lane * 2 + 1];
        }
        float inverse = 1.0f / std::sqrt(partial[0] / 5120.0f + 1.0e-6f);
        for (int k = 0; k < 5120; ++k) {
            dst[row * 5120 + k] = static_cast<bfloat16_t>(
                static_cast<float>(dst[row * 5120 + k]) * inverse * static_cast<float>(gamma[k]));
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> p0, p1, p2, p3, sum;
            vlds(p0, mix, (row + row_offset) * 4, BRC_B32);
            vlds(p1, mix, (row + row_offset) * 4 + 1, BRC_B32);
            vlds(p2, mix, (row + row_offset) * 4 + 2, BRC_B32);
            vlds(p3, mix, (row + row_offset) * 4 + 3, BRC_B32);
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t k = 0; k < 80; ++k) {
                RegTensor<float> x0, x1, x2, x3, a, b, value, square;
                RegTensor<bfloat16_t> rounded;
                vlds(x0, src, row * 20480 + k * 64, NORM);
                vlds(x1, src, row * 20480 + 5120 + k * 64, NORM);
                vlds(x2, src, row * 20480 + 10240 + k * 64, NORM);
                vlds(x3, src, row * 20480 + 15360 + k * 64, NORM);
                vmul(x0, x0, p0, full, MODE_ZEROING);
                vmul(x1, x1, p1, full, MODE_ZEROING);
                vmul(x2, x2, p2, full, MODE_ZEROING);
                vmul(x3, x3, p3, full, MODE_ZEROING);
                vadd(a, x0, x1, full, MODE_ZEROING);
                vadd(b, x2, x3, full, MODE_ZEROING);
                vadd(value, a, b, full, MODE_ZEROING);
                vcvt(rounded, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(rounded, dst, row * 5120 + k * 64, PK_B32, full);
                if constexpr (StoreCollapsed) {
                    vsts(rounded, raw, row * 5120 + k * 64, PK_B32, full);
                }
                vcvt(value, rounded, full, PART_EVEN);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            RegTensor<float> inverse, one;
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(inverse, sum, 1.0f / 5120.0f, full, MODE_ZEROING);
            vadds(inverse, inverse, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inverse, inverse, full, MODE_ZEROING);
            vdup(one, 1.0f, full, MODE_ZEROING);
            vdiv(inverse, one, inverse, full, MODE_ZEROING);
            vdup(inverse, inverse, full, POS_LOWEST, MODE_ZEROING);
            mem_bar(VST_VLD);
            for (uint16_t k = 0; k < 80; ++k) {
                RegTensor<float> value, gain;
                RegTensor<bfloat16_t> rounded;
                vlds(rounded, dst, row * 5120 + k * 64, UNPK_B16);
                vcvt(value, rounded, full, PART_EVEN);
                vlds(rounded, gamma, k * 64, UNPK_B16);
                vcvt(gain, rounded, full, PART_EVEN);
                vmul(value, value, inverse, full, MODE_ZEROING);
                vmul(value, value, gain, full, MODE_ZEROING);
                vcvt(rounded, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(rounded, dst, row * 5120 + k * 64, PK_B32, full);
            }
        }
    }
#endif
}
#endif
}
template <bool StoreCollapsed>
__aicore__ inline void normalize(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    constexpr int row_tile = StoreCollapsed ? 2 : 8;
    for (int t0 = get_block_idx(args) * row_tile; t0 < args[5]; t0 += get_block_num(args) * row_tile) {
        int rows = args[5] - t0;
        rows = rows > row_tile ? row_tile : rows;
        // Two 80-KiB input slots and two 10-KiB output slots precede gamma and premix.
        Vec<bfloat16_t, 1, 5120> weight;
        Premix mix(rows * 4);
        TASSIGN(weight, 184320);
        TASSIGN(mix, 194560);
        GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, 5120>, BaseShape2D<bfloat16_t, 1, 5120>> gamma(
            data<bfloat16_t>(args, 2));
        using MixShape = TileShape2D<float, 1, DYNAMIC>;
        GlobalTensor<float, MixShape, BaseShape2D<float, 1, 32>> premix(data<float>(args, 1) + t0 * 4, MixShape(1, rows * 4));
        TLOAD(weight, gamma);
        TLOAD(mix, premix);
        using InShape = TileShape2D<float, 1, 20480>;
        using OutShape = TileShape2D<bfloat16_t, 1, 5120>;
        Input first(1);
        TASSIGN(first, 0);
        GlobalTensor<float, InShape, BaseShape2D<float, 1, 20480>> first_source(data<float>(args, 0) + t0 * 20480);
        TLOAD(first, first_source);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // V-to-MTE2 credits protect input reuse; MTE3-to-V credits protect output reuse.
        for (int local = 0; local < rows; ++local) {
            int slot = local % 2;
            event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
            if (local + 1 < rows) {
                if (local > 0) wait_flag(PIPE_V, PIPE_MTE2, next);
                Input pending(1);
                TASSIGN(pending, (1 - slot) * 81920);
                GlobalTensor<float, InShape, BaseShape2D<float, 1, 20480>> source(
                    data<float>(args, 0) + (t0 + local + 1) * 20480);
                TLOAD(pending, source);
                set_flag(PIPE_MTE2, PIPE_V, next);
            }
            Input input(1);
            Output output(1);
            TASSIGN(input, slot * 81920);
            TASSIGN(output, 163840 + slot * 10240);
            wait_flag(PIPE_MTE2, PIPE_V, current);
            wait_flag(PIPE_MTE3, PIPE_V, current);
            Output collapsed(1);
            TASSIGN(collapsed, 194688 + slot * 10240);
            if constexpr (StoreCollapsed) {
                collapse_norm<true>(output.data(), input.data(), weight.data(), mix.data(), 1, local, collapsed.data());
                if (t0 + local >= args[6]) {
                    pipe_barrier(PIPE_ALL);
                    TEXPANDS(output, 0.0f);
                }
            } else {
                collapse_norm<false>(output.data(), input.data(), weight.data(), mix.data(), 1, local, output.data());
            }
            set_flag(PIPE_V, PIPE_MTE2, current);
            set_flag(PIPE_V, PIPE_MTE3, current);
            wait_flag(PIPE_V, PIPE_MTE3, current);
            GlobalTensor<bfloat16_t, OutShape, BaseShape2D<bfloat16_t, 1, 5120>> target(
                data<bfloat16_t>(args, 3) + (t0 + local) * 5120);
            TSTORE(target, output);
            if constexpr (StoreCollapsed) {
                GlobalTensor<bfloat16_t, OutShape, BaseShape2D<bfloat16_t, 1, 5120>> raw_target(
                    data<bfloat16_t>(args, 4) + (t0 + local) * 5120);
                TSTORE(raw_target, collapsed);
            }
            set_flag(PIPE_MTE3, PIPE_V, current);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if (rows > 1) wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }
#endif
}

// Four-stream HC collapse with pairwise FP32 sums and BF16 rounding.
namespace hc_collapse_detail {
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
#if defined(__DAV_VEC__)
using Input = Tile<TileType::Vec, float, 1, 20480, BLayout::RowMajor, DYNAMIC, 20480>;
using Output = Tile<TileType::Vec, bfloat16_t, 1, 5120, BLayout::RowMajor, DYNAMIC, 5120>;
using Premix = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
__tf__ __aicore__ inline void collapse(
    Output::TileDType __out__ output, Input::TileDType __in__ input,
    Premix::TileDType __in__ premix, uint16_t rows, uint16_t row_offset) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto mix = (__ubuf__ float*)__cce_get_tile_ptr(premix);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < 5120; ++k) {
            float y0 = src[row * 20480 + k] * mix[(row + row_offset) * 4];
            float y1 = src[row * 20480 + 5120 + k] * mix[(row + row_offset) * 4 + 1];
            float y2 = src[row * 20480 + 10240 + k] * mix[(row + row_offset) * 4 + 2];
            float y3 = src[row * 20480 + 15360 + k] * mix[(row + row_offset) * 4 + 3];
            bfloat16_t rounded = static_cast<bfloat16_t>((y0 + y1) + (y2 + y3));
            dst[row * 5120 + k] = rounded;
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> p0, p1, p2, p3;
            vlds(p0, mix, (row + row_offset) * 4, BRC_B32);
            vlds(p1, mix, (row + row_offset) * 4 + 1, BRC_B32);
            vlds(p2, mix, (row + row_offset) * 4 + 2, BRC_B32);
            vlds(p3, mix, (row + row_offset) * 4 + 3, BRC_B32);
            for (uint16_t k = 0; k < 80; ++k) {
                RegTensor<float> x0, x1, x2, x3, a, b, value;
                RegTensor<bfloat16_t> rounded;
                vlds(x0, src, row * 20480 + k * 64, NORM);
                vlds(x1, src, row * 20480 + 5120 + k * 64, NORM);
                vlds(x2, src, row * 20480 + 10240 + k * 64, NORM);
                vlds(x3, src, row * 20480 + 15360 + k * 64, NORM);
                vmul(x0, x0, p0, full, MODE_ZEROING);
                vmul(x1, x1, p1, full, MODE_ZEROING);
                vmul(x2, x2, p2, full, MODE_ZEROING);
                vmul(x3, x3, p3, full, MODE_ZEROING);
                vadd(a, x0, x1, full, MODE_ZEROING);
                vadd(b, x2, x3, full, MODE_ZEROING);
                vadd(value, a, b, full, MODE_ZEROING);
                vcvt(rounded, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(rounded, dst, row * 5120 + k * 64, PK_B32, full);
            }
        }
    }
#endif
}
#endif
__aicore__ inline void run(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int tokens = args[5];
    for (int t0 = get_block_idx(args) * 8; t0 < tokens; t0 += get_block_num(args) * 8) {
        int rows = tokens - t0;
        rows = rows > 8 ? 8 : rows;
        // Two 80-KiB input slots and two 10-KiB output slots precede premix.
        Premix mix(rows * 4);
        TASSIGN(mix, 184320);
        using MixShape = TileShape2D<float, 1, DYNAMIC>;
        GlobalTensor<float, MixShape, BaseShape2D<float, 1, 32>> premix(data<float>(args, 1) + t0 * 4, MixShape(1, rows * 4));
        TLOAD(mix, premix);
        using InShape = TileShape2D<float, 1, 20480>;
        using OutShape = TileShape2D<bfloat16_t, 1, 5120>;
        Input first(1);
        TASSIGN(first, 0);
        GlobalTensor<float, InShape, BaseShape2D<float, 1, 20480>> first_source(data<float>(args, 0) + t0 * 20480);
        TLOAD(first, first_source);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // V-to-MTE2 credits protect input reuse; MTE3-to-V credits protect output reuse.
        for (int local = 0; local < rows; ++local) {
            int slot = local % 2;
            event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
            event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
            if (local + 1 < rows) {
                if (local > 0) wait_flag(PIPE_V, PIPE_MTE2, next);
                Input pending(1);
                TASSIGN(pending, (1 - slot) * 81920);
                GlobalTensor<float, InShape, BaseShape2D<float, 1, 20480>> source(
                    data<float>(args, 0) + (t0 + local + 1) * 20480);
                TLOAD(pending, source);
                set_flag(PIPE_MTE2, PIPE_V, next);
            }
            Input input(1);
            Output output(1);
            TASSIGN(input, slot * 81920);
            TASSIGN(output, 163840 + slot * 10240);
            wait_flag(PIPE_MTE2, PIPE_V, current);
            wait_flag(PIPE_MTE3, PIPE_V, current);
            collapse(output.data(), input.data(), mix.data(), 1, local);
            set_flag(PIPE_V, PIPE_MTE2, current);
            set_flag(PIPE_V, PIPE_MTE3, current);
            wait_flag(PIPE_V, PIPE_MTE3, current);
            GlobalTensor<bfloat16_t, OutShape, BaseShape2D<bfloat16_t, 1, 5120>> target(
                data<bfloat16_t>(args, 3) + (t0 + local) * 5120);
            TSTORE(target, output);
            set_flag(PIPE_MTE3, PIPE_V, current);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if (rows > 1) wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }
#endif
}
} // namespace hc_collapse_detail

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    if (args[7] == 0) {
        hc_collapse_detail::run(args);
    } else if (args[7] == 1) {
        normalize<false>(args);
    } else {
        normalize<true>(args);
    }
#endif
}
