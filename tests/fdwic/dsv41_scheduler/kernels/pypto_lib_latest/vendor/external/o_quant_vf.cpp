// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <cstring>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "o_quant_vf.hpp"
using namespace pto;
namespace {
template <typename T, int R, int C> using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T> __aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}
__aicore__ inline int64_t stride(__gm__ int64_t* args, int i) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->strides[0];
}
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <int Rows>
__tf__ __aicore__ inline void rotate_tail(
    typename Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512>::TileDType __out__ output, typename Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512>::TileDType __in__ input,
    typename Tile<TileType::Vec, float, Rows, 32, BLayout::RowMajor, DYNAMIC, 32>::TileDType __in__ cosine, typename Tile<TileType::Vec, float, Rows, 32, BLayout::RowMajor, DYNAMIC, 32>::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cos = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sin = (__ubuf__ float*)__cce_get_tile_ptr(sine);
    __VEC_SCOPE__ {
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
        for (uint16_t row = 0; row < rows; ++row) {
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
            vsts(raw, dst, row * 512 + 448, PK_B32, all);
        }
    }
}
template <int Rows>
__aicore__ inline void run_native(__gm__ int64_t* args) {
    int active = args[5], groups = args[6], heads = groups * 8, capacity_rows = args[7];
    using Input = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Prefix = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 448>;
    using Tail = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 64>;
    using Trig = Tile<TileType::Vec, float, Rows, 32, BLayout::RowMajor, DYNAMIC, 32>;
    using TrigShape = TileShape2D<float, DYNAMIC, 32>;
    using PrefixShape = TileShape2D<bfloat16_t, DYNAMIC, 448>;
    using TailShape = TileShape2D<bfloat16_t, DYNAMIC, 64>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int work = get_block_idx(args); work < ((capacity_rows + Rows - 1) / Rows) * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * Rows;
        int capacity = capacity_rows - t0; capacity = capacity > Rows ? Rows : capacity;
        int rows = active - t0; rows = rows < 0 ? 0 : (rows > capacity ? capacity : rows);
        Input input(rows);
        Trig cosine(rows), sine(rows);
        TASSIGN(cosine, Rows * 1600);
        TASSIGN(sine, Rows * 1728);
        Tile<TileType::Vec, float8_e4m3_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512> output(capacity);
        Tile<TileType::Vec, uint8_t, Rows / 16, 256, BLayout::RowMajor, DYNAMIC, 256> scales(capacity / 16);
        TASSIGN(input, 0);
        TASSIGN(output, Rows * 1024); TASSIGN(scales, Rows * 1536);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if (rows > 0) {
            Prefix prefix(rows);
            Tail tail(rows);
            TASSIGN(prefix, 0);
            TASSIGN(tail, 896);
            GlobalTensor<bfloat16_t, PrefixShape, DynamicStride> input_gm(
                data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + h * 512,
                PrefixShape(rows, 448), DynamicStride(1, 1, 1, stride(args, 0), 1));
            GlobalTensor<bfloat16_t, TailShape, DynamicStride> tail_gm(
                data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + h * 512 + 448,
                TailShape(rows, 64), DynamicStride(1, 1, 1, stride(args, 0), 1));
            GlobalTensor<float, TrigShape, DynamicStride> cos_gm(data<float>(args, 1) + t0 * stride(args, 1),
                TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 1), 1));
            GlobalTensor<float, TrigShape, DynamicStride> sin_gm(data<float>(args, 2) + t0 * stride(args, 2),
                TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 2), 1));
            TLOAD(cosine, cos_gm);
            TLOAD(sine, sin_gm);
            TLOAD(prefix, input_gm);
            TLOAD(tail, tail_gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        rotate_tail<Rows>(input.data(), input.data(), cosine.data(), sine.data(), rows);
        pipe_barrier(PIPE_ALL);
        o_quant::quant_vf<decltype(output), decltype(scales), Input>(output.data(), scales.data(), input.data(), rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int group = h / 8, local_h = h % 8;
        using OutShape = TileShape2D<float8_e4m3_t, DYNAMIC, 512>;
        GlobalTensor<float8_e4m3_t, OutShape, DynamicStride> output_gm(data<float8_e4m3_t>(args, 3) + (group * capacity_rows + t0) * stride(args, 3) + local_h * 512,
            OutShape(capacity, 512), DynamicStride(1, 1, 1, stride(args, 3), 1));
        using ScaleShape = TileShape2D<uint8_t, DYNAMIC, 256>;
        GlobalTensor<uint8_t, ScaleShape, DynamicStride> scale_gm(
            data<uint8_t>(args, 4) + group * stride(args, 4) + t0 * 128 + local_h * 256,
            ScaleShape(capacity / 16, 256), DynamicStride(1, 1, 1, 2048, 1));
        TSTORE(output_gm, output); TSTORE(scale_gm, scales);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
__aicore__ inline void run_cpu(__gm__ int64_t* args) {
    int active = args[5], heads = args[6] * 8, capacity_rows = args[7];
    for (int work = get_block_idx(args); work < (capacity_rows / 16) * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * 16;
        for (int r = 0; r < 16; ++r) {
            float value[512] = {};
            if (t0 + r < active) {
                auto src = data<bfloat16_t>(args, 0) + (t0 + r) * stride(args, 0) + h * 512;
                for (int d = 0; d < 448; ++d) value[d] = static_cast<float>(src[d]);
                for (int pair = 0; pair < 32; ++pair) {
                    float c = data<float>(args, 1)[(t0 + r) * stride(args, 1) + pair];
                    float si = -data<float>(args, 2)[(t0 + r) * stride(args, 2) + pair];
                    float re = static_cast<float>(src[448 + pair * 2]);
                    float im = static_cast<float>(src[449 + pair * 2]);
                    value[448 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(re * c - im * si));
                    value[449 + pair * 2] = static_cast<float>(static_cast<bfloat16_t>(im * c + re * si));
                }
            }
            for (int g = 0; g < 16; ++g) {
                float maximum = 0;
                for (int d = 0; d < 32; ++d) maximum = std::max(maximum, std::fabs(value[g * 32 + d]));
                uint32_t bits; std::memcpy(&bits, &maximum, 4);
                int e = std::max(int(bits >> 23) - 8, 0);
                uint32_t dbits = uint32_t(std::max(e, 1)) << 23; float divisor; std::memcpy(&divisor, &dbits, 4);
                for (int d = 0; d < 32; ++d) {
                    float v = value[g * 32 + d] / divisor * float(e != 0);
                    v = std::min(std::max(v, -448.0f), 448.0f);
                    data<float8_e4m3_t>(args, 3)[(h / 8 * capacity_rows + t0 + r) * stride(args, 3) + h % 8 * 512 + g * 32 + d] = float8_e4m3_t(v, RoundMode::CAST_RINT);
                }
                data<uint8_t>(args, 4)[h / 8 * stride(args, 4) + t0 * 128 + h % 8 * 256 + g / 2 * 32 + r * 2 + g % 2] = e;
            }
        }
    }
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__CPU_SIM)
    run_cpu(args);
#elif defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    if (args[6] >= 8) run_native<80>(args);
    else if (args[6] >= 4) run_native<48>(args);
    else if (args[6] >= 2) run_native<32>(args);
    else run_native<16>(args);
#endif
}
