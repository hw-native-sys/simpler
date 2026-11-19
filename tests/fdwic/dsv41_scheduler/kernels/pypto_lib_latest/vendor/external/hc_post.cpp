// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Four-stream HC post mixing with ordered FP32 multiply and add.
#include <cstdint>
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
using Hidden = Vec<bfloat16_t, 1, 2560>;
using Streams = Vec<float, 4, 2560>;
using Mix = Vec<float, 1, 32>;
__tf__ __aicore__ inline void mix_streams(
    Streams::TileDType __out__ output, Hidden::TileDType __in__ hidden,
    Streams::TileDType __in__ residual, Mix::TileDType __in__ coefficients,
    uint16_t first, uint16_t count) {
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto x = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(hidden);
    auto r = (__ubuf__ float*)__cce_get_tile_ptr(residual);
    auto c = (__ubuf__ float*)__cce_get_tile_ptr(coefficients);
#if defined(__CPU_SIM)
    for (int h = first; h < first + count; ++h) {
        for (int k = 0; k < 2560; ++k) {
            float value = static_cast<float>(x[k]) * c[h];
            for (int i = 0; i < 4; ++i) value = value + r[i * 2560 + k] * c[8 + i * 4 + h];
            dst[h * 2560 + k] = value;
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t k = 0; k < 40; ++k) {
            RegTensor<bfloat16_t> rounded;
            RegTensor<float> value, x0, r0, r1, r2, r3, coef, term;
            vlds(rounded, x, k * 64, UNPK_B16);
            vcvt(x0, rounded, full, PART_EVEN);
            vlds(r0, r, k * 64, NORM);
            vlds(r1, r, 2560 + k * 64, NORM);
            vlds(r2, r, 5120 + k * 64, NORM);
            vlds(r3, r, 7680 + k * 64, NORM);
            for (uint16_t h = first; h < first + count; ++h) {
                vlds(coef, c, h, BRC_B32);
                vmul(value, x0, coef, full, MODE_ZEROING);
                vlds(coef, c, 8 + h, BRC_B32);
                vmul(term, r0, coef, full, MODE_ZEROING);
                vadd(value, value, term, full, MODE_ZEROING);
                vlds(coef, c, 12 + h, BRC_B32);
                vmul(term, r1, coef, full, MODE_ZEROING);
                vadd(value, value, term, full, MODE_ZEROING);
                vlds(coef, c, 16 + h, BRC_B32);
                vmul(term, r2, coef, full, MODE_ZEROING);
                vadd(value, value, term, full, MODE_ZEROING);
                vlds(coef, c, 20 + h, BRC_B32);
                vmul(term, r3, coef, full, MODE_ZEROING);
                vadd(value, value, term, full, MODE_ZEROING);
                vsts(value, dst, h * 2560 + k * 64, NORM_B32, full);
            }
        }
    }
#endif
}
__aicore__ inline void process(__gm__ int64_t* args, int chunk, int first, int count) {
    int row = chunk / 2, column = chunk % 2 * 2560;
    Hidden hidden;
    Streams residual, output;
    using Post = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4>;
    using ResMix = Vec<float, 1, 16>;
    Post post;
    ResMix res_mix;
    Mix mix;
    TASSIGN(hidden, 0);
    TASSIGN(residual, 5120);
    TASSIGN(output, 46080);
    TASSIGN(post, 87040);
    TASSIGN(res_mix, 87072);
    TASSIGN(mix, 87040);
    GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, 2560>, BaseShape2D<bfloat16_t, 1, 5120>> x(
        data<bfloat16_t>(args, 0) + row * 5120 + column);
    GlobalTensor<float, TileShape2D<float, 4, 2560>, BaseShape2D<float, 4, 5120>> r(
        data<float>(args, 1) + row * 20480 + column);
    GlobalTensor<float, TileShape2D<float, 1, 4>, BaseShape2D<float, 1, 4>> p(
        data<float>(args, 2) + row * 4);
    GlobalTensor<float, TileShape2D<float, 1, 16>, BaseShape2D<float, 1, 16>> m(
        data<float>(args, 3) + row * 16);
    TLOAD(hidden, x);
    TLOAD(residual, r);
    TLOAD(post, p);
    TLOAD(res_mix, m);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    mix_streams(output.data(), hidden.data(), residual.data(), mix.data(), first, count);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (count == 4) {
        GlobalTensor<float, TileShape2D<float, 4, 2560>, BaseShape2D<float, 4, 5120>> y(
            data<float>(args, 4) + row * 20480 + column);
        TSTORE(y, output);
    } else {
        Vec<float, 1, 2560> tail;
        TASSIGN(tail, 46080 + first * 10240);
        GlobalTensor<float, TileShape2D<float, 1, 2560>, BaseShape2D<float, 1, 5120>> y(
            data<float>(args, 4) + row * 20480 + first * 5120 + column);
        TSTORE(y, tail);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int worker = get_block_idx(args);
    int total = args[5] * 2, full = total / 64 * 64;
    for (int chunk = worker; chunk < full; chunk += 64) process(args, chunk, 0, 4);
    for (int tail = worker; tail < (total - full) * 4; tail += 64) {
        process(args, full + tail / 4, tail % 4, 1);
    }
#endif
}
