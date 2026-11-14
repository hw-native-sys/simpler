// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// RMS normalization, latent output, rotary embedding, and C1A cache publication.
#if defined(__CPU_SIM)
#include <algorithm>
#include <bit>
#include <cmath>
#endif
#include "compressor_publish.h"
namespace {
#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void normalize(
    Batch<bfloat16_t, 512>::TileDType __out__ output,
    Batch<bfloat16_t, 512>::TileDType __in__ input,
    Vec<bfloat16_t, 512>::TileDType __in__ weight, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float square = 0;
        for (int col = 0; col < 512; ++col) {
            float value = static_cast<float>(src[row * 512 + col]);
            square += value * value;
        }
        float inv = 1.0f / std::sqrt(square / 512.0f + 1.0e-6f);
        for (int col = 0; col < 512; ++col)
            dst[row * 512 + col] = static_cast<bfloat16_t>(static_cast<float>(src[row * 512 + col]) * inv * static_cast<float>(gamma[col]));
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> sum, value, square, inv, one;
            RegTensor<bfloat16_t> bf;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                vlds(bf, src, row * 512 + chunk * 64, UNPK_B16);
                vcvt(value, bf, full, PART_EVEN);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(inv, sum, 1.0f / 512.0f, full, MODE_ZEROING);
            vadds(inv, inv, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inv, inv, full, MODE_ZEROING);
            vdup(one, 1.0f, full, MODE_ZEROING);
            vdiv(inv, one, inv, full, MODE_ZEROING);
            vdup(inv, inv, full, POS_LOWEST, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<float> gain;
                vlds(bf, src, row * 512 + chunk * 64, UNPK_B16);
                vcvt(value, bf, full, PART_EVEN);
                vlds(bf, gamma, chunk * 64, UNPK_B16);
                vcvt(gain, bf, full, PART_EVEN);
                vmul(value, value, inv, full, MODE_ZEROING);
                vmul(value, value, gain, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(bf, dst, row * 512 + chunk * 64, PK_B32, full);
            }
        }
    }
#endif
}

__aicore__ inline void flush(__gm__ int64_t* args, int rows, Vec<int64_t, 8>& targets, Vec<int32_t, 8>& tokens) {
    Batch<bfloat16_t, 512> input;
    Vec<bfloat16_t, 512> weight;
    Batch<float, 32> cosine, sine;
    Batch<uint8_t, 256> packed;
    Batch<bfloat16_t, 32> scales;
    Vec<float, 32> scratch;
    TASSIGN(input, 0);
    TASSIGN(packed, 8192);
    TASSIGN(scales, 10240);
    TASSIGN(scratch, 10752);
    TASSIGN(cosine, 11200);
    TASSIGN(sine, 12224);
    TASSIGN(weight, 13312);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    normalize(input.data(), input.data(), weight.data(), rows);
    auto cache_tensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
    auto cache_data = data<uint8_t>(args, 2);
    auto scale_data = reinterpret_cast<__gm__ bfloat16_t*>(cache_data + 256);
    int64_t scale_stride = cache_tensor->strides[1] / sizeof(bfloat16_t);
    publish_rows<true>(input, cosine, sine, packed, scales, scratch, targets, tokens, rows,
                       0, 8192, 10240, data<bfloat16_t>(args, 6), cache_data, scale_data,
                       cache_tensor->strides[1], scale_stride);
}

__aicore__ inline void run(__gm__ int64_t* args) {
    using Slots = Tile<TileType::Vec, int64_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
    using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 32>>;
    Slots slots(32);
    Vec<int64_t, 8> targets;
    Vec<int32_t, 8> tokens;
    Vec<bfloat16_t, 512> weight;
    TASSIGN(slots, 10880);
    TASSIGN(targets, 11136);
    TASSIGN(tokens, 14336);
    TASSIGN(weight, 13312);
    GM<bfloat16_t, 512> weight_gm(data<bfloat16_t>(args, 5));
    TLOAD(weight, weight_gm);
    int count_tokens = args[7];
    int worker = get_block_idx(args), workers = get_block_num(args);
    int pending = 0;
    for (int base = 0; base < count_tokens; base += 32) {
        int count = count_tokens - base < 32 ? count_tokens - base : 32;
        slots.SetValidCol(count);
        SlotGM slot_gm(data<int64_t>(args, 1) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
        TLOAD(slots, slot_gm);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int lane = 0; lane < count; ++lane) {
            int token = base + lane;
            int64_t slot = slots.GetValue(lane);
            int owner = slot < 0 ? (token / 8) % workers : (slot / 128) % workers;
            if (owner != worker) continue;
            targets.SetValue(pending, slot);
            tokens.SetValue(pending, token);
            Vec<bfloat16_t, 512> input;
            Vec<float, 32> cosine, sine;
            TASSIGN(input, pending * 1024);
            TASSIGN(cosine, 11200 + pending * 128);
            TASSIGN(sine, 12224 + pending * 128);
            GM<bfloat16_t, 512> input_gm(data<bfloat16_t>(args, 0) + (int64_t)token * 512);
            GM<float, 32> cos_gm(data<float>(args, 3) + (int64_t)token * 32);
            GM<float, 32> sin_gm(data<float>(args, 4) + (int64_t)token * 32);
            TLOAD(input, input_gm);
            TLOAD(cosine, cos_gm);
            TLOAD(sine, sin_gm);
            if (++pending == 8) {
                flush(args, pending, targets, tokens);
                pending = 0;
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
    if (pending > 0) flush(args, pending, targets, tokens);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    run(args);
#endif
}
