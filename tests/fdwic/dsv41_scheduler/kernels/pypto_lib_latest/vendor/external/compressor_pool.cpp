// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Pair pooling, RMS normalization, rotary cache publication, and ring-state update.
#if defined(__CPU_SIM)
#include <algorithm>
#include <bit>
#include <cmath>
#endif
#include "compressor_publish.h"
namespace {
#if defined(__DAV_VEC__)
template <typename T, int R, int C>
using TileRC = Tile<TileType::Vec, T, R, C>;
__tf__ __aicore__ inline void pool_rows(
    TileRC<bfloat16_t, 8, 512>::TileDType __out__ output,
    TileRC<float, 8, 512>::TileDType __in__ current_kv,
    TileRC<float, 8, 512>::TileDType __in__ current_score,
    TileRC<float, 8, 512>::TileDType __in__ previous_kv,
    TileRC<float, 8, 512>::TileDType __in__ previous_score,
    TileRC<bfloat16_t, 1, 512>::TileDType __in__ weight,
    TileRC<float, 1, 512>::TileDType __out__ scratch, uint16_t mask, uint16_t rows) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto ck = (__ubuf__ float*)__cce_get_tile_ptr(current_kv);
    auto cs = (__ubuf__ float*)__cce_get_tile_ptr(current_score);
    auto pk = (__ubuf__ float*)__cce_get_tile_ptr(previous_kv);
    auto ps = (__ubuf__ float*)__cce_get_tile_ptr(previous_score);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto rounded = (__ubuf__ float*)__cce_get_tile_ptr(scratch);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float square = 0;
        for (int col = 0; col < 512; ++col) {
            float v = 0;
            if (mask & (1u << row)) {
                int i = row * 512 + col;
                float highest = std::max(cs[i], ps[i]);
                float a = std::exp(ps[i] - highest), b = std::exp(cs[i] - highest);
                v = static_cast<float>(static_cast<bfloat16_t>(pk[i] * (a / (a + b)) + ck[i] * (b / (a + b))));
            }
            rounded[col] = v;
            square += v * v;
        }
        float inv = 1.0f / std::sqrt(square / 512.0f + 1.0e-6f);
        for (int col = 0; col < 512; ++col)
            dst[row * 512 + col] = static_cast<bfloat16_t>(rounded[col] * inv * static_cast<float>(gamma[col]));
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            if (!(mask & (1u << row))) {
                RegTensor<bfloat16_t> zero;
                MaskReg full16 = pset_b16(PAT_ALL);
                vdup(zero, (bfloat16_t)0.0f, full16, MODE_ZEROING);
                for (uint16_t chunk = 0; chunk < 4; ++chunk)
                    vsts(zero, dst, row * 512 + chunk * 128, NORM_B16, full16);
                continue;
            }
            RegTensor<float> sum;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                int offset = row * 512 + chunk * 64;
                RegTensor<float> a, b, high, av, bv, denominator, value, square;
                RegTensor<bfloat16_t> bf;
                vlds(a, ps, offset, NORM);
                vlds(b, cs, offset, NORM);
                vmax(high, a, b, full, MODE_ZEROING);
                vsub(a, a, high, full);
                vsub(b, b, high, full);
                vexp(a, a, full, MODE_ZEROING);
                vexp(b, b, full, MODE_ZEROING);
                vlds(av, pk, offset, NORM);
                vlds(bv, ck, offset, NORM);
                vadd(denominator, a, b, full, MODE_ZEROING);
                vdiv(a, a, denominator, full, MODE_ZEROING);
                vdiv(b, b, denominator, full, MODE_ZEROING);
                vmul(av, av, a, full, MODE_ZEROING);
                vmul(bv, bv, b, full, MODE_ZEROING);
                vadd(value, av, bv, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vcvt(value, bf, full, PART_EVEN);
                vsts(value, rounded, chunk * 64, NORM_B32, full);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(sum, sum, 1.0f / 512.0f, full, MODE_ZEROING);
            vadds(sum, sum, 1.0e-6f, full, MODE_ZEROING);
            RegTensor<float> one, inv, root;
            vdup(one, 1.0f, full, MODE_ZEROING);
            vsqrt(root, sum, full, MODE_ZEROING);
            vdiv(inv, one, root, full, MODE_ZEROING);
            // Broadcast the reduced RMS factor from lane zero.
            vdup(inv, inv, full, POS_LOWEST, MODE_ZEROING);
            mem_bar(VST_VLD);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<float> value, g;
                RegTensor<bfloat16_t> bf;
                vlds(value, rounded, chunk * 64, NORM);
                vlds(bf, gamma, chunk * 64, UNPK_B16);
                vcvt(g, bf, full, PART_EVEN);
                vmul(value, value, inv, full, MODE_ZEROING);
                vmul(value, value, g, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(bf, dst, row * 512 + chunk * 64, PK_B32, full);
            }
        }
    }
#endif
}

__aicore__ inline int read_metadata(__gm__ int32_t* ptr) {
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}

__aicore__ inline bool can_publish(__gm__ int64_t* args, int token) {
    int requests = reinterpret_cast<__gm__ Tensor*>(args[8])->shapes[0];
    int request = read_metadata(data<int32_t>(args, 7) + token);
    if (request < 0 || request >= requests) return false;
    int block = read_metadata(data<int32_t>(args, 8) + request);
    if (block < 0 || block >= args[14]) return false;
    int position = read_metadata(data<int32_t>(args, 3) + token);
    if (position < 0 || position % 2 == 0) return false;
    int start = read_metadata(data<int32_t>(args, 6) + request);
    int end = read_metadata(data<int32_t>(args, 6) + request + 1);
    return token >= start && token < end;
}

__aicore__ inline void flush(__gm__ int64_t* args, int rows, uint16_t active,
                            Vec<int64_t, 8>& targets, Vec<int32_t, 8>& tokens) {
    Batch<float, 512> current_kv, current_score, previous_kv, previous_score;
    Batch<bfloat16_t, 512> output;
    Vec<bfloat16_t, 512> weight;
    Vec<float, 512> pool_scratch;
    Batch<float, 32> cosine, sine;
    Batch<uint8_t, 256> packed;
    Batch<bfloat16_t, 32> scales;
    Vec<float, 32> quant_scratch;
    TASSIGN(current_kv, 0);
    TASSIGN(current_score, 16384);
    TASSIGN(previous_kv, 32768);
    TASSIGN(previous_score, 49152);
    TASSIGN(output, 65536);
    TASSIGN(weight, 73728);
    TASSIGN(pool_scratch, 74752);
    TASSIGN(cosine, 77504);
    TASSIGN(sine, 78528);
    TASSIGN(packed, 79552);
    TASSIGN(scales, 81600);
    TASSIGN(quant_scratch, 82112);
    // Only the final eligible occurrence of a cache slot writes it across state owners.
    using Slots = Tile<TileType::Vec, int64_t, 1, 128, BLayout::RowMajor, 1, DYNAMIC>;
    using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 128>>;
    Slots future(128);
    TASSIGN(future, 82240);
    int count_tokens = args[13];
    int first = tokens.GetValue(0);
    for (int row = 1; row < rows; ++row) {
        if (tokens.GetValue(row) < first) first = tokens.GetValue(row);
    }
    for (int base = first / 128 * 128; base < count_tokens; base += 128) {
        int count = count_tokens - base < 128 ? count_tokens - base : 128;
        future.SetValidCol(count);
        SlotGM slot_gm(data<int64_t>(args, 9) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
        TLOAD(future, slot_gm);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int row = 0; row < rows; ++row) {
            int64_t slot = targets.GetValue(row);
            if (slot < 0) continue;
            int start = tokens.GetValue(row) + 1 - base;
            if (start < 0) start = 0;
            for (int lane = start; lane < count; ++lane) {
                if (future.GetValue(lane) == slot && can_publish(args, base + lane)) {
                    targets.SetValue(row, -1);
                    break;
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pool_rows(output.data(), current_kv.data(), current_score.data(), previous_kv.data(),
              previous_score.data(), weight.data(), pool_scratch.data(), active, rows);
    auto cache_tensor = reinterpret_cast<__gm__ Tensor*>(args[12]);
    auto cache_data = data<uint8_t>(args, 12);
    auto scale_data = reinterpret_cast<__gm__ bfloat16_t*>(cache_data + 256);
    int64_t scale_stride = cache_tensor->strides[1] / sizeof(bfloat16_t);
    publish_rows<false>(output, cosine, sine, packed, scales, quant_scratch, targets, tokens, rows,
                        65536, 79552, 81600, data<bfloat16_t>(args, 5), cache_data, scale_data,
                        cache_tensor->strides[1], scale_stride);
}

__aicore__ inline void run(__gm__ int64_t* args) {
    int count_tokens = args[13], blocks = args[14];
    int worker = get_block_idx(args), workers = get_block_num(args);
    int request_count = reinterpret_cast<__gm__ Tensor*>(args[8])->shapes[0];
    using Meta = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
    using MetaGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 32>>;
    using Slots = Tile<TileType::Vec, int64_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
    using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 32>>;
    using Blocks = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, DYNAMIC>;
    using Bounds = Tile<TileType::Vec, int32_t, 1, 136, BLayout::RowMajor, 1, DYNAMIC>;
    using BlocksGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 128>>;
    using BoundsGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 136>>;
    Meta positions(32), requests(32);
    Blocks state_blocks(128);
    Bounds boundaries(129);
    Slots slots(32);
    Vec<int64_t, 8> targets;
    Vec<int32_t, 8> tokens;
    Vec<bfloat16_t, 512> weight;
    TASSIGN(positions, 76800);
    TASSIGN(requests, 76928);
    TASSIGN(slots, 77056);
    TASSIGN(boundaries, 83776);
    TASSIGN(state_blocks, 83264);
    TASSIGN(targets, 77408);
    TASSIGN(tokens, 77472);
    TASSIGN(weight, 73728);
    GM<bfloat16_t, 512> weight_gm(data<bfloat16_t>(args, 4));
    TLOAD(weight, weight_gm);
    int cached_base = -1;
    // A physical state block has one owner; all its historical reads precede writes.
    for (int phase = 0; phase < 2; ++phase) {
        int pending = 0;
        uint16_t active = 0;
        for (int request = 0; request < request_count + 2; ++request) {
            int lookup = request < request_count ? request : (request == request_count ? 0 : request_count);
            int request_base = lookup / 128 * 128;
            if (cached_base != request_base) {
                int request_rows = request_count - request_base < 128 ? request_count - request_base : 128;
                state_blocks.SetValidCol(request_rows);
                boundaries.SetValidCol(request_rows + 1);
                BoundsGM bounds_gm(data<int32_t>(args, 6) + request_base,
                                  TileShape2D<int32_t, 1, DYNAMIC>(1, request_rows + 1));
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
                TLOAD(boundaries, bounds_gm);
                if (request_rows > 0) {
                    BlocksGM block_gm(data<int32_t>(args, 8) + request_base,
                                      TileShape2D<int32_t, 1, DYNAMIC>(1, request_rows));
                    TLOAD(state_blocks, block_gm);
                }
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                cached_base = request_base;
            }
            int block = request < request_count ? state_blocks.GetValue(request - cached_base) : -1;
            bool valid_block = block >= 0 && block < blocks;
            int owner = valid_block ? block % workers : request % workers;
            if (owner != worker) continue;
            int start, end;
            if (request < request_count) {
                start = boundaries.GetValue(request - cached_base);
                end = boundaries.GetValue(request - cached_base + 1);
            } else if (request == request_count) {
                start = 0;
                end = boundaries.GetValue(0);
            } else {
                start = boundaries.GetValue(request_count - cached_base);
                end = count_tokens;
            }
            if (start < 0) start = 0;
            if (end > count_tokens) end = count_tokens;
            if (phase == 1 && !valid_block) continue;
            int begin = phase == 1 && end - 4 > start ? end - 4 : start;
            for (int base = begin; base < end; base += 32) {
                int count = end - base < 32 ? end - base : 32;
                positions.SetValidCol(count);
                requests.SetValidCol(count);
                slots.SetValidCol(count);
                MetaGM pos_gm(data<int32_t>(args, 3) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
                MetaGM req_gm(data<int32_t>(args, 7) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
                TLOAD(positions, pos_gm);
                TLOAD(requests, req_gm);
                if (phase == 0) {
                    SlotGM slot_gm(data<int64_t>(args, 9) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
                    TLOAD(slots, slot_gm);
                }
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                for (int lane = 0; lane < count; ++lane) {
                    int token = base + lane, position = positions.GetValue(lane);
                    bool valid = valid_block && requests.GetValue(lane) == request && position >= 0;
                    if (phase == 1) {
                        if (!valid) continue;
                        Vec<float, 512> kv, score;
                        TASSIGN(kv, 0);
                        TASSIGN(score, 16384);
                        GM<float, 512> kv_gm(data<float>(args, 0) + (int64_t)token * 512);
                        GM<float, 512> score_gm(data<float>(args, 1) + (int64_t)token * 512);
                        TLOAD(kv, kv_gm);
                        TLOAD(score, score_gm);
                        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                        int64_t slot = (int64_t)block * 4 + position % 4;
                        GM<float, 512> state_kv(data<float>(args, 2) + slot * 1024);
                        GM<float, 512> state_score(data<float>(args, 2) + slot * 1024 + 512);
                        TSTORE(state_kv, kv);
                        TSTORE(state_score, score);
                        pipe_barrier(PIPE_ALL);
                        continue;
                    }
                    tokens.SetValue(pending, token);
                    targets.SetValue(pending, valid && position % 2 ? slots.GetValue(lane) : int64_t(-1));
                    Vec<float, 32> cosine, sine;
                    TASSIGN(cosine, 77504 + pending * 128);
                    TASSIGN(sine, 78528 + pending * 128);
                    GM<float, 32> cos_gm(data<float>(args, 10) + (int64_t)token * 32);
                    GM<float, 32> sin_gm(data<float>(args, 11) + (int64_t)token * 32);
                    TLOAD(cosine, cos_gm);
                    TLOAD(sine, sin_gm);
                    if (valid && position % 2) {
                        active |= 1u << pending;
                        Vec<float, 512> ck, cs, pk, ps;
                        TASSIGN(ck, pending * 2048);
                        TASSIGN(cs, 16384 + pending * 2048);
                        TASSIGN(pk, 32768 + pending * 2048);
                        TASSIGN(ps, 49152 + pending * 2048);
                        GM<float, 512> ck_gm(data<float>(args, 0) + (int64_t)token * 512);
                        GM<float, 512> cs_gm(data<float>(args, 1) + (int64_t)token * 512);
                        __gm__ float* prev_kv;
                        __gm__ float* prev_score;
                        if (token > start) {
                            prev_kv = data<float>(args, 0) + (int64_t)(token - 1) * 512;
                            prev_score = data<float>(args, 1) + (int64_t)(token - 1) * 512;
                        } else {
                            int64_t slot = (int64_t)block * 4 + (position - 1) % 4;
                            prev_kv = data<float>(args, 2) + slot * 1024;
                            prev_score = prev_kv + 512;
                        }
                        GM<float, 512> pk_gm(prev_kv), ps_gm(prev_score);
                        TLOAD(ck, ck_gm);
                        TLOAD(cs, cs_gm);
                        TLOAD(pk, pk_gm);
                        TLOAD(ps, ps_gm);
                    }
                    if (++pending == 8) {
                        flush(args, pending, active, targets, tokens);
                        pending = 0;
                        active = 0;
                    }
                }
                set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            }
        }
        if (pending > 0) flush(args, pending, active, targets, tokens);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    run(args);
#endif
}
