// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Select candidate block maxima in stable logical order with radix histograms.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T, int R, int C> using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T> __aicore__ inline __gm__ T *data(__gm__ int64_t *args, int i) {
    auto tensor = reinterpret_cast<__gm__ Tensor *>(args[i]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}
__aicore__ inline int read(__gm__ int32_t *ptr) {
    dcci((__gm__ void *)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}
__aicore__ inline int minimum(int a, int b) {
    return a < b ? a : b;
}
__aicore__ inline int maximum(int a, int b) {
    return a > b ? a : b;
}
template <typename T> __aicore__ inline __ubuf__ T *ub(int offset) {
    Vec<T, 1, 32> tile;
    TASSIGN(tile, offset);
    return tile.data();
}
template <bool PackUpper = false>
__tf__ __aicore__ void encode(__ubuf__ float* source, __ubuf__ uint32_t* keys, int begin, int last, int width) {
    __VEC_SCOPE__ {
        vector_f32 value;
        vector_u32 sign, positive, negative, key, tail, index;
        vector_bool below, newest, packed_mask;
        vector_u16 packed;
        vector_u32 upper;
        auto all = pset_b32(PAT_ALL);
        vbr(sign, uint32_t(0x80000000));
        vbr(tail, uint32_t(0xf149f2ca));
        for (uint16_t chunk = 0; chunk < (width + 63) / 64; ++chunk) {
            vlds(value, source, chunk * 64, NORM);
            vmaxs(value, value, -1e30f, all, MODE_ZEROING);
            vadds(value, value, 0.0f, all, MODE_ZEROING);
            vcmps_lt(below, (vector_s32 &)value, int32_t(0), all);
            vxor(positive, (vector_u32 &)value, sign, all, MODE_ZEROING);
            vnot(negative, (vector_u32 &)value, all, MODE_ZEROING);
            vsel(key, negative, positive, below);
            vci((vector_s32 &)index, int32_t(begin + chunk * 64), INC_ORDER);
            vcmps_eq(newest, (vector_s32 &)index, last, all);
            vsel(key, tail, key, newest);
            vsts(key, keys, chunk * 64, NORM_B32, all);
            if constexpr (PackUpper) {
                vshrs(upper, key, 16, all, MODE_ZEROING);
                vpack(packed, upper, LOWER);
                ppack(packed_mask, all, LOWER);
                vsts(packed, (__ubuf__ uint16_t*)source, chunk * 64, NORM_B16, packed_mask);
            }
        }
    }
}
__tf__ __aicore__ void prepare_keys(__ubuf__ uint32_t* source, __ubuf__ uint32_t* keys, int width) {
    __VEC_SCOPE__ {
        vector_u32 key, lower, tail, index, upper;
        vector_u16 packed;
        vector_bool newest, packed_mask;
        auto full = pset_b32(PAT_ALL);
        vbr(lower, 0x0eb60d35u);
        vbr(tail, 0xf149f2cau);
        for (uint16_t c = 0; c < (width + 63) / 64; ++c) {
            vlds(key, source, c * 64, NORM);
            vmax(key, key, lower, full, MODE_ZEROING);
            vci((vector_s32&)index, int32_t(c * 64), INC_ORDER);
            vcmps_eq(newest, index, uint32_t(width - 1), full);
            vsel(key, tail, key, newest);
            vsts(key, keys, c * 64, NORM_B32, full);
            vshrs(upper, key, 16, full, MODE_ZEROING);
            vpack(packed, upper, LOWER);
            ppack(packed_mask, full, LOWER);
            vsts(packed, (__ubuf__ uint16_t*)source, c * 64, NORM_B16, packed_mask);
        }
    }
}
__tf__ __aicore__ void pack_upper_keys(__ubuf__ uint32_t* keys, __ubuf__ uint16_t* output, int width) {
    __VEC_SCOPE__ {
        vector_u32 key;
        vector_u16 packed;
        vector_bool half;
        uint32_t remaining = width;
        for (uint16_t chunk = 0; chunk < (width + 63) / 64; ++chunk) {
            auto mask = CreatePredicate<uint32_t>(remaining);
            vlds(key, keys, chunk * 64, NORM);
            vshrs(key, key, 16, mask, MODE_ZEROING);
            vpack(packed, key, LOWER);
            ppack(half, mask, LOWER);
            vsts(packed, output, chunk * 64, NORM_B16, half);
        }
        mem_bar(VST_VLD);
    }
}
template <int SHIFT>
__tf__ __aicore__ void packed_histogram(__ubuf__ uint16_t* keys, __ubuf__ uint16_t* hist, int width,
                                        uint32_t prefix) {
    __VEC_SCOPE__ {
        vector_u16 h0, h1, h2, h3;
        vector_u8 lo, hi;
        vector_bool selected;
        vbr(h0, uint16_t(0));
        vbr(h1, uint16_t(0));
        vbr(h2, uint16_t(0));
        vbr(h3, uint16_t(0));
        uint32_t remaining = width;
        const int blocks = (width + 255) / 256;
        for (uint16_t c = 0; c < blocks; c += 2) {
            auto mask = CreatePredicate<uint8_t>(remaining);
            vlds(lo, hi, (__ubuf__ uint8_t*)keys, c * 512, DINTLV_B8);
            if constexpr (SHIFT == 8) {
                chistv2(h0, hi, mask, Bin_N0);
                chistv2(h1, hi, mask, Bin_N1);
            } else {
                vcmps_eq(selected, hi, uint8_t(prefix >> 8), mask);
                chistv2(h0, lo, selected, Bin_N0);
                chistv2(h1, lo, selected, Bin_N1);
            }
            if (c + 1 < blocks) {
                mask = CreatePredicate<uint8_t>(remaining);
                vlds(lo, hi, (__ubuf__ uint8_t*)keys, (c + 1) * 512, DINTLV_B8);
                if constexpr (SHIFT == 8) {
                    chistv2(h2, hi, mask, Bin_N0);
                    chistv2(h3, hi, mask, Bin_N1);
                } else {
                    vcmps_eq(selected, hi, uint8_t(prefix >> 8), mask);
                    chistv2(h2, lo, selected, Bin_N0);
                    chistv2(h3, lo, selected, Bin_N1);
                }
            }
        }
        auto full = pset_b16(PAT_ALL);
        vadd(h0, h0, h2, full, MODE_ZEROING);
        vadd(h1, h1, h3, full, MODE_ZEROING);
        vsts(h0, hist, 0, NORM_B16, full);
        vsts(h1, hist, 128, NORM_B16, full);
    }
}
template <int SHIFT>
__tf__ __aicore__ void histogram(__ubuf__ uint32_t *keys, __ubuf__ uint16_t *hist, int width,
                                 uint32_t prefix) {
    __VEC_SCOPE__ {
        vector_u16 h0, h1, packed;
        vector_u32 key, part, upper;
        vector_u8 bytes;
        vector_bool selected, mask16, mask8;
        vbr(h0, uint16_t(0));
        vbr(h1, uint16_t(0));
        uint32_t remaining = width;
        for (uint16_t chunk = 0; chunk < (width + 63) / 64; ++chunk) {
            auto mask = CreatePredicate<uint32_t>(remaining);
            vlds(key, keys, chunk * 64, NORM);
            if constexpr (SHIFT < 24) {
                vshrs(upper, key, SHIFT + 8, mask, MODE_ZEROING);
                vcmps_eq(selected, upper, prefix >> (SHIFT + 8), mask);
            } else
                selected = mask;
            if constexpr (SHIFT == 0)
                part = key;
            else
                vshrs(part, key, SHIFT, mask, MODE_ZEROING);
            vpack(packed, part, LOWER);
            vpack(bytes, packed, LOWER);
            ppack(mask16, selected, LOWER);
            ppack(mask8, mask16, LOWER);
            chistv2(h0, bytes, mask8, Bin_N0);
            chistv2(h1, bytes, mask8, Bin_N1);
        }
        auto all = pset_b16(PAT_ALL);
        vsts(h0, hist, 0, NORM_B16, all);
        vsts(h1, hist, 128, NORM_B16, all);
    }
}
__aicore__ inline void choose_bin(__ubuf__ uint16_t *hist, int shift, uint32_t &key, int &rank, int &count) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    int lo = 0, hi = 255;
    for (int step = 0; step < 8; ++step) {
        int mid = (lo + hi) / 2;
        if (hist[mid] >= rank)
            hi = mid;
        else
            lo = mid + 1;
    }
    int previous = lo ? hist[lo - 1] : 0;
    count = hist[lo] - previous;
    rank -= previous;
    key |= uint32_t(lo) << shift;
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
__tf__ __aicore__ void equal_indices(__ubuf__ uint32_t *keys, __ubuf__ uint16_t *output, int width,
                                     uint32_t cutoff) {
    __VEC_SCOPE__ {
        vector_u32 key, index;
        vector_u16 small, squeezed;
        vector_bool equal, mask16;
        vector_align align;
        sprclr(SPR_AR);
        uint32_t remaining = width;
        for (uint16_t chunk = 0; chunk < (width + 63) / 64; ++chunk) {
            auto mask = CreatePredicate<uint32_t>(remaining);
            vlds(key, keys, chunk * 64, NORM);
            vci((vector_s32 &)index, int32_t(chunk * 64), INC_ORDER);
            vcmps_eq(equal, key, cutoff, mask);
            vpack(small, index, LOWER);
            ppack(mask16, equal, LOWER);
            vsqz(squeezed, small, mask16, MODE_STORED);
            vstur(align, squeezed, output, POST_UPDATE);
        }
        vstar(align, output);
    }
}
__tf__ __aicore__ void select(__ubuf__ uint32_t *keys, __ubuf__ uint32_t *output, int width, uint32_t cutoff,
                              int boundary, bool all_visible) {
    __VEC_SCOPE__ {
        vector_u32 key, index, squeezed, invalid;
        vector_bool strict, equal, earlier, keep;
        vector_align align;
        auto all = pset_b32(PAT_ALL);
        vbr(invalid, uint32_t(0xffffffff));
        for (uint16_t chunk = 0; chunk < 32; ++chunk)
            vsts(invalid, output, chunk * 64, NORM_B32, all);
        mem_bar(VST_VST);
        sprclr(SPR_AR);
        uint32_t remaining = width;
        for (uint16_t chunk = 0; chunk < (width + 63) / 64; ++chunk) {
            auto mask = CreatePredicate<uint32_t>(remaining);
            vci((vector_s32 &)index, int32_t(chunk * 64), INC_ORDER);
            if (all_visible)
                keep = mask;
            else {
                vlds(key, keys, chunk * 64, NORM);
                vcmps_gt(strict, key, cutoff, mask);
                vcmps_eq(equal, key, cutoff, mask);
                vcmps_le(earlier, (vector_s32 &)index, boundary, mask);
                pand(equal, equal, earlier, mask);
                por(keep, strict, equal, mask);
            }
            vsqz(squeezed, index, keep, MODE_STORED);
            vstur(align, squeezed, output, POST_UPDATE);
        }
        vstar(align, output);
    }
}
__tf__ __aicore__ void count_blocks(__ubuf__ int32_t* lengths, int tokens, int active, int width,
                                    int capacity) {
    __VEC_SCOPE__ {
        vector_s32 value, lane, zero;
        vector_bool enabled;
        vbr(zero, int32_t(0));
        uint32_t remaining = tokens;
        for (uint16_t chunk = 0; chunk < (tokens + 63) / 64; ++chunk) {
            auto mask = CreatePredicate<uint32_t>(remaining);
            vlds(value, lengths, chunk * 64, NORM);
            vmaxs(value, value, int32_t(0), mask, MODE_ZEROING);
            vmins(value, value, int32_t(width), mask, MODE_ZEROING);
            vadds(value, value, int32_t(7), mask, MODE_ZEROING);
            vshrs(value, value, 3, mask, MODE_ZEROING);
            vmins(value, value, int32_t(capacity), mask, MODE_ZEROING);
            vci(lane, int32_t(chunk * 64), INC_ORDER);
            vcmps_lt(enabled, lane, active, mask);
            vsel(value, value, zero, enabled);
            vsts(value, lengths, chunk * 64, NORM_B32, mask);
        }
    }
}
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(-1, -1);
    auto scores = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto out = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int tokens = out->shapes[0], capacity = out->shapes[1], width = scores->shapes[1], active = args[4];
    Vec<float, 1, 8192> full;
    Tile<TileType::Vec, float, 1024, 8> grouped, tmp;
    Tile<TileType::Vec, float, 1024, 1, BLayout::ColMajor> maxima;
    TASSIGN(full, 0);
    TASSIGN(grouped, 0);
    TASSIGN(tmp, 32768);
    TASSIGN(maxima, 65536);
    for (int token = get_block_idx(args); token < tokens; token += get_block_num(args)) {
        int visible = token < active ? maximum(0, minimum(width, read(data<int32_t>(args, 1) + token))) : 0;
        int blocks = (visible + 7) / 8, selected = minimum(blocks, minimum(capacity, 2048));
        uint32_t cutoff = 0;
        int boundary = -1;
        if (blocks > selected) {
            if (args[5]) {
                Tile<TileType::Vec, float, 1, 32768, BLayout::RowMajor, 1, DYNAMIC> input(blocks);
                TASSIGN(input, 0);
                GlobalTensor<float, TileShape2D<float, 1, DYNAMIC>, BaseShape2D<float, 1, DYNAMIC>> gm(
                    data<float>(args, 0) + int64_t(token) * width,
                    TileShape2D<float, 1, DYNAMIC>(1, blocks), BaseShape2D<float, 1, DYNAMIC>(1, blocks));
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                TLOAD(input, gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (args[5] == 2) prepare_keys((__ubuf__ uint32_t*)input.data(), ub<uint32_t>(131072), blocks);
                else encode<true>(input.data(), ub<uint32_t>(131072), 0, blocks - 1, blocks);
            } else {
                for (int block = 0; block < blocks; block += 1024) {
                    int valid = minimum(8192, width - block * 8);
                    TEXPANDS(full, -1e30f);
                    Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, DYNAMIC> input(valid);
                    TASSIGN(input, 0);
                    GlobalTensor<float, TileShape2D<float, 1, DYNAMIC>, BaseShape2D<float, 1, DYNAMIC>> gm(
                        data<float>(args, 0) + int64_t(token) * width + block * 8,
                        TileShape2D<float, 1, DYNAMIC>(1, valid), BaseShape2D<float, 1, DYNAMIC>(1, valid));
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    TLOAD(input, gm);
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    TROWMAX(maxima, grouped, tmp);
                    encode(maxima.data(), ub<uint32_t>(131072) + block, block, blocks - 1, 1024);
                }
            }
            pipe_barrier(PIPE_ALL);
            int rank = blocks - selected + 1, count = 0;
            if (!args[5]) pack_upper_keys(ub<uint32_t>(131072), ub<uint16_t>(0), blocks);
            packed_histogram<8>(ub<uint16_t>(0), ub<uint16_t>(73728), blocks, 0);
            choose_bin(ub<uint16_t>(73728), 24, cutoff, rank, count);
            packed_histogram<0>(ub<uint16_t>(0), ub<uint16_t>(73728), blocks, cutoff >> 16);
            choose_bin(ub<uint16_t>(73728), 16, cutoff, rank, count);
            // BF16 maxima use two radix bytes; clamp and newest-block scores use four.
            if (args[5] && cutoff != 0x0eb60000u && cutoff != 0xf1490000u) {
                if (cutoff < 0x80000000u)
                    cutoff |= 0xffffu;
            } else {
                histogram<8>(ub<uint32_t>(131072), ub<uint16_t>(73728), blocks, cutoff);
                choose_bin(ub<uint16_t>(73728), 8, cutoff, rank, count);
                histogram<0>(ub<uint32_t>(131072), ub<uint16_t>(73728), blocks, cutoff);
                choose_bin(ub<uint16_t>(73728), 0, cutoff, rank, count);
            }
            equal_indices(ub<uint32_t>(131072), ub<uint16_t>(0), blocks, cutoff);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            boundary = ub<uint16_t>(0)[count - rank];
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }
        select(ub<uint32_t>(131072), ub<uint32_t>(65536), blocks, cutoff, boundary, blocks <= selected);
        Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, 1, DYNAMIC> ids(capacity);
        TASSIGN(ids, 65536);
        GlobalTensor<uint32_t, TileShape2D<uint32_t, 1, DYNAMIC>, BaseShape2D<uint32_t, 1, DYNAMIC>> gm(
            data<uint32_t>(args, 2) + token * capacity, TileShape2D<uint32_t, 1, DYNAMIC>(1, capacity),
            BaseShape2D<uint32_t, 1, DYNAMIC>(1, capacity));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(gm, ids);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    if (get_block_idx(args) == 0) {
        for (int begin = 0; begin < tokens; begin += 256) {
            int count = minimum(256, tokens - begin);
            Tile<TileType::Vec, int32_t, 1, 256, BLayout::RowMajor, 1, DYNAMIC> lengths(count);
            TASSIGN(lengths, 0);
            using Shape = TileShape2D<int32_t, 1, DYNAMIC>;
            using Base = BaseShape2D<int32_t, 1, DYNAMIC>;
            GlobalTensor<int32_t, Shape, Base> source(data<int32_t>(args, 1) + begin, Shape(1, count),
                                                      Base(1, count));
            GlobalTensor<int32_t, Shape, Base> dest(data<int32_t>(args, 3) + begin, Shape(1, count),
                                                    Base(1, count));
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            TLOAD(lengths, source);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            count_blocks(lengths.data(), count, active - begin, width, minimum(capacity, 2048));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dest, lengths);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
