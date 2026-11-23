// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Sharded paged scores, local TopK, and candidate merge in one MIX kernel.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
namespace score_core {
using F4 = float4_e2m1x2_t;
using S = float8_e8m0_t;

template <typename T, int R, int C, int BR, int BC, Layout L = Layout::ND>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using QueryScale = Tile<TileType::Mat, S, 192, 4, BLayout::RowMajor, DYNAMIC, 4, SLayout::RowMajor, 32>;
using KeyScale = Tile<TileType::Mat, S, 4, 128, BLayout::ColMajor, 4, 128, SLayout::ColMajor, 32>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int read(__gm__ int32_t* ptr) {
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}

__aicore__ inline int minimum(int a, int b) { return a < b ? a : b; }

__aicore__ inline int active_shards(__gm__ int64_t* args, int groups, int panels, int workers) {
    if (args[19]) return int(args[21]);
    int count = workers;
    if (panels > 100) {
        int needed = (groups * panels + 99) / 100;
        if (needed > count) count = needed;
    }
    return minimum(int(args[21]), count);
}


#if defined(__DAV_VEC__)
__aicore__ inline void prepare_weights(__gm__ int64_t* args, int begin, int count) {
    Tile<TileType::Vec, bfloat16_t, 6, 32, BLayout::RowMajor, DYNAMIC, 32> weight(count);
    Vec<bfloat16_t, 1, 192> weight_raw;
    Vec<float, 1, 192> scaled;
    Vec<bfloat16_t, 1, 192> rounded;
    TASSIGN(weight, 256000);
    TASSIGN(weight_raw, 256000);
    TASSIGN(scaled, 256512);
    TASSIGN(rounded, 257280);
    GM<bfloat16_t, DYNAMIC, 32, DYNAMIC, 32> wg(data<bfloat16_t>(args, 2) + begin * 32,
        TileShape2D<bfloat16_t, DYNAMIC, 32>(count, 32), BaseShape2D<bfloat16_t, DYNAMIC, 32>(count, 32));
    // Preserve the previous fragment's vector reads before reusing weight storage.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(weight, wg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(scaled, weight_raw, RoundMode::CAST_RINT);
    TMULS(scaled, scaled, 0.015625f);
    TCVT(rounded, scaled, RoundMode::CAST_RINT);
}

#endif
}
}

using namespace score_core;
namespace {
    namespace fused {
        // Main-phase UB: QK [0, 98304), three uint16 caches [98304, 251904),
        // cumulative histograms [251904, 253440), and weights above 256000.
        // Local TopK reuses the drained QK slots; merge reuses the main-phase buffers.
#if defined(__DAV_VEC__)
        template <typename T>
        __aicore__ inline __ubuf__ T* ub(int offset) {
            Vec<T, 1, 32> tile;
            TASSIGN(tile, offset);
            return tile.data();
        }
        using CandidateKeys = Vec<uint16_t, 1, 512>;
        using CandidateIndices = Vec<uint32_t, 1, 512>;
        template <int HistogramTile = 128>
        __aicore__ inline void write_cached_row(RegTensor<bfloat16_t>& a, RegTensor<bfloat16_t>& b,
        __ubuf__ uint16_t* dst, __ubuf__ uint16_t* hist, int valid0, int valid1,
        int span0, int span1, bool reset, __ubuf__ uint16_t* pool_valid, bool gathered) {
            vector_u16 h0, h1, pos, negkey, sign, key, high, invalid, lane_valid;
            vector_u8 bytes, unused;
            vector_u16 first_key;
            vector_bool negative, pool_mask;
            auto full = pset_b16(PAT_ALL);
            mem_bar(VST_VLD);
            if (reset) {
                vbr(h0, uint16_t(0));
                vbr(h1, uint16_t(0));
            }
            else {
                vlds(h0, hist, 0, NORM);
                vlds(h1, hist, 128, NORM);
            }
            vbr(sign, uint16_t(0x8000));
            vbr(invalid, uint16_t(0x007f));
            vadds(a, a, bfloat16_t(0), full, MODE_ZEROING);
            vcmps_lt(negative, (vector_s16&)a, int16_t(0), full);
            vxor(pos, (vector_u16&)a, sign, full, MODE_ZEROING);
            vnot(negkey, (vector_u16&)a, full, MODE_ZEROING);
            vsel(key, negkey, pos, negative);
            uint32_t remaining = valid0;
            auto valid = CreatePredicate<uint16_t>(remaining);
            vsel(key, key, invalid, valid);
            if (gathered) {
                vlds(lane_valid, pool_valid, 0, NORM);
                vcmps_eq(pool_mask, lane_valid, uint16_t(1), full);
                vsel(key, key, invalid, pool_mask);
            }
            vsts(key, dst, 0, NORM_B16, full);
            first_key = key;
            if constexpr (HistogramTile == 128) {
                vshrs(high, key, 8, full, MODE_ZEROING);
                vpack(bytes, high, LOWER);
                remaining = span0;
                auto hist_mask = CreatePredicate<uint8_t>(remaining);
                chistv2(h0, bytes, hist_mask, Bin_N0);
                chistv2(h1, bytes, hist_mask, Bin_N1);
            }
            vadds(b, b, bfloat16_t(0), full, MODE_ZEROING);
            vcmps_lt(negative, (vector_s16&)b, int16_t(0), full);
            vxor(pos, (vector_u16&)b, sign, full, MODE_ZEROING);
            vnot(negkey, (vector_u16&)b, full, MODE_ZEROING);
            vsel(key, negkey, pos, negative);
            remaining = valid1;
            valid = CreatePredicate<uint16_t>(remaining);
            vsel(key, key, invalid, valid);
            if (gathered) {
                vlds(lane_valid, pool_valid, 128, NORM);
                vcmps_eq(pool_mask, lane_valid, uint16_t(1), full);
                vsel(key, key, invalid, pool_mask);
            }
            vsts(key, dst, 128, NORM_B16, full);
            if constexpr (HistogramTile == 256) {
                vdintlv(unused, bytes, (vector_u8&)first_key, (vector_u8&)key);
                remaining = span0 + span1;
                auto mask = CreatePredicate<uint8_t>(remaining);
                chistv2(h0, bytes, mask, Bin_N0);
                chistv2(h1, bytes, mask, Bin_N1);
            } else {
                vshrs(high, key, 8, full, MODE_ZEROING);
                vpack(bytes, high, LOWER);
                remaining = span1;
                auto hist_mask = CreatePredicate<uint8_t>(remaining);
                chistv2(h0, bytes, hist_mask, Bin_N0);
                chistv2(h1, bytes, hist_mask, Bin_N1);
            }
            vsts(h0, hist, 0, NORM_B16, full);
            vsts(h1, hist, 128, NORM_B16, full);
        }
        __tf__ __aicore__ void compact_validity(__ubuf__ uint32_t* out,
                                                int p0, int p1, int p2, int p3) {
            __VEC_SCOPE__ {
                vector_u32 value;
                auto full = pset_b32(PAT_ALL);
                for (uint16_t piece = 0; piece < 4; ++piece) {
                    int page = piece == 0 ? p0 : piece == 1 ? p1 : piece == 2 ? p2 : p3;
                    vbr(value, uint32_t(page >= 0));
                    vsts(value, out, piece * 64, NORM_B32, full);
                }
            }
        }
        template <int HistogramTile>
        __tf__ __aicore__ void score_cache(__ubuf__ bfloat16_t* src, __ubuf__ bfloat16_t* weights,
        __ubuf__ uint16_t* cache, __ubuf__ uint16_t* hist, int column, int span0, int span1,
        int v00, int v01, int v10, int v11, int v20, int v21, bool have_dot,
        __ubuf__ uint16_t* pool_valid, bool gathered) {
            __VEC_SCOPE__ {
                RegTensor<bfloat16_t> s00, s01, s10, s11, s20, s21, d00, d01, d10, d11, d20, d21, w0, w1, w2;
                auto full = pset_b16(PAT_ALL);
                vbr(s00, bfloat16_t(0));
                vbr(s01, bfloat16_t(0));
                vbr(s10, bfloat16_t(0));
                vbr(s11, bfloat16_t(0));
                vbr(s20, bfloat16_t(0));
                vbr(s21, bfloat16_t(0));
                if (have_dot) {
#pragma unroll 4
                    for (uint16_t h = 0; h < 32; ++h) {
                        vlds(w0, weights, h, BRC_B16);
                        vlds(w1, weights, h + 32, BRC_B16);
                        vlds(w2, weights, h + 64, BRC_B16);
                        vlds(d00, src, h * 256, NORM);
                        vlds(d01, src, h * 256 + 128, NORM);
                        vlds(d10, src, (h + 32) * 256, NORM);
                        vlds(d11, src, (h + 32) * 256 + 128, NORM);
                        vlds(d20, src, (h + 64) * 256, NORM);
                        vlds(d21, src, (h + 64) * 256 + 128, NORM);
                        vmaxs(d00, d00, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d01, d01, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d10, d10, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d11, d11, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d20, d20, bfloat16_t(0), full, MODE_ZEROING);
                        vmaxs(d21, d21, bfloat16_t(0), full, MODE_ZEROING);
                        vmadd(d00, w0, s00, full, MODE_ZEROING);
                        vmadd(d01, w0, s01, full, MODE_ZEROING);
                        vmadd(d10, w1, s10, full, MODE_ZEROING);
                        vmadd(d11, w1, s11, full, MODE_ZEROING);
                        vmadd(d20, w2, s20, full, MODE_ZEROING);
                        vmadd(d21, w2, s21, full, MODE_ZEROING);
                        s00 = d00;
                        s01 = d01;
                        s10 = d10;
                        s11 = d11;
                        s20 = d20;
                        s21 = d21;
                    }
                }
                write_cached_row<HistogramTile>(s00, s01, cache+column, hist, v00, v01, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row<HistogramTile>(s10, s11, cache+25600+column, hist+256, v10, v11, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row<HistogramTile>(s20, s21, cache+51200+column, hist+512, v20, v21, span0, span1, column == 0, pool_valid, gathered);
            }
        }
        __aicore__ inline void threshold_bin(vector_u16 h0, vector_u16 h1, vector_u16& rank,
                                            vector_u16& bin, vector_u16& residual) {
            vector_u16 i0, i1, c0, c1, previous0, previous1, zero, missing;
            vector_bool ge0, ge1, before0, before1;
            auto full = pset_b16(PAT_ALL);
            vci((vector_s16&)i0, int16_t(0), INC_ORDER);
            vci((vector_s16&)i1, int16_t(128), INC_ORDER);
            vbr(zero, uint16_t(0));
            vbr(missing, uint16_t(65535));
            vcmp_ge(ge0, h0, rank, full);
            vcmp_ge(ge1, h1, rank, full);
            vsel(c0, i0, missing, ge0);
            vsel(c1, i1, missing, ge1);
            vmin(c0, c0, c1, full, MODE_ZEROING);
            vcmin(c1, c0, full, MODE_ZEROING);
            vdup(bin, c1, full, POS_LOWEST, MODE_ZEROING);
            vcmp_lt(before0, i0, bin, full);
            vcmp_lt(before1, i1, bin, full);
            vsel(previous0, h0, zero, before0);
            vsel(previous1, h1, zero, before1);
            vmax(previous0, previous0, previous1, full, MODE_ZEROING);
            vcmax(previous1, previous0, full, MODE_ZEROING);
            vdup(previous0, previous1, full, POS_LOWEST, MODE_ZEROING);
            vcmp_eq(ge0, i0, bin, full);
            vcmp_eq(ge1, i1, bin, full);
            vsel(c0, h0, zero, ge0);
            vsel(c1, h1, zero, ge1);
            vmax(c0, c0, c1, full, MODE_ZEROING);
            vcmax(c1, c0, full, MODE_ZEROING);
            vdup(c0, c1, full, POS_LOWEST, MODE_ZEROING);
            vsub(residual, c0, rank, full, MODE_ZEROING);
            vsub(rank, rank, previous0, full, MODE_ZEROING);
        }
        __aicore__ inline void find_threshold(__ubuf__ uint16_t* keys, int width,
                                              __ubuf__ uint16_t* scratch, bool cached,
                                              __ubuf__ uint16_t* cached_hist,
                                              vector_u16& key_cutoff, vector_u16& key_boundary) {
            vector_u16 h0, h1, h2, h3, rank, bin, residual, cutoff;
            vector_u8 lo, hi, prefix;
            vector_bool selected;
            auto full = pset_b16(PAT_ALL);
            if (cached) {
                vlds(h0, cached_hist, 0, NORM);
                vlds(h1, cached_hist, 128, NORM);
            } else {
                vbr(h0, uint16_t(0));
                vbr(h1, uint16_t(0));
                vbr(h2, uint16_t(0));
                vbr(h3, uint16_t(0));
                uint32_t remaining = width;
                for (uint16_t c = 0; c < (width + 255) / 256; c += 2) {
                    auto mask = CreatePredicate<uint8_t>(remaining);
                    vlds(lo, hi, (__ubuf__ uint8_t*)keys, c * 512, DINTLV_B8);
                    chistv2(h0, hi, mask, Bin_N0);
                    chistv2(h1, hi, mask, Bin_N1);
                    if (c + 1 < (width + 255) / 256) {
                        mask = CreatePredicate<uint8_t>(remaining);
                        vlds(lo, hi, (__ubuf__ uint8_t*)keys, (c + 1) * 512, DINTLV_B8);
                        chistv2(h2, hi, mask, Bin_N0);
                        chistv2(h3, hi, mask, Bin_N1);
                    }
                }
                vadd(h0, h0, h2, full, MODE_ZEROING);
                vadd(h1, h1, h3, full, MODE_ZEROING);
            }
            vbr(rank, uint16_t(width - 511));
            threshold_bin(h0, h1, rank, bin, residual);
            vpack(prefix, bin, LOWER);
            auto full8 = pset_b8(PAT_ALL);
            vdup(prefix, prefix, full8, POS_LOWEST, MODE_ZEROING);
            vshls(cutoff, bin, 8, full, MODE_ZEROING);
            vbr(h0, uint16_t(0));
            vbr(h1, uint16_t(0));
            vbr(h2, uint16_t(0));
            vbr(h3, uint16_t(0));
            uint32_t remaining = width;
            for (uint16_t c = 0; c < (width + 255) / 256; c += 2) {
                auto mask = CreatePredicate<uint8_t>(remaining);
                vlds(lo, hi, (__ubuf__ uint8_t*)keys, c * 512, DINTLV_B8);
                vcmp_eq(selected, hi, prefix, mask);
                chistv2(h0, lo, selected, Bin_N0);
                chistv2(h1, lo, selected, Bin_N1);
                if (c + 1 < (width + 255) / 256) {
                    mask = CreatePredicate<uint8_t>(remaining);
                    vlds(lo, hi, (__ubuf__ uint8_t*)keys, (c + 1) * 512, DINTLV_B8);
                    vcmp_eq(selected, hi, prefix, mask);
                    chistv2(h2, lo, selected, Bin_N0);
                    chistv2(h3, lo, selected, Bin_N1);
                }
            }
            vadd(h0, h0, h2, full, MODE_ZEROING);
            vadd(h1, h1, h3, full, MODE_ZEROING);
            threshold_bin(h0, h1, rank, bin, residual);
            vor(cutoff, cutoff, bin, full, MODE_ZEROING);
            vector_u16 key, lane, squeezed;
            vector_align align;
            auto out = scratch;
            sprclr(SPR_AR);
            remaining = width;
            for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                auto mask = CreatePredicate<uint16_t>(remaining);
                vlds(key, keys, c * 128, NORM);
                vci((vector_s16&)lane, int16_t(c * 128), INC_ORDER);
                vcmp_eq(selected, key, cutoff, mask);
                vsqz(squeezed, lane, selected, MODE_STORED);
                vstur(align, squeezed, out, POST_UPDATE);
            }
            vstar(align, out);
            mem_bar(VST_VLD);
            vector_u32 index, shift, packed;
            auto full32 = pset_b32(PAT_ALL);
            vcvt(index, residual, full, PART_EVEN);
            vbr(shift, 1u);
            vand(shift, shift, index, full32, MODE_ZEROING);
            vshls(shift, shift, 4, full32, MODE_ZEROING);
            vshrs(index, index, 1, full32, MODE_ZEROING);
            vgather2(packed, (__ubuf__ uint32_t*)scratch, index, full32);
            vshr(packed, packed, (vector_s32&)shift, full32, MODE_ZEROING);
            vpack(lane, packed, LOWER);
            key_cutoff = cutoff;
            vdup(key_boundary, lane, full, POS_LOWEST, MODE_ZEROING);
        }
        __tf__ __aicore__ void vector_threshold(__ubuf__ uint16_t* keys, int width,
                                               __ubuf__ uint16_t* scratch, __ubuf__ uint16_t* result) {
            __VEC_SCOPE__ {
                vector_u16 cutoff, boundary;
                find_threshold(keys, width, scratch, false, (__ubuf__ uint16_t*)0, cutoff, boundary);
                auto full = pset_b16(PAT_ALL);
                vsts(cutoff, result, 0, NORM_B16, full);
                vsts(boundary, result, 128, NORM_B16, full);
            }
        }
        __aicore__ inline void packed_threshold(__ubuf__ uint16_t* keys, int width, int scratch, int hist_base,
        uint32_t& cutoff, int& boundary) {
            cutoff = 0x007f;
            boundary = -1;
            if (width>512) {
                vector_threshold(keys, width, ub<uint16_t>(scratch), ub<uint16_t>(hist_base));
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                cutoff = ub<uint16_t>(hist_base)[0];
                boundary = ub<uint16_t>(hist_base)[128];
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            }
        }
        __tf__ __aicore__ void packed_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
        __ubuf__ uint16_t* src, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* table,
        int width, uint32_t cutoff, int boundary, bool gathered, bool map_output, int page_shift = 7) {
            auto result = outi;
            __VEC_SCOPE__ {
                vector_u32 key, lane, logical, tmp, page, squeezed, invalid, shift;
                vector_u16 key16, zero16;
                vector_bool eq, strict, early, keep, valid, packed_mask;
                vector_align align;
                auto full = pset_b32(PAT_ALL);
                auto full16 = pset_b16(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                vbr(zero16, uint16_t(0));
                for (uint16_t c = 0;c<8;++c)vsts(invalid, outi, c*64, NORM_B32, full);
                for (uint16_t c = 0;c<4;++c)vsts(zero16, outk, c*128, NORM_B16, full16);
                mem_bar(VST_VST);
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0;c<(width+63)/64;++c) {
                    auto mask = CreatePredicate<uint32_t>(remaining);
                    vlds((vector_u16&)key, src, c*64, UNPK_B16);
                    vci((vector_s32&)lane, int32_t(c*64), INC_ORDER);
                    vcmps_gt(strict, key, cutoff, mask);
                    vcmps_eq(eq, key, cutoff, mask);
                    vcmps_le(early, (vector_s32&)lane, boundary, mask);
                    pand(eq, eq, early, mask);
                    por(keep, strict, eq, mask);
                    vcmps_gt(valid, key, 0x007fu, mask);
                    pand(keep, keep, valid, mask);
                    {
                        vlds(logical, ids, c*64, NORM);
                    }
                    vsqz(squeezed, logical, keep, MODE_STORED);
                    vstur(align, squeezed, outi, POST_UPDATE);
                }
                vstar(align, outi);
                // Map only the selected rows, retaining their logical order.
                if (map_output) {
                    mem_bar(VST_VLD);
                    for (uint16_t c = 0; c < 8; ++c) {
                        vlds(logical, result, c * 64, NORM);
                        vcmps_ne(valid, logical, 0xffffffffu, full);
                        if (gathered) {
                            vgather2(page, table, logical, valid);
                            logical = page;
                        } else {
                            vshrs(tmp, logical, page_shift, full, MODE_ZEROING);
                            vgather2(page, table, tmp, valid);
                            vshls(page, page, page_shift, full, MODE_ZEROING);
                            vbr(tmp, uint32_t((1 << page_shift) - 1));
                            vand(tmp, logical, tmp, full, MODE_ZEROING);
                            vadd(logical, page, tmp, full, MODE_ZEROING);
                        }
                        vsel(logical, logical, invalid, valid);
                        vsts(logical, result, c * 64, NORM_B32, full);
                    }
                }
            }
        }
        __tf__ __aicore__ void merge_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
        __ubuf__ uint16_t* src, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* table,
        int width, bool gathered, bool map_output, __ubuf__ uint16_t* scratch, int page_shift = 7) {
            auto result = outi;
            __VEC_SCOPE__ {
                vector_u16 cutoff16, boundary16;
                vector_u32 cutoff, boundary;
                if (width > 512) {
                    find_threshold(src, width, scratch, false, (__ubuf__ uint16_t*)0, cutoff16, boundary16);
                    mem_bar(VLD_VST);
                } else {
                    vbr(cutoff16, uint16_t(0x007f));
                    vbr(boundary16, uint16_t(65535));
                }
                auto convert_mask = pset_b16(PAT_ALL);
                vcvt(cutoff, cutoff16, convert_mask, PART_EVEN);
                vcvt(boundary, boundary16, convert_mask, PART_EVEN);
                vector_u32 key, lane, logical, tmp, page, squeezed, invalid, shift;
                vector_u16 key16, zero16;
                vector_bool eq, strict, early, keep, valid, packed_mask;
                vector_align align;
                auto full = pset_b32(PAT_ALL);
                auto full16 = pset_b16(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                vbr(zero16, uint16_t(0));
                for (uint16_t c = 0;c<8;++c)vsts(invalid, outi, c*64, NORM_B32, full);
                for (uint16_t c = 0;c<4;++c)vsts(zero16, outk, c*128, NORM_B16, full16);
                mem_bar(VST_VST);
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0;c<(width+63)/64;++c) {
                    auto mask = CreatePredicate<uint32_t>(remaining);
                    vlds((vector_u16&)key, src, c*64, UNPK_B16);
                    vci((vector_s32&)lane, int32_t(c*64), INC_ORDER);
                    vcmp_gt(strict, key, cutoff, mask);
                    vcmp_eq(eq, key, cutoff, mask);
                    vcmp_le(early, lane, boundary, mask);
                    pand(eq, eq, early, mask);
                    por(keep, strict, eq, mask);
                    vcmps_gt(valid, key, 0x007fu, mask);
                    pand(keep, keep, valid, mask);
                    {
                        vlds(logical, ids, c*64, NORM);
                    }
                    vsqz(squeezed, logical, keep, MODE_STORED);
                    vstur(align, squeezed, outi, POST_UPDATE);
                }
                vstar(align, outi);
                // Map only the selected rows, retaining their logical order.
                if (map_output) {
                    mem_bar(VST_VLD);
                    for (uint16_t c = 0; c < 8; ++c) {
                        vlds(logical, result, c * 64, NORM);
                        vcmps_ne(valid, logical, 0xffffffffu, full);
                        if (gathered) {
                            vgather2(page, table, logical, valid);
                            logical = page;
                        } else {
                            vshrs(tmp, logical, page_shift, full, MODE_ZEROING);
                            vgather2(page, table, tmp, valid);
                            vshls(page, page, page_shift, full, MODE_ZEROING);
                            vbr(tmp, uint32_t((1 << page_shift) - 1));
                            vand(tmp, logical, tmp, full, MODE_ZEROING);
                            vadd(logical, page, tmp, full, MODE_ZEROING);
                        }
                        vsel(logical, logical, invalid, valid);
                        vsts(logical, result, c * 64, NORM_B32, full);
                    }
                }
            }
        }
        __aicore__ inline void select_local(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
                                            __ubuf__ uint16_t* src, int width, vector_u16& cutoff,
                                            vector_u16& boundary, int logical_begin) {
            vector_u16 key, lane, squeezed, zero;
            vector_bool eq, strict, early, keep, valid;
            vector_align align;
            auto full = pset_b16(PAT_ALL);
            auto full32 = pset_b32(PAT_ALL);
            auto packed = (__ubuf__ uint16_t*)0;
            vbr(zero, uint16_t(65535));
            for (uint16_t c = 0; c < 4; ++c) vsts(zero, packed, c * 128, NORM_B16, full);
            mem_bar(VST_VST);
            sprclr(SPR_AR);
            uint32_t remaining = width;
            for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                auto mask = CreatePredicate<uint16_t>(remaining);
                vlds(key, src, c * 128, NORM);
                vci((vector_s16&)lane, int16_t(c * 128), INC_ORDER);
                vcmp_gt(strict, key, cutoff, mask);
                vcmp_eq(eq, key, cutoff, mask);
                vcmp_le(early, (vector_s16&)lane, (vector_s16&)boundary, mask);
                pand(eq, eq, early, mask);
                por(keep, strict, eq, mask);
                vcmps_gt(valid, key, uint16_t(0x007f), mask);
                pand(keep, keep, valid, mask);
                vsqz(squeezed, lane, keep, MODE_STORED);
                vstur(align, squeezed, packed, POST_UPDATE);
            }
            vstar(align, packed);
            mem_bar(VST_VLD);
            vector_u16 interleaved, unused;
            vector_u32 index, local, gathered, shift, invalid;
            vbr(invalid, 0xffffffffu);
            for (uint16_t c = 0; c < 8; ++c) {
                vlds(lane, (__ubuf__ uint16_t*)0, c * 64, NORM);
                vintlv(interleaved, unused, lane, lane);
                vcvt(local, interleaved, full, PART_EVEN);
                vcmps_ne(valid, local, 65535u, full32);
                vadds(index, local, uint32_t(logical_begin), full32, MODE_ZEROING);
                vsel(index, index, invalid, valid);
                vsts(index, outi, c * 64, NORM_B32, full32);
                vbr(shift, 1u);
                vand(shift, local, shift, full32, MODE_ZEROING);
                vshls(shift, shift, 4, full32, MODE_ZEROING);
                vshrs(local, local, 1, full32, MODE_ZEROING);
                vbr(gathered, 0u);
                vgather2(gathered, (__ubuf__ uint32_t*)src, local, valid);
                vshr(gathered, gathered, (vector_s32&)shift, full32, MODE_ZEROING);
                vpack(key, gathered, LOWER);
                vector_bool packed_mask;
                ppack(packed_mask, full32, LOWER);
                vsts(key, outk, c * 64, NORM_B16, packed_mask);
            }
        }
        __tf__ __aicore__ void local_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
                                           __ubuf__ uint16_t* keys, int width, int logical_begin,
                                           __ubuf__ uint16_t* cached_hist) {
            __VEC_SCOPE__ {
                vector_u16 cutoff, boundary;
                if (width > 512) {
                    find_threshold(keys, width, (__ubuf__ uint16_t*)0, true, cached_hist, cutoff, boundary);
                    mem_bar(VLD_VST);
                } else {
                    vbr(cutoff, uint16_t(0x007f));
                    vbr(boundary, uint16_t(65535));
                }
                select_local(outk, outi, keys, width, cutoff, boundary, logical_begin);
            }
        }
        __aicore__ inline void local_topk(__gm__ int64_t* args, int begin, int count, int logical, int width, int shard) {
            int shards = args[21];
            for (int row = 0;row<3;++row) {
                int group_offset = get_subblockid()*3+row;
                if (group_offset >= count)continue;
                auto keys = ub<uint16_t>(98304+row*51200);
                CandidateKeys selected_keys;
                CandidateIndices selected_indices;
                TASSIGN(selected_keys, 51200 + row * 3072);
                TASSIGN(selected_indices, 52224 + row * 3072);
                local_select(selected_keys.data(), selected_indices.data(), keys, width, logical,
                             ub<uint16_t>(251904 + row * 512));
            }
            // Publish all local query rows through one strided DMA per output.
            int first = get_subblockid() * 3;
            int rows = minimum(3, count - first);
            if (rows <= 0) return;
            Tile<TileType::Vec, uint16_t, 3, 1536, BLayout::RowMajor, DYNAMIC, 512> keys(rows);
            Tile<TileType::Vec, uint32_t, 3, 768, BLayout::RowMajor, DYNAMIC, 512> indices(rows);
            TASSIGN(keys, 51200);
            TASSIGN(indices, 52224);
            int index = ((begin + first) * shards + shard) * 512;
            GM<uint16_t, DYNAMIC, 512, DYNAMIC, DYNAMIC> kg(data<uint16_t>(args, 10) + index,
                TileShape2D<uint16_t, DYNAMIC, 512>(rows, 512), BaseShape2D<uint16_t, DYNAMIC, DYNAMIC>(rows, shards * 512));
            GM<uint32_t, DYNAMIC, 512, DYNAMIC, DYNAMIC> ig(data<uint32_t>(args, 11) + index,
                TileShape2D<uint32_t, DYNAMIC, 512>(rows, 512), BaseShape2D<uint32_t, DYNAMIC, DYNAMIC>(rows, shards * 512));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(kg, keys);
            TSTORE(ig, indices);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        __tf__ __aicore__ void compact_keys(__ubuf__ uint16_t* out, __ubuf__ uint16_t* keys,
                                            int width, uint32_t cutoff, int boundary) {
            __VEC_SCOPE__ {
                vector_u16 key, selected;
                vector_s16 lane;
                vector_bool strict, equal, early, keep, valid;
                vector_align align;
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0; c < (width + 127) / 128; ++c) {
                    auto mask = CreatePredicate<uint16_t>(remaining);
                    vlds(key, keys, c * 128, NORM);
                    vci(lane, int16_t(c * 128), INC_ORDER);
                    vcmps_gt(strict, key, uint16_t(cutoff), mask);
                    vcmps_eq(equal, key, uint16_t(cutoff), mask);
                    vcmps_le(early, lane, int16_t(boundary), mask);
                    pand(equal, equal, early, mask);
                    por(keep, strict, equal, mask);
                    vcmps_gt(valid, key, uint16_t(0x007f), mask);
                    pand(keep, keep, valid, mask);
                    vsqz(selected, key, keep, MODE_STORED);
                    vstur(align, selected, out, POST_UPDATE);
                }
                vstar(align, out);
            }
        }
        __tf__ __aicore__ void unpack_scores(__ubuf__ bfloat16_t* dst, __ubuf__ uint16_t* src) {
            __VEC_SCOPE__ {
                vector_u16 key, positive, negative, bits, sign;
                vector_bool mask;
                auto full = pset_b16(PAT_ALL);
                vbr(sign, uint16_t(0x8000));
                for (uint16_t c = 0; c < 2; ++c) {
                    vlds(key, src, c * 128, NORM);
                    vcmps_ge(mask, key, uint16_t(0x8000), full);
                    vxor(positive, key, sign, full, MODE_ZEROING);
                    vnot(negative, key, full, MODE_ZEROING);
                    vsel(bits, positive, negative, mask);
                    vsts(bits, (__ubuf__ uint16_t*)dst, c * 128, NORM_B16, full);
                }
            }
        }
        __tf__ __aicore__ void block_max_keys(__ubuf__ uint32_t* dst, __ubuf__ uint16_t* src, int width) {
            __VEC_SCOPE__ {
                vector_u32 keys, maxima, lower, zero, negative_tail;
                vector_bool negative;
                auto full = pset_b32(PAT_ALL);
                uint32_t count = 8;
                auto reduced = CreatePredicate<uint32_t>(count);
                vbr(zero, 0u);
                vbr(negative_tail, 0xffffu);
                for (uint16_t c = 0; c < (width + 63) / 64; ++c) {
                    vlds((vector_u16&)keys, src, c * 64, UNPK_B16);
                    vcgmax(maxima, keys, full);
                    vcmps_lt(negative, maxima, 0x8000u, reduced);
                    vsel(lower, negative_tail, zero, negative);
                    vshls(maxima, maxima, 16, reduced, MODE_ZEROING);
                    vor(maxima, maxima, lower, reduced, MODE_ZEROING);
                    vsts(maxima, dst, c * 8, NORM_B32, reduced);
                }
            }
        }
        __tf__ __aicore__ void block_maxima(__ubuf__ float* dst, __ubuf__ uint16_t* src, int width) {
            __VEC_SCOPE__ {
                vector_u32 keys, maxima, positive, negative, bits, sign;
                vector_bool positive_mask;
                auto full = pset_b32(PAT_ALL);
                uint32_t count = 8;
                auto reduced = CreatePredicate<uint32_t>(count);
                vbr(sign, 0x8000u);
                for (uint16_t c = 0; c < (width + 63) / 64; ++c) {
                    vlds((vector_u16&)keys, src, c * 64, UNPK_B16);
                    vcgmax(maxima, keys, full);
                    vcmps_ge(positive_mask, maxima, 0x8000u, reduced);
                    vxor(positive, maxima, sign, reduced, MODE_ZEROING);
                    vnot(negative, maxima, reduced, MODE_ZEROING);
                    vsel(bits, positive, negative, positive_mask);
                    vshls(bits, bits, 16, reduced, MODE_ZEROING);
                    vsts((vector_f32&)bits, dst, c * 8, NORM_B32, reduced);
                }
            }
        }
        __aicore__ inline void store_scores(__gm__ int64_t* args, int begin, int count,
                                            int logical, int cached_width) {
            int width = args[15];
            Vec<bfloat16_t, 1, 256> rounded;
            Vec<float, 1, 256> values;
            TASSIGN(rounded, 0);
            TASSIGN(values, 512);
            for (int row = 0; row < 3; ++row) {
                int offset = get_subblockid() * 3 + row;
                if (offset >= count) continue;
                if (args[20] >= 2) {
                    if (args[20] == 3) block_max_keys(ub<uint32_t>(4096), ub<uint16_t>(98304 + row * 51200), cached_width);
                    else block_maxima(ub<float>(4096), ub<uint16_t>(98304 + row * 51200), cached_width);
                } else {
                    for (int column = 0; column < cached_width; column += 256) {
                    unpack_scores(rounded.data(), ub<uint16_t>(98304 + row * 51200) + column);
                    TCVT(values, rounded, RoundMode::CAST_RINT);
                    {
                        int valid = minimum(256, cached_width - column);
                        Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, DYNAMIC> output(valid);
                        TASSIGN(output, 512);
                        GM<float, 1, DYNAMIC, 1, DYNAMIC> gm(
                            data<float>(args, 14) + (begin + offset) * width + logical + column,
                            TileShape2D<float, 1, DYNAMIC>(1, valid), BaseShape2D<float, 1, DYNAMIC>(1, width));
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        TSTORE(gm, output);
                        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    }
                    }
                }
                if (args[20] >= 2) {
                    int blocks = (cached_width + 7) / 8;
                    Tile<TileType::Vec, float, 1, 3200, BLayout::RowMajor, 1, DYNAMIC> maxima(blocks);
                    TASSIGN(maxima, 4096);
                    GM<float, 1, DYNAMIC, 1, DYNAMIC> gm(
                        data<float>(args, 14) + (begin + offset) * width + logical / 8,
                        TileShape2D<float, 1, DYNAMIC>(1, blocks), BaseShape2D<float, 1, DYNAMIC>(1, width));
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(gm, maxima);
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                }
            }
        }
        __tf__ __aicore__ void short_page(__ubuf__ uint32_t* output, int base, int count, int page_rows) {
            __VEC_SCOPE__ {
                vector_u32 lane, physical, invalid, result;
                vector_bool valid;
                auto full = pset_b32(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                for (uint16_t chunk = 0; chunk < page_rows / 64; ++chunk) {
                    vci((vector_s32&)lane, int32_t(chunk * 64), INC_ORDER);
                    vadds(physical, lane, uint32_t(base), full, MODE_ZEROING);
                    vcmps_lt(valid, (vector_s32&)lane, count, full);
                    vsel(result, physical, invalid, valid);
                    vsts(result, output, chunk * 64, NORM_B32, full);
                }
            }
        }

        __aicore__ inline void write_short_topk(__gm__ int64_t* args, int token,
                                               int length, CandidateIndices& indices) {
            int table_width = args[17];
            int cache_pages = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0];
            int page_rows = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[1];
            int request = read(data<int32_t>(args, 6) + token);
            for (int block = 0; block < 512 / page_rows; ++block) {
                int count = minimum(page_rows, length - block * page_rows);
                int page = -1;
                if (count > 0 && block < table_width) {
                    page = read(data<int32_t>(args, 5) + request * table_width + block);
                }
                if (page < 0 || page >= cache_pages) {
                    page = 0;
                    count = 0;
                }
                short_page(indices.data() + block * page_rows, page * page_rows, count, page_rows);
            }
            GM<uint32_t, 1, 512, 1, 512> out(data<uint32_t>(args, 8) + token * 512);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(out, indices);
            pipe_barrier(PIPE_ALL);
        }

        __aicore__ inline void merge(__gm__ int64_t* args) {
            int width = args[15], stride = args[16], table_width = args[17], tokens = args[18];
            int workers = get_block_num(args), worker = get_block_idx(args) * 2 + get_subblockid();
            int gathered_rows = args[19], shards = args[21];
            int panels = (width + 255) / 256, total = read(data<int32_t>(args, 9) + 3 * stride) * panels;
            CandidateKeys selected_keys;
            CandidateIndices selected_indices;
            TASSIGN(selected_keys, 131584);
            TASSIGN(selected_indices, 132608);
            for (int token = worker; token < tokens; token += workers * 2) {
                if (args[22]) {
                    int length = read(data<int32_t>(args, 7) + token);
                    if (length <= args[22]) {
                        write_short_topk(args, token, length, selected_indices);
                        continue;
                    }
                }
                GM<uint32_t, 1, 512, 1, 512> out(data<uint32_t>(args, 8) + token * 512);
                if (total == 0) {
                    TEXPANDS(selected_indices, 0xffffffffu);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    TSTORE(out, selected_indices);
                    pipe_barrier(PIPE_ALL);
                    continue;
                }
                int group = read(data<int32_t>(args, 9) + 3 * stride + token + 1);
                int owners = minimum(active_shards(args, total / panels, panels, workers), total);
                int first = ((group * panels + 1) * owners - 1) / total;
                int last = (((group + 1) * panels) * owners - 1) / total;
                int carry = 0;
                for (int start = first; start <= last;) {
                    int batch = minimum(last - start + 1, 32 - carry / 512);
                    int loaded = batch * 512, candidates = carry + loaded;
                    int index = (token * shards + start) * 512;
                    using KeyInput = Tile<TileType::Vec, uint16_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC>;
                    using IndexInput = Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC>;
                    KeyInput ki(loaded);
                    IndexInput ii(loaded);
                    TASSIGN(ki, carry * 2);
                    TASSIGN(ii, 65536 + carry * 4);
                    GM<uint16_t, 1, DYNAMIC, 1, DYNAMIC> kg(data<uint16_t>(args, 10) + index,
                        TileShape2D<uint16_t, 1, DYNAMIC>(1, loaded), BaseShape2D<uint16_t, 1, DYNAMIC>(1, loaded));
                    GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> ig(data<uint32_t>(args, 11) + index,
                        TileShape2D<uint32_t, 1, DYNAMIC>(1, loaded), BaseShape2D<uint32_t, 1, DYNAMIC>(1, loaded));
                    TLOAD(ki, kg);
                    TLOAD(ii, ig);
                    bool final = start + batch > last;
                    auto mapping = ub<uint32_t>(134656);
                    if (final && gathered_rows) {
                        Tile<TileType::Vec, uint32_t, 1, 24576, BLayout::RowMajor, 1, DYNAMIC> rows(width);
                        TASSIGN(rows, 143360);
                        GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> pg(data<uint32_t>(args, 13) + token * gathered_rows,
                            TileShape2D<uint32_t, 1, DYNAMIC>(1, width), BaseShape2D<uint32_t, 1, DYNAMIC>(1, width));
                        TLOAD(rows, pg);
                        mapping = rows.data();
                    } else if (final) {
                        int page_rows = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[1];
                        int request = read(data<int32_t>(args, 6) + token), page_count = (width + page_rows - 1) / page_rows;
                        Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> pages(page_count);
                        TASSIGN(pages, 134656);
                        GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC> pg(data<uint32_t>(args, 5) + request * table_width,
                            TileShape2D<uint32_t, 1, DYNAMIC>(1, page_count), BaseShape2D<uint32_t, 1, DYNAMIC>(1, table_width));
                        TLOAD(pages, pg);
                    }
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    uint32_t cutoff;
                    int boundary;
                    if (final) {
                        merge_select(selected_keys.data(), selected_indices.data(), ub<uint16_t>(0),
                            ub<uint32_t>(65536), mapping, candidates, gathered_rows != 0, final, ub<uint16_t>(32768),
                            reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[1] == 64 ? 6 : 7);
                    } else {
                        packed_threshold(ub<uint16_t>(0), candidates, 32768, 131072, cutoff, boundary);
                        packed_select(selected_keys.data(), selected_indices.data(), ub<uint16_t>(0),
                            ub<uint32_t>(65536), mapping, candidates, cutoff, boundary, gathered_rows != 0, final,
                            reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[1] == 64 ? 6 : 7);
                    }
                    if (final) {
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        TSTORE(out, selected_indices);
                    } else {
                        compact_keys(selected_keys.data(), ub<uint16_t>(0), candidates, cutoff, boundary);
                        CandidateKeys retained_keys;
                        CandidateIndices retained_indices;
                        TASSIGN(retained_keys, 0);
                        TASSIGN(retained_indices, 65536);
                        TMOV(retained_keys, selected_keys);
                        TMOV(retained_indices, selected_indices);
                        carry = 512;
                    }
                    pipe_barrier(PIPE_ALL);
                    start += batch;
                }
            }
        }
#endif
        template <typename T, typename G> __aicore__ inline void load_value(T& tile, G global) {
            TLOAD(tile, global);
        }
        template <typename G, typename T> __aicore__ inline void store_value(G global, T& tile) {
            TSTORE(global, tile);
        }
#if defined(__DAV_VEC__)
        __tf__ __aicore__ void pool_score(__ubuf__ bfloat16_t* src, __ubuf__ bfloat16_t* weights,
                                          __ubuf__ uint16_t* cache, __ubuf__ uint16_t* hist,
                                          __ubuf__ uint16_t* valid, int column) {
            __VEC_SCOPE__ {
                RegTensor<bfloat16_t> a, b, c, d, x, y, z, t, weight;
                auto mask = pset_b16(PAT_ALL);
                vbr(a, bfloat16_t(0));
                vbr(b, bfloat16_t(0));
                vbr(c, bfloat16_t(0));
                vbr(d, bfloat16_t(0));
#pragma unroll 4
                for (uint16_t h = 0; h < 32; ++h) {
                    vlds(weight, weights, h, BRC_B16);
                    vlds(x, src, h * 512, NORM);
                    vlds(y, src, h * 512 + 128, NORM);
                    vlds(z, src, h * 512 + 256, NORM);
                    vlds(t, src, h * 512 + 384, NORM);

                    vmula(a, x, weight, mask, MODE_ZEROING);
                    vmula(b, y, weight, mask, MODE_ZEROING);
                    vmula(c, z, weight, mask, MODE_ZEROING);
                    vmula(d, t, weight, mask, MODE_ZEROING);
                }
                write_cached_row(a, b, cache + column, hist, 128, 128, 128, 128, column == 0, valid, true);
                write_cached_row(c, d, cache + column + 256, hist, 128, 128, 128, 128, false, valid + 256,
                                 true);
            }
        }

        __tf__ __aicore__ void pool_map(__ubuf__ uint32_t* indices, __ubuf__ uint32_t* candidates,
                                        __ubuf__ uint32_t* pages) {
            __VEC_SCOPE__ {
                vector_u32 pos, block, logical, physical, within, mask7, mask127;
                vector_bool valid;
                auto mask = pset_b32(PAT_ALL);
                vbr(mask7, uint32_t(7));
                vbr(mask127, uint32_t(127));
                for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                    vlds(pos, indices, chunk * 64, NORM);
                    vcmps_ne(valid, pos, uint32_t(-1), mask);
                    vshrs(block, pos, 3, mask, MODE_ZEROING);
                    vgather2(logical, candidates, block, valid);
                    vshls(logical, logical, 3, mask, MODE_ZEROING);
                    vand(within, pos, mask7, mask, MODE_ZEROING);
                    vadd(logical, logical, within, mask, MODE_ZEROING);
                    vshrs(block, logical, 7, mask, MODE_ZEROING);
                    vgather2(physical, pages, block, valid);
                    vshls(physical, physical, 7, mask, MODE_ZEROING);
                    vand(within, logical, mask127, mask, MODE_ZEROING);
                    vadd(physical, physical, within, mask, MODE_ZEROING);
                    vsel(pos, physical, pos, valid);
                    vsts(pos, indices, chunk * 64, NORM_B32, mask);
                }
            }
        }

        __tf__ __aicore__ void prepare_pool_metadata(__ubuf__ int32_t* ids, __ubuf__ int32_t* pages,
                                                     __ubuf__ int32_t* direct, int capacity, int count,
                                                     int visible, int table_width, int cache_pages) {
            __VEC_SCOPE__ {
                vector_u32 candidate, page_index, page, invalid, zero, lane, first, expected, error, reduced,
                    within;
                vector_bool active, bounded, mapped, contiguous, aligned;
                auto all = pset_b32(PAT_ALL);
                vbr(invalid, uint32_t(-1));
                vbr(zero, uint32_t(0));
                uint32_t remaining = capacity;
                for (uint16_t chunk = 0; chunk < (capacity + 63) / 64; ++chunk) {
                    auto mask = CreatePredicate<uint32_t>(remaining);
                    vlds(candidate, (__ubuf__ uint32_t*)ids, chunk * 64, NORM);
                    vci((vector_s32&)lane, int32_t(chunk * 64), INC_ORDER);
                    vcmps_lt(active, lane, uint32_t(count), mask);
                    vcmps_lt(bounded, candidate, uint32_t((visible + 7) / 8), mask);
                    pand(active, active, bounded, mask);
                    vshrs(page_index, candidate, 4, mask, MODE_ZEROING);
                    vcmps_lt(bounded, page_index, uint32_t(table_width), mask);
                    pand(active, active, bounded, mask);
                    vgather2(page, (__ubuf__ uint32_t*)pages, page_index, active);
                    vcmps_lt(mapped, page, uint32_t(cache_pages), mask);
                    pand(active, active, mapped, mask);
                    vsel(candidate, candidate, invalid, active);
                    vsts(candidate, (__ubuf__ uint32_t*)ids, chunk * 64, NORM_B32, mask);
                }
                mem_bar(VST_VLD);
                uint32_t sixteen = 16, one = 1;
                auto block = CreatePredicate<uint32_t>(sixteen);
                auto scalar = CreatePredicate<uint32_t>(one);
                vci((vector_s32&)lane, 0, INC_ORDER);
                for (uint16_t half = 0; half < capacity / 16; ++half) {
                    vlds(candidate, (__ubuf__ uint32_t*)ids, half * 16, NORM);
                    vlds(first, (__ubuf__ uint32_t*)ids, half * 16, BRC_B32);
                    vadd(expected, first, lane, block, MODE_ZEROING);
                    vxor(error, candidate, expected, block, MODE_ZEROING);
                    vcmax(reduced, error, block, MODE_ZEROING);
                    vcmps_eq(contiguous, reduced, uint32_t(0), scalar);
                    vshrs(page_index, first, 4, scalar, MODE_ZEROING);
                    vcmps_lt(bounded, page_index, uint32_t(table_width), scalar);
                    pand(contiguous, contiguous, bounded, scalar);
                    vbr(within, uint32_t(15));
                    vand(within, first, within, scalar, MODE_ZEROING);
                    vcmps_eq(aligned, within, uint32_t(0), scalar);
                    pand(contiguous, contiguous, aligned, scalar);
                    vgather2(page, (__ubuf__ uint32_t*)pages, page_index, contiguous);
                    vsel(page, page, invalid, contiguous);
                    vsts(page, (__ubuf__ uint32_t*)direct, half * 8, NORM_B32, scalar);
                }
            }
        }

        __tf__ __aicore__ void pool_validity(__ubuf__ int32_t* ids, __ubuf__ int32_t* output, int visible,
                                             int count) {
            __VEC_SCOPE__ {
                vector_u32 row, block, logical, within, seven, one, zero, result;
                vector_bool causal, active;
                auto full = pset_b32(PAT_ALL);
                vbr(seven, uint32_t(7));
                vbr(one, uint32_t(1));
                vbr(zero, uint32_t(0));
                for (uint16_t part = 0; part < 8; ++part) {
                    vci((vector_s32&)row, int32_t(part * 64), INC_ORDER);
                    vshrs(block, row, 3, full, MODE_ZEROING);
                    vgather2(logical, (__ubuf__ uint32_t*)ids, block, full);
                    vshls(logical, logical, 3, full, MODE_ZEROING);
                    vand(within, row, seven, full, MODE_ZEROING);
                    vadd(logical, logical, within, full, MODE_ZEROING);
                    vcmps_lt(causal, logical, uint32_t(visible), full);
                    vcmps_lt(active, row, uint32_t(count), full);
                    pand(active, active, causal, full);
                    vsel(result, one, zero, active);
                    vsts(result, (__ubuf__ uint32_t*)output, part * 64, NORM_B32, full);
                }
            }
        }

        __tf__ __aicore__ void pool_physical_rows(__ubuf__ int32_t* ids, __ubuf__ int32_t* pages,
                                                  __ubuf__ uint32_t* output, int capacity) {
            __VEC_SCOPE__ {
                vector_u32 candidate, page, physical, within, fifteen, zero;
                vector_bool valid;
                vbr(fifteen, uint32_t(15));
                vbr(zero, uint32_t(0));
                uint32_t remaining = capacity;
                for (uint16_t chunk = 0; chunk < (capacity + 63) / 64; ++chunk) {
                    auto mask = CreatePredicate<uint32_t>(remaining);
                    vlds(candidate, (__ubuf__ uint32_t*)ids, chunk * 64, NORM);
                    vcmps_ne(valid, candidate, uint32_t(-1), mask);
                    vshrs(page, candidate, 4, mask, MODE_ZEROING);
                    vgather2(physical, (__ubuf__ uint32_t*)pages, page, valid);
                    vshls(physical, physical, 7, mask, MODE_ZEROING);
                    vand(within, candidate, fifteen, mask, MODE_ZEROING);
                    vshls(within, within, 3, mask, MODE_ZEROING);
                    vadd(physical, physical, within, mask, MODE_ZEROING);
                    vsel(physical, physical, zero, valid);
                    vsts(physical, output, chunk * 64, NORM_B32, mask);
                }
            }
        }

        __aicore__ inline void gather_pool_half(__gm__ int64_t* args, int panel, __gm__ uint8_t* ring,
                                                int slot, bool reuse) {
            int lane = get_subblockid();
            int ub_base = lane == 0 ? (slot == 0 ? 0 : 147456) : (slot == 0 ? 65536 : 222208);
            auto physical = ub<uint32_t>(lane == 0 ? 49152 : 82944) + panel * 64 + lane * 32;
            auto payload = data<uint8_t>(args, 3);
            auto scale = data<uint8_t>(args, 4);
            if (reuse)
                wait_flag(PIPE_MTE3, PIPE_MTE2, event_t(4 + slot));
#pragma unroll 8
            for (int pair = 0; pair < 16; ++pair) {
                uint64_t addresses = reinterpret_cast<__ubuf__ uint64_t*>(physical)[pair];
                int64_t first = uint32_t(addresses), second = uint32_t(addresses >> 32);
                if (second >= first + 8) {
                    int64_t payload_stride = (second - first) * 64;
                    int64_t scale_stride = (second - first) * 4;
                    Vec<uint8_t, 2, 512> keys;
                    Vec<uint8_t, 2, 32> scales;
                    TASSIGN(keys, ub_base + pair * 1024);
                    TASSIGN(scales, ub_base + 16384 + pair * 64);
                    load_value(keys, (GM<uint8_t, 2, 512, 2, DYNAMIC>(
                                         payload + first * 64, TileShape2D<uint8_t, 2, 512>(2, 512),
                                         BaseShape2D<uint8_t, 2, DYNAMIC>(2, payload_stride))));
                    load_value(scales, (GM<uint8_t, 2, 32, 2, DYNAMIC>(
                                           scale + first * 4, TileShape2D<uint8_t, 2, 32>(2, 32),
                                           BaseShape2D<uint8_t, 2, DYNAMIC>(2, scale_stride))));
                } else {
                    for (int half = 0; half < 2; ++half) {
                        int64_t row = half == 0 ? first : second;
                        Vec<uint8_t, 8, 64> keys;
                        Vec<uint8_t, 1, 32> scales;
                        TASSIGN(keys, ub_base + pair * 1024 + half * 512);
                        TASSIGN(scales, ub_base + 16384 + pair * 64 + half * 32);
                        load_value(keys, (GM<uint8_t, 8, 64, 8, 64>(payload + row * 64)));
                        load_value(scales, (GM<uint8_t, 1, 32, 1, 32>(scale + row * 4)));
                    }
                }
            }
            Vec<uint8_t, 256, 64> keys;
            Vec<uint8_t, 1, 1024> scales;
            TASSIGN(keys, ub_base);
            TASSIGN(scales, ub_base + 16384);
            set_flag(PIPE_MTE2, PIPE_MTE3, event_t(4 + slot));
            wait_flag(PIPE_MTE2, PIPE_MTE3, event_t(4 + slot));
            store_value((GM<uint8_t, 256, 64, 256, 64>(ring + lane * 16384)), keys);
            store_value((GM<uint8_t, 1, 1024, 1, 1024>(ring + 32768 + lane * 1024)), scales);
            set_flag(PIPE_MTE3, PIPE_MTE2, event_t(4 + slot));
        }
#endif

        // Each worker owns two 34 KiB gather slots and 12 KiB of sanitized pool metadata.
        // AIV0 gathers the first 256 tokens; AIV1 gathers the second half and reduces scores.
        __aicore__ inline void run_pool(__gm__ int64_t* args) {
            int worker = get_block_idx(args), workers = get_block_num(args);
            int capacity = reinterpret_cast<__gm__ Tensor*>(args[12])->shapes[1];
            int tokens = args[18], table_width = args[17];
            auto cache_payload = data<F4>(args, 3);
            auto cache_scale = data<S>(args, 4);
            int padded_capacity = (capacity + 63) / 64 * 64;
            int token_panels = padded_capacity / 64;
            int owners = minimum(workers, tokens * token_panels);
            if (worker >= owners) return;
            int first = tokens * token_panels * worker / owners;
            int finish = tokens * token_panels * (worker + 1) / owners;
            for (int cursor = first; cursor < finish;) {
                int token_group = cursor / token_panels;
                int token = token_group;
                if (tokens % workers == 0) {
                    token = (token_group - first / token_panels) * workers + worker;
                }
                int first_panel = cursor % token_panels;
                int next = minimum(finish, (token_group + 1) * token_panels);
                int last_panel = next - token_group * token_panels;
                int count = minimum(capacity, read(data<int32_t>(args, 13) + token));
                if (count < 0)
                    count = 0;
                int request = read(data<int32_t>(args, 6) + token);
                int visible = read(data<int32_t>(args, 7) + token);
                if (visible < 0)
                    visible = 0;
                if (request < 0 || request >= reinterpret_cast<__gm__ Tensor*>(args[5])->shapes[0]) {
                    request = 0;
                    count = 0;
                }
                last_panel = minimum(last_panel, (count + 63) / 64);
                int panels = last_panel > first_panel ? last_panel - first_panel : 0;
                auto metadata = data<int32_t>(args, 14) + worker * 20480 + 17408;
                bool gathered_slot0 = false, gathered_slot1 = false;
#if defined(__DAV_CUBE__)
                Tile<TileType::Mat, F4, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor> query;
                TASSIGN(query, 0);
                GM<F4, 32, 128, 32, 128> qdata(data<F4>(args, 0) + token * 2048);
                TLOAD(query, qdata);
                QueryScale query_scale(32);
                TASSIGN(query_scale, 49152);
                using QS = TileShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                using QD = BaseShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                GlobalTensor<S, QS, QD, Layout::MX_A_ND> qg(data<S>(args, 1) + token * 128, QS(32, 4),
                                                            QD(32, 4));
                TLOAD(query_scale, qg);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                TileLeft<F4, 32, 128> left;
                TileLeftScale<S, 32, 4> left_scale;
                TASSIGN(left, 0);
                TASSIGN(left_scale, GetScaleAddr(left.data()));
                TEXTRACT(left, query, 0, 0);
                TEXTRACT(left_scale, query_scale, 0, 0);
#else
                prepare_weights(args, token, 1);
                if (get_subblockid() == 0) {
                    Tile<TileType::Vec, int32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> pages(table_width);
                    TASSIGN(pages, 73728);
                    load_value(pages, (GM<int32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                          data<int32_t>(args, 5) + request * table_width,
                                          TileShape2D<int32_t, 1, DYNAMIC>(1, table_width),
                                          BaseShape2D<int32_t, 1, DYNAMIC>(1, table_width))));
                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                    Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, DYNAMIC> ids(capacity);
                    TASSIGN(ids, 32768);
                    Vec<int32_t, 1, 2048> padded_ids;
                    TASSIGN(padded_ids, 32768);
                    TEXPANDS(padded_ids, int32_t(-1));
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    load_value(ids, (GM<int32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                        data<int32_t>(args, 12) + token * capacity,
                                        TileShape2D<int32_t, 1, DYNAMIC>(1, capacity),
                                        BaseShape2D<int32_t, 1, DYNAMIC>(1, capacity))));
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    prepare_pool_metadata(ids.data(), pages.data(), ub<int32_t>(40960), padded_capacity,
                                          count, visible, table_width,
                                          reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0]);
                    pool_physical_rows(ids.data(), pages.data(), ub<uint32_t>(49152), padded_capacity);
                    Vec<int32_t, 1, 1024> direct;
                    TASSIGN(direct, 40960);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    store_value((GM<int32_t, 1, 2048, 1, 2048>(metadata)), padded_ids);
                    store_value((GM<int32_t, 1, 1024, 1, 1024>(metadata + 2048)), direct);
                    set_intra_block(PIPE_MTE3, 7);
                    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                } else {
                    wait_intra_block(PIPE_MTE2, 7);
                    Tile<TileType::Vec, int32_t, 1, 2048, BLayout::RowMajor, 1, DYNAMIC> ids(padded_capacity);
                    TASSIGN(ids, 134656);
                    load_value(ids, (GM<int32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                        metadata, TileShape2D<int32_t, 1, DYNAMIC>(1, padded_capacity),
                                        BaseShape2D<int32_t, 1, DYNAMIC>(1, padded_capacity))));
                    Tile<TileType::Vec, int32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> pages(table_width);
                    Vec<int32_t, 1, 1024> direct;
                    TASSIGN(pages, 143360);
                    TASSIGN(direct, 212992);
                    load_value(pages, (GM<int32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                          data<int32_t>(args, 5) + request * table_width,
                                          TileShape2D<int32_t, 1, DYNAMIC>(1, table_width),
                                          BaseShape2D<int32_t, 1, DYNAMIC>(1, table_width))));
                    load_value(direct, (GM<int32_t, 1, 1024, 1, 1024>(metadata + 2048)));
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    pool_physical_rows(ids.data(), pages.data(), ub<uint32_t>(82944), padded_capacity);
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                }
#endif
#if defined(__DAV_CUBE__)
                wait_intra_block(PIPE_S, 7);
                set_intra_block(PIPE_S, 23);
#endif
                for (int local_panel = 0; local_panel < panels; ++local_panel) {
                    int panel = first_panel + local_panel;
                    int slot = local_panel % 2;
                    auto ring = data<uint8_t>(args, 14) + worker * 81920 + slot * 34816;
#if defined(__DAV_CUBE__)
                    int page0 = read(metadata + 2048 + panel * 32);
                    int page1 = metadata[2048 + panel * 32 + 8];
                    int page2 = read(metadata + 2048 + panel * 32 + 16);
                    int page3 = metadata[2048 + panel * 32 + 24];
                    bool direct = page0 >= 0 && page1 >= 0 && page2 >= 0 && page3 >= 0;
                    if (!direct) {
                        wait_intra_block(PIPE_MTE2, 8 + slot * 2);
                        wait_intra_block(PIPE_MTE2, 24 + slot * 2);
                    }
                    using Key =
                        Tile<TileType::Mat, F4, 128, 512, BLayout::RowMajor, 128, 512, SLayout::ColMajor>;
                    using Scale =
                        Tile<TileType::Mat, S, 4, 512, BLayout::ColMajor, 4, 512, SLayout::ColMajor, 32>;
                    Key key;
                    Scale key_scale;
                    TileRight<F4, 128, 512> right;
                    TileRightScale<S, 4, 512> right_scale;
                    TileAcc<float, 32, 512> dot;
                    TASSIGN(key, 65536 + slot * 65536);
                    TASSIGN(key_scale, 98304 + slot * 65536);
                    TASSIGN(right, slot * 32768);
                    TASSIGN(right_scale, GetScaleAddr(right.data()));
                    TASSIGN(dot, 0);
                    if (local_panel >= 2)
                        wait_flag(PIPE_MTE1, PIPE_MTE2, event_t(2 + slot));
                    bool contiguous = !direct || (page1 == page0 + 1 && page2 == page0 + 2 && page3 == page0 + 3);
                    if (contiguous) {
                        auto kp = direct ? cache_payload + int64_t(page0) * 8192
                                         : reinterpret_cast<__gm__ F4*>(ring);
                        auto sp = direct ? cache_scale + int64_t(page0) * 512
                                         : reinterpret_cast<__gm__ S*>(ring + 32768);
                        GM<F4, 128, 512, 128, 512, Layout::DN> kg(kp);
                        GM<S, 4, 512, 4, 512, Layout::MX_B_DN> sg(sp);
                        TLOAD(key, kg);
                        TLOAD(key_scale, sg);
                    } else {
                        for (int piece = 0; piece < 4; ++piece) {
                            using KeyPart =
                                Tile<TileType::Mat, F4, 128, 512, BLayout::RowMajor, 128, 128, SLayout::ColMajor>;
                            KeyPart part;
                            KeyScale part_scale;
                            TASSIGN(part, 65536 + slot * 65536 + piece * 4096);
                            TASSIGN(part_scale, 98304 + slot * 65536 + piece * 512);
                            int page = piece == 0 ? page0 : piece == 1 ? page1 : piece == 2 ? page2 : page3;
                            auto kp = direct ? cache_payload + int64_t(page) * 8192
                                             : reinterpret_cast<__gm__ F4*>(ring + piece * 8192);
                            auto sp = direct ? cache_scale + int64_t(page) * 512
                                             : reinterpret_cast<__gm__ S*>(ring + 32768 + piece * 512);
                            GM<F4, 128, 128, 128, 128, Layout::DN> kg(kp);
                            GM<S, 4, 128, 4, 128, Layout::MX_B_DN> sg(sp);
                            TLOAD(part, kg);
                            TLOAD(part_scale, sg);
                        }
                    }
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    if (local_panel >= 2)
                        wait_flag(PIPE_M, PIPE_MTE1, event_t(2 + slot));
                    TMOV(right, key);
                    TMOV(right_scale, key_scale);
                    set_flag(PIPE_MTE1, PIPE_MTE2, event_t(2 + slot));
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    TMATMUL_MX<AccPhase::Final>(dot, left, left_scale, right, right_scale);
                    set_flag(PIPE_M, PIPE_MTE1, event_t(2 + slot));
                    if (local_panel >= 2)
                        wait_intra_block(PIPE_FIX, slot * 2 + 17);
                    using Received = Vec<bfloat16_t, 32, 512>;
                    Received received;
                    TASSIGN(received, slot * 32768);
                    TMOV<STPhase::Final, Received, TileAcc<float, 32, 512>, AccToVecMode::SingleModeVec1,
                         ReluPreMode::NormalRelu>(received, dot);
                    set_intra_block(PIPE_FIX, slot * 2 + 16);
                    if (!direct) {
                        set_intra_block(PIPE_MTE2, 9 + slot * 2);
                        set_intra_block(PIPE_MTE2, 25 + slot * 2);
                    }
#else
                    int direct_base = get_subblockid() == 0 ? 40960 : 212992;
                    bool direct = ub<int32_t>(direct_base)[panel * 32] >= 0 &&
                                  ub<int32_t>(direct_base)[panel * 32 + 8] >= 0;
                    direct = direct && ub<int32_t>(direct_base)[panel * 32 + 16] >= 0 &&
                             ub<int32_t>(direct_base)[panel * 32 + 24] >= 0;
                    if (!direct) {
                        if (slot == 0 ? gathered_slot0 : gathered_slot1)
                            wait_intra_block(PIPE_MTE2, 9 + slot * 2);
                        gather_pool_half(args, panel, ring, slot, slot == 0 ? gathered_slot0 : gathered_slot1);
                        set_intra_block(PIPE_MTE3, 8 + slot * 2);
                        if (slot == 0)
                            gathered_slot0 = true;
                        else
                            gathered_slot1 = true;
                    }
                    if (get_subblockid() == 1) {
                        wait_intra_block(PIPE_V, slot * 2);
                        Vec<int32_t, 1, 512> raw_valid;
                        Vec<uint16_t, 1, 512> valid;
                        TASSIGN(raw_valid, 240640 + slot * 2048);
                        TASSIGN(valid, 244736 + slot * 1024);
                        pool_validity(ub<int32_t>(134656) + panel * 64, raw_valid.data(), visible,
                                      minimum(512, (count - panel * 64) * 8));
                        TCVT(valid, raw_valid, RoundMode::CAST_NONE);
                        pool_score(ub<bfloat16_t>(slot * 32768), ub<bfloat16_t>(257280), ub<uint16_t>(98304),
                                   ub<uint16_t>(251904), valid.data(), local_panel * 512);
                        set_intra_block(PIPE_V, slot * 2 + 1);
                    }
#endif
                }
#if defined(__DAV_CUBE__)
                for (int slot = 0; slot < minimum(2, panels); ++slot) {
                    wait_flag(PIPE_MTE1, PIPE_MTE2, event_t(2 + slot));
                    wait_flag(PIPE_M, PIPE_MTE1, event_t(2 + slot));
                }
                for (int slot = 0; slot < minimum(2, panels); ++slot)
                    wait_intra_block(PIPE_FIX, slot * 2 + 17);
                wait_intra_block(PIPE_S, 22);
                set_intra_block(PIPE_S, 4);
#else
                if (gathered_slot0)
                    wait_intra_block(PIPE_MTE2, 9);
                if (gathered_slot1)
                    wait_intra_block(PIPE_MTE2, 11);
                if (gathered_slot0)
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
                if (gathered_slot1)
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
                if (get_subblockid() == 1) {
                    CandidateKeys keys;
                    CandidateIndices indices;
                    TASSIGN(keys, 51200);
                    TASSIGN(indices, 52224);
                    // Histogram counts include the masked suffix of the last 512-token tile.
                    int cached_width = panels * 512;
                    local_select(keys.data(), indices.data(), ub<uint16_t>(98304), cached_width,
                                 first_panel * 512, ub<uint16_t>(251904));
                    bool complete = first_panel == 0 && next == (token_group + 1) * token_panels;
                    if (complete) {
                        pool_map(indices.data(), ub<uint32_t>(134656), ub<uint32_t>(143360));
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        store_value((GM<uint32_t, 1, 512, 1, 512>(data<uint32_t>(args, 8) + token * 512)),
                                    indices);
                    } else {
                        int output_offset = (token * workers + worker) * 512;
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        store_value((GM<uint16_t, 1, 512, 1, 512>(data<uint16_t>(args, 10) + output_offset)),
                                    keys);
                        store_value((GM<uint32_t, 1, 512, 1, 512>(data<uint32_t>(args, 11) + output_offset)),
                                    indices);
                    }
                    pipe_barrier(PIPE_ALL);
                    set_intra_block(PIPE_MTE3, 6);
                } else {
                    // Keep the current pool metadata live until both consumers finish.
                    wait_intra_block(PIPE_S, 4);
                }
#endif
                cursor = next;
            }
        }

#if defined(__DAV_VEC__)
        __aicore__ inline void merge_pool(__gm__ int64_t* args) {
            int worker = get_block_idx(args) * 2 + get_subblockid();
            int workers = get_block_num(args), tokens = args[18];
            int capacity = reinterpret_cast<__gm__ Tensor*>(args[12])->shapes[1];
            int panels = (capacity + 63) / 64, total = tokens * panels, table_width = args[17];
            int owners = minimum(workers, total);
            for (int token = worker; token < tokens; token += workers * 2) {
                int first = ((token * panels + 1) * owners - 1) / total;
                int last = (((token + 1) * panels) * owners - 1) / total;
                if (first == last)
                    continue;
                int count = (last - first + 1) * 512;
                int offset = (token * workers + first) * 512;
                Tile<TileType::Vec, uint16_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> keys(count);
                Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> indices(count);
                TASSIGN(keys, 0);
                TASSIGN(indices, 65536);
                load_value(keys,
                           (GM<uint16_t, 1, DYNAMIC, 1, DYNAMIC>(
                               data<uint16_t>(args, 10) + offset, TileShape2D<uint16_t, 1, DYNAMIC>(1, count),
                               BaseShape2D<uint16_t, 1, DYNAMIC>(1, count))));
                load_value(indices,
                           (GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC>(
                               data<uint32_t>(args, 11) + offset, TileShape2D<uint32_t, 1, DYNAMIC>(1, count),
                               BaseShape2D<uint32_t, 1, DYNAMIC>(1, count))));
                Tile<TileType::Vec, uint32_t, 1, 2048, BLayout::RowMajor, 1, DYNAMIC> candidates(capacity);
                Tile<TileType::Vec, uint32_t, 1, 16384, BLayout::RowMajor, 1, DYNAMIC> pages(table_width);
                TASSIGN(candidates, 134656);
                TASSIGN(pages, 143360);
                int request = read(data<int32_t>(args, 6) + token);
                if (request < 0 || request >= reinterpret_cast<__gm__ Tensor*>(args[5])->shapes[0])
                    request = 0;
                load_value(candidates, (GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                           data<uint32_t>(args, 12) + token * capacity,
                                           TileShape2D<uint32_t, 1, DYNAMIC>(1, capacity),
                                           BaseShape2D<uint32_t, 1, DYNAMIC>(1, capacity))));
                load_value(pages, (GM<uint32_t, 1, DYNAMIC, 1, DYNAMIC>(
                                      data<uint32_t>(args, 5) + request * table_width,
                                      TileShape2D<uint32_t, 1, DYNAMIC>(1, table_width),
                                      BaseShape2D<uint32_t, 1, DYNAMIC>(1, table_width))));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                uint32_t cutoff;
                int boundary;
                packed_threshold(keys.data(), count, 32768, 131072, cutoff, boundary);
                CandidateKeys selected_keys;
                CandidateIndices selected_indices;
                TASSIGN(selected_keys, 131584);
                TASSIGN(selected_indices, 132608);
                packed_select(selected_keys.data(), selected_indices.data(), keys.data(), indices.data(),
                              (__ubuf__ uint32_t*)0, count, cutoff, boundary, false, false);
                pool_map(selected_indices.data(), candidates.data(), pages.data());
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                store_value((GM<uint32_t, 1, 512, 1, 512>(data<uint32_t>(args, 8) + token * 512)),
                            selected_indices);
                pipe_barrier(PIPE_ALL);
            }
        }
#endif

#if defined(__DAV_CUBE__)
        template <int PageRows>
        __aicore__ inline bool prefetch_key(__gm__ int64_t* args, int request, int maximum,
                                            int first_panel, int panel, int cache_pages) {
            int logical = (first_panel + panel) * 256;
            int table_width = args[17];
            int pages[256 / PageRows];
            int fallback = -1;
            for (int piece = 0; piece < 256 / PageRows; ++piece) {
                int position = logical + piece * PageRows;
                int page = position < maximum && position / PageRows < table_width
                    ? read(data<int32_t>(args, 5) + request * table_width + position / PageRows) : -1;
                pages[piece] = page >= 0 && page < cache_pages ? page : -1;
                if (pages[piece] >= 0) fallback = pages[piece];
            }
            int slot = panel % 4;
            int event = slot + 2;
            if (panel >= 4) wait_flag(PIPE_MTE1, PIPE_MTE2, event);
            if (fallback >= 0) {
                using KeyPart = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, PageRows, SLayout::ColMajor>;
                using ScalePart = Tile<TileType::Mat, S, 4, PageRows, BLayout::ColMajor, 4, PageRows, SLayout::ColMajor, 32>;
                for (int piece = 0; piece < 256 / PageRows; ++piece) {
                    int page = pages[piece] >= 0 ? pages[piece] : fallback;
                    KeyPart key;
                    ScalePart scale;
                    TASSIGN(key, 65536 + slot * 65536 + piece * PageRows * 32);
                    TASSIGN(scale, 81920 + slot * 65536 + piece * PageRows * 4);
                    GM<F4, 128, PageRows, 128, PageRows, Layout::DN> kg(data<F4>(args, 3) + static_cast<int64_t>(page) * PageRows * 64);
                    GM<S, 4, PageRows, 4, PageRows, Layout::MX_B_DN> sg(data<S>(args, 4) + page * PageRows * 4);
                    TLOAD(key, kg);
                    TLOAD(scale, sg);
                }
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, event);
            return fallback >= 0;
        }

#endif

        template <int KeySlots, int PageRows = 128>
        __aicore__ inline void run(__gm__ int64_t* args) {
            int width = args[15], stride = args[16], table_width = args[17];
            int gathered_rows = args[19];
            int cache_pages = reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[0];
            int groups = read(data<int32_t>(args, 9) + 3 * stride);
            int panels = (width + 255) / 256;
            int block = get_block_idx(args), workers = get_block_num(args);
            int owners = minimum(active_shards(args, groups, panels, workers), groups * panels);
            int transferred = 0;
            for (int shard = block; shard < owners; shard += workers) {
                int first = groups * panels * shard / owners;
                int end = groups * panels * (shard + 1) / owners;
                for (int cursor = first; cursor < end;) {
                    int group_id = cursor / panels;
                    int finish = minimum(end, (group_id + 1) * panels);
                    int first_panel = cursor % panels;
                    int panel_count = finish - cursor;
                    int begin = read(data<int32_t>(args, 9) + group_id);
                    int count = read(data<int32_t>(args, 9) + stride + group_id) - begin;
                    int maximum = read(data<int32_t>(args, 9) + 2 * stride + group_id);
                    int request = read(data<int32_t>(args, 6) + begin);
#if defined(__DAV_CUBE__)
                    Tile<TileType::Mat, F4, 192, 128, BLayout::ColMajor, DYNAMIC, 128, SLayout::RowMajor> query(count * 32);
                    TASSIGN(query, 0);
                    GM<F4, DYNAMIC, 128, DYNAMIC, 128> qdata(data<F4>(args, 0) + begin * 2048,
                        TileShape2D<F4, DYNAMIC, 128>(count * 32, 128),
                        BaseShape2D<F4, DYNAMIC, 128>(count * 32, 128));
                    // Query and scale L1 slots are reused across fragments.
                    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                    TLOAD(query, qdata);
                    QueryScale query_scale(count * 32);
                    TASSIGN(query_scale, 83200);
                    using QS = TileShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                    using QD = BaseShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
                    GlobalTensor<S, QS, QD, Layout::MX_A_ND> qg(data<S>(args, 1) + begin * 128,
                    QS(count * 32, 4), QD(count * 32, 4));
                    TLOAD(query_scale, qg);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    TileLeft<F4, 96, 128> left0, left1;
                    TileLeftScale<S, 96, 4> scale0, scale1;
                    TASSIGN(left0, 0);
                    TASSIGN(left1, 16384);
                    TASSIGN(scale0, GetScaleAddr(left0.data()));
                    TASSIGN(scale1, GetScaleAddr(left1.data()));
                    // Both query halves must finish MMAD before their L0A slots are replaced.
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    TEXTRACT(left0, query, 0, 0);
                    TEXTRACT(left1, query, 96, 0);
                    TEXTRACT(scale0, query_scale, 0, 0);
                    TEXTRACT(scale1, query_scale, 96, 0);
                    int valid_panels = 0;
                    if constexpr (KeySlots == 4) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                        for (int panel = 0; panel < minimum(3, panel_count); ++panel) {
                            bool valid = prefetch_key<PageRows>(args, request, maximum, first_panel, panel, cache_pages);
                            if (valid) valid_panels |= 1 << panel;
                        }
                    }
#else
                    prepare_weights(args, begin, count);
                    int offset = get_subblockid() * 3;
                    int visible0 = offset < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset), width)) : 0;
                    int visible1 = offset + 1 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 1), width)) : 0;
                    int visible2 = offset + 2 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 2), width)) : 0;
#endif
                    for (int panel = 0; panel < panel_count; ++panel) {
#if defined(__DAV_CUBE__)
                        if constexpr (KeySlots == 4) {
                            int future = panel + 3;
                            if (future < panel_count) {
                                bool valid = prefetch_key<PageRows>(args, request, maximum, first_panel, future, cache_pages);
                                int bit = 1 << (future % 4);
                                valid_panels = (valid_panels & ~bit) | (valid ? bit : 0);
                            }
                        }
#endif
                        int logical = (first_panel + panel) * 256;
                        int page0 = -1, page1 = -1;
                        int compact_pages[4] = {-1, -1, -1, -1};
#if defined(__DAV_CUBE__)
                        if constexpr (KeySlots == 2) {
#endif
                        page0 = !gathered_rows && logical < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128) : -1;
                        page1 = !gathered_rows && logical + 128 < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128 + 1) : -1;
                        if (gathered_rows) {
                            page0 = logical < maximum ? (begin * gathered_rows + logical) / 128 : -1;
                            page1 = logical + 128 < maximum ? (begin * gathered_rows + logical) / 128 + 1 : -1;
                        }
                        if (!gathered_rows) {
                            if (page0 >= cache_pages) page0 = -1;
                            if (page1 >= cache_pages) page1 = -1;
                        }
#if defined(__DAV_CUBE__)
                        }
#endif
                        bool have_dot = page0 >= 0 || page1 >= 0;
#if defined(__DAV_VEC__)
                        if constexpr (PageRows == 64) {
                            have_dot = false;
                            for (int piece = 0; piece < 4; ++piece) {
                                int position = logical + piece * 64;
                                int page = position < maximum && position / 64 < table_width
                                    ? read(data<int32_t>(args, 5) + request * table_width + position / 64) : -1;
                                compact_pages[piece] = page >= 0 && page < cache_pages ? page : -1;
                                have_dot = have_dot || compact_pages[piece] >= 0;
                            }
                            page0 = page1 = have_dot ? 0 : -1;
                        }
#endif
#if defined(__DAV_CUBE__)
                        if constexpr (KeySlots == 4) {
                            have_dot = (valid_panels & (1 << (panel % 4))) != 0;
                            wait_flag(PIPE_MTE2, PIPE_MTE1, panel % 4 + 2);
                        }
                        if (have_dot) {
                            using Key = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 256, SLayout::ColMajor>;
                            using KeyPart = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 128, SLayout::ColMajor>;
                            using Scale = Tile<TileType::Mat, S, 4, 256, BLayout::ColMajor, 4, 256, SLayout::ColMajor, 32>;
                            Key key;
                            Scale key_scale;
                            TileRight<F4, 128, 256> right;
                            TileRightScale<S, 4, 256> right_scale;
                            TileAcc<float, 96, 256> dot0, dot1;
                            int key_slot = transferred % 2;
                            int key_event = key_slot + 2;
                            int l1_slot = KeySlots == 4 ? panel % KeySlots : key_slot;
                            TASSIGN(key, 65536 + l1_slot * 65536);
                            TASSIGN(key_scale, 81920 + l1_slot * 65536);
                            TASSIGN(right, key_slot * 16384);
                            TASSIGN(right_scale, GetScaleAddr(right.data()));
                            TASSIGN(dot0, 0);
                            TASSIGN(dot1, 98304);
                            if constexpr (KeySlots == 2) {
                                if (transferred >= 2) wait_flag(PIPE_MTE1, PIPE_MTE2, key_event);
                                for (int piece = 0; piece < 2; ++piece) {
                                    int page = piece == 0 ? page0 : page1;
                                    if (page < 0) page = page0 >= 0 ? page0 : page1;
                                    KeyPart part;
                                    KeyScale part_scale;
                                    TASSIGN(part, 65536 + key_slot * 65536 + piece * 4096);
                                    TASSIGN(part_scale, 81920 + key_slot * 65536 + piece * 512);
                                    GM<F4, 128, 128, 128, 128, Layout::DN> kg(data<F4>(args, 3) + static_cast<int64_t>(page) * 8192);
                                    GM<S, 4, 128, 4, 128, Layout::MX_B_DN> sg(data<S>(args, 4) + page * 512);
                                    TLOAD(part, kg);
                                    TLOAD(part_scale, sg);
                                }
                                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                            }
                            if (transferred >= 2) wait_flag(PIPE_M, PIPE_MTE1, key_event);
                            TMOV(right, key);
                            TMOV(right_scale, key_scale);
                            set_flag(PIPE_MTE1, PIPE_MTE2, KeySlots == 4 ? panel % KeySlots + 2 : key_event);
                            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            int slot = transferred % 2;
                            using Received = Tile<TileType::Vec, bfloat16_t, 96, 256, BLayout::RowMajor, 96, 256>;
                            using DotPart = Tile<TileType::Acc, float, 96, 256, BLayout::ColMajor, 96, 256, SLayout::RowMajor>;
                            if (transferred > 0) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            TMATMUL_MX(dot0, left0, scale0, right, right_scale);
                            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            if (transferred > 0) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
                            TMATMUL_MX(dot1, left1, scale1, right, right_scale);
                            set_flag(PIPE_M, PIPE_MTE1, key_event);
                            set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            if (transferred >= 2) wait_intra_block(PIPE_FIX, slot * 2 + 1);
                            {
                                Received received;
                                DotPart half;
                                TASSIGN(received, slot * 49152);
                                TASSIGN(half, 0);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec0>(received, half);
                            }
                            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            set_intra_block(PIPE_FIX, slot * 2);
                            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                            if (transferred >= 2) wait_intra_block(PIPE_FIX, slot * 2 + 17);
                            {
                                Received received;
                                DotPart half;
                                TASSIGN(received, slot * 49152);
                                TASSIGN(half, 98304);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec1>(received, half);
                            }
                            set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
                            set_intra_block(PIPE_FIX, slot * 2 + 16);
                            ++transferred;
                        } else if constexpr (KeySlots == 4) {
                            set_flag(PIPE_MTE1, PIPE_MTE2, panel % 4 + 2);
                        }
#else
                        if (have_dot) wait_intra_block(PIPE_V, (transferred % 2) * 2);
                        if (gathered_rows) {
                            Vec<int32_t, 1, 256> raw_valid;
                            Vec<uint16_t, 1, 256> pool_valid;
                            TASSIGN(raw_valid, 253440);
                            TASSIGN(pool_valid, 254464);
                            GM<int32_t, 1, 256, 1, 256> vg(data<int32_t>(args, 12) + begin * gathered_rows + logical);
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                            TLOAD(raw_valid, vg);
                            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                            TCVT(pool_valid, raw_valid, RoundMode::CAST_NONE);
                        }
                        if constexpr (PageRows == 64) {
                            Vec<int32_t, 1, 256> raw_valid;
                            Vec<uint16_t, 1, 256> page_valid;
                            TASSIGN(raw_valid, 253440);
                            TASSIGN(page_valid, 254464);
                            compact_validity((__ubuf__ uint32_t*)raw_valid.data(),
                                             compact_pages[0], compact_pages[1], compact_pages[2], compact_pages[3]);
                            pipe_barrier(PIPE_ALL);
                            TCVT(page_valid, raw_valid, RoundMode::CAST_NONE);
                        }
                        int n0 = visible0 - logical;
                        int n1 = visible1 - logical;
                        int n2 = visible2 - logical;
                        if (n0<0)n0 = 0;
                        if (n1<0)n1 = 0;
                        if (n2<0)n2 = 0;
                        int v00 = page0 >= 0?minimum(n0, 128):0, v01 = page1 >= 0&&n0>128?minimum(n0-128, 128):0;
                        int v10 = page0 >= 0?minimum(n1, 128):0, v11 = page1 >= 0&&n1>128?minimum(n1-128, 128):0;
                        int v20 = page0 >= 0?minimum(n2, 128):0, v21 = page1 >= 0&&n2>128?minimum(n2-128, 128):0;
                        int span0 = minimum(128, width-logical), span1 = minimum(128, width-logical-128);
                        if (span0<0)span0 = 0;
                        if (span1<0)span1 = 0;
                        score_cache<KeySlots == 4 ? 256 : 128>(ub<bfloat16_t>((transferred%2)*49152), ub<bfloat16_t>(257280+offset*64),
                        ub<uint16_t>(98304), ub<uint16_t>(251904), panel*256, span0, span1, v00, v01, v10, v11, v20, v21, have_dot, ub<uint16_t>(254464), (gathered_rows != 0 || PageRows == 64));
                        if (have_dot) {
                            set_intra_block(PIPE_V, (transferred%2)*2+1);
                            ++transferred;
                        }
#endif
                    }
#if defined(__DAV_CUBE__)
                    if constexpr (KeySlots == 4) {
                        for (int slot = 0; slot < minimum(4, panel_count); ++slot) {
                            wait_flag(PIPE_MTE1, PIPE_MTE2, slot + 2);
                        }
                    }
                    wait_intra_block(PIPE_FIX, 6);
                    wait_intra_block(PIPE_FIX, 22);
#else
                    int cached_width = minimum(panel_count * 256, width - first_panel * 256);
                    local_topk(args, begin, count, first_panel * 256, cached_width, shard);
                    if (args[20]) store_scores(args, begin, count, first_panel * 256, cached_width);
                    set_intra_block(PIPE_MTE3, 6);
#endif
                    cursor = finish;
                }
            }
#if defined(__DAV_CUBE__)
            if (transferred > 0) {
                for (int slot = 0; slot < minimum(2, transferred); ++slot) {
                    if constexpr (KeySlots == 2) wait_flag(PIPE_MTE1, PIPE_MTE2, slot + 2);
                    wait_flag(PIPE_M, PIPE_MTE1, slot + 2);
                }
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
            }
            for (int slot = 0; slot < minimum(2, transferred); ++slot) {
                wait_intra_block(PIPE_FIX, slot * 2 + 1);
                wait_intra_block(PIPE_FIX, slot * 2 + 17);
            }
#endif
        }
    }
    // namespace fused
}
// namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    set_atomic_none();
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
#else
    set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
#endif
    if (args[19] < 0) {
        fused::run_pool(args);
        SYNCALL<SyncCoreType::AIVOnly>();
#if defined(__DAV_VEC__)
        fused::merge_pool(args);
#endif
        return;
    }
    if (args[19] == 0) {
        if (reinterpret_cast<__gm__ Tensor*>(args[3])->shapes[1] == 64) fused::run<4, 64>(args);
        else fused::run<4, 128>(args);
    }
    else fused::run<2>(args);
    SYNCALL<SyncCoreType::AIVOnly>();
#if defined(__DAV_VEC__)
    fused::merge(args);
#endif
}
