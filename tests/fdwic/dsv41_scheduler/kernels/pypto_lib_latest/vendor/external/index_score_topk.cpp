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
using QueryPipe = TPipe<4, Direction::DIR_V2C, 16384, 2, 2, true>;

template <typename T, int R, int C, int BR, int BC, Layout L = Layout::ND>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using QueryMat = Tile<TileType::Mat, F4, 192, 128, BLayout::ColMajor, 192, 128, SLayout::RowMajor>;
using QueryVec = Tile<TileType::Vec, F4, 192, 128, BLayout::ColMajor, 192, 128, SLayout::RowMajor>;
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
__aicore__ inline void prepare_query(__gm__ int64_t* args, int begin, int count, QueryPipe& pipe) {
    if (get_subblockid() == 0) {
        Tile<TileType::Vec, uint8_t, 192, 64, BLayout::RowMajor, DYNAMIC, 64> input(count * 32);
        Vec<F4, 192, 128> packed;
        QueryVec nz;
        TASSIGN(input, 131072);
        TASSIGN(packed, 131072);
        TASSIGN(nz, 143360);
        GM<uint8_t, DYNAMIC, 64, DYNAMIC, 64> gm(data<uint8_t>(args, 0) + begin * 2048,
            TileShape2D<uint8_t, DYNAMIC, 64>(count * 32, 64),
            BaseShape2D<uint8_t, DYNAMIC, 64>(count * 32, 64));
        TLOAD(input, gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMOV(nz, packed);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TPUSH<QueryPipe, QueryVec, TileSplitAxis::TILE_NO_SPLIT>(pipe, nz);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

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
    // Complete query layout conversion before loading weights.
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
        __aicore__ inline void write_cached_row(RegTensor<bfloat16_t>& a, RegTensor<bfloat16_t>& b,
        __ubuf__ uint16_t* dst, __ubuf__ uint16_t* hist, int valid0, int valid1,
        int span0, int span1, bool reset, __ubuf__ uint16_t* pool_valid, bool gathered) {
            vector_u16 h0, h1, pos, negkey, sign, key, high, invalid, lane_valid;
            vector_u8 bytes;
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
            vshrs(high, key, 8, full, MODE_ZEROING);
            vpack(bytes, high, LOWER);
            remaining = span0;
            auto hist_mask = CreatePredicate<uint8_t>(remaining);
            chistv2(h0, bytes, hist_mask, Bin_N0);
            chistv2(h1, bytes, hist_mask, Bin_N1);
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
            vshrs(high, key, 8, full, MODE_ZEROING);
            vpack(bytes, high, LOWER);
            remaining = span1;
            hist_mask = CreatePredicate<uint8_t>(remaining);
            chistv2(h0, bytes, hist_mask, Bin_N0);
            chistv2(h1, bytes, hist_mask, Bin_N1);
            vsts(h0, hist, 0, NORM_B16, full);
            vsts(h1, hist, 128, NORM_B16, full);
        }
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
                write_cached_row(s00, s01, cache+column, hist, v00, v01, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row(s10, s11, cache+25600+column, hist+256, v10, v11, span0, span1, column == 0, pool_valid, gathered);
                write_cached_row(s20, s21, cache+51200+column, hist+512, v20, v21, span0, span1, column == 0, pool_valid, gathered);
            }
        }
        template <int SHIFT>
        __tf__ __aicore__ void packed_histogram(__ubuf__ uint16_t* keys, __ubuf__ uint16_t* hist, int width, uint32_t prefix) {
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
                    }
                    else {
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
                        }
                        else {
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
        __aicore__ inline void select_bin(__ubuf__ uint16_t* hist, int shift, uint32_t& key, int& rank, int& count) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            int lo = 0, hi = 255;
            for (int step = 0;step<8;++step) {
                int mid = (lo+hi)/2;
                if (hist[mid] >= rank)hi = mid;
                else lo = mid+1;
            }
            int previous = lo?hist[lo-1]:0;
            count = hist[lo]-previous;
            rank -= previous;
            key |= uint32_t(lo)<<shift;
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        }
        __tf__ __aicore__ void packed_equal(__ubuf__ uint16_t* out, __ubuf__ uint16_t* in, int width, uint32_t cutoff) {
            __VEC_SCOPE__ {
                vector_u16 key, index, squeezed;
                vector_bool equal;
                vector_align align;
                sprclr(SPR_AR);
                uint32_t remaining = width;
                for (uint16_t c = 0;c<(width+127)/128;++c) {
                    auto mask = CreatePredicate<uint16_t>(remaining);
                    vlds(key, in, c*128, NORM);
                    vci((vector_s16&)index, int16_t(c*128), INC_ORDER);
                    vcmps_eq(equal, key, uint16_t(cutoff), mask);
                    vsqz(squeezed, index, equal, MODE_STORED);
                    vstur(align, squeezed, out, POST_UPDATE);
                }
                vstar(align, out);
            }
        }
        __aicore__ inline void packed_threshold(__ubuf__ uint16_t* keys, int width, int scratch, int hist_base,
        bool cached, int cached_hist, uint32_t& cutoff, int& boundary) {
            cutoff = 0x007f;
            boundary = -1;
            if (width>512) {
                auto hist = ub<uint16_t>(hist_base);
                int rank = width-511, count = 0;
                cutoff = 0;
                if (!cached)packed_histogram<8>(keys, hist, width, 0);
                select_bin(cached?ub<uint16_t>(cached_hist):hist, 8, cutoff, rank, count);
                packed_histogram<0>(keys, hist, width, cutoff);
                select_bin(hist, 0, cutoff, rank, count);
                packed_equal(ub<uint16_t>(scratch), keys, width, cutoff);
                set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                boundary = (ub<uint16_t>(scratch))[count-rank];
                set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            }
        }
        template <bool MERGE>
        __tf__ __aicore__ void packed_select(__ubuf__ uint16_t* outk, __ubuf__ uint32_t* outi,
        __ubuf__ uint16_t* src, __ubuf__ uint32_t* ids, __ubuf__ uint32_t* table,
        int width, uint32_t cutoff, int boundary, int logical_begin, bool gathered, bool map_output = true) {
            auto output_indices = outi;
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
                    if constexpr(MERGE) {
                        vlds(logical, ids, c*64, NORM);
                        if (map_output && gathered) {
                            vgather2(page, table, logical, keep);
                            logical = page;
                        } else if (map_output) {
                            vshrs(tmp, logical, 7, mask, MODE_ZEROING);
                            vgather2(page, table, tmp, keep);
                            vshls(page, page, 7, mask, MODE_ZEROING);
                            vbr(tmp, 127u);
                            vand(tmp, logical, tmp, mask, MODE_ZEROING);
                            vadd(logical, page, tmp, mask, MODE_ZEROING);
                        }
                    }
                    else vadds(logical, lane, uint32_t(logical_begin), mask, MODE_ZEROING);
                    vsqz(squeezed, logical, keep, MODE_STORED);
                    vstur(align, squeezed, outi, POST_UPDATE);
                }
                vstar(align, outi);
                if constexpr(!MERGE) {
                    mem_bar(VST_VLD);
                    for (uint16_t c = 0;c<8;++c) {
                        vlds(logical, output_indices, c*64, NORM);
                        vcmps_ne(valid, logical, 0xffffffffu, full);
                        vadds(tmp, logical, uint32_t(-logical_begin), full, MODE_ZEROING);
                        vbr(shift, 1u);
                        vand(shift, tmp, shift, full, MODE_ZEROING);
                        vshls(shift, shift, 4, full, MODE_ZEROING);
                        vshrs(tmp, tmp, 1, full, MODE_ZEROING);
                        vbr(key, 0u);
                        vgather2(key, (__ubuf__ uint32_t*)src, tmp, valid);
                        vshr(key, key, (vector_s32&)shift, full, MODE_ZEROING);
                        vpack(key16, key, LOWER);
                        ppack(packed_mask, full, LOWER);
                        vsts(key16, outk, c*64, NORM_B16, packed_mask);
                    }
                }
            }
        }
        __aicore__ inline void local_topk(__gm__ int64_t* args, int begin, int count, int logical, int width, int shard) {
            int shards = args[21];
            for (int row = 0;row<3;++row) {
                int group_offset = get_subblockid()*3+row;
                if (group_offset >= count)continue;
                auto keys = ub<uint16_t>(98304+row*51200);
                uint32_t cutoff;
                int boundary;
                packed_threshold(keys, width, 0, 54272, true, 251904+row*512, cutoff, boundary);
                CandidateKeys selected_keys;
                CandidateIndices selected_indices;
                TASSIGN(selected_keys, 51200);
                TASSIGN(selected_indices, 52224);
                packed_select<false>(selected_keys.data(), selected_indices.data(), keys, (__ubuf__ uint32_t*)0,
                (__ubuf__ uint32_t*)0, width, cutoff, boundary, logical, false);
                int index = ((begin+group_offset)*shards+shard)*512;
                GM<uint16_t, 1, 512, 1, 512> kg(data<uint16_t>(args, 10)+index);
                GM<uint32_t, 1, 512, 1, 512> ig(data<uint32_t>(args, 11)+index);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(kg, selected_keys);
                TSTORE(ig, selected_indices);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }
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
                for (int column = 0; column < cached_width; column += 256) {
                    unpack_scores(rounded.data(), ub<uint16_t>(98304 + row * 51200) + column);
                    TCVT(values, rounded, RoundMode::CAST_RINT);
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
        __tf__ __aicore__ void short_page(__ubuf__ uint32_t* output, int base, int count) {
            __VEC_SCOPE__ {
                vector_u32 lane, physical, invalid, result;
                vector_bool valid;
                auto full = pset_b32(PAT_ALL);
                vbr(invalid, 0xffffffffu);
                for (uint16_t chunk = 0; chunk < 2; ++chunk) {
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
            int request = read(data<int32_t>(args, 6) + token);
            for (int block = 0; block < 4; ++block) {
                int count = minimum(128, length - block * 128);
                int page = -1;
                if (count > 0 && block < table_width) {
                    page = read(data<int32_t>(args, 5) + request * table_width + block);
                }
                if (page < 0 || page >= cache_pages) {
                    page = 0;
                    count = 0;
                }
                short_page(indices.data() + block * 128, page * 128, count);
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
                    if (length <= 512) {
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
                        int request = read(data<int32_t>(args, 6) + token), page_count = (width + 127) / 128;
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
                    packed_threshold(ub<uint16_t>(0), candidates, 32768, 131072, false, 0, cutoff, boundary);
                    packed_select<true>(selected_keys.data(), selected_indices.data(), ub<uint16_t>(0),
                        ub<uint32_t>(65536), mapping, candidates, cutoff, boundary, 0, gathered_rows != 0, final);
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
        __aicore__ inline void run(__gm__ int64_t* args) {
            QueryPipe query_pipe(nullptr, 0, 0);
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
                    QueryMat query;
                    QueryScale query_scale(count * 32);
                    TASSIGN(query_scale, 83200);
                    TPOP<QueryPipe, QueryMat, TileSplitAxis::TILE_NO_SPLIT>(query_pipe, query);
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
                    TEXTRACT(left0, query, 0, 0);
                    TEXTRACT(left1, query, 96, 0);
                    TEXTRACT(scale0, query_scale, 0, 0);
                    TEXTRACT(scale1, query_scale, 96, 0);
#else
                    prepare_query(args, begin, count, query_pipe);
                    int offset = get_subblockid() * 3;
                    int visible0 = offset < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset), width)) : 0;
                    int visible1 = offset + 1 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 1), width)) : 0;
                    int visible2 = offset + 2 < count ? (gathered_rows ? maximum : minimum(read(data<int32_t>(args, 7) + begin + offset + 2), width)) : 0;
#endif
                    for (int panel = 0; panel < panel_count; ++panel) {
                        int logical = (first_panel + panel) * 256;
                        int page0 = !gathered_rows && logical < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128) : -1;
                        int page1 = !gathered_rows && logical + 128 < maximum ? read(data<int32_t>(args, 5) + request * table_width + logical / 128 + 1) : -1;
                        if (gathered_rows) {
                            page0 = logical < maximum ? (begin * gathered_rows + logical) / 128 : -1;
                            page1 = logical + 128 < maximum ? (begin * gathered_rows + logical) / 128 + 1 : -1;
                        }
                        if (!gathered_rows) {
                            if (page0 >= cache_pages) page0 = -1;
                            if (page1 >= cache_pages) page1 = -1;
                        }
                        bool have_dot = page0 >= 0 || page1 >= 0;
#if defined(__DAV_CUBE__)
                        if (have_dot) {
                            using Key = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 256, SLayout::ColMajor>;
                            using KeyPart = Tile<TileType::Mat, F4, 128, 256, BLayout::RowMajor, 128, 128, SLayout::ColMajor>;
                            using Scale = Tile<TileType::Mat, S, 4, 256, BLayout::ColMajor, 4, 256, SLayout::ColMajor, 32>;
                            Key key;
                            Scale key_scale;
                            TileRight<F4, 128, 256> right;
                            TileRightScale<S, 4, 256> right_scale;
                            TileAcc<float, 96, 256> dot0, dot1;
                            TASSIGN(key, 65536);
                            TASSIGN(key_scale, 81920);
                            TASSIGN(right, 0);
                            TASSIGN(right_scale, GetScaleAddr(right.data()));
                            TASSIGN(dot0, 0);
                            TASSIGN(dot1, 98304);
                            if (transferred > 0) wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
                            for (int piece = 0; piece < 2; ++piece) {
                                int page = piece == 0 ? page0 : page1;
                                if (page < 0) page = page0 >= 0 ? page0 : page1;
                                KeyPart part;
                                KeyScale part_scale;
                                TASSIGN(part, 65536 + piece * 4096);
                                TASSIGN(part_scale, 81920 + piece * 512);
                                GM<F4, 128, 128, 128, 128, Layout::DN> kg(data<F4>(args, 3) + static_cast<int64_t>(page) * 8192);
                                GM<S, 4, 128, 4, 128, Layout::MX_B_DN> sg(data<S>(args, 4) + page * 512);
                                TLOAD(part, kg);
                                TLOAD(part_scale, sg);
                            }
                            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                            if (transferred > 0) wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                            TMOV(right, key);
                            TMOV(right_scale, key_scale);
                            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
                            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                            if (transferred > 0) wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            TMATMUL_MX(dot0, left0, scale0, right, right_scale);
                            TMATMUL_MX(dot1, left1, scale1, right, right_scale);
                            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                            int slot = transferred % 2;
                            if (transferred >= 2) {
                                wait_intra_block(PIPE_FIX, slot * 2 + 1);
                                wait_intra_block(PIPE_FIX, slot * 2 + 17);
                            }
                            using Received = Tile<TileType::Vec, bfloat16_t, 96, 256, BLayout::RowMajor, 96, 128>;
                            using DotPart = Tile<TileType::Acc, float, 96, 256, BLayout::ColMajor, 96, 128, SLayout::RowMajor>;
                            for (int piece = 0; piece < 2; ++piece) {
                                Received received;
                                DotPart first_half, second_half;
                                TASSIGN(received, slot * 49152 + piece * 256);
                                TASSIGN(first_half, piece * 49152);
                                TASSIGN(second_half, 98304 + piece * 49152);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec0>(received, first_half);
                                TMOV<Received, DotPart, AccToVecMode::SingleModeVec1>(received, second_half);
                            }
                            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                            set_intra_block(PIPE_FIX, slot * 2);
                            set_intra_block(PIPE_FIX, slot * 2 + 16);
                            ++transferred;
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
                        score_cache(ub<bfloat16_t>((transferred%2)*49152), ub<bfloat16_t>(257280+offset*64),
                        ub<uint16_t>(98304), ub<uint16_t>(251904), panel*256, span0, span1, v00, v01, v10, v11, v20, v21, have_dot, ub<uint16_t>(254464), gathered_rows != 0);
                        if (have_dot) {
                            set_intra_block(PIPE_V, (transferred%2)*2+1);
                            ++transferred;
                        }
#endif
                    }
#if defined(__DAV_CUBE__)
                    TFREE<QueryPipe, TileSplitAxis::TILE_NO_SPLIT>(query_pipe);
                    wait_intra_block(PIPE_S, 6);
                    wait_intra_block(PIPE_S, 22);
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
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
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
    fused::run(args);
    SYNCALL<SyncCoreType::Mix>();
#if defined(__DAV_VEC__)
    fused::merge(args);
#endif
}
