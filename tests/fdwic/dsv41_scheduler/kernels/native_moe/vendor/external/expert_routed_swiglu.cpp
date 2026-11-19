/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "intrinsic.h"

#if defined(__CPU_SIM)
#include <cmath>
#include <cstring>
#endif

using namespace pto;

namespace {

constexpr int ROW_TILE = 16;
constexpr int COL_TILE = 768;
constexpr int SCALE_COLS = COL_TILE / 32;
constexpr int SCALE_STRIDE = 32;
constexpr int INPUT_BYTES = ROW_TILE * COL_TILE * sizeof(bfloat16_t);

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

AICORE inline void activateQuantize(__ubuf__ bfloat16_t* gate, __ubuf__ bfloat16_t* up,
                                   __ubuf__ float* weights, __ubuf__ uint8_t* out,
                                   __ubuf__ uint8_t* codes, uint16_t validRows, uint16_t validCols)
{
#if defined(__CPU_SIM)
    using Input = Tile<TileType::Vec, bfloat16_t, ROW_TILE, COL_TILE>;
    using Weight = Tile<TileType::Vec, float, 1, ROW_TILE>;
    using Output = Tile<TileType::Vec, uint8_t, ROW_TILE, COL_TILE>;
    using Codes = Tile<TileType::Vec, uint8_t, 1, ROW_TILE * SCALE_COLS>;
    Input gateTile, upTile;
    Weight weightTile;
    Output outputTile;
    Codes codeTile;
    TASSIGN(gateTile, reinterpret_cast<uintptr_t>(gate));
    TASSIGN(upTile, reinterpret_cast<uintptr_t>(up));
    TASSIGN(weightTile, reinterpret_cast<uintptr_t>(weights));
    TASSIGN(outputTile, reinterpret_cast<uintptr_t>(out));
    TASSIGN(codeTile, reinterpret_cast<uintptr_t>(codes));
    gate = gateTile.data();
    up = upTile.data();
    weights = weightTile.data();
    out = outputTile.data();
    codes = codeTile.data();
    for (uint16_t row = 0; row < validRows; ++row) {
        for (uint16_t group = 0; group < validCols / 32; ++group) {
            float values[32] = {};
            float maximum = 1e-4f;
            for (uint16_t col = 0; col < 32; ++col) {
                if (row < validRows) {
                    const uint16_t offset = row * COL_TILE + group * 32 + col;
                    const float g = std::fmin(static_cast<float>(gate[offset]), 10.0f);
                    const float u = std::fmax(std::fmin(static_cast<float>(up[offset]), 10.0f), -10.0f);
                    values[col] = (g / (1.0f + std::exp(-g))) * u * weights[row];
                }
                maximum = std::fmax(maximum, std::fabs(values[col]));
            }
            maximum *= 1.0f / 448.0f;
            uint32_t bits;
            std::memcpy(&bits, &maximum, sizeof(bits));
            const uint32_t exponent = (bits + 8388607) >> 23;
            codes[row * SCALE_STRIDE + group] = static_cast<uint8_t>(exponent);
            bits = (254 - exponent) << 23;
            float reciprocal;
            std::memcpy(&reciprocal, &bits, sizeof(reciprocal));
            for (uint16_t col = 0; col < 32; ++col) {
                const float normalized = std::fmax(std::fmin(values[col] * reciprocal, 448.0f), -448.0f);
                const float8_e4m3_t value(normalized, RoundMode::CAST_RINT, SaturationMode::ON);
                out[row * COL_TILE + group * 32 + col] = value.RawData();
            }
        }
    }
#else
    __VEC_SCOPE__ {
        constexpr auto halfStore = static_cast<::DistVST>(pto::DistVST::DIST_ONEPT_B16);
        vector_bool all = pset_b32(PAT_ALL);
        vector_bool all16 = pset_b16(PAT_ALL);
        vector_bool lower = pset_b32(PAT_VL32);
        vector_bool upper;
        pnot(upper, lower, all);
        vector_bool four = pset_b32(PAT_VL4);
        vector_bool one = pset_b32(PAT_VL1);
        vector_bool one16 = pset_b16(PAT_VL1);
        vector_s32 maxReciprocalExponent;
        vbr(maxReciprocalExponent, 254);
        for (uint16_t row = 0; row < validRows; ++row) {
            vector_f32 weight;
            vlds(weight, weights, row, BRC_B32);
            for (uint16_t col = 0; col < validCols; col += 128) {
                vector_f32 g0, g1, u0, u1;
                RegTensor<bfloat16_t> g16, u16;
                vlds(g16, gate, row * COL_TILE + col, NORM);
                vlds(u16, up, row * COL_TILE + col, NORM);
                vcvt(g0, g16, all16, PART_EVEN);
                vcvt(g1, g16, all16, PART_ODD);
                vcvt(u0, u16, all16, PART_EVEN);
                vcvt(u1, u16, all16, PART_ODD);
                vmins(g0, g0, 10.0f, all);
                vmins(g1, g1, 10.0f, all);
                vmins(u0, u0, 10.0f, all);
                vmins(u1, u1, 10.0f, all);
                vmaxs(u0, u0, -10.0f, all);
                vmaxs(u1, u1, -10.0f, all);
                vector_f32 sigmoid0, sigmoid1, hiddenLow, hiddenHigh;
                vmuls(sigmoid0, g0, -1.0f, all);
                vmuls(sigmoid1, g1, -1.0f, all);
                vexp(sigmoid0, sigmoid0, all);
                vexp(sigmoid1, sigmoid1, all);
                vadds(sigmoid0, sigmoid0, 1.0f, all);
                vadds(sigmoid1, sigmoid1, 1.0f, all);
                vdiv(hiddenLow, g0, sigmoid0, all);
                vdiv(hiddenHigh, g1, sigmoid1, all);
                vmul(hiddenLow, hiddenLow, u0, all);
                vmul(hiddenHigh, hiddenHigh, u1, all);
                vmul(hiddenLow, hiddenLow, weight, all);
                vmul(hiddenHigh, hiddenHigh, weight, all);
                vector_f32 h0 = hiddenLow;
                vector_f32 h1 = hiddenHigh;
                vector_f32 magnitude0, magnitude1, maximum, groups, even, odd;
                vabs(magnitude0, h0, all);
                vabs(magnitude1, h1, all);
                vmax(maximum, magnitude0, magnitude1, all);
                vcgmax(groups, maximum, all);
                vdintlv(even, odd, groups, groups);
                vmax(maximum, even, odd, four);
                vmaxs(maximum, maximum, 1e-4f, four);
                vmuls(maximum, maximum, 1.0f / 448.0f, four);
                vector_s32 exponent, reciprocal, expanded, unused;
                vadds(exponent, (vector_s32&)maximum, 8388607, four);
                vshrs(exponent, exponent, 23, four);
                vector_u16 packed16;
                vector_u8 packed8;
                vpack(packed16, (vector_u32&)exponent, LOWER);
                vpack(packed8, packed16, LOWER);
                vsts((vector_u32&)packed8, (__ubuf__ uint32_t*)codes,
                     (row * SCALE_STRIDE + col / 32) / 4, ONEPT_B32, one);
                vsub(reciprocal, maxReciprocalExponent, exponent, four);
                vshls(reciprocal, reciprocal, 23, four);
                vintlv(expanded, unused, reciprocal, reciprocal);
                vintlv(reciprocal, unused, expanded, expanded);
                vintlv(expanded, unused, reciprocal, reciprocal);
                vintlv(reciprocal, unused, expanded, expanded);
                vmul(h0, h0, (vector_f32&)reciprocal, all);
                vmul(h1, h1, (vector_f32&)reciprocal, all);
                vector_f32 low, high;
                vintlv(low, high, h0, h1);
                vector_f8e4m3 payload0, payload1;
                vcvt(payload0, low, all, ROUND_R, RS_ENABLE, PART_P0);
                vcvt(payload1, high, all, ROUND_R, RS_ENABLE, PART_P0);
                vsts((vector_u8&)payload0, out, row * COL_TILE + col, PK4_B32, all);
                vsts((vector_u8&)payload1, out, row * COL_TILE + col + 64, PK4_B32, all);
            }
        }

    }
#endif
}

}  // namespace


AICORE inline void reduceCounts(__ubuf__ int32_t* input, uint16_t count)
{
#if defined(__CPU_SIM)
    int32_t total = 0;
    for (uint16_t e = 0; e < count; ++e) total += input[e];
    input[0] = total;
#else
    __VEC_SCOPE__ {
        vector_s32 total;
        vbr(total, 0);
        vector_bool one = pset_b32(PAT_VL1);
        uint32_t remaining = count;
        for (uint16_t offset = 0; offset < count; offset += 64) {
            vector_bool mask = plt_b32(remaining, POST_UPDATE);
            vector_s32 values, sum;
            vlds(values, input, offset, NORM);
            vcadd(sum, values, mask, MODE_ZEROING);
            vadd(total, total, sum, one);
        }
        vsts(total, input, 0, ONEPT_B32, one);
    }
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    auto* pair = tensorData<bfloat16_t>(args, 0);
    auto* weights = tensorData<float>(args, 1);
    auto* counts = tensorData<int32_t>(args, 2);
    auto* out = tensorData<uint8_t>(args, 3);
    auto* scales = tensorData<uint8_t>(args, 4);
    const int64_t experts = args[5];
    const int64_t width = args[7];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    using Counts = Tile<TileType::Vec, int32_t, 1, 256, BLayout::RowMajor, 1, DYNAMIC>;
    using GlobalCounts = GlobalTensor<int32_t, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
    Counts localCounts;
    TASSIGN(localCounts, 65536);
    int64_t rows = 0;
    for (int64_t first = 0; first < experts; first += 256) {
        const int64_t entries = experts - first < 256 ? experts - first : 256;
        localCounts.SetValidCol(entries);
        GlobalCounts gc(counts + first, GlobalCounts::Shape(1, 1, 1, 1, entries));
        TLOAD(localCounts, gc);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        reduceCounts(localCounts.data(), entries);
        set_flag(PIPE_V, PIPE_S, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
        rows += localCounts.data()[0];
    }
    using Input = Tile<TileType::Vec, bfloat16_t, ROW_TILE, COL_TILE, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Weight = Tile<TileType::Vec, float, 1, ROW_TILE, BLayout::RowMajor, 1, DYNAMIC>;
    using Output = Tile<TileType::Vec, uint8_t, ROW_TILE, COL_TILE, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Codes = Tile<TileType::Vec, uint8_t, ROW_TILE, SCALE_STRIDE, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    Input gateTile, upTile;
    Weight weightTile;
    Output outputTile;
    Codes codeTile;
    TASSIGN(gateTile, 0);
    TASSIGN(upTile, INPUT_BYTES);
    TASSIGN(weightTile, 2 * INPUT_BYTES);
    TASSIGN(outputTile, 2 * INPUT_BYTES + 64);
    TASSIGN(codeTile, 2 * INPUT_BYTES + 64 + ROW_TILE * COL_TILE);
    using GI = GlobalTensor<bfloat16_t, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    using GW = GlobalTensor<float, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
    using GO = GlobalTensor<uint8_t, Shape<1, 1, 1, DYNAMIC, DYNAMIC>, pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    const int64_t colTiles = (width + COL_TILE - 1) / COL_TILE;
    const int64_t workCount = ((rows + ROW_TILE - 1) / ROW_TILE) * colTiles;
    for (int64_t work = block; work < workCount; work += blocks) {
        const int64_t row = work / colTiles * ROW_TILE;
        const int64_t col = work % colTiles * COL_TILE;
        const uint16_t validRows = rows - row < ROW_TILE ? rows - row : ROW_TILE;
        const uint16_t validCols = width - col < COL_TILE ? width - col : COL_TILE;
        gateTile.SetValidRow(validRows);
        gateTile.SetValidCol(validCols);
        upTile.SetValidRow(validRows);
        upTile.SetValidCol(validCols);
        weightTile.SetValidCol(validRows);
        outputTile.SetValidRow(validRows);
        outputTile.SetValidCol(validCols);
        codeTile.SetValidRow(validRows);
        codeTile.SetValidCol(validCols / 32);
        GI gg(pair + row * width * 2 + col, GI::Shape(1, 1, 1, validRows, validCols), GI::Stride(1, 1, 1, width * 2, 1));
        GI gu(pair + row * width * 2 + width + col, GI::Shape(1, 1, 1, validRows, validCols), GI::Stride(1, 1, 1, width * 2, 1));
        GW gw(weights + row, GW::Shape(1, 1, 1, 1, validRows));
        TLOAD(gateTile, gg);
        TLOAD(upTile, gu);
        TLOAD(weightTile, gw);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        activateQuantize(gateTile.data(), upTile.data(), weightTile.data(), outputTile.data(), codeTile.data(), validRows, validCols);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GO gy(out + row * width + col, GO::Shape(1, 1, 1, validRows, validCols), GO::Stride(1, 1, 1, width, 1));
        GO gs(scales + row * (width / 32) + col / 32, GO::Shape(1, 1, 1, validRows, validCols / 32), GO::Stride(1, 1, 1, width / 32, 1));
        TSTORE(gy, outputTile);
        TSTORE(gs, codeTile);
        pipe_barrier(PIPE_ALL);
    }
}
