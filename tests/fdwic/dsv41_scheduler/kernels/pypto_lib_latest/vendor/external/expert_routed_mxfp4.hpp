/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "tensor.h"
#include "intrinsic.h"

namespace grouped_mxfp4 {

using namespace pto;

constexpr uint32_t M_TILE = 64;
constexpr uint32_t N_TILE = 256;
constexpr uint32_t K_TILE = 256;
constexpr uint32_t SCALE_K_TILE = K_TILE / 32;
constexpr uint32_t L0_K_TILE = 128;
constexpr uint32_t A_STAGE_K = 1024;
constexpr uint32_t SCALE_STAGE_K = 4096;
constexpr uint32_t SCALE_STAGE_Q = SCALE_STAGE_K / 32;
constexpr uint32_t L0_SCALE_K_TILE = L0_K_TILE / 32;
constexpr uint32_t VEC_K_TILE = K_TILE / 2;
constexpr uint32_t PACKED_COLS = N_TILE / 2;
constexpr uint32_t UB_SOURCE_BYTES = VEC_K_TILE * PACKED_COLS;
constexpr uint32_t UB_NZ_BYTES = ((VEC_K_TILE + 15) / 16 * 16 + 1) * N_TILE;
constexpr uint32_t PREFETCH_SLOTS = 4;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t L1_BASE = 2 * M_TILE * A_STAGE_K + 2 * (M_TILE + N_TILE) * SCALE_STAGE_Q;
constexpr uint32_t L1_SLOT_BYTES = K_TILE * N_TILE;
constexpr uint32_t L1_SLOT_COUNT = 2;
constexpr uint8_t VEC_CORE_ID_OFFSET = 16;

static_assert(UB_SOURCE_BYTES + UB_NZ_BYTES <= 64 * 1024);
static_assert(M_TILE * K_TILE <= L0_PINGPONG_BYTES);
static_assert(L0_K_TILE * N_TILE <= L0_PINGPONG_BYTES);

static_assert(L1_BASE + L1_SLOT_COUNT * L1_SLOT_BYTES <= 512 * 1024);

#ifdef __DAV_CUBE__
AICORE inline void releaseSlot(uint32_t slot)
{
    const uint8_t freeFlag = static_cast<uint8_t>(1 + 2 * slot);
    set_intra_block(PIPE_MTE1, freeFlag);
    set_intra_block(PIPE_MTE1, freeFlag + VEC_CORE_ID_OFFSET);
}

AICORE inline void waitSlot(uint32_t slot)
{
    const uint8_t readyFlag = static_cast<uint8_t>(2 * slot);
    wait_intra_block(PIPE_MTE1, readyFlag);
    wait_intra_block(PIPE_MTE1, readyFlag + VEC_CORE_ID_OFFSET);
}
#endif

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

struct Chunk {
    int64_t expert;
    int64_t stageRow;
    int64_t validRows;
};

AICORE inline int64_t acquireCount(__gm__ int32_t* counts, int64_t expert)
{
    auto* count = counts + expert;
    dcci(count, SINGLE_CACHE_LINE);
    return *count;
}

#ifdef __DAV_VEC__
AICORE inline void produceWeight(__gm__ uint8_t* weight, int64_t k, int64_t n,
                                int64_t expert, int64_t n0, int64_t kBegin = 0, int64_t kEnd = 0)
{
    constexpr uint32_t PACKED_ROW_BYTES = 256 / 2;
    constexpr uint32_t PACKED_TILE_BYTES = 256 * 256 / 2;
    static_assert(PREFETCH_SLOTS * UB_SOURCE_BYTES + 2 * UB_NZ_BYTES <= 192 * 1024);
    using Packed = Tile<TileType::Vec, float4_e2m1x2_t, VEC_K_TILE, N_TILE>;
    using Raw = Tile<TileType::Vec, uint8_t, VEC_K_TILE, PACKED_COLS>;
    using NZ = Tile<TileType::Vec, float8_e4m3_t, VEC_K_TILE, N_TILE, BLayout::ColMajor,
                    VEC_K_TILE, N_TILE, SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
    static_assert(NZ::StorageBytes == UB_NZ_BYTES);
    using GlobalPacked = GlobalTensor<uint8_t, Shape<1, 1, 1, VEC_K_TILE, PACKED_COLS>,
                                      pto::Stride<1, 1, 1, PACKED_ROW_BYTES, 1>>;
    using MatSlot = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                         K_TILE, N_TILE, SLayout::RowMajor, 512>;
    MatSlot matSlot;
    Packed packed[PREFETCH_SLOTS];
    Raw raw[PREFETCH_SLOTS];
    NZ nz[2];
    for (uint32_t slot = 0; slot < PREFETCH_SLOTS; ++slot) {
        TASSIGN(packed[slot], slot * UB_SOURCE_BYTES);
        TASSIGN(raw[slot], slot * UB_SOURCE_BYTES);
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        TASSIGN(nz[slot], PREFETCH_SLOTS * UB_SOURCE_BYTES + slot * UB_NZ_BYTES);
    }
    const int64_t lane = get_subblockid();
    const int64_t expertBase = expert * k * n / 2;
    const int64_t panelBase = (n0 / 256) * (k / K_TILE) * PACKED_TILE_BYTES;
    const int64_t sliceBase = (kBegin / K_TILE) * PACKED_TILE_BYTES;
    const int64_t laneBase = lane * VEC_K_TILE * PACKED_ROW_BYTES + (n0 % 256) / 2 + sliceBase;
    const int64_t iterations = ((kEnd == 0 ? k : kEnd) - kBegin) / K_TILE;
    // Prefetch four packed panels into independent source slots.
    for (int64_t first = 0; first < PREFETCH_SLOTS && first < iterations; ++first) {
        GlobalPacked globalWeight(weight + expertBase + panelBase + first * PACKED_TILE_BYTES + laneBase);
        TLOAD(raw[first], globalWeight);
        set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(first));
    }
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const uint32_t slot = iteration % 2;
        const uint32_t rawSlot = iteration % PREFETCH_SLOTS;
        const auto inputEvent = static_cast<event_t>(rawSlot);
        const auto event = static_cast<event_t>(slot);
        wait_flag(PIPE_MTE2, PIPE_V, inputEvent);
        wait_flag(PIPE_MTE3, PIPE_V, event);
        TCVT(nz[slot], packed[rawSlot], RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE2, inputEvent);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE2, inputEvent);
        if (iteration + PREFETCH_SLOTS < iterations) {
            const int64_t offset = expertBase + panelBase + (iteration + PREFETCH_SLOTS) * PACKED_TILE_BYTES + laneBase;
            GlobalPacked globalWeight(weight + offset);
            TLOAD(raw[rawSlot], globalWeight);
            set_flag(PIPE_MTE2, PIPE_V, inputEvent);
        }
        wait_flag(PIPE_V, PIPE_MTE3, event);
        wait_intra_block(PIPE_MTE3, static_cast<uint8_t>(1 + 2 * slot));
        TASSIGN(matSlot, L1_BASE + slot * L1_SLOT_BYTES);
        TINSERT(matSlot, nz[slot], static_cast<uint16_t>(lane * VEC_K_TILE), static_cast<uint16_t>(0));
        set_intra_block(PIPE_MTE3, static_cast<uint8_t>(2 * slot));
        set_flag(PIPE_MTE3, PIPE_V, event);
    }
}
#endif

#ifdef __DAV_CUBE__
template <typename AStage, typename GlobalA>
AICORE inline void loadAStage(AStage& target, __gm__ float8_e4m3_t* x,
                             int64_t k, int64_t rows, Chunk chunk, int64_t start, uint32_t slot, int64_t limit)
{
    const auto event = static_cast<event_t>(2 + slot);
    const int64_t validK = limit - start < A_STAGE_K ? limit - start : A_STAGE_K;
    target.SetValidCol(validK);
    GlobalA globalA(x + chunk.stageRow * k + start,
                    typename GlobalA::Shape(1, 1, 1, chunk.validRows, validK),
                    typename GlobalA::Stride(rows * k, rows * k, rows * k, k, 1));
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    TLOAD(target, globalA);
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
}

template <typename AScale, typename BScale, typename GlobalAScale, typename GlobalBScale>
AICORE inline void loadScaleStage(AScale& aScale, BScale& bScale,
                                 __gm__ float8_e8m0_t* xScale, __gm__ float8_e8m0_t* weightScale,
                                 int64_t k, int64_t n, int64_t experts, int64_t rows,
                                 Chunk chunk, int64_t n0, int64_t start, uint32_t slot, int64_t limit)
{
    const auto event = static_cast<event_t>(4 + slot);
    const int64_t q = k / 32;
    const int64_t globalQ = experts * q;
    const int64_t validQ = (limit - start < SCALE_STAGE_K ? limit - start : SCALE_STAGE_K) / 32;
#ifdef COMPACT_ACTIVATIONS
    // W2 scale rows are contiguous, with no padding between experts.
    // Bound the last row tile by this expert's actual count.
    const int64_t alignedRows = chunk.validRows;
#else
    const int64_t alignedRows = ((chunk.validRows + 15) / 16) * 16;
#endif
    aScale.SetValidRow(alignedRows);
    aScale.SetValidCol(validQ);
    bScale.SetValidRow(validQ);
#ifdef COMPACT_ACTIVATIONS
    // MX_A_ND uses row-major offsets; W13 input scales remain MX_A_ZZ.
    const int64_t aOffset = chunk.stageRow * q + start / 32;
#else
    const int64_t aOffset = chunk.stageRow * q + (start / 32) * 16;
#endif
    const int64_t bOffset = n0 * globalQ + (chunk.expert * q + start / 32) * 16;
    GlobalAScale globalA(xScale + aOffset, typename GlobalAScale::Shape(alignedRows, validQ),
                        typename GlobalAScale::Stride(rows, q));
    GlobalBScale globalB(weightScale + bOffset, typename GlobalBScale::Shape(validQ, N_TILE),
                        typename GlobalBScale::Stride(globalQ, n));
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    TLOAD(aScale, globalA);
    TLOAD(bScale, globalB);
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
}

template <uint32_t M, typename OutputSink>
AICORE inline void computeProjection(__gm__ float8_e4m3_t* x, __gm__ float8_e8m0_t* xScale,
                                     __gm__ float8_e8m0_t* weightScale, int64_t k, int64_t n,
                                     int64_t experts, int64_t rows, Chunk chunk, int64_t n0, OutputSink sink,
                                     int64_t kBegin = 0, int64_t kEnd = 0)
{
    const int64_t limit = kEnd == 0 ? k : kEnd;
    constexpr int VALID_M = DYNAMIC;
    using MatA = Tile<TileType::Mat, float8_e4m3_t, M, A_STAGE_K, BLayout::ColMajor,
                      VALID_M, DYNAMIC, SLayout::RowMajor, 512>;
    using MatB = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                      K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using MatAScale = Tile<TileType::Mat, float8_e8m0_t, M, SCALE_STAGE_Q, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::RowMajor, 32>;
    using MatBScale = Tile<TileType::Mat, float8_e8m0_t, SCALE_STAGE_Q, N_TILE, BLayout::ColMajor,
                           DYNAMIC, N_TILE, SLayout::ColMajor, 32>;
    using Left = TileLeftCompact<float8_e4m3_t, M, L0_K_TILE, VALID_M>;
    using Right = TileRight<float8_e4m3_t, L0_K_TILE, N_TILE>;
    using LeftScale = TileLeftScaleCompact<float8_e8m0_t, M, L0_SCALE_K_TILE, DYNAMIC>;
    using RightScale = TileRightScale<float8_e8m0_t, L0_SCALE_K_TILE, N_TILE>;
    using Acc = TileAccCompact<float, M, N_TILE, VALID_M>;
    using GlobalA = GlobalTensor<float8_e4m3_t, Shape<1, 1, 1, VALID_M, DYNAMIC>,
                                 pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
#ifdef COMPACT_ACTIVATIONS
    using GlobalAScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ND>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ND>, Layout::MX_A_ND>;
#else
    using GlobalAScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ZZ>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ZZ>, Layout::MX_A_ZZ>;
#endif
    using GlobalBScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, N_TILE, Layout::MX_B_NN>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_B_NN>, Layout::MX_B_NN>;
    MatA aMat[2];
    MatB bMat;
    MatAScale aScaleMat[2];
    MatBScale bScaleMat[2];
    Left a[2];
    Right b[2];
    LeftScale aScale[2];
    RightScale bScale[2];
    Acc acc;
    {
        aMat[0].SetValidRow(chunk.validRows);
        aMat[1].SetValidRow(chunk.validRows);
        a[0].SetValidRow(chunk.validRows);
        a[1].SetValidRow(chunk.validRows);
        acc.SetValidRow(chunk.validRows);
        aScale[0].SetValidRow(chunk.validRows);
        aScale[1].SetValidRow(chunk.validRows);
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        TASSIGN(aMat[slot], slot * M_TILE * A_STAGE_K);
        TASSIGN(aScaleMat[slot], 2 * M_TILE * A_STAGE_K + slot * M_TILE * SCALE_STAGE_Q);
        TASSIGN(bScaleMat[slot], 2 * M_TILE * (A_STAGE_K + SCALE_STAGE_Q) + slot * N_TILE * SCALE_STAGE_Q);
        TASSIGN(a[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(b[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(aScale[slot], GetScaleAddr(a[slot].data()));
        TASSIGN(bScale[slot], GetScaleAddr(b[slot].data()));
    }
    TASSIGN(acc, 0);
    loadAStage<MatA, GlobalA>(aMat[0], x, k, rows, chunk, kBegin, 0, limit);
    loadScaleStage<MatAScale, MatBScale, GlobalAScale, GlobalBScale>(
        aScaleMat[0], bScaleMat[0], xScale, weightScale, k, n, experts, rows, chunk, n0, kBegin, 0, limit);
    for (int64_t k0 = kBegin; k0 < limit; k0 += K_TILE) {
        const uint32_t slot = ((k0 - kBegin) / K_TILE) % 2;
        const uint32_t aSlot = ((k0 - kBegin) / A_STAGE_K) % 2;
        const uint32_t scaleSlot = ((k0 - kBegin) / SCALE_STAGE_K) % 2;
        const uint16_t aOffset = (k0 - kBegin) % A_STAGE_K;
        const uint16_t scaleOffset = ((k0 - kBegin) % SCALE_STAGE_K) / 32;
        if (aOffset == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2 + aSlot));
        }
        if (scaleOffset == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(4 + scaleSlot));
        }
        waitSlot(slot);
        TASSIGN(bMat, L1_BASE + slot * L1_SLOT_BYTES);
        for (uint32_t half = 0; half < 2; ++half) {
            const auto event = static_cast<event_t>(half);
            const uint16_t offset = half * L0_K_TILE;
            const uint16_t qOffset = half * L0_SCALE_K_TILE;
            wait_flag(PIPE_M, PIPE_MTE1, event);
            TEXTRACT(a[half], aMat[aSlot], 0, static_cast<uint16_t>(aOffset + offset));
            TEXTRACT(b[half], bMat, offset, 0);
            TEXTRACT(aScale[half], aScaleMat[scaleSlot], 0, static_cast<uint16_t>(scaleOffset + qOffset));
            TEXTRACT(bScale[half], bScaleMat[scaleSlot], static_cast<uint16_t>(scaleOffset + qOffset), 0);
            if (half == 1) {
                releaseSlot(slot);
                if (aOffset + K_TILE == A_STAGE_K || k0 + K_TILE == limit) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + aSlot));
                }
                if ((k0 - kBegin) % SCALE_STAGE_K + K_TILE == SCALE_STAGE_K || k0 + K_TILE == limit) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + scaleSlot));
                }
            }
            set_flag(PIPE_MTE1, PIPE_M, event);
            wait_flag(PIPE_MTE1, PIPE_M, event);
            if (k0 == kBegin && half == 0) {
                TMATMUL_MX<AccPhase::Partial>(acc, a[half], aScale[half], b[half], bScale[half]);
            } else if (k0 + K_TILE == limit && half == 1) {
                TMATMUL_MX<AccPhase::Final>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            } else {
                TMATMUL_MX<AccPhase::Partial>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            }
            set_flag(PIPE_M, PIPE_MTE1, event);
        }
        if (aOffset == 0 && k0 + A_STAGE_K < limit) {
            loadAStage<MatA, GlobalA>(aMat[1-aSlot], x, k, rows, chunk, k0 + A_STAGE_K, 1-aSlot, limit);
        }
        if (scaleOffset == 0 && k0 + SCALE_STAGE_K < limit) {
            loadScaleStage<MatAScale, MatBScale, GlobalAScale, GlobalBScale>(
                aScaleMat[1-scaleSlot], bScaleMat[1-scaleSlot], xScale, weightScale,
                k, n, experts, rows, chunk, n0, k0 + SCALE_STAGE_K, 1-scaleSlot, limit);
        }
    }
    sink(acc);
}

#endif

AICORE inline void beginGemm()
{
    for (uint32_t slot = 0; slot < 2; ++slot) {
#ifdef __DAV_CUBE__
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + slot));
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + slot));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        releaseSlot(slot);
#endif
#ifdef __DAV_VEC__
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
#endif
    }
}

AICORE inline void endGemm()
{
    for (uint32_t slot = 0; slot < 2; ++slot) {
#ifdef __DAV_CUBE__
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + slot));
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + slot));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
#endif
#ifdef __DAV_VEC__
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
        wait_intra_block(PIPE_MTE3, static_cast<uint8_t>(1 + 2 * slot));
#endif
    }
    pipe_barrier(PIPE_ALL);
}

}  // namespace grouped_mxfp4
