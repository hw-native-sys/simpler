// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Split-K HC projection with nearest-away HF32 operands and FP32 partials.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
constexpr int M_TILE = 80;
constexpr int N_TILE = 32;
constexpr int K_TILE = 64;
constexpr int K_L1_TILE = 512;
constexpr int K = 20480;
constexpr int N = 24;
constexpr int SPLITS = 4;
constexpr int K_SPLIT = K / SPLITS;
constexpr int A_L1_BYTES = M_TILE * K_L1_TILE * sizeof(float);
constexpr int L1_SLOT_BYTES = (M_TILE + N_TILE) * K_L1_TILE * sizeof(float);
static_assert(K_SPLIT % K_L1_TILE == 0 && K_L1_TILE % K_TILE == 0);
using AM = Tile<TileType::Mat, float, M_TILE, K_L1_TILE, BLayout::ColMajor, DYNAMIC, K_L1_TILE, SLayout::RowMajor>;
using BM = Tile<TileType::Mat, float, K_L1_TILE, N_TILE, BLayout::RowMajor, K_L1_TILE, N, SLayout::ColMajor>;
// Compact L0 pitches follow the valid row count on partial M tiles.
using AL = TileLeftCompact<float, M_TILE, K_TILE, DYNAMIC, K_TILE>;
using BR = TileRight<float, K_TILE, N_TILE, K_TILE, N>;
using ACC = TileAccCompact<float, M_TILE, N_TILE, DYNAMIC, N>;

__aicore__ inline __gm__ float* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ float*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__CPU_SIM)
template <typename T>
void round_hf32(T& tile) {
    for (int row = 0; row < tile.GetValidRow(); ++row) {
        for (int col = 0; col < tile.GetValidCol(); ++col) {
            union { float value; uint32_t bits; } x;
            x.value = tile.GetElement(row, col);
            if ((x.bits & 0x7f800000) != 0x7f800000) x.bits = (x.bits + 4096) & 0xffffe000;
            tile.SetElement(row, col, x.value);
        }
    }
}
#endif

__aicore__ inline void load(__gm__ float* x, __gm__ float* weight, int rows, int k, int slot) {
    AM a(rows);
    BM b;
    TASSIGN(a, slot * L1_SLOT_BYTES);
    TASSIGN(b, slot * L1_SLOT_BYTES + A_L1_BYTES);
    using AShape = TileShape2D<float, DYNAMIC, K_L1_TILE>;
    using AStride = BaseShape2D<float, DYNAMIC, K>;
    GlobalTensor<float, AShape, AStride> ag(x + k, AShape(rows, K_L1_TILE), AStride(rows, K));
    using BShape = TileShape2D<float, K_L1_TILE, N, Layout::DN>;
    using BStride = BaseShape2D<float, K, N, Layout::DN>;
    GlobalTensor<float, BShape, BStride, Layout::DN> bg(weight + k);
    TLOAD(a, ag);
    TLOAD(b, bg);
#if defined(__CPU_SIM)
    round_hf32(a);
    round_hf32(b);
#endif
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    int rows = args[3];
    int padded_rows = args[4];
    int tasks = (rows + M_TILE - 1) / M_TILE * SPLITS;
    ACC acc;
    TASSIGN(acc, 0);
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_atomic_none();
    uint64_t saved_ctrl = get_ctrl();
    acc.SetMadTF32Mode(RoundMode::CAST_ROUND);
#endif
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int row = task / SPLITS * M_TILE;
        int split = task % SPLITS;
        int valid_rows = rows - row < M_TILE ? rows - row : M_TILE;
        __gm__ float* x = data(args, 0) + row * K + split * K_SPLIT;
        __gm__ float* weight = data(args, 1) + split * K_SPLIT;
        acc.SetValidRow(valid_rows);
        for (int slot = 0; slot < 2; ++slot) {
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            load(x, weight, valid_rows, slot * K_L1_TILE, slot);
        }
        for (int step = 0; step < K_SPLIT / K_L1_TILE; ++step) {
            int slot = step % 2;
            event_t event = static_cast<event_t>(slot);
            AM a(valid_rows);
            BM b;
            TASSIGN(a, slot * L1_SLOT_BYTES);
            TASSIGN(b, slot * L1_SLOT_BYTES + A_L1_BYTES);
            wait_flag(PIPE_MTE2, PIPE_MTE1, event);
            for (int kb = 0; kb < K_L1_TILE / K_TILE; ++kb) {
                int inner_slot = (step * (K_L1_TILE / K_TILE) + kb) % 2;
                event_t inner_event = static_cast<event_t>(inner_slot);
                AL left(valid_rows);
                BR right;
                TASSIGN(left, inner_slot * M_TILE * K_TILE * sizeof(float));
                TASSIGN(right, inner_slot * K_TILE * N_TILE * sizeof(float));
                wait_flag(PIPE_M, PIPE_MTE1, inner_event);
                TEXTRACT(left, a, 0, kb * K_TILE);
                TEXTRACT(right, b, kb * K_TILE, 0);
                set_flag(PIPE_MTE1, PIPE_M, inner_event);
                wait_flag(PIPE_MTE1, PIPE_M, inner_event);
                if (step == 0 && kb == 0) TMATMUL(acc, left, right);
                else TMATMUL_ACC(acc, acc, left, right);
                set_flag(PIPE_M, PIPE_MTE1, inner_event);
            }
            if (step + 2 < K_SPLIT / K_L1_TILE) {
                set_flag(PIPE_MTE1, PIPE_MTE2, event);
                wait_flag(PIPE_MTE1, PIPE_MTE2, event);
                load(x, weight, valid_rows, (step + 2) * K_L1_TILE, slot);
            }
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        using OutShape = TileShape2D<float, DYNAMIC, N>;
        using OutStride = BaseShape2D<float, DYNAMIC, N_TILE>;
        GlobalTensor<float, OutShape, OutStride> output(data(args, 2) + (split * padded_rows + row) * N_TILE,
            OutShape(valid_rows, N), OutStride(valid_rows, N_TILE));
        TSTORE(output, acc);
        pipe_barrier(PIPE_ALL);
    }
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_ctrl(saved_ctrl);
#endif
}
