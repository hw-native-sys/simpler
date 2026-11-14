// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 matrix projection with buffered operands and an FP32 accumulator emitter.
#pragma once

#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

namespace bf16_projection {
using namespace pto;
constexpr int M_TILE = 80;
using BFloat = bfloat16_t;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename T>
struct StoreOutput {
    static constexpr bool full_rows = false;
    __gm__ T* output;
    int64_t stride;

    template <typename Acc>
    __aicore__ inline void operator()(Acc& acc, int rows, int column, int row) {
        using Shape = TileShape2D<T, DYNAMIC, Acc::Cols>;
        using Base = BaseShape2D<T, DYNAMIC, DYNAMIC>;
        GlobalTensor<T, Shape, Base> gm(output + row * stride + column, Shape(rows, Acc::Cols), Base(rows, stride));
        TSTORE(gm, acc);
    }
};

template <int K, int N, int NT, int KT, int KK, int Slots, Layout WeightLayout, int MatRows = 80, int AccSlots = 1>
struct PanelPipeline {
    static_assert(K % KT == 0 && KT % KK == 0 && Slots <= K / KT);
    static_assert(AccSlots > 0 && AccSlots <= 2);
    static_assert(MatRows * KK * sizeof(BFloat) <= 32768);
    static_assert((MatRows + NT) * KT * sizeof(BFloat) * Slots <= 524288);
    static_assert(2 * KK * NT * sizeof(BFloat) <= 65536);
    static_assert(AccSlots * MatRows * NT * sizeof(float) <= 262144);
    static_assert(WeightLayout == Layout::ND || WeightLayout == Layout::NZ);
    static constexpr int SlotBytes = (MatRows + NT) * KT * sizeof(BFloat);
    using AM = Tile<TileType::Mat, BFloat, MatRows, KT, BLayout::ColMajor, DYNAMIC, KT, SLayout::RowMajor>;
    using BM = Tile<TileType::Mat, BFloat, KT, NT, BLayout::ColMajor, KT, NT, SLayout::RowMajor>;
    using AL = TileLeftCompact<BFloat, MatRows, KK, DYNAMIC, KK>;
    using BR = TileRight<BFloat, KK, NT>;
    using AC = TileAccCompact<float, MatRows, NT, DYNAMIC, NT>;

    __aicore__ static inline void load(
        __gm__ BFloat* hidden, __gm__ BFloat* weight, int rows, int step, int worker,
        int workers, int slot, int64_t hidden_stride, int64_t weight_stride, int column_base) {
        int k = step % (K / KT) * KT;
        int n = column_base + (worker + step / (K / KT) * workers) * NT;
        AM a(rows);
        BM b;
        TASSIGN(a, slot * SlotBytes);
        TASSIGN(b, slot * SlotBytes + MatRows * KT * sizeof(BFloat));
        if (rows > 0) {
            using Shape = TileShape2D<BFloat, DYNAMIC, KT>;
            using Base = BaseShape2D<BFloat, DYNAMIC, DYNAMIC>;
            GlobalTensor<BFloat, Shape, Base> gm(hidden + k, Shape(rows, KT), Base(rows, hidden_stride));
            TLOAD(a, gm);
        }
        if constexpr (WeightLayout == Layout::NZ) {
            using Shape = TileShape2D<BFloat, KT, NT, Layout::NZ>;
            using Base = BaseShape2D<BFloat, K, N, Layout::NZ>;
            GlobalTensor<BFloat, Shape, Base, Layout::NZ> gm(weight + n * K + k * 16);
            TLOAD<TLoadL2Hint::NotAllocKeep>(b, gm);
        } else {
            using Shape = TileShape2D<BFloat, KT, NT>;
            using Base = BaseShape2D<BFloat, DYNAMIC, DYNAMIC>;
            GlobalTensor<BFloat, Shape, Base> gm(weight + k * weight_stride + n, Shape(), Base(K, weight_stride));
            TLOAD<TLoadL2Hint::NotAllocKeep>(b, gm);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
    }

    template <typename Emit>
    __aicore__ static inline void run(
        __gm__ BFloat* hidden, __gm__ BFloat* weight, int rows, int input_rows,
        int worker, int workers, int panels, int64_t hidden_stride, int64_t weight_stride,
        int column_base, int row, Emit& emit) {
        if (panels <= 0) return;
        int compute_rows = Emit::full_rows ? MatRows : rows;
        AC acc(compute_rows);
        if constexpr (AccSlots == 1) TASSIGN(acc, 0);
        for (int slot = 0; slot < AccSlots; ++slot) {
            set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        for (int slot = 0; slot < 2; ++slot) {
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        }
        for (int slot = 0; slot < Slots; ++slot) {
            load(hidden, weight, input_rows, slot, worker, workers, slot, hidden_stride, weight_stride, column_base);
        }
        for (int step = 0; step < panels * (K / KT); ++step) {
            int slot = step % Slots;
            int acc_slot = step / (K / KT) % AccSlots;
            auto acc_event = static_cast<event_t>(4 + acc_slot);
            if constexpr (AccSlots > 1) TASSIGN(acc, acc_slot * MatRows * NT * sizeof(float));
            int k = step % (K / KT) * KT;
            int n = column_base + (worker + step / (K / KT) * workers) * NT;
            AM am(compute_rows);
            BM bm;
            TASSIGN(am, slot * SlotBytes);
            TASSIGN(bm, slot * SlotBytes + MatRows * KT * sizeof(BFloat));
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
            for (int inner = 0; inner < KT / KK; ++inner) {
                int inner_slot = inner % 2;
                auto event = static_cast<event_t>(inner_slot);
                AL left(compute_rows);
                BR right;
                TASSIGN(left, inner_slot * 32768);
                TASSIGN(right, inner_slot * 32768);
                wait_flag(PIPE_M, PIPE_MTE1, event);
                TEXTRACT(left, am, 0, inner * KK);
                TEXTRACT(right, bm, inner * KK, 0);
                set_flag(PIPE_MTE1, PIPE_M, event);
                wait_flag(PIPE_MTE1, PIPE_M, event);
                if (k == 0 && inner == 0) {
                    wait_flag(PIPE_FIX, PIPE_M, acc_event);
                    TMATMUL(acc, left, right);
                } else {
                    TMATMUL_ACC(acc, acc, left, right);
                }
                set_flag(PIPE_M, PIPE_MTE1, event);
            }
            if (step + Slots < panels * (K / KT)) {
                set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                load(hidden, weight, input_rows, step + Slots, worker, workers, slot, hidden_stride, weight_stride, column_base);
            }
            if (k + KT == K) {
                set_flag(PIPE_M, PIPE_FIX, acc_event);
                wait_flag(PIPE_M, PIPE_FIX, acc_event);
                emit(acc, rows, n, row);
                set_flag(PIPE_FIX, PIPE_M, acc_event);
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        }
        for (int slot = 0; slot < AccSlots; ++slot) {
            wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        pipe_barrier(PIPE_ALL);
    }
};

template <int K, int N, int NT, int KT, int KK, int Slots, Layout WeightLayout, int MatRows = 80, int AccSlots = 1, typename Emit>
__aicore__ inline void project(
    __gm__ BFloat* hidden, __gm__ BFloat* weight, int rows, int input_rows,
    int64_t hidden_stride, int64_t weight_stride, int row_tile, int worker, int workers, Emit& emit) {
    constexpr int Tail = N % NT;
    row_tile = row_tile < MatRows ? row_tile : MatRows;
    int panels = (N / NT + workers - 1 - worker) / workers;
    for (int row = 0; row < rows; row += row_tile) {
        int valid_rows = rows - row < row_tile ? rows - row : row_tile;
        int loaded_rows = input_rows - row < valid_rows ? input_rows - row : valid_rows;
        loaded_rows = loaded_rows > 0 ? loaded_rows : 0;
        auto input = hidden + row * hidden_stride;
        PanelPipeline<K, N, NT, KT, KK, Slots, WeightLayout, MatRows, AccSlots>::run(
            input, weight, valid_rows, loaded_rows, worker, workers, panels, hidden_stride, weight_stride, 0, row, emit);
        if constexpr (Tail > 0) {
            if (worker == (N / NT) % workers) {
                PanelPipeline<K, N, Tail, KT, KK, Slots, WeightLayout, MatRows, AccSlots>::run(
                    input, weight, valid_rows, loaded_rows, 0, 1, 1, hidden_stride, weight_stride, N / NT * NT, row, emit);
            }
        }
    }
}
}  // namespace bf16_projection
