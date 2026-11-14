// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Inactive route initialization and padded router score fill.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
constexpr int PadRows = 80;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
__aicore__ inline void initialize(__gm__ int64_t* args) {
    auto index_meta = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto score_meta = reinterpret_cast<__gm__ Tensor*>(args[2]);
    int capacity = index_meta->shapes[0], topk = index_meta->shapes[1];
    int padded_rows = score_meta->shapes[0], padded_columns = score_meta->shapes[1];
    int active = args[3], experts = args[4];
    active = active < 0 ? 0 : (active > capacity ? capacity : active);
    int count = (capacity - active) * topk;
    Tile<TileType::Vec, int32_t, 1, 512> zeros;
    TASSIGN(zeros, 0);
    if (count > 0) {
        TEXPANDS(zeros, 0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        for (int offset = 0; offset < count; offset += 512) {
            int values = count - offset; values = values > 512 ? 512 : values;
            Tile<TileType::Vec, int32_t, 1, 512, BLayout::RowMajor, 1, DYNAMIC> index(values);
            Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, DYNAMIC> weight(values);
            TASSIGN(index, 0); TASSIGN(weight, 0);
            using IndexShape = TileShape2D<int32_t, 1, DYNAMIC>;
            using WeightShape = TileShape2D<float, 1, DYNAMIC>;
            using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
            GlobalTensor<int32_t, IndexShape, Stride> index_gm(
                data<int32_t>(args, 0) + active * topk + offset,
                IndexShape(1, values), Stride(1, 1, 1, values, 1));
            GlobalTensor<float, WeightShape, Stride> weight_gm(
                data<float>(args, 1) + active * topk + offset,
                WeightShape(1, values), Stride(1, 1, 1, values, 1));
            TSTORE(index_gm, index); TSTORE(weight_gm, weight);
        }
    }
    int padding = padded_columns - experts;
    if (padding > 0) {
        Tile<TileType::Vec, float, PadRows, 384, BLayout::RowMajor, DYNAMIC, DYNAMIC> pad(PadRows, padding);
        TASSIGN(pad, 4096);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (int row = 0; row < padded_rows; row += PadRows) {
            int rows = padded_rows - row; rows = rows > PadRows ? PadRows : rows;
            pad.SetValidShape(rows, padding);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TEXPANDS(pad, -3.4028234663852886e38f);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            using Shape = TileShape2D<float, DYNAMIC, DYNAMIC>;
            using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
            GlobalTensor<float, Shape, Stride> destination(
                data<float>(args, 2) + row * padded_columns + experts,
                Shape(rows, padding), Stride(1, 1, 1, padded_columns, 1));
            TSTORE(destination, pad);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
#endif
    initialize(args);
#endif
}
