// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Router Top-K selection and normalized weight publication.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
constexpr int Rows = 8;
constexpr int TopK = 6;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using Indices = Vec<int32_t, Rows, 8>;
using Selected = Vec<float, Rows, 8>;
using Denominator = Tile<TileType::Vec, float, Rows, 1, BLayout::ColMajor>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
#if defined(PTO_NPU_ARCH_A5)
template <int RowStride>
__tf__ __aicore__ inline void gather_indices(
    Indices::TileDType __out__ output,
    typename Vec<float, Rows, RowStride>::TileDType __in__ sorted) {
    auto src = (__ubuf__ int32_t*)__cce_get_tile_ptr(sorted);
    auto dst = (__ubuf__ int32_t*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<uint32_t> lane, offset;
        RegTensor<int32_t> index;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vmuls(offset, lane, 2u, all, MODE_ZEROING);
        if constexpr (RowStride != 16) {
            RegTensor<uint32_t> row;
            vshrs(row, lane, 3u, all, MODE_ZEROING);
            vmuls(row, row, static_cast<uint32_t>(RowStride - 16), all, MODE_ZEROING);
            vadd(offset, offset, row, all, MODE_ZEROING);
        }
        vadds(offset, offset, 1u, all, MODE_ZEROING);
        vgather2(index, src, offset, all);
        vsts(index, dst, 0, NORM_B32, all);
    }
}

template <int Columns>
__tf__ __aicore__ inline void gather_scores(
    Selected::TileDType __out__ output, Indices::TileDType __in__ indices,
    typename Vec<float, Rows, Columns>::TileDType __in__ scores) {
    auto idx = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), active;
        RegTensor<uint32_t> lane, row, column, bits, index, offset;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(row, lane, 3u, all, MODE_ZEROING);
        vdup(bits, 7u, all, MODE_ZEROING);
        vand(column, lane, bits, all, MODE_ZEROING);
        vcmps_lt(active, column, static_cast<uint32_t>(TopK), all);
        vlds((RegTensor<int32_t>&)index, idx, 0, NORM);
        vmuls(offset, row, static_cast<uint32_t>(Columns), all, MODE_ZEROING);
        vadd(index, index, offset, all, MODE_ZEROING);
        RegTensor<float> value, zero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vgather2(value, src, index, all);
        vsel(value, value, zero, active);
        vsts(value, dst, 0, NORM_B32, all);
    }
}

__tf__ __aicore__ inline void normalize_weights(
    Selected::TileDType __out__ output, Selected::TileDType __in__ selected,
    Denominator::TileDType __in__ denominator, float route_scale) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(selected);
    auto inv = (__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<uint32_t> lane, row;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(row, lane, 3u, all, MODE_ZEROING);
        RegTensor<float> value, divisor;
        vlds(value, src, 0, NORM);
        vgather2(divisor, inv, row, all);
        vdiv(value, value, divisor, all, MODE_ZEROING);
        vmuls(value, value, route_scale, all, MODE_ZEROING);
        vsts(value, dst, 0, NORM_B32, all);
    }
}
#endif

template <int Columns>
__aicore__ inline void run(__gm__ int64_t* args, int block) {
    int capacity = reinterpret_cast<__gm__ Tensor*>(args[2])->shapes[0];
    int active = args[4];
    active = active < 0 ? 0 : (active > capacity ? capacity : active);
    int first = block * Rows;
    int valid = active - first;
    valid = valid > Rows ? Rows : valid;
    if (valid <= 0) return;
    float route_scale = *reinterpret_cast<__gm__ float*>(args + 5);
    using Input = Vec<float, Rows, Columns>;
    using SortedBatch = Vec<float, Rows, 2 * Columns>;
    using SortedFlat = Vec<float, 1, Rows * 2 * Columns>;
    using Prefix = Vec<float, 1, 16>;
    using PrefixBatch = Vec<float, Rows, 16>;
    using Part = Vec<float, 1, 16>;
    using Scratch = Vec<float, 1, 48>;
    using RangeRow = Vec<uint32_t, 1, Columns>;
    using Range = Vec<uint32_t, Rows, Columns>;
    Input biased, scores;
    Range range;
    RangeRow range_row;
    SortedBatch sorted32, sorted64;
    PrefixBatch sorted;
    SortedFlat sorted32_flat, sorted64_flat;
    Indices indices;
    Selected selected, weights;
    Denominator denominator;
    Vec<float, Rows, 128> sum_scratch;
    TASSIGN(biased, 0); TASSIGN(scores, 16384);
    TASSIGN(sorted32, 32768); TASSIGN(sorted32_flat, 32768);
    TASSIGN(sorted64_flat, 65536); TASSIGN(sorted64, 65536); TASSIGN(sorted, 98304);
    TASSIGN(range, 102400); TASSIGN(range_row, 163840); TASSIGN(indices, 167936);
    TASSIGN(selected, 168192); TASSIGN(weights, 168448);
    TASSIGN(denominator, 168704); TASSIGN(sum_scratch, 169984);
    using Shape = TileShape2D<float, Rows, Columns>;
    using Stride = BaseShape2D<float, Rows, Columns>;
    GlobalTensor<float, Shape, Stride> biased_gm(data<float>(args, 0) + first * Columns);
    GlobalTensor<float, Shape, Stride> scores_gm(data<float>(args, 1) + first * Columns);
    TLOAD(biased, biased_gm); TLOAD(scores, scores_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCI<RangeRow, uint32_t, 0>(range_row, 0u);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCOLEXPAND(range, range_row);
    pipe_barrier(PIPE_ALL);
    TSORT32(sorted32, biased, range);
    pipe_barrier(PIPE_ALL);
    if constexpr (Columns > 32) {
        TMRGSORT(sorted64_flat, sorted32_flat, 64);
        pipe_barrier(PIPE_ALL);
    }
    if constexpr (Columns > 128) {
        for (int row = 0; row < Rows; ++row) {
            int offset = row * 2 * Columns * sizeof(float);
            Prefix sorted_row;
            Scratch scratch_row;
            TASSIGN(sorted_row, 98304 + row * 16 * sizeof(float));
            TASSIGN(scratch_row, 131072 + offset);
            Part p0, p1, p2;
            TASSIGN(p0, 65536 + offset);
            TASSIGN(p1, 66560 + offset);
            TASSIGN(p2, 67584 + offset);
            MrgSortExecutedNumList executed{0, 0, 0, 0};
            if constexpr (Columns == 256) {
                TMRGSORT<Prefix, Scratch, Part, Part, false>(sorted_row, executed, scratch_row, p0, p1);
            } else {
                TMRGSORT<Prefix, Scratch, Part, Part, Part, false>(sorted_row, executed, scratch_row, p0, p1, p2);
            }
        }
        pipe_barrier(PIPE_ALL);
    }
#if defined(PTO_NPU_ARCH_A5)
    if constexpr (Columns == 32) gather_indices<2 * Columns>(indices.data(), sorted32.data());
    else if constexpr (Columns == 128) gather_indices<2 * Columns>(indices.data(), sorted64.data());
    else gather_indices<16>(indices.data(), sorted.data());
#else
    for (int row = 0; row < Rows; ++row) {
        Vec<float, 1, 16> prefix;
        Vec<int32_t, 1, 8> row_indices;
        constexpr int PrefixBase = Columns == 32 ? 32768 : Columns == 128 ? 65536 : 98304;
        constexpr int PrefixStride = Columns <= 128 ? 2 * Columns : 16;
        TASSIGN(prefix, PrefixBase + row * PrefixStride * sizeof(float));
        TASSIGN(row_indices, 167936 + row * 32);
        TGATHER<decltype(row_indices), decltype(prefix), MaskPattern::P1010>(row_indices, prefix);
    }
#endif
    pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
    gather_scores<Columns>(selected.data(), indices.data(), scores.data());
#else
    Vec<int32_t, Rows, 8> offsets, row_index;
    Vec<int32_t, 1, 64> linear;
    Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, DYNAMIC, DYNAMIC> gathered(Rows, 8);
    TASSIGN(offsets, 176128); TASSIGN(row_index, 176384); TASSIGN(linear, 176384);
    TASSIGN(gathered, 176640);
    TCI<decltype(linear), int32_t, 0>(linear, 0);
    TDIVS(row_index, row_index, 8);
    TMULS(offsets, row_index, Columns);
    TADD(offsets, offsets, indices);
    Vec<int32_t, Rows, 8> gather_scratch;
    TASSIGN(gather_scratch, 177152);
    TGATHER(gathered, scores, offsets, gather_scratch);
    gathered.SetValidShape(Rows, TopK);
    Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, Rows, 8,
         SLayout::NoneBox, 512, PadValue::Zero> padded;
    TASSIGN(padded, 168192);
    TFILLPAD(padded, gathered);
#endif
    pipe_barrier(PIPE_ALL);
    TROWSUM(denominator, selected, sum_scratch);
    pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
    normalize_weights(weights.data(), selected.data(), denominator.data(), route_scale);
#else
    TROWEXPANDDIV(weights, selected, denominator);
    TMULS(weights, weights, route_scale);
#endif
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using OutputIndex = Tile<TileType::Vec, int32_t, Rows, 8, BLayout::RowMajor, DYNAMIC, TopK>;
    using OutputWeight = Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, DYNAMIC, TopK>;
    OutputIndex output_index(valid);
    OutputWeight output_weight(valid);
    TASSIGN(output_index, 167936); TASSIGN(output_weight, 168448);
    using IndexShape = TileShape2D<int32_t, DYNAMIC, TopK>;
    using WeightShape = TileShape2D<float, DYNAMIC, TopK>;
    GlobalTensor<int32_t, IndexShape, BaseShape2D<int32_t, Rows, TopK>> index_gm(
        data<int32_t>(args, 2) + first * TopK, IndexShape(valid, TopK), BaseShape2D<int32_t, Rows, TopK>());
    GlobalTensor<float, WeightShape, BaseShape2D<float, Rows, TopK>> weight_gm(
        data<float>(args, 3) + first * TopK, WeightShape(valid, TopK), BaseShape2D<float, Rows, TopK>());
    TSTORE(index_gm, output_index); TSTORE(weight_gm, output_weight);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
#endif
    int columns = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[1];
    int tasks = (args[4] + Rows - 1) / Rows;
    for (int block = get_block_idx(args); block < tasks; block += get_block_num(args)) {
        if (columns == 32) run<32>(args, block);
        else if (columns == 128) run<128>(args, block);
        else if (columns == 256) run<256>(args, block);
        else run<384>(args, block);
    }
#endif
}
