/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
// Kernel Function: residual_rms_cast_1

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

// --- ptoas-generated code ---

enum class PTOAutoSyncTailMode : int {
    kBarrierAll = 0,
    kSetWaitMte3ToSEvent0 = 1,
};

static __aicore__ inline void ptoas_auto_sync_tail(PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
    switch (mode) {
    case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        break;
    case PTOAutoSyncTailMode::kBarrierAll:
    default:
        pipe_barrier(PIPE_ALL);
        break;
    }
}

// The two stores below address the band relatively: the task's `mlp_norm_in` and
// `post_norm_partial` arguments are views narrowed to [k_base, k_base + 1024), so
// their base already carries k_base and only the in-band offset is added. The
// loads keep the parent views and so still index absolutely.
static __aicore__ void residual_rms_cast_1(
    __gm__ bfloat16_t *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, int64_t v6,
    int64_t v7
) {
    SaturationMode v8 = SaturationMode::OFF;
    RoundMode v9 = RoundMode::CAST_ROUND;
    const int64_t v10 = 256;
    const int64_t v11 = 2;
    const int64_t v12 = 4;
    const int64_t v13 = 1;
    const int64_t v14 = 5120;
    const int64_t v15 = 16;
    const int64_t v16 = 0;
    const int64_t v17 = 51200;
    const int64_t v18 = 34816;
    const int64_t v19 = 33792;
    const int64_t v20 = 17408;
    const int64_t v21 = 1024;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    for (size_t v22 = (size_t)v16; v22 < ((size_t)v12); v22 += (size_t)v11) {
        int64_t v23 = (int64_t)((uint64_t)((int64_t)v22) * (uint64_t)v10);
        int64_t v24 = (int64_t)((uint64_t)v6 + (uint64_t)v23);
        int64_t v25 = (int64_t)((uint64_t)v6 + (uint64_t)((int64_t)(uint64_t)v23 + (uint64_t)v10));
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v26 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v27 = (uint64_t)v21;
        TASSIGN(v26, v27);
        pto::Shape<1, 1, 1, 16, 256> v28 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v29 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v30 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v3 + (v16 + v16 * v14 + v24 * v13), v28, v29
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v26, v30);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v31 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v32 = (uint64_t)v20;
        TASSIGN(v31, v32);
        pto::Shape<1, 1, 1, 16, 256> v33 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v34 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v35 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v4 + (v16 + v16 * v14 + v24 * v13), v33, v34
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v31, v35);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v36 = Tile<
                TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v10);
        uint64_t v37 = (uint64_t)v19;
        TASSIGN(v36, v37);
        pto::Shape<1, 1, 1, 1, 256> v38 = pto::Shape<1, 1, 1, 1, 256>();
        pto::Stride<5120, 5120, 5120, 5120, 1> v39 = pto::Stride<5120, 5120, 5120, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v40 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(
                v5 + (v16 + v7 * v14 + v24 * v13), v38, v39
            );
        TLOAD(v36, v40);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v41 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v42 = (uint64_t)v18;
        TASSIGN(v41, v42);
        pto::Shape<1, 1, 1, 16, 256> v43 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v44 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v45 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v3 + (v16 + v16 * v14 + v25 * v13), v43, v44
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v41, v45);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v46 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v47 = (uint64_t)v17;
        TASSIGN(v46, v47);
        pto::Shape<1, 1, 1, 16, 256> v48 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v49 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v50 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v4 + (v16 + v16 * v14 + v25 * v13), v48, v49
            );
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(v46, v50);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        Tile<
            TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v51 = Tile<
                TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v13, v10);
        uint64_t v52 = (uint64_t)v16;
        TASSIGN(v51, v52);
        pto::Shape<1, 1, 1, 1, 256> v53 = pto::Shape<1, 1, 1, 1, 256>();
        pto::Stride<5120, 5120, 5120, 5120, 1> v54 = pto::Stride<5120, 5120, 5120, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v55 =
            GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(
                v5 + (v16 + v7 * v14 + v25 * v13), v53, v54
            );
        TLOAD(v51, v55);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v56 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v57 = (uint64_t)v21;
        TASSIGN(v56, v57);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(v56, v26, v31);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v59 = (uint64_t)v20;
        TASSIGN(v58, v59);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCOLEXPANDMUL(v58, v56, v36);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        Tile<
            TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v60 = Tile<
                TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v61 = (uint64_t)v20;
        TASSIGN(v60, v61);
        pipe_barrier(PIPE_V);
        TCVT(v60, v58, v9, v8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pto::Shape<1, 1, 1, 16, 256> v62 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v63 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v64 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v16 + v16 * v14 + v23 * v13), v62, v63
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v64, v56);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pto::Shape<1, 1, 1, 16, 256> v65 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v66 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v67 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v16 + v16 * v14 + v23 * v13), v65, v66
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v67, v60);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v68 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v69 = (uint64_t)v18;
        TASSIGN(v68, v69);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TADD(v68, v41, v46);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        Tile<
            TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v70 = Tile<
                TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v71 = (uint64_t)v17;
        TASSIGN(v70, v71);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TCOLEXPANDMUL(v70, v68, v51);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        Tile<
            TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v72 = Tile<
                TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v73 = (uint64_t)v17;
        TASSIGN(v72, v73);
        pipe_barrier(PIPE_V);
        TCVT(v72, v70, v9, v8);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        pto::Shape<1, 1, 1, 16, 256> v74 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v75 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v76 = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v2 + (v16 + v16 * v14 + (v23 + v10) * v13), v74, v75
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v76, v68);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        pto::Shape<1, 1, 1, 16, 256> v77 = pto::Shape<1, 1, 1, 16, 256>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v78 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v79 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v16 + v16 * v14 + (v23 + v10) * v13), v77, v78
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v79, v72);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: mlp_norm_in_inline71__rv_v5
    __gm__ Tensor *mlp_norm_in_inline71__rv_v5_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *mlp_norm_in_inline71__rv_v5 =
        reinterpret_cast<__gm__ bfloat16_t *>(mlp_norm_in_inline71__rv_v5_tensor->buffer.addr) +
        mlp_norm_in_inline71__rv_v5_tensor->start_offset;

    // Unpack tensor: post_norm_partial_inline118__rv_v5
    __gm__ Tensor *post_norm_partial_inline118__rv_v5_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *post_norm_partial_inline118__rv_v5 =
        reinterpret_cast<__gm__ float *>(post_norm_partial_inline118__rv_v5_tensor->buffer.addr) +
        post_norm_partial_inline118__rv_v5_tensor->start_offset;

    // Unpack tensor: attn_proj_fp32_inline220__ssa_v7
    __gm__ Tensor *attn_proj_fp32_inline220__ssa_v7_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *attn_proj_fp32_inline220__ssa_v7 =
        reinterpret_cast<__gm__ float *>(attn_proj_fp32_inline220__ssa_v7_tensor->buffer.addr) +
        attn_proj_fp32_inline220__ssa_v7_tensor->start_offset;

    // Unpack tensor: cur__iter_v6
    __gm__ Tensor *cur__iter_v6_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *cur__iter_v6 =
        reinterpret_cast<__gm__ float *>(cur__iter_v6_tensor->buffer.addr) + cur__iter_v6_tensor->start_offset;

    // Unpack tensor: post_rms_weight__ssa_v0
    __gm__ Tensor *post_rms_weight__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *post_rms_weight__ssa_v0 =
        reinterpret_cast<__gm__ float *>(post_rms_weight__ssa_v0_tensor->buffer.addr) +
        post_rms_weight__ssa_v0_tensor->start_offset;

    // Unpack scalar: k_base_inline111__ssa_v2
    union {
        uint64_t u64;
        int64_t val;
    } k_base_inline111__ssa_v2_conv;
    k_base_inline111__ssa_v2_conv.u64 = args[5];
    int64_t k_base_inline111__ssa_v2 = k_base_inline111__ssa_v2_conv.val;

    // Unpack scalar: i__idx_v0
    union {
        uint64_t u64;
        int64_t val;
    } i__idx_v0_conv;
    i__idx_v0_conv.u64 = args[6];
    int64_t i__idx_v0 = i__idx_v0_conv.val;

    // Forward to ptoas-generated function
    residual_rms_cast_1(
        mlp_norm_in_inline71__rv_v5, post_norm_partial_inline118__rv_v5, attn_proj_fp32_inline220__ssa_v7, cur__iter_v6,
        post_rms_weight__ssa_v0, k_base_inline111__ssa_v2, i__idx_v0
    );
}
