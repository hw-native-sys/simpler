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
// Kernel Function: out_proj

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

static __aicore__ void
out_proj(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int64_t v4, int64_t v5, int64_t v6) {
    const int64_t v7 = 960;
    const int64_t v8 = 2;
    const int64_t v9 = 15;
    const int64_t v10 = 32;
    const int64_t v11 = 512;
    const int64_t v12 = 64;
    const int64_t v13 = 1;
    const int64_t v14 = 5120;
    const int64_t v15 = 16;
    const int64_t v16 = 1024;
    const int64_t v17 = 32768;
    const int64_t v18 = 3072;
    const int64_t v19 = 2048;
    const int64_t v20 = 0;
    const int64_t v21 = 69632;
    const int64_t v22 = 4096;
    using T = float;

#if defined(__DAV_CUBE__)
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v23 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v12);
    uint64_t v24 = (uint64_t)v22;
    TASSIGN(v23, v24);
    pto::Shape<1, 1, 1, 16, 64> v25 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v26 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v27 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v20 + v20 * v14 + v4 * v13), v25, v26
        );
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TLOAD(v23, v27);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<
        TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v28 = Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v11);
    uint64_t v29 = (uint64_t)v21;
    TASSIGN(v28, v29);
    int64_t v30 = (int64_t)((uint64_t)v5 + (uint64_t)v4);
    pto::Shape<1, 1, 1, 64, 512> v31 = pto::Shape<1, 1, 1, 64, 512>();
    pto::Stride<327680, 327680, 327680, 5120, 1> v32 = pto::Stride<327680, 327680, 327680, 5120, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
        v33 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>(
            v2 + (v20 + v30 * v14 + v6 * v13), v31, v32
        );
    TLOAD(v28, v33);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v34 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v10);
    uint64_t v35 = (uint64_t)v19;
    TASSIGN(v34, v35);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(v34, v23, v20, v20);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v36 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v10, v11);
    uint64_t v37 = (uint64_t)v20;
    TASSIGN(v36, v37);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(v36, v28, v20, v20);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v38 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v10);
    uint64_t v39 = (uint64_t)v18;
    TASSIGN(v38, v39);
    TEXTRACT(v38, v23, v20, v10);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v40 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v10, v11);
    uint64_t v41 = (uint64_t)v17;
    TASSIGN(v40, v41);
    TEXTRACT(v40, v28, v10, v20);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v42 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v43 = (uint64_t)v20;
    TASSIGN(v42, v43);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v42, v34, v36);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v44 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v45 = (uint64_t)v20;
    TASSIGN(v44, v45);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v44, v44, v38, v40);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (size_t v46 = (size_t)v13; v46 < ((size_t)v9); v46 += (size_t)v8) {
        int64_t v47 = (int64_t)((uint64_t)((int64_t)v46) * (uint64_t)v12);
        int64_t v48 = (int64_t)((uint64_t)v47 + (uint64_t)v12);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v49 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        uint64_t v50 = (uint64_t)v20;
        TASSIGN(v49, v50);
        pto::Shape<1, 1, 1, 16, 64> v51 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v52 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v53 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v20 + v20 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v47) * v13), v51, v52
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v49, v53);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v54 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v11);
        uint64_t v55 = (uint64_t)v21;
        TASSIGN(v54, v55);
        pto::Shape<1, 1, 1, 64, 512> v56 = pto::Shape<1, 1, 1, 64, 512>();
        pto::Stride<327680, 327680, 327680, 5120, 1> v57 = pto::Stride<327680, 327680, 327680, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
            v58 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>,
                pto::Layout::ND>(v2 + (v20 + (int64_t)((uint64_t)v30 + (uint64_t)v47) * v14 + v6 * v13), v56, v57);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v54, v58);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v59 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v12);
        uint64_t v60 = (uint64_t)v19;
        TASSIGN(v59, v60);
        pto::Shape<1, 1, 1, 16, 64> v61 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v62 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v63 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v20 + v20 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v48) * v13), v61, v62
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(v59, v63);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v64 = Tile<
                TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v12, v11);
        uint64_t v65 = (uint64_t)v22;
        TASSIGN(v64, v65);
        pto::Shape<1, 1, 1, 64, 512> v66 = pto::Shape<1, 1, 1, 64, 512>();
        pto::Stride<327680, 327680, 327680, 5120, 1> v67 = pto::Stride<327680, 327680, 327680, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
            v68 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>,
                pto::Layout::ND>(v2 + (v20 + (int64_t)((uint64_t)v30 + (uint64_t)v48) * v14 + v6 * v13), v66, v67);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        TLOAD(v64, v68);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v69 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v70 = (uint64_t)v19;
        TASSIGN(v69, v70);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v69, v49, v20, v20);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v71 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        uint64_t v72 = (uint64_t)v20;
        TASSIGN(v71, v72);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v71, v54, v20, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v74 = (uint64_t)v18;
        TASSIGN(v73, v74);
        TEXTRACT(v73, v49, v20, v10);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v75 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        uint64_t v76 = (uint64_t)v17;
        TASSIGN(v75, v76);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v75, v54, v10, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v77 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v78 = (uint64_t)v20;
        TASSIGN(v77, v78);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v77, v77, v69, v71);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v79 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v80 = (uint64_t)v20;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v79, v79, v73, v75);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v81 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v82 = (uint64_t)v20;
        TASSIGN(v81, v82);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v81, v59, v20, v20);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v83 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        uint64_t v84 = (uint64_t)v20;
        TASSIGN(v83, v84);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        TEXTRACT(v83, v64, v20, v20);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v85 = Tile<
                TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v10);
        uint64_t v86 = (uint64_t)v16;
        TASSIGN(v85, v86);
        TEXTRACT(v85, v59, v20, v10);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v87 = Tile<
                TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
                CompactMode::Null>(v10, v11);
        uint64_t v88 = (uint64_t)v17;
        TASSIGN(v87, v88);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        TEXTRACT(v87, v64, v10, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v89 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v90 = (uint64_t)v20;
        TASSIGN(v89, v90);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v89, v89, v81, v83);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v91 = Tile<
                TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v92 = (uint64_t)v20;
        TASSIGN(v91, v92);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v91, v91, v85, v87);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v93 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v12);
    uint64_t v94 = (uint64_t)v22;
    TASSIGN(v93, v94);
    pto::Shape<1, 1, 1, 16, 64> v95 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v96 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v97 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v20 + v20 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v7) * v13), v95, v96
        );
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    TLOAD(v93, v97);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    Tile<
        TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v98 = Tile<
            TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v12, v11);
    uint64_t v99 = (uint64_t)v21;
    TASSIGN(v98, v99);
    pto::Shape<1, 1, 1, 64, 512> v100 = pto::Shape<1, 1, 1, 64, 512>();
    pto::Stride<327680, 327680, 327680, 5120, 1> v101 = pto::Stride<327680, 327680, 327680, 5120, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>
        v102 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<327680, 327680, 327680, 5120, 1>, pto::Layout::ND>(
            v2 + (v20 + (int64_t)((uint64_t)v30 + (uint64_t)v7) * v14 + v6 * v13), v100, v101
        );
    TLOAD(v98, v102);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v103 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v10);
    uint64_t v104 = (uint64_t)v19;
    TASSIGN(v103, v104);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
    TEXTRACT(v103, v93, v20, v20);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v105 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v10, v11);
    uint64_t v106 = (uint64_t)v20;
    TASSIGN(v105, v106);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    TEXTRACT(v105, v98, v20, v20);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    Tile<
        TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v107 = Tile<
            TileType::Left, bfloat16_t, 16, 32, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v10);
    uint64_t v108 = (uint64_t)v18;
    TASSIGN(v107, v108);
    TEXTRACT(v107, v93, v20, v10);
    Tile<
        TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
        CompactMode::Null>
        v109 = Tile<
            TileType::Right, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>(v10, v11);
    uint64_t v110 = (uint64_t)v17;
    TASSIGN(v109, v110);
    TEXTRACT(v109, v98, v10, v20);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v111 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v112 = (uint64_t)v20;
    TASSIGN(v111, v112);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v111, v111, v103, v105);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v113 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v114 = (uint64_t)v20;
    TASSIGN(v113, v114);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    TMATMUL_ACC(v113, v113, v107, v109);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    Tile<
        TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v115 = Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v116 = (uint64_t)v20;
    TASSIGN(v115, v116);
    pto::Shape<1, 1, 1, 16, 512> v117 = pto::Shape<1, 1, 1, 16, 512>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v118 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v119 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v3 + (v20 + v20 * v14), v117, v118
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE<
        Tile<
            TileType::Acc, float, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>,
        AtomicType::AtomicNone>(v119, v115);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: attn_out_inline282__ssa_v4
    __gm__ Tensor *attn_out_inline282__ssa_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *attn_out_inline282__ssa_v4 =
        reinterpret_cast<__gm__ bfloat16_t *>(attn_out_inline282__ssa_v4_tensor->buffer.addr) +
        attn_out_inline282__ssa_v4_tensor->start_offset;

    // Unpack tensor: wo__ssa_v0
    __gm__ Tensor *wo__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *wo__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(wo__ssa_v0_tensor->buffer.addr) + wo__ssa_v0_tensor->start_offset;

    // Unpack tensor: attn_proj_fp32_inline220__iter_v4
    __gm__ Tensor *attn_proj_fp32_inline220__iter_v4_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *attn_proj_fp32_inline220__iter_v4 =
        reinterpret_cast<__gm__ float *>(attn_proj_fp32_inline220__iter_v4_tensor->buffer.addr) +
        attn_proj_fp32_inline220__iter_v4_tensor->start_offset;

    // Unpack scalar: k_op_inline266__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } k_op_inline266__ssa_v0_conv;
    k_op_inline266__ssa_v0_conv.u64 = args[3];
    int64_t k_op_inline266__ssa_v0 = k_op_inline266__ssa_v0_conv.val;

    // Unpack scalar: layer_hidden_base_inline151__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } layer_hidden_base_inline151__ssa_v0_conv;
    layer_hidden_base_inline151__ssa_v0_conv.u64 = args[4];
    int64_t layer_hidden_base_inline151__ssa_v0 = layer_hidden_base_inline151__ssa_v0_conv.val;

    // Unpack scalar: n_op_inline64__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } n_op_inline64__ssa_v0_conv;
    n_op_inline64__ssa_v0_conv.u64 = args[5];
    int64_t n_op_inline64__ssa_v0 = n_op_inline64__ssa_v0_conv.val;

    // Forward to ptoas-generated function
    out_proj(
        attn_out_inline282__ssa_v4, wo__ssa_v0, attn_proj_fp32_inline220__iter_v4, k_op_inline266__ssa_v0,
        layer_hidden_base_inline151__ssa_v0, n_op_inline64__ssa_v0
    );
}
