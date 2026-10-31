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
// Kernel Function: up_proj_4

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
up_proj_4(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ float *v3, int64_t v4, int64_t v5, int64_t v6) {
    const int64_t v7 = 960;
    const int64_t v8 = 2;
    const int64_t v9 = 15;
    const int64_t v10 = 32;
    const int64_t v11 = 64;
    const int64_t v12 = 17408;
    const int64_t v13 = 1;
    const int64_t v14 = 5120;
    const int64_t v15 = 16;
    const int64_t v16 = 512;
    const int64_t v17 = 2048;
    const int64_t v18 = 32768;
    const int64_t v19 = 1536;
    const int64_t v20 = 1024;
    const int64_t v21 = 0;
    const int64_t v22 = 135168;
    const int64_t v23 = 4096;
    using T = float;

#if defined(__DAV_CUBE__)
    size_t v24 = (size_t)v11;
    size_t v25 = (size_t)v21;
    size_t v26 = (size_t)v10;
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v27 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v28 = (uint64_t)v23;
    TASSIGN(v27, v28);
    pto::Shape<1, 1, 1, 16, 64> v29 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v30 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v31 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v21 + v21 * v14 + v4 * v13), v29, v30
        );
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v27, v31);
    Tile<
        TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v32 = Tile<
            TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v20);
    uint64_t v33 = (uint64_t)v22;
    TASSIGN(v32, v33);
    int64_t v34 = (int64_t)((uint64_t)v5 + (uint64_t)v4);
    pto::Shape<1, 1, 1, 64, 1024> v35 = pto::Shape<1, 1, 1, 64, 1024>();
    pto::Stride<1114112, 1114112, 1114112, 17408, 1> v36 = pto::Stride<1114112, 1114112, 1114112, 17408, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>, pto::Layout::ND>
        v37 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
            pto::Layout::ND>(v2 + (v21 + v34 * v12 + v6 * v13), v35, v36);
    TLOAD(v32, v37);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (size_t v38 = v25; v38 < v24; v38 += v26) {
        int64_t v39 = (int64_t)v38;
        Tile<
            TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v40 = Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v15);
        uint64_t v41 = (uint64_t)v20;
        TASSIGN(v40, v41);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(v40, v27, v21, v38);
        Tile<
            TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v42 = Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v15, v20);
        uint64_t v43 = (uint64_t)v21;
        TASSIGN(v42, v43);
        TEXTRACT(v42, v32, v38, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        Tile<
            TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v44 = Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v15);
        uint64_t v45 = (uint64_t)v19;
        TASSIGN(v44, v45);
        int64_t v46 = (int64_t)((uint64_t)v39 + (uint64_t)v15);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v44, v27, v21, v46);
        Tile<
            TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v47 = Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v15, v20);
        uint64_t v48 = (uint64_t)v18;
        TASSIGN(v47, v48);
        TEXTRACT(v47, v32, v46, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v39 == v21) {
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v49 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v50 = (uint64_t)v21;
            TASSIGN(v49, v50);
            pipe_barrier(PIPE_M);
            TMATMUL(v49, v40, v42);
        } else {
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v51 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v52 = (uint64_t)v21;
            TASSIGN(v51, v52);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v51, v51, v40, v42);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        Tile<
            TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v53 = Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v20);
        uint64_t v54 = (uint64_t)v21;
        TASSIGN(v53, v54);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(v53, v53, v44, v47);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    for (size_t v55 = (size_t)v13; v55 < ((size_t)v9); v55 += (size_t)v8) {
        int64_t v56 = (int64_t)((uint64_t)((int64_t)v55) * (uint64_t)v11);
        int64_t v57 = (int64_t)((uint64_t)v56 + (uint64_t)v11);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v58 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v59 = (uint64_t)v21;
        TASSIGN(v58, v59);
        pto::Shape<1, 1, 1, 16, 64> v60 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v61 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v62 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v21 + v21 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v56) * v13), v60, v61
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v58, v62);
        Tile<
            TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v63 = Tile<
                TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v20);
        uint64_t v64 = (uint64_t)v22;
        TASSIGN(v63, v64);
        pto::Shape<1, 1, 1, 64, 1024> v65 = pto::Shape<1, 1, 1, 64, 1024>();
        pto::Stride<1114112, 1114112, 1114112, 17408, 1> v66 = pto::Stride<1114112, 1114112, 1114112, 17408, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
            pto::Layout::ND>
            v67 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
                pto::Layout::ND>(v2 + (v21 + (int64_t)((uint64_t)v34 + (uint64_t)v56) * v12 + v6 * v13), v65, v66);
        TLOAD(v63, v67);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v68 = Tile<
                TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v11);
        uint64_t v69 = (uint64_t)v17;
        TASSIGN(v68, v69);
        pto::Shape<1, 1, 1, 16, 64> v70 = pto::Shape<1, 1, 1, 16, 64>();
        pto::Stride<81920, 81920, 81920, 5120, 1> v71 = pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            v72 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + (v21 + v21 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v57) * v13), v70, v71
            );
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v68, v72);
        Tile<
            TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v73 = Tile<
                TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v11, v20);
        uint64_t v74 = (uint64_t)v23;
        TASSIGN(v73, v74);
        pto::Shape<1, 1, 1, 64, 1024> v75 = pto::Shape<1, 1, 1, 64, 1024>();
        pto::Stride<1114112, 1114112, 1114112, 17408, 1> v76 = pto::Stride<1114112, 1114112, 1114112, 17408, 1>();
        GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
            pto::Layout::ND>
            v77 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
                pto::Layout::ND>(v2 + (v21 + (int64_t)((uint64_t)v34 + (uint64_t)v57) * v12 + v6 * v13), v75, v76);
        TLOAD(v73, v77);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        for (size_t v78 = v25; v78 < v24; v78 += v26) {
            Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v79 = Tile<
                    TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            uint64_t v80 = (uint64_t)v20;
            TASSIGN(v79, v80);
            pipe_barrier(PIPE_MTE1);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            TEXTRACT(v79, v58, v21, v78);
            Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v81 = Tile<
                    TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v20);
            uint64_t v82 = (uint64_t)v21;
            TASSIGN(v81, v82);
            TEXTRACT(v81, v63, v78, v21);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v83 = Tile<
                    TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            uint64_t v84 = (uint64_t)v19;
            TASSIGN(v83, v84);
            int64_t v85 = (int64_t)((uint64_t)((int64_t)v78) + (uint64_t)v15);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
            TEXTRACT(v83, v58, v21, v85);
            Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v86 = Tile<
                    TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v20);
            uint64_t v87 = (uint64_t)v18;
            TASSIGN(v86, v87);
            TEXTRACT(v86, v63, v85, v21);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v88 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v89 = (uint64_t)v21;
            TASSIGN(v88, v89);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v88, v88, v79, v81);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v90 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v91 = (uint64_t)v21;
            TASSIGN(v90, v91);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
            TMATMUL_ACC(v90, v90, v83, v86);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        for (size_t v92 = v25; v92 < v24; v92 += v26) {
            Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v93 = Tile<
                    TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            uint64_t v94 = (uint64_t)v21;
            TASSIGN(v93, v94);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            TEXTRACT(v93, v68, v21, v92);
            Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v95 = Tile<
                    TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v20);
            uint64_t v96 = (uint64_t)v21;
            TASSIGN(v95, v96);
            TEXTRACT(v95, v73, v92, v21);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>
                v97 = Tile<
                    TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v15);
            uint64_t v98 = (uint64_t)v16;
            TASSIGN(v97, v98);
            int64_t v99 = (int64_t)((uint64_t)((int64_t)v92) + (uint64_t)v15);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            TEXTRACT(v97, v68, v21, v99);
            Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>
                v100 = Tile<
                    TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                    PadValue::Null, CompactMode::Null>(v15, v20);
            uint64_t v101 = (uint64_t)v18;
            TASSIGN(v100, v101);
            TEXTRACT(v100, v73, v99, v21);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v102 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v103 = (uint64_t)v21;
            TASSIGN(v102, v103);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v102, v102, v93, v95);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
            Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>
                v104 = Tile<
                    TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                    CompactMode::Null>(v15, v20);
            uint64_t v105 = (uint64_t)v21;
            TASSIGN(v104, v105);
            pipe_barrier(PIPE_M);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
            TMATMUL_ACC(v104, v104, v97, v100);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        };
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    Tile<
        TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v106 = Tile<
            TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v15, v11);
    uint64_t v107 = (uint64_t)v23;
    TASSIGN(v106, v107);
    pto::Shape<1, 1, 1, 16, 64> v108 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v109 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
        v110 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
            v1 + (v21 + v21 * v14 + (int64_t)((uint64_t)v4 + (uint64_t)v7) * v13), v108, v109
        );
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    TLOAD(v106, v110);
    Tile<
        TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::Null>
        v111 = Tile<
            TileType::Mat, bfloat16_t, 64, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>(v11, v20);
    uint64_t v112 = (uint64_t)v22;
    TASSIGN(v111, v112);
    pto::Shape<1, 1, 1, 64, 1024> v113 = pto::Shape<1, 1, 1, 64, 1024>();
    pto::Stride<1114112, 1114112, 1114112, 17408, 1> v114 = pto::Stride<1114112, 1114112, 1114112, 17408, 1>();
    GlobalTensor<
        bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>, pto::Layout::ND>
        v115 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 64, 1024>, pto::Stride<1114112, 1114112, 1114112, 17408, 1>,
            pto::Layout::ND>(v2 + (v21 + (int64_t)((uint64_t)v34 + (uint64_t)v7) * v12 + v6 * v13), v113, v114);
    TLOAD(v111, v115);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (size_t v116 = v25; v116 < v24; v116 += v26) {
        Tile<
            TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v117 = Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v15);
        uint64_t v118 = (uint64_t)v20;
        TASSIGN(v117, v118);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v117, v106, v21, v116);
        Tile<
            TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v119 = Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v15, v20);
        uint64_t v120 = (uint64_t)v21;
        TASSIGN(v119, v120);
        TEXTRACT(v119, v111, v116, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        Tile<
            TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
            CompactMode::Null>
            v121 = Tile<
                TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null,
                CompactMode::Null>(v15, v15);
        uint64_t v122 = (uint64_t)v19;
        TASSIGN(v121, v122);
        int64_t v123 = (int64_t)((uint64_t)((int64_t)v116) + (uint64_t)v15);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v121, v106, v21, v123);
        Tile<
            TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null,
            CompactMode::Null>
            v124 = Tile<
                TileType::Right, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512,
                PadValue::Null, CompactMode::Null>(v15, v20);
        uint64_t v125 = (uint64_t)v18;
        TASSIGN(v124, v125);
        TEXTRACT(v124, v111, v123, v21);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        Tile<
            TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v126 = Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v20);
        uint64_t v127 = (uint64_t)v21;
        TASSIGN(v126, v127);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v126, v126, v117, v119);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        Tile<
            TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>
            v128 = Tile<
                TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
                CompactMode::Null>(v15, v20);
        uint64_t v129 = (uint64_t)v21;
        TASSIGN(v128, v129);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
        TMATMUL_ACC(v128, v128, v121, v124);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    Tile<
        TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
        CompactMode::Null>
        v130 = Tile<
            TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>(v15, v20);
    uint64_t v131 = (uint64_t)v21;
    TASSIGN(v130, v131);
    pto::Shape<1, 1, 1, 16, 1024> v132 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<278528, 278528, 278528, 17408, 1> v133 = pto::Stride<278528, 278528, 278528, 17408, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>
        v134 = GlobalTensor<
            float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>(
            v3 + (v21 + v21 * v12), v132, v133
        );
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE<
        Tile<
            TileType::Acc, float, 16, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null,
            CompactMode::Null>,
        GlobalTensor<
            float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>,
        AtomicType::AtomicNone>(v134, v130);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
#endif  // __DAV_CUBE__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: mlp_norm_in_inline71__rv_v14
    __gm__ Tensor *mlp_norm_in_inline71__rv_v14_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *mlp_norm_in_inline71__rv_v14 =
        reinterpret_cast<__gm__ bfloat16_t *>(mlp_norm_in_inline71__rv_v14_tensor->buffer.addr) +
        mlp_norm_in_inline71__rv_v14_tensor->start_offset;

    // Unpack tensor: w_up__ssa_v0
    __gm__ Tensor *w_up__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *w_up__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(w_up__ssa_v0_tensor->buffer.addr) + w_up__ssa_v0_tensor->start_offset;

    // Unpack tensor: up_acc_all_inline303__iter_v11
    __gm__ Tensor *up_acc_all_inline303__iter_v11_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *up_acc_all_inline303__iter_v11 =
        reinterpret_cast<__gm__ float *>(up_acc_all_inline303__iter_v11_tensor->buffer.addr) +
        up_acc_all_inline303__iter_v11_tensor->start_offset;

    // Unpack scalar: k0_inline113__ssa_v7
    union {
        uint64_t u64;
        int64_t val;
    } k0_inline113__ssa_v7_conv;
    k0_inline113__ssa_v7_conv.u64 = args[3];
    int64_t k0_inline113__ssa_v7 = k0_inline113__ssa_v7_conv.val;

    // Unpack scalar: layer_hidden_base_inline151__ssa_v0
    union {
        uint64_t u64;
        int64_t val;
    } layer_hidden_base_inline151__ssa_v0_conv;
    layer_hidden_base_inline151__ssa_v0_conv.u64 = args[4];
    int64_t layer_hidden_base_inline151__ssa_v0 = layer_hidden_base_inline151__ssa_v0_conv.val;

    // Unpack scalar: n0_inline122__ssa_v6
    union {
        uint64_t u64;
        int64_t val;
    } n0_inline122__ssa_v6_conv;
    n0_inline122__ssa_v6_conv.u64 = args[5];
    int64_t n0_inline122__ssa_v6 = n0_inline122__ssa_v6_conv.val;

    // Forward to ptoas-generated function
    up_proj_4(
        mlp_norm_in_inline71__rv_v14, w_up__ssa_v0, up_acc_all_inline303__iter_v11, k0_inline113__ssa_v7,
        layer_hidden_base_inline151__ssa_v0, n0_inline122__ssa_v6
    );
}
