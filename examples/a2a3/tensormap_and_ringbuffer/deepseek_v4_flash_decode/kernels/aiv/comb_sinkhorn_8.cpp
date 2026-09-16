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
// Kernel Function: comb_sinkhorn_8

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
#include "intrinsic.h"

#if defined(__CPU_SIM)
// PTOAS v0.50+ emits cache_line_t::ENTIRE_DATA_CACHE / SINGLE_CACHE_LINE as
// scoped identifiers, but the pto-isa cpu_stub.hpp defines them as bare macros
// (#define ENTIRE_DATA_CACHE 0) — which breaks cache_line_t::ENTIRE_DATA_CACHE
// into cache_line_t::0. Undefine the macros and provide proper namespace-scoped
// constexpr constants. The same headers also #define dcci/dsb as macros that
// would expand our own inlines, so undefine + redefine all of them here.
#include <atomic>

// Forward-declare the overloads so the undefs below don't break chained includes.
namespace pypto_sim_detail {
template <typename... Args>
static inline void sim_dcci(Args...);  // defined after the undefs
static inline void sim_dsb(int kind);  // ditto
}  // namespace pypto_sim_detail

// Undefine conflicting macros from pto-isa cpu_stub.hpp / inner_kernel.h
// so our namespace-scoped constants and inline functions are used instead.
#undef ENTIRE_DATA_CACHE
#undef SINGLE_CACHE_LINE
#undef DSB_DDR
#undef dcci
#undef dsb
#undef CACHELINE_OUT

namespace cache_line_t {
constexpr int ENTIRE_DATA_CACHE = 0;
constexpr int SINGLE_CACHE_LINE = 0;
constexpr int CACHELINE_OUT = 0;
}  // namespace cache_line_t
typedef int mem_dsb_t;
#define DSB_DDR 0

static inline void dcci(...) { std::atomic_thread_fence(std::memory_order_seq_cst); }
static inline void dsb(mem_dsb_t) { std::atomic_thread_fence(std::memory_order_seq_cst); }
#endif  // __CPU_SIM

using namespace pto;

// --- ptoas-generated code ---

template <typename TensorT>
static __aicore__ inline auto PTOAS__GLOBAL_TENSOR_DATA(TensorT &tensor) -> decltype(tensor.data()) {
    return tensor.data();
}

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

template <typename Ptr>
static __aicore__ inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
    dcci((__gm__ void *)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

static __aicore__ void comb_sinkhorn_8(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, float v5, int64_t v6, int32_t v7, int32_t v8
) {
    unsigned v9 = 20;
    unsigned v10 = 16;
    unsigned v11 = 12;
    unsigned v12 = 8;
    const int64_t v13 = 20;
    const int64_t v14 = 16;
    const int64_t v15 = 12;
    const int64_t v16 = 32;
    const int64_t v17 = 2;
    const int64_t v18 = 18;
    const float v19 = 9.99999997E-7f;
    const int64_t v20 = 4;
    const int64_t v21 = 8;
    const int64_t v22 = 1;
    const int64_t v23 = 1280;
    const int64_t v24 = 0;
    const int64_t v25 = 1024;
    const int64_t v26 = 768;
    const int64_t v27 = 512;
    const int64_t v28 = 256;
    const int64_t v29 = 4480;
    const int64_t v30 = 4224;
    const int64_t v31 = 3968;
    const int64_t v32 = 2112;
    const int64_t v33 = 2080;
    const int64_t v34 = 2048;
    const int64_t v35 = 1536;
    const int64_t v36 = 3712;
    const int64_t v37 = 3456;
    const int64_t v38 = 2688;
    const int64_t v39 = 1792;
    const int64_t v40 = 3200;
    const int64_t v41 = 2944;
    const int64_t v42 = 2432;
    const int64_t v43 = 2176;
    const int64_t v44 = 2144;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %inv_rms_inline15360__ssa_v1_view
    int64_t v45 = v6 * v22;
    // pto: %inv_rms_inline15360__ssa_v1_view
    int64_t v46 = v22 * v45;
    // pto: %inv_rms_inline15360__ssa_v1_view
    pto::Shape<1, 1, 1, -1, -1> v47 = pto::Shape<1, 1, 1, -1, -1>(v22, v22, v22, v6, v22);
    // pto: %inv_rms_inline15360__ssa_v1_view
    pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v46, v46, v45, v22, v6);
    // pto: %inv_rms_inline15360__ssa_v1_view
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v49 =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v1, v47, v48
        );
    // pto: %ob_inline15364__ssa_v0, %56
    int64_t v50 = (int64_t)((uint64_t)((int64_t)v7) * (uint64_t)v21);
    // pto: %inv_col_t_inline15416__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v51 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %inv_col_t_inline15416__ssa_v0
    uint64_t v52 = (uint64_t)v44;
    TASSIGN(v51, v52);
    // pto: %57
    int64_t v53 = v50 < v24 ? v24 : v50;
    // pto: %inv_rms_inline15360__ssa_v1_pview
    __gm__ float *v54 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %inv_rms_inline15360__ssa_v1_pview
    int64_t v55 = v21 * v22;
    // pto: %inv_rms_inline15360__ssa_v1_pview
    int64_t v56 = v22 * v55;
    // pto: %inv_rms_inline15360__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 1> v57 = pto::Shape<1, 1, 1, 8, 1>(v22, v22, v22, v21, v22);
    // pto: %inv_rms_inline15360__ssa_v1_pview
    pto::Stride<-1, -1, -1, -1, -1> v58 = pto::Stride<-1, -1, -1, -1, -1>(v22 * v56, v56, v55, v22, v6);
    // pto: %inv_rms_inline15360__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v59 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v54 + ((v24 + v53 * v22) + v24 * v6), v57, v58
        );
    TLOAD(v51, v59);
    // pto: %mix_g0_inline15423__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v60 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %mix_g0_inline15423__ssa_v0
    uint64_t v61 = (uint64_t)v43;
    TASSIGN(v60, v61);
    // pto: %mixes_raw_inline15383__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 4> v62 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %mixes_raw_inline15383__ssa_v1_pview
    pto::Stride<256, 256, 256, 32, 1> v63 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %mixes_raw_inline15383__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v64 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v2 + (v21 + v53 * v16), v62, v63
        );
    TLOAD(v60, v64);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %mix_g1_inline15375__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v65 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %mix_g1_inline15375__ssa_v0
    uint64_t v66 = (uint64_t)v42;
    TASSIGN(v65, v66);
    // pto: %60
    pto::Shape<1, 1, 1, 8, 4> v67 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %60
    pto::Stride<256, 256, 256, 32, 1> v68 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %60
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v69 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v2 + (v15 + v53 * v16), v67, v68
        );
    TLOAD(v65, v69);
    // pto: %mix_g2_inline15361__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v70 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %mix_g2_inline15361__ssa_v0
    uint64_t v71 = (uint64_t)v41;
    TASSIGN(v70, v71);
    // pto: %62
    pto::Shape<1, 1, 1, 8, 4> v72 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %62
    pto::Stride<256, 256, 256, 32, 1> v73 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %62
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v74 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v2 + (v14 + v53 * v16), v72, v73
        );
    TLOAD(v70, v74);
    // pto: %mix_g3_inline15359__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v75 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %mix_g3_inline15359__ssa_v0
    uint64_t v76 = (uint64_t)v40;
    TASSIGN(v75, v76);
    // pto: %64
    pto::Shape<1, 1, 1, 8, 4> v77 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %64
    pto::Stride<256, 256, 256, 32, 1> v78 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %64
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v79 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v2 + (v13 + v53 * v16), v77, v78
        );
    TLOAD(v75, v79);
    // pto: %cb0_inline15357__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v80 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v20);
    // pto: %cb0_inline15357__ssa_v0
    uint64_t v81 = (uint64_t)v39;
    TASSIGN(v80, v81);
    // pto: %hc_base_2d_inline15420__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4> v82 = pto::Shape<1, 1, 1, 1, 4>();
    // pto: %hc_base_2d_inline15420__ssa_v0_pview
    pto::Stride<24, 24, 24, 24, 1> v83 = pto::Stride<24, 24, 24, 24, 1>();
    // pto: %hc_base_2d_inline15420__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v84 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(
            v3 + v12, v82, v83
        );
    TLOAD(v80, v84);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %cb1_inline15356__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v85 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v20);
    // pto: %cb1_inline15356__ssa_v0
    uint64_t v86 = (uint64_t)v38;
    TASSIGN(v85, v86);
    // pto: %65
    pto::Shape<1, 1, 1, 1, 4> v87 = pto::Shape<1, 1, 1, 1, 4>();
    // pto: %65
    pto::Stride<24, 24, 24, 24, 1> v88 = pto::Stride<24, 24, 24, 24, 1>();
    // pto: %65
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v89 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(
            v3 + v11, v87, v88
        );
    TLOAD(v85, v89);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %cb2_inline15426__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v90 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v20);
    // pto: %cb2_inline15426__ssa_v0
    uint64_t v91 = (uint64_t)v37;
    TASSIGN(v90, v91);
    // pto: %66
    pto::Shape<1, 1, 1, 1, 4> v92 = pto::Shape<1, 1, 1, 1, 4>();
    // pto: %66
    pto::Stride<24, 24, 24, 24, 1> v93 = pto::Stride<24, 24, 24, 24, 1>();
    // pto: %66
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v94 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(
            v3 + v10, v92, v93
        );
    TLOAD(v90, v94);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %cb3_inline15377__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v95 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v20);
    // pto: %cb3_inline15377__ssa_v0
    uint64_t v96 = (uint64_t)v36;
    TASSIGN(v95, v96);
    // pto: %67
    pto::Shape<1, 1, 1, 1, 4> v97 = pto::Shape<1, 1, 1, 1, 4>();
    // pto: %67
    pto::Stride<24, 24, 24, 24, 1> v98 = pto::Stride<24, 24, 24, 24, 1>();
    // pto: %67
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v99 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(
            v3 + v9, v97, v98
        );
    TLOAD(v95, v99);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %t__tmp_v2017
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v100 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2017
    uint64_t v101 = (uint64_t)v35;
    TASSIGN(v100, v101);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v100, v60, v51);
    // pto: %t__tmp_v2018
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v102 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2018
    uint64_t v103 = (uint64_t)v35;
    TASSIGN(v102, v103);
    pipe_barrier(PIPE_V);
    TMULS(v102, v100, v5);
    // pto: %t__tmp_v2019
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v104 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2019
    uint64_t v105 = (uint64_t)v43;
    TASSIGN(v104, v105);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCOLEXPAND(v104, v80);
    // pto: %row0_inline15414__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v106 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %row0_inline15414__ssa_v0
    uint64_t v107 = (uint64_t)v43;
    TASSIGN(v106, v107);
    pipe_barrier(PIPE_V);
    TADD(v106, v102, v104);
    // pto: %t__tmp_v2020
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v108 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2020
    uint64_t v109 = (uint64_t)v35;
    TASSIGN(v108, v109);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v108, v65, v51);
    // pto: %t__tmp_v2021
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v110 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2021
    uint64_t v111 = (uint64_t)v35;
    TASSIGN(v110, v111);
    pipe_barrier(PIPE_V);
    TMULS(v110, v108, v5);
    // pto: %t__tmp_v2022
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v112 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2022
    uint64_t v113 = (uint64_t)v42;
    TASSIGN(v112, v113);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TCOLEXPAND(v112, v85);
    // pto: %row1_inline15355__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v114 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %row1_inline15355__ssa_v0
    uint64_t v115 = (uint64_t)v42;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    TADD(v114, v110, v112);
    // pto: %t__tmp_v2023
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v116 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2023
    uint64_t v117 = (uint64_t)v35;
    TASSIGN(v116, v117);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v116, v70, v51);
    // pto: %t__tmp_v2024
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v118 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2024
    uint64_t v119 = (uint64_t)v35;
    TASSIGN(v118, v119);
    pipe_barrier(PIPE_V);
    TMULS(v118, v116, v5);
    // pto: %t__tmp_v2025
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v120 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2025
    uint64_t v121 = (uint64_t)v41;
    TASSIGN(v120, v121);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TCOLEXPAND(v120, v90);
    // pto: %row2_inline15428__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v122 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %row2_inline15428__ssa_v0
    uint64_t v123 = (uint64_t)v41;
    TASSIGN(v122, v123);
    pipe_barrier(PIPE_V);
    TADD(v122, v118, v120);
    // pto: %t__tmp_v2026
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v124 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2026
    uint64_t v125 = (uint64_t)v35;
    TASSIGN(v124, v125);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v124, v75, v51);
    // pto: %t__tmp_v2027
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v126 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2027
    uint64_t v127 = (uint64_t)v35;
    TASSIGN(v126, v127);
    pipe_barrier(PIPE_V);
    TMULS(v126, v124, v5);
    // pto: %t__tmp_v2028
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v128 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %t__tmp_v2028
    uint64_t v129 = (uint64_t)v40;
    TASSIGN(v128, v129);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TCOLEXPAND(v128, v95);
    // pto: %row3_inline15424__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v130 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v20);
    // pto: %row3_inline15424__ssa_v0
    uint64_t v131 = (uint64_t)v40;
    TASSIGN(v130, v131);
    pipe_barrier(PIPE_V);
    TADD(v130, v126, v128);
    // pto: %row0_p_inline15427__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v132 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row0_p_inline15427__ssa_v0
    uint64_t v133 = (uint64_t)v43;
    TASSIGN(v132, v133);
    TFILLPAD(v132, v106);
    // pto: %row1_p_inline15415__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v134 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row1_p_inline15415__ssa_v0
    uint64_t v135 = (uint64_t)v42;
    TASSIGN(v134, v135);
    TFILLPAD(v134, v114);
    // pto: %row2_p_inline15429__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v136 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row2_p_inline15429__ssa_v0
    uint64_t v137 = (uint64_t)v41;
    TASSIGN(v136, v137);
    TFILLPAD(v136, v122);
    // pto: %row3_p_inline15430__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v138 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row3_p_inline15430__ssa_v0
    uint64_t v139 = (uint64_t)v40;
    TASSIGN(v138, v139);
    pipe_barrier(PIPE_V);
    TFILLPAD(v138, v130);
    // pto: %row_max_tmp_inline15432__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v140 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v21);
    // pto: %row_max_tmp_inline15432__ssa_v0
    uint64_t v141 = (uint64_t)v35;
    TASSIGN(v140, v141);
    // pto: %row_sum_tmp_inline15434__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v142 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v21);
    // pto: %row_sum_tmp_inline15434__ssa_v0
    uint64_t v143 = (uint64_t)v39;
    TASSIGN(v142, v143);
    // pto: %row0_max_inline15435__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v144 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row0_max_inline15435__ssa_v0
    uint64_t v145 = (uint64_t)v44;
    TASSIGN(v144, v145);
    TROWMAX(v144, v132, v140);
    // pto: %row1_max_inline15437__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v146 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row1_max_inline15437__ssa_v0
    uint64_t v147 = (uint64_t)v34;
    TASSIGN(v146, v147);
    pipe_barrier(PIPE_V);
    TROWMAX(v146, v134, v140);
    // pto: %row2_max_inline15455__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v148 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row2_max_inline15455__ssa_v0
    uint64_t v149 = (uint64_t)v33;
    TASSIGN(v148, v149);
    pipe_barrier(PIPE_V);
    TROWMAX(v148, v136, v140);
    // pto: %row3_max_inline15438__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v150 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row3_max_inline15438__ssa_v0
    uint64_t v151 = (uint64_t)v32;
    TASSIGN(v150, v151);
    pipe_barrier(PIPE_V);
    TROWMAX(v150, v138, v140);
    // pto: %t__tmp_v2029
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v152 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2029
    uint64_t v153 = (uint64_t)v43;
    TASSIGN(v152, v153);
    TROWEXPANDSUB(v152, v132, v144);
    // pto: %row0_exp_inline15378__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v154 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row0_exp_inline15378__ssa_v0
    uint64_t v155 = (uint64_t)v43;
    TASSIGN(v154, v155);
    pipe_barrier(PIPE_V);
    TEXP(v154, v152);
    // pto: %t__tmp_v2030
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v156 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2030
    uint64_t v157 = (uint64_t)v42;
    TASSIGN(v156, v157);
    TROWEXPANDSUB(v156, v134, v146);
    // pto: %row1_exp_inline15441__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v158 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row1_exp_inline15441__ssa_v0
    uint64_t v159 = (uint64_t)v42;
    TASSIGN(v158, v159);
    pipe_barrier(PIPE_V);
    TEXP(v158, v156);
    // pto: %t__tmp_v2031
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v160 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2031
    uint64_t v161 = (uint64_t)v41;
    TASSIGN(v160, v161);
    TROWEXPANDSUB(v160, v136, v148);
    // pto: %row2_exp_inline15442__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v162 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row2_exp_inline15442__ssa_v0
    uint64_t v163 = (uint64_t)v41;
    TASSIGN(v162, v163);
    pipe_barrier(PIPE_V);
    TEXP(v162, v160);
    // pto: %t__tmp_v2032
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v164 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2032
    uint64_t v165 = (uint64_t)v40;
    TASSIGN(v164, v165);
    TROWEXPANDSUB(v164, v138, v150);
    // pto: %row3_exp_inline15390__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v166 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row3_exp_inline15390__ssa_v0
    uint64_t v167 = (uint64_t)v40;
    TASSIGN(v166, v167);
    pipe_barrier(PIPE_V);
    TEXP(v166, v164);
    // pto: %row0_sum_inline15439__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v168 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row0_sum_inline15439__ssa_v0
    uint64_t v169 = (uint64_t)v44;
    TASSIGN(v168, v169);
    TROWSUM(v168, v154, v142);
    // pto: %row1_sum_inline15444__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v170 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row1_sum_inline15444__ssa_v0
    uint64_t v171 = (uint64_t)v34;
    TASSIGN(v170, v171);
    pipe_barrier(PIPE_V);
    TROWSUM(v170, v158, v142);
    // pto: %row2_sum_inline15460__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v172 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row2_sum_inline15460__ssa_v0
    uint64_t v173 = (uint64_t)v33;
    TASSIGN(v172, v173);
    pipe_barrier(PIPE_V);
    TROWSUM(v172, v162, v142);
    // pto: %row3_sum_inline15445__ssa_v0
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v174 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %row3_sum_inline15445__ssa_v0
    uint64_t v175 = (uint64_t)v32;
    TASSIGN(v174, v175);
    pipe_barrier(PIPE_V);
    TROWSUM(v174, v166, v142);
    // pto: %t__tmp_v2033
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v176 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2033
    uint64_t v177 = (uint64_t)v43;
    TASSIGN(v176, v177);
    TROWEXPANDDIV(v176, v154, v168);
    // pto: %row0_soft_inline15409__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v178 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row0_soft_inline15409__ssa_v0
    uint64_t v179 = (uint64_t)v43;
    TASSIGN(v178, v179);
    pipe_barrier(PIPE_V);
    TADDS(v178, v176, v19);
    // pto: %t__tmp_v2034
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v180 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2034
    uint64_t v181 = (uint64_t)v42;
    TASSIGN(v180, v181);
    TROWEXPANDDIV(v180, v158, v170);
    // pto: %row1_soft_inline15433__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v182 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row1_soft_inline15433__ssa_v0
    uint64_t v183 = (uint64_t)v42;
    TASSIGN(v182, v183);
    pipe_barrier(PIPE_V);
    TADDS(v182, v180, v19);
    // pto: %t__tmp_v2035
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v184 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2035
    uint64_t v185 = (uint64_t)v41;
    TASSIGN(v184, v185);
    TROWEXPANDDIV(v184, v162, v172);
    // pto: %row2_soft_inline15446__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v186 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row2_soft_inline15446__ssa_v0
    uint64_t v187 = (uint64_t)v41;
    TASSIGN(v186, v187);
    pipe_barrier(PIPE_V);
    TADDS(v186, v184, v19);
    // pto: %t__tmp_v2036
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v188 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2036
    uint64_t v189 = (uint64_t)v40;
    TASSIGN(v188, v189);
    TROWEXPANDDIV(v188, v166, v174);
    // pto: %row3_soft_inline15372__ssa_v0
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>
        v190 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min,
            CompactMode::Null>(v21, v21);
    // pto: %row3_soft_inline15372__ssa_v0
    uint64_t v191 = (uint64_t)v40;
    TASSIGN(v190, v191);
    pipe_barrier(PIPE_V);
    TADDS(v190, v188, v19);
    v178.SetValidShape(v21, v20);
    v182.SetValidShape(v21, v20);
    v186.SetValidShape(v21, v20);
    v190.SetValidShape(v21, v20);
    // pto: %row0_eff_inline15396__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v192 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row0_eff_inline15396__ssa_v0
    uint64_t v193 = (uint64_t)v43;
    TASSIGN(v192, v193);
    TFILLPAD(v192, v178);
    // pto: %row1_eff_inline15449__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v194 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row1_eff_inline15449__ssa_v0
    uint64_t v195 = (uint64_t)v42;
    TASSIGN(v194, v195);
    TFILLPAD(v194, v182);
    // pto: %row2_eff_inline15450__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v196 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row2_eff_inline15450__ssa_v0
    uint64_t v197 = (uint64_t)v41;
    TASSIGN(v196, v197);
    TFILLPAD(v196, v186);
    // pto: %row3_eff_inline15451__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v198 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row3_eff_inline15451__ssa_v0
    uint64_t v199 = (uint64_t)v40;
    TASSIGN(v198, v199);
    pipe_barrier(PIPE_V);
    TFILLPAD(v198, v190);
    // pto: %row_sum_tmp_iter_inline15436__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v200 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v21);
    // pto: %row_sum_tmp_iter_inline15436__ssa_v0
    uint64_t v201 = (uint64_t)v35;
    TASSIGN(v200, v201);
    // pto: %t__tmp_v2037
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v202 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2037
    uint64_t v203 = (uint64_t)v39;
    TASSIGN(v202, v203);
    TADD(v202, v192, v194);
    // pto: %t__tmp_v2038
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v204 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %t__tmp_v2038
    uint64_t v205 = (uint64_t)v38;
    TASSIGN(v204, v205);
    pipe_barrier(PIPE_V);
    TADD(v204, v196, v198);
    // pto: %col_sum_inline15452__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v206 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %col_sum_inline15452__ssa_v0
    uint64_t v207 = (uint64_t)v39;
    TASSIGN(v206, v207);
    pipe_barrier(PIPE_V);
    TADD(v206, v202, v204);
    // pto: %col_sum_v1_inline15454__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v208 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %col_sum_v1_inline15454__ssa_v0
    uint64_t v209 = (uint64_t)v39;
    TASSIGN(v208, v209);
    pipe_barrier(PIPE_V);
    TADDS(v208, v206, v19);
    // pto: %row0_cur_inline15458__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v210 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row0_cur_inline15458__ssa_v0
    uint64_t v211 = (uint64_t)v43;
    TASSIGN(v210, v211);
    pipe_barrier(PIPE_V);
    TDIV(v210, v192, v208);
    // pto: %row1_cur_inline15459__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v212 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row1_cur_inline15459__ssa_v0
    uint64_t v213 = (uint64_t)v42;
    TASSIGN(v212, v213);
    TDIV(v212, v194, v208);
    // pto: %row2_cur_inline15470__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v214 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row2_cur_inline15470__ssa_v0
    uint64_t v215 = (uint64_t)v41;
    TASSIGN(v214, v215);
    TDIV(v214, v196, v208);
    // pto: %row3_cur_inline15461__ssa_v0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v216 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row3_cur_inline15461__ssa_v0
    uint64_t v217 = (uint64_t)v40;
    TASSIGN(v216, v217);
    TDIV(v216, v198, v208);
    for (int64_t i218 = v24; i218 < v18; i218 += v17) {
        // pto: %t__tmp_v2039
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v219 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %t__tmp_v2039
        uint64_t v220 = (uint64_t)v44;
        TASSIGN(v219, v220);
        pipe_barrier(PIPE_V);
        TROWSUM(v219, v210, v200);
        // pto: %row0_rowsum_inline15464__rm_a0_tmp_v0
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v221 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row0_rowsum_inline15464__rm_a0_tmp_v0
        uint64_t v222 = (uint64_t)v44;
        TASSIGN(v221, v222);
        // pto: %row0_rowsum_inline15464__row_major_tmp_v1
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v223 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row0_rowsum_inline15464__row_major_tmp_v1
        uint64_t v224 = (uint64_t)v37;
        TASSIGN(v223, v224);
        pipe_barrier(PIPE_V);
        TADDS(v223, v221, v19);
        // pto: %row0_rowsum_inline15464__ssa_v0
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v225 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %row0_rowsum_inline15464__ssa_v0
        uint64_t v226 = (uint64_t)v37;
        TASSIGN(v225, v226);
        // pto: %t__tmp_v2040
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v227 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %t__tmp_v2040
        uint64_t v228 = (uint64_t)v44;
        TASSIGN(v227, v228);
        pipe_barrier(PIPE_V);
        TROWSUM(v227, v212, v200);
        // pto: %row1_rowsum_inline15480__rm_a0_tmp_v2
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v229 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row1_rowsum_inline15480__rm_a0_tmp_v2
        uint64_t v230 = (uint64_t)v44;
        TASSIGN(v229, v230);
        // pto: %row1_rowsum_inline15480__row_major_tmp_v3
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v231 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row1_rowsum_inline15480__row_major_tmp_v3
        uint64_t v232 = (uint64_t)v36;
        TASSIGN(v231, v232);
        pipe_barrier(PIPE_V);
        TADDS(v231, v229, v19);
        // pto: %row1_rowsum_inline15480__ssa_v0
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v233 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %row1_rowsum_inline15480__ssa_v0
        uint64_t v234 = (uint64_t)v36;
        TASSIGN(v233, v234);
        // pto: %t__tmp_v2041
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v235 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %t__tmp_v2041
        uint64_t v236 = (uint64_t)v44;
        TASSIGN(v235, v236);
        pipe_barrier(PIPE_V);
        TROWSUM(v235, v214, v200);
        // pto: %row2_rowsum_inline15467__rm_a0_tmp_v4
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v237 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row2_rowsum_inline15467__rm_a0_tmp_v4
        uint64_t v238 = (uint64_t)v44;
        TASSIGN(v237, v238);
        // pto: %row2_rowsum_inline15467__row_major_tmp_v5
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v239 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row2_rowsum_inline15467__row_major_tmp_v5
        uint64_t v240 = (uint64_t)v31;
        TASSIGN(v239, v240);
        pipe_barrier(PIPE_V);
        TADDS(v239, v237, v19);
        // pto: %row2_rowsum_inline15467__ssa_v0
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v241 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %row2_rowsum_inline15467__ssa_v0
        uint64_t v242 = (uint64_t)v31;
        TASSIGN(v241, v242);
        // pto: %t__tmp_v2042
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v243 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %t__tmp_v2042
        uint64_t v244 = (uint64_t)v44;
        TASSIGN(v243, v244);
        pipe_barrier(PIPE_V);
        TROWSUM(v243, v216, v200);
        // pto: %row3_rowsum_inline15425__rm_a0_tmp_v6
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v245 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row3_rowsum_inline15425__rm_a0_tmp_v6
        uint64_t v246 = (uint64_t)v44;
        TASSIGN(v245, v246);
        // pto: %row3_rowsum_inline15425__row_major_tmp_v7
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v247 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %row3_rowsum_inline15425__row_major_tmp_v7
        uint64_t v248 = (uint64_t)v30;
        TASSIGN(v247, v248);
        pipe_barrier(PIPE_V);
        TADDS(v247, v245, v19);
        // pto: %row3_rowsum_inline15425__ssa_v0
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v249 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %row3_rowsum_inline15425__ssa_v0
        uint64_t v250 = (uint64_t)v30;
        TASSIGN(v249, v250);
        // pto: %row0_norm_inline15384__ssa_v0
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v251 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row0_norm_inline15384__ssa_v0
        uint64_t v252 = (uint64_t)v38;
        TASSIGN(v251, v252);
        TROWEXPANDDIV(v251, v210, v225);
        // pto: %row1_norm_inline15453__ssa_v0
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v253 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row1_norm_inline15453__ssa_v0
        uint64_t v254 = (uint64_t)v37;
        TASSIGN(v253, v254);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v253, v212, v233);
        // pto: %row2_norm_inline15468__ssa_v0
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v255 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row2_norm_inline15468__ssa_v0
        uint64_t v256 = (uint64_t)v36;
        TASSIGN(v255, v256);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v255, v214, v241);
        // pto: %row3_norm_inline15406__ssa_v0
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v257 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row3_norm_inline15406__ssa_v0
        uint64_t v258 = (uint64_t)v31;
        TASSIGN(v257, v258);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v257, v216, v249);
        // pto: %t__tmp_v2043
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v259 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %t__tmp_v2043
        uint64_t v260 = (uint64_t)v30;
        TASSIGN(v259, v260);
        pipe_barrier(PIPE_V);
        TADD(v259, v251, v253);
        // pto: %t__tmp_v2044
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v261 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %t__tmp_v2044
        uint64_t v262 = (uint64_t)v29;
        TASSIGN(v261, v262);
        TADD(v261, v255, v257);
        // pto: %col_sum_v1_inline15454__ssa_v3
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v263 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %col_sum_v1_inline15454__ssa_v3
        uint64_t v264 = (uint64_t)v30;
        TASSIGN(v263, v264);
        pipe_barrier(PIPE_V);
        TADD(v263, v259, v261);
        // pto: %col_sum_v1_inline15454__ssa_v4
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v265 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %col_sum_v1_inline15454__ssa_v4
        uint64_t v266 = (uint64_t)v30;
        TASSIGN(v265, v266);
        pipe_barrier(PIPE_V);
        TADDS(v265, v263, v19);
        // pto: %row0_cur_inline15458__ssa_v3
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v267 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row0_cur_inline15458__ssa_v3
        uint64_t v268 = (uint64_t)v38;
        TASSIGN(v267, v268);
        pipe_barrier(PIPE_V);
        TDIV(v267, v251, v265);
        // pto: %row1_cur_inline15459__ssa_v3
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v269 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row1_cur_inline15459__ssa_v3
        uint64_t v270 = (uint64_t)v37;
        TASSIGN(v269, v270);
        TDIV(v269, v253, v265);
        // pto: %row2_cur_inline15470__ssa_v3
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v271 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row2_cur_inline15470__ssa_v3
        uint64_t v272 = (uint64_t)v36;
        TASSIGN(v271, v272);
        TDIV(v271, v255, v265);
        // pto: %row3_cur_inline15461__ssa_v3
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v273 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %row3_cur_inline15461__ssa_v3
        uint64_t v274 = (uint64_t)v31;
        TASSIGN(v273, v274);
        TDIV(v273, v257, v265);
        // pto: %0
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v275 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %0
        uint64_t v276 = (uint64_t)v34;
        TASSIGN(v275, v276);
        pipe_barrier(PIPE_V);
        TROWSUM(v275, v267, v200);
        // pto: %1
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v277 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %1
        uint64_t v278 = (uint64_t)v34;
        TASSIGN(v277, v278);
        // pto: %2
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v279 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %2
        uint64_t v280 = (uint64_t)v28;
        TASSIGN(v279, v280);
        pipe_barrier(PIPE_V);
        TADDS(v279, v277, v19);
        // pto: %3
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v281 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %3
        uint64_t v282 = (uint64_t)v28;
        TASSIGN(v281, v282);
        // pto: %4
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v283 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %4
        uint64_t v284 = (uint64_t)v34;
        TASSIGN(v283, v284);
        pipe_barrier(PIPE_V);
        TROWSUM(v283, v269, v200);
        // pto: %5
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v285 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %5
        uint64_t v286 = (uint64_t)v34;
        TASSIGN(v285, v286);
        // pto: %6
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v287 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %6
        uint64_t v288 = (uint64_t)v27;
        TASSIGN(v287, v288);
        pipe_barrier(PIPE_V);
        TADDS(v287, v285, v19);
        // pto: %7
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v289 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %7
        uint64_t v290 = (uint64_t)v27;
        TASSIGN(v289, v290);
        // pto: %8
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v291 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %8
        uint64_t v292 = (uint64_t)v34;
        TASSIGN(v291, v292);
        pipe_barrier(PIPE_V);
        TROWSUM(v291, v271, v200);
        // pto: %9
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v293 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %9
        uint64_t v294 = (uint64_t)v34;
        TASSIGN(v293, v294);
        // pto: %10
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v295 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %10
        uint64_t v296 = (uint64_t)v26;
        TASSIGN(v295, v296);
        pipe_barrier(PIPE_V);
        TADDS(v295, v293, v19);
        // pto: %11
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v297 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %11
        uint64_t v298 = (uint64_t)v26;
        TASSIGN(v297, v298);
        // pto: %12
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v299 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %12
        uint64_t v300 = (uint64_t)v34;
        TASSIGN(v299, v300);
        pipe_barrier(PIPE_V);
        TROWSUM(v299, v273, v200);
        // pto: %13
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v301 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %13
        uint64_t v302 = (uint64_t)v34;
        TASSIGN(v301, v302);
        // pto: %14
        Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v303 = Tile<
                TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v22, v21);
        // pto: %14
        uint64_t v304 = (uint64_t)v25;
        TASSIGN(v303, v304);
        pipe_barrier(PIPE_V);
        TADDS(v303, v301, v19);
        // pto: %15
        Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v305 = Tile<
                TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v21, v22);
        // pto: %15
        uint64_t v306 = (uint64_t)v25;
        TASSIGN(v305, v306);
        // pto: %16
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v307 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %16
        uint64_t v308 = (uint64_t)v24;
        TASSIGN(v307, v308);
        TROWEXPANDDIV(v307, v267, v281);
        // pto: %17
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v309 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %17
        uint64_t v310 = (uint64_t)v28;
        TASSIGN(v309, v310);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v309, v269, v289);
        // pto: %18
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v311 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %18
        uint64_t v312 = (uint64_t)v27;
        TASSIGN(v311, v312);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v311, v271, v297);
        // pto: %19
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v313 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %19
        uint64_t v314 = (uint64_t)v26;
        TASSIGN(v313, v314);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(v313, v273, v305);
        // pto: %20
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v315 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %20
        uint64_t v316 = (uint64_t)v25;
        TASSIGN(v315, v316);
        pipe_barrier(PIPE_V);
        TADD(v315, v307, v309);
        // pto: %21
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v317 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %21
        uint64_t v318 = (uint64_t)v23;
        TASSIGN(v317, v318);
        TADD(v317, v311, v313);
        // pto: %22
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v319 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %22
        uint64_t v320 = (uint64_t)v25;
        TASSIGN(v319, v320);
        pipe_barrier(PIPE_V);
        TADD(v319, v315, v317);
        // pto: %23
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v321 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %23
        uint64_t v322 = (uint64_t)v39;
        TASSIGN(v321, v322);
        pipe_barrier(PIPE_V);
        TADDS(v321, v319, v19);
        // pto: %24
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v323 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %24
        uint64_t v324 = (uint64_t)v43;
        TASSIGN(v323, v324);
        pipe_barrier(PIPE_V);
        TDIV(v323, v307, v321);
        // pto: %25
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v325 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %25
        uint64_t v326 = (uint64_t)v42;
        TASSIGN(v325, v326);
        TDIV(v325, v309, v321);
        // pto: %26
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v327 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %26
        uint64_t v328 = (uint64_t)v41;
        TASSIGN(v327, v328);
        TDIV(v327, v311, v321);
        // pto: %27
        Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>
            v329 = Tile<
                TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
                CompactMode::Null>(v21, v21);
        // pto: %27
        uint64_t v330 = (uint64_t)v40;
        TASSIGN(v329, v330);
        TDIV(v329, v313, v321);
    }
    // pto: %28
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v331 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %28
    uint64_t v332 = (uint64_t)v44;
    TASSIGN(v331, v332);
    pipe_barrier(PIPE_V);
    TROWSUM(v331, v210, v200);
    // pto: %29
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v333 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %29
    uint64_t v334 = (uint64_t)v44;
    TASSIGN(v333, v334);
    // pto: %30
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v335 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %30
    uint64_t v336 = (uint64_t)v39;
    TASSIGN(v335, v336);
    pipe_barrier(PIPE_V);
    TADDS(v335, v333, v19);
    // pto: %31
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v337 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %31
    uint64_t v338 = (uint64_t)v39;
    TASSIGN(v337, v338);
    // pto: %32
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v339 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %32
    uint64_t v340 = (uint64_t)v44;
    TASSIGN(v339, v340);
    pipe_barrier(PIPE_V);
    TROWSUM(v339, v212, v200);
    // pto: %33
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v341 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %33
    uint64_t v342 = (uint64_t)v44;
    TASSIGN(v341, v342);
    // pto: %34
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v343 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %34
    uint64_t v344 = (uint64_t)v38;
    TASSIGN(v343, v344);
    pipe_barrier(PIPE_V);
    TADDS(v343, v341, v19);
    // pto: %35
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v345 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %35
    uint64_t v346 = (uint64_t)v38;
    TASSIGN(v345, v346);
    // pto: %36
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v347 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %36
    uint64_t v348 = (uint64_t)v44;
    TASSIGN(v347, v348);
    pipe_barrier(PIPE_V);
    TROWSUM(v347, v214, v200);
    // pto: %37
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v349 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %37
    uint64_t v350 = (uint64_t)v44;
    TASSIGN(v349, v350);
    // pto: %38
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v351 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %38
    uint64_t v352 = (uint64_t)v37;
    TASSIGN(v351, v352);
    pipe_barrier(PIPE_V);
    TADDS(v351, v349, v19);
    // pto: %39
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v353 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %39
    uint64_t v354 = (uint64_t)v37;
    TASSIGN(v353, v354);
    // pto: %40
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v355 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %40
    uint64_t v356 = (uint64_t)v44;
    TASSIGN(v355, v356);
    pipe_barrier(PIPE_V);
    TROWSUM(v355, v216, v200);
    // pto: %41
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v357 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %41
    uint64_t v358 = (uint64_t)v44;
    TASSIGN(v357, v358);
    // pto: %42
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v359 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v22, v21);
    // pto: %42
    uint64_t v360 = (uint64_t)v35;
    TASSIGN(v359, v360);
    pipe_barrier(PIPE_V);
    TADDS(v359, v357, v19);
    // pto: %43
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v361 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v21, v22);
    // pto: %43
    uint64_t v362 = (uint64_t)v35;
    TASSIGN(v361, v362);
    // pto: %44
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v363 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %44
    uint64_t v364 = (uint64_t)v43;
    TASSIGN(v363, v364);
    TROWEXPANDDIV(v363, v210, v337);
    // pto: %45
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v365 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %45
    uint64_t v366 = (uint64_t)v42;
    TASSIGN(v365, v366);
    TROWEXPANDDIV(v365, v212, v345);
    // pto: %46
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v367 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %46
    uint64_t v368 = (uint64_t)v41;
    TASSIGN(v367, v368);
    TROWEXPANDDIV(v367, v214, v353);
    // pto: %47
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v369 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %47
    uint64_t v370 = (uint64_t)v40;
    TASSIGN(v369, v370);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(v369, v216, v361);
    // pto: %48
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v371 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %48
    uint64_t v372 = (uint64_t)v35;
    TASSIGN(v371, v372);
    pipe_barrier(PIPE_V);
    TADD(v371, v363, v365);
    // pto: %49
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v373 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %49
    uint64_t v374 = (uint64_t)v39;
    TASSIGN(v373, v374);
    TADD(v373, v367, v369);
    // pto: %50
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v375 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %50
    uint64_t v376 = (uint64_t)v35;
    TASSIGN(v375, v376);
    pipe_barrier(PIPE_V);
    TADD(v375, v371, v373);
    // pto: %51
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v377 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %51
    uint64_t v378 = (uint64_t)v35;
    TASSIGN(v377, v378);
    pipe_barrier(PIPE_V);
    TADDS(v377, v375, v19);
    // pto: %52
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v379 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %52
    uint64_t v380 = (uint64_t)v43;
    TASSIGN(v379, v380);
    pipe_barrier(PIPE_V);
    TDIV(v379, v363, v377);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %53
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v381 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %53
    uint64_t v382 = (uint64_t)v42;
    TASSIGN(v381, v382);
    TDIV(v381, v365, v377);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %54
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v383 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %54
    uint64_t v384 = (uint64_t)v41;
    TASSIGN(v383, v384);
    TDIV(v383, v367, v377);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %55
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v385 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %55
    uint64_t v386 = (uint64_t)v40;
    TASSIGN(v385, v386);
    TDIV(v385, v369, v377);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // pto: %col_sum_v1_inline15454__rv_v2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v387 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %col_sum_v1_inline15454__rv_v2
    uint64_t v388 = (uint64_t)v35;
    TASSIGN(v387, v388);
    // pto: %row0_cur_inline15458__rv_v2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v389 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row0_cur_inline15458__rv_v2
    uint64_t v390 = (uint64_t)v43;
    TASSIGN(v389, v390);
    // pto: %row1_cur_inline15459__rv_v2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v391 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row1_cur_inline15459__rv_v2
    uint64_t v392 = (uint64_t)v42;
    TASSIGN(v391, v392);
    // pto: %row2_cur_inline15470__rv_v2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v393 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row2_cur_inline15470__rv_v2
    uint64_t v394 = (uint64_t)v41;
    TASSIGN(v393, v394);
    // pto: %row3_cur_inline15461__rv_v2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>
        v395 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero,
            CompactMode::Null>(v21, v21);
    // pto: %row3_cur_inline15461__rv_v2
    uint64_t v396 = (uint64_t)v40;
    TASSIGN(v395, v396);
    v389.SetValidShape(v21, v20);
    v391.SetValidShape(v21, v20);
    v393.SetValidShape(v21, v20);
    v395.SetValidShape(v21, v20);
    // pto: %comb_ffn_inline13000__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 4> v397 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %comb_ffn_inline13000__ssa_v0_pview
    pto::Stride<128, 128, 128, 16, 1> v398 = pto::Stride<128, 128, 128, 16, 1>();
    // pto: %comb_ffn_inline13000__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND> v399 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND>(
            v4 + (v24 + v53 * v14), v397, v398
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v399, v389);
    // pto: %70
    pto::Shape<1, 1, 1, 8, 4> v400 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %70
    pto::Stride<128, 128, 128, 16, 1> v401 = pto::Stride<128, 128, 128, 16, 1>();
    // pto: %70
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND> v402 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND>(
            v4 + (v20 + v53 * v14), v400, v401
        );
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v402, v391);
    // pto: %72
    pto::Shape<1, 1, 1, 8, 4> v403 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %72
    pto::Stride<128, 128, 128, 16, 1> v404 = pto::Stride<128, 128, 128, 16, 1>();
    // pto: %72
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND> v405 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND>(
            v4 + (v21 + v53 * v14), v403, v404
        );
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v405, v393);
    // pto: %74
    pto::Shape<1, 1, 1, 8, 4> v406 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %74
    pto::Stride<128, 128, 128, 16, 1> v407 = pto::Stride<128, 128, 128, 16, 1>();
    // pto: %74
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND> v408 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<128, 128, 128, 16, 1>, pto::Layout::ND>(
            v4 + (v15 + v53 * v14), v406, v407
        );
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v408, v395);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: inv_rms_inline15360__ssa_v1
    __gm__ Tensor *inv_rms_inline15360__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *inv_rms_inline15360__ssa_v1 =
        reinterpret_cast<__gm__ float *>(inv_rms_inline15360__ssa_v1_tensor->buffer.addr) +
        inv_rms_inline15360__ssa_v1_tensor->start_offset;

    // Unpack tensor: mixes_raw_inline15383__ssa_v1
    __gm__ Tensor *mixes_raw_inline15383__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *mixes_raw_inline15383__ssa_v1 =
        reinterpret_cast<__gm__ float *>(mixes_raw_inline15383__ssa_v1_tensor->buffer.addr) +
        mixes_raw_inline15383__ssa_v1_tensor->start_offset;

    // Unpack tensor: hc_base_2d_inline15420__ssa_v0
    __gm__ Tensor *hc_base_2d_inline15420__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *hc_base_2d_inline15420__ssa_v0 =
        reinterpret_cast<__gm__ float *>(hc_base_2d_inline15420__ssa_v0_tensor->buffer.addr) +
        hc_base_2d_inline15420__ssa_v0_tensor->start_offset;

    // Unpack tensor: comb_ffn_inline13000__ssa_v0
    __gm__ Tensor *comb_ffn_inline13000__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *comb_ffn_inline13000__ssa_v0 =
        reinterpret_cast<__gm__ float *>(comb_ffn_inline13000__ssa_v0_tensor->buffer.addr) +
        comb_ffn_inline13000__ssa_v0_tensor->start_offset;

    // Unpack tensor: hc_scale (scale2 read from GM instead of a host-passed scalar)
    __gm__ Tensor *hc_scale_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *hc_scale =
        reinterpret_cast<__gm__ float *>(hc_scale_tensor->buffer.addr) + hc_scale_tensor->start_offset;
    float scale2_inline15385__ssa_v0 = hc_scale[2];

    // Extract dynamic dim: t_linear_inline15410__ssa_v0
    int64_t t_linear_inline15410__ssa_v0 = static_cast<int64_t>(inv_rms_inline15360__ssa_v1_tensor->shapes[0]);

    // Forward to ptoas-generated function
    comb_sinkhorn_8(
        inv_rms_inline15360__ssa_v1, mixes_raw_inline15383__ssa_v1, hc_base_2d_inline15420__ssa_v0,
        comb_ffn_inline13000__ssa_v0, scale2_inline15385__ssa_v0, t_linear_inline15410__ssa_v0, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
