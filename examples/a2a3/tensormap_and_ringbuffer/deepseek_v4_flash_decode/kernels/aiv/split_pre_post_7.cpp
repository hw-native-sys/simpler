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
// Kernel Function: split_pre_post_7

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

static __aicore__ void split_pre_post_7(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4, __gm__ float *v5, float v6, float v7,
    int64_t v8, int32_t v9, int32_t v10
) {
    unsigned v11 = 4;
    const int64_t v12 = 32;
    const float v13 = 2.0f;
    const float v14 = 9.99999997E-7f;
    const float v15 = 1.0f;
    const int64_t v16 = 4;
    const int64_t v17 = 8;
    const int64_t v18 = 1;
    const int64_t v19 = 0;
    const int64_t v20 = 320;
    const int64_t v21 = 288;
    const int64_t v22 = 256;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %inv_rms_inline15113__ssa_v1_view
    int64_t v23 = v8 * v18;
    // pto: %inv_rms_inline15113__ssa_v1_view
    int64_t v24 = v18 * v23;
    // pto: %inv_rms_inline15113__ssa_v1_view
    pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v18, v18, v18, v8, v18);
    // pto: %inv_rms_inline15113__ssa_v1_view
    pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v24, v24, v23, v18, v8);
    // pto: %inv_rms_inline15113__ssa_v1_view
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v27 =
        GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v1, v25, v26
        );
    // pto: %ob_inline15129__ssa_v0, %14
    int64_t v28 = (int64_t)((uint64_t)((int64_t)v9) * (uint64_t)v17);
    // pto: %inv_col_inline15196__tile
    Tile<
        TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v29 = Tile<
            TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v18);
    // pto: %inv_col_inline15196__tile
    uint64_t v30 = (uint64_t)v22;
    TASSIGN(v29, v30);
    // pto: %15
    int64_t v31 = v28 < v19 ? v19 : v28;
    // pto: %inv_rms_inline15113__ssa_v1_pview
    __gm__ float *v32 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %inv_rms_inline15113__ssa_v1_pview
    int64_t v33 = v17 * v18;
    // pto: %inv_rms_inline15113__ssa_v1_pview
    int64_t v34 = v18 * v33;
    // pto: %inv_rms_inline15113__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 1> v35 = pto::Shape<1, 1, 1, 8, 1>(v18, v18, v18, v17, v18);
    // pto: %inv_rms_inline15113__ssa_v1_pview
    pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v34, v34, v33, v18, v8);
    // pto: %inv_rms_inline15113__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v37 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(
            v32 + ((v19 + v31 * v18) + v19 * v8), v35, v36
        );
    TLOAD(v29, v37);
    // pto: %t__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v38 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v17);
    // pto: %t__tile
    uint64_t v39 = (uint64_t)v21;
    TASSIGN(v38, v39);
    // pto: %hc_attn_base_last_inline509__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 8> v40 = pto::Shape<1, 1, 1, 1, 8>();
    // pto: %hc_attn_base_last_inline509__ssa_v0_pview
    pto::Stride<8, 8, 8, 8, 1> v41 = pto::Stride<8, 8, 8, 8, 1>();
    // pto: %hc_attn_base_last_inline509__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v42 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v2, v40, v41);
    TLOAD(v38, v42);
    // pto: %pre_base_inline15127__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v43 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v17);
    // pto: %pre_base_inline15127__tile
    uint64_t v44 = (uint64_t)v21;
    TASSIGN(v43, v44);
    // pto: %0
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v45 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %0
    uint64_t v46 = (uint64_t)v20;
    TASSIGN(v45, v46);
    // pto: %mixes_raw_inline15136__ssa_v1_pview
    pto::Shape<1, 1, 1, 8, 8> v47 = pto::Shape<1, 1, 1, 8, 8>();
    // pto: %mixes_raw_inline15136__ssa_v1_pview
    pto::Stride<256, 256, 256, 32, 1> v48 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %mixes_raw_inline15136__ssa_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v49 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v3 + (v19 + v31 * v12), v47, v48
        );
    TLOAD(v45, v49);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %1
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v50 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %1
    uint64_t v51 = (uint64_t)v20;
    TASSIGN(v50, v51);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v50, v45, v29);
    // pto: %pre_scaled_inline15160__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v52 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %pre_scaled_inline15160__tile
    uint64_t v53 = (uint64_t)v20;
    TASSIGN(v52, v53);
    pipe_barrier(PIPE_V);
    TMULS(v52, v50, v6);
    // pto: %2
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v54 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %2
    uint64_t v55 = (uint64_t)v19;
    TASSIGN(v54, v55);
    TCOLEXPAND(v54, v43);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %pre_logits_inline15126__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v56 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %pre_logits_inline15126__tile
    uint64_t v57 = (uint64_t)v20;
    TASSIGN(v56, v57);
    pipe_barrier(PIPE_V);
    TADD(v56, v52, v54);
    // pto: %3
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v58 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %3
    uint64_t v59 = (uint64_t)v20;
    TASSIGN(v58, v59);
    pipe_barrier(PIPE_V);
    TNEG(v58, v56);
    // pto: %4
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v60 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %4
    uint64_t v61 = (uint64_t)v20;
    TASSIGN(v60, v61);
    pipe_barrier(PIPE_V);
    TEXP(v60, v58);
    // pto: %5
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v62 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %5
    uint64_t v63 = (uint64_t)v20;
    TASSIGN(v62, v63);
    pipe_barrier(PIPE_V);
    TADDS(v62, v60, v15);
    // pto: %pre_sig_inline15124__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v64 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %pre_sig_inline15124__tile
    uint64_t v65 = (uint64_t)v19;
    TASSIGN(v64, v65);
    pipe_barrier(PIPE_V);
    TRECIP(v64, v62);
    // pto: %pre_val_inline15123__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v66 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %pre_val_inline15123__tile
    uint64_t v67 = (uint64_t)v20;
    TASSIGN(v66, v67);
    pipe_barrier(PIPE_V);
    TADDS(v66, v64, v14);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %pre_val_store_inline15141__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 8> v68 = pto::Shape<1, 1, 1, 8, 8>();
    // pto: %pre_val_store_inline15141__ssa_v0_pview
    pto::Stride<64, 64, 64, 8, 1> v69 = pto::Stride<64, 64, 64, 8, 1>();
    // pto: %pre_val_store_inline15141__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND> v70 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND>(
            v4 + (v19 + v31 * v17), v68, v69
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v70, v66);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %6
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v71 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v17);
    // pto: %6
    uint64_t v72 = (uint64_t)v21;
    TASSIGN(v71, v72);
    // pto: %18
    pto::Shape<1, 1, 1, 1, 8> v73 = pto::Shape<1, 1, 1, 1, 8>();
    // pto: %18
    pto::Stride<8, 8, 8, 8, 1> v74 = pto::Stride<8, 8, 8, 8, 1>();
    // pto: %18
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v75 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(v2 + v11, v73, v74);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v71, v75);
    // pto: %post_base_inline15120__tile
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v76 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v17);
    // pto: %post_base_inline15120__tile
    uint64_t v77 = (uint64_t)v21;
    TASSIGN(v76, v77);
    // pto: %7
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v78 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %7
    uint64_t v79 = (uint64_t)v20;
    TASSIGN(v78, v79);
    // pto: %20
    pto::Shape<1, 1, 1, 8, 8> v80 = pto::Shape<1, 1, 1, 8, 8>();
    // pto: %20
    pto::Stride<256, 256, 256, 32, 1> v81 = pto::Stride<256, 256, 256, 32, 1>();
    // pto: %20
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND> v82 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<256, 256, 256, 32, 1>, pto::Layout::ND>(
            v3 + (v16 + v31 * v12), v80, v81
        );
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v78, v82);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %8
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v83 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %8
    uint64_t v84 = (uint64_t)v20;
    TASSIGN(v83, v84);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v83, v78, v29);
    // pto: %post_scaled_inline15140__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v85 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %post_scaled_inline15140__tile
    uint64_t v86 = (uint64_t)v20;
    TASSIGN(v85, v86);
    pipe_barrier(PIPE_V);
    TMULS(v85, v83, v7);
    // pto: %9
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v87 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %9
    uint64_t v88 = (uint64_t)v19;
    TASSIGN(v87, v88);
    TCOLEXPAND(v87, v76);
    // pto: %post_logits_inline15215__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v89 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %post_logits_inline15215__tile
    uint64_t v90 = (uint64_t)v20;
    TASSIGN(v89, v90);
    pipe_barrier(PIPE_V);
    TADD(v89, v85, v87);
    // pto: %10
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v91 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %10
    uint64_t v92 = (uint64_t)v20;
    TASSIGN(v91, v92);
    pipe_barrier(PIPE_V);
    TNEG(v91, v89);
    // pto: %11
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v93 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %11
    uint64_t v94 = (uint64_t)v20;
    TASSIGN(v93, v94);
    pipe_barrier(PIPE_V);
    TEXP(v93, v91);
    // pto: %12
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v95 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %12
    uint64_t v96 = (uint64_t)v20;
    TASSIGN(v95, v96);
    pipe_barrier(PIPE_V);
    TADDS(v95, v93, v15);
    // pto: %post_sig_inline15122__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v97 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %post_sig_inline15122__tile
    uint64_t v98 = (uint64_t)v19;
    TASSIGN(v97, v98);
    pipe_barrier(PIPE_V);
    TRECIP(v97, v95);
    // pto: %post_pad_inline15118__tile
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v99 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v17);
    // pto: %post_pad_inline15118__tile
    uint64_t v100 = (uint64_t)v20;
    TASSIGN(v99, v100);
    pipe_barrier(PIPE_V);
    TMULS(v99, v97, v13);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %13
    Tile<
        TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v101 = Tile<
            TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v16);
    // pto: %13
    uint64_t v102 = (uint64_t)v20;
    TASSIGN(v101, v102);
    // pto: %slice_view
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v103;
    // pto: %slice_view
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v104 = v103;
    // pto: %slice_view
    uint64_t v105 = (uint64_t)v20;
    TASSIGN(v104, v105);
    // pto: %post_t_inline12482__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 4> v106 = pto::Shape<1, 1, 1, 8, 4>();
    // pto: %post_t_inline12482__ssa_v0_pview
    pto::Stride<32, 32, 32, 4, 1> v107 = pto::Stride<32, 32, 32, 4, 1>();
    // pto: %post_t_inline12482__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<32, 32, 32, 4, 1>, pto::Layout::ND> v108 =
        GlobalTensor<float, pto::Shape<1, 1, 1, 8, 4>, pto::Stride<32, 32, 32, 4, 1>, pto::Layout::ND>(
            v5 + (v19 + v31 * v16), v106, v107
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v108, v104);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: inv_rms_inline15113__ssa_v1
    __gm__ Tensor *inv_rms_inline15113__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *inv_rms_inline15113__ssa_v1 =
        reinterpret_cast<__gm__ float *>(inv_rms_inline15113__ssa_v1_tensor->buffer.addr) +
        inv_rms_inline15113__ssa_v1_tensor->start_offset;

    // Unpack tensor: hc_attn_base_last_inline509__ssa_v0
    __gm__ Tensor *hc_attn_base_last_inline509__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *hc_attn_base_last_inline509__ssa_v0 =
        reinterpret_cast<__gm__ float *>(hc_attn_base_last_inline509__ssa_v0_tensor->buffer.addr) +
        hc_attn_base_last_inline509__ssa_v0_tensor->start_offset;

    // Unpack tensor: mixes_raw_inline15136__ssa_v1
    __gm__ Tensor *mixes_raw_inline15136__ssa_v1_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *mixes_raw_inline15136__ssa_v1 =
        reinterpret_cast<__gm__ float *>(mixes_raw_inline15136__ssa_v1_tensor->buffer.addr) +
        mixes_raw_inline15136__ssa_v1_tensor->start_offset;

    // Unpack tensor: pre_val_store_inline15141__ssa_v0
    __gm__ Tensor *pre_val_store_inline15141__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *pre_val_store_inline15141__ssa_v0 =
        reinterpret_cast<__gm__ float *>(pre_val_store_inline15141__ssa_v0_tensor->buffer.addr) +
        pre_val_store_inline15141__ssa_v0_tensor->start_offset;

    // Unpack tensor: post_t_inline12482__ssa_v0
    __gm__ Tensor *post_t_inline12482__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *post_t_inline12482__ssa_v0 =
        reinterpret_cast<__gm__ float *>(post_t_inline12482__ssa_v0_tensor->buffer.addr) +
        post_t_inline12482__ssa_v0_tensor->start_offset;

    // Unpack tensor: hc_scale (scale0/scale1 read from GM instead of host-passed scalars)
    __gm__ Tensor *hc_scale_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *hc_scale =
        reinterpret_cast<__gm__ float *>(hc_scale_tensor->buffer.addr) + hc_scale_tensor->start_offset;
    float scale0_inline15152__ssa_v0 = hc_scale[0];
    float scale1_inline15172__ssa_v0 = hc_scale[1];

    // Extract dynamic dim: t_linear_inline15163__ssa_v0
    int64_t t_linear_inline15163__ssa_v0 = static_cast<int64_t>(inv_rms_inline15113__ssa_v1_tensor->shapes[0]);

    // Forward to ptoas-generated function
    split_pre_post_7(
        inv_rms_inline15113__ssa_v1, hc_attn_base_last_inline509__ssa_v0, mixes_raw_inline15136__ssa_v1,
        pre_val_store_inline15141__ssa_v0, post_t_inline12482__ssa_v0, scale0_inline15152__ssa_v0,
        scale1_inline15172__ssa_v0, t_linear_inline15163__ssa_v0, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
