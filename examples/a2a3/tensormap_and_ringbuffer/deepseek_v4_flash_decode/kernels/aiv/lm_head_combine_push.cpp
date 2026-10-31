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
// Kernel Function: lm_head_combine_push

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
#include "platform_comm/comm_context.h"
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

static __aicore__ void lm_head_combine_push(
    __gm__ float *v1, __gm__ float *v2, __gm__ int32_t *v3, int32_t v4, __gm__ CommContext *v5, __gm__ CommContext *v6,
    int32_t v7, int32_t v8
) {
    pto::comm::NotifyOp v9 = pto::comm::NotifyOp::AtomicAdd;
    const int32_t v10 = 1;
    const int64_t v11 = 1152;
    const int64_t v12 = 63488;
    const int64_t v13 = 7;
    const int64_t v14 = 4;
    const int64_t v15 = 2048;
    const int64_t v16 = 24;
    const int64_t v17 = 31;
    const int64_t v18 = 2;
    const int64_t v19 = 64640;
    const int64_t v20 = 1;
    const int64_t v21 = 8;
    const int64_t v22 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %blk_inline13_inline13299__ssa_v0
    int64_t v23 = (int64_t)v7;
    // pto: %1
    int64_t v24 = (int64_t)v4;
    // pto: %2
    int64_t v25 = v24 % v18;
    // pto: %4
    int64_t v26 = (int64_t)((uint64_t)v25 * (uint64_t)v19);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    for (int64_t i27 = v22; i27 < v18; i27 += v20) {
        // pto: %5
        int64_t v28 = (int64_t)((uint64_t)i27 * (uint64_t)v21);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        for (int64_t j29 = v23; j29 < v17; j29 += v16) {
            // pto: %6
            int64_t v30 = (int64_t)((uint64_t)j29 * (uint64_t)v15);
            // pto: %tput_stage
            Tile<
                TileType::Vec, float, 8, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v31 = Tile<
                    TileType::Vec, float, 8, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v21, v15);
            // pto: %tput_stage
            uint64_t v32 = (uint64_t)v22;
            TASSIGN(v31, v32);
            // pto: %7
            int64_t v33 = (int64_t)((uint64_t)v26 + (uint64_t)v30);
            // pto: %16
            int64_t v34 = static_cast<int64_t>(v5->rankId);
            // pto: %17, %18, %19, %20
            int64_t v35 = static_cast<int64_t>(v5->windowsIn[v34]);
            // pto: %12, %13, %15, %21, %22
            int64_t v36 =
                static_cast<int64_t>(v5->windowsIn[((int64_t)((uint64_t)(v24 / v18) * (uint64_t)v18)) + (uint64_t)i27]);
            // pto: %lm_head_logits_window__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 8, 2048> v37 = pto::Shape<1, 1, 1, 8, 2048>();
            // pto: %lm_head_logits_window__ssa_v0_peer_pview
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v38 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %23, %24, %26, %8, %lm_head_logits_window__ssa_v0_peer_pview
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v39 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(
                    (v1 + (int64_t)((uint64_t)v36 - (uint64_t)v35) / v14) + (v22 + (v33 < v22 ? v22 : v33)), v37, v38
                );
            // pto: %logits_shards_inline40_inline13289__rv_v2_local_pview
            pto::Shape<1, 1, 1, 8, 2048> v40 = pto::Shape<1, 1, 1, 8, 2048>();
            // pto: %logits_shards_inline40_inline13289__rv_v2_local_pview
            pto::Stride<517120, 517120, 517120, 64640, 1> v41 = pto::Stride<517120, 517120, 517120, 64640, 1>();
            // pto: %9, %logits_shards_inline40_inline13289__rv_v2_local_pview, %10
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<517120, 517120, 517120, 64640, 1>, pto::Layout::ND>
                v42 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<517120, 517120, 517120, 64640, 1>,
                    pto::Layout::ND>(v2 + ((v22 + (v28 < v22 ? v22 : v28) * v19) + (v30 < v22 ? v22 : v30)), v40, v41);
            pipe_barrier(PIPE_ALL);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
            pipe_barrier(PIPE_MTE2);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            pto::comm::TPUT(v39, v42, v31);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
            pipe_barrier(PIPE_ALL);
            __gm__ float *v43 = PTOAS__GLOBAL_TENSOR_DATA(v39);
            PTOAS__DCCI_SINGLE_CACHE_LINE(v43);
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        // pto: %28
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        if (v23 == v13) {
            // pto: %0
            Tile<
                TileType::Vec, float, 8, 1152, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v44 = Tile<
                    TileType::Vec, float, 8, 1152, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v21, v11);
            // pto: %0
            uint64_t v45 = (uint64_t)v22;
            TASSIGN(v44, v45);
            // pto: %29
            int64_t v46 = (int64_t)((uint64_t)v26 + (uint64_t)v12);
            // pto: %38
            int64_t v47 = static_cast<int64_t>(v5->rankId);
            // pto: %39, %40, %41, %42
            int64_t v48 = static_cast<int64_t>(v5->windowsIn[v47]);
            // pto: %34, %35, %37, %43, %44
            int64_t v49 =
                static_cast<int64_t>(v5->windowsIn[((int64_t)((uint64_t)(v24 / v18) * (uint64_t)v18)) + (uint64_t)i27]);
            // pto: %50
            pto::Shape<1, 1, 1, 8, 1152> v50 = pto::Shape<1, 1, 1, 8, 1152>();
            // pto: %50
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v51 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %45, %46, %48, %30, %50
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v52 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(
                    (v1 + (int64_t)((uint64_t)v49 - (uint64_t)v48) / v14) + (v22 + (v46 < v22 ? v22 : v46)), v50, v51
                );
            // pto: %51
            pto::Shape<1, 1, 1, 8, 1152> v53 = pto::Shape<1, 1, 1, 8, 1152>();
            // pto: %51
            pto::Stride<517120, 517120, 517120, 64640, 1> v54 = pto::Stride<517120, 517120, 517120, 64640, 1>();
            // pto: %31, %51
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<517120, 517120, 517120, 64640, 1>, pto::Layout::ND>
                v55 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<517120, 517120, 517120, 64640, 1>,
                    pto::Layout::ND>(v2 + (v12 + (v28 < v22 ? v22 : v28) * v19), v53, v54);
            pipe_barrier(PIPE_ALL);
            pipe_barrier(PIPE_MTE2);
            pipe_barrier(PIPE_MTE3);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
            pto::comm::TPUT(v52, v55, v44);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
            pipe_barrier(PIPE_ALL);
            __gm__ float *v56 = PTOAS__GLOBAL_TENSOR_DATA(v52);
            PTOAS__DCCI_SINGLE_CACHE_LINE(v56);
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    for (int64_t i57 = v22; i57 < v18; i57 += v20) {
        // pto: %55
        if (i57 != v25) {
            // pto: %61
            int64_t v58 = static_cast<int64_t>(v6->rankId);
            // pto: %62, %63, %64, %65
            int64_t v59 = static_cast<int64_t>(v6->windowsIn[v58]);
            // pto: %57, %58, %60, %66, %67
            int64_t v60 =
                static_cast<int64_t>(v6->windowsIn[((int64_t)((uint64_t)(v24 / v18) * (uint64_t)v18)) + (uint64_t)i57]);
            // pto: %lm_head_logits_done__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 1> v61 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %lm_head_logits_done__ssa_v0_peer_pview
            pto::Stride<1, 1, 1, 1, 2> v62 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %68, %69, %71, %75, %lm_head_logits_done__ssa_v0_peer_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v63 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    (v3 + (int64_t)((uint64_t)v60 - (uint64_t)v59) / v14) + (v22 + (v25 < v22 ? v22 : v25)), v61, v62
                );
            pipe_barrier(PIPE_ALL);
            pto::comm::TNOTIFY(v63, v10, v9);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: lm_head_logits_window__ssa_v0
    __gm__ Tensor *lm_head_logits_window__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *lm_head_logits_window__ssa_v0 =
        reinterpret_cast<__gm__ float *>(lm_head_logits_window__ssa_v0_tensor->buffer.addr) +
        lm_head_logits_window__ssa_v0_tensor->start_offset;

    // Unpack tensor: logits_shards_inline40_inline13289__rv_v2
    __gm__ Tensor *logits_shards_inline40_inline13289__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *logits_shards_inline40_inline13289__rv_v2 =
        reinterpret_cast<__gm__ float *>(logits_shards_inline40_inline13289__rv_v2_tensor->buffer.addr) +
        logits_shards_inline40_inline13289__rv_v2_tensor->start_offset;

    // Unpack tensor: lm_head_logits_done__ssa_v0
    __gm__ Tensor *lm_head_logits_done__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *lm_head_logits_done__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(lm_head_logits_done__ssa_v0_tensor->buffer.addr) +
        lm_head_logits_done__ssa_v0_tensor->start_offset;

    // Unpack scalar: my_rank__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } my_rank__ssa_v0_conv;
    my_rank__ssa_v0_conv.u64 = args[3];
    int32_t my_rank__ssa_v0 = my_rank__ssa_v0_conv.val;

    // Unpack CommContext: lm_head_logits_window_ctx
    __gm__ CommContext *lm_head_logits_window_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    // Unpack CommContext: lm_head_logits_done_ctx
    __gm__ CommContext *lm_head_logits_done_ctx = reinterpret_cast<__gm__ CommContext *>(args[5]);

    // Forward to ptoas-generated function
    lm_head_combine_push(
        lm_head_logits_window__ssa_v0, logits_shards_inline40_inline13289__rv_v2, lm_head_logits_done__ssa_v0,
        my_rank__ssa_v0, lm_head_logits_window_ctx, lm_head_logits_done_ctx, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
