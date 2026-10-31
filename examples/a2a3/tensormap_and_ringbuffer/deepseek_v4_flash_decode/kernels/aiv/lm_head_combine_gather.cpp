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
// Kernel Function: lm_head_combine_gather

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

static __aicore__ void
lm_head_combine_gather(__gm__ float *v1, __gm__ float *v2, __gm__ CommContext *v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 1152;
    const int64_t v7 = 63488;
    const int64_t v8 = 7;
    const int64_t v9 = 2048;
    const int64_t v10 = 24;
    const int64_t v11 = 31;
    const int64_t v12 = 64640;
    const int64_t v13 = 2;
    const int64_t v14 = 1;
    const int64_t v15 = 8;
    const int64_t v16 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %gblk_inline7_inline13334__ssa_v0
    int64_t v17 = (int64_t)v4;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int64_t i18 = v16; i18 < v13; i18 += v14) {
        // pto: %1
        int64_t v19 = (int64_t)((uint64_t)i18 * (uint64_t)v12);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        for (int64_t j20 = v17; j20 < v11; j20 += v10) {
            // pto: %3, %2
            int64_t v21 = (int64_t)((uint64_t)v19 + (uint64_t)((int64_t)((uint64_t)j20 * (uint64_t)v9)));
            // pto: %t__tile
            Tile<
                TileType::Vec, float, 8, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v22 = Tile<
                    TileType::Vec, float, 8, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v15, v9);
            // pto: %t__tile
            uint64_t v23 = (uint64_t)v16;
            TASSIGN(v22, v23);
            // pto: %4
            int64_t v24 = v21 < v16 ? v16 : v21;
            // pto: %lm_head_logits_window__ssa_v0_pview
            pto::Shape<1, 1, 1, 8, 2048> v25 = pto::Shape<1, 1, 1, 8, 2048>();
            // pto: %lm_head_logits_window__ssa_v0_pview
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v26 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %lm_head_logits_window__ssa_v0_pview
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v27 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(v2 + (v16 + v24), v25, v26);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            TLOAD(v22, v27);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %logits__iter_v3_pview
            pto::Shape<1, 1, 1, 8, 2048> v28 = pto::Shape<1, 1, 1, 8, 2048>();
            // pto: %logits__iter_v3_pview
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v29 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %logits__iter_v3_pview
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v30 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 2048>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(v1 + (v16 + v24), v28, v29);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(v30, v22);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        // pto: %6
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        if (v17 == v8) {
            // pto: %7
            int64_t v31 = (int64_t)((uint64_t)v19 + (uint64_t)v7);
            // pto: %0
            Tile<
                TileType::Vec, float, 8, 1152, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v32 = Tile<
                    TileType::Vec, float, 8, 1152, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v15, v6);
            // pto: %0
            uint64_t v33 = (uint64_t)v16;
            TASSIGN(v32, v33);
            // pto: %8
            int64_t v34 = v31 < v16 ? v16 : v31;
            // pto: %9
            pto::Shape<1, 1, 1, 8, 1152> v35 = pto::Shape<1, 1, 1, 8, 1152>();
            // pto: %9
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v36 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %9
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v37 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(v2 + (v16 + v34), v35, v36);
            TLOAD(v32, v37);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            // pto: %logits__rv_v4_pview
            pto::Shape<1, 1, 1, 8, 1152> v38 = pto::Shape<1, 1, 1, 8, 1152>();
            // pto: %logits__rv_v4_pview
            pto::Stride<1034240, 1034240, 1034240, 129280, 1> v39 = pto::Stride<1034240, 1034240, 1034240, 129280, 1>();
            // pto: %logits__rv_v4_pview
            GlobalTensor<
                float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>, pto::Layout::ND>
                v40 = GlobalTensor<
                    float, pto::Shape<1, 1, 1, 8, 1152>, pto::Stride<1034240, 1034240, 1034240, 129280, 1>,
                    pto::Layout::ND>(v1 + (v16 + v34), v38, v39);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            TSTORE(v40, v32);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: logits__ssa_v0
    __gm__ Tensor *logits__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *logits__ssa_v0 =
        reinterpret_cast<__gm__ float *>(logits__ssa_v0_tensor->buffer.addr) + logits__ssa_v0_tensor->start_offset;

    // Unpack tensor: lm_head_logits_window__ssa_v0
    __gm__ Tensor *lm_head_logits_window__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *lm_head_logits_window__ssa_v0 =
        reinterpret_cast<__gm__ float *>(lm_head_logits_window__ssa_v0_tensor->buffer.addr) +
        lm_head_logits_window__ssa_v0_tensor->start_offset;

    // Unpack CommContext: lm_head_logits_window_ctx
    __gm__ CommContext *lm_head_logits_window_ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);

    // Forward to ptoas-generated function
    lm_head_combine_gather(
        logits__ssa_v0, lm_head_logits_window__ssa_v0, lm_head_logits_window_ctx, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
