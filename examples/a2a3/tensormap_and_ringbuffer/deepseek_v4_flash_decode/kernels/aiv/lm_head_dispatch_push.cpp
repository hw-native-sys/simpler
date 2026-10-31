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
// Kernel Function: lm_head_dispatch_push

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

static __aicore__ void lm_head_dispatch_push(
    __gm__ bfloat16_t *v1, __gm__ int32_t *v2, __gm__ bfloat16_t *v3, __gm__ bfloat16_t *v4, __gm__ int32_t *v5,
    int32_t v6, __gm__ CommContext *v7, __gm__ CommContext *v8, int64_t v9, int32_t v10, int32_t v11
) {
    pto::comm::NotifyOp v12 = pto::comm::NotifyOp::AtomicAdd;
    const int32_t v13 = 1;
    const int64_t v14 = 4;
    const bfloat16_t v15 = 0.0f;
    const int64_t v16 = 2;
    const int64_t v17 = 8;
    const int64_t v18 = 1;
    const int64_t v19 = 4096;
    const int64_t v20 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %row_inline43_inline13303__ssa_v0
    int64_t v21 = (int64_t)v10;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    // pto: %source_row_raw_inline24_inline13301__tile
    int32_t v22 = (v2)[v21];
    // pto: %1
    int64_t v23 = (int64_t)v22;
    // pto: %2
    int64_t v24 = (int64_t)((uint64_t)v9 - (uint64_t)v18);
    // pto: %3
    int64_t v25 = v23 < v24 ? v23 : v24;
    // pto: %4
    int64_t v26 = v25 < v20 ? v20 : v25;
    // pto: %t__tile
    Tile<
        TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v27 = Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v19);
    // pto: %t__tile
    uint64_t v28 = (uint64_t)v20;
    TASSIGN(v27, v28);
    TEXPANDS(v27, v15);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %selected_hidden_inline51_inline13294__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v29 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %selected_hidden_inline51_inline13294__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v30 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %5, %selected_hidden_inline51_inline13294__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
        v31 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
            v3 + (v20 + (v21 < v20 ? v20 : v21) * v19), v29, v30
        );
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v31, v27);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %7
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    if (v23 >= v20) {
        // pto: %0
        Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v32 = Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v19);
        // pto: %0
        uint64_t v33 = (uint64_t)v20;
        TASSIGN(v32, v33);
        // pto: %x_out__rv_v2_pview
        pto::Shape<1, 1, 1, 1, 4096> v34 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %x_out__rv_v2_pview
        pto::Stride<4096, 4096, 4096, 4096, 1> v35 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %9, %x_out__rv_v2_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v36 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v1 + (v20 + (v26 < v20 ? v20 : v26) * v19), v34, v35
            );
        TLOAD(v32, v36);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
        TSTORE(v31, v32);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int64_t i37 = v20; i37 < v16; i37 += v18) {
        // pto: %tput_stage
        Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v38 = Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v18, v19);
        // pto: %tput_stage
        uint64_t v39 = (uint64_t)v20;
        TASSIGN(v38, v39);
        // pto: %12
        int64_t v40 = (int64_t)v6;
        // pto: %13, %15, %16
        int64_t v41 = (int64_t)((uint64_t)((int64_t)((uint64_t)(v40 % v16) * (uint64_t)v17)) + (uint64_t)v21);
        // pto: %24
        int64_t v42 = static_cast<int64_t>(v7->rankId);
        // pto: %25, %26, %27, %28
        int64_t v43 = static_cast<int64_t>(v7->windowsIn[v42]);
        // pto: %20, %21, %23, %29, %30
        int64_t v44 =
            static_cast<int64_t>(v7->windowsIn[((int64_t)((uint64_t)(v40 / v16) * (uint64_t)v16)) + (uint64_t)i37]);
        // pto: %lm_head_hidden_window__ssa_v0_peer_pview
        pto::Shape<1, 1, 1, 1, 4096> v45 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %lm_head_hidden_window__ssa_v0_peer_pview
        pto::Stride<4096, 4096, 4096, 4096, 1> v46 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %31, %32, %34, %17, %lm_head_hidden_window__ssa_v0_peer_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v47 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                (v4 + (int64_t)((uint64_t)v44 - (uint64_t)v43) / v16) + (v20 + (v41 < v20 ? v20 : v41) * v19), v45, v46
            );
        pipe_barrier(PIPE_ALL);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        pipe_barrier(PIPE_MTE2);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
        pto::comm::TPUT(v47, v31, v38);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        pipe_barrier(PIPE_ALL);
        __gm__ bfloat16_t *v48 = PTOAS__GLOBAL_TENSOR_DATA(v47);
        PTOAS__DCCI_SINGLE_CACHE_LINE(v48);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }
    for (int64_t i49 = v20; i49 < v16; i49 += v18) {
        // pto: %36
        int64_t v50 = (int64_t)v6;
        // pto: %37
        int64_t v51 = v50 % v16;
        // pto: %39
        if (i49 != v51) {
            // pto: %45
            int64_t v52 = static_cast<int64_t>(v8->rankId);
            // pto: %46, %47, %48, %49
            int64_t v53 = static_cast<int64_t>(v8->windowsIn[v52]);
            // pto: %41, %42, %44, %50, %51
            int64_t v54 =
                static_cast<int64_t>(v8->windowsIn[((int64_t)((uint64_t)(v50 / v16) * (uint64_t)v16)) + (uint64_t)i49]);
            // pto: %lm_head_hidden_done__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 1> v55 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %lm_head_hidden_done__ssa_v0_peer_pview
            pto::Stride<1, 1, 1, 1, 2> v56 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %52, %53, %55, %59, %lm_head_hidden_done__ssa_v0_peer_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v57 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    (v5 + (int64_t)((uint64_t)v54 - (uint64_t)v53) / v14) + (v20 + (v51 < v20 ? v20 : v51)), v55, v56
                );
            pipe_barrier(PIPE_ALL);
            pto::comm::TNOTIFY(v57, v13, v12);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
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

    // Unpack tensor: x_out__rv_v2
    __gm__ Tensor *x_out__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *x_out__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(x_out__rv_v2_tensor->buffer.addr) + x_out__rv_v2_tensor->start_offset;

    // Unpack tensor: logit_row_indices__ssa_v0
    __gm__ Tensor *logit_row_indices__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *logit_row_indices__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(logit_row_indices__ssa_v0_tensor->buffer.addr) +
        logit_row_indices__ssa_v0_tensor->start_offset;

    // Unpack tensor: selected_hidden_inline51_inline13294__ssa_v0
    __gm__ Tensor *selected_hidden_inline51_inline13294__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *selected_hidden_inline51_inline13294__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(selected_hidden_inline51_inline13294__ssa_v0_tensor->buffer.addr) +
        selected_hidden_inline51_inline13294__ssa_v0_tensor->start_offset;

    // Unpack tensor: lm_head_hidden_window__ssa_v0
    __gm__ Tensor *lm_head_hidden_window__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ bfloat16_t *lm_head_hidden_window__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(lm_head_hidden_window__ssa_v0_tensor->buffer.addr) +
        lm_head_hidden_window__ssa_v0_tensor->start_offset;

    // Unpack tensor: lm_head_hidden_done__ssa_v0
    __gm__ Tensor *lm_head_hidden_done__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *lm_head_hidden_done__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(lm_head_hidden_done__ssa_v0_tensor->buffer.addr) +
        lm_head_hidden_done__ssa_v0_tensor->start_offset;

    // Unpack scalar: my_rank__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } my_rank__ssa_v0_conv;
    my_rank__ssa_v0_conv.u64 = args[5];
    int32_t my_rank__ssa_v0 = my_rank__ssa_v0_conv.val;

    // Unpack CommContext: lm_head_hidden_window_ctx
    __gm__ CommContext *lm_head_hidden_window_ctx = reinterpret_cast<__gm__ CommContext *>(args[6]);

    // Unpack CommContext: lm_head_hidden_done_ctx
    __gm__ CommContext *lm_head_hidden_done_ctx = reinterpret_cast<__gm__ CommContext *>(args[7]);

    // Extract dynamic dim: T_DYN
    int64_t T_DYN = static_cast<int64_t>(x_out__rv_v2_tensor->shapes[0]);

    // Forward to ptoas-generated function
    lm_head_dispatch_push(
        x_out__rv_v2, logit_row_indices__ssa_v0, selected_hidden_inline51_inline13294__ssa_v0,
        lm_head_hidden_window__ssa_v0, lm_head_hidden_done__ssa_v0, my_rank__ssa_v0, lm_head_hidden_window_ctx,
        lm_head_hidden_done_ctx, T_DYN, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
