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
// Kernel Function: combine_3

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

static __aicore__ void combine_3(
    __gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ bfloat16_t *v3, __gm__ bfloat16_t *v4, __gm__ CommContext *v5,
    int32_t v6, int32_t v7
) {
    const int64_t v9 = 4096;
    const int64_t v10 = 16;
    const int64_t v11 = 1;
    const int64_t v12 = 32;
    const int64_t v13 = 2;
    const int64_t v14 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %e_inline2820_inline12944__ssa_v0
    int64_t v15 = (int64_t)v6;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %b_inline2829_inline12869__rv_v2
    int64_t v16;
    v16 = v14;
    for (int64_t i17 = v14; i17 < v13; i17 += v11) {
        // pto: %b_inline2829_inline12869__rv_v2
        int64_t v18 = v16;
        // pto: %flat_offset_mul, %flat_offset, %t__tile
        int32_t v19 = (v1)[(int64_t)((uint64_t)((int64_t)((uint64_t)i17 * (uint64_t)v12)) + (uint64_t)v15)];
        // pto: %2
        int64_t v20 = (int64_t)v19;
        for (int64_t j21 = v14; j21 < v20; j21 += v11) {
            // pto: %0, %6, %3
            int64_t v22 = (int64_t)((uint64_t)((int64_t)((uint64_t)v15 * (uint64_t)v10)) +
                                    (uint64_t)((int64_t)((uint64_t)v18 + (uint64_t)j21)));
            // pto: %4
            int32_t v23 = (v2)[v22];
            // pto: %7
            int64_t v24 = (int64_t)v23;
            // pto: %tput_stage
            Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v25 = Tile<
                    TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v11, v9);
            // pto: %tput_stage
            uint64_t v26 = (uint64_t)v14;
            TASSIGN(v25, v26);
            // pto: %11
            int64_t v27 = static_cast<int64_t>(v5->rankId);
            // pto: %12, %13, %14, %15
            int64_t v28 = static_cast<int64_t>(v5->windowsIn[v27]);
            // pto: %16, %17
            int64_t v29 = static_cast<int64_t>(v5->windowsIn[i17]);
            // pto: %routed_y_buf__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 4096> v30 = pto::Shape<1, 1, 1, 1, 4096>();
            // pto: %routed_y_buf__ssa_v0_peer_pview
            pto::Stride<4096, 4096, 4096, 4096, 1> v31 = pto::Stride<4096, 4096, 4096, 4096, 1>();
            // pto: %18, %19, %21, %8, %routed_y_buf__ssa_v0_peer_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                v32 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                    (v3 + (int64_t)((uint64_t)v29 - (uint64_t)v28) / v13) + (v14 + (v24 < v14 ? v14 : v24) * v9), v30,
                    v31
                );
            // pto: %recv_y_flat_inline2828_inline12871__ssa_v0_local_pview
            pto::Shape<1, 1, 1, 1, 4096> v33 = pto::Shape<1, 1, 1, 1, 4096>();
            // pto: %recv_y_flat_inline2828_inline12871__ssa_v0_local_pview
            pto::Stride<4096, 4096, 4096, 4096, 1> v34 = pto::Stride<4096, 4096, 4096, 4096, 1>();
            // pto: %10, %recv_y_flat_inline2828_inline12871__ssa_v0_local_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                v35 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                    v4 + (v14 + (v22 < v14 ? v14 : v22) * v9), v33, v34
                );
            pipe_barrier(PIPE_ALL);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            pipe_barrier(PIPE_MTE2);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            pto::comm::TPUT(v32, v35, v25);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            pipe_barrier(PIPE_ALL);
            __gm__ bfloat16_t *v36 = PTOAS__GLOBAL_TENSOR_DATA(v32);
            PTOAS__DCCI_SINGLE_CACHE_LINE(v36);
            pipe_barrier(PIPE_ALL);
            dsb(DSB_DDR);
        }
        // pto: %23
        v16 = (int64_t)((uint64_t)v18 + (uint64_t)v20);
    }
    // pto: %b_inline2829_inline12869__rv_v2
    int64_t v37 = v16;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: recv_meta_local_inline13136__ssa_v0
    __gm__ Tensor *recv_meta_local_inline13136__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *recv_meta_local_inline13136__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_meta_local_inline13136__ssa_v0_tensor->buffer.addr) +
        recv_meta_local_inline13136__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_r_route_out_inline13100__ssa_v0
    __gm__ Tensor *recv_r_route_out_inline13100__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *recv_r_route_out_inline13100__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_r_route_out_inline13100__ssa_v0_tensor->buffer.addr) +
        recv_r_route_out_inline13100__ssa_v0_tensor->start_offset;

    // Unpack tensor: routed_y_buf__ssa_v0
    __gm__ Tensor *routed_y_buf__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *routed_y_buf__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(routed_y_buf__ssa_v0_tensor->buffer.addr) +
        routed_y_buf__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_y_flat_inline2828_inline12871__ssa_v0
    __gm__ Tensor *recv_y_flat_inline2828_inline12871__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ bfloat16_t *recv_y_flat_inline2828_inline12871__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(recv_y_flat_inline2828_inline12871__ssa_v0_tensor->buffer.addr) +
        recv_y_flat_inline2828_inline12871__ssa_v0_tensor->start_offset;

    // Unpack CommContext: routed_y_buf_ctx
    __gm__ CommContext *routed_y_buf_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    // Forward to ptoas-generated function
    combine_3(
        recv_meta_local_inline13136__ssa_v0, recv_r_route_out_inline13100__ssa_v0, routed_y_buf__ssa_v0,
        recv_y_flat_inline2828_inline12871__ssa_v0, routed_y_buf_ctx, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
