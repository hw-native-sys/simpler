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
// Kernel Function: dispatch_gather

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

static __aicore__ void dispatch_gather(
    __gm__ int8_t *v1, __gm__ int32_t *v2, __gm__ int8_t *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ int32_t *v7, __gm__ int32_t *v8, __gm__ CommContext *v9, __gm__ CommContext *v10, __gm__ CommContext *v11,
    int32_t v12, int32_t v13
) {
    const int64_t v14 = 16;
    const int64_t v15 = 8;
    const int64_t v16 = 32;
    const int64_t v17 = 2;
    const int64_t v18 = 1;
    const int64_t v19 = 4096;
    const int64_t v20 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %e_inline2695_inline9418__ssa_v1
    int64_t v21 = (int64_t)v12;
    // pto: %0
    int64_t v22 = (int64_t)((uint64_t)v21 * (uint64_t)v14);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %b_inline2690_inline9285__rv_v2
    int64_t v23;
    v23 = v20;
    for (int64_t i24 = v20; i24 < v17; i24 += v18) {
        // pto: %b_inline2690_inline9285__rv_v2
        int64_t v25 = v23;
        // pto: %flat_offset_mul, %flat_offset, %2
        int32_t v26 = (v2)[(int64_t)((uint64_t)((int64_t)((uint64_t)i24 * (uint64_t)v16)) + (uint64_t)v21)];
        // pto: %3
        int64_t v27 = (int64_t)v26;
        for (int64_t j28 = v20; j28 < v27; j28 += v18) {
            // pto: %5, %4, %6
            int64_t v29 =
                (int64_t)((uint64_t)((int64_t)((uint64_t)v22 + (uint64_t)((int64_t)((uint64_t)i24 * (uint64_t)v15)))) +
                          (uint64_t)j28);
            // pto: %8, %7
            int64_t v30 = (int64_t)((uint64_t)v22 + (uint64_t)((int64_t)((uint64_t)v25 + (uint64_t)j28)));
            // pto: %t__tile
            Tile<
                TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v31 = Tile<
                    TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v18, v19);
            // pto: %t__tile
            uint64_t v32 = (uint64_t)v20;
            TASSIGN(v31, v32);
            // pto: %recv_x__ssa_v0_pview
            pto::Shape<1, 1, 1, 1, 4096> v33 = pto::Shape<1, 1, 1, 1, 4096>();
            // pto: %recv_x__ssa_v0_pview
            pto::Stride<4096, 4096, 4096, 4096, 1> v34 = pto::Stride<4096, 4096, 4096, 4096, 1>();
            // pto: %9, %recv_x__ssa_v0_pview
            GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                v35 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                    v3 + (v20 + (v29 < v20 ? v20 : v29) * v19), v33, v34
                );
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(v31, v35);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            // pto: %recv_x_out_flat_inline2699_inline9324__iter_v3_pview
            pto::Shape<1, 1, 1, 1, 4096> v36 = pto::Shape<1, 1, 1, 1, 4096>();
            // pto: %recv_x_out_flat_inline2699_inline9324__iter_v3_pview
            pto::Stride<4096, 4096, 4096, 4096, 1> v37 = pto::Stride<4096, 4096, 4096, 4096, 1>();
            // pto: %10, %recv_x_out_flat_inline2699_inline9324__iter_v3_pview
            GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                v38 = GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                    v1 + (v20 + (v30 < v20 ? v20 : v30) * v19), v36, v37
                );
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(v38, v31);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            // pto: %12
            int64_t v39 = (int64_t)((uint64_t)v29 * (uint64_t)v15);
            // pto: %11
            float v40 = (v4)[v39];
            (v5)[v30] = v40;
            // pto: %18, %16
            float v41 = (v4)[(int64_t)((uint64_t)v39 + (uint64_t)v18)];
            (v6)[v30] = v41;
            // pto: %21
            int32_t v42 = (v7)[v39];
            (v8)[v30] = v42;
        }
        // pto: %26
        v23 = (int64_t)((uint64_t)v25 + (uint64_t)v27);
    }
    // pto: %b_inline2690_inline9285__rv_v2
    int64_t v43 = v23;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
#endif  // __DAV_VEC__

    pipe_barrier(PIPE_ALL);
    dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
    dsb((mem_dsb_t)0);
    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: recv_x_out_flat_inline2699_inline9324__ssa_v0
    __gm__ Tensor *recv_x_out_flat_inline2699_inline9324__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int8_t *recv_x_out_flat_inline2699_inline9324__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(recv_x_out_flat_inline2699_inline9324__ssa_v0_tensor->buffer.addr) +
        recv_x_out_flat_inline2699_inline9324__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_meta_local_inline9405__ssa_v0
    __gm__ Tensor *recv_meta_local_inline9405__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *recv_meta_local_inline9405__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_meta_local_inline9405__ssa_v0_tensor->buffer.addr) +
        recv_meta_local_inline9405__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_x__ssa_v0
    __gm__ Tensor *recv_x__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *recv_x__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(recv_x__ssa_v0_tensor->buffer.addr) + recv_x__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_aux__ssa_v0
    __gm__ Tensor *recv_aux__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *recv_aux__ssa_v0 =
        reinterpret_cast<__gm__ float *>(recv_aux__ssa_v0_tensor->buffer.addr) + recv_aux__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_scale_out_inline9172__ssa_v0
    __gm__ Tensor *recv_scale_out_inline9172__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *recv_scale_out_inline9172__ssa_v0 =
        reinterpret_cast<__gm__ float *>(recv_scale_out_inline9172__ssa_v0_tensor->buffer.addr) +
        recv_scale_out_inline9172__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_w_out_inline9360__ssa_v0
    __gm__ Tensor *recv_w_out_inline9360__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *recv_w_out_inline9360__ssa_v0 =
        reinterpret_cast<__gm__ float *>(recv_w_out_inline9360__ssa_v0_tensor->buffer.addr) +
        recv_w_out_inline9360__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_route__ssa_v0
    __gm__ Tensor *recv_route__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ int32_t *recv_route__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(recv_route__ssa_v0_tensor->buffer.addr) +
                                         recv_route__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_r_route_out_inline9369__ssa_v0
    __gm__ Tensor *recv_r_route_out_inline9369__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ int32_t *recv_r_route_out_inline9369__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_r_route_out_inline9369__ssa_v0_tensor->buffer.addr) +
        recv_r_route_out_inline9369__ssa_v0_tensor->start_offset;

    // Unpack CommContext: recv_x_ctx
    __gm__ CommContext *recv_x_ctx = reinterpret_cast<__gm__ CommContext *>(args[8]);

    // Unpack CommContext: recv_aux_ctx
    __gm__ CommContext *recv_aux_ctx = reinterpret_cast<__gm__ CommContext *>(args[9]);

    // Unpack CommContext: recv_route_ctx
    __gm__ CommContext *recv_route_ctx = reinterpret_cast<__gm__ CommContext *>(args[10]);

    // Forward to ptoas-generated function
    dispatch_gather(
        recv_x_out_flat_inline2699_inline9324__ssa_v0, recv_meta_local_inline9405__ssa_v0, recv_x__ssa_v0,
        recv_aux__ssa_v0, recv_scale_out_inline9172__ssa_v0, recv_w_out_inline9360__ssa_v0, recv_route__ssa_v0,
        recv_r_route_out_inline9369__ssa_v0, recv_x_ctx, recv_aux_ctx, recv_route_ctx, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
