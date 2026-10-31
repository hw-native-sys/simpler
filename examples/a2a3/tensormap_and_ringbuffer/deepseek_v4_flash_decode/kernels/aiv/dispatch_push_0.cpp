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
// Kernel Function: dispatch_push_0

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

static __aicore__ void dispatch_push_0(
    __gm__ int32_t *v1, __gm__ int8_t *v2, __gm__ int8_t *v3, __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ int32_t *v7, __gm__ int32_t *v8, int32_t v9, int32_t v10, __gm__ CommContext *v11, __gm__ CommContext *v12,
    __gm__ CommContext *v13, __gm__ CommContext *v14, int32_t v15, int32_t v16
) {
    pto::comm::NotifyOp v17 = pto::comm::NotifyOp::AtomicAdd;
    const int32_t v18 = 1;
    const int64_t v19 = 4;
    const int64_t v20 = 32;
    const float v21 = 0.0f;
    const int64_t v22 = 16;
    const int32_t v23 = 0;
    const int64_t v24 = 2;
    const int64_t v25 = 1;
    const int64_t v26 = 6;
    const int64_t v27 = 8;
    const int64_t v28 = 0;
    const int64_t v29 = 4128;
    const int64_t v30 = 4096;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %loc_e_inline2717_inline10140__ssa_v1
    int64_t v31 = (int64_t)v15;
    // pto: %0
    int64_t v32 = (int64_t)v9;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    // pto: %1, %active_tokens_v1_inline2691_inline10184__phi_v4
    // pto: %slot_ctr_inline2697_inline10186__ssa_v0
    int32_t v33[2];
    for (int64_t i34 = v28; i34 < v24; i34 += v25) {
        v33[i34] = v23;
    }
    // pto: %4
    int64_t v35 = (int64_t)v10;
    // pto: %aux_tile_inline2693_inline10115__ssa_v0
    Tile<
        TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>
        v36 = Tile<
            TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v25, v27);
    // pto: %aux_tile_inline2693_inline10115__ssa_v0
    uint64_t v37 = (uint64_t)v30;
    TASSIGN(v36, v37);
    TEXPANDS(v36, v21);
    // pto: %route_tile_inline2714_inline10190__ssa_v0
    Tile<
        TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v38 = Tile<
            TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v25, v27);
    // pto: %route_tile_inline2714_inline10190__ssa_v0
    uint64_t v39 = (uint64_t)v29;
    TASSIGN(v38, v39);
    TEXPANDS(v38, v23);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (int64_t i40 = v28; i40 < (v32 > v27 ? v27 : v32); i40 += v25) {
        for (int64_t j41 = v28; j41 < v26; j41 += v25) {
            // pto: %flat_offset_mul, %flat_offset
            int64_t v42 = (int64_t)((uint64_t)((int64_t)((uint64_t)i40 * (uint64_t)v26)) + (uint64_t)j41);
            // pto: %eid_inline2694_inline10172__tile
            int32_t v43 = (v1)[v42];
            // pto: %7
            int64_t v44 = (int64_t)v43;
            // pto: %8
            int64_t v45 = v44 / v20;
            // pto: %11, %10, %12
            if ((int64_t)((uint64_t)v44 - (uint64_t)((int64_t)((uint64_t)v45 * (uint64_t)v20))) == v31) {
                // pto: %slot_inline2686_inline10021__ssa_v0
                // pto: %slot_inline2686_inline10021__ssa_v0
                int32_t v46 = v33[v45];
                // pto: %13
                int64_t v47 = (int64_t)v46;
                // pto: %14, %15
                v33[v45] = (int32_t)((int64_t)((uint64_t)v47 + (uint64_t)v25));
                // pto: %3, %6, %5, %17
                int64_t v48 = (int64_t)((uint64_t)((int64_t)((uint64_t)((int64_t)((uint64_t)v31 * (uint64_t)v22)) +
                                                             (uint64_t)((int64_t)((uint64_t)v35 * (uint64_t)v27)))) +
                                        (uint64_t)v47);
                // pto: %tput_stage
                Tile<
                    TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>
                    v49 = Tile<
                        TileType::Vec, int8_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                        PadValue::Null, CompactMode::Null>(v25, v30);
                // pto: %tput_stage
                uint64_t v50 = (uint64_t)v28;
                TASSIGN(v49, v50);
                // pto: %18
                int64_t v51 = v48 < v28 ? v28 : v48;
                // pto: %20
                int64_t v52 = static_cast<int64_t>(v11->rankId);
                // pto: %21, %22, %23, %24
                int64_t v53 = static_cast<int64_t>(v11->windowsIn[v52]);
                // pto: %25
                // pto: %26
                int64_t v55 = static_cast<int64_t>(v11->windowsIn[v45]);
                // pto: %recv_x__ssa_v0_peer_pview
                pto::Shape<1, 1, 1, 1, 4096> v56 = pto::Shape<1, 1, 1, 1, 4096>();
                // pto: %recv_x__ssa_v0_peer_pview
                pto::Stride<4096, 4096, 4096, 4096, 1> v57 = pto::Stride<4096, 4096, 4096, 4096, 1>();
                // pto: %27, %30, %recv_x__ssa_v0_peer_pview
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                    v58 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                        (v2 + (int64_t)((uint64_t)v55 - (uint64_t)v53)) + (v28 + v51 * v30), v56, v57
                    );
                // pto: %x_norm_i8_inline10027__rv_v2_local_pview
                pto::Shape<1, 1, 1, 1, 4096> v59 = pto::Shape<1, 1, 1, 1, 4096>();
                // pto: %x_norm_i8_inline10027__rv_v2_local_pview
                pto::Stride<4096, 4096, 4096, 4096, 1> v60 = pto::Stride<4096, 4096, 4096, 4096, 1>();
                // pto: %19, %x_norm_i8_inline10027__rv_v2_local_pview
                GlobalTensor<
                    int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                    v61 = GlobalTensor<
                        int8_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                        v3 + (v28 + (i40 < v28 ? v28 : i40) * v30), v59, v60
                    );
                pipe_barrier(PIPE_ALL);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                pipe_barrier(PIPE_MTE2);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                pto::comm::TPUT(v58, v61, v49);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                pipe_barrier(PIPE_ALL);
                __gm__ int8_t *v62 = PTOAS__GLOBAL_TENSOR_DATA(v58);
                PTOAS__DCCI_SINGLE_CACHE_LINE(v62);
                pipe_barrier(PIPE_ALL);
                dsb(DSB_DDR);
                // pto: %t__tile
                float v63 = (v4)[i40];
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                v36.SetValue(v28, v63);
                // pto: %34
                float v64 = (v5)[v42];
                v36.SetValue(v25, v64);
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                // pto: %37
                int64_t v65 = static_cast<int64_t>(v12->rankId);
                // pto: %38, %39, %40, %41
                int64_t v66 = static_cast<int64_t>(v12->windowsIn[v65]);
                // pto: %43
                int64_t v67 = static_cast<int64_t>(v12->windowsIn[v45]);
                // pto: %recv_aux__ssa_v0_peer_pview
                pto::Shape<1, 1, 1, 1, 8> v68 = pto::Shape<1, 1, 1, 1, 8>();
                // pto: %recv_aux__ssa_v0_peer_pview
                pto::Stride<8, 8, 8, 8, 1> v69 = pto::Stride<8, 8, 8, 8, 1>();
                // pto: %44, %45, %47, %recv_aux__ssa_v0_peer_pview
                GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v70 =
                    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(
                        (v6 + (int64_t)((uint64_t)v67 - (uint64_t)v66) / v19) + (v28 + v51 * v27), v68, v69
                    );
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                TSTORE(v70, v36);
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                __gm__ float *v71 = PTOAS__GLOBAL_TENSOR_DATA(v70);
                PTOAS__DCCI_SINGLE_CACHE_LINE(v71);
                pipe_barrier(PIPE_ALL);
                dsb(DSB_DDR);
                // pto: %52
                int32_t v72 = (int32_t)v42;
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
                v38.SetValue(v28, v72);
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                // pto: %53
                int64_t v73 = static_cast<int64_t>(v13->rankId);
                // pto: %54, %55, %56, %57
                int64_t v74 = static_cast<int64_t>(v13->windowsIn[v73]);
                // pto: %59
                int64_t v75 = static_cast<int64_t>(v13->windowsIn[v45]);
                // pto: %recv_route__ssa_v0_peer_pview
                pto::Shape<1, 1, 1, 1, 8> v76 = pto::Shape<1, 1, 1, 1, 8>();
                // pto: %recv_route__ssa_v0_peer_pview
                pto::Stride<8, 8, 8, 8, 1> v77 = pto::Stride<8, 8, 8, 8, 1>();
                // pto: %60, %61, %63, %recv_route__ssa_v0_peer_pview
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND> v78 =
                    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>(
                        (v7 + (int64_t)((uint64_t)v75 - (uint64_t)v74) / v19) + (v28 + v51 * v27), v76, v77
                    );
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                TSTORE(v78, v38);
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
                __gm__ int32_t *v79 = PTOAS__GLOBAL_TENSOR_DATA(v78);
                PTOAS__DCCI_SINGLE_CACHE_LINE(v79);
                pipe_barrier(PIPE_ALL);
                dsb(DSB_DDR);
            }
        }
    }
    for (int64_t i80 = v28; i80 < v24; i80 += v25) {
        // pto: %67
        if (i80 != v35) {
            // pto: %68
            int64_t v81 = static_cast<int64_t>(v14->rankId);
            // pto: %69, %70, %71, %72
            int64_t v82 = static_cast<int64_t>(v14->windowsIn[v81]);
            // pto: %73, %74
            int64_t v83 = static_cast<int64_t>(v14->windowsIn[i80]);
            // pto: %data_arrived__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 1> v84 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %data_arrived__ssa_v0_peer_pview
            pto::Stride<1, 1, 1, 1, 2> v85 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %75, %76, %78, %80, %data_arrived__ssa_v0_peer_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v86 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    (v8 + (int64_t)((uint64_t)v83 - (uint64_t)v82) / v19) + (v28 + (v35 < v28 ? v28 : v35)), v84, v85
                );
            pipe_barrier(PIPE_ALL);
            pto::comm::TNOTIFY(v86, v18, v17);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: indices_inline10048__ssa_v0
    __gm__ Tensor *indices_inline10048__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *indices_inline10048__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(indices_inline10048__ssa_v0_tensor->buffer.addr) +
        indices_inline10048__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_x__ssa_v0
    __gm__ Tensor *recv_x__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int8_t *recv_x__ssa_v0 =
        reinterpret_cast<__gm__ int8_t *>(recv_x__ssa_v0_tensor->buffer.addr) + recv_x__ssa_v0_tensor->start_offset;

    // Unpack tensor: x_norm_i8_inline10027__rv_v2
    __gm__ Tensor *x_norm_i8_inline10027__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int8_t *x_norm_i8_inline10027__rv_v2 =
        reinterpret_cast<__gm__ int8_t *>(x_norm_i8_inline10027__rv_v2_tensor->buffer.addr) +
        x_norm_i8_inline10027__rv_v2_tensor->start_offset;

    // Unpack tensor: x_norm_scale_inline10007__ssa_v0
    __gm__ Tensor *x_norm_scale_inline10007__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ float *x_norm_scale_inline10007__ssa_v0 =
        reinterpret_cast<__gm__ float *>(x_norm_scale_inline10007__ssa_v0_tensor->buffer.addr) +
        x_norm_scale_inline10007__ssa_v0_tensor->start_offset;

    // Unpack tensor: weights_inline10038__ssa_v0
    __gm__ Tensor *weights_inline10038__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ float *weights_inline10038__ssa_v0 =
        reinterpret_cast<__gm__ float *>(weights_inline10038__ssa_v0_tensor->buffer.addr) +
        weights_inline10038__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_aux__ssa_v0
    __gm__ Tensor *recv_aux__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ float *recv_aux__ssa_v0 =
        reinterpret_cast<__gm__ float *>(recv_aux__ssa_v0_tensor->buffer.addr) + recv_aux__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_route__ssa_v0
    __gm__ Tensor *recv_route__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ int32_t *recv_route__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(recv_route__ssa_v0_tensor->buffer.addr) +
                                         recv_route__ssa_v0_tensor->start_offset;

    // Unpack tensor: data_arrived__ssa_v0
    __gm__ Tensor *data_arrived__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ int32_t *data_arrived__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(data_arrived__ssa_v0_tensor->buffer.addr) +
        data_arrived__ssa_v0_tensor->start_offset;

    // Unpack scalar: nt_inline677__rv_v2
    union {
        uint64_t u64;
        int32_t val;
    } nt_inline677__rv_v2_conv;
    nt_inline677__rv_v2_conv.u64 = args[8];
    int32_t nt_inline677__rv_v2 = nt_inline677__rv_v2_conv.val;

    // Unpack scalar: my_rank__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } my_rank__ssa_v0_conv;
    my_rank__ssa_v0_conv.u64 = args[9];
    int32_t my_rank__ssa_v0 = my_rank__ssa_v0_conv.val;

    // Unpack CommContext: recv_x_ctx
    __gm__ CommContext *recv_x_ctx = reinterpret_cast<__gm__ CommContext *>(args[10]);

    // Unpack CommContext: recv_aux_ctx
    __gm__ CommContext *recv_aux_ctx = reinterpret_cast<__gm__ CommContext *>(args[11]);

    // Unpack CommContext: recv_route_ctx
    __gm__ CommContext *recv_route_ctx = reinterpret_cast<__gm__ CommContext *>(args[12]);

    // Unpack CommContext: data_arrived_ctx
    __gm__ CommContext *data_arrived_ctx = reinterpret_cast<__gm__ CommContext *>(args[13]);

    // Forward to ptoas-generated function
    dispatch_push_0(
        indices_inline10048__ssa_v0, recv_x__ssa_v0, x_norm_i8_inline10027__rv_v2, x_norm_scale_inline10007__ssa_v0,
        weights_inline10038__ssa_v0, recv_aux__ssa_v0, recv_route__ssa_v0, data_arrived__ssa_v0, nt_inline677__rv_v2,
        my_rank__ssa_v0, recv_x_ctx, recv_aux_ctx, recv_route_ctx, data_arrived_ctx, __pypto_spmd_block_idx,
        __pypto_spmd_block_num
    );
}
