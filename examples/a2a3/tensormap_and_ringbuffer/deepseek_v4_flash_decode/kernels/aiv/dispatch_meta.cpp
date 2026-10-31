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
// Kernel Function: dispatch_meta

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

static __aicore__ void dispatch_meta(
    __gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ int32_t *v3, __gm__ int32_t *v4, __gm__ int32_t *v5, int32_t v6,
    int32_t v7, __gm__ CommContext *v8, __gm__ CommContext *v9
) {
    pto::comm::WaitCmp v10 = pto::comm::WaitCmp::GE;
    pto::comm::NotifyOp v11 = pto::comm::NotifyOp::AtomicAdd;
    const int32_t v12 = 1;
    const int64_t v13 = 4;
    const int32_t v14 = 0;
    const int64_t v15 = 32;
    const int64_t v16 = 2;
    const int64_t v17 = 1;
    const int64_t v18 = 6;
    const int64_t v19 = 8;
    const int64_t v20 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %0
    int64_t v21 = (int64_t)v6;
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    // pto: %1, %active_tokens_inline2713_inline9397__phi_v4
    // pto: %cursor_inline2708_inline9345__ssa_v0
    int32_t v22[64];
    for (int64_t i23 = v20; i23 < v16; i23 += v17) {
        for (int64_t j24 = v20; j24 < v15; j24 += v17) {
            // pto: %3, %4
            int64_t v25 = (int64_t)((uint64_t)((int64_t)((uint64_t)i23 * (uint64_t)v15)) + (uint64_t)j24);
            v22[v25] = v14;
        }
    }
    for (int64_t i26 = v20; i26 < (v21 > v19 ? v19 : v21); i26 += v17) {
        for (int64_t j27 = v20; j27 < v18; j27 += v17) {
            // pto: %flat_offset_mul, %flat_offset, %eid_inline2694_inline9410__tile
            int32_t v28 = (v1)[(int64_t)((uint64_t)((int64_t)((uint64_t)i26 * (uint64_t)v18)) + (uint64_t)j27)];
            // pto: %5
            int64_t v29 = (int64_t)v28;
            // pto: %t__tmp_v270
            // pto: %t__tmp_v270
            int32_t v30 = v22[v29];
            // pto: %14, %15, %16
            v22[v29] = (int32_t)((int64_t)((uint64_t)((int64_t)v30) + (uint64_t)v17));
        }
    }
    // pto: %meta_tile_inline2721_inline9413__ssa_v0
    Tile<
        TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v31 = Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v17, v15);
    // pto: %meta_tile_inline2721_inline9413__ssa_v0
    uint64_t v32 = (uint64_t)v20;
    TASSIGN(v31, v32);
    TEXPANDS(v31, v14);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int64_t i33 = v20; i33 < v16; i33 += v17) {
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        for (int64_t j34 = v20; j34 < v15; j34 += v17) {
            // pto: %17, %18
            int64_t v35 = (int64_t)((uint64_t)((int64_t)((uint64_t)i33 * (uint64_t)v15)) + (uint64_t)j34);
            // pto: %t__tmp_v271
            // pto: %t__tmp_v271
            int32_t v36 = v22[v35];
            v31.SetValue(j34, v36);
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        // pto: %21
        int64_t v37 = static_cast<int64_t>(v8->rankId);
        // pto: %22, %23, %24, %25
        int64_t v38 = static_cast<int64_t>(v8->windowsIn[v37]);
        // pto: %26
        // pto: %27
        int64_t v40 = static_cast<int64_t>(v8->windowsIn[i33]);
        // pto: %my_rank__ssa_v0_idx
        int64_t v41 = (int64_t)v7;
        // pto: %33
        int64_t v42 = v41 < v20 ? v20 : v41;
        // pto: %recv_meta__ssa_v0_peer_pview
        pto::Shape<1, 1, 1, 1, 32> v43 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %recv_meta__ssa_v0_peer_pview
        pto::Stride<32, 32, 32, 32, 1> v44 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %28, %29, %31, %recv_meta__ssa_v0_peer_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v45 =
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                (v2 + (int64_t)((uint64_t)v40 - (uint64_t)v38) / v13) + (v20 + v42 * v15), v43, v44
            );
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v45, v31);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        __gm__ int32_t *v46 = PTOAS__GLOBAL_TENSOR_DATA(v45);
        PTOAS__DCCI_SINGLE_CACHE_LINE(v46);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        // pto: %35
        if (i33 != v41) {
            // pto: %36
            int64_t v47 = static_cast<int64_t>(v9->rankId);
            // pto: %37, %38, %39, %40
            int64_t v48 = static_cast<int64_t>(v9->windowsIn[v47]);
            // pto: %42
            int64_t v49 = static_cast<int64_t>(v9->windowsIn[i33]);
            // pto: %arrived__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 1> v50 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %arrived__ssa_v0_peer_pview
            pto::Stride<1, 1, 1, 1, 2> v51 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %43, %44, %46, %arrived__ssa_v0_peer_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v52 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    (v3 + (int64_t)((uint64_t)v49 - (uint64_t)v48) / v13) + (v20 + v42), v50, v51
                );
            pipe_barrier(PIPE_ALL);
            pto::comm::TNOTIFY(v52, v12, v11);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    for (int64_t i53 = v20; i53 < v16; i53 += v17) {
        // pto: %50, %51
        if (i53 != (int64_t)v7) {
            // pto: %arrived__ssa_v0_local_pview
            pto::Shape<1, 1, 1, 1, 1> v54 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %arrived__ssa_v0_local_pview
            pto::Stride<1, 1, 1, 1, 2> v55 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %52, %arrived__ssa_v0_local_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v56 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    v3 + (v20 + (i53 < v20 ? v20 : i53)), v54, v55
                );
            pto::comm::TWAIT(v56, v12, v10);
            dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    for (int64_t i57 = v20; i57 < v15; i57 += v17) {
        // pto: %acc_inline2718_inline9457__rv_v2
        int32_t v58;
        v58 = v14;
        for (int64_t j59 = v20; j59 < v16; j59 += v17) {
            // pto: %acc_inline2718_inline9457__rv_v2
            int32_t v60 = v58;
            // pto: %54, %55
            int64_t v61 = (int64_t)((uint64_t)((int64_t)((uint64_t)j59 * (uint64_t)v15)) + (uint64_t)i57);
            // pto: %count_inline2688_inline9411__tile
            int32_t v62 = (v2)[v61];
            (v4)[v61] = v62;
            // pto: %58
            v58 = (int32_t)((uint32_t)v60 + (uint32_t)v62);
        }
        // pto: %acc_inline2718_inline9457__rv_v2
        int32_t v63 = v58;
        (v5)[i57] = v63;
    }
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
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
    // Unpack tensor: indices_inline9286__ssa_v0
    __gm__ Tensor *indices_inline9286__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *indices_inline9286__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(indices_inline9286__ssa_v0_tensor->buffer.addr) +
        indices_inline9286__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_meta__ssa_v0
    __gm__ Tensor *recv_meta__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *recv_meta__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(recv_meta__ssa_v0_tensor->buffer.addr) +
                                        recv_meta__ssa_v0_tensor->start_offset;

    // Unpack tensor: arrived__ssa_v0
    __gm__ Tensor *arrived__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *arrived__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(arrived__ssa_v0_tensor->buffer.addr) + arrived__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_meta_local_inline9405__ssa_v0
    __gm__ Tensor *recv_meta_local_inline9405__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *recv_meta_local_inline9405__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_meta_local_inline9405__ssa_v0_tensor->buffer.addr) +
        recv_meta_local_inline9405__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_count_out_inline9338__ssa_v0
    __gm__ Tensor *recv_count_out_inline9338__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *recv_count_out_inline9338__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_count_out_inline9338__ssa_v0_tensor->buffer.addr) +
        recv_count_out_inline9338__ssa_v0_tensor->start_offset;

    // Unpack scalar: nt_inline677__rv_v2
    union {
        uint64_t u64;
        int32_t val;
    } nt_inline677__rv_v2_conv;
    nt_inline677__rv_v2_conv.u64 = args[5];
    int32_t nt_inline677__rv_v2 = nt_inline677__rv_v2_conv.val;

    // Unpack scalar: my_rank__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } my_rank__ssa_v0_conv;
    my_rank__ssa_v0_conv.u64 = args[6];
    int32_t my_rank__ssa_v0 = my_rank__ssa_v0_conv.val;

    // Unpack CommContext: recv_meta_ctx
    __gm__ CommContext *recv_meta_ctx = reinterpret_cast<__gm__ CommContext *>(args[7]);

    // Unpack CommContext: arrived_ctx
    __gm__ CommContext *arrived_ctx = reinterpret_cast<__gm__ CommContext *>(args[8]);

    // Forward to ptoas-generated function
    dispatch_meta(
        indices_inline9286__ssa_v0, recv_meta__ssa_v0, arrived__ssa_v0, recv_meta_local_inline9405__ssa_v0,
        recv_count_out_inline9338__ssa_v0, nt_inline677__rv_v2, my_rank__ssa_v0, recv_meta_ctx, arrived_ctx
    );
}
