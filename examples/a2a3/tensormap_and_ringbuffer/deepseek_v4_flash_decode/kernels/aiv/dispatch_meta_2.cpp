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
// Kernel Function: dispatch_meta_2

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

static __aicore__ void dispatch_meta_2(
    __gm__ int32_t *v1, __gm__ int32_t *v2, __gm__ int32_t *v3, __gm__ int32_t *v4, __gm__ int32_t *v5, int32_t v6,
    int32_t v7, int32_t v8, __gm__ CommContext *v9, __gm__ CommContext *v10
) {
    pto::comm::WaitCmp v11 = pto::comm::WaitCmp::GE;
    pto::comm::NotifyOp v12 = pto::comm::NotifyOp::AtomicAdd;
    const int32_t v13 = 1;
    const int64_t v14 = 4;
    const int32_t v15 = 0;
    const int64_t v16 = 32;
    const int64_t v17 = 2;
    const int64_t v18 = 1;
    const int64_t v19 = 6;
    const int64_t v20 = 8;
    const int64_t v21 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %0
    int64_t v22 = (int64_t)v6;
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    // pto: %1, %active_tokens_inline2713_inline12083__phi_v4
    // pto: %cursor_inline2708_inline12031__ssa_v0
    int32_t v23[64];
    for (int64_t i24 = v21; i24 < v17; i24 += v18) {
        for (int64_t j25 = v21; j25 < v16; j25 += v18) {
            // pto: %3, %4
            int64_t v26 = (int64_t)((uint64_t)((int64_t)((uint64_t)i24 * (uint64_t)v16)) + (uint64_t)j25);
            v23[v26] = v15;
        }
    }
    for (int64_t i27 = v21; i27 < (v22 > v20 ? v20 : v22); i27 += v18) {
        for (int64_t j28 = v21; j28 < v19; j28 += v18) {
            // pto: %flat_offset_mul, %flat_offset, %eid_inline2694_inline12096__tile
            int32_t v29 = (v1)[(int64_t)((uint64_t)((int64_t)((uint64_t)i27 * (uint64_t)v19)) + (uint64_t)j28)];
            // pto: %5
            int64_t v30 = (int64_t)v29;
            // pto: %t__tmp_v1538
            // pto: %t__tmp_v1538
            int32_t v31 = v23[v30];
            // pto: %14, %15, %16
            v23[v30] = (int32_t)((int64_t)((uint64_t)((int64_t)v31) + (uint64_t)v18));
        }
    }
    // pto: %meta_tile_inline2721_inline12099__ssa_v0
    Tile<
        TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v32 = Tile<
            TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v18, v16);
    // pto: %meta_tile_inline2721_inline12099__ssa_v0
    uint64_t v33 = (uint64_t)v21;
    TASSIGN(v32, v33);
    TEXPANDS(v32, v15);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int64_t i34 = v21; i34 < v17; i34 += v18) {
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        for (int64_t j35 = v21; j35 < v16; j35 += v18) {
            // pto: %17, %18
            int64_t v36 = (int64_t)((uint64_t)((int64_t)((uint64_t)i34 * (uint64_t)v16)) + (uint64_t)j35);
            // pto: %t__tmp_v1539
            // pto: %t__tmp_v1539
            int32_t v37 = v23[v36];
            v32.SetValue(j35, v37);
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        // pto: %21
        int64_t v38 = static_cast<int64_t>(v9->rankId);
        // pto: %22, %23, %24, %25
        int64_t v39 = static_cast<int64_t>(v9->windowsIn[v38]);
        // pto: %26
        // pto: %27
        int64_t v41 = static_cast<int64_t>(v9->windowsIn[i34]);
        // pto: %my_rank__ssa_v0_idx
        int64_t v42 = (int64_t)v7;
        // pto: %33
        int64_t v43 = v42 < v21 ? v21 : v42;
        // pto: %recv_meta__ssa_v0_peer_pview
        pto::Shape<1, 1, 1, 1, 32> v44 = pto::Shape<1, 1, 1, 1, 32>();
        // pto: %recv_meta__ssa_v0_peer_pview
        pto::Stride<32, 32, 32, 32, 1> v45 = pto::Stride<32, 32, 32, 32, 1>();
        // pto: %28, %29, %31, %recv_meta__ssa_v0_peer_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v46 =
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(
                (v2 + (int64_t)((uint64_t)v41 - (uint64_t)v39) / v14) + (v21 + v43 * v16), v44, v45
            );
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v46, v32);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        __gm__ int32_t *v47 = PTOAS__GLOBAL_TENSOR_DATA(v46);
        PTOAS__DCCI_SINGLE_CACHE_LINE(v47);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        // pto: %35
        if (i34 != v42) {
            // pto: %36
            int64_t v48 = static_cast<int64_t>(v10->rankId);
            // pto: %37, %38, %39, %40
            int64_t v49 = static_cast<int64_t>(v10->windowsIn[v48]);
            // pto: %42
            int64_t v50 = static_cast<int64_t>(v10->windowsIn[i34]);
            // pto: %arrived__ssa_v0_peer_pview
            pto::Shape<1, 1, 1, 1, 1> v51 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %arrived__ssa_v0_peer_pview
            pto::Stride<1, 1, 1, 1, 2> v52 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %43, %44, %46, %arrived__ssa_v0_peer_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v53 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    (v3 + (int64_t)((uint64_t)v50 - (uint64_t)v49) / v14) + (v21 + v43), v51, v52
                );
            pipe_barrier(PIPE_ALL);
            pto::comm::TNOTIFY(v53, v13, v12);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    for (int64_t i54 = v21; i54 < v17; i54 += v18) {
        // pto: %50, %51
        if (i54 != (int64_t)v7) {
            // pto: %arrived__ssa_v0_local_pview
            pto::Shape<1, 1, 1, 1, 1> v55 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %arrived__ssa_v0_local_pview
            pto::Stride<1, 1, 1, 1, 2> v56 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %52, %arrived__ssa_v0_local_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v57 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    v3 + (v21 + (i54 < v21 ? v21 : i54)), v55, v56
                );
            pto::comm::TWAIT(v57, v8, v11);
            dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    for (int64_t i58 = v21; i58 < v16; i58 += v18) {
        // pto: %acc_inline2718_inline12143__rv_v2
        int32_t v59;
        v59 = v15;
        for (int64_t j60 = v21; j60 < v17; j60 += v18) {
            // pto: %acc_inline2718_inline12143__rv_v2
            int32_t v61 = v59;
            // pto: %53, %54
            int64_t v62 = (int64_t)((uint64_t)((int64_t)((uint64_t)j60 * (uint64_t)v16)) + (uint64_t)i58);
            // pto: %count_inline2688_inline12097__tile
            int32_t v63 = (v2)[v62];
            (v4)[v62] = v63;
            // pto: %57
            v59 = (int32_t)((uint32_t)v61 + (uint32_t)v63);
        }
        // pto: %acc_inline2718_inline12143__rv_v2
        int32_t v64 = v59;
        (v5)[i58] = v64;
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
    // Unpack tensor: indices_inline11972__ssa_v0
    __gm__ Tensor *indices_inline11972__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *indices_inline11972__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(indices_inline11972__ssa_v0_tensor->buffer.addr) +
        indices_inline11972__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_meta__ssa_v0
    __gm__ Tensor *recv_meta__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *recv_meta__ssa_v0 = reinterpret_cast<__gm__ int32_t *>(recv_meta__ssa_v0_tensor->buffer.addr) +
                                        recv_meta__ssa_v0_tensor->start_offset;

    // Unpack tensor: arrived__ssa_v0
    __gm__ Tensor *arrived__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *arrived__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(arrived__ssa_v0_tensor->buffer.addr) + arrived__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_meta_local_inline12091__ssa_v0
    __gm__ Tensor *recv_meta_local_inline12091__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ int32_t *recv_meta_local_inline12091__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_meta_local_inline12091__ssa_v0_tensor->buffer.addr) +
        recv_meta_local_inline12091__ssa_v0_tensor->start_offset;

    // Unpack tensor: recv_count_out_inline12024__ssa_v0
    __gm__ Tensor *recv_count_out_inline12024__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ int32_t *recv_count_out_inline12024__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(recv_count_out_inline12024__ssa_v0_tensor->buffer.addr) +
        recv_count_out_inline12024__ssa_v0_tensor->start_offset;

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

    // Unpack scalar: hca_moe_epoch_inline716__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } hca_moe_epoch_inline716__ssa_v0_conv;
    hca_moe_epoch_inline716__ssa_v0_conv.u64 = args[7];
    int32_t hca_moe_epoch_inline716__ssa_v0 = hca_moe_epoch_inline716__ssa_v0_conv.val;

    // Unpack CommContext: recv_meta_ctx
    __gm__ CommContext *recv_meta_ctx = reinterpret_cast<__gm__ CommContext *>(args[8]);

    // Unpack CommContext: arrived_ctx
    __gm__ CommContext *arrived_ctx = reinterpret_cast<__gm__ CommContext *>(args[9]);

    // Forward to ptoas-generated function
    dispatch_meta_2(
        indices_inline11972__ssa_v0, recv_meta__ssa_v0, arrived__ssa_v0, recv_meta_local_inline12091__ssa_v0,
        recv_count_out_inline12024__ssa_v0, nt_inline677__rv_v2, my_rank__ssa_v0, hca_moe_epoch_inline716__ssa_v0,
        recv_meta_ctx, arrived_ctx
    );
}
