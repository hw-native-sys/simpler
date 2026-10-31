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
// Kernel Function: lm_head_dispatch_gather

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
lm_head_dispatch_gather(__gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ CommContext *v3, int32_t v4, int32_t v5) {
    const int64_t v6 = 512;
    const int64_t v7 = 16;
    const int64_t v8 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %gkb_inline49_inline13312__ssa_v0, %0
    int64_t v9 = (int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v6);
    // pto: %t__tile
    Tile<
        TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::Null>
        v10 = Tile<
            TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>(v7, v6);
    // pto: %t__tile
    uint64_t v11 = (uint64_t)v8;
    TASSIGN(v10, v11);
    // pto: %1
    int64_t v12 = v9 < v8 ? v8 : v9;
    // pto: %lm_head_hidden_window__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 512> v13 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %lm_head_hidden_window__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v14 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %lm_head_hidden_window__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
        v15 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v1 + (v8 + v12), v13, v14
        );
    TLOAD(v10, v15);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %owner_hiddens_inline50_inline13292__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 512> v16 = pto::Shape<1, 1, 1, 16, 512>();
    // pto: %owner_hiddens_inline50_inline13292__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v17 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %owner_hiddens_inline50_inline13292__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>
        v18 = GlobalTensor<
            bfloat16_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(
            v2 + (v8 + v12), v16, v17
        );
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v18, v10);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: lm_head_hidden_window__ssa_v0
    __gm__ Tensor *lm_head_hidden_window__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *lm_head_hidden_window__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(lm_head_hidden_window__ssa_v0_tensor->buffer.addr) +
        lm_head_hidden_window__ssa_v0_tensor->start_offset;

    // Unpack tensor: owner_hiddens_inline50_inline13292__ssa_v0
    __gm__ Tensor *owner_hiddens_inline50_inline13292__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *owner_hiddens_inline50_inline13292__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(owner_hiddens_inline50_inline13292__ssa_v0_tensor->buffer.addr) +
        owner_hiddens_inline50_inline13292__ssa_v0_tensor->start_offset;

    // Unpack CommContext: lm_head_hidden_window_ctx
    __gm__ CommContext *lm_head_hidden_window_ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);

    // Forward to ptoas-generated function
    lm_head_dispatch_gather(
        lm_head_hidden_window__ssa_v0, owner_hiddens_inline50_inline13292__ssa_v0, lm_head_hidden_window_ctx,
        __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
