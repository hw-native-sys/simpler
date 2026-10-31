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
// Kernel Function: lm_head_signal_clear

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

static __aicore__ void lm_head_signal_clear(
    __gm__ float *v1, __gm__ int32_t *v2, __gm__ int32_t *v3, __gm__ CommContext *v4, __gm__ CommContext *v5
) {
    const int64_t v6 = 0;
    const int64_t v7 = 2;
    const int64_t v8 = 1;
    const int32_t v9 = 0;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (int64_t i10 = v6; i10 < v7; i10 += v8) {
        (v2)[i10] = v9;
        // pto: %lm_head_hidden_done__ssa_v0_pview
        pto::Shape<1, 1, 1, 2, 1> v11 = pto::Shape<1, 1, 1, 2, 1>();
        // pto: %lm_head_hidden_done__ssa_v0_pview
        pto::Stride<2, 2, 2, 1, 2> v12 = pto::Stride<2, 2, 2, 1, 2>();
        // pto: %lm_head_hidden_done__ssa_v0_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN> v13 =
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN>(v2, v11, v12);
        __gm__ int32_t *v14 = PTOAS__GLOBAL_TENSOR_DATA(v13);
        PTOAS__DCCI_SINGLE_CACHE_LINE(v14);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        (v3)[i10] = v9;
        // pto: %lm_head_logits_done__ssa_v0_pview
        pto::Shape<1, 1, 1, 2, 1> v15 = pto::Shape<1, 1, 1, 2, 1>();
        // pto: %lm_head_logits_done__ssa_v0_pview
        pto::Stride<2, 2, 2, 1, 2> v16 = pto::Stride<2, 2, 2, 1, 2>();
        // pto: %lm_head_logits_done__ssa_v0_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN> v17 =
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 2, 1>, pto::Stride<2, 2, 2, 1, 2>, pto::Layout::DN>(v3, v15, v16);
        __gm__ int32_t *v18 = PTOAS__GLOBAL_TENSOR_DATA(v17);
        PTOAS__DCCI_SINGLE_CACHE_LINE(v18);
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }
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
    // Unpack tensor: logits__rv_v2
    __gm__ Tensor *logits__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *logits__rv_v2 =
        reinterpret_cast<__gm__ float *>(logits__rv_v2_tensor->buffer.addr) + logits__rv_v2_tensor->start_offset;

    // Unpack tensor: lm_head_hidden_done__ssa_v0
    __gm__ Tensor *lm_head_hidden_done__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *lm_head_hidden_done__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(lm_head_hidden_done__ssa_v0_tensor->buffer.addr) +
        lm_head_hidden_done__ssa_v0_tensor->start_offset;

    // Unpack tensor: lm_head_logits_done__ssa_v0
    __gm__ Tensor *lm_head_logits_done__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ int32_t *lm_head_logits_done__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(lm_head_logits_done__ssa_v0_tensor->buffer.addr) +
        lm_head_logits_done__ssa_v0_tensor->start_offset;

    // Unpack CommContext: lm_head_hidden_done_ctx
    __gm__ CommContext *lm_head_hidden_done_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);

    // Unpack CommContext: lm_head_logits_done_ctx
    __gm__ CommContext *lm_head_logits_done_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    // Forward to ptoas-generated function
    lm_head_signal_clear(
        logits__rv_v2, lm_head_hidden_done__ssa_v0, lm_head_logits_done__ssa_v0, lm_head_hidden_done_ctx,
        lm_head_logits_done_ctx
    );
}
