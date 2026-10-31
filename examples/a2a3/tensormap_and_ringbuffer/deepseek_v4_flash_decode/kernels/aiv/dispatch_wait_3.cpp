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
// Kernel Function: dispatch_wait_3

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

static __aicore__ void
dispatch_wait_3(__gm__ int32_t *v1, __gm__ int32_t *v2, int32_t v3, int32_t v4, __gm__ CommContext *v5) {
    pto::comm::WaitCmp v6 = pto::comm::WaitCmp::GE;
    const int64_t v7 = 32;
    const int64_t v8 = 0;
    const int64_t v9 = 2;
    const int64_t v10 = 1;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (int64_t i11 = v8; i11 < v9; i11 += v10) {
        // pto: %0, %1
        if (i11 != (int64_t)v3) {
            // pto: %data_arrived__ssa_v0_local_pview
            pto::Shape<1, 1, 1, 1, 1> v12 = pto::Shape<1, 1, 1, 1, 1>();
            // pto: %data_arrived__ssa_v0_local_pview
            pto::Stride<1, 1, 1, 1, 2> v13 = pto::Stride<1, 1, 1, 1, 2>();
            // pto: %2, %data_arrived__ssa_v0_local_pview
            GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v14 =
                GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(
                    v2 + (v8 + (i11 < v8 ? v8 : i11)), v12, v13
                );
            // pto: %3, %4, %5
            int32_t v15 = (int32_t)((int64_t)((uint64_t)((int64_t)v4) * (uint64_t)v7));
            pto::comm::TWAIT(v14, v15, v6);
            dcci((__gm__ void *)0, cache_line_t::ENTIRE_DATA_CACHE);
        }
    }
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: indices_inline13017__ssa_v0
    __gm__ Tensor *indices_inline13017__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int32_t *indices_inline13017__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(indices_inline13017__ssa_v0_tensor->buffer.addr) +
        indices_inline13017__ssa_v0_tensor->start_offset;

    // Unpack tensor: data_arrived__ssa_v0
    __gm__ Tensor *data_arrived__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *data_arrived__ssa_v0 =
        reinterpret_cast<__gm__ int32_t *>(data_arrived__ssa_v0_tensor->buffer.addr) +
        data_arrived__ssa_v0_tensor->start_offset;

    // Unpack scalar: my_rank__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } my_rank__ssa_v0_conv;
    my_rank__ssa_v0_conv.u64 = args[2];
    int32_t my_rank__ssa_v0 = my_rank__ssa_v0_conv.val;

    // Unpack scalar: last_moe_epoch_inline624__ssa_v0
    union {
        uint64_t u64;
        int32_t val;
    } last_moe_epoch_inline624__ssa_v0_conv;
    last_moe_epoch_inline624__ssa_v0_conv.u64 = args[3];
    int32_t last_moe_epoch_inline624__ssa_v0 = last_moe_epoch_inline624__ssa_v0_conv.val;

    // Unpack CommContext: data_arrived_ctx
    __gm__ CommContext *data_arrived_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    // Forward to ptoas-generated function
    dispatch_wait_3(
        indices_inline13017__ssa_v0, data_arrived__ssa_v0, my_rank__ssa_v0, last_moe_epoch_inline624__ssa_v0,
        data_arrived_ctx
    );
}
