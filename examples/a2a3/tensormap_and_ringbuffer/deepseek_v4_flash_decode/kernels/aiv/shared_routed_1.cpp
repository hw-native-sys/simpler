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
// Kernel Function: shared_routed_1

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

static __aicore__ void shared_routed_1(
    __gm__ bfloat16_t *v1, __gm__ bfloat16_t *v2, __gm__ bfloat16_t *v3, int64_t v4, __gm__ CommContext *v5, int32_t v6,
    int32_t v7
) {
    RoundMode v8 = RoundMode::CAST_RINT;
    SaturationMode v9 = SaturationMode::OFF;
    RoundMode v10 = RoundMode::CAST_ROUND;
    const int64_t v11 = 6;
    const int64_t v12 = 1;
    const int64_t v13 = 4096;
    const int64_t v14 = 16384;
    const int64_t v15 = 0;
    const int64_t v16 = 24576;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    // pto: %t_inline2815_inline10939__ssa_v0
    int64_t v17 = (int64_t)v6;
    // pto: %5
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (v17 < v4) {
        // pto: %t__tile
        Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v18 = Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %t__tile
        uint64_t v19 = (uint64_t)v16;
        TASSIGN(v18, v19);
        // pto: %6
        int64_t v20 = v17 < v15 ? v15 : v17;
        // pto: %sh_inline11139__rv_v2_pview
        pto::Shape<1, 1, 1, 1, 4096> v21 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %sh_inline11139__rv_v2_pview
        pto::Stride<4096, 4096, 4096, 4096, 1> v22 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %sh_inline11139__rv_v2_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v23 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v1 + (v15 + v20 * v13), v21, v22
            );
        TLOAD(v18, v23);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %acc_inline2814_inline11132__tile
        Tile<
            TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v24 = Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %acc_inline2814_inline11132__tile
        uint64_t v25 = (uint64_t)v15;
        TASSIGN(v24, v25);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(v24, v18, v10, v9);
        for (int64_t i26 = v15; i26 < v11; i26 += v12) {
            // pto: %7, %8
            int64_t v27 = (int64_t)((uint64_t)((int64_t)((uint64_t)v17 * (uint64_t)v11)) + (uint64_t)i26);
            // pto: %0
            Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v28 = Tile<
                    TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                    PadValue::Null, CompactMode::Null>(v12, v13);
            // pto: %0
            uint64_t v29 = (uint64_t)v14;
            TASSIGN(v28, v29);
            // pto: %routed_y_buf__ssa_v0_pview
            pto::Shape<1, 1, 1, 1, 4096> v30 = pto::Shape<1, 1, 1, 1, 4096>();
            // pto: %routed_y_buf__ssa_v0_pview
            pto::Stride<4096, 4096, 4096, 4096, 1> v31 = pto::Stride<4096, 4096, 4096, 4096, 1>();
            // pto: %9, %routed_y_buf__ssa_v0_pview
            GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
                v32 = GlobalTensor<
                    bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                    v2 + (v15 + (v27 < v15 ? v15 : v27) * v13), v30, v31
                );
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            TLOAD(v28, v32);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            // pto: %1
            Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v33 = Tile<
                    TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v12, v13);
            // pto: %1
            uint64_t v34 = (uint64_t)v16;
            TASSIGN(v33, v34);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            pipe_barrier(PIPE_V);
            TCVT(v33, v28, v10, v9);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            // pto: %2
            Tile<
                TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>
                v35 = Tile<
                    TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                    CompactMode::Null>(v12, v13);
            // pto: %2
            uint64_t v36 = (uint64_t)v15;
            TASSIGN(v35, v36);
            pipe_barrier(PIPE_V);
            TADD(v35, v24, v33);
        }
        // pto: %3
        Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v37 = Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %3
        uint64_t v38 = (uint64_t)v15;
        TASSIGN(v37, v38);
        pipe_barrier(PIPE_V);
        TCVT(v37, v24, v8, v9);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %ffn_out_inline10948__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 4096> v39 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %ffn_out_inline10948__ssa_v0_pview
        pto::Stride<4096, 4096, 4096, 4096, 1> v40 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %ffn_out_inline10948__ssa_v0_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v41 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v3 + (v15 + v20 * v13), v39, v40
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v41, v37);
    } else {
        // pto: %4
        Tile<
            TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            v42 = Tile<
                TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(v12, v13);
        // pto: %4
        uint64_t v43 = (uint64_t)v15;
        TASSIGN(v42, v43);
        // pto: %11
        int64_t v44 = v17 < v15 ? v15 : v17;
        // pto: %12
        pto::Shape<1, 1, 1, 1, 4096> v45 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %12
        pto::Stride<4096, 4096, 4096, 4096, 1> v46 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %12
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v47 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v1 + (v15 + v44 * v13), v45, v46
            );
        TLOAD(v42, v47);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // pto: %15
        pto::Shape<1, 1, 1, 1, 4096> v48 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %15
        pto::Stride<4096, 4096, 4096, 4096, 1> v49 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %15
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>
            v50 = GlobalTensor<
                bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(
                v3 + (v15 + v44 * v13), v48, v49
            );
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v50, v42);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // Unpack tensor: sh_inline11139__rv_v2
    __gm__ Tensor *sh_inline11139__rv_v2_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ bfloat16_t *sh_inline11139__rv_v2 =
        reinterpret_cast<__gm__ bfloat16_t *>(sh_inline11139__rv_v2_tensor->buffer.addr) +
        sh_inline11139__rv_v2_tensor->start_offset;

    // Unpack tensor: routed_y_buf__ssa_v0
    __gm__ Tensor *routed_y_buf__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ bfloat16_t *routed_y_buf__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(routed_y_buf__ssa_v0_tensor->buffer.addr) +
        routed_y_buf__ssa_v0_tensor->start_offset;

    // Unpack tensor: ffn_out_inline10948__ssa_v0
    __gm__ Tensor *ffn_out_inline10948__ssa_v0_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ bfloat16_t *ffn_out_inline10948__ssa_v0 =
        reinterpret_cast<__gm__ bfloat16_t *>(ffn_out_inline10948__ssa_v0_tensor->buffer.addr) +
        ffn_out_inline10948__ssa_v0_tensor->start_offset;

    // Unpack scalar: active_tokens_inline2816_inline11163__phi_v4
    union {
        uint64_t u64;
        int64_t val;
    } active_tokens_inline2816_inline11163__phi_v4_conv;
    active_tokens_inline2816_inline11163__phi_v4_conv.u64 = args[3];
    int64_t active_tokens_inline2816_inline11163__phi_v4 = active_tokens_inline2816_inline11163__phi_v4_conv.val;

    // Unpack CommContext: routed_y_buf_ctx
    __gm__ CommContext *routed_y_buf_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    // Forward to ptoas-generated function
    shared_routed_1(
        sh_inline11139__rv_v2, routed_y_buf__ssa_v0, ffn_out_inline10948__ssa_v0,
        active_tokens_inline2816_inline11163__phi_v4, routed_y_buf_ctx, __pypto_spmd_block_idx, __pypto_spmd_block_num
    );
}
