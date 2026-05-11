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
/**
 * Local-expert kernel — placeholder for the production moe_expert in this
 * 2-rank demo. Replaced by the real moe_expert without changing the
 * surrounding orchestration / combine wiring.
 *
 * Behavior (one element-wise multiply per row):
 *   recv_y[e, s, :] = recv_x[e, s, :] * recv_w[e, s]      for s in [0, recv_count[e])
 *
 *   recv_x          : BF16  [L, R, D]   (dispatch OUTPUT_EXISTING, reused as INPUT)
 *   recv_w          : FP32  [L, R]      (dispatch OUTPUT_EXISTING, reused as INPUT)
 *   recv_count      : INT32 [L, 1]      (dispatch OUTPUT_EXISTING, reused as INPUT)
 *   recv_y          : BF16  [L, R, D]   (this kernel's OUTPUT_EXISTING)
 *
 * Per-expert n_rows = recv_count[e] decides how many rows to process. We
 * skip padding rows entirely — they stay whatever recv_y was previously
 * initialized to, but combine reads them only via pub_counts-driven slabs
 * that never reach into padding, so the value doesn't matter.
 *
 * Pure local — no CommRemotePtr, no signals, no scratch. BF16 round-trip
 * happens once per row at `cast(x*w, bf16)`.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Demo dimensions — must match dispatch.cpp / main.py.
static constexpr int L = 4;
static constexpr int R = 32;
static constexpr int D = 64;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *recv_x_tensor      = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recv_w_tensor      = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *recv_count_tensor  = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *recv_y_tensor      = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);
    (void)comm_ctx;  // unused; kept for ABI symmetry with dispatch / combine

    __gm__ bfloat16_t *recv_x     = reinterpret_cast<__gm__ bfloat16_t *>(recv_x_tensor->buffer.addr) +
                                    recv_x_tensor->start_offset;
    __gm__ float      *recv_w     = reinterpret_cast<__gm__ float      *>(recv_w_tensor->buffer.addr) +
                                    recv_w_tensor->start_offset;
    __gm__ int32_t    *recv_count = reinterpret_cast<__gm__ int32_t    *>(recv_count_tensor->buffer.addr) +
                                    recv_count_tensor->start_offset;
    __gm__ bfloat16_t *recv_y     = reinterpret_cast<__gm__ bfloat16_t *>(recv_y_tensor->buffer.addr) +
                                    recv_y_tensor->start_offset;

    using XShape = Shape<1, 1, 1, 1, D>;
    using XStride = Stride<D, D, D, D, 1>;
    using XGlobalBF = GlobalTensor<bfloat16_t, XShape, XStride>;
    using XTileBF = Tile<TileType::Vec, bfloat16_t, 1, D, BLayout::RowMajor>;
    using XTileF = Tile<TileType::Vec, float, 1, D, BLayout::RowMajor>;

    XTileBF x_bf;
    XTileF  x_f;
    TASSIGN(x_bf, 0x0);
    TASSIGN(x_f,  0x10000);

    for (int e = 0; e < L; ++e) {
        int n_rows = recv_count[e];
        for (int slot = 0; slot < n_rows; ++slot) {
            int row = e * R + slot;
            float w = recv_w[row];

            __gm__ bfloat16_t *x_src = recv_x + row * D;
            __gm__ bfloat16_t *y_dst = recv_y + row * D;
            XGlobalBF x_g(x_src);
            XGlobalBF y_g(y_dst);

            TLOAD(x_bf, x_g);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(x_f, x_bf, RoundMode::CAST_ROUND);
            pipe_barrier(PIPE_V);
            TMULS(x_f, x_f, w);
            pipe_barrier(PIPE_V);
            TCVT(x_bf, x_f, RoundMode::CAST_ROUND);
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(y_g, x_bf);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
