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
 * FFN residual add: ``ffn_out = routed_y + sh``.
 *
 *   routed_y  : FP32  [T, D]   (combine output)
 *   sh        : BF16  [T, D]   (moe_expert shared-expert output)
 *   ffn_out   : BF16  [T, D]
 */

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <cstdint>

#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

static constexpr int T = 16;
static constexpr int D = 4096;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *routed_y_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *sh_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *ffn_out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    __gm__ float *routed_y =
        reinterpret_cast<__gm__ float *>(routed_y_tensor->buffer.addr) + routed_y_tensor->start_offset;
    __gm__ bfloat16_t *sh =
        reinterpret_cast<__gm__ bfloat16_t *>(sh_tensor->buffer.addr) + sh_tensor->start_offset;
    __gm__ bfloat16_t *ffn_out =
        reinterpret_cast<__gm__ bfloat16_t *>(ffn_out_tensor->buffer.addr) + ffn_out_tensor->start_offset;

    using RowFpG = GlobalTensor<float, Shape<1, 1, 1, 1, D>, Stride<D, D, D, D, 1>>;
    using RowBfG = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, D>, Stride<D, D, D, D, 1>>;
    using RowFpTile = Tile<TileType::Vec, float, 1, D, BLayout::RowMajor, 1, D>;
    using RowBfTile = Tile<TileType::Vec, bfloat16_t, 1, D, BLayout::RowMajor, 1, D>;

    // 4 tiles laid out within the 192 KiB AIV UB (0x30000 is OOB so the 4th
    // slot uses a tighter 0x28000 offset instead of the usual 64 KB pitch).
    RowFpTile routed_fp;
    RowFpTile sh_fp;
    RowBfTile sh_bf;
    RowBfTile out_bf;
    TASSIGN(routed_fp, 0x0);
    TASSIGN(sh_fp, 0x10000);
    TASSIGN(sh_bf, 0x20000);
    TASSIGN(out_bf, 0x28000);

    for (int t = 0; t < T; ++t) {
        RowFpG r_g(routed_y + t * D);
        RowBfG s_g(sh + t * D);
        RowBfG o_g(ffn_out + t * D);

        TLOAD(routed_fp, r_g);
        TLOAD(sh_bf, s_g);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(sh_fp, sh_bf, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TADD(routed_fp, routed_fp, sh_fp);
        pipe_barrier(PIPE_V);
        TCVT(out_bf, routed_fp, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(o_g, out_bf);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
