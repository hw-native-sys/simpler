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
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
#include <wse/pipe_sync.h>

extern "C" void kernel_entry(int64_t *args) {
    constexpr int N = 32;
    auto pointer = [&](int i) {
        auto *t = reinterpret_cast<Tensor *>(args[i]);
        return reinterpret_cast<float *>(t->buffer.addr) + t->start_offset;
    };
    const int block = get_block_idx(args);
    auto *out = pointer(2) + block * N * N;
    using Global = GlobalTensor<float, Shape<1, 1, 1, N, N>, pto::Stride<N * N, N * N, N * N, N, 1>>;
    using Mat = Tile<TileType::Mat, float, N, N, BLayout::ColMajor, N, N, SLayout::RowMajor, 512>;
    Mat am, bm;
    TileLeft<float, N, N, N, N> a;
    TileRight<float, N, N, N, N> b;
    TileAcc<float, N, N, N, N> acc;
    TASSIGN(am, 0);
    TASSIGN(bm, 0x20000);
    TASSIGN(a, 0);
    TASSIGN(b, 0);
    TASSIGN(acc, 0);
    Global ga(pointer(0)), gb(pointer(1)), gc(out);
    TLOAD(am, ga);
    TLOAD(bm, gb);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(a, am);
    TMOV(b, bm);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(acc, a, b);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(gc, acc);
    Tile<TileType::Vec, float, N, N, BLayout::RowMajor, N, N> v, result;
    TASSIGN(v, 0);
    TASSIGN(result, 0x10000);
    TLOAD(v, gc);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(result, v, float(block + 1));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gc, result);
    pointer(3)[2 * block] = get_core_id(args);
    pointer(3)[2 * block + 1] = get_pe_id(args);
    pipe_sync();
}
