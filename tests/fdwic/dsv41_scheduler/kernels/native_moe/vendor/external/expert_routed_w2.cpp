/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define COMPACT_ACTIVATIONS
#include "expert_routed_mxfp4.hpp"

using namespace pto;
using namespace grouped_mxfp4;

#ifdef __DAV_CUBE__
namespace {
struct StoreOutput {
    __gm__ bfloat16_t* out;
    int64_t width;
    int64_t stride;
    int64_t row;
    int64_t col;

    template <typename Acc>
    AICORE inline void operator()(Acc& acc) const
    {
        using Global = GlobalTensor<bfloat16_t, Shape<1, 1, 1, DYNAMIC, N_TILE>,
                                    pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
        Global target(out + row * width + col,
                      Global::Shape(1, 1, 1, acc.GetValidRow(), N_TILE),
                      Global::Stride(stride * width, stride * width, stride * width, width, 1));
        TSTORE<STPhase::Final>(target, acc);
    }
};
}  // namespace
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    auto* counts = tensorData<int32_t>(args, 2);
    const int64_t experts = args[6];
    const int64_t stride = args[7];
    const int64_t width = args[8];
    const int64_t hidden = args[9];
    const int64_t outputRows = args[10];
    const int64_t activationRows = args[11];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    const int64_t nTiles = width / N_TILE;
#ifdef __DAV_VEC__
    auto* weight = tensorData<uint8_t>(args, 3);
#endif
#ifdef __DAV_CUBE__
    auto* x = tensorData<float8_e4m3_t>(args, 0);
    auto* xScale = tensorData<float8_e8m0_t>(args, 1);
    auto* weightScale = tensorData<float8_e8m0_t>(args, 4);
    auto* out = tensorData<bfloat16_t>(args, 5);
#endif
    beginGemm();
    int64_t compactBase = 0;
    int64_t activationBase = 0;
    int64_t workBase = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t count = acquireCount(counts, expert);
        const int64_t workCount = ((count + M_TILE - 1) / M_TILE) * nTiles;
        const int64_t first = (block + blocks - workBase % blocks) % blocks;
        for (int64_t work = first; work < workCount; work += blocks) {
            const int64_t row = work / nTiles * M_TILE;
            const int64_t col = work % nTiles * N_TILE;
            const int64_t valid = count - row < M_TILE ? count - row : M_TILE;
#ifdef __DAV_VEC__
            produceWeight(weight, hidden, width, expert, col);
#endif
#ifdef __DAV_CUBE__
            const Chunk chunk{expert, activationBase + row, valid};
            computeProjection<M_TILE>(x, xScale, weightScale, hidden, width, experts,
                                      activationRows, chunk, col,
                                      StoreOutput{out, width, outputRows, compactBase + row, col});
#endif
        }
        workBase += workCount;
        compactBase += count;
        activationBase += count;
    }
    endGemm();
}
