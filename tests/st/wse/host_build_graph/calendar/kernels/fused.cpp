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
#include <wse/tile_ops.h>
#include "intrinsic.h"
#include "tensor.h"

namespace {
wse::TensorView tile_view(int64_t arg, size_t index) {
    const auto &tensor = *reinterpret_cast<Tensor *>(arg);
    const size_t offset = index * 4;
    if (offset >= tensor.numel()) throw std::out_of_range("matrix index outside Graph argument");
    return {reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset + offset,
            tensor.numel() - offset,
            2,
            2,
            2,
            1};
}
}  // namespace

extern "C" void kernel_entry(int64_t *args) {
    const size_t block = get_block_idx(args);
    const size_t tiles_per_block = args[3];
    wse::Tile<2, 2> input, weight, accum, result;
    for (size_t i = 0; i < tiles_per_block; ++i) {
        wse::TLOAD(input, tile_view(args[0], i));
        wse::TLOAD(weight, tile_view(args[1], block + i));
        wse::TMATMUL_ACC(accum, accum, input, weight);
    }
    wse::TRELU(result, accum);
    wse::TSTORE(tile_view(args[2], block), result);
}
