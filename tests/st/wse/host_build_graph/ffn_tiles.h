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
#pragma once
#include <wse/tile_ops.h>
#include "intrinsic.h"

using FfnMemory = pto::global_tensor<float, pto::RowMajor<32, 32>>;

inline void ffn_relu(FfnMemory input, FfnMemory output) {
    pto::VecTileM32<float, 32, 32> value;
    TLOAD(value, input);
    TRELU(value, value);
    TSTORE(output, value);
}

template <unsigned Slot, class Input, class Weight>
void ffn_project(int64_t *args, uint32_t publisher, size_t chunks, Input input, Weight weight, FfnMemory output) {
    using WeightTile = pto::SharedMatrixRight<float, 32, 32>;
    wse::SharedTileSlot<Slot, WeightTile> weights(
        publisher, 15, get_wse_block_scope(args), get_wse_operation_trace(args)
    );
    pto::CubeAccumulatorM32<float, 32, 32> result;
    for (size_t k = 0; k < chunks; ++k) {
        if (get_pe_id(args) == publisher) weights.publish(k + 1, weight(k));
        auto input_shard = input(k);
        pto::CubeTileM32<float, 32, 32> left;
        TLOAD(left, input_shard);
        // Shared Right is stored as [N, K]. Four Local Left shards cover the 128 group rows.
        auto right = weights.read(k + 1);
        if (k == 0) TMATMUL(result, left, right, 128);
        else TMATMUL_ACC(result, result, left, right, 128);
    }
    TSTORE(output, result);
}
