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

extern "C" void kernel_entry(int64_t *args) {
    constexpr int N = 32;
    auto pointer = [&](int i) {
        const auto &tensor = *reinterpret_cast<const Tensor *>(args[i]);
        return reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset;
    };
    const int block = get_block_idx(args);
    using Global = pto::global_tensor<float, pto::RowMajor<N, N>>;
    Global left(pointer(0)), right(pointer(1)), out(pointer(2) + block * N * N);
    pto::CubeTileM32<float, N, N> a;
    pto::CubeTileN8<float, N, N> b;
    pto::CubeAccumulatorM32<float, N, N> product;
    TLOAD(a, left);
    TLOAD(b, right);
    TMATMUL(product, a, b);
    TSTORE(out, product);

    pto::Tile<pto::Location::Vec, float, N, N> v, result;
    TLOAD(v, out);
    TADDS(result, v, static_cast<float>(block + 1));
    TSTORE(out, result);
    pointer(3)[2 * block] = get_core_id(args);
    pointer(3)[2 * block + 1] = get_pe_id(args);
}
