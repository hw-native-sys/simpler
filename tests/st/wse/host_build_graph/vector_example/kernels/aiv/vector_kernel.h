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
#include <bit>
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

namespace wse_test {
using Global = pto::global_tensor<float, pto::RowMajor<128, 128>>;
using Vec = pto::Tile<pto::Location::Vec, float, 128, 128>;

inline Global global(int64_t value) {
    const auto &tensor = *reinterpret_cast<const Tensor *>(value);
    return Global(reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset);
}

template <class Compute>
void binary(int64_t *args, Compute compute) {
    auto left = global(args[0]), right = global(args[1]), output = global(args[2]);
    Vec lhs, rhs, result;
    TLOAD(lhs, left);
    TLOAD(rhs, right);
    compute(result, lhs, rhs);
    TSTORE(output, result);
}

inline void add_scalar(int64_t *args) {
    auto source = global(args[0]), output = global(args[1]);
    Vec input, result;
    TLOAD(input, source);
    TADDS(result, input, std::bit_cast<float>(static_cast<uint32_t>(args[2])));
    TSTORE(output, result);
}
}  // namespace wse_test
