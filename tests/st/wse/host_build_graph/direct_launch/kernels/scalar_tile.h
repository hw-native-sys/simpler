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
#include <pto/pto-inst.hpp>
#include "tensor.h"

namespace wse_test {
// One valid lane in a minimum-size Linx vector Tile.
template <class T>
using ScalarTile = pto::Tile<pto::Location::Vec, T, 1, 32, pto::BLayout::RowMajor, 1, 1>;

template <class T>
auto scalar_memory(int64_t arg, size_t offset = 0) {
    const auto &tensor = *reinterpret_cast<const Tensor *>(arg);
    return pto::global_tensor<T, pto::RowMajor<1, 1>>(
        reinterpret_cast<T *>(tensor.buffer.addr) + tensor.start_offset + offset
    );
}

template <class T>
void write_scalar(int64_t arg, T value, size_t offset = 0) {
    auto output = scalar_memory<T>(arg, offset);
    ScalarTile<T> tile;
    TEXPANDS(tile, value);
    TSTORE(output, tile);
}
}  // namespace wse_test
