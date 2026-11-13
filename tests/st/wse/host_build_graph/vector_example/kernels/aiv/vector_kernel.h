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
#include <cstdint>
#include <cstring>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include <wse/pipe_sync.h>

namespace wse_test {
using Global = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<1, 1, 1, 128, 1>>;
using Tile = pto::Tile<pto::TileType::Vec, float, 128, 128, pto::BLayout::RowMajor, -1, -1>;
inline float *data(int64_t value) {
    const auto &tensor = *reinterpret_cast<const Tensor *>(value);
    return reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset;
}
inline void inputs_ready() {
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}
inline void store(Tile &tile, int64_t target) {
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Global output(data(target));
    pto::TSTORE(output, tile);
    pipe_sync();
}
template <class Compute>
void binary(int64_t *args, Compute compute) {
    Tile lhs(128, 128), rhs(128, 128), result(128, 128);
    pto::TASSIGN(lhs, 0x0);
    pto::TASSIGN(rhs, 0x10000);
    pto::TASSIGN(result, 0x20000);
    Global left(data(args[0])), right(data(args[1]));
    pto::TLOAD(lhs, left);
    pto::TLOAD(rhs, right);
    inputs_ready();
    compute(result, lhs, rhs);
    store(result, args[2]);
}
inline void add_scalar(int64_t *args) {
    Tile input(128, 128), result(128, 128);
    pto::TASSIGN(input, 0x0);
    pto::TASSIGN(result, 0x10000);
    Global source(data(args[0]));
    pto::TLOAD(input, source);
    inputs_ready();
    const auto bits = static_cast<uint32_t>(args[2]);
    float scalar;
    std::memcpy(&scalar, &bits, sizeof(scalar));
    pto::TADDS(result, input, scalar);
    store(result, args[1]);
}
}  // namespace wse_test
