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

#include <limits>
#include <wse/counter_ops.h>

#include "tensor.h"

namespace wse {

// Tensor offsets/strides count elements; buffer.size counts bytes. No memory is accessed here.
inline CounterStatus make_counter_view(const Tensor &tensor, CounterView &view) noexcept {
    view = {};
    if (tensor.dtype != DataType::UINT64 || tensor.ndims != 1 || tensor.shapes[0] == 0 || tensor.strides[0] != 1)
        return CounterStatus::INVALID_ARGUMENT;
    constexpr auto max = std::numeric_limits<uintptr_t>::max();
    const auto base = tensor.buffer.addr;
    const auto bytes = tensor.buffer.size;
    if (!base || base > max || bytes > max - base || tensor.start_offset > bytes / sizeof(uint64_t))
        return CounterStatus::INVALID_ARGUMENT;
    const auto offset = tensor.start_offset * sizeof(uint64_t);
    if (tensor.shapes[0] > (bytes - offset) / sizeof(uint64_t) || (base + offset) % alignof(uint64_t) != 0)
        return CounterStatus::INVALID_ARGUMENT;
    view = {reinterpret_cast<uint64_t *>(static_cast<uintptr_t>(base + offset)), tensor.shapes[0]};
    return CounterStatus::OK;
}

}  // namespace wse
