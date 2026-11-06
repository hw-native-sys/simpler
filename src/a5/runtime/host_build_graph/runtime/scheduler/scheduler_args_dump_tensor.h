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

#include "host_build_graph/tensor.h"

// Visit the physical cache lines touched by the actual view, including a
// non-unit innermost stride. No protocol/storage line is inferred from an
// allocation-wide or whole-cache operation.
template <typename Visit>
inline __aicore__ bool scheduler_args_dump_tensor_lines(const __gm__ simpler::hbg::Tensor *tensor, Visit visit) {
    if (tensor == nullptr || tensor->ndims == 0 || tensor->ndims > MAX_TENSOR_DIMS) return false;
    for (uint32_t dimension = 0; dimension < tensor->ndims; ++dimension)
        if (tensor->shapes[dimension] == 0) return true;
    const uint64_t element_bytes = get_element_size(tensor->dtype);
    if (element_bytes == 0 || tensor->buffer.addr == 0) return false;
    const uint64_t maximum_elements = UINT64_MAX / element_bytes;
    uint64_t elements = 1;
    for (uint32_t dimension = 0; dimension < tensor->ndims; ++dimension) {
        const uint64_t shape = tensor->shapes[dimension];
        if (elements > maximum_elements / shape || tensor->strides[dimension] == 0) return false;
        elements *= shape;
    }
    const uint64_t base = tensor->buffer.addr;
    if (base > UINT64_MAX - tensor->buffer.size) return false;
    if (tensor->start_offset > tensor->buffer.size / element_bytes) return false;
    const uint64_t origin = tensor->start_offset * element_bytes;
    if (origin > tensor->buffer.size) return false;
    if (tensor->is_contiguous) {
        if (elements > (tensor->buffer.size - origin) / element_bytes) return false;
        const uint64_t start = base + origin;
        const uint64_t end = start + elements * element_bytes;
        for (uint64_t line = start & ~UINT64_C(63); line < end;) {
            visit(line);
            if (end - line <= 64) break;
            line += 64;
        }
        return true;
    }
    uint64_t previous = UINT64_MAX;
    if (element_bytes > tensor->buffer.size) return false;
    const uint64_t maximum_source = (tensor->buffer.size - element_bytes) / element_bytes;
    for (uint64_t element = 0; element < elements; ++element) {
        uint64_t position = element;
        uint64_t source = tensor->start_offset;
        if (source > maximum_source) return false;
        for (int32_t dimension = tensor->ndims - 1; dimension >= 0; --dimension) {
            const uint64_t coordinate = position % tensor->shapes[dimension];
            position /= tensor->shapes[dimension];
            const uint64_t stride = tensor->strides[dimension];
            if (coordinate > (maximum_source - source) / stride) return false;
            source += coordinate * stride;
        }
        const uint64_t offset = source * element_bytes;
        if (offset > tensor->buffer.size || element_bytes > tensor->buffer.size - offset) return false;
        const uint64_t start = base + offset;
        const uint64_t end = start + element_bytes;
        for (uint64_t line = start & ~UINT64_C(63); line < end;) {
            if (line != previous) visit(line);
            previous = line;
            if (end - line <= 64) break;
            line += 64;
        }
    }
    return true;
}
