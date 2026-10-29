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

#include "data_type.h"

// Saturating u64 arithmetic for tensor offsets and extents. Shapes and strides are u32;
// their products and sums can exceed a representable byte extent.
inline uint64_t tensor_mul_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_mul_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a != 0 && b > UINT64_MAX / a) ? UINT64_MAX : a * b;
#endif
}
inline uint64_t tensor_add_sat(uint64_t a, uint64_t b) {
#if defined(__clang__) || defined(__GNUC__)
    uint64_t result = 0;
    return __builtin_add_overflow(a, b, &result) ? UINT64_MAX : result;
#else
    return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
#endif
}

// Sentinel for a view whose byte extent does not fit 64 bits.
inline constexpr uint64_t TENSOR_EXTENT_UNREPRESENTABLE = UINT64_MAX;

// Byte extent of a (possibly strided) view: the last addressable element, plus one element.
// Saturates at TENSOR_EXTENT_UNREPRESENTABLE rather than wrapping, so an extent a hostile
// shape/stride cannot express is rejected instead of comparing as a small, in-bounds value.
// Callers that have not yet validated the view must not trust the result for anything but a bound.
inline uint64_t view_extent_bytes(const uint32_t *shapes, const uint32_t *strides, uint32_t ndims, DataType dtype) {
    uint64_t last_elem = 0;
    for (uint32_t i = 0; i < ndims && i < static_cast<uint32_t>(MAX_TENSOR_DIMS); ++i) {
        if (shapes[i] == 0) continue;
        last_elem = tensor_add_sat(
            last_elem, tensor_mul_sat(static_cast<uint64_t>(shapes[i] - 1), static_cast<uint64_t>(strides[i]))
        );
    }
    return tensor_mul_sat(tensor_add_sat(last_elem, 1), get_element_size(dtype));
}
