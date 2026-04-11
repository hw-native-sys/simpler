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

#include <gtest/gtest.h>

#include "pto_orchestration_api.h"

TEST(A2A3TmrTensorOffsets, ExternalTensorStartsAtZeroOffset) {
    uint32_t shapes[2] = {8, 16};
    Tensor tensor = make_tensor_external(reinterpret_cast<void *>(0x1000), shapes, 2, DataType::FLOAT32);

    EXPECT_EQ(tensor.start_offset, 0U);
}

TEST(A2A3TmrTensorOffsets, ViewCachesFlatStartOffsetAtConstruction) {
    uint32_t shapes[2] = {8, 16};
    Tensor tensor = make_tensor_external(reinterpret_cast<void *>(0x1000), shapes, 2, DataType::FLOAT32);

    uint32_t view_shapes[2] = {2, 4};
    uint32_t view_offsets[2] = {3, 5};
    Tensor view = tensor.view(view_shapes, view_offsets);

    EXPECT_EQ(view.start_offset, 3U * 16U + 5U);
}

TEST(A2A3TmrTensorOffsets, NestedViewKeepsAccumulatedStartOffset) {
    uint32_t shapes[2] = {8, 16};
    Tensor tensor = make_tensor_external(reinterpret_cast<void *>(0x1000), shapes, 2, DataType::FLOAT32);

    uint32_t outer_shapes[2] = {4, 8};
    uint32_t outer_offsets[2] = {2, 3};
    Tensor outer = tensor.view(outer_shapes, outer_offsets);

    uint32_t inner_shapes[2] = {2, 4};
    uint32_t inner_offsets[2] = {1, 2};
    Tensor inner = outer.view(inner_shapes, inner_offsets);

    EXPECT_EQ(inner.start_offset, (2U + 1U) * 16U + (3U + 2U));
}
