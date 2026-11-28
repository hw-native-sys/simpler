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

#include "tensor.h"

namespace {

TEST(ChipTensorContiguity, SingletonStridesDoNotChangeStorageExtent) {
    const uint32_t shapes[] = {1, 64, 1};
    const uint32_t strides[] = {999, 1, 64};
    const auto tensor = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 3);
    EXPECT_TRUE(tensor.is_contiguous());
    EXPECT_EQ(tensor.extent_elem(), 64U);
    EXPECT_EQ(tensor.buffer.size, 64U * sizeof(float));
    EXPECT_EQ(tensor.strides[0], 999U);
    EXPECT_EQ(tensor.strides[2], 64U);
}

TEST(ChipTensorContiguity, SingletonDoesNotHideGappedStorage) {
    const uint32_t shapes[] = {64, 1};
    const uint32_t strides[] = {2, 64};
    const auto tensor = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 2);
    EXPECT_FALSE(tensor.is_contiguous());
    EXPECT_EQ(tensor.extent_elem(), 127U);
    EXPECT_EQ(tensor.buffer.size, 127U * sizeof(float));
}

}  // namespace
