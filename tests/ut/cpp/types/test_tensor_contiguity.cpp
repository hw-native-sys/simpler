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

#include <cstdlib>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/tensor.h"
#include "task_interface/tensor.h"

[[noreturn]] void assert_impl(const char *, const char *, int) { std::abort(); }

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

TEST(HbgTensorContiguity, BoundarySingletonStridesAllowZeroCopyReshape) {
    const uint32_t shapes[] = {1, 64, 1};
    const uint32_t strides[] = {999, 1, 64};
    const auto arg = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 3);
    const auto tensor = simpler::hbg::Tensor::from_boundary(arg);
    ASSERT_TRUE(tensor.is_contiguous);
    EXPECT_EQ(tensor.extent_elem_cache, 64U);
    EXPECT_EQ(tensor.strides[0], 999U);
    EXPECT_EQ(tensor.strides[2], 64U);
    const uint32_t reshaped[] = {1, 64};
    const auto view = tensor.reshape(reshaped, 2);
    EXPECT_EQ(view.buffer.addr, arg.buffer.addr);
    EXPECT_EQ(view.start_offset, arg.start_offset);
    EXPECT_EQ(view.strides[0], 64U);
    EXPECT_EQ(view.strides[1], 1U);
    EXPECT_TRUE(view.to_boundary().is_contiguous());
}

TEST(HbgTensorContiguity, TransposedColumnAllowsZeroCopyReshape) {
    const uint32_t shapes[] = {1, 64};
    const auto arg = make_tensor_external(reinterpret_cast<void *>(0x1000), shapes, 2);
    const auto tensor = simpler::hbg::Tensor::from_boundary(arg).transpose(0, 1);
    ASSERT_EQ(tensor.shapes[0], 64U);
    ASSERT_EQ(tensor.shapes[1], 1U);
    ASSERT_EQ(tensor.strides[0], 1U);
    ASSERT_EQ(tensor.strides[1], 64U);
    ASSERT_TRUE(tensor.is_contiguous);
    const uint32_t reshaped[] = {64};
    const auto view = tensor.reshape(reshaped, 1);
    EXPECT_EQ(view.buffer.addr, arg.buffer.addr);
    EXPECT_EQ(view.numel(), 64U);
}

TEST(HbgTensorContiguity, SingletonDoesNotAllowReshapingGappedStorage) {
    const uint32_t shapes[] = {64, 1};
    const uint32_t strides[] = {2, 64};
    const auto arg = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 2);
    const auto tensor = simpler::hbg::Tensor::from_boundary(arg);
    EXPECT_FALSE(tensor.is_contiguous);
    EXPECT_EQ(tensor.extent_elem_cache, 127U);
    const uint32_t reshaped[] = {64};
    EXPECT_DEATH(tensor.reshape(reshaped, 1), "");
}

TEST(GraphTensorContiguity, AcceptsSingletonStridesAndRejectsForgedFlag) {
    const uint32_t shapes[] = {64, 1};
    const uint32_t strides[] = {1, 64};
    const auto arg = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 2);
    auto packed = graph_tensor_pack(simpler::hbg::Tensor::from_boundary(arg));
    packed.is_contiguous = 1;
    EXPECT_TRUE(graph_tensor_wire_valid(packed));
    packed.is_contiguous = 0;
    EXPECT_FALSE(graph_tensor_wire_valid(packed));
}

TEST(GraphTensorContiguity, SingletonDoesNotHideGapsOrInvalidExtent) {
    const uint32_t shapes[] = {64, 1};
    const uint32_t strides[] = {2, 64};
    const auto arg = make_tensor_strided(reinterpret_cast<void *>(0x1000), shapes, strides, 2);
    auto packed = graph_tensor_pack(simpler::hbg::Tensor::from_boundary(arg));
    ASSERT_TRUE(graph_tensor_wire_valid(packed));
    packed.is_contiguous = 1;
    EXPECT_FALSE(graph_tensor_wire_valid(packed));
    packed.is_contiguous = 0;
    packed.extent_elem = 64;
    EXPECT_FALSE(graph_tensor_wire_valid(packed));
}

}  // namespace
