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

#include <cstdint>
#include <vector>

#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_args_dump_tensor.h"

namespace {
simpler::hbg::Tensor tensor_view(uint64_t address = 0x1000, uint64_t bytes = 128) {
    simpler::hbg::Tensor tensor{};
    tensor.buffer.addr = address;
    tensor.buffer.size = bytes;
    tensor.dtype = DataType::UINT32;
    tensor.ndims = 1;
    tensor.shapes[0] = 1;
    tensor.strides[0] = 1;
    tensor.is_contiguous = true;
    return tensor;
}

TEST(ArgsDumpTensorLines, ContiguousAndStridedViewsVisitOnlyTheirPhysicalLines) {
    std::vector<uint64_t> lines;
    const auto visit = [&](uint64_t address) {
        lines.push_back(address);
    };
    auto tensor = tensor_view(0x103c, 8);
    tensor.shapes[0] = 2;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, visit));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000, 0x1040}));

    lines.clear();
    tensor = tensor_view();
    tensor.ndims = 2;
    tensor.shapes[0] = 2;
    tensor.shapes[1] = 3;
    tensor.strides[0] = 16;
    tensor.strides[1] = 2;
    tensor.start_offset = 1;
    tensor.is_contiguous = false;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, visit));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000, 0x1040}));

    lines.clear();
    tensor = tensor_view(0x1000, 256);
    tensor.shapes[0] = 3;
    tensor.strides[0] = 17;
    tensor.is_contiguous = false;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, visit));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000, 0x1040, 0x1080}));

    lines.clear();
    tensor = tensor_view(0x103f, 80);
    tensor.dtype = DataType::UINT64;
    tensor.shapes[0] = 2;
    tensor.strides[0] = 9;
    tensor.is_contiguous = false;
    ASSERT_TRUE(scheduler_args_dump_tensor_lines(&tensor, visit));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000, 0x1040, 0x1080}));
}

TEST(ArgsDumpTensorLines, EmptyViewsPerformNoCacheOperation) {
    auto tensor = tensor_view(0, 0);
    tensor.shapes[0] = 0;
    tensor.strides[0] = 0;
    bool visited = false;
    EXPECT_TRUE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t) {
        visited = true;
    }));
    EXPECT_FALSE(visited);
}

TEST(ArgsDumpTensorLines, InvalidGeometryCannotVisitAnOutOfRangeLine) {
    for (int mutation = 0; mutation < 8; ++mutation) {
        SCOPED_TRACE(mutation);
        auto tensor = tensor_view();
        switch (mutation) {
        case 0:
            tensor.ndims = 0;
            break;
        case 1:
            tensor.ndims = MAX_TENSOR_DIMS + 1;
            break;
        case 2:
            tensor.dtype = static_cast<DataType>(UINT8_MAX);
            break;
        case 3:
            tensor.buffer.addr = 0;
            break;
        case 4:
            tensor.buffer.addr = UINT64_MAX - 31;
            break;
        case 5:
            tensor.start_offset = 33;
            break;
        case 6:
            tensor.strides[0] = 0;
            break;
        case 7:
            tensor.ndims = MAX_TENSOR_DIMS;
            for (uint32_t d = 0; d < MAX_TENSOR_DIMS; ++d) {
                tensor.shapes[d] = UINT32_MAX;
                tensor.strides[d] = 1;
            }
            break;
        }
        bool visited = false;
        EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t) {
            visited = true;
        }));
        EXPECT_FALSE(visited);
    }
    auto tensor = tensor_view(0x1000, 16);
    tensor.is_contiguous = false;
    tensor.shapes[0] = 2;
    tensor.strides[0] = UINT32_MAX;
    std::vector<uint64_t> lines;
    EXPECT_FALSE(scheduler_args_dump_tensor_lines(&tensor, [&](uint64_t address) {
        lines.push_back(address);
    }));
    EXPECT_EQ(lines, (std::vector<uint64_t>{0x1000}));
}
}  // namespace
