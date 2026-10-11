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

/**
 * Packed-output layout totals: the sum of a task's 1024B-aligned outputs is the
 * size of its packed region, so it must keep its full value past 2 GiB. A
 * wrapped total (negative, or wrapped back to a small positive value)
 * under-allocates the region and silently aliases later tasks' outputs.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "host_build_graph/orchestrator_internal.h"

namespace {

TEST(HbgOutputLayout, TwoOneGiBOutputsKeepFullTotal) {
    uint32_t shape[] = {1u << 28};  // FLOAT32 → 1 GiB per output, 1024-aligned
    TensorCreateInfo first(shape, 1, DataType::FLOAT32);
    TensorCreateInfo second(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(first, second);

    OutputLayout layout = calculate_output_layout(args);

    EXPECT_EQ(layout.buffer_sizes[0], 1ull << 30);
    EXPECT_EQ(layout.buffer_sizes[1], 1ull << 30);
    EXPECT_EQ(layout.offsets[0], 0ull);
    EXPECT_EQ(layout.offsets[1], 1ull << 30);
    EXPECT_EQ(static_cast<uint64_t>(layout.total_output_size), 2ull << 30);
}

TEST(HbgOutputLayout, TotalsPastFourGiBDoNotWrapToASmallPositiveValue) {
    uint32_t shape_4g[] = {1u << 30};  // FLOAT32 → 4 GiB
    uint32_t shape_1g[] = {1u << 28};  // FLOAT32 → 1 GiB
    TensorCreateInfo big(shape_4g, 1, DataType::FLOAT32);
    TensorCreateInfo small(shape_1g, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(big, small);

    OutputLayout layout = calculate_output_layout(args);

    EXPECT_EQ(layout.offsets[1], 4ull << 30);
    EXPECT_EQ(static_cast<uint64_t>(layout.total_output_size), 5ull << 30);
}

}  // namespace
