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

#include <limits>

#include "tensormap_and_ringbuffer/kernel_clear_plan.h"

namespace {
using namespace simpler::tmr;

TEST(TmrKernelCoordinationBinding, AcceptsDisjointOrAdjacentRegions) {
    const TmrKernelClearBinding binding{
        11, {0x10000, sizeof(TmrLaunchControl)}, {0x20000, 3 * sizeof(TmrCoreReport)}, 3
    };
    EXPECT_TRUE(valid_tmr_clear_binding(binding));
    auto adjacent = binding;
    adjacent.reports.address = adjacent.control.address + adjacent.control.bytes;
    EXPECT_TRUE(valid_tmr_clear_binding(adjacent));
    adjacent.control.address = adjacent.reports.address + adjacent.reports.bytes;
    EXPECT_TRUE(valid_tmr_clear_binding(adjacent));
}

TEST(TmrKernelCoordinationBinding, RejectsInvalidRegions) {
    const TmrKernelClearBinding binding{
        11, {0x10000, sizeof(TmrLaunchControl)}, {0x20000, 3 * sizeof(TmrCoreReport)}, 3
    };
    auto invalid = binding;
    invalid.context_generation = 0;
    EXPECT_FALSE(valid_tmr_clear_binding(invalid));
    for (int32_t count : {0, -1, std::numeric_limits<int32_t>::min()}) {
        invalid = binding;
        invalid.worker_count = count;
        EXPECT_FALSE(valid_tmr_clear_binding(invalid));
    }
    invalid = binding;
    invalid.reports.address = invalid.control.address;
    EXPECT_FALSE(valid_tmr_clear_binding(invalid));
    invalid = binding;
    invalid.control.bytes = 0;
    EXPECT_FALSE(valid_tmr_clear_binding(invalid));
    invalid = binding;
    invalid.reports.bytes = sizeof(TmrCoreReport);
    EXPECT_FALSE(valid_tmr_clear_binding(invalid));
    invalid = binding;
    invalid.reports.address = std::numeric_limits<uint64_t>::max() - 63;
    EXPECT_FALSE(valid_tmr_clear_binding(invalid));
}

}  // namespace
