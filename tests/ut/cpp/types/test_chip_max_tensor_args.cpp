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
// Regression test for #786: CHIP_MAX_TENSOR_ARGS was 64, which blocked the
// DeepSeek-V4 decode-CSA orchestration at 69 entry tensors. After the bump
// to 128 we verify (a) a ChipStorageTaskArgs accepts the 65th..128th tensor
// and (b) view_to_chip_storage still rejects 129+.

#include <stdexcept>

#include <gtest/gtest.h>

#include "task_args.h"

namespace {

ContinuousTensor make_tensor(uint64_t addr) {
    ContinuousTensor t{};
    t.data = addr;
    t.shapes[0] = 1;
    t.ndims = 1;
    t.dtype = DataType::FLOAT32;
    return t;
}

}  // namespace

TEST(ChipMaxTensorArgs, CapIsAtLeast128) {
    // The cap is the contract — the storage struct and the chip callable
    // signature_[] array both size off it. Lock in 128 explicitly so a
    // future change that lowers it is caught.
    static_assert(CHIP_MAX_TENSOR_ARGS >= 128, "CHIP_MAX_TENSOR_ARGS regressed below 128 (see #786)");
    EXPECT_GE(CHIP_MAX_TENSOR_ARGS, 128);
}

TEST(ChipMaxTensorArgs, ChipStorageHoldsAtLeast128Tensors) {
    // Pre-#786 this throws std::out_of_range at the 65th add_tensor.
    ChipStorageTaskArgs args;
    for (int i = 0; i < 128; ++i) {
        ASSERT_NO_THROW(args.add_tensor(make_tensor(static_cast<uint64_t>(0x1000 + i))));
    }
    EXPECT_EQ(args.tensor_count(), 128);
    EXPECT_EQ(args.tensor(0).data, 0x1000u);
    EXPECT_EQ(args.tensor(127).data, 0x1000u + 127);
}

TEST(ChipMaxTensorArgs, ChipStorageRejectsOverflow) {
    ChipStorageTaskArgs args;
    for (int i = 0; i < CHIP_MAX_TENSOR_ARGS; ++i) {
        args.add_tensor(make_tensor(static_cast<uint64_t>(i)));
    }
    EXPECT_THROW(args.add_tensor(make_tensor(0xDEAD)), std::out_of_range);
}

TEST(ChipMaxTensorArgs, ViewToChipStorageAcceptsCap) {
    std::vector<ContinuousTensor> tensors;
    tensors.reserve(CHIP_MAX_TENSOR_ARGS);
    for (int i = 0; i < CHIP_MAX_TENSOR_ARGS; ++i) {
        tensors.push_back(make_tensor(static_cast<uint64_t>(0x2000 + i)));
    }
    TaskArgsView view{CHIP_MAX_TENSOR_ARGS, 0, tensors.data(), nullptr};

    ChipStorageTaskArgs chip;
    ASSERT_NO_THROW(chip = view_to_chip_storage(view));
    EXPECT_EQ(chip.tensor_count(), CHIP_MAX_TENSOR_ARGS);
    EXPECT_EQ(chip.tensor(CHIP_MAX_TENSOR_ARGS - 1).data, 0x2000u + CHIP_MAX_TENSOR_ARGS - 1);
}

TEST(ChipMaxTensorArgs, ViewToChipStorageRejectsOverflow) {
    std::vector<ContinuousTensor> tensors(CHIP_MAX_TENSOR_ARGS + 1, make_tensor(0));
    TaskArgsView view{CHIP_MAX_TENSOR_ARGS + 1, 0, tensors.data(), nullptr};
    EXPECT_THROW(view_to_chip_storage(view), std::out_of_range);
}
