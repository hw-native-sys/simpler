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
#include <cstddef>
#include <cstdint>
#include "runtime.h"

TEST(RuntimeWorkersBoundary, TmrDescriptorUploadIncludesOnlyUsedTensorSlots) {
    Runtime runtime;
    constexpr size_t tensors_at =
        offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + offsetof(simpler::tmr::EntryArgsStorage, tensors_);
    for (int32_t count : {0, 1, 4, CHIP_MAX_TENSOR_ARGS}) {
        runtime.dev.orch_args_storage_.tensor_count_ = count;
        EXPECT_EQ(
            runtime_device_copy_size(runtime), tensors_at + static_cast<size_t>(count) * sizeof(simpler::tmr::Tensor)
        );
        EXPECT_EQ(runtime_device_extent_size(runtime), sizeof(DeviceRuntimeLaunchDesc));
        EXPECT_EQ(runtime_device_initialized_prefix_size(runtime), sizeof(DeviceRuntimeLaunchDesc));
    }
}

TEST(RuntimeWorkersBoundary, KernelLaunchDescriptorExcludesEntryCapacity) {
    Runtime runtime;
    for (int32_t count : {0, 1, 4, CHIP_MAX_TENSOR_ARGS}) {
        runtime.dev.orch_args_storage_.tensor_count_ = count;
        EXPECT_EQ(runtime_device_launch_size(runtime), offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_));
    }
    EXPECT_LT(runtime_device_launch_size(runtime), runtime_device_copy_size(runtime));
}
