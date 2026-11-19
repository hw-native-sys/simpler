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

namespace {
constexpr size_t kWorkersBytes = sizeof(DeviceRuntimeLaunchDesc::workers);

// Where this runtime's entry tensors start inside the descriptor, and the slot
// width the steady length counts in.
constexpr size_t kTensorsAt =
    offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + offsetof(simpler::tmr::EntryArgsStorage, tensors_);
constexpr size_t kTensorBytes = sizeof(simpler::tmr::Tensor);

}  // namespace

// This runtime's steady prefix is bounded by the run's own tensor count, so it
// is the one length of the three that is not a constant of the variant. The
// count alone selects it — `used_prefix_bytes` reads no tensor — so the count
// is what this sets.
TEST(RuntimeWorkersBoundary, TheSteadyStatePrefixIsBoundedByThisRunsTensorCount) {
    Runtime runtime;

    runtime.dev.orch_args_storage_.tensor_count_ = 0;
    EXPECT_EQ(runtime_device_copy_size(runtime), kTensorsAt) << "with no tensors the prefix ends where they begin";

    for (const int32_t count : {1, 4, 47, CHIP_MAX_TENSOR_ARGS - 1}) {
        runtime.dev.orch_args_storage_.tensor_count_ = count;
        EXPECT_EQ(runtime_device_copy_size(runtime), kTensorsAt + static_cast<size_t>(count) * kTensorBytes)
            << "count " << count;
        EXPECT_LT(runtime_device_copy_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers)) << "count " << count;
    }

    runtime.dev.orch_args_storage_.tensor_count_ = CHIP_MAX_TENSOR_ARGS;
    EXPECT_EQ(runtime_device_copy_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers))
        << "a run that fills every slot reaches exactly the handshake region's first byte, and no further";
}

// What the two prefixes differ by, which on this runtime is the handshake
// region plus the tensor slots this run left unfilled — nothing else, at any
// count.
TEST(RuntimeWorkersBoundary, ASteadyStateRunRepublishesNeitherTheHandshakeRegionNorUnusedCapacity) {
    Runtime runtime;
    for (const int32_t count : {0, 1, 4, 47, CHIP_MAX_TENSOR_ARGS - 1, CHIP_MAX_TENSOR_ARGS}) {
        runtime.dev.orch_args_storage_.tensor_count_ = count;
        const size_t unused = static_cast<size_t>(CHIP_MAX_TENSOR_ARGS - count) * kTensorBytes;
        EXPECT_EQ(
            runtime_device_initialized_prefix_size(runtime) - runtime_device_copy_size(runtime), kWorkersBytes + unused
        ) << "count "
          << count;
    }

    // At capacity the difference is the handshake region alone, which is the
    // whole of what the first publication adds.
    runtime.dev.orch_args_storage_.tensor_count_ = CHIP_MAX_TENSOR_ARGS;
    EXPECT_EQ(runtime_device_initialized_prefix_size(runtime) - runtime_device_copy_size(runtime), kWorkersBytes);
}

// The first publication's range and the allocation are properties of the
// variant, not of a run: a count that shortens the steady prefix must move
// neither.
TEST(RuntimeWorkersBoundary, TheCountBoundedPrefixMovesNeitherTheInitializedRangeNorTheExtent) {
    Runtime runtime;
    const size_t initialized = runtime_device_initialized_prefix_size(runtime);
    const size_t extent = runtime_device_extent_size(runtime);

    for (const int32_t count : {0, 4, CHIP_MAX_TENSOR_ARGS}) {
        runtime.dev.orch_args_storage_.tensor_count_ = count;
        EXPECT_EQ(runtime_device_initialized_prefix_size(runtime), initialized) << "count " << count;
        EXPECT_EQ(runtime_device_extent_size(runtime), extent) << "count " << count;
    }
    EXPECT_EQ(initialized, offsetof(DeviceRuntimeLaunchDesc, workers) + kWorkersBytes);
    EXPECT_EQ(extent, sizeof(DeviceRuntimeLaunchDesc));
}
