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
// The three device lengths of a Runtime descriptor, checked against the real
// layout of whichever variant this executable was compiled for. One source file
// registered against all four arch/runtime combinations: the contract is that
// they answer the same shape, not that they answer the same numbers.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "runtime.h"

namespace {

// Bytes the handshake region occupies in this variant.
constexpr size_t kWorkersBytes = sizeof(DeviceRuntimeLaunchDesc::workers);

}  // namespace

TEST(RuntimeWorkersBoundary, TheThreeLengthsAreOrderedAndNameDifferentBoundaries) {
    const Runtime runtime;
    const size_t copy = runtime_device_copy_size(runtime);
    const size_t initialized = runtime_device_initialized_prefix_size(runtime);
    const size_t extent = runtime_device_extent_size(runtime);

    EXPECT_LT(copy, initialized) << "a steady-state run must publish strictly less than a first one";
    EXPECT_LE(initialized, extent) << "no publication may run past the allocation";
    EXPECT_EQ(extent, sizeof(DeviceRuntimeLaunchDesc));
}

TEST(RuntimeWorkersBoundary, TheSteadyStatePrefixNeverReachesTheHandshakeRegion) {
    const Runtime runtime;
    // The property every variant owes, whatever bounds its length: a
    // steady-state publication stops at or before the first byte the device
    // owns. The exact length is per-runtime, below.
    EXPECT_LE(runtime_device_copy_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers));
}

TEST(RuntimeWorkersBoundary, TheInitializingPrefixEndsWithTheHandshakeRegion) {
    const Runtime runtime;
    EXPECT_EQ(
        runtime_device_initialized_prefix_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers) + kWorkersBytes
    );
}

#ifdef SIMPLER_UT_TRB_RUNTIME

namespace {

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

#else

// This runtime publishes a fixed steady prefix: it ends exactly where the
// handshake region begins, for every run.
TEST(RuntimeWorkersBoundary, TheSteadyStatePrefixStopsWhereTheHandshakeRegionBegins) {
    const Runtime runtime;
    EXPECT_EQ(runtime_device_copy_size(runtime), offsetof(DeviceRuntimeLaunchDesc, workers));
}

// What the move is for, stated as the difference between the two prefixes.
TEST(RuntimeWorkersBoundary, ASteadyStateRunNoLongerRepublishesTheHandshakeRegion) {
    const Runtime runtime;
    EXPECT_EQ(runtime_device_initialized_prefix_size(runtime) - runtime_device_copy_size(runtime), kWorkersBytes)
        << "the two prefixes must differ by exactly the handshake region and nothing else";
}

#endif

TEST(RuntimeWorkersBoundary, EveryHandshakeOccupiesItsOwnCacheLine) {
    EXPECT_EQ(sizeof(Handshake), 64U);
    EXPECT_EQ(alignof(Handshake), 64U);
    EXPECT_EQ(offsetof(DeviceRuntimeLaunchDesc, workers) % 64U, 0U);

    // Not just the first one: the array has to keep the property element by
    // element, because each AICore writes its own line back whole.
    const Runtime runtime;
    const auto base = reinterpret_cast<uintptr_t>(&runtime.dev.workers[0]);
    for (int i = 1; i < RUNTIME_MAX_WORKER; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(&runtime.dev.workers[i]);
        ASSERT_EQ((address - base) % 64U, 0U) << "worker " << i << " does not start a line";
    }
}

// The first publication carries this region straight from the host object, so
// what the constructor leaves there is what a fresh device block starts from.
TEST(RuntimeWorkersBoundary, AFreshRuntimeHasAZeroedHandshakeRegion) {
    const Runtime runtime;
    const auto *bytes = reinterpret_cast<const unsigned char *>(&runtime.dev.workers[0]);
    for (size_t i = 0; i < kWorkersBytes; ++i) {
        ASSERT_EQ(bytes[i], 0u) << "byte " << i << " of the handshake region is not zeroed";
    }
}

TEST(RuntimeWorkersBoundary, TheLaunchShapeRuleIsHostStateAndNotTheHandshakeRegion) {
    Runtime runtime;

    // A sentinel, not zero: CoreType::AIC is 0, so a region left zeroed cannot
    // distinguish "untouched" from "written with AIC".
    constexpr unsigned char kSentinel = 0x3c;
    auto *region = reinterpret_cast<unsigned char *>(&runtime.dev.workers[0]);
    std::memset(region, kSentinel, kWorkersBytes);

    runtime.set_core_type_rule(/*worker_count=*/9, /*aic_count=*/3);

    EXPECT_EQ(runtime.core_type_rule_count(), 9U);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(runtime.core_type_rule(i), CoreType::AIC) << "worker " << i;
    for (int i = 3; i < 9; ++i)
        EXPECT_EQ(runtime.core_type_rule(i), CoreType::AIV) << "worker " << i;

    // Recording the rule must not write the region the device owns.
    for (size_t i = 0; i < kWorkersBytes; ++i) {
        ASSERT_EQ(region[i], kSentinel) << "byte " << i << " of the handshake region was written";
    }

    // A later shape replaces the rule rather than leaving a longer predecessor
    // visible past its end.
    runtime.set_core_type_rule(/*worker_count=*/3, /*aic_count=*/1);
    EXPECT_EQ(runtime.core_type_rule_count(), 3U);
    EXPECT_EQ(runtime.core_type_rule(0), CoreType::AIC);
    EXPECT_EQ(runtime.core_type_rule(2), CoreType::AIV);
}
