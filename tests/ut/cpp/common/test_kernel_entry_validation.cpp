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

#include "host/kernel_entry_validation.h"

namespace {

int dummy_ctx_storage = 0;
void *const kCtx = &dummy_ctx_storage;
const uint8_t kBinary[4] = {1, 2, 3, 4};
const int kConfigStorage = 0;
const void *const kConfig = &kConfigStorage;
int dummy_stream_storage = 0;
void *const kStream = &dummy_stream_storage;
alignas(ChipCallable) const unsigned char kCallableImage[sizeof(ChipCallable)] = {};

TEST(KernelEntryValidation, BinarySpanRequiresPointerAndSizeTogether) {
    EXPECT_TRUE(kernel_binary_span_is_consistent(nullptr, 0));
    EXPECT_TRUE(kernel_binary_span_is_consistent(kBinary, sizeof(kBinary)));
    EXPECT_FALSE(kernel_binary_span_is_consistent(nullptr, 4));
    EXPECT_FALSE(kernel_binary_span_is_consistent(kBinary, 0));
}

TEST(KernelEntryValidation, InitAcceptsConsistentArgs) {
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        0
    );
}

TEST(KernelEntryValidation, InitRejectsEachStructuralViolation) {
    EXPECT_EQ(
        validate_kernel_init_args(
            nullptr, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1
        ),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, nullptr, 1),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, -1, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 0),
        PTO_RUNTIME_ERR_INTERNAL
    );
    // One inconsistent span per position, in both directions.
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, nullptr, 4, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, 0, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), kBinary, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INTERNAL
    );
}

TEST(KernelEntryValidation, PrepareCallableChecksOutputPointerAndImageSize) {
    SimplerCallableHandle id{-1, 0};
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable), &id), 0);
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(nullptr, kCallableImage, sizeof(ChipCallable), &id),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, nullptr, sizeof(ChipCallable), &id), PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable), nullptr),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable) - 1, &id),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage + 1, sizeof(ChipCallable), &id),
        PTO_RUNTIME_ERR_INTERNAL
    );
}

TEST(KernelEntryValidation, LaunchChecksPointersAndIdRange) {
    EXPECT_EQ(validate_kernel_launch_args(kCtx, {0, 0}, kCallableImage, kStream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, {0, 1}, kCallableImage, kStream), 0);
    EXPECT_EQ(validate_kernel_launch_args(nullptr, {0, 1}, kCallableImage, kStream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, {0, 1}, nullptr, kStream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, {0, 1}, kCallableImage, nullptr), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, {-1, 1}, kCallableImage, kStream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(
        validate_kernel_launch_args(kCtx, {MAX_REGISTERED_CALLABLE_IDS, 1}, kCallableImage, kStream),
        PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED
    );
}

}  // namespace
