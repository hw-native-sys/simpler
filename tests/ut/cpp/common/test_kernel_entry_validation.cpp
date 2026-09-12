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

#include <cstring>

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
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage, kKernelCallableByteLimit + 1, &id),
        PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED
    );
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

TEST(KernelEntryValidation, PrepareRejectsInvalidInvocationSignatureBeforeRegistration) {
    SimplerCallableHandle handle{-1, 0};
    alignas(ChipCallable) unsigned char image[sizeof(ChipCallable)] = {};
    auto set_count = [&](size_t offset, int32_t count) {
        std::memcpy(image + offset, &count, sizeof(count));
    };
    auto set_direction = [&](int32_t index, ArgDirection direction) {
        std::memcpy(
            image + offsetof(ChipCallable, signature_) + index * sizeof(direction), &direction, sizeof(direction)
        );
    };
    set_count(offsetof(ChipCallable, sig_count_), -1);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), PTO_RUNTIME_ERR_INTERNAL);
    set_count(offsetof(ChipCallable, sig_count_), CHIP_MAX_TENSOR_ARGS + 1);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), PTO_RUNTIME_ERR_INTERNAL);
    set_count(offsetof(ChipCallable, sig_count_), 2);
    set_direction(0, ArgDirection::IN);
    set_direction(1, ArgDirection::SCALAR);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), 0);
    set_count(offsetof(ChipCallable, scalar_count_), 1);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), 0);
    set_count(offsetof(ChipCallable, scalar_count_), 2);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), PTO_RUNTIME_ERR_INTERNAL);
    set_count(offsetof(ChipCallable, scalar_count_), 0);
    set_direction(0, ArgDirection::SCALAR);
    set_direction(1, ArgDirection::OUT);
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), PTO_RUNTIME_ERR_INTERNAL);
    set_direction(0, static_cast<ArgDirection>(99));
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, image, sizeof(image), &handle), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelEntryValidation, PrepareBoundsCanonicalImageBeforeHashOrUpload) {
    SimplerCallableHandle handle{-1, 0};
    const uint8_t binary[] = {1, 2, 3};
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary));
    const int32_t func_id = 0;
    const auto valid = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, 0, "entry", binary, sizeof(binary), &func_id, &child, 1, "config"
    );
    ASSERT_EQ(validate_kernel_prepare_callable_args(kCtx, valid.data(), valid.size(), &handle), 0);
    const auto *header = reinterpret_cast<const ChipCallable *>(valid.data());
    const size_t child_start = offsetof(ChipCallable, storage_) + header->child_offset(0);
    auto reject_word = [&](size_t offset, uint32_t value) {
        auto image = valid;
        std::memcpy(image.data() + offset, &value, sizeof(value));
        EXPECT_EQ(
            validate_kernel_prepare_callable_args(kCtx, image.data(), image.size(), &handle), PTO_RUNTIME_ERR_INTERNAL
        );
    };
    reject_word(offsetof(ChipCallable, binary_size_), UINT32_MAX);
    reject_word(offsetof(ChipCallable, child_count_), UINT32_MAX);
    reject_word(offsetof(ChipCallable, child_count_), 1025);
    reject_word(offsetof(ChipCallable, child_offsets_), UINT32_MAX);
    reject_word(offsetof(ChipCallable, child_offsets_), 1);
    reject_word(offsetof(ChipCallable, child_offsets_), 0);
    reject_word(child_start + offsetof(CoreCallable, binary_size_), UINT32_MAX);
    reject_word(child_start + offsetof(CoreCallable, sig_count_), CORE_MAX_TENSOR_ARGS + 1);
    reject_word(offsetof(ChipCallable, func_name_len_), CALLABLE_FUNC_NAME_MAX);
    reject_word(offsetof(ChipCallable, config_name_len_), CALLABLE_FUNC_NAME_MAX);
    auto unterminated = valid;
    std::memset(unterminated.data() + offsetof(ChipCallable, func_name_), 'x', CALLABLE_FUNC_NAME_MAX);
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, unterminated.data(), unterminated.size(), &handle),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, valid.data(), child_start + sizeof(CoreCallable) - 1, &handle),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, valid.data(), valid.size() - 1, &handle), PTO_RUNTIME_ERR_INTERNAL
    );
    auto trailing = valid;
    trailing.push_back(0);
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, trailing.data(), trailing.size(), &handle), PTO_RUNTIME_ERR_INTERNAL
    );
}

}  // namespace
