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

TEST(KernelEntryValidation, RunRejectsTransfersBeforeReadingAnyTensorBytes) {
    ChipStorageTaskArgs args{};
    const uint32_t shape[] = {1};
    args.add_tensor(make_tensor_external(
        reinterpret_cast<void *>(1), shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE
    ));
    EXPECT_EQ(validate_kernel_run_args(&args), 0);
    args.tensor(0).address_space = AddressSpace::HOST;
    EXPECT_EQ(validate_kernel_run_args(&args), 0);
    for (auto space : {AddressSpace::HOST, AddressSpace::DEVICE}) {
        args.tensor(0).address_space = space;
        for (auto request : {TensorTransfer::H2D, TensorTransfer::D2H, static_cast<TensorTransfer>(255)}) {
            args.tensor(0).transfer = request;
            EXPECT_EQ(validate_kernel_run_args(&args), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
        }
    }
    args.tensor(0).transfer = TensorTransfer::NONE;
    args.tensor(0).address_space = static_cast<AddressSpace>(255);
    EXPECT_EQ(validate_kernel_run_args(&args), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_run_args(nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    args.tensor_count_ = CHIP_MAX_TENSOR_ARGS + 1;
    EXPECT_EQ(validate_kernel_run_args(&args), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    args.tensor_count_ = 0;
    args.scalar_count_ = CHIP_MAX_SCALAR_ARGS + 1;
    EXPECT_EQ(validate_kernel_run_args(&args), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

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
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, nullptr, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, -1, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // One inconsistent span per position, in both directions.
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, nullptr, 4, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, 0, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), nullptr, 4, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, 0, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), kBinary, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 4, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, PrepareCallableChecksIdRangeAndImageSize) {
    EXPECT_EQ(validate_callable_registration_args(kCtx, 0, kCallableImage, sizeof(ChipCallable)), 0);
    EXPECT_EQ(
        validate_callable_registration_args(
            kCtx, MAX_REGISTERED_CALLABLE_IDS - 1, kCallableImage, sizeof(ChipCallable)
        ),
        0
    );
    EXPECT_EQ(
        validate_callable_registration_args(nullptr, 0, kCallableImage, sizeof(ChipCallable)),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_callable_registration_args(kCtx, 0, nullptr, sizeof(ChipCallable)), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_callable_registration_args(kCtx, -1, kCallableImage, sizeof(ChipCallable)),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_callable_registration_args(kCtx, MAX_REGISTERED_CALLABLE_IDS, kCallableImage, sizeof(ChipCallable)),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_callable_registration_args(kCtx, 0, kCallableImage, sizeof(ChipCallable) - 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // storage_ sits at a CALLABLE_CHILD_ALIGN offset from the header, so an
    // image the caller placed off-alignment puts every child off-alignment.
    static_assert(alignof(ChipCallable) > 1, "a misaligned image must be expressible");
    EXPECT_EQ(
        validate_callable_registration_args(kCtx, 0, kCallableImage + 1, sizeof(ChipCallable)),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, CallableImageAcceptsExpandedChildCapacity) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    for (int32_t count : {1024, 1025, 1427, ChipCallable::MAX_CHILDREN - 1, ChipCallable::MAX_CHILDREN}) {
        SCOPED_TRACE(count);
        std::vector<int32_t> ids(count);
        std::vector<std::vector<uint8_t>> children(count, child);
        for (int32_t i = 0; i < count; ++i)
            ids[i] = i;
        auto image =
            make_chip_callable(nullptr, 0, "orch", kBinary, sizeof(kBinary), ids.data(), children.data(), count, "");
        EXPECT_EQ(validate_callable_registration_args(kCtx, 0, image.data(), image.size()), 0);
        EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), 0);
        auto *callable = reinterpret_cast<ChipCallable *>(image.data());
        callable->child_offsets_[count - 1] = static_cast<uint32_t>(image.size());
        EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    }
}

TEST(KernelEntryValidation, CallableImageRejectsChildCountOverflow) {
    auto image = make_chip_callable(nullptr, 0, "orch", nullptr, 0, nullptr, nullptr, 0, "");
    auto *callable = reinterpret_cast<ChipCallable *>(image.data());
    for (int32_t count : {-1, ChipCallable::MAX_CHILDREN + 1}) {
        SCOPED_TRACE(count);
        callable->child_count_ = count;
        EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    }
}

TEST(KernelEntryValidation, CallableImageRejectsTruncatedVariableTail) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    const int32_t child_id = 7;
    const std::vector<uint8_t> children[] = {child};
    const auto image = make_chip_callable(nullptr, 0, "orch", kBinary, sizeof(kBinary), &child_id, children, 1, "");
    ASSERT_GT(image.size(), sizeof(ChipCallable));
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), 0);
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size() - 1), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(KernelEntryValidation, CallableImageRejectsOutOfRangeChildMetadata) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    const int32_t child_id = 7;
    const std::vector<uint8_t> children[] = {child};
    auto image = make_chip_callable(nullptr, 0, "orch", nullptr, 0, &child_id, children, 1, "");
    auto *callable = reinterpret_cast<ChipCallable *>(image.data());
    callable->child_offsets_[0] = static_cast<uint32_t>(image.size());
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

}  // namespace
