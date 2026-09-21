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

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "callable.h"

namespace {

std::vector<uint8_t> build_chip_callable(int32_t scalar_entries) {
    std::vector<ArgDirection> sig{ArgDirection::IN, ArgDirection::OUT};
    for (int32_t i = 0; i < scalar_entries; ++i)
        sig.push_back(ArgDirection::SCALAR);

    ArgDirection core_sig[1] = {ArgDirection::IN};
    const uint8_t kernel[] = {0x01, 0x02, 0x03, 0x04};
    auto core = make_callable<CORE_MAX_TENSOR_ARGS>(core_sig, 1, kernel, sizeof(kernel));

    const uint8_t fake_orch_so[] = {0x7f, 'E', 'L', 'F'};
    int32_t child_ids[1] = {3};
    std::vector<uint8_t> children[1] = {std::move(core)};

    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        sig.data(), static_cast<int32_t>(sig.size()), "orch_fn", fake_orch_so, sizeof(fake_orch_so), child_ids,
        children, 1, "cfg_name"
    );
}

}  // namespace

TEST(CallableScalarCount, DerivesTensorScalarSplitFromSignature) {
    for (int32_t count : {0, 1, 7, CHIP_MAX_SCALAR_ARGS, CHIP_MAX_TENSOR_ARGS - 2}) {
        auto buf = build_chip_callable(count);
        const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
        EXPECT_EQ(callable->scalar_count(), count);
        EXPECT_EQ(callable->sig_count(), 2 + count);
        EXPECT_EQ(callable->sig_count() - callable->scalar_count(), 2);
    }
}

TEST(CallableScalarCount, EmptySignatureHasNoScalars) {
    auto buf = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_fn", nullptr, 0, nullptr, nullptr, 0, ""
    );
    const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
    EXPECT_EQ(callable->scalar_count(), 0);
}

TEST(CallableScalarCount, CachedBlobUsesSignatureRegardlessOfPadding) {
    auto buf = build_chip_callable(9);
    const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
    constexpr size_t padding_begin = offsetof(ChipCallable, config_name_len_) + sizeof(uint32_t);
    constexpr size_t padding_size = offsetof(ChipCallable, storage_) - padding_begin;
    for (size_t i = padding_begin; i < offsetof(ChipCallable, storage_); ++i) {
        ASSERT_EQ(buf[i], 0);
    }
    EXPECT_EQ(callable->scalar_count(), 9);
    std::memset(buf.data() + padding_begin, 0xff, padding_size);
    EXPECT_EQ(callable->scalar_count(), 9);
    EXPECT_EQ(callable->sig_count(), 11);
    EXPECT_EQ(std::string(callable->func_name(), callable->func_name_len()), "orch_fn");
    EXPECT_EQ(std::string(callable->config_name(), callable->config_name_len()), "cfg_name");
    ASSERT_EQ(callable->child_count(), 1);
    EXPECT_EQ(callable->child_func_id(0), 3);
    EXPECT_EQ(callable->child(0).binary_size(), 4u);
}

TEST(CallableScalarCount, RejectsCorruptSignatureCountBeforeScanning) {
    for (int32_t count : {-1, CHIP_MAX_TENSOR_ARGS + 1}) {
        auto buf = build_chip_callable(9);
        auto *callable = reinterpret_cast<ChipCallable *>(buf.data());
        callable->sig_count_ = count;
        EXPECT_THROW(callable->scalar_count(), std::invalid_argument);
    }
}
