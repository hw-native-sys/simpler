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

// ChipCallable::scalar_count_ contract: the factory validates and records the
// count, the field occupies former header padding only, and a blob written by
// a producer that predates the field (whose padding make_callable
// zero-initialized) reads back as scalar_count == 0.

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "callable.h"

namespace {

std::vector<uint8_t> build_chip_callable(int32_t scalar_count) {
    ArgDirection sig[2] = {ArgDirection::IN, ArgDirection::OUT};

    ArgDirection core_sig[1] = {ArgDirection::IN};
    const uint8_t kernel[] = {0x01, 0x02, 0x03, 0x04};
    auto core = make_callable<CORE_MAX_TENSOR_ARGS>(core_sig, 1, kernel, sizeof(kernel));

    const uint8_t fake_orch_so[] = {0x7f, 'E', 'L', 'F'};
    int32_t child_ids[1] = {3};
    std::vector<uint8_t> children[1] = {std::move(core)};

    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        sig, 2, scalar_count, "orch_fn", fake_orch_so, sizeof(fake_orch_so), child_ids, children, 1, "cfg_name"
    );
}

}  // namespace

TEST(CallableScalarCount, RoundTripsThroughFactory) {
    for (int32_t count : {0, 1, 7, CHIP_MAX_SCALAR_ARGS}) {
        auto buf = build_chip_callable(count);
        const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
        EXPECT_EQ(callable->scalar_count(), count);
    }
}

TEST(CallableScalarCount, RejectsOutOfRangeWithValueAndLimit) {
    for (int32_t count : {-1, CHIP_MAX_SCALAR_ARGS + 1}) {
        try {
            (void)build_chip_callable(count);
            FAIL() << "expected scalar count validation to fail for " << count;
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(std::to_string(count)), std::string::npos) << message;
            EXPECT_NE(message.find(std::to_string(CHIP_MAX_SCALAR_ARGS)), std::string::npos) << message;
        }
    }
}

// A legacy blob is byte-identical to a current one built with the same inputs
// except that the four bytes now holding scalar_count_ were zero-initialized
// padding. Zeroing them reproduces such a blob exactly.
TEST(CallableScalarCount, LegacyBlobReadsZero) {
    auto buf = build_chip_callable(9);
    std::memset(buf.data() + offsetof(ChipCallable, scalar_count_), 0, sizeof(int32_t));

    const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
    EXPECT_EQ(callable->scalar_count(), 0);
    EXPECT_EQ(callable->sig_count(), 2);
    EXPECT_EQ(std::string(callable->func_name(), callable->func_name_len()), "orch_fn");
    EXPECT_EQ(std::string(callable->config_name(), callable->config_name_len()), "cfg_name");
    ASSERT_EQ(callable->child_count(), 1);
    EXPECT_EQ(callable->child_func_id(0), 3);
    EXPECT_EQ(callable->child(0).binary_size(), 4u);
}

// Two builds differing only in scalar_count must differ only in that field's
// four bytes — every other header byte, the orchestration binary, and the
// child payload keep their positions and values.
TEST(CallableScalarCount, OnlyTheFieldBytesVary) {
    auto zero = build_chip_callable(0);
    auto nine = build_chip_callable(9);
    ASSERT_EQ(zero.size(), nine.size());

    constexpr size_t field_begin = offsetof(ChipCallable, scalar_count_);
    constexpr size_t field_end = field_begin + sizeof(int32_t);
    for (size_t i = 0; i < zero.size(); ++i) {
        if (i >= field_begin && i < field_end) continue;
        ASSERT_EQ(zero[i], nine[i]) << "byte " << i << " must not depend on scalar_count";
    }
    EXPECT_NE(std::memcmp(zero.data() + field_begin, nine.data() + field_begin, sizeof(int32_t)), 0);
}
