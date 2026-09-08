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

// ChipCallable::scalar_count_ contract: the field is a cached derivation of
// the signature's SCALAR entries (0 also meaning "not recorded"), the factory
// validates range and signature agreement, the field occupies former header
// padding only, and a blob written by a producer that predates the field
// (whose padding make_callable zero-initialized) reads back as
// scalar_count == 0.

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "callable.h"

namespace {

// Builds a chip callable whose signature carries `scalar_entries` trailing
// SCALAR entries after one IN and one OUT tensor, declaring `declared_count`
// as its scalar count.
std::vector<uint8_t> build_chip_callable(int32_t declared_count, int32_t scalar_entries) {
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
        sig.data(), static_cast<int32_t>(sig.size()), declared_count, "orch_fn", fake_orch_so, sizeof(fake_orch_so),
        child_ids, children, 1, "cfg_name"
    );
}

// scalar_count() reads 0 both for a scalar-free orchestration and for an
// artifact built before the field existed, so the wire contract has a consumer
// derive the effective count from the signature when the field is 0 rather
// than subtracting it. This pins the answer the naive formula gets wrong, so a
// consumer that reintroduces it fails here rather than on device.
TEST(CallableScalarCount, SubtractingAnUnrecordedCountMiscountsScalarsAsTensors) {
    auto buf = build_chip_callable(0, 9);
    const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
    ASSERT_EQ(callable->sig_count(), 11);
    ASSERT_EQ(callable->scalar_count(), 0);

    int32_t signature_scalars = 0;
    for (int32_t i = 0; i < callable->sig_count(); ++i) {
        if (callable->sig(i) == ArgDirection::SCALAR) ++signature_scalars;
    }
    ASSERT_EQ(signature_scalars, 9);

    const int32_t effective_scalars = callable->scalar_count() != 0 ? callable->scalar_count() : signature_scalars;
    EXPECT_EQ(callable->sig_count() - effective_scalars, 2);
    EXPECT_EQ(callable->sig_count() - callable->scalar_count(), 11);
}

}  // namespace

TEST(CallableScalarCount, RoundTripsThroughFactory) {
    for (int32_t count : {0, 1, 7, CHIP_MAX_SCALAR_ARGS}) {
        auto buf = build_chip_callable(count, count);
        const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
        EXPECT_EQ(callable->scalar_count(), count);
        EXPECT_EQ(callable->sig_count(), 2 + count);
    }
}

TEST(CallableScalarCount, RejectsOutOfRangeWithValueAndLimit) {
    for (int32_t count : {-1, CHIP_MAX_SCALAR_ARGS + 1}) {
        try {
            (void)build_chip_callable(count, count > 0 ? count : 0);
            FAIL() << "expected scalar count validation to fail for " << count;
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(std::to_string(count)), std::string::npos) << message;
            EXPECT_NE(message.find(std::to_string(CHIP_MAX_SCALAR_ARGS)), std::string::npos) << message;
        }
    }
}

// A nonzero count is a cached derivation of the signature, so a value that
// disagrees with the signature's SCALAR entries is rejected naming both
// numbers. Zero stays valid with any signature (legacy "not recorded").
TEST(CallableScalarCount, RejectsCountDisagreeingWithSignature) {
    struct Case {
        int32_t declared;
        int32_t entries;
    };
    for (const Case &c : {Case{5, 0}, Case{3, 9}}) {
        try {
            (void)build_chip_callable(c.declared, c.entries);
            FAIL() << "expected consistency validation to fail for declared=" << c.declared << " entries=" << c.entries;
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(std::to_string(c.declared)), std::string::npos) << message;
            EXPECT_NE(message.find(std::to_string(c.entries)), std::string::npos) << message;
        }
    }
    auto unrecorded = build_chip_callable(0, 9);
    EXPECT_EQ(reinterpret_cast<const ChipCallable *>(unrecorded.data())->scalar_count(), 0);
}

// A legacy blob is byte-identical to a current one built with the same inputs
// except that the four bytes now holding scalar_count_ were zero-initialized
// padding. Zeroing them reproduces such a blob exactly.
TEST(CallableScalarCount, LegacyBlobReadsZero) {
    auto buf = build_chip_callable(9, 9);
    std::memset(buf.data() + offsetof(ChipCallable, scalar_count_), 0, sizeof(int32_t));

    const auto *callable = reinterpret_cast<const ChipCallable *>(buf.data());
    EXPECT_EQ(callable->scalar_count(), 0);
    EXPECT_EQ(callable->sig_count(), 11);
    EXPECT_EQ(std::string(callable->func_name(), callable->func_name_len()), "orch_fn");
    EXPECT_EQ(std::string(callable->config_name(), callable->config_name_len()), "cfg_name");
    ASSERT_EQ(callable->child_count(), 1);
    EXPECT_EQ(callable->child_func_id(0), 3);
    EXPECT_EQ(callable->child(0).binary_size(), 4u);
}

// Two builds with the same signature that differ only in the declared count
// (0 = "not recorded" vs the real derivation) must differ only in that
// field's four bytes — every other header byte, the orchestration binary,
// and the child payload keep their positions and values.
TEST(CallableScalarCount, OnlyTheFieldBytesVary) {
    auto unrecorded = build_chip_callable(0, 9);
    auto nine = build_chip_callable(9, 9);
    ASSERT_EQ(unrecorded.size(), nine.size());

    constexpr size_t field_begin = offsetof(ChipCallable, scalar_count_);
    constexpr size_t field_end = field_begin + sizeof(int32_t);
    for (size_t i = 0; i < unrecorded.size(); ++i) {
        if (i >= field_begin && i < field_end) continue;
        ASSERT_EQ(unrecorded[i], nine[i]) << "byte " << i << " must not depend on scalar_count";
    }
    EXPECT_NE(std::memcmp(unrecorded.data() + field_begin, nine.data() + field_begin, sizeof(int32_t)), 0);
}
