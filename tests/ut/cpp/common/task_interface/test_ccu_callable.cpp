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

#include "chip_callable_layout.h"

namespace {
std::vector<uint8_t> module_bytes() {
    std::vector<uint8_t> binary(64, 0);
    std::memcpy(binary.data(), "\177ELF", 4);
    binary[4] = 2;
    binary[5] = 1;
    binary[6] = 1;
    binary[16] = 3;
    binary[20] = 1;
    return binary;
}

std::vector<uint8_t> callable_bytes(const std::vector<CcuFunction> &functions = {}) {
    const uint8_t kernel[] = {1, 2, 3};
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kernel, sizeof(kernel));
    const int32_t id = 0;
    return make_chip_callable(nullptr, 0, "orch", kernel, sizeof(kernel), &id, &child, 1, "", functions);
}
}  // namespace

TEST(CcuCallable, RoundTripPreservesIndependentOrderAndSharesModule) {
    const auto module = module_bytes();
    auto buffer = callable_bytes({{module, "build_copy"}, {module, "build_reduce"}});
    ASSERT_NO_THROW(validate_chip_callable_buffer(buffer.data(), buffer.size()));
    const auto *callable = reinterpret_cast<const ChipCallable *>(buffer.data());
    const auto functions = callable->ccu_functions();
    ASSERT_EQ(functions.size(), 2u);
    EXPECT_EQ(functions[0].entry, "build_copy");
    EXPECT_EQ(functions[1].entry, "build_reduce");
    EXPECT_EQ(functions[0].binary, module);
    EXPECT_EQ(functions[1].binary, module);
    EXPECT_EQ(callable->child_func_id(0), 0);
    CcuDefinitionsHeader header{};
    std::memcpy(&header, callable->storage_ + callable->ccu_offset_, sizeof(header));
    EXPECT_EQ(header.module_count, 1u);
    EXPECT_EQ(header.function_count, 2u);
    EXPECT_EQ(compute_chip_callable_layout(callable).total_size, buffer.size());
}

TEST(CcuCallable, DefinitionsAffectCallableHashButNotAicoreImage) {
    const auto module = module_bytes();
    auto first = callable_bytes({{module, "copy"}, {module, "reduce"}});
    auto second = callable_bytes({{module, "reduce"}, {module, "copy"}});
    const auto a = compute_chip_callable_layout(reinterpret_cast<const ChipCallable *>(first.data()));
    const auto b = compute_chip_callable_layout(reinterpret_cast<const ChipCallable *>(second.data()));
    EXPECT_NE(a.content_hash, b.content_hash);
    EXPECT_EQ(a.aicore_image_hash, b.aicore_image_hash);
}

TEST(CcuCallable, OrdinaryCallableHasNoDefinitions) {
    auto bytes = callable_bytes();
    const auto *callable = reinterpret_cast<const ChipCallable *>(bytes.data());
    EXPECT_NO_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()));
    EXPECT_TRUE(callable->ccu_functions().empty());
    EXPECT_EQ(compute_chip_callable_layout(callable).total_size, bytes.size());
}

TEST(CcuCallable, RejectsLegacyVersionAndTruncation) {
    auto bytes = callable_bytes({{module_bytes(), "copy"}});
    EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), 16), std::invalid_argument);
    EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size() - 1), std::invalid_argument);
    auto *callable = reinterpret_cast<ChipCallable *>(bytes.data());
    callable->wire_version_ = 0;
    EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()), std::invalid_argument);
}

TEST(CcuCallable, RejectsMalformedDefinitionRangesAndAbi) {
    auto original = callable_bytes({{module_bytes(), "copy"}});
    for (bool corrupt_abi : {false, true}) {
        auto bytes = original;
        auto *callable = reinterpret_cast<ChipCallable *>(bytes.data());
        auto *record_bytes = callable->storage_ + callable->ccu_offset_ + sizeof(CcuDefinitionsHeader);
        CcuModuleRecord record{};
        std::memcpy(&record, record_bytes, sizeof(record));
        if (corrupt_abi) record.abi_version = CCU_BUILDER_ABI_VERSION + 1;
        else record.offset = 0xffffffff;
        std::memcpy(record_bytes, &record, sizeof(record));
        EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()), std::invalid_argument);
    }
}

TEST(CcuCallable, RejectsInvalidModuleAndEntry) {
    EXPECT_THROW(pack_ccu_functions({{{1, 2, 3}, "copy"}}), std::invalid_argument);
    for (const auto &entry : {"", "copy\n", "../copy", "1copy"})
        EXPECT_THROW(pack_ccu_functions({{module_bytes(), entry}}), std::invalid_argument);
}

TEST(CcuCallable, RejectsOverlappingChildrenAndDefinitions) {
    auto bytes = callable_bytes({{module_bytes(), "copy"}});
    auto *callable = reinterpret_cast<ChipCallable *>(bytes.data());
    callable->ccu_offset_ = 0;
    EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()), std::invalid_argument);
    callable->child_offsets_[0] = 0xffffffff;
    EXPECT_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()), std::invalid_argument);
}
