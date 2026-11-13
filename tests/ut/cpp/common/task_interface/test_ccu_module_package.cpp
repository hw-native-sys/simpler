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
#include <cstring>
#include <vector>

#include "callable.h"
#include "ccu_module_package.h"

TEST(CcuModulePackageTest, RoundTripsExactModuleBytesInOrder) {
    const std::vector<std::vector<uint8_t>> modules{{0x7f, 'E', 'L', 'F', 0, 1}, {9, 8, 7, 6, 5}};
    const auto packed = pack_ccu_modules(modules);

    EXPECT_EQ(unpack_ccu_modules(packed.data(), packed.size()), modules);
}

TEST(CcuModulePackageTest, RejectsSizeRangeAndTrailingByteChanges) {
    const std::vector<std::vector<uint8_t>> modules{{1, 2, 3}};
    auto packed = pack_ccu_modules(modules);

    auto reserved = packed;
    auto *header = reinterpret_cast<CcuModulePackageHeader *>(reserved.data());
    header->struct_size = 1;
    EXPECT_THROW(unpack_ccu_modules(reserved.data(), reserved.size()), std::invalid_argument);

    auto range = packed;
    auto *record = reinterpret_cast<CcuModulePackageRecord *>(range.data() + sizeof(CcuModulePackageHeader));
    ++record->module_offset;
    EXPECT_THROW(unpack_ccu_modules(range.data(), range.size()), std::invalid_argument);

    packed.push_back(0);
    EXPECT_THROW(unpack_ccu_modules(packed.data(), packed.size()), std::invalid_argument);
}

TEST(CcuModulePackageTest, ChipCallableRoundTripsAndValidatesModulePackage) {
    const std::vector<std::vector<uint8_t>> modules{{1, 3, 5, 7}, {2, 4, 6, 8}};
    auto bytes = make_chip_callable(nullptr, 0, "orch", "x", 1, nullptr, nullptr, 0, "", modules);
    const auto *callable = reinterpret_cast<const ChipCallable *>(bytes.data());

    ASSERT_NO_THROW(validate_chip_callable_buffer(bytes.data(), bytes.size()));
    EXPECT_EQ(callable->ccu_modules(), modules);

    auto corrupt = bytes;
    auto *corrupt_callable = reinterpret_cast<ChipCallable *>(corrupt.data());
    ++corrupt_callable->ccu_modules_offset_;
    EXPECT_THROW(validate_chip_callable_buffer(corrupt.data(), corrupt.size()), std::invalid_argument);
}

TEST(CcuModulePackageTest, RejectsIncorrectRecordSize) {
    auto packed = pack_ccu_modules({{1, 2, 3}});
    auto *record = reinterpret_cast<CcuModulePackageRecord *>(packed.data() + sizeof(CcuModulePackageHeader));
    ASSERT_EQ(record->struct_size, sizeof(CcuModulePackageRecord));
    record->struct_size = 1;
    EXPECT_THROW(unpack_ccu_modules(packed.data(), packed.size()), std::invalid_argument);
}
