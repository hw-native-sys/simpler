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
#include <fstream>
#include <iterator>

#include <gtest/gtest.h>

#include "platform_comm/ccu/ccu_module.h"

namespace {
using namespace simpler::ccu;

std::vector<uint8_t> fixture_bytes(const char *path = CCU_MODULE_FIXTURE_PATH) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("CCU module fixture is unavailable");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

TEST(CcuModule, SharedEmbeddedModuleResolvesOrderedEntries) {
    ModuleCache cache;
    auto binary = fixture_bytes();
    auto functions = cache.load_functions({{binary, "build_copy"}, {binary, "build_reduce"}});
    ASSERT_EQ(functions.size(), 2);
    EXPECT_EQ(functions[0].module, functions[1].module);
    auto other = cache.load_functions({{binary, "build_reduce"}});
    EXPECT_EQ(functions[0].module, other[0].module);
    binary.clear();
    Domain domain{{1, 2, 3}, 0, {{0x1000, 0x100, 0}}, std::make_shared<int>(1)};
    Instance instance(functions, domain);
    instance.prepare();
    EXPECT_EQ(instance.device_program(0), 42);
    EXPECT_EQ(instance.device_program(1), 43);
    EXPECT_TRUE(instance.close());
}

TEST(CcuModule, RejectsMissingEntryAndMalformedModule) {
    ModuleCache cache;
    EXPECT_THROW(cache.load_functions({{fixture_bytes(), "missing_entry"}}), std::runtime_error);
    EXPECT_THROW(cache.load_functions({{{1, 2, 3}, "build_copy"}}), std::invalid_argument);
    auto binary = fixture_bytes();
    binary[4] = 1;
    EXPECT_THROW(cache.load_functions({{binary, "build_copy"}}), std::invalid_argument);
}

TEST(CcuModule, FailedLoadsDiscardUnownedModuleBytes) {
    ModuleCache cache;
    auto binary = fixture_bytes();
    EXPECT_THROW(cache.load_functions({{binary, "missing_entry"}}), std::runtime_error);
    EXPECT_EQ(cache.cached_module_count(), 0);
    EXPECT_THROW(cache.load_functions({{binary, "build_copy"}, {binary, "missing_entry"}}), std::runtime_error);
    EXPECT_EQ(cache.cached_module_count(), 0);

    auto wrong_machine = binary;
    wrong_machine[18] = 0xff;
    wrong_machine[19] = 0xff;
    EXPECT_THROW(cache.load_functions({{wrong_machine, "build_copy"}}), std::runtime_error);
    EXPECT_EQ(cache.cached_module_count(), 0);

    auto retained = cache.load_functions({{binary, "build_copy"}});
    EXPECT_THROW(cache.load_functions({{binary, "missing_entry"}}), std::runtime_error);
    EXPECT_EQ(cache.cached_module_count(), 1);
    EXPECT_NE(retained[0].module, nullptr);
}

TEST(CcuModule, RejectsWrongExportedModuleAbi) {
    ModuleCache cache;
    try {
        (void)cache.load_functions({{fixture_bytes(CCU_WRONG_ABI_FIXTURE_PATH), "build_copy"}});
        FAIL() << "unsupported module ABI was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "unsupported CCU module ABI version");
    }
    EXPECT_EQ(cache.cached_module_count(), 0);
}
}  // namespace
