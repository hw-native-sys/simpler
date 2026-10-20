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

#include <dlfcn.h>
#include <unistd.h>

#include <memory>
#include <stdexcept>

#include "runtime_module.h"

namespace {

int unloads = 0;
using Value = int (*)();
using CountUnloads = void (*)(int *);

class RuntimeModuleTest : public ::testing::Test {
protected:
    void SetUp() override { unloads = 0; }
    static std::shared_ptr<RuntimeModule> open_module() {
        auto module = std::make_shared<RuntimeModule>(RUNTIME_MODULE_FIXTURE_PATH);
        module->required<CountUnloads>("runtime_module_fixture_count_unloads")(&unloads);
        return module;
    }
};

TEST_F(RuntimeModuleTest, UnloadsOnlyAfterTheLastOwner) {
    auto module = open_module();
    auto dependent = module;
    std::weak_ptr<RuntimeModule> weak = module;
    const auto value = module->required<Value>("runtime_module_fixture_value");
    module.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(unloads, 0);
    EXPECT_EQ(value(), 37);
    dependent.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(unloads, 1);
}

TEST_F(RuntimeModuleTest, IndependentLoadsEachRetainADynamicLoaderReference) {
    auto first = open_module();
    auto second = open_module();
    first.reset();
    EXPECT_EQ(unloads, 0);
    EXPECT_EQ(second->required<Value>("runtime_module_fixture_value")(), 37);
    second.reset();
    EXPECT_EQ(unloads, 1);
}

TEST_F(RuntimeModuleTest, MissingRequiredSymbolLeavesTheOwnerUsable) {
    auto module = open_module();
    EXPECT_THROW(module->required<Value>("runtime_module_fixture_missing"), std::runtime_error);
    EXPECT_EQ(unloads, 0);
    EXPECT_EQ(module->required<Value>("runtime_module_fixture_value")(), 37);
    module.reset();
    EXPECT_EQ(unloads, 1);
}

TEST_F(RuntimeModuleTest, OptionalLookupDoesNotPoisonRequiredLookup) {
    auto module = open_module();
    EXPECT_EQ(module->optional<Value>("runtime_module_fixture_missing"), nullptr);
    EXPECT_EQ(module->required<Value>("runtime_module_fixture_value")(), 37);
}

TEST_F(RuntimeModuleTest, RuntimeSymbolsStayOutOfGlobalLookup) {
    auto module = open_module();
    EXPECT_NE(module->optional<Value>("runtime_module_fixture_value"), nullptr);
    EXPECT_EQ(dlsym(RTLD_DEFAULT, "runtime_module_fixture_value"), nullptr);
}

TEST_F(RuntimeModuleTest, FailedOpenOwnsNoModule) {
    EXPECT_THROW(RuntimeModule("runtime-module-does-not-exist.so"), std::runtime_error);
    EXPECT_EQ(unloads, 0);
}

TEST_F(RuntimeModuleTest, TerminalRetentionKeepsCodeAfterEveryCppOwnerLeaves) {
    EXPECT_EXIT(
        {
            auto module = open_module();
            const auto value = module->required<Value>("runtime_module_fixture_value");
            module->retain_until_process_exit();
            module.reset();
            _exit(unloads == 0 && value() == 37 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

}  // namespace
