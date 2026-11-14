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

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "platform_comm/ccu/ccu_native_module.h"
#include "task_interface/ccu_module_package.h"

extern "C" int simpler_ccu_test_kernel_a(const SimplerCcuRegistrationBinding *);
extern "C" {
int simpler_ccu_test_manifest_variant = 0;
}

namespace {

std::vector<uint8_t> module_bytes() {
    Dl_info info{};
    // Taking the linked function address may identify the executable's PLT on non-PIE builds.
    const void *entry = ::dlsym(RTLD_DEFAULT, "simpler_ccu_test_kernel_a");
    if (entry == nullptr || ::dladdr(entry, &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate CCU test module");
    std::ifstream input(info.dli_fname, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

simpler::ccu::ExpectedBuildIdentity expected_identity() {
    simpler::ccu::ExpectedBuildIdentity identity{};
    identity.sdk.fill(0x11);
    identity.pto_isa.fill(0x22);
    return identity;
}

}  // namespace

TEST(CcuNativeModuleTest, LoadsExactBytesWithLocalEntriesAndStableCacheIdentity) {
    simpler_ccu_test_manifest_variant = 0;
    ASSERT_EQ(simpler_ccu_test_kernel_a(nullptr), 3);
    const auto bytes = module_bytes();
    simpler::ccu::NativeModuleCache cache;
    auto first = cache.load(bytes, expected_identity());
    auto second = cache.load(bytes, expected_identity());

    ASSERT_EQ(first, second);
    ASSERT_EQ(cache.cached_module_count(), 1U);
    EXPECT_EQ(first->module_bytes(), bytes);
    ASSERT_EQ(first->kernels().size(), 2U);
    EXPECT_EQ(first->kernels()[0].record.kernel_id, 3U);
    EXPECT_EQ(first->kernels()[0].entry_symbol, "simpler_ccu_test_kernel_a");
    EXPECT_NE(first->kernels()[0].entry, nullptr);
    EXPECT_FALSE(first->cache_identity().empty());
    EXPECT_EQ(first->kernels()[1].record.kernel_id, 4U);
}

TEST(CcuNativeModuleTest, BuildIdentityParticipatesInValidation) {
    simpler_ccu_test_manifest_variant = 0;
    auto wrong = expected_identity();
    wrong.sdk[0] = 0;
    EXPECT_THROW(simpler::ccu::NativeModule::load(module_bytes(), wrong), std::invalid_argument);
}

TEST(CcuNativeModuleTest, DistinctLiveModulesExecuteTheirOwnExactBytes) {
    simpler_ccu_test_manifest_variant = 0;
    const auto first_bytes = module_bytes();
    std::ifstream input(CCU_TEST_OTHER_MODULE_PATH, std::ios::binary);
    const std::vector<uint8_t> other_bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(other_bytes.empty());
    ASSERT_NE(first_bytes, other_bytes);
    simpler::ccu::NativeModuleCache cache;
    const auto first = cache.load(first_bytes, expected_identity());
    const auto second = cache.load(other_bytes, expected_identity());
    const auto first_entry =
        reinterpret_cast<SimplerCcuNativeKernelEntry>(const_cast<void *>(first->kernels()[0].entry));
    const auto second_entry =
        reinterpret_cast<SimplerCcuNativeKernelEntry>(const_cast<void *>(second->kernels()[0].entry));
    EXPECT_EQ(first_entry(nullptr), 3);
    EXPECT_EQ(second_entry(nullptr), 33);
    EXPECT_NE(first->kernels()[0].entry, second->kernels()[0].entry);
    EXPECT_EQ(cache.cached_module_count(), 2U);
}

TEST(CcuNativeModuleTest, RejectsSizeKernelAndEntryViolations) {
    for (int variant = 1; variant <= 18; ++variant) {
        simpler_ccu_test_manifest_variant = variant;
        EXPECT_THROW(simpler::ccu::NativeModule::load(module_bytes(), expected_identity()), std::invalid_argument)
            << "variant=" << variant;
    }
    simpler_ccu_test_manifest_variant = 0;
}

TEST(CcuNativeModuleTest, RejectsNonElfBytesBeforeLoading) {
    EXPECT_THROW(
        simpler::ccu::NativeModule::load(std::vector<uint8_t>(64, 0), expected_identity()), std::invalid_argument
    );
}

TEST(CcuNativeModuleTest, CacheHitStillValidatesTheReturnedManifest) {
    simpler_ccu_test_manifest_variant = 0;
    const auto bytes = module_bytes();
    simpler::ccu::NativeModuleCache cache;
    const auto first = cache.load(bytes, expected_identity());
    simpler_ccu_test_manifest_variant = 2;
    EXPECT_THROW(cache.load(bytes, expected_identity()), std::invalid_argument);
    simpler_ccu_test_manifest_variant = 0;
    EXPECT_EQ(cache.load(bytes, expected_identity()), first);
}

TEST(CcuNativeAbiTest, PinsRegistrationAndLaunchWordContracts) {
    static_assert(offsetof(SimplerCcuRegistrationBinding, channels) == 16);
    static_assert(offsetof(SimplerCcuRegistrationBinding, ready_event_handle) == 528);
    static_assert(offsetof(SimplerCcuRegistrationBinding, trigger_event_handle) == 536);
    static_assert(offsetof(SimplerCcuRegistrationBinding, done_event_handle) == 544);
    static_assert(offsetof(SimplerCcuControl, parameter_token) == 40);
    static_assert(offsetof(SimplerCcuControl, ready) == 48);
    static_assert(offsetof(SimplerCcuControl, trigger) == 64);
    static_assert(offsetof(SimplerCcuControl, done) == 80);

    const SimplerCcuLaunchArguments arguments{20, 30};
    const auto *words = reinterpret_cast<const uint64_t *>(&arguments);
    ASSERT_EQ(sizeof(arguments) / sizeof(words[0]), SIMPLER_CCU_TASK_ARGUMENT_COUNT);
    EXPECT_EQ(words[0], 20U);
    EXPECT_EQ(words[1], 30U);
}

TEST(CcuNativeModuleTest, SelectedDefinitionIdentityIgnoresUnrelatedModules) {
    simpler_ccu_test_manifest_variant = 0;
    const auto bytes = module_bytes();
    std::ifstream input(CCU_TEST_OTHER_MODULE_PATH, std::ios::binary);
    const std::vector<uint8_t> other{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(other.empty());
    simpler::ccu::NativeModuleCache cache;
    const auto first = simpler::ccu::select_native_kernel({bytes, bytes}, 1, expected_identity(), cache);
    const auto changed = simpler::ccu::select_native_kernel({bytes, other}, 1, expected_identity(), cache);
    EXPECT_EQ(first.cache_identity, changed.cache_identity);
    EXPECT_EQ(first.module, changed.module);
    EXPECT_EQ(first.kernel_index, 1U);
    const auto different_entry = simpler::ccu::select_native_kernel({bytes}, 0, expected_identity(), cache);
    EXPECT_NE(first.cache_identity, different_entry.cache_identity);
    const auto different_bytes = simpler::ccu::select_native_kernel({other}, 1, expected_identity(), cache);
    EXPECT_NE(first.cache_identity, different_bytes.cache_identity);
}

TEST(CcuNativeModuleTest, DefinitionIdentityIsIndependentOfCallablePosition) {
    simpler_ccu_test_manifest_variant = 0;
    const auto bytes = module_bytes();
    simpler::ccu::NativeModuleCache cache;
    const auto first = simpler::ccu::select_native_kernel({bytes}, 0, expected_identity(), cache);
    const auto shifted = simpler::ccu::select_native_kernel({bytes, bytes}, 2, expected_identity(), cache);
    EXPECT_EQ(first.cache_identity, shifted.cache_identity);
    EXPECT_EQ(first.module, shifted.module);
    EXPECT_THROW(simpler::ccu::select_native_kernel({bytes}, 2, expected_identity(), cache), std::invalid_argument);
}
