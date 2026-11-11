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

static_assert(CCU_MODULE_ABI_VERSION == SIMPLER_CCU_MODULE_ABI_VERSION);

extern "C" int simpler_ccu_test_kernel_a(const SimplerCcuRegistrationBindingV2 *);
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
    ASSERT_EQ(first->instances().size(), 3U);
    EXPECT_EQ(first->instances()[0].mission_index, 0U);
    EXPECT_EQ(first->instances()[1].mission_index, 1U);
    EXPECT_EQ(first->instances()[0].kernel_id, first->instances()[1].kernel_id);
    EXPECT_EQ(first->instances()[2].kernel_id, 4U);
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
        reinterpret_cast<SimplerCcuNativeKernelEntryV2>(const_cast<void *>(first->kernels()[0].entry));
    const auto second_entry =
        reinterpret_cast<SimplerCcuNativeKernelEntryV2>(const_cast<void *>(second->kernels()[0].entry));
    EXPECT_EQ(first_entry(nullptr), 3);
    EXPECT_EQ(second_entry(nullptr), 33);
    EXPECT_NE(first->kernels()[0].entry, second->kernels()[0].entry);
    EXPECT_EQ(cache.cached_module_count(), 2U);
}

TEST(CcuNativeModuleTest, RejectsVersionReservedKernelAndEntryViolations) {
    for (int variant = 1; variant <= 24; ++variant) {
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
    static_assert(offsetof(SimplerCcuRegistrationBindingV2, channel_handle) == 32);
    static_assert(offsetof(SimplerCcuRegistrationBindingV2, ready_event_handle) == 40);
    static_assert(offsetof(SimplerCcuRegistrationBindingV2, trigger_event_handle) == 48);
    static_assert(offsetof(SimplerCcuRegistrationBindingV2, done_event_handle) == 56);
    static_assert(offsetof(SimplerCcuMissionDescriptorV2, parameter_token) == 48);
    static_assert(offsetof(SimplerCcuMissionDescriptorV2, ready) == 56);
    static_assert(offsetof(SimplerCcuMissionDescriptorV2, trigger) == 72);
    static_assert(offsetof(SimplerCcuMissionDescriptorV2, done) == 88);

    const SimplerCcuLaunchArgumentsV2 arguments{SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION, 20, 30, 0};
    const auto *words = reinterpret_cast<const uint64_t *>(&arguments);
    ASSERT_EQ(sizeof(arguments) / sizeof(words[0]), SIMPLER_CCU_TASK_ARGUMENT_COUNT);
    EXPECT_EQ(words[0], 2U);
    EXPECT_EQ(words[1], 20U);
    EXPECT_EQ(words[2], 30U);
    EXPECT_EQ(words[3], 0U);
}
