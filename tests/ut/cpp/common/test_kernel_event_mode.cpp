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

#include <acl/acl.h>
#include <runtime/rt.h>

#include <string>

#include "kernel_platform_ops.h"

namespace {
constexpr rtError_t configured_error = 107000;
uint8_t event_mode = 0;
bool configured = false;
bool competing_hardware_setter = false;
rtError_t query_error = 0;
rtError_t setter_error = 0;
rtError_t requery_error = 0;
std::string last_error_log;
int query_calls = 0;
int setter_calls = 0;
int resource_calls = 0;

class KernelEventMode : public testing::Test {
protected:
    void SetUp() override {
        event_mode = 0;
        configured = false;
        competing_hardware_setter = false;
        query_error = setter_error = requery_error = 0;
        last_error_log.clear();
        query_calls = setter_calls = resource_calls = 0;
    }
    void TearDown() override { EXPECT_EQ(resource_calls, 0); }
};
}  // namespace

extern "C" rtError_t rtEventWorkModeGet(uint8_t *mode) {
    ++query_calls;
    if (query_error != 0) return query_error;
    if (query_calls > 1 && requery_error != 0) return requery_error;
    *mode = event_mode;
    return RT_ERROR_NONE;
}

extern "C" rtError_t rtEventWorkModeSet(uint8_t mode) {
    ++setter_calls;
    EXPECT_EQ(mode, 1);
    if (competing_hardware_setter) {
        configured = true;
        event_mode = 1;
    }
    if (configured) return configured_error;
    if (setter_error != 0) return setter_error;
    configured = true;
    event_mode = mode;
    return RT_ERROR_NONE;
}

TEST_F(KernelEventMode, DefaultSoftwareBecomesHardwareAndRepeatedInitReusesIt) {
    ASSERT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(event_mode, 1);
    EXPECT_EQ(setter_calls, 1);
    ASSERT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(setter_calls, 1);
}

TEST_F(KernelEventMode, ExistingHardwareModeNeverCallsSetter) {
    event_mode = 1;
    configured = true;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(setter_calls, 0);
}

TEST_F(KernelEventMode, ExplicitSoftwareModeLogsAndContinues) {
    configured = true;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(event_mode, 0);
    EXPECT_EQ(setter_calls, 1);
    EXPECT_NE(last_error_log.find("continuing with software events"), std::string::npos);
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(event_mode, 0);
}

TEST_F(KernelEventMode, SoftwareConflictRequiresSuccessfulModeRequery) {
    configured = true;
    requery_error = 507000;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), configured_error);
    EXPECT_EQ(last_error_log.find("continuing with software events"), std::string::npos);
}

TEST_F(KernelEventMode, ConcurrentHardwareSelectionIsAcceptedAfterRequery) {
    competing_hardware_setter = true;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(event_mode, 1);
    EXPECT_EQ(query_calls, 2);
}

TEST_F(KernelEventMode, FixedHardwarePlatformsDoNotNeedTheModeApi) {
    query_error = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(setter_calls, 0);
}

TEST_F(KernelEventMode, QueryFailureIsReturnedWithoutSettingMode) {
    query_error = 507000;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), query_error);
    EXPECT_EQ(setter_calls, 0);
}

TEST_F(KernelEventMode, SetterFailureIsNotCachedAcrossInitializationAttempts) {
    setter_error = 507000;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), setter_error);
    setter_error = 0;
    EXPECT_EQ(ensure_onboard_kernel_hardware_events(), 0);
    EXPECT_EQ(setter_calls, 2);
}

// No device resource is touched by the event-mode initialization step.
extern "C" aclError aclrtGetDevice(int32_t *) { return ++resource_calls; }
extern "C" aclError aclrtSetStreamFailureMode(aclrtStream, uint64_t) { return ++resource_calls; }
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *, uint32_t) { return ++resource_calls; }
extern "C" aclError aclrtDestroyEvent(aclrtEvent) { return ++resource_calls; }
extern "C" rtError_t rtStreamCreate(rtStream_t *, int32_t) { return ++resource_calls; }
extern "C" rtError_t rtStreamDestroy(rtStream_t) { return ++resource_calls; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }
extern "C" void unified_log_error(const char *, const char *format, ...) { last_error_log = format; }
