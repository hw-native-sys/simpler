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
/**
 * Unit tests for common/acl_hal_device.h::acl_to_hal_device_id — the ACL-logical
 * to driver-visible device-id translation applied at direct-HAL call sites so a
 * chip forked under ASCEND_RT_VISIBLE_DEVICES isolation targets the right device
 * (regression barrier for the halMemCtl rc=42 this PR fixes). Pure + hardware-
 * free: it only parses the env var.
 */
#include <gtest/gtest.h>

#include <cstdlib>

#include "common/acl_hal_device.h"

// Minimal logger stubs: the header emits LOG_WARN / LOG_DEBUG on the diagnostic
// paths. The unified_log symbols have C linkage (see common/unified_log.h);
// these tests assert on the returned id, not log text.
extern "C" {
void unified_log_error(const char *, const char *, ...) {}
void unified_log_warn(const char *, const char *, ...) {}
void unified_log_timing(const char *, const char *, ...) {}
void unified_log_info(const char *, const char *, ...) {}
void unified_log_debug(const char *, const char *, ...) {}
}

namespace {

void set_vis(const char *value) {
    if (value != nullptr) {
        setenv("ASCEND_RT_VISIBLE_DEVICES", value, 1);
    } else {
        unsetenv("ASCEND_RT_VISIBLE_DEVICES");
    }
}

TEST(AclToHalDeviceId, IdentityWhenUnsetOrEmpty) {
    set_vis(nullptr);
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 0);
    EXPECT_EQ(pto::acl_to_hal_device_id(3), 3);
    set_vis("");
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 0);
}

TEST(AclToHalDeviceId, RemapsLogicalToDriverVisible) {
    set_vis("4,5,6,7");
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 4);
    EXPECT_EQ(pto::acl_to_hal_device_id(2), 6);  // the reproduced VIS -> wrong-device case
    EXPECT_EQ(pto::acl_to_hal_device_id(3), 7);
    set_vis("6");
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 6);
}

TEST(AclToHalDeviceId, WhitespaceAndEmptyFieldsTolerated) {
    set_vis(" 4 , 5 , 6 ");
    EXPECT_EQ(pto::acl_to_hal_device_id(1), 5);
    set_vis("4,,6");
    EXPECT_EQ(pto::acl_to_hal_device_id(1), 6);
}

TEST(AclToHalDeviceId, MalformedListFallsBackToIdentity) {
    set_vis("-1");
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 0);
    set_vis("99999999999999999999");  // overflows an int -> rejected
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 0);
    set_vis("6x");  // token not terminated by a delimiter
    EXPECT_EQ(pto::acl_to_hal_device_id(0), 0);
    set_vis("4,5,6,7x");  // a bad token anywhere invalidates the whole list, even after the match
    EXPECT_EQ(pto::acl_to_hal_device_id(2), 2);
}

TEST(AclToHalDeviceId, OutOfRangeAndNegativeLogicalFallBack) {
    set_vis("4,5,6,7");
    EXPECT_EQ(pto::acl_to_hal_device_id(4), 4);    // logical id beyond the visible list
    EXPECT_EQ(pto::acl_to_hal_device_id(-3), -3);  // negative logical id
}

}  // namespace
