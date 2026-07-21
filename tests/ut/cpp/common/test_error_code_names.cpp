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
 * Error-code name tables: every code an operator can see must translate to text.
 *
 * The e2e coverage in tests/st/runtime_fatal_codes only reaches the codes a workload can
 * actually provoke. Codes 10, 11 and 103 are argued unreachable there and so have no ST --
 * but an unreachable code is exactly the one that shows up as a bare number on the day it
 * finally fires. These tests hold the tables complete regardless of reachability.
 */

#include <gtest/gtest.h>

#include <cstring>

#include "pto_runtime_status.h"

#include "runtime_status/error_names.h"

#include "host/acl_error_names.h"

namespace {

bool is_blank(const char *s) { return s == nullptr || s[0] == '\0'; }

// Every PTO2 error code, including the ones no ST can provoke. 6 is retired and absent from
// the tensormap_and_ringbuffer header this test compiles against.
constexpr int32_t kAllRuntimeCodes[] = {
    PTO2_ERROR_SCOPE_DEADLOCK,        PTO2_ERROR_HEAP_RING_DEADLOCK,
    PTO2_ERROR_FLOW_CONTROL_DEADLOCK, PTO2_ERROR_DEP_POOL_OVERFLOW,
    PTO2_ERROR_INVALID_ARGS,          PTO2_ERROR_REQUIRE_SYNC_START_INVALID,
    PTO2_ERROR_TENSOR_WAIT_TIMEOUT,   PTO2_ERROR_EXPLICIT_ORCH_FATAL,
    PTO2_ERROR_SCOPE_TASKS_OVERFLOW,  PTO2_ERROR_TENSORMAP_OVERFLOW,
    PTO2_ERROR_SCHEDULER_TIMEOUT,     PTO2_ERROR_ASYNC_COMPLETION_INVALID,
    PTO2_ERROR_ASYNC_WAIT_OVERFLOW,   PTO2_ERROR_ASYNC_REGISTRATION_FAILED,
};

TEST(RuntimeErrorNames, EveryCodeHasNameDescAndHint) {
    for (int32_t code : kAllRuntimeCodes) {
        EXPECT_STRNE(error_name(code), "unknown") << "code " << code << " is missing from the name table";
        EXPECT_FALSE(is_blank(error_desc(code))) << "code " << code << " has no description";
        EXPECT_FALSE(is_blank(error_hint(code))) << "code " << code << " has no triage hint";
    }
}

TEST(RuntimeErrorNames, NamesAreDistinct) {
    for (size_t i = 0; i < std::size(kAllRuntimeCodes); ++i) {
        for (size_t j = i + 1; j < std::size(kAllRuntimeCodes); ++j) {
            EXPECT_STRNE(error_name(kAllRuntimeCodes[i]), error_name(kAllRuntimeCodes[j]))
                << "codes " << kAllRuntimeCodes[i] << " and " << kAllRuntimeCodes[j] << " share a name";
        }
    }
}

TEST(RuntimeErrorNames, NoErrorIsNotAnError) {
    EXPECT_STREQ(error_name(PTO2_ERROR_NONE), "none");
    EXPECT_TRUE(is_blank(error_hint(PTO2_ERROR_NONE)));
}

// An unrecognised code must be reported as such. Annotating it with a neighbouring code's
// text would be worse than printing the bare number.
TEST(RuntimeErrorNames, UnknownCodeFallsBack) {
    for (int32_t code : {6, 12, 42, 99, 104, 9999, -1}) {
        EXPECT_STREQ(error_name(code), "unknown") << "code " << code;
        EXPECT_TRUE(is_blank(error_desc(code))) << "code " << code;
        EXPECT_TRUE(is_blank(error_hint(code))) << "code " << code;
    }
}

// The orchestrator and the scheduler each latch into their own field, and at most one is
// ever non-zero. The annotation has to name the field the code actually came from.
TEST(RuntimeErrorNames, LatchedCodePicksTheNonZeroField) {
    EXPECT_EQ(latched_error_code(PTO2_ERROR_SCOPE_DEADLOCK, PTO2_ERROR_NONE), PTO2_ERROR_SCOPE_DEADLOCK);
    EXPECT_STREQ(latched_error_field(PTO2_ERROR_SCOPE_DEADLOCK, PTO2_ERROR_NONE), "orch_error_code");

    EXPECT_EQ(latched_error_code(PTO2_ERROR_NONE, PTO2_ERROR_SCHEDULER_TIMEOUT), PTO2_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_STREQ(latched_error_field(PTO2_ERROR_NONE, PTO2_ERROR_SCHEDULER_TIMEOUT), "sched_error_code");

    EXPECT_EQ(latched_error_code(PTO2_ERROR_NONE, PTO2_ERROR_NONE), PTO2_ERROR_NONE);
}

// The stall sub-class already had a name table; keep it and the error table consistent so a
// SCHEDULER_TIMEOUT annotation and its sub_class= line cannot disagree.
TEST(RuntimeErrorNames, StallDetailStillNamed) {
    EXPECT_STREQ(stall_detail_name(PTO2_STALL_DETAIL_RUNNING_STALLED), "S1:running-stalled");
    EXPECT_STREQ(stall_detail_name(PTO2_STALL_DETAIL_UNKNOWN), "unknown:accounting/corruption");
}

// ---------------------------------------------------------------------------
// ACL / RT codes (host side)
// ---------------------------------------------------------------------------

// The codes conftest treats as device-poisoning, plus the ones the onboard rules doc triages.
// If a code is worth branching on, it is worth naming in the log.
constexpr int32_t kKnownAclCodes[] = {
    SIMPLER_DEVICE_UNUSABLE, 107022, 207001, 507000, 507014, 507015, 507017, 507018, 507046, 507899,
};

TEST(AclErrorNames, KnownCodesHaveNameDescAndHint) {
    for (int32_t rc : kKnownAclCodes) {
        ASSERT_NE(acl_error_name(rc), nullptr) << "rc " << rc << " is missing from the name table";
        EXPECT_FALSE(is_blank(acl_error_desc(rc))) << "rc " << rc << " has no description";
        EXPECT_FALSE(is_blank(acl_error_hint(rc))) << "rc " << rc << " has no triage hint";
    }
}

// 207001 is ACL_ERROR_RT_MEMORY_ALLOCATION per CANN's rt_error_codes.h -- not the "AICore
// launch failure" that this repo's comments long claimed. Pin the authoritative names so the
// log cannot re-inherit the folklore.
TEST(AclErrorNames, NamesMatchCann) {
    EXPECT_STREQ(acl_error_name(207001), "ACL_ERROR_RT_MEMORY_ALLOCATION");
    EXPECT_STREQ(acl_error_name(507000), "ACL_ERROR_RT_INTERNAL_ERROR");
    EXPECT_STREQ(acl_error_name(507018), "ACL_ERROR_RT_AICPU_EXCEPTION");
    EXPECT_STREQ(acl_error_name(507046), "ACL_ERROR_RT_STREAM_SYNC_TIMEOUT");
    EXPECT_STREQ(acl_error_name(507899), "ACL_ERROR_RT_DRV_INTERNAL_ERROR");
}

TEST(AclErrorNames, UnknownCodeIsNotGuessedAt) {
    for (int32_t rc : {0, 1, 507019, 600000}) {
        EXPECT_EQ(acl_error_name(rc), nullptr) << "rc " << rc;
        EXPECT_EQ(acl_error_desc(rc), nullptr) << "rc " << rc;
        EXPECT_EQ(acl_error_hint(rc), nullptr) << "rc " << rc;
    }
}

}  // namespace
