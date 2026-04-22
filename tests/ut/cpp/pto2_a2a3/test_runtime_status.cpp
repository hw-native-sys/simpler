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
 * UT for pto2_runtime_status_from_error_codes (pto_runtime_status.h).
 *
 * The helper merges an orchestrator code (1-99) and a scheduler code (100+)
 * into a single negative AICPU return code. Orchestrator code wins when both
 * are non-zero; positive codes are negated; already-negative codes pass
 * through unchanged.
 *
 * Pure header-only function -- no runtime linkage required.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "pto_runtime_status.h"

// ---------- happy path ----------

TEST(RuntimeStatus, BothZero_ReturnsZero) {
    EXPECT_EQ(pto2_runtime_status_from_error_codes(PTO2_ERROR_NONE, PTO2_ERROR_NONE), 0);
}

// ---------- single-channel ----------

TEST(RuntimeStatus, OrchOnly_NegatesPositiveCode) {
    EXPECT_EQ(
        pto2_runtime_status_from_error_codes(PTO2_ERROR_SCOPE_DEADLOCK, PTO2_ERROR_NONE), -PTO2_ERROR_SCOPE_DEADLOCK
    );
}

TEST(RuntimeStatus, SchedOnly_NegatesPositiveCode) {
    EXPECT_EQ(
        pto2_runtime_status_from_error_codes(PTO2_ERROR_NONE, PTO2_ERROR_SCHEDULER_TIMEOUT),
        -PTO2_ERROR_SCHEDULER_TIMEOUT
    );
}

// ---------- precedence: orchestrator wins ----------

TEST(RuntimeStatus, BothNonZero_OrchTakesPrecedence) {
    int32_t result = pto2_runtime_status_from_error_codes(PTO2_ERROR_INVALID_ARGS, PTO2_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(result, -PTO2_ERROR_INVALID_ARGS);
}

// ---------- already-negative passthrough (idempotency) ----------

TEST(RuntimeStatus, NegativeOrchCode_PassesThrough) {
    EXPECT_EQ(pto2_runtime_status_from_error_codes(-7, PTO2_ERROR_NONE), -7);
}

TEST(RuntimeStatus, NegativeSchedCode_PassesThrough) {
    EXPECT_EQ(pto2_runtime_status_from_error_codes(PTO2_ERROR_NONE, -101), -101);
}

// ---------- equivalence-class coverage of every defined code ----------

TEST(RuntimeStatus, AllOrchestratorCodes_AreNegated) {
    const int32_t codes[] = {
        PTO2_ERROR_SCOPE_DEADLOCK,
        PTO2_ERROR_HEAP_RING_DEADLOCK,
        PTO2_ERROR_FLOW_CONTROL_DEADLOCK,
        PTO2_ERROR_DEP_POOL_OVERFLOW,
        PTO2_ERROR_INVALID_ARGS,
        PTO2_ERROR_DEPENDENCY_OVERFLOW,
        PTO2_ERROR_REQUIRE_SYNC_START_INVALID,
        PTO2_ERROR_TENSOR_WAIT_TIMEOUT,
        PTO2_ERROR_EXPLICIT_ORCH_FATAL,
    };
    for (int32_t code : codes) {
        SCOPED_TRACE(testing::Message() << "orch_code=" << code);
        EXPECT_EQ(pto2_runtime_status_from_error_codes(code, PTO2_ERROR_NONE), -code);
    }
}

// ---------- contract guard: PTO2_ERROR_NONE is the only zero ----------

TEST(RuntimeStatus, NoneIsZero) { EXPECT_EQ(PTO2_ERROR_NONE, 0); }

TEST(RuntimeStatus, OrchAndSchedRangesDoNotOverlap) {
    // Orchestrator codes occupy 1..99; scheduler codes occupy 100+.
    EXPECT_LT(PTO2_ERROR_EXPLICIT_ORCH_FATAL, 100);
    EXPECT_GE(PTO2_ERROR_SCHEDULER_TIMEOUT, 100);
}
