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

#include "host/workspace_staging.h"

namespace {

// The object `DeviceRunnerBase::stage_workspace_management`,
// `clear_staged_workspace` and `install_staged_workspace` are: recording a
// request before the execution mode is latched, and consuming it once
// afterwards. Exercised directly because the runner that owns it needs a
// device context to construct.

constexpr uint64_t kLimit = 4u << 20;

TEST(WorkspaceStaging, AContextThatAsksForNothingInstallsNothing) {
    WorkspaceStagingRequest staging;
    EXPECT_FALSE(staging.requested());
    EXPECT_EQ(staging.plan(), WorkspaceStagingRequest::Install::Nothing);
    EXPECT_EQ(staging.limit_bytes(), 0u);
}

TEST(WorkspaceStaging, ManagementAloneAndManagementWithALimitAreDifferentPlans) {
    WorkspaceStagingRequest manage_only;
    ASSERT_EQ(manage_only.record(0, /*installed=*/false), 0);
    EXPECT_EQ(manage_only.plan(), WorkspaceStagingRequest::Install::ManageOnly);
    EXPECT_EQ(manage_only.limit_bytes(), 0u);

    WorkspaceStagingRequest with_limit;
    ASSERT_EQ(with_limit.record(kLimit, /*installed=*/false), 0);
    EXPECT_EQ(with_limit.plan(), WorkspaceStagingRequest::Install::ManageWithLimit);
    // The install reads exactly what was asked for, so the prewarm that
    // follows it allocates under that limit and not under a default.
    EXPECT_EQ(with_limit.limit_bytes(), kLimit);
}

TEST(WorkspaceStaging, TheFirstAcceptedRequestStandsAgainstASecondOne) {
    WorkspaceStagingRequest staging;
    ASSERT_EQ(staging.record(kLimit, /*installed=*/false), 0);

    // Deferring the install must not turn "once" into a silent
    // reconfiguration: what this context enforces cannot depend on call order.
    EXPECT_EQ(staging.record(kLimit * 2, /*installed=*/false), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(staging.record(0, /*installed=*/false), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(staging.limit_bytes(), kLimit);
    EXPECT_EQ(staging.plan(), WorkspaceStagingRequest::Install::ManageWithLimit);
}

TEST(WorkspaceStaging, ARequestIsRefusedOnceManagementIsLive) {
    WorkspaceStagingRequest staging;
    EXPECT_EQ(staging.record(kLimit, /*installed=*/true), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(staging.requested());
}

TEST(WorkspaceStaging, AnInitThatRolledBackLeavesNothingStagedForTheNextOne) {
    WorkspaceStagingRequest staging;
    ASSERT_EQ(staging.record(kLimit, /*installed=*/false), 0);
    // What the rollback path does after destroying the device context: the
    // request goes with the generation that made it, so a retried init states
    // its own rather than inheriting one.
    staging.clear();
    EXPECT_FALSE(staging.requested());
    EXPECT_EQ(staging.plan(), WorkspaceStagingRequest::Install::Nothing);
    EXPECT_EQ(staging.limit_bytes(), 0u);

    ASSERT_EQ(staging.record(0, /*installed=*/false), 0);
    EXPECT_EQ(staging.plan(), WorkspaceStagingRequest::Install::ManageOnly);
}

}  // namespace
