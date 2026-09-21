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

#include <new>
#include <type_traits>
#include <utility>

#include "common/support/native_run_execution_peer.h"

namespace {

constexpr NativeRunIdentity kIdentity{17, 23, 29, 1};

static_assert(!std::is_copy_constructible_v<LaunchPermit>);
static_assert(!std::is_copy_assignable_v<LaunchPermit>);
static_assert(std::is_nothrow_move_constructible_v<LaunchPermit>);
static_assert(!std::is_copy_constructible_v<LaunchReceipt>);
static_assert(std::is_nothrow_move_constructible_v<LaunchReceipt>);

TEST(NativeRunExecutionTest, IdentityMismatchDoesNotSubmit) {
    NativeRunIdentity other = kIdentity;
    other.run_epoch++;
    int submissions = 0;

    LaunchTransactionResult result = exact_launch_transaction(
        other, NativeRunExecutionTestPeer::mint(kIdentity),
        [&](LaunchProgressSink &) {
            ++submissions;
            return 0;
        },
        [&](LaunchProgressSink &) {
            ++submissions;
            return 0;
        }
    );

    EXPECT_EQ(result.rc, -1);
    EXPECT_EQ(result.progress, LaunchProgress::NotStarted);
    EXPECT_EQ(submissions, 0);
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, AicoreFailureIsSafePrelaunchFailure) {
    int aicpu_submissions = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [](LaunchProgressSink &) {
            return 41;
        },
        [&](LaunchProgressSink &) {
            ++aicpu_submissions;
            return 0;
        }
    );

    EXPECT_EQ(result.rc, 41);
    EXPECT_EQ(result.progress, LaunchProgress::NotStarted);
    EXPECT_EQ(aicpu_submissions, 0);
    EXPECT_FALSE(result.poisoned());
}

TEST(NativeRunExecutionTest, ArmingFailureReportedAsRcStaysSafe) {
    int aicpu_submissions = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [&](LaunchProgressSink &) -> int {
            // The shape every real submit callback uses: the arming prologue
            // catches its own throws and reports them as an rc, so the failure
            // stays on the safe side of the first submission.
            try {
                throw std::bad_alloc();
            } catch (...) {
                return -1;
            }
        },
        [&](LaunchProgressSink &) {
            ++aicpu_submissions;
            return 0;
        }
    );

    EXPECT_EQ(result.rc, -1);
    EXPECT_EQ(result.progress, LaunchProgress::NotStarted);
    EXPECT_FALSE(result.poisoned());
    EXPECT_EQ(aicpu_submissions, 0);
}

// The record-after-launch shape: the kernel is already on the device when the
// step after it fails. Grading that NotStarted would hand the caller a run it
// believes never launched, and let it roll back resources the device holds.
TEST(NativeRunExecutionTest, AicoreFailureAfterMarkedSubmissionPoisons) {
    int aicpu_submissions = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [](LaunchProgressSink &sink) {
            sink.mark_submitted();
            return 41;
        },
        [&](LaunchProgressSink &) {
            ++aicpu_submissions;
            return 0;
        }
    );

    EXPECT_EQ(result.rc, 41);
    EXPECT_EQ(result.progress, LaunchProgress::Partial);
    EXPECT_TRUE(result.poisoned());
    EXPECT_EQ(aicpu_submissions, 0);
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, AicpuFailureAfterMarkedSubmissionStaysPartial) {
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [](LaunchProgressSink &sink) {
            sink.mark_submitted();
            return 0;
        },
        [](LaunchProgressSink &sink) {
            sink.mark_submitted();
            return 43;
        }
    );

    EXPECT_EQ(result.rc, 43);
    EXPECT_EQ(result.progress, LaunchProgress::Partial);
    EXPECT_TRUE(result.poisoned());
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, AicoreExceptionPoisonsWithoutTryingAicpu) {
    int aicpu_submissions = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [](LaunchProgressSink &) -> int {
            throw 41;
        },
        [&](LaunchProgressSink &) {
            ++aicpu_submissions;
            return 0;
        }
    );

    EXPECT_EQ(result.rc, -1);
    EXPECT_EQ(result.progress, LaunchProgress::Partial);
    EXPECT_EQ(aicpu_submissions, 0);
    EXPECT_TRUE(result.poisoned());
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, AicpuFailureAfterAicorePoisonsWithoutReceipt) {
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [](LaunchProgressSink &) {
            return 0;
        },
        [](LaunchProgressSink &) {
            return 43;
        }
    );

    EXPECT_EQ(result.rc, 43);
    EXPECT_EQ(result.progress, LaunchProgress::Partial);
    EXPECT_TRUE(result.poisoned());
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, AicpuExceptionAfterAicorePoisonsWithoutReceipt) {
    int aicore_submissions = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [&](LaunchProgressSink &) {
            ++aicore_submissions;
            return 0;
        },
        [](LaunchProgressSink &) -> int {
            throw 43;
        }
    );

    EXPECT_EQ(result.rc, -1);
    EXPECT_EQ(result.progress, LaunchProgress::Partial);
    EXPECT_EQ(aicore_submissions, 1);
    EXPECT_TRUE(result.poisoned());
    EXPECT_FALSE(result.receipt.valid());
}

TEST(NativeRunExecutionTest, BothSubmissionsProduceIdentityBoundReceipt) {
    int order = 0;
    LaunchTransactionResult result = exact_launch_transaction(
        kIdentity, NativeRunExecutionTestPeer::mint(kIdentity),
        [&](LaunchProgressSink &sink) {
            EXPECT_EQ(order++, 0);
            sink.mark_submitted();
            return 0;
        },
        [&](LaunchProgressSink &sink) {
            EXPECT_EQ(order++, 1);
            sink.mark_submitted();
            return 0;
        }
    );

    EXPECT_EQ(result.rc, 0);
    EXPECT_EQ(result.progress, LaunchProgress::Complete);
    EXPECT_EQ(order, 2);
    EXPECT_TRUE(result.receipt.matches(kIdentity));

    NativeRunIdentity stale = kIdentity;
    stale.generation++;
    EXPECT_FALSE(result.receipt.matches(stale));
}

TEST(NativeRunExecutionTest, PermitIsOneShot) {
    LaunchPermit permit = NativeRunExecutionTestPeer::mint(kIdentity);
    LaunchPermit first = std::move(permit);
    ASSERT_TRUE(first.valid());
    EXPECT_FALSE(permit.valid());

    LaunchTransactionResult success = exact_launch_transaction(
        kIdentity, std::move(first),
        [](LaunchProgressSink &) {
            return 0;
        },
        [](LaunchProgressSink &) {
            return 0;
        }
    );
    EXPECT_EQ(success.progress, LaunchProgress::Complete);
    EXPECT_FALSE(first.valid());
}

}  // namespace
