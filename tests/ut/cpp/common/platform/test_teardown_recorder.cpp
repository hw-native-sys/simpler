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
 * The teardown recorder's two scopes.
 *
 * Every case here replays a real shape of `DeviceRunner::finalize()`: the
 * normal reset arms, which have no probe; the fatal recovery loop, whose
 * wrapper return is a different value from the reset call's; and the returns
 * that never reach a reset arm at all. What is under test is that the record
 * keeps those apart, in particular that a last attempt which stopped in its
 * preamble does not erase an earlier attempt's reset call.
 */

#include <gtest/gtest.h>

#include "host/teardown_recorder.h"

namespace {

bool has_flag(const SimplerTeardownReport &report, uint32_t flag) {
    return (report.flags & static_cast<uint8_t>(flag)) != 0;
}

TEST(TeardownRecorderTest, NothingRecordedIsNotAnEmptyTeardown) {
    TeardownRecorder recorder;
    SimplerTeardownReport report{};
    EXPECT_FALSE(recorder.copy_to(&report));

    // Begun but not finished is still nothing: a teardown in flight has no
    // return value yet, so there is no record to hand out.
    recorder.begin(TEARDOWN_PATH_NORMAL, 3);
    EXPECT_FALSE(recorder.copy_to(&report));
}

TEST(TeardownRecorderTest, NormalResetSucceededWithNoProbe) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 3);
    recorder.note_reset_api(TEARDOWN_RESET_API_RT_DEVICE_RESET, 0);
    recorder.note_stage(TEARDOWN_STAGE_API_OK_NO_PROBE);
    recorder.finish(0);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.schema, TEARDOWN_REPORT_SCHEMA);
    EXPECT_EQ(report.path, TEARDOWN_PATH_NORMAL);
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_API_OK_NO_PROBE);
    EXPECT_EQ(report.reset_api, TEARDOWN_RESET_API_RT_DEVICE_RESET);
    EXPECT_TRUE(has_flag(report, TEARDOWN_FLAG_LAST_RESET_API_RC_VALID));
    EXPECT_EQ(report.last_reset_api_rc, 0);
    EXPECT_EQ(report.reset_api_invocations_total, 1);
    // No recovery wrapper ran, so its return value is absent rather than zero:
    // a zero here would read as a confirmation nobody obtained.
    EXPECT_FALSE(has_flag(report, TEARDOWN_FLAG_RECOVERY_SEQUENCE_RC_VALID));
    EXPECT_FALSE(has_flag(report, TEARDOWN_FLAG_PROBE_CONFIRMED));
    EXPECT_EQ(report.recovery_attempts_total, 0);
    EXPECT_EQ(report.teardown_rc, 0);
}

TEST(TeardownRecorderTest, NormalResetFailedIsNotAnOkStage) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 3);
    recorder.note_reset_api(TEARDOWN_RESET_API_ACL_RESET_DEVICE, 507899);
    recorder.note_stage(TEARDOWN_STAGE_API_FAILED);
    recorder.finish(507899);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_API_FAILED);
    EXPECT_EQ(report.reset_api, TEARDOWN_RESET_API_ACL_RESET_DEVICE);
    EXPECT_EQ(report.last_reset_api_rc, 507899);
    EXPECT_EQ(report.teardown_rc, 507899);
}

TEST(TeardownRecorderTest, EarlyReturnRecordsNoResetCall) {
    // The attach-failure return: finalize() leaves before any reset arm.
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 3);
    recorder.finish(-1001);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_NOT_ATTEMPTED);
    EXPECT_EQ(report.reset_api, TEARDOWN_RESET_API_NONE);
    EXPECT_EQ(report.reset_api_invocations_total, 0);
    EXPECT_FALSE(has_flag(report, TEARDOWN_FLAG_LAST_RESET_API_RC_VALID));
    EXPECT_EQ(report.teardown_rc, -1001);
}

TEST(TeardownRecorderTest, KernelContextRefusesWithoutAttempting) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_FATAL, 3);
    recorder.note_stage(TEARDOWN_STAGE_REFUSED);
    recorder.finish(-1003);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.path, TEARDOWN_PATH_FATAL);
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_REFUSED);
    EXPECT_EQ(report.recovery_attempts_total, 0);
    EXPECT_EQ(report.reset_api_invocations_total, 0);
}

TEST(TeardownRecorderTest, ForceResetSucceededButProbeDidNot) {
    // aclrtResetDeviceForce returns 0 and the post-reset probe then fails, so
    // the recovery wrapper returns the probe's code. Both must survive.
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_FATAL, 3);
    recorder.note_recovery_attempt();
    recorder.note_reset_api(TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE, 0);
    recorder.note_stage(TEARDOWN_STAGE_API_OK_PROBE_RUN);
    recorder.note_recovery_sequence(507899);
    recorder.finish(507899);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_API_OK_PROBE_RUN);
    EXPECT_TRUE(has_flag(report, TEARDOWN_FLAG_LAST_RESET_API_RC_VALID));
    EXPECT_EQ(report.last_reset_api_rc, 0);
    EXPECT_TRUE(has_flag(report, TEARDOWN_FLAG_RECOVERY_SEQUENCE_RC_VALID));
    EXPECT_EQ(report.recovery_sequence_rc, 507899);
    EXPECT_FALSE(has_flag(report, TEARDOWN_FLAG_PROBE_CONFIRMED));
}

TEST(TeardownRecorderTest, ConfirmedProbeIsTheOnlyProbeBackedClaim) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_FATAL, 3);
    recorder.note_recovery_attempt();
    recorder.note_reset_api(TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE, 0);
    recorder.note_stage(TEARDOWN_STAGE_API_OK_PROBE_RUN);
    recorder.note_probe_confirmed();
    recorder.note_recovery_sequence(0);
    recorder.finish(0);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_TRUE(has_flag(report, TEARDOWN_FLAG_PROBE_CONFIRMED));
    EXPECT_EQ(report.recovery_sequence_rc, 0);
    EXPECT_EQ(report.recovery_attempts_total, 1);
}

TEST(TeardownRecorderTest, PreambleFailureAfterAnEarlierApiCallKeepsThatCall) {
    // Attempt 1 binds and calls the force reset, which fails; attempt 2 cannot
    // even bind. `attempt_fatal_reset` returns the LAST attempt's error, so the
    // stage describes attempt 2 while the counters and the last API rc still
    // report attempt 1's call.
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_FATAL, 3);

    recorder.note_recovery_attempt();
    recorder.note_reset_api(TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE, 507899);
    recorder.note_stage(TEARDOWN_STAGE_API_FAILED);

    recorder.note_recovery_attempt();
    recorder.note_stage(TEARDOWN_STAGE_PREAMBLE_FAILED);

    recorder.note_recovery_sequence(-1001);
    recorder.finish(-1001);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_PREAMBLE_FAILED);
    // The whole point: a preamble failure last does not claim that no reset
    // API ever ran.
    EXPECT_EQ(report.reset_api_invocations_total, 1);
    EXPECT_TRUE(has_flag(report, TEARDOWN_FLAG_LAST_RESET_API_RC_VALID));
    EXPECT_EQ(report.last_reset_api_rc, 507899);
    EXPECT_EQ(report.reset_api, TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE);
    EXPECT_EQ(report.recovery_attempts_total, 2);
    EXPECT_EQ(report.recovery_sequence_rc, -1001);
}

TEST(TeardownRecorderTest, InvocationsCountEveryAttemptThatReachedTheApi) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_FATAL, 3);
    for (int attempt = 0; attempt < 3; ++attempt) {
        recorder.note_recovery_attempt();
        recorder.note_reset_api(TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE, 507899);
        recorder.note_stage(TEARDOWN_STAGE_API_FAILED);
    }
    recorder.note_recovery_sequence(507899);
    recorder.finish(507899);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.recovery_attempts_total, 3);
    EXPECT_EQ(report.reset_api_invocations_total, 3);
}

TEST(TeardownRecorderTest, FirstFinishWins) {
    // finalize() has several returns; the earliest one reached is the one that
    // describes this teardown, and a later begin/finish pair — an explicit
    // close retry — must not replace it at the runner either.
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 3);
    recorder.note_reset_api(TEARDOWN_RESET_API_RT_DEVICE_RESET, 5);
    recorder.note_stage(TEARDOWN_STAGE_API_FAILED);
    recorder.finish(5);
    recorder.finish(0);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.teardown_rc, 5);
    EXPECT_EQ(report.reset_stage, TEARDOWN_STAGE_API_FAILED);
}

TEST(TeardownRecorderTest, ChildPidIsLeftToTheTransport) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 7);
    recorder.finish(0);

    SimplerTeardownReport report{};
    ASSERT_TRUE(recorder.copy_to(&report));
    EXPECT_EQ(report.device_id, 7);
    // The runner observes no pid; the publishing process stamps it as the
    // record's generation key.
    EXPECT_EQ(report.child_pid, 0);
    EXPECT_EQ(report.reserved, 0u);
}

TEST(TeardownRecorderTest, NotesBeforeBeginAreDropped) {
    TeardownRecorder recorder;
    recorder.note_reset_api(TEARDOWN_RESET_API_RT_DEVICE_RESET, 7);
    recorder.note_stage(TEARDOWN_STAGE_API_FAILED);
    recorder.finish(7);

    SimplerTeardownReport report{};
    EXPECT_FALSE(recorder.copy_to(&report));
}

TEST(TeardownRecorderTest, CopyToRejectsANullDestination) {
    TeardownRecorder recorder;
    recorder.begin(TEARDOWN_PATH_NORMAL, 0);
    recorder.finish(0);
    EXPECT_FALSE(recorder.copy_to(nullptr));
}

}  // namespace
