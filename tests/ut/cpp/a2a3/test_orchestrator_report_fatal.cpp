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
 * UT for the orchestrator-side fatal reporting path.
 *
 * Targets orch_.report_fatal (pto_orchestrator.cpp) and verifies:
 *  - orch->fatal latches to true on any non-zero error code
 *  - the first non-zero code wins via CAS into sm_header->orch_error_code
 *  - subsequent fatal reports do NOT overwrite the first code
 *  - PTO2_ERROR_NONE never latches the shared-memory code (but still flips
 *    the local fatal flag -- by design, callers may use it to mark fatal
 *    without writing a code)
 *
 * This test exercises the real symbol against a fully-initialized
 * orchestrator + shared memory pair, complementing the fake-runtime test
 * (test_a2a3_pto2_fatal.cpp) that only validates the ops-table dispatch.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "pto_orchestrator.h"
#include "pto_runtime_status.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"

namespace {

constexpr uint64_t kHeapSize = 64 * 1024;
constexpr int32_t kWindowSize = 64;
constexpr int32_t kDepPool = 256;

class OrchestratorFatalTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle *sm_ = nullptr;
    PTO2SchedulerState sched_{};
    PTO2OrchestratorState orch_{};
    uint8_t *gm_heap_ = nullptr;
    bool sched_ok_ = false;
    bool orch_ok_ = false;

    void SetUp() override {
        sm_ = PTO2SharedMemoryHandle::create(kWindowSize, kHeapSize);
        ASSERT_NE(sm_, nullptr);

        gm_heap_ = static_cast<uint8_t *>(std::calloc(PTO2_MAX_RING_DEPTH, kHeapSize));
        ASSERT_NE(gm_heap_, nullptr);

        sched_ok_ = sched_.init(sm_->header, kDepPool);
        ASSERT_TRUE(sched_ok_);

        orch_ok_ = orch_.init(sm_->header, gm_heap_, kHeapSize, kDepPool);
        ASSERT_TRUE(orch_ok_);
    }

    void TearDown() override {
        if (orch_ok_) orch_.destroy();
        if (sched_ok_) sched_.destroy();
        if (gm_heap_) std::free(gm_heap_);
        if (sm_) sm_->destroy();
    }

    int32_t shared_orch_code() const { return sm_->header->orch_error_code.load(std::memory_order_acquire); }
};

}  // namespace

// ---------- baseline ----------

TEST_F(OrchestratorFatalTest, InitialState_NoFatalNoSharedCode) {
    // Verify no fatal state via the observable shared memory output
    EXPECT_FALSE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_NONE);
}

// ---------- happy path: single fatal latches both local flag and shared code ----------

TEST_F(OrchestratorFatalTest, ReportFatal_SetsLocalFlagAndSharedCode) {
    orch_.report_fatal(PTO2_ERROR_HEAP_RING_DEADLOCK, "test", "deadlock at ring %d", 3);

    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

// ---------- CAS first-writer-wins ----------

TEST_F(OrchestratorFatalTest, SecondReportFatal_DoesNotOverwriteSharedCode) {
    orch_.report_fatal(PTO2_ERROR_HEAP_RING_DEADLOCK, "test", nullptr);
    orch_.report_fatal(PTO2_ERROR_DEP_POOL_OVERFLOW, "test", nullptr);

    // Second report must NOT overwrite the first latched code.
    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(OrchestratorFatalTest, RepeatedSameCode_StaysLatched) {
    orch_.report_fatal(PTO2_ERROR_INVALID_ARGS, "test", nullptr);
    orch_.report_fatal(PTO2_ERROR_INVALID_ARGS, "test", nullptr);

    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_INVALID_ARGS);
}

// ---------- PTO2_ERROR_NONE: marks fatal locally, does NOT touch shared code ----------

TEST_F(OrchestratorFatalTest, ReportFatalWithErrorNone_DoesNotWriteSharedCode) {
    orch_.report_fatal(PTO2_ERROR_NONE, "test", nullptr);

    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_NONE);
}

// ---------- PTO2_ERROR_NONE first does not block a real code from latching ----------

TEST_F(OrchestratorFatalTest, ErrorNoneFirst_RealCodeStillLatchesAfter) {
    orch_.report_fatal(PTO2_ERROR_NONE, "test", nullptr);
    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_NONE);

    orch_.report_fatal(PTO2_ERROR_SCOPE_DEADLOCK, "test", nullptr);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_SCOPE_DEADLOCK);
}

// ---------- coverage of every defined orchestrator code ----------

TEST_F(OrchestratorFatalTest, EveryOrchCode_LatchesIntoSharedMemory) {
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
        // Reset latches between iterations. Direct field access is unavoidable here
        // since there is no public reset API for the orchestrator fatal state.
        sm_->header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_release);
        orch_.fatal = false;

        orch_.report_fatal(code, "test", "code=%d", code);

        SCOPED_TRACE(testing::Message() << "code=" << code);
        EXPECT_TRUE(orch_.fatal);
        EXPECT_EQ(shared_orch_code(), code);
    }
}

// ---------- format-string variants must not crash ----------

TEST_F(OrchestratorFatalTest, NullFmt_DoesNotCrash) {
    orch_.report_fatal(PTO2_ERROR_INVALID_ARGS, "func", nullptr);
    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_INVALID_ARGS);
}

TEST_F(OrchestratorFatalTest, EmptyFmt_DoesNotCrash) {
    orch_.report_fatal(PTO2_ERROR_INVALID_ARGS, "func", "");
    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_INVALID_ARGS);
}

TEST_F(OrchestratorFatalTest, FmtWithVarArgs_DoesNotCrash) {
    orch_.report_fatal(
        PTO2_ERROR_TENSOR_WAIT_TIMEOUT, "func", "tensor=%p slot=%d msg=%s", reinterpret_cast<void *>(0xdeadbeef), 17,
        "boom"
    );
    EXPECT_TRUE(orch_.fatal);
    EXPECT_EQ(shared_orch_code(), PTO2_ERROR_TENSOR_WAIT_TIMEOUT);
}

// ---------- end-to-end: status helper sees latched code ----------

TEST_F(OrchestratorFatalTest, StatusHelperReadsLatchedOrchCode) {
    orch_.report_fatal(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, "func", nullptr);

    int32_t orch_code = shared_orch_code();
    int32_t sched_code = sm_->header->sched_error_code.load(std::memory_order_acquire);
    EXPECT_EQ(runtime_status_from_error_codes(orch_code, sched_code), -PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
}
