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
 * Orchestrator submit-path UT.
 *
 * Covers pto2_submit_mixed_task, pto2_alloc_tensors, pto2_orchestrator_done,
 * and pto2_orchestrator_set_scheduler on a fully initialized
 * (TMR) system.
 *
 * Follows AAA and FIRST: each TEST_F builds a fresh TMRSystem, exercises
 * one behavior, and tears the system down in TearDown().
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "pto_orchestration_api.h"  // make_tensor_external, TensorCreateInfo ctor
#include "pto_orchestrator.h"
#include "pto_ring_buffer.h"
#include "scheduler/pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_submit_types.h"
#include "pto_tensormap.h"
#include "tensor.h"

namespace {

constexpr uint64_t kHeapSize = 64 * 1024;
constexpr int32_t kWindowSize = 64;
constexpr int32_t kDepPool = 256;

// -----------------------------------------------------------------------------
// Fixture: minimal TMR system for orchestrator-level tests.
// -----------------------------------------------------------------------------
class OrchestratorSubmitTest : public ::testing::Test {
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

        orch_.set_scheduler(&sched_);
    }

    void TearDown() override {
        if (orch_ok_) orch_.destroy();
        if (sched_ok_) sched_.destroy();
        if (gm_heap_) std::free(gm_heap_);
        if (sm_) sm_->destroy();
    }

    // Helper: build a minimal TensorCreateInfo owning one FP32 scalar output.
    static TensorCreateInfo make_scalar_ci() {
        static const uint32_t kShape[1] = {1};
        return TensorCreateInfo(kShape, 1, DataType::FLOAT32);
    }

    bool has_orch_error() const {
        return sm_->header->orch_error_code.load(std::memory_order_acquire) != PTO2_ERROR_NONE;
    }
};

}  // namespace

// ---------- set_scheduler ----------

TEST_F(OrchestratorSubmitTest, SetScheduler_StoresPointer) {
    PTO2SchedulerState other{};
    orch_.set_scheduler(&other);
    // Direct field read: no public getter exists for the scheduler pointer.
    EXPECT_EQ(orch_.scheduler, &other);

    // Restore for TearDown.
    orch_.set_scheduler(&sched_);
}

// ---------- alloc_tensors: argument validation ----------

TEST_F(OrchestratorSubmitTest, AllocTensors_EmptyArgs_MarksFatal) {
    Arg args;  // no tensors, no scalars

    TaskOutputTensors result = orch_.alloc_tensors(args);

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_orch_error());
}

TEST_F(OrchestratorSubmitTest, AllocTensors_WithScalars_MarksFatal) {
    TensorCreateInfo ci = make_scalar_ci();
    Arg args;
    args.add_output(ci);
    args.add_scalar(uint64_t{42});

    TaskOutputTensors result = orch_.alloc_tensors(args);

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_orch_error());
}

TEST_F(OrchestratorSubmitTest, AllocTensors_InputArg_MarksFatal) {
    // alloc_tensors only accepts OUTPUT TensorCreateInfo args.
    uint32_t shape[1] = {1};
    Tensor input = make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    Arg args;
    args.add_input(input);

    TaskOutputTensors result = orch_.alloc_tensors(args);

    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_orch_error());
}

TEST_F(OrchestratorSubmitTest, AllocTensors_OutputOnly_ReturnsMaterializedTensors) {
    // Arrange: two output CIs, inside an active scope.
    TensorCreateInfo ci1 = make_scalar_ci();
    TensorCreateInfo ci2 = make_scalar_ci();
    Arg args;
    args.add_output(ci1, ci2);

    // Act
    orch_.begin_scope();
    TaskOutputTensors result = orch_.alloc_tensors(args);
    orch_.end_scope();

    // Assert
    EXPECT_FALSE(has_orch_error());
    EXPECT_EQ(result.size(), 2U);
}

TEST_F(OrchestratorSubmitTest, AllocTensors_AfterFatal_ReturnsEmpty) {
    // Arrange: force fatal.
    orch_.report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "UT", nullptr);
    ASSERT_TRUE(has_orch_error());

    TensorCreateInfo ci = make_scalar_ci();
    Arg args;
    args.add_output(ci);

    // Act
    TaskOutputTensors result = orch_.alloc_tensors(args);

    // Assert
    EXPECT_TRUE(result.empty());
}

// ---------- submit_mixed_task ----------

TEST_F(OrchestratorSubmitTest, SubmitMixedTask_AfterFatal_ReturnsEmpty) {
    // Arrange: pre-fatal state
    orch_.report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "UT", nullptr);

    MixedKernels mixed;
    mixed.aic_kernel_id = 0;
    Arg args;

    // Act
    TaskOutputTensors result = orch_.submit_task(mixed, args);

    // Assert
    EXPECT_TRUE(result.empty());
}

TEST_F(OrchestratorSubmitTest, SubmitMixedTask_ArgWithError_MarksFatalInvalidArgs) {
    // Arrange: craft an Arg with has_error set.
    // Calling add_input after add_scalar triggers the ordering error path.
    uint32_t shape[1] = {1};
    Tensor t = make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1);
    Arg args;
    args.add_scalar(uint64_t{1});
    args.add_input(t);  // illegal ordering -> has_error = true
    ASSERT_TRUE(args.has_error);

    MixedKernels mixed;
    mixed.aic_kernel_id = 0;

    // Act
    orch_.begin_scope();
    TaskOutputTensors result = orch_.submit_task(mixed, args);
    orch_.end_scope();

    // Assert
    EXPECT_TRUE(result.empty());
    EXPECT_TRUE(has_orch_error());
}

TEST_F(OrchestratorSubmitTest, SubmitMixedTask_PureInputOnly_Succeeds) {
    // Arrange: one input tensor, one AIC kernel, within a scope.
    uint32_t shape[1] = {1};
    Tensor input = make_tensor_external(reinterpret_cast<void *>(0x2000), shape, 1);

    Arg args;
    args.add_input(input);
    ASSERT_FALSE(args.has_error);

    MixedKernels mixed;
    mixed.aic_kernel_id = 7;  // any non-invalid id

    // Act
    orch_.begin_scope();
    TaskOutputTensors result = orch_.submit_task(mixed, args);
    orch_.end_scope();

    // Assert: submit returns (no outputs), and no fatal state was set.
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(has_orch_error());
}

TEST_F(OrchestratorSubmitTest, SubmitMixedTask_OutputTensor_MaterializesResult) {
    // Arrange: one OUTPUT TensorCreateInfo -> task produces one tensor.
    TensorCreateInfo ci = make_scalar_ci();
    Arg args;
    args.add_output(ci);

    MixedKernels mixed;
    mixed.aic_kernel_id = 1;

    // Act
    orch_.begin_scope();
    TaskOutputTensors result = orch_.submit_task(mixed, args);
    orch_.end_scope();

    // Assert
    EXPECT_FALSE(has_orch_error());
    EXPECT_EQ(result.size(), 1U);
}

// ---------- orchestrator_done ----------

TEST_F(OrchestratorSubmitTest, OrchestratorDone_SetsSharedMemoryFlag) {
    // Arrange
    ASSERT_EQ(sm_->header->orchestrator_done.load(), 0);

    // Act
    orch_.mark_done();

    // Assert
    EXPECT_EQ(sm_->header->orchestrator_done.load(std::memory_order_acquire), 1);
}

TEST_F(OrchestratorSubmitTest, OrchestratorDone_IsIdempotent) {
    orch_.mark_done();
    orch_.mark_done();

    // Flag stays 1 -- store is release-set, not increment.
    EXPECT_EQ(sm_->header->orchestrator_done.load(std::memory_order_acquire), 1);
}
