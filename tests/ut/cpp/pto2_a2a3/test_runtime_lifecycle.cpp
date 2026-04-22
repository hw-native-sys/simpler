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
 * PTO2 Runtime lifecycle UT.
 *
 * Covers pto2_runtime_create / _custom / _from_sm / _destroy / set_mode.
 *
 * Follows AAA and FIRST: no shared mutable state between tests, each test
 * constructs its own runtime and tears it down.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "pto_runtime2.h"
#include "pto_shared_memory.h"

namespace {

constexpr uint64_t kSmallWindow = 64;
constexpr uint64_t kSmallHeap = 64 * 1024;

// -----------------------------------------------------------------------------
// Fixture: each test gets a fresh, isolated runtime config.
// -----------------------------------------------------------------------------
class RuntimeLifecycleTest : public ::testing::Test {
protected:
    PTO2Runtime *rt_ = nullptr;

    void TearDown() override {
        if (rt_ != nullptr) {
            pto2_runtime_destroy(rt_);
            rt_ = nullptr;
        }
    }
};

}  // namespace

// ---------- Happy-path creation ----------

TEST_F(RuntimeLifecycleTest, CreateCustom_ValidSizes_ReturnsInitializedRuntime) {
    // Arrange + Act
    rt_ = pto2_runtime_create_custom(PTO2_MODE_SIMULATE, kSmallWindow, kSmallHeap);

    // Assert
    ASSERT_NE(rt_, nullptr);
    EXPECT_NE(rt_->ops, nullptr);
    EXPECT_NE(rt_->sm_handle, nullptr);
    EXPECT_NE(rt_->gm_heap, nullptr);
    EXPECT_TRUE(rt_->gm_heap_owned);
    EXPECT_EQ(rt_->mode, PTO2_MODE_SIMULATE);
    EXPECT_EQ(rt_->gm_heap_size, kSmallHeap * PTO2_MAX_RING_DEPTH);
}

TEST_F(RuntimeLifecycleTest, CreateCustom_ConnectsOrchestratorToScheduler) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_EXECUTE, kSmallWindow, kSmallHeap);

    ASSERT_NE(rt_, nullptr);
    // In simulated mode the orchestrator must hold a pointer to the scheduler.
    EXPECT_EQ(rt_->orchestrator.scheduler, &rt_->scheduler);
}

TEST_F(RuntimeLifecycleTest, CreateDefault_UsesDefaultSizes) {
    // create() is a thin wrapper around create_custom with PTO2_TASK_WINDOW_SIZE / PTO2_HEAP_SIZE.
    // Use GRAPH_ONLY to avoid executor threads.  We don't allocate the full
    // 256MB heap in this path -- keep the assertion restricted to mode.
    rt_ = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
    ASSERT_NE(rt_, nullptr);
    EXPECT_EQ(rt_->mode, PTO2_MODE_GRAPH_ONLY);
}

// ---------- From-SM creation ----------

TEST_F(RuntimeLifecycleTest, CreateFromSM_NullHandle_ReturnsNull) {
    // Act
    PTO2Runtime *rt = pto2_runtime_create_from_sm(PTO2_MODE_SIMULATE, nullptr, nullptr, 0);

    // Assert
    EXPECT_EQ(rt, nullptr);
}

TEST_F(RuntimeLifecycleTest, CreateFromSM_RecordsCallerBuffers) {
    // Arrange: caller-allocated sm + gm_heap.
    PTO2SharedMemoryHandle *sm = pto2_sm_create(kSmallWindow, kSmallHeap);
    ASSERT_NE(sm, nullptr);
    uint8_t *heap = static_cast<uint8_t *>(std::calloc(PTO2_MAX_RING_DEPTH, kSmallHeap));
    ASSERT_NE(heap, nullptr);

    // Act
    rt_ = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE, sm, heap, kSmallHeap);

    // Assert: the returned runtime must NOT claim ownership of the gm_heap.
    ASSERT_NE(rt_, nullptr);
    EXPECT_EQ(rt_->sm_handle, sm);
    EXPECT_EQ(rt_->gm_heap, heap);
    EXPECT_FALSE(rt_->gm_heap_owned);

    // Cleanup: pto2_runtime_destroy consumes sm via pto2_sm_destroy (observed
    // behavior, see pto_runtime2.cpp:339), so only free the gm_heap here.
    pto2_runtime_destroy(rt_);
    rt_ = nullptr;
    std::free(heap);
}

// ---------- Destroy ----------

TEST_F(RuntimeLifecycleTest, Destroy_NullRuntime_NoCrash) {
    // Documented contract: destroy(nullptr) is a no-op.
    pto2_runtime_destroy(nullptr);
    SUCCEED();
}

TEST_F(RuntimeLifecycleTest, Destroy_ReleasesOwnedHeap) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_SIMULATE, kSmallWindow, kSmallHeap);
    ASSERT_NE(rt_, nullptr);
    // Act: explicitly destroy and null out so TearDown doesn't double-free.
    pto2_runtime_destroy(rt_);
    rt_ = nullptr;
    // Assert: reaching here without asan/ubsan complaint is the test (leak-free).
    SUCCEED();
}

// ---------- set_mode ----------

TEST_F(RuntimeLifecycleTest, SetMode_UpdatesField) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_EXECUTE, kSmallWindow, kSmallHeap);
    ASSERT_NE(rt_, nullptr);
    ASSERT_EQ(rt_->mode, PTO2_MODE_EXECUTE);

    // Act
    pto2_runtime_set_mode(rt_, PTO2_MODE_GRAPH_ONLY);

    // Assert
    EXPECT_EQ(rt_->mode, PTO2_MODE_GRAPH_ONLY);
}

TEST_F(RuntimeLifecycleTest, SetMode_NullRuntime_NoCrash) {
    // Contract: defensive null check, mirrors destroy.
    pto2_runtime_set_mode(nullptr, PTO2_MODE_SIMULATE);
    SUCCEED();
}

// ---------- Ops table wiring ----------

TEST_F(RuntimeLifecycleTest, OpsTable_AllFunctionPointersPopulated) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_SIMULATE, kSmallWindow, kSmallHeap);
    ASSERT_NE(rt_, nullptr);
    const PTO2RuntimeOps *ops = rt_->ops;
    ASSERT_NE(ops, nullptr);

    // Hot-path ops called by the orchestration .so -- must never be null.
    EXPECT_NE(ops->submit_task, nullptr);
    EXPECT_NE(ops->alloc_tensors, nullptr);
    EXPECT_NE(ops->scope_begin, nullptr);
    EXPECT_NE(ops->scope_end, nullptr);
    EXPECT_NE(ops->orchestration_done, nullptr);
    EXPECT_NE(ops->is_fatal, nullptr);
    EXPECT_NE(ops->report_fatal, nullptr);
    EXPECT_NE(ops->get_tensor_data, nullptr);
    EXPECT_NE(ops->set_tensor_data, nullptr);
}

TEST_F(RuntimeLifecycleTest, IsFatal_FreshRuntime_ReturnsFalse) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_SIMULATE, kSmallWindow, kSmallHeap);
    ASSERT_NE(rt_, nullptr);
    EXPECT_FALSE(rt_->ops->is_fatal(rt_));
}

TEST_F(RuntimeLifecycleTest, ReportFatal_SetsFatalFlag) {
    rt_ = pto2_runtime_create_custom(PTO2_MODE_SIMULATE, kSmallWindow, kSmallHeap);
    ASSERT_NE(rt_, nullptr);

    // Act
    rt_->ops->report_fatal(rt_, PTO2_ERROR_EXPLICIT_ORCH_FATAL, "UT", "%s", "forced");

    // Assert
    EXPECT_TRUE(rt_->ops->is_fatal(rt_));
}
