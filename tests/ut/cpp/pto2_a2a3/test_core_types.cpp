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
 * Unit tests for core types in pto_runtime2_types.h
 *
 * Tests PTO2TaskId encoding, alignment assertions, utility macros, and
 * task-state machine transitions / subtask-completion bitmask semantics.
 */

#include <gtest/gtest.h>

#include <atomic>

#include "pto_runtime2_types.h"

// =============================================================================
// PTO2TaskId encoding/extraction
// =============================================================================

TEST(TaskId, DefaultIsZero) {
    PTO2TaskId id{};
    EXPECT_EQ(id.raw, 0u);
    EXPECT_EQ(id.ring(), 0);
    EXPECT_EQ(id.local(), 0u);
}

TEST(TaskId, MakeAndExtract) {
    auto id = PTO2TaskId::make(2, 42);
    EXPECT_EQ(id.ring(), 2);
    EXPECT_EQ(id.local(), 42u);
}

TEST(TaskId, RingInUpperBits) {
    auto id = PTO2TaskId::make(3, 0);
    EXPECT_EQ(id.raw, static_cast<uint64_t>(3) << 32);
    EXPECT_EQ(id.ring(), 3);
    EXPECT_EQ(id.local(), 0u);
}

TEST(TaskId, MaxRingMaxLocal) {
    auto id = PTO2TaskId::make(255, 0xFFFFFFFF);
    EXPECT_EQ(id.ring(), 255);
    EXPECT_EQ(id.local(), 0xFFFFFFFF);
}

TEST(TaskId, Roundtrip) {
    for (uint8_t ring = 0; ring < PTO2_MAX_RING_DEPTH; ring++) {
        for (uint32_t local : {0u, 1u, 100u, 0xFFFFu, 0xFFFFFFFFu}) {
            auto id = PTO2TaskId::make(ring, local);
            EXPECT_EQ(id.ring(), ring);
            EXPECT_EQ(id.local(), local);
        }
    }
}

TEST(TaskId, Equality) {
    auto a = PTO2TaskId::make(1, 42);
    auto b = PTO2TaskId::make(1, 42);
    auto c = PTO2TaskId::make(1, 43);
    auto d = PTO2TaskId::make(2, 42);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a != d);
}

TEST(TaskId, SizeIs8Bytes) { EXPECT_EQ(sizeof(PTO2TaskId), 8u); }

// =============================================================================
// PTO2TaskSlotState size (cache-line aligned)
// =============================================================================

// ABI contract: size must match shared memory layout (cache-line aligned).
TEST(TaskSlotState, SizeIs64Bytes) { EXPECT_EQ(sizeof(PTO2TaskSlotState), 64u); }

// =============================================================================
// PTO2_ALIGN_UP macro
// =============================================================================

TEST(AlignUp, Zero) { EXPECT_EQ(PTO2_ALIGN_UP(0, 64), 0u); }

TEST(AlignUp, AlreadyAligned) {
    EXPECT_EQ(PTO2_ALIGN_UP(64, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(128, 64), 128u);
}

TEST(AlignUp, NotAligned) {
    EXPECT_EQ(PTO2_ALIGN_UP(1, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(63, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(65, 64), 128u);
}

TEST(AlignUp, SmallAlignment) {
    EXPECT_EQ(PTO2_ALIGN_UP(5, 4), 8u);
    EXPECT_EQ(PTO2_ALIGN_UP(4, 4), 4u);
    EXPECT_EQ(PTO2_ALIGN_UP(3, 4), 4u);
}

// =============================================================================
// Task state enum values
// =============================================================================

// ABI contract: values must match wire protocol / shared memory layout.
TEST(TaskState, EnumValues) {
    EXPECT_EQ(PTO2_TASK_PENDING, 0);
    EXPECT_EQ(PTO2_TASK_READY, 1);
    EXPECT_EQ(PTO2_TASK_RUNNING, 2);
    EXPECT_EQ(PTO2_TASK_COMPLETED, 3);
    EXPECT_EQ(PTO2_TASK_CONSUMED, 4);
}

// =============================================================================
// Error code constants
// =============================================================================

// ABI contract: values must match wire protocol / shared memory layout.
TEST(ErrorCodes, Values) {
    EXPECT_EQ(PTO2_ERROR_NONE, 0);
    EXPECT_EQ(PTO2_ERROR_SCOPE_DEADLOCK, 1);
    EXPECT_EQ(PTO2_ERROR_HEAP_RING_DEADLOCK, 2);
    EXPECT_EQ(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, 3);
    EXPECT_EQ(PTO2_ERROR_DEP_POOL_OVERFLOW, 4);
    EXPECT_EQ(PTO2_ERROR_INVALID_ARGS, 5);
    EXPECT_EQ(PTO2_ERROR_SCHEDULER_TIMEOUT, 100);
}

// =============================================================================
// Configuration constants
// =============================================================================

TEST(Config, TaskWindowSizeIsPowerOf2) {
    EXPECT_GT(PTO2_TASK_WINDOW_SIZE, 0);
    EXPECT_EQ(PTO2_TASK_WINDOW_SIZE & (PTO2_TASK_WINDOW_SIZE - 1), 0);
}

TEST(Config, MaxRingDepth) { EXPECT_EQ(PTO2_MAX_RING_DEPTH, 4); }

TEST(Config, AlignSize) { EXPECT_EQ(PTO2_ALIGN_SIZE, 64); }

// =============================================================================
// Task state machine: valid transitions PENDING -> READY -> RUNNING ->
// COMPLETED -> CONSUMED
// =============================================================================

TEST(TaskStateTest, ValidTransitions) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING);

    PTO2TaskState expected = PTO2_TASK_PENDING;
    bool ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_READY);

    expected = PTO2_TASK_READY;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_RUNNING);

    expected = PTO2_TASK_RUNNING;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_COMPLETED);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_COMPLETED);

    expected = PTO2_TASK_COMPLETED;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// Invalid transition: PENDING -> RUNNING (must go through READY)
TEST(TaskStateTest, InvalidTransition_PendingToRunning) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING);

    PTO2TaskState expected = PTO2_TASK_READY;
    bool ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING);
    EXPECT_FALSE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING);
}

// Subtask completion bitmask
TEST(TaskStateTest, SubtaskCompletion) {
    PTO2TaskSlotState slot{};
    slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    slot.subtask_done_mask.store(0);

    uint8_t prev = slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIC);
    EXPECT_EQ(prev, 0u);
    EXPECT_NE(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);

    slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIV0);
    EXPECT_NE(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);

    slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIV1);
    EXPECT_EQ(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);
}

// Fanin/fanout refcount correctness
TEST(TaskStateTest, FaninRefcount) {
    PTO2TaskSlotState slot{};
    slot.fanin_count = 3;
    slot.fanin_refcount.store(0);

    for (int i = 0; i < 3; i++) {
        slot.fanin_refcount.fetch_add(1);
    }

    EXPECT_EQ(slot.fanin_refcount.load(), slot.fanin_count);
}

TEST(TaskStateTest, FanoutRefcount) {
    PTO2TaskSlotState slot{};
    slot.fanout_count = 5;
    slot.fanout_refcount.store(0);

    for (int i = 0; i < 5; i++) {
        slot.fanout_refcount.fetch_add(1);
    }

    EXPECT_EQ(slot.fanout_refcount.load(), slot.fanout_count);
}
