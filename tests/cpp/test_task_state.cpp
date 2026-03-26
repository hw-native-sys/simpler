/**
 * Unit tests for PTO2 Task State Machine.
 *
 * Tests valid state transitions and subtask completion bitmask.
 */

#include <gtest/gtest.h>
#include <atomic>
#include "pto_runtime2_types.h"

// =============================================================================
// Valid transitions: PENDING → READY → RUNNING → COMPLETED → CONSUMED
// =============================================================================

TEST(TaskStateTest, ValidTransitions) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING);

    // PENDING → READY
    PTO2TaskState expected = PTO2_TASK_PENDING;
    bool ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_READY);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_READY);

    // READY → RUNNING
    expected = PTO2_TASK_READY;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_RUNNING);

    // RUNNING → COMPLETED
    expected = PTO2_TASK_RUNNING;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_COMPLETED);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_COMPLETED);

    // COMPLETED → CONSUMED
    expected = PTO2_TASK_COMPLETED;
    ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED);
    EXPECT_TRUE(ok);
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_CONSUMED);
}

// =============================================================================
// Invalid transition: PENDING → RUNNING (must go through READY)
// =============================================================================

TEST(TaskStateTest, InvalidTransition_PendingToRunning) {
    PTO2TaskSlotState slot{};
    slot.task_state.store(PTO2_TASK_PENDING);

    // Attempt PENDING → RUNNING should fail (CAS expects READY)
    PTO2TaskState expected = PTO2_TASK_READY;
    bool ok = slot.task_state.compare_exchange_strong(expected, PTO2_TASK_RUNNING);
    EXPECT_FALSE(ok);
    // State should remain PENDING
    EXPECT_EQ(slot.task_state.load(), PTO2_TASK_PENDING);
}

// =============================================================================
// Subtask completion bitmask
// =============================================================================

TEST(TaskStateTest, SubtaskCompletion) {
    PTO2TaskSlotState slot{};
    // Mixed task with all 3 subtask slots: AIC + AIV0 + AIV1
    slot.active_mask = PTO2_SUBTASK_MASK_AIC | PTO2_SUBTASK_MASK_AIV0 | PTO2_SUBTASK_MASK_AIV1;
    slot.subtask_done_mask.store(0);

    // AIC completes
    uint8_t prev = slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIC);
    EXPECT_EQ(prev, 0u);
    EXPECT_NE(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);

    // AIV0 completes
    slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIV0);
    EXPECT_NE(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);

    // AIV1 completes — now all done
    slot.subtask_done_mask.fetch_or(PTO2_SUBTASK_MASK_AIV1);
    EXPECT_EQ(slot.subtask_done_mask.load() & slot.active_mask, slot.active_mask);
}

// =============================================================================
// Fanin/fanout refcount correctness
// =============================================================================

TEST(TaskStateTest, FaninRefcount) {
    PTO2TaskSlotState slot{};
    slot.fanin_count = 3;
    slot.fanin_refcount.store(0);

    // Simulate 3 producers completing
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
