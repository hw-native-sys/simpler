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

#include <atomic>
#include <string>

#include "ring_buffer.h"
#include "tensormap_and_ringbuffer/task_id.h"

// CMake compiles this source against both the a2a3 and a5 runtime objects.
namespace {

constexpr int32_t WINDOW_SIZE = 16;
constexpr int32_t POOL_CAPACITY = 8;

void make_head_match_old_structural_predicate(
    ChipTaskSlotState &head, TaskDescriptor &descriptor, uint8_t ring_id, uint32_t local_task_id
) {
    descriptor.task_id = TaskId::make(ring_id, local_task_id);
    head.task = &descriptor;
    head.task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    head.fanout_count = FANOUT_SCOPE_BIT;
    head.fanout_refcount.store(0, std::memory_order_release);
}

}  // namespace

TEST(ScopeDeadlockDetectionTest, DepPoolUsesTimeoutForDifferentScopeHead) {
    DepListEntry entries[POOL_CAPACITY]{};
    std::atomic<int32_t> error_code{SIMPLER_ERROR_NONE};
    DepListPool pool;
    pool.init(entries, POOL_CAPACITY, &error_code);
    for (int32_t i = 0; i < POOL_CAPACITY; ++i) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    alignas(64) ChipTaskSlotState slot_states[WINDOW_SIZE]{};
    TaskDescriptor task_descriptors[WINDOW_SIZE]{};
    SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.task_window_size = WINDOW_SIZE;
    ring.task_window_mask = WINDOW_SIZE - 1;
    ring.slot_states = slot_states;
    ring.fc.current_task_index.store(2, std::memory_order_release);
    ring.fc.last_task_alive.store(0, std::memory_order_release);

    make_head_match_old_structural_predicate(slot_states[0], task_descriptors[0], 0, 0);
    ChipTaskSlotState *oldest_open_task = &slot_states[1];

    testing::internal::CaptureStderr();
    bool available = pool.ensure_space(ring, 1, oldest_open_task);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(available);
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
    EXPECT_NE(log.find("cannot reclaim space after ~500 ms"), std::string::npos);
    EXPECT_EQ(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}

TEST(ScopeDeadlockDetectionTest, DepPoolRejectsCurrentScopeHeadStructurally) {
    DepListEntry entries[POOL_CAPACITY]{};
    std::atomic<int32_t> error_code{SIMPLER_ERROR_NONE};
    DepListPool pool;
    pool.init(entries, POOL_CAPACITY, &error_code);
    for (int32_t i = 0; i < POOL_CAPACITY; ++i) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    alignas(64) ChipTaskSlotState slot_states[WINDOW_SIZE]{};
    TaskDescriptor task_descriptors[WINDOW_SIZE]{};
    SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.task_window_size = WINDOW_SIZE;
    ring.task_window_mask = WINDOW_SIZE - 1;
    ring.slot_states = slot_states;
    ring.fc.current_task_index.store(1, std::memory_order_release);
    ChipTaskSlotState *oldest_open_task = &slot_states[0];
    make_head_match_old_structural_predicate(slot_states[0], task_descriptors[0], 0, 0);

    testing::internal::CaptureStderr();
    bool available = pool.ensure_space(ring, 1, oldest_open_task);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(available);
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}
