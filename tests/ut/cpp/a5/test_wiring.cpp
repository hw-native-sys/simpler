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
 * Unit tests for scheduler wiring and completion paths:
 *
 * 1. wire_task()         — fanout wiring, early-finished detection,
 *                          fanin_count initialization, ready push
 * 2. on_mixed_task_complete() — COMPLETED transition, fanout traversal,
 *                               consumer fanin release
 * 3. on_task_release()   — fanin traversal, producer release,
 *                          self-CONSUMED check
 * 4. advance_ring_pointers() — CONSUMED slot scan, reset_for_reuse
 *
 * These tests exercise the core scheduling hot-paths that had zero coverage.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "scheduler/pto_scheduler.h"

// =============================================================================
// Fixture: sets up a scheduler with shared memory and provides helpers
// =============================================================================

class WiringTest : public ::testing::Test {
protected:
    PTO2SchedulerState sched{};
    PTO2SharedMemoryHandle *sm_handle = nullptr;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_default();
        ASSERT_NE(sm_handle, nullptr);
        bool ok = sched.init(sm_handle->header);
        ASSERT_TRUE(ok);
    }

    void TearDown() override {
        sched.destroy();
        if (sm_handle) {
            sm_handle->destroy();
        }
    }

    // Initialize a slot for testing wiring/completion
    void init_slot(
        PTO2TaskSlotState &slot, PTO2TaskState state, int32_t fanin_count, int32_t fanout_count, uint8_t ring_id = 0
    ) {
        memset(&slot, 0, sizeof(slot));
        slot.task_state.store(state);
        slot.fanin_count = fanin_count;
        slot.fanin_refcount.store(0);
        slot.fanout_count = fanout_count;
        slot.fanout_refcount.store(0);
        slot.fanout_lock.store(0);
        slot.fanout_head = nullptr;
        slot.ring_id = ring_id;
        slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        slot.dep_pool_mark = 0;
    }
};

// =============================================================================
// wire_task: no fanin (independent task)
// =============================================================================

TEST_F(WiringTest, WireTaskNoFaninBecomesReady) {
    // A task with 0 actual fanins should immediately be pushed to ready queue
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    sched.wire_task(rss, &task_slot, 0);

    // fanin_count set to 0 + 1 = 1 (the wiring "+1" sentinel)
    EXPECT_EQ(task_slot.fanin_count, 1);
    // fanin_refcount should be 1 (the +1 from no-fanin path)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);

    // Task should be in ready queue
    PTO2ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, &task_slot);
}

// =============================================================================
// wire_task: with fanin, all producers already completed (early-finished)
// =============================================================================

TEST_F(WiringTest, WireTaskAllProducersEarlyFinished) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer_slots[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    // Set up 2 producers that are already COMPLETED
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], PTO2_TASK_COMPLETED, 1, 2);
    }

    // Consumer task with 2 fanins
    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producer_slots[0];
    payload.fanin_inline_slot_states[1] = &producer_slots[1];

    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    sched.wire_task(rss, &task_slot, 2);

    // fanin_count = 2 + 1 = 3
    EXPECT_EQ(task_slot.fanin_count, 3);
    // early_finished = 2, init_rc = 2 + 1 = 3, so refcount should hit fanin_count
    EXPECT_GE(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // Task should be in ready queue
    PTO2ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, &task_slot);
}

// =============================================================================
// wire_task: with fanin, producers still pending (task NOT ready)
// =============================================================================

TEST_F(WiringTest, WireTaskProducersPendingTaskNotReady) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer_slots[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    // Producers are PENDING (not yet completed)
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], PTO2_TASK_PENDING, 1, 2);
    }

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producer_slots[0];
    payload.fanin_inline_slot_states[1] = &producer_slots[1];
    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    sched.wire_task(rss, &task_slot, 2);

    // fanin_count = 3 (2 + 1)
    EXPECT_EQ(task_slot.fanin_count, 3);
    // early_finished = 0, init_rc = 1 -> not ready
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);
    EXPECT_LT(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // Ready queue should be empty
    PTO2ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, nullptr);

    // Producers should have fanout_head pointing to task_slot
    EXPECT_NE(producer_slots[0].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[0].fanout_head->slot_state, &task_slot);
    EXPECT_NE(producer_slots[1].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[1].fanout_head->slot_state, &task_slot);
}

// =============================================================================
// wire_task: mixed early-finished and pending producers
// =============================================================================

TEST_F(WiringTest, WireTaskMixedProducerStates) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producers[3];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(producers[0], PTO2_TASK_COMPLETED, 1, 2);  // early finished
    init_slot(producers[1], PTO2_TASK_PENDING, 1, 2);    // in flight (< COMPLETED)
    init_slot(producers[2], PTO2_TASK_CONSUMED, 1, 2);   // early finished (>= COMPLETED)

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 3;
    for (int i = 0; i < 3; i++) {
        payload.fanin_inline_slot_states[i] = &producers[i];
    }
    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    sched.wire_task(rss, &task_slot, 3);

    // fanin_count = 4 (3 + 1)
    EXPECT_EQ(task_slot.fanin_count, 4);
    // early_finished = 2 (COMPLETED + CONSUMED), init_rc = 3
    // Not yet 4 -> not ready (one producer still running)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 3);

    // Only the running producer should have the consumer in its fanout chain
    EXPECT_EQ(producers[0].fanout_head, nullptr);  // early finished, no dep entry added
    EXPECT_NE(producers[1].fanout_head, nullptr);  // running, dep entry added
    EXPECT_EQ(producers[2].fanout_head, nullptr);  // early finished
}

// =============================================================================
// on_mixed_task_complete: notifies consumers via fanout chain
// =============================================================================

TEST_F(WiringTest, OnMixedTaskCompleteNotifiesConsumers) {
    alignas(64) PTO2TaskSlotState producer;
    alignas(64) PTO2TaskSlotState consumer1, consumer2;
    alignas(64) PTO2TaskPayload prod_payload;
    memset(&prod_payload, 0, sizeof(prod_payload));
    PTO2TaskDescriptor desc{};

    // Producer in flight (PENDING, not yet COMPLETED) with 2 consumers in fanout chain
    init_slot(producer, PTO2_TASK_PENDING, 1, 1);
    producer.payload = &prod_payload;
    producer.task = &desc;

    // Consumer1: needs 1 more fanin to become ready
    init_slot(consumer1, PTO2_TASK_PENDING, 2, 1);
    consumer1.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer1.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);

    // Consumer2: this release will make it ready
    init_slot(consumer2, PTO2_TASK_PENDING, 2, 1);
    consumer2.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer2.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);

    // Build fanout chain: producer -> consumer2 -> consumer1
    PTO2DepListEntry dep_entries[2];
    dep_entries[0].slot_state = &consumer1;
    dep_entries[0].next = nullptr;
    dep_entries[1].slot_state = &consumer2;
    dep_entries[1].next = &dep_entries[0];
    producer.fanout_head = &dep_entries[1];

    sched.on_mixed_task_complete(producer);

    // Producer should be COMPLETED
    EXPECT_EQ(producer.task_state.load(), PTO2_TASK_COMPLETED);

    // Both consumers should have fanin_refcount incremented
    EXPECT_EQ(consumer1.fanin_refcount.load(), 2);
    EXPECT_EQ(consumer2.fanin_refcount.load(), 2);

    // Both consumers should be ready (fanin_refcount == fanin_count)
    PTO2ResourceShape shape = consumer1.active_mask.to_shape();
    auto *r1 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    auto *r2 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_TRUE((r1 == &consumer1 && r2 == &consumer2) || (r1 == &consumer2 && r2 == &consumer1));
}

// =============================================================================
// on_task_release: releases producers via fanin traversal
// =============================================================================

TEST_F(WiringTest, OnTaskReleaseReleasesProducers) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producers[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    // 2 producers, each COMPLETED with fanout_count=1
    for (int i = 0; i < 2; i++) {
        init_slot(producers[i], PTO2_TASK_COMPLETED, 1, 1);
    }

    init_slot(task_slot, PTO2_TASK_COMPLETED, 3, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producers[0];
    payload.fanin_inline_slot_states[1] = &producers[1];
    // Need a valid fanin_spill_pool even though we don't spill
    PTO2FaninPool dummy_pool{};
    PTO2FaninSpillEntry dummy_entries[4];
    std::atomic<int32_t> dummy_error{PTO2_ERROR_NONE};
    dummy_pool.init(dummy_entries, 4, &dummy_error);
    payload.fanin_spill_pool = &dummy_pool;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    int32_t fanin_count = sched.on_task_release(task_slot);
    EXPECT_EQ(fanin_count, 2);

    // Each producer should have fanout_refcount incremented
    EXPECT_EQ(producers[0].fanout_refcount.load(), 1);
    EXPECT_EQ(producers[1].fanout_refcount.load(), 1);

    // Producers with fanout_refcount == fanout_count AND COMPLETED -> CONSUMED
    EXPECT_EQ(producers[0].task_state.load(), PTO2_TASK_CONSUMED);
    EXPECT_EQ(producers[1].task_state.load(), PTO2_TASK_CONSUMED);
}

// =============================================================================
// advance_ring_pointers: scans CONSUMED slots, resets, advances last_alive
// =============================================================================

TEST_F(WiringTest, AdvanceRingPointersScansConsumed) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    // Submit 3 tasks via flow control
    ring->fc.current_task_index.store(3, std::memory_order_release);

    // Mark all 3 as CONSUMED
    for (int i = 0; i < 3; i++) {
        auto &slot = ring->get_slot_state_by_task_id(i);
        slot.task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);
    }

    EXPECT_EQ(rss.last_task_alive, 0);
    rss.advance_ring_pointers();
    EXPECT_EQ(rss.last_task_alive, 3);

    // Verify SM was synced
    EXPECT_EQ(ring->fc.last_task_alive.load(), 3);
}

TEST_F(WiringTest, AdvanceRingPointersStopsAtNonConsumed) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(5, std::memory_order_release);

    // Tasks 0,1 CONSUMED; task 2 COMPLETED (not consumed)
    ring->get_slot_state_by_task_id(0).task_state.store(PTO2_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(1).task_state.store(PTO2_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(2).task_state.store(PTO2_TASK_COMPLETED);

    rss.advance_ring_pointers();
    EXPECT_EQ(rss.last_task_alive, 2) << "Should stop at first non-CONSUMED slot";
}

TEST_F(WiringTest, AdvanceRingPointersResetsSlots) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(1, std::memory_order_release);

    auto &slot = ring->get_slot_state_by_task_id(0);
    slot.task_state.store(PTO2_TASK_CONSUMED);
    slot.fanout_count = 5;
    slot.fanin_refcount.store(3);
    slot.fanout_refcount.store(2);
    slot.completed_subtasks.store(1);

    rss.advance_ring_pointers();

    // After reset_for_reuse: fanout_count=1, fanin_refcount=0, etc.
    EXPECT_EQ(slot.fanout_count, 1);
    EXPECT_EQ(slot.fanin_refcount.load(), 0);
    EXPECT_EQ(slot.fanout_refcount.load(), 0);
    EXPECT_EQ(slot.completed_subtasks.load(), 0);
    EXPECT_EQ(slot.fanout_head, nullptr);
}

// =============================================================================
// drain_wiring_queue: pushes tasks through SPSC queue
// =============================================================================

TEST_F(WiringTest, DrainWiringQueueProcessesTasks) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    // Push into wiring SPSC queue (orchestrator side)
    ASSERT_TRUE(sched.wiring.queue.push(&task_slot));

    // Drain (scheduler thread 0 side)
    int wired = sched.drain_wiring_queue(true /* force_drain */);
    EXPECT_EQ(wired, 1);

    // Task should be ready
    PTO2ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, &task_slot);
}

TEST_F(WiringTest, DrainWiringQueueBackoffDefers) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    sched.wiring.queue.push(&task_slot);

    // Without force_drain, single item < BATCH_SIZE → backoff
    sched.wiring.backoff_counter = 0;
    int wired = sched.drain_wiring_queue(false);
    EXPECT_EQ(wired, 0) << "Backoff should defer when queue < BATCH_SIZE";
    EXPECT_EQ(sched.wiring.backoff_counter, 1);
}

TEST_F(WiringTest, DrainWiringQueueBackoffLimitForcesProcess) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    sched.wiring.queue.push(&task_slot);

    // Set backoff at limit → should process
    sched.wiring.backoff_counter = PTO2SchedulerState::WiringState::BACKOFF_LIMIT;
    int wired = sched.drain_wiring_queue(false);
    EXPECT_EQ(wired, 1) << "Backoff limit reached should force processing";
}
