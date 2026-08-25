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
 * Unit tests for Orch-side wiring and scheduler completion paths:
 *
 * 1. Orch-side wiring    — fanout wiring, early-finished detection,
 *                          fanin_count initialization, ready push
 * 2. on_task_complete() — COMPLETED transition, fanout traversal,
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

#include "orchestrator.h"
#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "tensormap_and_ringbuffer/task_id_encoding.h"

// =============================================================================
// Fixture: sets up runtime state with shared memory and provides helpers
// =============================================================================

class WiringTest : public ::testing::Test {
protected:
    OrchestratorState orch{};
    SchedulerState sched{};
    SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    TaskDescriptor slot_task_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = SchedulerState::reserve_layout(sched_arena);
        ASSERT_NE(sched_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, sched_arena, sm_handle->header));
        sched.wire_arena_pointers(layout, sched_arena);
        orch.set_scheduler(&sched);
        sched.set_publication_batching_enabled(true);
    }

    void TearDown() override {
        sched.destroy();
        sched_arena.release();
        sm_arena.release();
    }

    // Initialize a slot for testing wiring/completion
    void init_slot(
        ChipTaskSlotState &slot, ChipTaskState state, int32_t fanin_count, int32_t fanout_count, uint8_t ring_id = 0
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
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.completed_subtasks.store(0);
        slot.total_required_subtasks = 1;
        slot.logical_block_num = 1;
        slot.dep_pool_mark = 0;
        TaskPayload &slot_pl = slot_payload_pool_[slot_payload_pool_idx_++ % kSlotPayloadPoolSize];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot.payload = &slot_pl;
        TaskDescriptor &slot_task = slot_task_pool_[(slot_payload_pool_idx_ - 1) % kSlotPayloadPoolSize];
        memset(&slot_task, 0, sizeof(slot_task));
        slot.task = &slot_task;
    }

    void publish_no_fanin(ChipTaskSlotState &slot) {
        slot.fanin_count = 1;
        slot.fanin_refcount.store(1, std::memory_order_release);
        orch.mark_dep_pool_position(slot);
        sched.push_ready_routed(&slot);
    }

    void wire_fanin(ChipTaskSlotState &slot, int32_t wfanin) {
        auto &rss = sched.ring_sched_states[slot.ring_id];
        bool ok = rss.dep_pool.ensure_space(*rss.ring, wfanin);
        if (ok) {
            orch.wire_fanin_task(slot, wfanin);
        }
        ASSERT_TRUE(ok);
    }

    // Ask the scheduler for an exact publication on ring 0, the way a blocked
    // reclaim consumer does once it has spun without progress.
    void request_and_drain_publication(int32_t ring_id = 0) {
        sched.publication_request_mask.store(ring_mask_bit(ring_id), std::memory_order_release);
        ASSERT_TRUE(sched.drain_publication_requests());
    }
};

TEST(ReclaimHeadMatchTest, ComparesExactRingAndLocalTaskId) {
    TaskDescriptor descriptor{};
    ChipTaskSlotState slot{};
    slot.task = &descriptor;
    descriptor.task_id = simpler::tmr::make_task_id(0, 16);

    EXPECT_FALSE(reclaim_head_matches_open_task(0, 0, &slot));
    EXPECT_TRUE(reclaim_head_matches_open_task(16, 0, &slot));
    EXPECT_FALSE(reclaim_head_matches_open_task(16, 1, &slot));
    EXPECT_FALSE(reclaim_head_matches_open_task(16, 0, nullptr));
}

// =============================================================================
// Orch-side publish: no fanin (independent task)
// =============================================================================

TEST_F(WiringTest, NoFaninTaskBecomesReady) {
    // A task with 0 actual fanins should immediately be pushed to ready queue
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    payload.fanin_wait_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    publish_no_fanin(task_slot);

    // fanin_count set to 0 + 1 = 1 (the wiring "+1" sentinel)
    EXPECT_EQ(task_slot.fanin_count, 1);
    // fanin_refcount should be 1 (the +1 from no-fanin path)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);

    // Task should be in ready queue
    ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, &task_slot);
}

// =============================================================================
// Orch-side wiring: with fanin, all producers already completed (early-finished)
// =============================================================================

TEST_F(WiringTest, WireTaskAllProducersEarlyFinished) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) ChipTaskSlotState producer_slots[2];
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    // Set up 2 producers that are already COMPLETED
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], CHIP_TASK_COMPLETED, 1, 2);
    }

    // Consumer task with 2 fanins
    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_wait_count = 2;
    payload.fanin_inline_edges[0].set(&producer_slots[0], DEP_WAIT | DEP_RETAIN);
    payload.fanin_inline_edges[1].set(&producer_slots[1], DEP_WAIT | DEP_RETAIN);

    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);

    // fanin_count = 2 + 1 = 3
    EXPECT_EQ(task_slot.fanin_count, 3);
    // completed_fanin = 2, init_rc = 2 + 1 = 3, so refcount should hit fanin_count
    EXPECT_GE(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // Task should be in ready queue
    ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, &task_slot);
}

// =============================================================================
// Orch-side wiring: with fanin, producers still pending (task NOT ready)
// =============================================================================

TEST_F(WiringTest, WireTaskProducersPendingTaskNotReady) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) ChipTaskSlotState producer_slots[2];
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    // Producers are PENDING (not yet completed)
    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], CHIP_TASK_PENDING, 1, 2);
    }

    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_wait_count = 2;
    payload.fanin_inline_edges[0].set(&producer_slots[0], DEP_WAIT | DEP_RETAIN);
    payload.fanin_inline_edges[1].set(&producer_slots[1], DEP_WAIT | DEP_RETAIN);
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);

    // fanin_count = 3 (2 + 1)
    EXPECT_EQ(task_slot.fanin_count, 3);
    // completed_fanin = 0, init_rc = 1 -> not ready
    EXPECT_EQ(task_slot.fanin_refcount.load(), 1);
    EXPECT_LT(task_slot.fanin_refcount.load(), task_slot.fanin_count);

    // Ready queue should be empty
    ResourceShape shape = task_slot.active_mask.to_shape();
    auto *popped = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_EQ(popped, nullptr);

    // Producers should have fanout_head pointing to task_slot
    EXPECT_NE(producer_slots[0].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[0].fanout_head->slot_state, &task_slot);
    EXPECT_NE(producer_slots[1].fanout_head, nullptr);
    EXPECT_EQ(producer_slots[1].fanout_head->slot_state, &task_slot);
}

// =============================================================================
// Orch-side wiring: mixed early-finished and pending producers
// =============================================================================

TEST_F(WiringTest, WireTaskMixedProducerStates) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) ChipTaskSlotState producers[3];
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    init_slot(producers[0], CHIP_TASK_COMPLETED, 1, 2);  // early finished
    init_slot(producers[1], CHIP_TASK_PENDING, 1, 2);    // in flight (< COMPLETED)
    init_slot(producers[2], CHIP_TASK_CONSUMED, 1, 2);   // early finished (>= COMPLETED)

    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 3;
    payload.fanin_wait_count = 3;
    for (int i = 0; i < 3; i++) {
        payload.fanin_inline_edges[i].set(&producers[i], DEP_WAIT | DEP_RETAIN);
    }
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 3);

    // fanin_count = 4 (3 + 1)
    EXPECT_EQ(task_slot.fanin_count, 4);
    // completed_fanin = 2 (COMPLETED + CONSUMED), init_rc = 3
    // Not yet 4 -> not ready (one producer still running)
    EXPECT_EQ(task_slot.fanin_refcount.load(), 3);

    // Only the running producer should have the consumer in its fanout chain
    EXPECT_EQ(producers[0].fanout_head, nullptr);  // early finished, no dep entry added
    EXPECT_NE(producers[1].fanout_head, nullptr);  // running, dep entry added
    EXPECT_EQ(producers[2].fanout_head, nullptr);  // early finished
}

TEST_F(WiringTest, SyncStartDoorbellPassHasOneOwner) {
    TaskPayload payload{};

    for (int iteration = 0; iteration < 1000; iteration++) {
        payload.early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        std::atomic<bool> start{false};
        bool first_won = false;
        bool second_won = false;

        std::thread first([&] {
            while (!start.load(std::memory_order_acquire)) {}
            first_won = SchedulerState::try_claim_early_dispatch_launch(payload);
        });
        std::thread second([&] {
            while (!start.load(std::memory_order_acquire)) {}
            second_won = SchedulerState::try_claim_early_dispatch_launch(payload);
        });

        start.store(true, std::memory_order_release);
        first.join();
        second.join();

        EXPECT_NE(first_won, second_won);
        EXPECT_EQ(payload.early_dispatch_launch_state.load(std::memory_order_acquire), EARLY_DISPATCH_LAUNCH_RINGING);
    }
}

TEST_F(WiringTest, SyncStartStagingFinalizeRetriesProducerFirstRendezvous) {
    alignas(64) ChipTaskSlotState sync_consumer, downstream;
    init_slot(sync_consumer, CHIP_TASK_PENDING, 1, 1);
    init_slot(downstream, CHIP_TASK_PENDING, 1, 1);

    sync_consumer.active_mask = ActiveMask(SUBTASK_MASK_AIV0);
    sync_consumer.task_attrs.set_sync_start();
    sync_consumer.task_attrs.set_early_resolve(true);
    sync_consumer.logical_block_num = 2;
    sync_consumer.next_block_idx.store(2, std::memory_order_relaxed);
    sched.record_published_blocks(sync_consumer, sync_consumer.logical_block_num);
    sync_consumer.payload->staged_core_mask[0].store(0b11, std::memory_order_relaxed);
    sync_consumer.payload->running_slot_count.store(0, std::memory_order_relaxed);
    sync_consumer.payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    downstream.payload->fanin_actual_count = 1;
    DepListEntry dep{};
    dep.slot_state = &downstream;
    sync_consumer.fanout_head = &dep;

    EXPECT_TRUE(sched.try_early_dispatch_release(sync_consumer));
    EXPECT_EQ(sync_consumer.payload->early_dispatch_state.load(), EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(sync_consumer.payload->early_dispatch_launch_state.load(), EARLY_DISPATCH_LAUNCH_NONE);
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 0);

    sync_consumer.payload->running_slot_count.store(2, std::memory_order_seq_cst);
    EXPECT_TRUE(sched.retry_sync_start_rendezvous_after_staging(sync_consumer));
    EXPECT_EQ(sync_consumer.payload->early_dispatch_launch_state.load(), EARLY_DISPATCH_LAUNCH_COMPLETE);
    EXPECT_TRUE(sync_consumer.has_dispatch_propagated());
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);

    EXPECT_FALSE(sched.retry_sync_start_rendezvous_after_staging(sync_consumer));
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);
}

TEST_F(WiringTest, SyncStartProducerReleaseCompletesStagerFirstRendezvous) {
    alignas(64) ChipTaskSlotState sync_consumer, downstream;
    init_slot(sync_consumer, CHIP_TASK_PENDING, 1, 1);
    init_slot(downstream, CHIP_TASK_PENDING, 1, 1);

    sync_consumer.active_mask = ActiveMask(SUBTASK_MASK_AIV0);
    sync_consumer.task_attrs.set_sync_start();
    sync_consumer.task_attrs.set_early_resolve(true);
    sync_consumer.logical_block_num = 2;
    sync_consumer.next_block_idx.store(2, std::memory_order_relaxed);
    sched.record_published_blocks(sync_consumer, sync_consumer.logical_block_num);
    sync_consumer.payload->staged_core_mask[0].store(0b11, std::memory_order_relaxed);
    sync_consumer.payload->running_slot_count.store(2, std::memory_order_relaxed);
    sync_consumer.payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    downstream.payload->fanin_actual_count = 1;
    DepListEntry dep{};
    dep.slot_state = &downstream;
    sync_consumer.fanout_head = &dep;

    EXPECT_FALSE(sched.retry_sync_start_rendezvous_after_staging(sync_consumer));
    EXPECT_TRUE(sched.try_early_dispatch_release(sync_consumer));
    EXPECT_EQ(sync_consumer.payload->early_dispatch_state.load(), EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(sync_consumer.payload->early_dispatch_launch_state.load(), EARLY_DISPATCH_LAUNCH_COMPLETE);
    EXPECT_TRUE(sync_consumer.has_dispatch_propagated());
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);

    EXPECT_FALSE(sched.retry_sync_start_rendezvous_after_staging(sync_consumer));
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);
}

TEST_F(WiringTest, EarlySyncFinishBetweenReleasePhasesRetainsOwnerCompleteState) {
    alignas(64) ChipTaskSlotState sync_consumer;
    init_slot(sync_consumer, CHIP_TASK_PENDING, 1, 1);
    sync_consumer.active_mask = ActiveMask(SUBTASK_MASK_AIV0);
    sync_consumer.task_attrs.set_sync_start();
    sync_consumer.logical_block_num = 2;
    sync_consumer.payload->early_dispatch_state.store(EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(SchedulerState::try_claim_early_sync_drain(*sync_consumer.payload));
    SchedulerState::mark_early_sync_drain_armed(*sync_consumer.payload);

    EXPECT_FALSE(sched.try_early_dispatch_release(sync_consumer));
    sync_consumer.next_block_idx.store(sync_consumer.logical_block_num, std::memory_order_seq_cst);
    SchedulerState::finish_early_sync_drain(*sync_consumer.payload);

    EXPECT_TRUE(SchedulerState::publish_ready_to_early_sync_drain(*sync_consumer.payload));
    EXPECT_EQ(
        sync_consumer.payload->early_sync_drain_state.load(),
        EARLY_SYNC_DRAIN_OWNER | EARLY_SYNC_DRAIN_ARMED | EARLY_SYNC_DRAIN_READY | EARLY_SYNC_DRAIN_COMPLETE
    );
}

// =============================================================================
// on_task_complete: notifies consumers via fanout chain
// =============================================================================

TEST_F(WiringTest, OnMixedTaskCompleteNotifiesConsumers) {
    alignas(64) ChipTaskSlotState producer;
    alignas(64) ChipTaskSlotState consumer1, consumer2;
    alignas(64) TaskPayload prod_payload;
    memset(&prod_payload, 0, sizeof(prod_payload));
    TaskDescriptor desc{};

    // Producer in flight (PENDING, not yet COMPLETED) with 2 consumers in fanout chain
    init_slot(producer, CHIP_TASK_PENDING, 1, 1);
    producer.payload = &prod_payload;
    producer.task = &desc;

    // Consumer1: needs 1 more fanin to become ready
    init_slot(consumer1, CHIP_TASK_PENDING, 2, 1);
    consumer1.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer1.active_mask = ActiveMask(SUBTASK_MASK_AIC);

    // Consumer2: this release will make it ready
    init_slot(consumer2, CHIP_TASK_PENDING, 2, 1);
    consumer2.fanin_refcount.store(1);  // 1 of 2 satisfied
    consumer2.active_mask = ActiveMask(SUBTASK_MASK_AIC);

    // Build fanout chain: producer -> consumer2 -> consumer1
    DepListEntry dep_entries[2];
    dep_entries[0].slot_state = &consumer1;
    dep_entries[0].next = nullptr;
    dep_entries[1].slot_state = &consumer2;
    dep_entries[1].next = &dep_entries[0];
    producer.fanout_head = &dep_entries[1];

#if SIMPLER_SCHED_PROFILING
    auto completion_stats = sched.on_task_complete(producer, /*thread_idx=*/0);
    EXPECT_EQ(completion_stats.fanout_edges, 2U);
#else
    EXPECT_EQ(sched.on_task_complete(producer), 2U);
#endif

    // Producer should be COMPLETED
    EXPECT_EQ(producer.task_state.load(), CHIP_TASK_COMPLETED);

    // Both consumers should have fanin_refcount incremented
    EXPECT_EQ(consumer1.fanin_refcount.load(), 2);
    EXPECT_EQ(consumer2.fanin_refcount.load(), 2);

    // Both consumers should be ready (fanin_refcount == fanin_count)
    ResourceShape shape = consumer1.active_mask.to_shape();
    auto *r1 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    auto *r2 = sched.ready_queues[static_cast<int32_t>(shape)].pop();
    EXPECT_TRUE((r1 == &consumer1 && r2 == &consumer2) || (r1 == &consumer2 && r2 == &consumer1));
}

// =============================================================================
// on_task_release: releases producers via fanin traversal
// =============================================================================

TEST_F(WiringTest, OnTaskReleaseReleasesProducers) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) ChipTaskSlotState producers[2];
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    // 2 producers, each COMPLETED with fanout_count=1
    for (int i = 0; i < 2; i++) {
        init_slot(producers[i], CHIP_TASK_COMPLETED, 1, 1);
    }

    init_slot(task_slot, CHIP_TASK_COMPLETED, 3, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_wait_count = 2;
    payload.fanin_inline_edges[0].set(&producers[0], DEP_WAIT | DEP_RETAIN);
    payload.fanin_inline_edges[1].set(&producers[1], DEP_WAIT | DEP_RETAIN);
    // Need a valid fanin_spill_pool even though we don't spill
    FaninPool dummy_pool{};
    FaninSpillEntry dummy_entries[4];
    std::atomic<int32_t> dummy_error{SIMPLER_ERROR_NONE};
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
    EXPECT_EQ(producers[0].task_state.load(), CHIP_TASK_CONSUMED);
    EXPECT_EQ(producers[1].task_state.load(), CHIP_TASK_CONSUMED);
}

// =============================================================================
// WAIT/RETAIN split (issue #1375): an ordering-only (DEP_WAIT) producer drops
// its submit->wire pin at wiring; a retention (DEP_WAIT|DEP_RETAIN) producer
// keeps it until on_task_release. Both are linked for completion notification.
// =============================================================================

TEST_F(WiringTest, OrderingOnlyReleasedAtWiringRetentionHeldUntilRelease) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) ChipTaskSlotState wait_producer;    // DEP_WAIT only (modifier)
    alignas(64) ChipTaskSlotState retain_producer;  // DEP_WAIT|DEP_RETAIN (creator)
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    // Both live (PENDING) with a single submit pin (fanout_count = 1).
    init_slot(wait_producer, CHIP_TASK_PENDING, 1, 1);
    init_slot(retain_producer, CHIP_TASK_PENDING, 1, 1);

    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_wait_count = 2;
    payload.fanin_inline_edges[0].set(&wait_producer, DEP_WAIT);
    payload.fanin_inline_edges[1].set(&retain_producer, DEP_WAIT | DEP_RETAIN);
    FaninPool dummy_pool{};
    FaninSpillEntry dummy_entries[4];
    std::atomic<int32_t> dummy_error{SIMPLER_ERROR_NONE};
    dummy_pool.init(dummy_entries, 4, &dummy_error);
    payload.fanin_spill_pool = &dummy_pool;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    // Both WAIT edges gate readiness (wfanin = 2) and both link onto fanout_head.
    wire_fanin(task_slot, 2);
    EXPECT_NE(wait_producer.fanout_head, nullptr);
    EXPECT_NE(retain_producer.fanout_head, nullptr);

    // Ordering-only pin released at wiring; retention pin still held.
    EXPECT_EQ(wait_producer.fanout_refcount.load(), 1);
    EXPECT_EQ(retain_producer.fanout_refcount.load(), 0);

    // Release: only the retention edge releases here; the ordering edge is not
    // released a second time.
    sched.on_task_release(task_slot);
    EXPECT_EQ(wait_producer.fanout_refcount.load(), 1);
    EXPECT_EQ(retain_producer.fanout_refcount.load(), 1);
}

// on_task_release must honor per-edge flags in the spill region too: a spilled
// DEP_RETAIN edge is released; inline ordering-only edges are skipped.
TEST_F(WiringTest, ReleaseHonorsRetainFlagInSpillRegion) {
    alignas(64) ChipTaskSlotState filler;        // 64 inline DEP_WAIT-only edges
    alignas(64) ChipTaskSlotState spill_retain;  // 1 spilled DEP_RETAIN edge
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    // filler carries a large fanout_count so releasing it can never consume it.
    init_slot(filler, CHIP_TASK_COMPLETED, 1, 100);
    init_slot(spill_retain, CHIP_TASK_COMPLETED, 1, 1);
    init_slot(task_slot, CHIP_TASK_COMPLETED, 0, 1);

    for (int i = 0; i < CHIP_FANIN_INLINE_CAP; i++) {
        payload.fanin_inline_edges[i].set(&filler, DEP_WAIT);
    }
    FaninPool spill_pool{};
    FaninSpillEntry spill_entries[4];
    std::atomic<int32_t> err{SIMPLER_ERROR_NONE};
    spill_pool.init(spill_entries, 4, &err);
    auto *e = spill_pool.alloc();
    int32_t spill_start = spill_pool.top - 1;
    e->set(&spill_retain, DEP_WAIT | DEP_RETAIN);

    payload.fanin_actual_count = CHIP_FANIN_INLINE_CAP + 1;

    payload.fanin_wait_count = CHIP_FANIN_INLINE_CAP + 1;
    payload.fanin_spill_start = spill_start;
    payload.fanin_spill_pool = &spill_pool;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    sched.on_task_release(task_slot);

    // Ordering-only inline edges are skipped; filler is untouched.
    EXPECT_EQ(filler.fanout_refcount.load(), 0);
    // The spilled retention edge is released (and consumed: rc == fc, COMPLETED).
    EXPECT_EQ(spill_retain.fanout_refcount.load(), 1);
    EXPECT_EQ(spill_retain.task_state.load(), CHIP_TASK_CONSUMED);
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
        slot.task_state.store(CHIP_TASK_CONSUMED, std::memory_order_release);
    }

    EXPECT_EQ(rss.last_task_alive, 0);
    rss.advance_ring_pointers();
    EXPECT_EQ(rss.last_task_alive, 3);

    // Verify SM was synced
    EXPECT_EQ(ring->fc.last_task_alive.load(), 3);
}

TEST_F(WiringTest, AdvanceRingPointersBatchesSharedMemoryPublication) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(18, std::memory_order_release);
    for (int i = 0; i < 17; i++) {
        ring->get_slot_state_by_task_id(i).task_state.store(CHIP_TASK_CONSUMED);
    }

    rss.advance_ring_pointers();
    EXPECT_EQ(rss.last_task_alive, 17);
    EXPECT_EQ(ring->fc.last_task_alive.load(), 17);

    ring->fc.last_task_alive.store(0);
    rss.last_task_alive = 0;
    rss.last_published_to_sm = 0;
    for (int advances : {1, 15, 16, 17}) {
        for (int i = 0; i < advances; i++) {
            ring->get_slot_state_by_task_id(i).task_state.store(CHIP_TASK_CONSUMED);
        }
        ring->get_slot_state_by_task_id(advances).task_state.store(CHIP_TASK_COMPLETED);
        rss.advance_ring_pointers();
        EXPECT_EQ(ring->fc.last_task_alive.load(), advances >= 16 ? advances : 0);

        ring->fc.last_task_alive.store(0);
        rss.last_task_alive = 0;
        rss.last_published_to_sm = 0;
    }
}

TEST_F(WiringTest, AdvanceRingPointersPublishesDrainedTail) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    for (int advances : {1, 15, 16, 17}) {
        ring->fc.current_task_index.store(advances, std::memory_order_release);
        for (int i = 0; i < advances; i++) {
            ring->get_slot_state_by_task_id(i).task_state.store(CHIP_TASK_CONSUMED);
        }

        rss.advance_ring_pointers();
        EXPECT_EQ(rss.last_task_alive, advances);
        EXPECT_EQ(ring->fc.last_task_alive.load(), advances);

        ring->fc.last_task_alive.store(0);
        rss.last_task_alive = 0;
        rss.last_published_to_sm = 0;
    }
}

// =============================================================================
// Withheld reclaim progress: request -> publish -> reclaim
//
// Batched publication lets the shared watermark trail scheduler-local
// reclamation by PUBLISH_INTERVAL_K - 1 tasks, so a reclaim consumer can run out
// of space while the scheduler has already retired what it needs. The consumer
// asks for an exact publication and resumes once it lands.
//
// What the tests below assert is that handshake's three steps, each in one
// thread. They deliberately do not assert the step in between — that a spinning
// consumer reaches its request after 10 ms of no reclaim progress. That step is
// a duration, and the same spin carries a 500 ms deadlock backstop, so a test
// that raced a servicing thread against it would be asserting that the servicing
// thread gets scheduled in time. Under `ctest -j` on a small runner it does not,
// and the production code then does exactly what it is specified to do — report
// the stall and give up — which such a test reports as a failed assertion.
// =============================================================================

TEST(ReclaimPublicationRequestTest, AcknowledgmentIsScopedToTheOutstandingRequest) {
    std::atomic<uint32_t> request_mask{0};
    std::atomic<uint32_t> ack_mask{0};
    ReclaimPublicationRequest request(&request_mask, &ack_mask, 1);
    ASSERT_TRUE(request.enabled());

    // An ack that predates the request describes an older watermark, so it is
    // cleared by the request rather than consumed by it.
    ack_mask.store(ring_mask_bit(1), std::memory_order_release);
    EXPECT_FALSE(request.poll_acknowledged());

    request.request();
    EXPECT_EQ(request_mask.load(std::memory_order_acquire), ring_mask_bit(1));
    EXPECT_EQ(ack_mask.load(std::memory_order_acquire) & ring_mask_bit(1), 0u);
    EXPECT_FALSE(request.poll_acknowledged());

    ack_mask.fetch_or(ring_mask_bit(1), std::memory_order_release);
    EXPECT_TRUE(request.poll_acknowledged());
    // One acknowledgment proves one publication: a later reclaim spin must
    // request again rather than re-reading this one.
    EXPECT_FALSE(request.poll_acknowledged());
}

TEST(ReclaimPublicationRequestTest, AnUnwiredRequestCountsAsSynchronized) {
    ReclaimPublicationRequest request(nullptr, nullptr, 0);

    EXPECT_FALSE(request.enabled());
    // No publisher to ask means the shared watermark is already the only one
    // there is, so the reclaim spin must not wait for an ack that cannot come.
    EXPECT_TRUE(request.poll_acknowledged());
}

TEST_F(WiringTest, DrainPublicationRequestsPublishesWithheldProgress) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(129, std::memory_order_release);
    ring->get_slot_state_by_task_id(128).task_state.store(CHIP_TASK_PENDING);
    rss.last_task_alive = 128;
    rss.last_published_to_sm = 113;
    ring->fc.last_task_alive.store(113, std::memory_order_release);

    // 15 retired tasks short of the publish interval: batching withholds them.
    rss.sync_to_sm();
    ASSERT_EQ(ring->fc.last_task_alive.load(), 113);

    sched.publication_request_mask.store(ring_mask_bit(0), std::memory_order_release);
    EXPECT_TRUE(sched.drain_publication_requests());

    EXPECT_EQ(ring->fc.last_task_alive.load(), 128);
    EXPECT_EQ(rss.last_published_to_sm, 128);
    EXPECT_EQ(sched.publication_ack_mask.load() & ring_mask_bit(0), ring_mask_bit(0));
    EXPECT_EQ(sched.publication_request_mask.load() & ring_mask_bit(0), 0u);

    // The return value reports whether the watermark moved, not whether the
    // request was serviced: a request with nothing withheld is still acked.
    sched.publication_request_mask.store(ring_mask_bit(0), std::memory_order_release);
    EXPECT_FALSE(sched.drain_publication_requests());
    EXPECT_EQ(sched.publication_ack_mask.load() & ring_mask_bit(0), ring_mask_bit(0));
    EXPECT_EQ(sched.publication_request_mask.load() & ring_mask_bit(0), 0u);
}

TEST_F(WiringTest, TaskWindowOpensOnceWithheldProgressIsPublished) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    alignas(64) char heap[64] = {};
    ring->fc.current_task_index.store(3, std::memory_order_release);
    ring->get_task_by_task_id(0).packed_buffer_end = heap;
    ring->get_slot_state_by_task_id(0).task_state.store(CHIP_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(1).task_state.store(CHIP_TASK_PENDING);
    rss.advance_ring_pointers();

    ASSERT_EQ(rss.last_task_alive, 1);
    ASSERT_EQ(ring->fc.last_task_alive.load(), 0);

    TaskAllocator allocator;
    auto *orch_err = sm_layout::orch_error_code_addr(sm_handle->sm_base);
    allocator.init(
        ring->task_descriptors, 4, &ring->fc.current_task_index, &ring->fc.last_task_alive, heap, sizeof(heap),
        orch_err, ring->slot_states, 3, 0
    );
    allocator.set_reclaim_publication_request(&sched.publication_request_mask, &sched.publication_ack_mask);

    // A window of 4 admits task 3 only once the watermark retires task 0, and
    // the watermark the allocator can see still says every task is alive.
    ASSERT_EQ(allocator.window_size(), 4);
    ASSERT_EQ(allocator.task_head(), 3);
    ASSERT_EQ(allocator.task_tail(), 0);
    ASSERT_EQ(allocator.active_count(), allocator.window_size() - 1);

    request_and_drain_publication();
    ASSERT_EQ(allocator.task_tail(), 1);

    TaskAllocResult result = allocator.alloc(0, &ring->get_slot_state_by_task_id(0));

    EXPECT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 3);
    EXPECT_EQ(ring->fc.last_task_alive.load(), 1);
}

TEST_F(WiringTest, HeapReclaimsOnceWithheldProgressIsPublished) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    alignas(64) char heap[64] = {};
    TaskAllocator allocator;
    auto *orch_err = sm_layout::orch_error_code_addr(sm_handle->sm_base);
    allocator.init(
        ring->task_descriptors, 8, &ring->fc.current_task_index, &ring->fc.last_task_alive, heap, sizeof(heap),
        orch_err, ring->slot_states, 0, 0
    );
    allocator.set_reclaim_publication_request(&sched.publication_request_mask, &sched.publication_ack_mask);

    auto full_heap = allocator.alloc(sizeof(heap));
    ASSERT_FALSE(full_heap.failed());
    ring->get_task_by_task_id(full_heap.task_id).packed_buffer_end = full_heap.packed_end;
    auto live_tail = allocator.alloc(0);
    ASSERT_FALSE(live_tail.failed());
    ring->get_task_by_task_id(live_tail.task_id).packed_buffer_end = live_tail.packed_end;

    ring->get_slot_state_by_task_id(0).task_state.store(CHIP_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(1).task_state.store(CHIP_TASK_PENDING);
    rss.advance_ring_pointers();

    ASSERT_EQ(rss.last_task_alive, 1);
    ASSERT_EQ(ring->fc.last_task_alive.load(), 0);

    // Every heap byte belongs to task 0, which the visible watermark still
    // counts as alive, so the heap tail cannot move off it.
    ASSERT_EQ(allocator.heap_available(), 0u);
    ASSERT_EQ(allocator.heap_tail(), 0u);

    request_and_drain_publication();
    ASSERT_EQ(allocator.task_tail(), 1);

    TaskAllocResult result = allocator.alloc(8);

    EXPECT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 2);
    EXPECT_EQ(result.packed_base, heap);
    EXPECT_EQ(ring->fc.last_task_alive.load(), 1);
}

TEST_F(WiringTest, DependencyPoolReclaimsOnceWithheldProgressIsPublished) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(129, std::memory_order_release);
    ring->fc.last_task_alive.store(113, std::memory_order_release);
    ring->get_slot_state_by_task_id(127).dep_pool_mark = 5;
    ring->get_slot_state_by_task_id(128).task_state.store(CHIP_TASK_PENDING);
    rss.last_task_alive = 128;
    rss.last_published_to_sm = 113;
    rss.dep_pool.capacity = 8;
    rss.dep_pool.top = 9;
    rss.dep_pool.tail = 1;
    rss.dep_pool.last_reclaimed = 64;

    // The pool is full, and the watermark it can see is not a reclaim point:
    // cleanup runs every 64 retired tasks and 113 is under 64 + 64.
    ASSERT_EQ(rss.dep_pool.available(), 0);
    rss.dep_pool.reclaim(*ring, ring->fc.last_task_alive.load());
    ASSERT_EQ(rss.dep_pool.tail, 1);

    request_and_drain_publication();
    ASSERT_EQ(ring->fc.last_task_alive.load(), 128);

    // 128 is a reclaim point, and task 127's mark says entries below 5 are dead.
    EXPECT_TRUE(rss.dep_pool.ensure_space(*ring, 1));
    EXPECT_EQ(rss.dep_pool.tail, 5);
}

TEST_F(WiringTest, FaninPoolReclaimsOnceWithheldProgressIsPublished) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(2, std::memory_order_release);
    ring->get_slot_state_by_task_id(0).task_state.store(CHIP_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(1).task_state.store(CHIP_TASK_PENDING);
    rss.advance_ring_pointers();
    ASSERT_EQ(rss.last_task_alive, 1);
    ASSERT_EQ(ring->fc.last_task_alive.load(), 0);

    FaninSpillEntry entries[4]{};
    FaninPool pool{};
    auto *orch_err = sm_layout::orch_error_code_addr(sm_handle->sm_base);
    pool.init(entries, 4, orch_err);
    pool.set_reclaim_publication_request(&sched.publication_request_mask, &sched.publication_ack_mask, 0);
    for (int i = 0; i < 4; i++) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    auto &payload = ring->get_payload_by_task_id(0);
    payload.fanin_actual_count = CHIP_FANIN_INLINE_CAP + 1;
    payload.fanin_wait_count = CHIP_FANIN_INLINE_CAP + 1;
    payload.fanin_spill_start = 1;
    payload.fanin_spill_pool = &pool;

    // The pool is full and the only task holding spill entries is task 0, which
    // the visible watermark has not retired.
    ASSERT_EQ(pool.available(), 0);
    pool.reclaim(*ring, ring->fc.last_task_alive.load());
    ASSERT_EQ(pool.tail, 1);

    request_and_drain_publication();
    ASSERT_EQ(ring->fc.last_task_alive.load(), 1);

    EXPECT_TRUE(pool.ensure_space(*ring, 1));
    EXPECT_EQ(pool.tail, 2);
}

TEST_F(WiringTest, RingReuseResetsPublicationShadow) {
    auto &rss = sched.ring_sched_states[0];
    rss.last_task_alive = 17;
    rss.last_published_to_sm = 16;

    auto *orch_err = sm_layout::orch_error_code_addr(sm_handle->sm_base);
    rss.reset_for_reuse(sm_handle->sm_base, 0, orch_err);

    EXPECT_EQ(rss.last_task_alive, 0);
    EXPECT_EQ(rss.last_published_to_sm, 0);
    EXPECT_FALSE(rss.publication_batching_enabled);
    EXPECT_EQ(rss.ring, sm_layout::ring_header_addr(sm_handle->sm_base, 0));
}

TEST_F(WiringTest, AdvanceRingPointersStopsAtNonConsumed) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(5, std::memory_order_release);

    // Tasks 0,1 CONSUMED; task 2 COMPLETED (not consumed)
    ring->get_slot_state_by_task_id(0).task_state.store(CHIP_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(1).task_state.store(CHIP_TASK_CONSUMED);
    ring->get_slot_state_by_task_id(2).task_state.store(CHIP_TASK_COMPLETED);

    rss.advance_ring_pointers();
    EXPECT_EQ(rss.last_task_alive, 2) << "Should stop at first non-CONSUMED slot";
}

TEST_F(WiringTest, AdvanceRingPointersResetsSlots) {
    auto &rss = sched.ring_sched_states[0];
    auto *ring = rss.ring;

    ring->fc.current_task_index.store(1, std::memory_order_release);

    auto &slot = ring->get_slot_state_by_task_id(0);
    slot.task_state.store(CHIP_TASK_CONSUMED);
    slot.fanout_count = 5;
    slot.fanin_refcount.store(3);
    slot.fanout_refcount.store(2);
    slot.completed_subtasks.store(1);

    rss.advance_ring_pointers();

    // After reset_for_reuse: fanout_count=FANOUT_SCOPE_BIT (bit31 owning-scope
    // ref, 0 consumers), fanin_refcount=0, etc.
    EXPECT_EQ(slot.fanout_count, FANOUT_SCOPE_BIT);
    EXPECT_EQ(slot.fanin_refcount.load(), 0);
    EXPECT_EQ(slot.fanout_refcount.load(), 0);
    EXPECT_EQ(slot.completed_subtasks.load(), 0);
    EXPECT_EQ(slot.fanout_head, nullptr);
}

TEST_F(WiringTest, NoEdgePublishRecordsDepPoolMark) {
    alignas(64) ChipTaskSlotState task_slot;
    alignas(64) TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    TaskDescriptor desc{};

    init_slot(task_slot, CHIP_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    payload.fanin_wait_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    int32_t before_top = rss.dep_pool.top;
    publish_no_fanin(task_slot);
    EXPECT_EQ(task_slot.dep_pool_mark, before_top);
}

TEST_F(WiringTest, BatchPushReportsFullInsteadOfSpinning) {
    alignas(64) ChipTaskSlotState filler;
    init_slot(filler, CHIP_TASK_PENDING, 0, 1);
    auto &queue = sched.early_dispatch_queues[static_cast<int32_t>(filler.active_mask.to_shape())];
    for (uint64_t i = 0; i < queue.capacity; i++) {
        ASSERT_TRUE(queue.push_tagged(&filler, i));
    }

    ChipTaskSlotState *items[1] = {&filler};
    uint64_t tags[1] = {queue.capacity};
    // A full queue must end the call, not spin waiting for a consumer. Reaching
    // the next line at all is the assertion.
    EXPECT_FALSE(queue.push_batch_tagged(items, tags, 1));
    EXPECT_EQ(queue.size(), queue.capacity);

    // A batch larger than the queue can never be satisfied and must not spin.
    EXPECT_FALSE(queue.push_batch_tagged(items, tags, static_cast<int>(queue.capacity) + 1));
}

TEST_F(WiringTest, BatchPushSucceedsAfterSpaceIsReclaimed) {
    alignas(64) ChipTaskSlotState filler;
    init_slot(filler, CHIP_TASK_PENDING, 0, 1);
    auto &queue = sched.early_dispatch_queues[static_cast<int32_t>(filler.active_mask.to_shape())];
    for (uint64_t i = 0; i < queue.capacity; i++) {
        ASSERT_TRUE(queue.push_tagged(&filler, i));
    }
    ASSERT_NE(queue.pop(), nullptr);
    ASSERT_NE(queue.pop(), nullptr);

    ChipTaskSlotState *items[2] = {&filler, &filler};
    uint64_t tags[2] = {7, 8};
    EXPECT_TRUE(queue.push_batch_tagged(items, tags, 2));
    EXPECT_EQ(queue.size(), queue.capacity);
}

TEST_F(WiringTest, EarlyDispatchQueueOverflowFallsBackToNormalDispatch) {
    alignas(64) ChipTaskSlotState filler, consumer;
    init_slot(filler, CHIP_TASK_PENDING, 0, 1);
    init_slot(consumer, CHIP_TASK_PENDING, 1, 1);

    ResourceShape shape = consumer.active_mask.to_shape();
    auto &queue = sched.early_dispatch_queues[static_cast<int32_t>(shape)];
    for (uint64_t i = 0; i < queue.capacity; i++) {
        ASSERT_TRUE(queue.push_tagged(&filler, i));
    }

    sched.try_enqueue_early_dispatch_candidate(consumer);
    ASSERT_EQ(consumer.payload->early_dispatch_state.load(), EARLY_DISPATCH_NONE);

    // The overflowed candidate carries no early-dispatch claim, so the producer
    // release routes every block through the ordinary ready queue.
    EXPECT_TRUE(sched.route_ready_once(consumer));
    EXPECT_EQ(consumer.payload->early_dispatch_state.load(), EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(sched.ready_queues[static_cast<int32_t>(shape)].pop(), &consumer);
}

TEST_F(WiringTest, EarlyDispatchSyncStartQueueOverflowFallsBackToSyncReadyQueue) {
    alignas(64) ChipTaskSlotState filler, consumer;
    init_slot(filler, CHIP_TASK_PENDING, 0, 1);
    init_slot(consumer, CHIP_TASK_PENDING, 1, 1);
    consumer.task_attrs.set_sync_start();
    ASSERT_TRUE(consumer.task_attrs.requires_sync_start());

    auto &queue = sched.early_sync_start_queue;
    for (uint64_t i = 0; i < queue.capacity; i++) {
        ASSERT_TRUE(queue.push_tagged(&filler, i));
    }

    sched.try_enqueue_early_dispatch_candidate(consumer);
    EXPECT_EQ(consumer.payload->early_dispatch_state.load(), EARLY_DISPATCH_NONE);
    EXPECT_EQ(queue.size(), queue.capacity);

    // A sync_start cohort that never reached its drain falls back to the
    // shape's sync ready queue, not the plain one.
    ResourceShape shape = consumer.active_mask.to_shape();
    EXPECT_TRUE(sched.route_ready_once(consumer));
    EXPECT_EQ(sched.ready_sync_queues[static_cast<int32_t>(shape)].pop(), &consumer);
}
