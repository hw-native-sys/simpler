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

#include "pto_orchestrator.h"
#include "utils/device_arena.h"
#include "scheduler/pto_scheduler.h"

void reset_test_reg_stub();
uint64_t get_test_reg_stub_value();
uint64_t get_test_reg_stub_base_addr();

// =============================================================================
// Fixture: sets up runtime state with shared memory and provides helpers
// =============================================================================

class WiringTest : public ::testing::Test {
protected:
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    DeviceArena sm_arena;
    DeviceArena sched_arena;

    // Each init_slot()'d slot gets a distinct zeroed payload from this pool,
    // mirroring orch::prepare_task's bind_buffers: every production slot has a
    // payload, and the scheduler's release/propagate paths dereference it.
    static constexpr int kSlotPayloadPoolSize = 16;
    PTO2TaskPayload slot_payload_pool_[kSlotPayloadPoolSize];
    int slot_payload_pool_idx_ = 0;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = PTO2SchedulerState::reserve_layout(sched_arena);
        ASSERT_NE(sched_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, sched_arena, sm_handle->header));
        sched.wire_arena_pointers(layout, sched_arena);
        orch.set_scheduler(&sched);
    }

    void TearDown() override {
        sched.destroy();
        sched_arena.release();
        sm_arena.release();
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
        PTO2TaskPayload &slot_pl = slot_payload_pool_[slot_payload_pool_idx_++ % kSlotPayloadPoolSize];
        memset(&slot_pl, 0, sizeof(slot_pl));
        slot.payload = &slot_pl;
    }

    void publish_no_fanin(PTO2TaskSlotState &slot) {
        slot.fanin_count = 1;
        slot.fanin_refcount.store(1, std::memory_order_release);
        orch.mark_dep_pool_position(slot);
        sched.push_ready_routed(&slot);
    }

    void wire_fanin(PTO2TaskSlotState &slot, int32_t wfanin) {
        auto &rss = sched.ring_sched_states[slot.ring_id];
        bool ok = rss.dep_pool.ensure_space(*rss.ring, wfanin);
        if (ok) {
            orch.wire_fanin_task(slot, wfanin);
        }
        ASSERT_TRUE(ok);
    }
};

// =============================================================================
// Orch-side publish: no fanin (independent task)
// =============================================================================

TEST_F(WiringTest, NoFaninTaskBecomesReady) {
    // A task with 0 actual fanins should immediately be pushed to ready queue
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    publish_no_fanin(task_slot);

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
// Orch-side wiring: with fanin, all producers already completed (early-finished)
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

    wire_fanin(task_slot, 2);

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
// Orch-side wiring: with fanin, producers still pending (task NOT ready)
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

    wire_fanin(task_slot, 2);

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
// Orch-side wiring: mixed early-finished and pending producers
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

    wire_fanin(task_slot, 3);

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
// Early-dispatch seed (direct-only): a consumer becomes an early-dispatch
// candidate ONLY when every producer is codegen-flagged (allow_early_resolve,
// now a slot_state field) and fully published or pre-completed. A single
// unflagged producer leaves dispatch_fanin short forever. Auto-chain inheritance
// was removed, so flagged-ness is the producer's own static hint — never
// propagated down a chain.
// =============================================================================

TEST_F(WiringTest, WireTaskAllFlaggedPrecompletedSeedsDispatchFanin) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer_slots[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    for (int i = 0; i < 2; i++) {
        init_slot(producer_slots[i], PTO2_TASK_COMPLETED, 1, 2);
        producer_slots[i].allow_early_resolve = true;  // codegen-flagged
    }

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producer_slots[0];
    payload.fanin_inline_slot_states[1] = &producer_slots[1];
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);

    // Every producer flagged + pre-completed -> seeded to fanin_actual_count, so
    // the consumer is already an early-dispatch candidate at wiring time.
    EXPECT_EQ(payload.dispatch_fanin.load(), payload.fanin_actual_count);
}

TEST_F(WiringTest, WireTaskUnflaggedPrecompletedProducerDoesNotSeed) {
    // Regression: an unflagged producer already complete at wiring must NOT seed
    // dispatch_fanin. It never dispatches, so it can never be the flagged-and-
    // dispatched contributor the candidate compare expects; seeding it would let
    // the consumer become an early-dispatch candidate it should stay off.
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(producer, PTO2_TASK_COMPLETED, 1, 1);
    producer.allow_early_resolve = false;  // unflagged

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 1;
    payload.fanin_inline_slot_states[0] = &producer;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 1);

    EXPECT_EQ(payload.dispatch_fanin.load(), 0);  // NOT seeded (direct-only)
    // Readiness is unaffected: early_finished(1) + 1 == fanin_count(2) -> ready.
    EXPECT_GE(task_slot.fanin_refcount.load(), task_slot.fanin_count);
}

TEST_F(WiringTest, WireTaskOneUnflaggedProducerDisqualifiesSeed) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producers[2];
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(producers[0], PTO2_TASK_COMPLETED, 1, 2);
    producers[0].allow_early_resolve = true;  // flagged
    init_slot(producers[1], PTO2_TASK_COMPLETED, 1, 2);
    producers[1].allow_early_resolve = false;  // one unflagged -> disqualifies

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &producers[0];
    payload.fanin_inline_slot_states[1] = &producers[1];
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);

    EXPECT_EQ(payload.dispatch_fanin.load(), 0);  // disqualified: seed stays 0
}

TEST_F(WiringTest, EarlyDispatchWaitsForAllProducerBlocksPublished) {
    // A flagged, still-pending producer seeds nothing at wiring (not
    // pre-completed); only publishing every logical block bumps the consumer
    // to fanin_actual_count and makes it an early-dispatch candidate.
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState producer;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(producer, PTO2_TASK_PENDING, 1, 1);
    producer.allow_early_resolve = true;
    producer.logical_block_num = 3;

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 1;
    payload.fanin_inline_slot_states[0] = &producer;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 1);
    EXPECT_EQ(payload.dispatch_fanin.load(), 0);  // pending -> nothing seeded yet

    sched.record_published_blocks(producer, 2);
    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(payload.dispatch_fanin.load(), 0);
    EXPECT_EQ(producer.payload->dispatch_propagated.load(), 0);

    sched.record_published_blocks(producer, 1);
    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(payload.dispatch_fanin.load(), payload.fanin_actual_count);
    EXPECT_EQ(producer.payload->dispatch_propagated.load(), 1);

    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(payload.dispatch_fanin.load(), payload.fanin_actual_count);
}

TEST_F(WiringTest, ConcurrentBlockRangeClaimsDoNotOverlap) {
    alignas(64) PTO2TaskSlotState task_slot;
    init_slot(task_slot, PTO2_TASK_PENDING, 1, 1);
    task_slot.logical_block_num = 8;

    struct ClaimedRange {
        int32_t start = -1;
        int32_t count = 0;
    } ranges[2];
    std::atomic<int32_t> ready{0};
    std::atomic<bool> start{false};

    auto claim = [&](int32_t index) {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {}
        ranges[index].count = task_slot.claim_block_range(task_slot.logical_block_num, 5, ranges[index].start);
    };

    std::thread first(claim, 0);
    std::thread second(claim, 1);
    while (ready.load(std::memory_order_acquire) != 2) {}
    start.store(true, std::memory_order_release);
    first.join();
    second.join();

    ClaimedRange *lower = ranges[0].start < ranges[1].start ? &ranges[0] : &ranges[1];
    ClaimedRange *upper = lower == &ranges[0] ? &ranges[1] : &ranges[0];
    EXPECT_EQ(lower->start, 0);
    EXPECT_EQ(lower->count, 5);
    EXPECT_EQ(upper->start, 5);
    EXPECT_EQ(upper->count, 3);
    EXPECT_EQ(lower->start + lower->count, upper->start);
    EXPECT_EQ(task_slot.next_block_idx.load(std::memory_order_relaxed), task_slot.logical_block_num);
}

TEST_F(WiringTest, PartialStagedReleaseRoutesRemainderToReadyQueue) {
    alignas(64) PTO2TaskSlotState consumer;
    init_slot(consumer, PTO2_TASK_PENDING, 1, 1);
    consumer.logical_block_num = 5;
    consumer.next_block_idx.store(2, std::memory_order_relaxed);
    consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);
    consumer.payload->staged_core_mask[0].store(1, std::memory_order_relaxed);

    EXPECT_TRUE(sched.route_ready_once(consumer));
    EXPECT_EQ(consumer.payload->early_dispatch_state.load(), PTO2_EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(consumer.next_block_idx.load(), 2);

    PTO2ResourceShape shape = consumer.active_mask.to_shape();
    EXPECT_EQ(sched.ready_queues[static_cast<int32_t>(shape)].pop(), &consumer);

    int32_t remaining_start = -1;
    EXPECT_EQ(consumer.claim_block_range(consumer.logical_block_num, 5, remaining_start), 3);
    EXPECT_EQ(remaining_start, 2);
}

TEST_F(WiringTest, EarlyDispatchDoorbellBitsHaveOneOwner) {
    constexpr uint64_t all_bits = 0b1111;
    constexpr uint64_t late_bits = 0b1010;

    std::atomic<uint64_t> release_first{all_bits};
    uint64_t release_owned = PTO2SchedulerState::claim_all_staged_doorbell_bits(release_first);
    uint64_t late_owned = PTO2SchedulerState::claim_late_staged_doorbell_bits(release_first, late_bits);
    EXPECT_EQ(release_owned, all_bits);
    EXPECT_EQ(late_owned, 0);

    std::atomic<uint64_t> late_first{all_bits};
    late_owned = PTO2SchedulerState::claim_late_staged_doorbell_bits(late_first, late_bits);
    release_owned = PTO2SchedulerState::claim_all_staged_doorbell_bits(late_first);
    EXPECT_EQ(late_owned, late_bits);
    EXPECT_EQ(release_owned, all_bits & ~late_bits);
    EXPECT_EQ(release_owned & late_owned, 0);
    EXPECT_EQ(release_owned | late_owned, all_bits);
    EXPECT_EQ(late_first.load(std::memory_order_acquire), 0);

    std::atomic<uint64_t> published_after_release{0};
    release_owned = PTO2SchedulerState::claim_all_staged_doorbell_bits(published_after_release);
    published_after_release.fetch_or(late_bits, std::memory_order_seq_cst);
    late_owned = PTO2SchedulerState::claim_late_staged_doorbell_bits(published_after_release, late_bits);
    EXPECT_EQ(release_owned, 0);
    EXPECT_EQ(late_owned, late_bits);
    EXPECT_EQ(published_after_release.load(std::memory_order_acquire), 0);
}

TEST_F(WiringTest, EarlyDispatchClaimStaysGatedAfterRelease) {
    EXPECT_TRUE(PTO2SchedulerState::should_gate_early_dispatch(true, PTO2_EARLY_DISPATCH_DISPATCHED));
    EXPECT_TRUE(PTO2SchedulerState::should_gate_early_dispatch(false, PTO2_EARLY_DISPATCH_STAGING));
    EXPECT_FALSE(PTO2SchedulerState::should_gate_early_dispatch(false, PTO2_EARLY_DISPATCH_DISPATCHED));
}

TEST_F(WiringTest, EarlyDispatchLaunchHasSingleOwner) {
    alignas(64) PTO2TaskPayload payload{};
    std::atomic<bool> start{false};
    bool won[2] = {false, false};

    std::thread contenders[2];
    for (int i = 0; i < 2; i++) {
        contenders[i] = std::thread([&, i] {
            while (!start.load(std::memory_order_acquire)) {}
            won[i] = PTO2SchedulerState::try_claim_early_dispatch_launch(payload);
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &contender : contenders)
        contender.join();

    EXPECT_NE(won[0], won[1]);
    EXPECT_EQ(payload.early_dispatch_launch_state.load(), PTO2_EARLY_DISPATCH_LAUNCH_RINGING);
}

TEST_F(WiringTest, EarlyDispatchFanoutWaitsForDoorbellPass) {
    alignas(64) PTO2TaskSlotState producer, consumer;
    init_slot(producer, PTO2_TASK_PENDING, 1, 1);
    init_slot(consumer, PTO2_TASK_PENDING, 1, 1);

    producer.allow_early_resolve = true;
    producer.payload->published_block_count.store(1, std::memory_order_relaxed);
    consumer.payload->fanin_actual_count = 1;

    PTO2DepListEntry dep{};
    dep.slot_state = &consumer;
    producer.fanout_head = &dep;

    producer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);
    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(consumer.payload->dispatch_fanin.load(), 0);
    EXPECT_EQ(producer.payload->dispatch_propagated.load(), 0);

    producer.payload->early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_RINGING);
    producer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_DISPATCHED, std::memory_order_release);
    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(consumer.payload->dispatch_fanin.load(), 0);
    EXPECT_EQ(producer.payload->dispatch_propagated.load(), 0);

    producer.payload->early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_seq_cst);
    sched.propagate_dispatch_fanin(producer);
    EXPECT_EQ(consumer.payload->dispatch_fanin.load(), 1);
    EXPECT_EQ(producer.payload->dispatch_propagated.load(), 1);
}

TEST_F(WiringTest, LateStagerRingsCapturedDoorbellAfterTableReuse) {
    constexpr int core_id = 3;
    constexpr uint64_t captured_addr = 0x12340000;
    constexpr uint64_t reused_addr = 0x56780000;
    constexpr uint32_t captured_token = 7;
    constexpr uint32_t reused_token = 9;

    reset_test_reg_stub();
    sched.early_dispatch_doorbell_table[core_id].addr = reused_addr;
    sched.early_dispatch_doorbell_table[core_id].token = reused_token;

    uint64_t claimed = 1ULL << core_id;
    EXPECT_TRUE(PTO2SchedulerState::ring_claimed_local_doorbell(claimed, core_id, captured_addr, captured_token));
    EXPECT_EQ(get_test_reg_stub_base_addr(), captured_addr);
    EXPECT_EQ(get_test_reg_stub_value(), (static_cast<uint64_t>(captured_token) << 32) | captured_token);
}

TEST_F(WiringTest, EarlyDispatchReleaseConsumesDoorbellMask) {
    constexpr int core_id = 5;
    constexpr uint64_t reg_addr = 0x98760000;
    constexpr uint32_t token = 11;
    alignas(64) PTO2TaskSlotState task;
    init_slot(task, PTO2_TASK_PENDING, 1, 1);
    task.allow_early_resolve = true;
    task.next_block_idx.store(1, std::memory_order_relaxed);
    task.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);
    task.payload->staged_core_mask[0].store(1ULL << core_id, std::memory_order_relaxed);
    sched.early_dispatch_doorbell_table[core_id].addr = reg_addr;
    sched.early_dispatch_doorbell_table[core_id].token = token;
    bool released_before =
        task.payload->early_dispatch_state.load(std::memory_order_seq_cst) == PTO2_EARLY_DISPATCH_DISPATCHED;

    reset_test_reg_stub();
    EXPECT_FALSE(released_before);
    EXPECT_TRUE(sched.try_early_dispatch_release(task));
    EXPECT_EQ(task.payload->early_dispatch_state.load(), PTO2_EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(task.payload->early_dispatch_launch_state.load(), PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE);
    EXPECT_EQ(task.payload->staged_core_mask[0].load(), 0);
    EXPECT_EQ(get_test_reg_stub_base_addr(), reg_addr);
    EXPECT_EQ(get_test_reg_stub_value(), (static_cast<uint64_t>(token) << 32) | token);
    EXPECT_EQ(task.payload->dispatch_propagated.load(), 0);

    sched.record_published_blocks(task, 1);
    // The staging path must retry even though its earlier state snapshot was
    // STAGING; release already missed this final publication count.
    sched.propagate_dispatch_fanin(task);
    EXPECT_EQ(task.payload->dispatch_propagated.load(), 1);

    sched.early_dispatch_doorbell_table[core_id].addr = 0x11110000;
    sched.early_dispatch_doorbell_table[core_id].token = 12;
    reset_test_reg_stub();
    EXPECT_TRUE(sched.try_early_dispatch_release(task));
    EXPECT_EQ(get_test_reg_stub_base_addr(), 0);
    EXPECT_EQ(get_test_reg_stub_value(), 0);
}

TEST_F(WiringTest, SyncStartDoorbellPassHasOneOwner) {
    PTO2TaskPayload payload{};

    for (int iteration = 0; iteration < 1000; iteration++) {
        payload.early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        std::atomic<bool> start{false};
        bool first_won = false;
        bool second_won = false;

        std::thread first([&] {
            while (!start.load(std::memory_order_acquire)) {}
            first_won = PTO2SchedulerState::try_claim_early_dispatch_launch(payload);
        });
        std::thread second([&] {
            while (!start.load(std::memory_order_acquire)) {}
            second_won = PTO2SchedulerState::try_claim_early_dispatch_launch(payload);
        });

        start.store(true, std::memory_order_release);
        first.join();
        second.join();

        EXPECT_NE(first_won, second_won);
        EXPECT_EQ(
            payload.early_dispatch_launch_state.load(std::memory_order_acquire), PTO2_EARLY_DISPATCH_LAUNCH_RINGING
        );
    }
}

TEST_F(WiringTest, SyncStartDrainFinalizeRetriesProducerFirstRendezvous) {
    alignas(64) PTO2TaskSlotState sync_consumer, downstream;
    init_slot(sync_consumer, PTO2_TASK_PENDING, 1, 1);
    init_slot(downstream, PTO2_TASK_PENDING, 1, 1);

    sync_consumer.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
    sync_consumer.active_mask.set_sync_start();
    sync_consumer.allow_early_resolve = true;
    sync_consumer.logical_block_num = 2;
    sync_consumer.next_block_idx.store(2, std::memory_order_relaxed);
    sched.record_published_blocks(sync_consumer, sync_consumer.logical_block_num);
    sync_consumer.payload->staged_core_mask[0].store(0b11, std::memory_order_relaxed);
    sync_consumer.payload->running_slot_count.store(0, std::memory_order_relaxed);
    sync_consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    downstream.payload->fanin_actual_count = 1;
    PTO2DepListEntry dep{};
    dep.slot_state = &downstream;
    sync_consumer.fanout_head = &dep;

    // Producer release wins before drain publishes its running-slot seed. With no pending
    // promotions, drain finalize is the only remaining rendezvous retry.
    EXPECT_TRUE(sched.try_early_dispatch_release(sync_consumer));
    EXPECT_EQ(sync_consumer.payload->early_dispatch_state.load(), PTO2_EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(sync_consumer.payload->early_dispatch_launch_state.load(), PTO2_EARLY_DISPATCH_LAUNCH_NONE);
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 0);

    sync_consumer.payload->running_slot_count.store(2, std::memory_order_seq_cst);
    EXPECT_TRUE(sched.retry_sync_start_rendezvous_after_drain(sync_consumer));
    EXPECT_EQ(sync_consumer.payload->early_dispatch_launch_state.load(), PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE);
    EXPECT_EQ(sync_consumer.payload->dispatch_propagated.load(), 1);
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);

    EXPECT_FALSE(sched.retry_sync_start_rendezvous_after_drain(sync_consumer));
    EXPECT_EQ(downstream.payload->dispatch_fanin.load(), 1);
}

TEST_F(WiringTest, ArmedEarlySyncDrainOwnsFinalReadyRoute) {
    alignas(64) PTO2TaskSlotState sync_consumer;
    init_slot(sync_consumer, PTO2_TASK_PENDING, 1, 1);
    sync_consumer.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
    sync_consumer.active_mask.set_sync_start();
    sync_consumer.logical_block_num = 2;
    sync_consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(PTO2SchedulerState::try_claim_early_sync_drain(*sync_consumer.payload));
    PTO2SchedulerState::mark_early_sync_drain_armed(*sync_consumer.payload);
    EXPECT_TRUE(sched.release_fanin_and_check_ready(sync_consumer));
    auto shape = static_cast<int32_t>(sync_consumer.active_mask.to_shape());
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), nullptr);
    EXPECT_EQ(sync_consumer.payload->early_dispatch_state.load(), PTO2_EARLY_DISPATCH_DISPATCHED);
    EXPECT_EQ(
        sync_consumer.payload->early_sync_drain_state.load(),
        PTO2_EARLY_SYNC_DRAIN_OWNER | PTO2_EARLY_SYNC_DRAIN_ARMED | PTO2_EARLY_SYNC_DRAIN_READY
    );
}

TEST_F(WiringTest, ArmedEarlySyncDrainKeepsEveryStagerGatedAfterReady) {
    alignas(64) PTO2TaskPayload payload{};
    payload.early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(PTO2SchedulerState::try_claim_early_sync_drain(payload));
    PTO2SchedulerState::mark_early_sync_drain_armed(payload);

    std::atomic<bool> start{false};
    std::atomic<int> before_ready{0};
    std::atomic<bool> ready_published{false};
    bool gated[4] = {false, false, false, false};
    std::thread stagers[4];
    for (int i = 0; i < 4; i++) {
        stagers[i] = std::thread([&, i] {
            while (!start.load(std::memory_order_acquire)) {}
            if (i >= 2) {
                while (!ready_published.load(std::memory_order_acquire)) {}
            }
            bool force_gate = PTO2SchedulerState::owns_early_sync_drain(payload);
            gated[i] = PTO2SchedulerState::should_gate_early_dispatch(
                force_gate, payload.early_dispatch_state.load(std::memory_order_relaxed)
            );
            if (i < 2) before_ready.fetch_add(1, std::memory_order_release);
        });
    }
    start.store(true, std::memory_order_release);
    while (before_ready.load(std::memory_order_acquire) != 2) {}
    payload.early_dispatch_state.store(PTO2_EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst);
    bool ready_has_owner = PTO2SchedulerState::publish_ready_to_early_sync_drain(payload);
    ready_published.store(true, std::memory_order_release);
    for (auto &stager : stagers)
        stager.join();

    EXPECT_TRUE(ready_has_owner);
    for (bool was_gated : gated)
        EXPECT_TRUE(was_gated);

    alignas(64) PTO2TaskPayload normal_ready{};
    normal_ready.early_dispatch_state.store(PTO2_EARLY_DISPATCH_DISPATCHED, std::memory_order_relaxed);
    EXPECT_FALSE(PTO2SchedulerState::publish_ready_to_early_sync_drain(normal_ready));
    EXPECT_FALSE(PTO2SchedulerState::owns_early_sync_drain(normal_ready));
    EXPECT_FALSE(
        PTO2SchedulerState::should_gate_early_dispatch(
            PTO2SchedulerState::owns_early_sync_drain(normal_ready),
            normal_ready.early_dispatch_state.load(std::memory_order_relaxed)
        )
    );
}

TEST_F(WiringTest, DrainFinishBetweenReleasePhasesRetainsOwner) {
    alignas(64) PTO2TaskSlotState sync_consumer;
    init_slot(sync_consumer, PTO2_TASK_PENDING, 1, 1);
    sync_consumer.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
    sync_consumer.active_mask.set_sync_start();
    sync_consumer.logical_block_num = 2;
    sync_consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(PTO2SchedulerState::try_claim_early_sync_drain(*sync_consumer.payload));
    PTO2SchedulerState::mark_early_sync_drain_armed(*sync_consumer.payload);

    // READY publication may lag the STAGING-to-DISPATCHED release transition.
    EXPECT_FALSE(sched.try_early_dispatch_release(sync_consumer));
    sync_consumer.next_block_idx.store(sync_consumer.logical_block_num, std::memory_order_seq_cst);
    PTO2SchedulerState::finish_early_sync_drain(*sync_consumer.payload);

    EXPECT_TRUE(PTO2SchedulerState::publish_ready_to_early_sync_drain(*sync_consumer.payload));
    EXPECT_EQ(
        sync_consumer.payload->early_sync_drain_state.load(),
        PTO2_EARLY_SYNC_DRAIN_OWNER | PTO2_EARLY_SYNC_DRAIN_ARMED | PTO2_EARLY_SYNC_DRAIN_READY |
            PTO2_EARLY_SYNC_DRAIN_COMPLETE
    );
}

TEST_F(WiringTest, CancelledEarlySyncDrainRoutesProducerRelease) {
    alignas(64) PTO2TaskSlotState sync_consumer;
    init_slot(sync_consumer, PTO2_TASK_PENDING, 1, 1);
    sync_consumer.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
    sync_consumer.active_mask.set_sync_start();
    sync_consumer.logical_block_num = 2;
    sync_consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(PTO2SchedulerState::try_claim_early_sync_drain(*sync_consumer.payload));
    sched.cancel_early_sync_drain(sync_consumer);
    EXPECT_TRUE(sched.release_fanin_and_check_ready(sync_consumer));

    auto shape = static_cast<int32_t>(sync_consumer.active_mask.to_shape());
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), &sync_consumer);
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), nullptr);
    EXPECT_EQ(sched.early_sync_start_queue.pop(), &sync_consumer);
}

TEST_F(WiringTest, ProducerReleaseTransfersReadyRouteToCancellingDrain) {
    alignas(64) PTO2TaskSlotState sync_consumer;
    init_slot(sync_consumer, PTO2_TASK_PENDING, 1, 1);
    sync_consumer.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIV0);
    sync_consumer.active_mask.set_sync_start();
    sync_consumer.logical_block_num = 2;
    sync_consumer.payload->early_dispatch_state.store(PTO2_EARLY_DISPATCH_STAGING, std::memory_order_relaxed);

    ASSERT_TRUE(PTO2SchedulerState::try_claim_early_sync_drain(*sync_consumer.payload));
    EXPECT_TRUE(sched.release_fanin_and_check_ready(sync_consumer));
    auto shape = static_cast<int32_t>(sync_consumer.active_mask.to_shape());
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), nullptr);

    sched.cancel_early_sync_drain(sync_consumer);
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), &sync_consumer);
    EXPECT_EQ(sched.ready_sync_queues[shape].pop(), nullptr);
    EXPECT_EQ(sched.early_sync_start_queue.pop(), nullptr);
}

TEST_F(WiringTest, EarlyDispatchBlockedByUnflaggedProducer) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState p_flagged, q_unflagged;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(p_flagged, PTO2_TASK_PENDING, 1, 1);
    p_flagged.allow_early_resolve = true;
    init_slot(q_unflagged, PTO2_TASK_PENDING, 1, 1);
    q_unflagged.allow_early_resolve = false;

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &p_flagged;
    payload.fanin_inline_slot_states[1] = &q_unflagged;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);
    EXPECT_EQ(payload.dispatch_fanin.load(), 0);  // q unflagged -> disqualified seed

    sched.record_published_blocks(p_flagged, p_flagged.logical_block_num);
    sched.propagate_dispatch_fanin(p_flagged);    // bumps to 1
    sched.propagate_dispatch_fanin(q_unflagged);  // gate returns, no bump
    EXPECT_EQ(payload.dispatch_fanin.load(), 1);  // never reaches 2 -> not a candidate
}

TEST_F(WiringTest, UnflaggedProducerDoesNotPropagate) {
    // Auto-chain removed: an unflagged producer never propagates, so its
    // consumers' dispatch_fanin stays untouched even after it dispatches, and the
    // once-guard is not even consumed (the gate returns first).
    alignas(64) PTO2TaskSlotState producer, consumer;
    alignas(64) PTO2TaskPayload prod_payload, cons_payload;
    memset(&prod_payload, 0, sizeof(prod_payload));
    memset(&cons_payload, 0, sizeof(cons_payload));

    init_slot(producer, PTO2_TASK_PENDING, 1, 1);
    producer.allow_early_resolve = false;  // unflagged
    producer.payload = &prod_payload;

    init_slot(consumer, PTO2_TASK_PENDING, 1, 1);
    consumer.payload = &cons_payload;
    cons_payload.fanin_actual_count = 1;

    PTO2DepListEntry dep{};
    dep.slot_state = &consumer;
    dep.next = nullptr;
    producer.fanout_head = &dep;

    sched.propagate_dispatch_fanin(producer);

    EXPECT_EQ(cons_payload.dispatch_fanin.load(), 0);       // no bump
    EXPECT_EQ(prod_payload.dispatch_propagated.load(), 0);  // gate returned before once-guard
}

TEST_F(WiringTest, FlaggedPrecompletedCreatorTransparentToEarlyDispatch) {
    // Models the alloc-creator case: a consumer depends on a pre-completed,
    // FLAGGED buffer creator (alloc_tensors flags its inline-completed task) plus
    // a flagged, still-pending compute producer. The creator must be transparent
    // (seeded), not a disqualifier: once the compute producer dispatches, the
    // consumer reaches fanin_actual_count and becomes an early-dispatch candidate.
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskSlotState creator, compute;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(creator, PTO2_TASK_COMPLETED, 1, 1);
    creator.allow_early_resolve = true;  // alloc creator: flagged -> transparent
    init_slot(compute, PTO2_TASK_PENDING, 1, 1);
    compute.allow_early_resolve = true;  // flagged compute producer

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 2;
    payload.fanin_inline_slot_states[0] = &creator;
    payload.fanin_inline_slot_states[1] = &compute;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    wire_fanin(task_slot, 2);
    EXPECT_EQ(payload.dispatch_fanin.load(), 1);  // creator seeded (transparent), not disqualified

    sched.record_published_blocks(compute, compute.logical_block_num);
    sched.propagate_dispatch_fanin(compute);                               // compute dispatches -> bumps to 2
    EXPECT_EQ(payload.dispatch_fanin.load(), payload.fanin_actual_count);  // candidate
}

// =============================================================================
// on_task_complete: notifies consumers via fanout chain
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

    sched.on_task_complete(producer);

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

    // After reset_for_reuse: fanout_count=PTO2_FANOUT_SCOPE_BIT (bit31 owning-scope
    // ref, 0 consumers), fanin_refcount=0, etc.
    EXPECT_EQ(slot.fanout_count, PTO2_FANOUT_SCOPE_BIT);
    EXPECT_EQ(slot.fanin_refcount.load(), 0);
    EXPECT_EQ(slot.fanout_refcount.load(), 0);
    EXPECT_EQ(slot.completed_subtasks.load(), 0);
    EXPECT_EQ(slot.fanout_head, nullptr);
}

TEST_F(WiringTest, NoEdgePublishRecordsDepPoolMark) {
    alignas(64) PTO2TaskSlotState task_slot;
    alignas(64) PTO2TaskPayload payload;
    memset(&payload, 0, sizeof(payload));
    PTO2TaskDescriptor desc{};

    init_slot(task_slot, PTO2_TASK_PENDING, 0, 1);
    payload.fanin_actual_count = 0;
    task_slot.payload = &payload;
    task_slot.task = &desc;

    auto &rss = sched.ring_sched_states[0];
    int32_t before_top = rss.dep_pool.top;
    publish_no_fanin(task_slot);
    EXPECT_EQ(task_slot.dep_pool_mark, before_top);
}
