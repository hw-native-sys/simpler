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

#include "chip_call_config.h"
#include "dist_ring.h"
#include "dist_orchestrator.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"
#include "task_args.h"

// ---------------------------------------------------------------------------
// Fixture: wires the Orchestrator components together (no Scheduler thread)
// ---------------------------------------------------------------------------

struct OrchestratorFixture : public ::testing::Test {
    static constexpr int32_t N = DIST_TASK_WINDOW_SIZE;

    std::unique_ptr<DistTaskSlotState[]> slots;
    DistTensorMap tm;
    DistRing allocator;
    DistScope scope;
    DistReadyQueue rq;
    DistOrchestrator orch;
    ChipCallConfig cfg;

    void SetUp() override {
        slots = std::make_unique<DistTaskSlotState[]>(N);
        allocator.init(N, /*heap_bytes=*/1ULL << 20);
        orch.init(&tm, &allocator, &scope, &rq, slots.get(), N);
    }

    void TearDown() override { allocator.shutdown(); }

    // Helper: build a TaskArgs whose only tensor has the given (data, tag).
    static TaskArgs single_tensor_args(uint64_t data_ptr, TensorArgType tag) {
        TaskArgs a;
        ContinuousTensor t{};
        t.data = data_ptr;
        t.ndims = 1;
        t.shapes[0] = 1;
        t.dtype = DataType::UINT8;
        a.add_tensor(t, tag);
        return a;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(OrchestratorFixture, IndependentTaskIsImmediatelyReady) {
    auto a = single_tensor_args(0xCAFE, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level(/*callable=*/0xDEAD, a, cfg);
    EXPECT_NE(res.task_slot, DIST_INVALID_SLOT);

    DistTaskSlot slot;
    EXPECT_TRUE(rq.try_pop(slot));
    EXPECT_EQ(slot, res.task_slot);
    EXPECT_EQ(slots[slot].state.load(), TaskState::READY);
}

TEST_F(OrchestratorFixture, DependentTaskIsPending) {
    // Task A produces an OUTPUT at key 0xBEEF
    auto args_a = single_tensor_args(0xBEEF, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xDEAD, args_a, cfg);
    DistTaskSlot a_slot;
    rq.try_pop(a_slot);

    // Task B reads INPUT at the same key — depends on A
    auto args_b = single_tensor_args(0xBEEF, TensorArgType::INPUT);
    auto b = orch.submit_next_level(0xDEAD, args_b, cfg);
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::PENDING);
    EXPECT_EQ(slots[b.task_slot].fanin_count, 1);

    DistTaskSlot extra;
    EXPECT_FALSE(rq.try_pop(extra));  // B should NOT be in ready queue
}

TEST_F(OrchestratorFixture, TensorMapTracksProducer) {
    auto args_a = single_tensor_args(0x1234, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xDEAD, args_a, cfg);
    DistTaskSlot drain_slot;
    rq.try_pop(drain_slot);

    EXPECT_EQ(tm.lookup(0x1234), a.task_slot);
}

TEST_F(OrchestratorFixture, OnConsumedCleansUpTensorMap) {
    auto args_a = single_tensor_args(0x42, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xDEAD, args_a, cfg);
    DistTaskSlot slot;
    rq.try_pop(slot);

    EXPECT_EQ(tm.lookup(0x42), slot);

    slots[slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    orch.on_consumed(slot);

    EXPECT_EQ(tm.lookup(0x42), DIST_INVALID_SLOT);
    EXPECT_EQ(slots[slot].state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, ScopeRegistersAndReleasesRef) {
    orch.scope_begin();
    auto args_a = single_tensor_args(0x77, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xDEAD, args_a, cfg);
    DistTaskSlot slot;
    rq.try_pop(slot);

    {
        std::lock_guard<std::mutex> lk(slots[slot].fanout_mu);
        EXPECT_EQ(slots[slot].fanout_total, 1);
    }

    // Simulate the completion path that would run if this test drove the
    // full scheduler: state -> COMPLETED + the self try_consume that
    // on_task_complete would normally fire (bumps fanout_released by 1).
    // Without this simulated self-release, the `>= total + 1` threshold in
    // release_ref / try_consume cannot be met from scope_end alone.
    slots[slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    slots[slot].fanout_released.fetch_add(1, std::memory_order_relaxed);
    orch.scope_end();

    EXPECT_EQ(slots[slot].state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, NoDepTagSkipsDependencyTracking) {
    // OUTPUT-tagged input registers a producer
    auto args_a = single_tensor_args(0xAAAA, TensorArgType::OUTPUT);
    auto a = orch.submit_next_level(0xDEAD, args_a, cfg);
    DistTaskSlot drain_slot;
    rq.try_pop(drain_slot);

    // Second task references same key but tagged NO_DEP — should be independent
    auto args_b = single_tensor_args(0xAAAA, TensorArgType::NO_DEP);
    auto b = orch.submit_next_level(0xDEAD, args_b, cfg);
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::READY);
    EXPECT_EQ(slots[b.task_slot].fanin_count, 0);
}

TEST_F(OrchestratorFixture, GroupTaskHasAllChipStorageEntries) {
    TaskArgs a0 = single_tensor_args(0xA0, TensorArgType::OUTPUT);
    TaskArgs a1 = single_tensor_args(0xA1, TensorArgType::OUTPUT);
    auto res = orch.submit_next_level_group(0xDEAD, {a0, a1}, cfg);

    EXPECT_NE(res.task_slot, DIST_INVALID_SLOT);
    EXPECT_TRUE(slots[res.task_slot].is_group());
    EXPECT_EQ(slots[res.task_slot].group_size(), 2);
    EXPECT_EQ(slots[res.task_slot].chip_storage_list.size(), 2u);

    // Both keys registered as producers for the group slot.
    EXPECT_EQ(tm.lookup(0xA0), res.task_slot);
    EXPECT_EQ(tm.lookup(0xA1), res.task_slot);
}

TEST_F(OrchestratorFixture, OutputAutoAllocsFromHeapRing) {
    // An OUTPUT tensor submitted with `data == 0` is auto-allocated from
    // the HeapRing: the slot's chip_storage tensor ends up with a non-zero
    // data pointer that falls inside the allocator's mmap'd region, and
    // the TensorMap routes that pointer to the slot.
    TaskArgs args;
    ContinuousTensor t{};
    t.data = 0;
    t.ndims = 1;
    t.shapes[0] = 1024;  // 1024 * 1 byte = 1024, one aligned slab
    t.dtype = DataType::UINT8;
    args.add_tensor(t, TensorArgType::OUTPUT);

    auto res = orch.submit_next_level(0xDEAD, args, cfg);
    ASSERT_NE(res.task_slot, DIST_INVALID_SLOT);

    const auto &chip = slots[res.task_slot].chip_storage_list.front();
    uint64_t data = chip.tensor(0).data;
    ASSERT_NE(data, 0u);
    uintptr_t base = reinterpret_cast<uintptr_t>(allocator.heap_base());
    EXPECT_GE(data, base);
    EXPECT_LT(data, base + allocator.heap_size());
    EXPECT_EQ(data % DIST_HEAP_ALIGN, 0u);

    EXPECT_EQ(tm.lookup(data), res.task_slot);
}

TEST_F(OrchestratorFixture, InoutWiresCreatorAsFanin) {
    // INOUT is the only tag that pulls in the prior writer as a fanin
    // producer — matching L2's pto_orchestrator.cpp Step B where only
    // INPUT / INOUT do tensor_map.lookup. Users who want a WaW dep on
    // the alloc-slot (so its HeapRing slab stays live while they write)
    // must tag the buffer INOUT.
    auto creator_args = single_tensor_args(0xFEED, TensorArgType::OUTPUT);
    auto creator = orch.submit_next_level(0xDEAD, creator_args, cfg);
    DistTaskSlot drain;
    rq.try_pop(drain);
    // Mark the creator COMPLETED so the new submit mimics the alloc-slot
    // path (COMPLETED producer with non-zero fanout).
    slots[creator.task_slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);

    auto writer_args = single_tensor_args(0xFEED, TensorArgType::INOUT);
    auto writer = orch.submit_next_level(0xDEAD, writer_args, cfg);
    DistTaskSlot writer_slot;
    rq.try_pop(writer_slot);

    // TensorMap now points at the new writer.
    EXPECT_EQ(tm.lookup(0xFEED), writer.task_slot);
    // Writer has the creator recorded as a fanin producer (via INOUT
    // lookup) but no *live* fanin since the creator is already COMPLETED.
    EXPECT_EQ(slots[writer.task_slot].fanin_count, 0);
    ASSERT_EQ(slots[writer.task_slot].fanin_producers.size(), 1u);
    EXPECT_EQ(slots[writer.task_slot].fanin_producers[0], creator.task_slot);
    // Creator's fanout_total bumped so it waits for writer before CONSUMED.
    {
        std::lock_guard<std::mutex> lk(slots[creator.task_slot].fanout_mu);
        EXPECT_EQ(slots[creator.task_slot].fanout_total, 1);
        ASSERT_EQ(slots[creator.task_slot].fanout_consumers.size(), 1u);
        EXPECT_EQ(slots[creator.task_slot].fanout_consumers[0], writer.task_slot);
    }
}

TEST_F(OrchestratorFixture, OutputAndOutputExistingAreInsertOnly) {
    // Contrast with INOUT: plain OUTPUT and OUTPUT_EXISTING are pure
    // overwrites — insert into TensorMap, no lookup, so no fanin wire
    // on the prior writer. Matches L2 semantics for both tags. Users
    // who need creator lifetime must tag the buffer INOUT.
    struct Case {
        uint64_t key;
        TensorArgType tag;
    };
    for (Case c : {Case{0xABCD, TensorArgType::OUTPUT}, Case{0xBEEF, TensorArgType::OUTPUT_EXISTING}}) {
        auto prior_args = single_tensor_args(c.key, TensorArgType::OUTPUT);
        auto prior = orch.submit_next_level(0xDEAD, prior_args, cfg);
        DistTaskSlot drain;
        rq.try_pop(drain);
        slots[prior.task_slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);

        auto writer_args = single_tensor_args(c.key, c.tag);
        auto writer = orch.submit_next_level(0xDEAD, writer_args, cfg);

        EXPECT_EQ(tm.lookup(c.key), writer.task_slot);
        EXPECT_EQ(slots[writer.task_slot].fanin_count, 0);
        EXPECT_TRUE(slots[writer.task_slot].fanin_producers.empty()) << "tag=" << static_cast<int>(c.tag);
        {
            std::lock_guard<std::mutex> lk(slots[prior.task_slot].fanout_mu);
            EXPECT_EQ(slots[prior.task_slot].fanout_total, 0) << "tag=" << static_cast<int>(c.tag);
        }
    }
}
