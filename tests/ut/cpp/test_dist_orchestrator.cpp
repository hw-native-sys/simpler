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

#include "dist_orchestrator.h"
#include "dist_ring.h"
#include "dist_scope.h"
#include "dist_tensormap.h"
#include "dist_types.h"

// ---------------------------------------------------------------------------
// Fixture: wires the Orchestrator components together (no Scheduler thread)
// ---------------------------------------------------------------------------

struct OrchestratorFixture : public ::testing::Test {
    static constexpr int32_t N = DIST_TASK_WINDOW_SIZE;

    std::unique_ptr<DistTaskSlotState[]> slots;
    DistTensorMap tm;
    DistRing ring;
    DistScope scope;
    DistReadyQueue rq;
    DistOrchestrator orch;

    void SetUp() override {
        slots = std::make_unique<DistTaskSlotState[]>(N);
        ring.init(N);
        orch.init(&tm, &ring, &scope, &rq, slots.get(), N);
    }

    void TearDown() override { ring.shutdown(); }

    // Submit a CHIP task with the given input/output specs.
    DistSubmitResult submit_chip(const std::vector<DistInputSpec> &inputs, const std::vector<DistOutputSpec> &outputs) {
        WorkerPayload p;
        p.worker_type = WorkerType::CHIP;
        return orch.submit(WorkerType::CHIP, p, inputs, outputs);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(OrchestratorFixture, IndependentTaskIsImmediatelyReady) {
    auto res = submit_chip({}, {{64}});
    EXPECT_NE(res.task_slot, DIST_INVALID_SLOT);
    ASSERT_EQ(res.outputs.size(), 1u);
    EXPECT_NE(res.outputs[0].ptr, nullptr);

    DistTaskSlot slot;
    EXPECT_TRUE(rq.try_pop(slot));
    EXPECT_EQ(slot, res.task_slot);
    EXPECT_EQ(slots[slot].state.load(), TaskState::READY);
}

TEST_F(OrchestratorFixture, DependentTaskIsPending) {
    // Task A produces a buffer
    auto a = submit_chip({}, {{128}});
    DistTaskSlot a_slot;
    rq.try_pop(a_slot);  // drain ready queue

    uint64_t a_out = reinterpret_cast<uint64_t>(a.outputs[0].ptr);

    // Task B depends on A's output
    auto b = submit_chip({{a_out}}, {{64}});
    EXPECT_EQ(slots[b.task_slot].state.load(), TaskState::PENDING);
    EXPECT_EQ(slots[b.task_slot].fanin_count, 1);

    DistTaskSlot extra;
    EXPECT_FALSE(rq.try_pop(extra));  // B should NOT be in ready queue
}

TEST_F(OrchestratorFixture, TensorMapTracksProducer) {
    auto a = submit_chip({}, {{256}});
    DistTaskSlot drain_slot;
    rq.try_pop(drain_slot);

    uint64_t key = reinterpret_cast<uint64_t>(a.outputs[0].ptr);
    EXPECT_EQ(tm.lookup(key), a.task_slot);
}

TEST_F(OrchestratorFixture, OnConsumedCleansUpTensorMap) {
    auto a = submit_chip({}, {{64}});
    DistTaskSlot slot;
    rq.try_pop(slot);

    uint64_t key = reinterpret_cast<uint64_t>(a.outputs[0].ptr);
    EXPECT_EQ(tm.lookup(key), slot);

    // Simulate task completion + consumed
    slots[slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    orch.on_consumed(slot);

    EXPECT_EQ(tm.lookup(key), DIST_INVALID_SLOT);
    EXPECT_EQ(slots[slot].state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, ScopeRegistersAndReleasesRef) {
    orch.scope_begin();
    auto a = submit_chip({}, {{64}});
    DistTaskSlot slot;
    rq.try_pop(slot);

    // Inside scope: fanout_total should be 1 (scope ref)
    {
        std::lock_guard<std::mutex> lk(slots[slot].fanout_mu);
        EXPECT_EQ(slots[slot].fanout_total, 1);
    }

    // scope_end releases the scope ref; if task is completed it becomes consumed
    slots[slot].state.store(TaskState::COMPLETED, std::memory_order_relaxed);
    orch.scope_end();

    // After scope_end the consumed callback should have fired
    EXPECT_EQ(slots[slot].state.load(), TaskState::CONSUMED);
}

TEST_F(OrchestratorFixture, MultipleOutputsAllocated) {
    auto res = submit_chip({}, {{32}, {64}, {128}});
    ASSERT_EQ(res.outputs.size(), 3u);
    EXPECT_EQ(res.outputs[0].size, 32u);
    EXPECT_EQ(res.outputs[1].size, 64u);
    EXPECT_EQ(res.outputs[2].size, 128u);
    for (const auto &o : res.outputs)
        EXPECT_NE(o.ptr, nullptr);
}
