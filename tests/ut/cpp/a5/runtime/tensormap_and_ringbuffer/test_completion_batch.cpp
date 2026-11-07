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

#include <functional>

#include "runtime.h"  // Handshake
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"

namespace {
// Only the two hardware boundaries are replaced. Classification, slab handling,
// mailbox publication, credit counting and fanout callbacks are production code.
std::function<void(const volatile uint32_t *)> on_read;
std::function<void()> on_doorbell;

struct Task {
    TaskDescriptor descriptor{};
    TaskPayload payload{};
    ChipTaskSlotState slot{};
    explicit Task(int16_t total, int id = 0) {
        descriptor.task_id = TaskId::make(0, id);
        slot.task = &descriptor;
        slot.payload = &payload;
        slot.total_required_subtasks = total;
        slot.task_state.store(CHIP_TASK_PENDING);
    }
};

struct PollResult {
    int32_t tasks{0}, physical{0}, release_count{0};
    bool progress{false};
    ChipTaskSlotState *released[16]{};
};
}  // namespace

uint32_t reg_load_acquire(const volatile uint32_t *ptr) {
    if (on_read) on_read(ptr);
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

volatile uint32_t *get_reg_ptr(uint64_t addr, RegId) {
    if (on_doorbell) on_doorbell();
    return reinterpret_cast<volatile uint32_t *>(addr);
}

// Named as SchedulerContext's friend, so every probe reads its private state.
class CompletionBatchTestPeer : public testing::Test {
protected:
    SchedulerContext context{};
    RuntimeContext runtime{};
    SharedMemoryHeader header{};
    AICoreCompletionMailbox mailbox{};
    uint32_t cond[9]{};
    Handshake handshake[9]{};
    Task gate{1, 60};  // callback_on's consumer: resolved by a producer, never dispatched
    DepListEntry gate_edge{};
    uint64_t gate_doorbell{0};

    void SetUp() override {
        on_read = nullptr;
        on_doorbell = nullptr;
        runtime.aicore_mailbox = &mailbox;
        runtime.scheduler.sm_header = &header;
        context.sched_ = &runtime.scheduler;
        context.rt_ = &runtime;
        tracker().init(3);
        for (int c = 0; c < 3; ++c)
            tracker().set_cluster(c, c * 3, c * 3 + 1, c * 3 + 2);
    }
    CoreTracker &tracker() { return context.core_trackers_[0]; }
    DeferredCompletionSlab &slab(int core, int token = 2) { return context.deferred_slab_per_core_[core][token & 1]; }
    void running(int id, Task &task, int token = 2) {
        CoreExecState &core = context.core_exec_states_[id];
        core.running_slot_state = &task.slot;
        core.running_reg_task_id = token;
        core.running_subslot = SubtaskSlot::AIC;
        core.pending_reg_task_id = AICPU_TASK_INVALID;
        core.cond_ptr = &cond[id];
        cond[id] = static_cast<uint32_t>(MAKE_FIN_VALUE(token));
        tracker().change_core_state(id);
        tracker().set_pending_occupied(id);
        tracker().set_running_groupable(id, task.slot.total_required_subtasks > 1);
    }
    // A pending FIN completes this core's pending slot and then its running slot.
    void pending(int id, Task &task, int token = 3) {
        CoreExecState &core = context.core_exec_states_[id];
        core.pending_slot_state = &task.slot;
        core.pending_reg_task_id = token;
        core.pending_subslot = SubtaskSlot::AIC;
        cond[id] = static_cast<uint32_t>(MAKE_FIN_VALUE(token));
        tracker().set_pending_groupable(id, task.slot.total_required_subtasks > 1);
    }
    // A CONDITION is published per deferred slab entry before that event's credit is counted.
    int registered() { return static_cast<int>(mailbox.head.load() - mailbox.tail.load()); }
    // The one production callback running inside the apply phase: a retiring `producer`
    // resolves this gated consumer and rings its doorbell, between two events of a batch.
    void callback_on(Task &producer, int core = 8) {
        gate.payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
        gate.payload.staged_core_mask[0].store(1ULL << core);
        gate.slot.fanin_count = 1;
        gate.slot.next_block_idx.store(1);
        runtime.scheduler.early_dispatch_doorbell_table[core].addr = reinterpret_cast<uint64_t>(&gate_doorbell);
        gate_edge.slot_state = &gate.slot;
        producer.slot.fanout_head = &gate_edge;
    }
    PollResult poll() {
        PollResult r;
        context.check_running_cores_for_completion(
            0, handshake, r.tasks, r.physical, r.progress, r.released, r.release_count
        );
        return r;
    }
};
using CompletionBatch = CompletionBatchTestPeer;

TEST_F(CompletionBatch, CreditsStayUnpublishedUntilTheGroupsLastEvent) {
    Task grouped(2), singleton(1, 1);
    running(0, grouped);
    running(1, singleton);
    running(2, grouped);
    slab(0).count = 1;  // each grouped event registers one deferred condition
    slab(2).count = 1;
    callback_on(singleton);
    int callbacks = 0;
    on_doorbell = [&] {
        ++callbacks;
        // The group's first event registered its condition and still holds its credit.
        EXPECT_EQ(registered(), 1);
        EXPECT_EQ(grouped.slot.completed_subtasks.load(), 0);
    };
    auto result = poll();
    ASSERT_EQ(callbacks, 1);  // non-vacuity: the mid-batch probe ran
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    ASSERT_EQ(result.release_count, 1);
    EXPECT_EQ(result.released[0], &singleton.slot);
}

TEST_F(CompletionBatch, DistinctTasksInOneBatchNeverShareACredit) {
    Task first(2), second(2, 1), single(1, 2);
    running(0, single);  // a singleton and a group's event on one core's two slots
    pending(0, first);
    running(1, first);
    cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(9));  // stale id: unmatched, no event
    running(2, first);
    pending(2, second);
    running(3, second);
    auto result = poll();
    EXPECT_EQ(first.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(second.slot.completed_subtasks.load(), 2);
    EXPECT_EQ(single.slot.completed_subtasks.load(), 1);
    EXPECT_EQ(result.release_count, 3);
}

TEST_F(CompletionBatch, ErrorAtTheLastEventPublishesTheCreditsHeldSoFar) {
    Task task(3);
    for (int i = 0; i < 3; ++i)
        running(i, task);
    slab(2).error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    auto result = poll();
    EXPECT_EQ(task.slot.completed_subtasks.load(), 2);  // the healthy events, not the failed one
    EXPECT_FALSE(task.slot.is_completion_flag_set());
    EXPECT_EQ(result.tasks, 0);  // nothing retired
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_F(CompletionBatch, AGroupNeverSpansTwoBatches) {
    Task task(9);
    for (int i = 0; i < 9; ++i)
        running(i, task);
    int reads = 0;
    on_read = [&](const volatile uint32_t *) {
        // Eight observations per barrier: the ninth opens a batch, after eight publish.
        EXPECT_EQ(task.slot.completed_subtasks.load(), reads < 8 ? 0 : 8);
        ++reads;
    };
    auto result = poll();
    ASSERT_EQ(reads, 9);  // non-vacuity: every observation was probed
    EXPECT_EQ(result.physical, 9);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 9);
    EXPECT_TRUE(task.slot.is_completion_flag_set());
}
