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
#include <memory>
#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "runtime.h"  // Handshake
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"

namespace {
// Only hardware/profiling boundaries are replaced. Classification, slab handling,
// mailbox publication, credit counting and fanout callbacks are production code.
std::function<void(const volatile uint32_t *)> on_read;
std::function<void()> on_doorbell;
uint64_t clock_tick;
#if SIMPLER_DFX
std::function<void(int)> on_ack;
std::function<void(int, uint64_t)> on_finish;
#endif

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

uint64_t get_sys_cnt_aicpu() { return ++clock_tick; }

#if SIMPLER_DFX
void chip_swimlane_aicpu_on_aicore_ack(int core_id, int, uint32_t) {
    if (on_ack) on_ack(core_id);
}
int chip_swimlane_aicpu_complete_task(int core_id, int, uint32_t, uint64_t, uint64_t finish) {
    if (on_finish) on_finish(core_id, finish);
    return 0;
}
void chip_swimlane_aicpu_record_sched_phase(
    int, SchedPhaseKind, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, const int16_t *, const int16_t *
) {}
extern "C" bool is_pmu_enabled() { return false; }
void pmu_aicpu_complete_record(int, int, uint32_t, TaskId, uint32_t, CoreType) { ADD_FAILURE() << "PMU is disabled"; }
#endif

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
    DevicePhaseBufferStorage<PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH> timing{};

    void SetUp() override {
        on_read = nullptr;
        on_doorbell = nullptr;
        clock_tick = 100;
        reset_device_phase_buffer(&timing, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH);
        set_platform_phase_base(reinterpret_cast<uint64_t>(&timing));
#if SIMPLER_DFX
        on_ack = nullptr;
        on_finish = nullptr;
        context.chip_swimlane_level_ = ChipSwimlaneLevel::SCHEDULE_TIMING;
        context.sched_chip_swimlane_[0].chip_swimlane_enabled = true;
#endif
        runtime.aicore_mailbox = &mailbox;
        runtime.scheduler.sm_header = &header;
        context.sched_ = &runtime.scheduler;
        context.rt_ = &runtime;
        tracker().init(3);
        for (int c = 0; c < 3; ++c)
            tracker().set_cluster(c, c * 3, c * 3 + 1, c * 3 + 2);
    }
    void TearDown() override { set_platform_phase_base(0); }
    CoreTracker &tracker() { return context.core_trackers_[0]; }
    CoreExecState &core(int id) { return context.core_exec_states_[id]; }
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

TEST_F(CompletionBatch, AFullBatchCanCompleteSixteenDistinctSlots) {
    std::vector<std::unique_ptr<Task>> tasks;
    for (int i = 0; i < 16; ++i)
        tasks.push_back(std::make_unique<Task>(2, i));
    for (int i = 0; i < 8; ++i) {
        running(i, *tasks[2 * i]);
        pending(i, *tasks[2 * i + 1]);
    }
    int reads = 0;
    on_read = [&](const volatile uint32_t *) {
        ++reads;
    };
    auto result = poll();
    EXPECT_EQ(reads, 8);
    EXPECT_EQ(result.physical, 16);
    EXPECT_EQ(result.tasks, 0);
    for (const auto &task : tasks) {
        EXPECT_EQ(task->slot.completed_subtasks.load(), 1);
        EXPECT_FALSE(task->slot.is_completion_flag_set());
    }
}

TEST_F(CompletionBatch, TaggedFinishesPrecedeTheFirstCallback) {
    Task first(1), running_task(2, 1), pending_task(2, 2);
    running(0, first);
    running(1, running_task);
    pending(1, pending_task);
    running_task.slot.task_attrs.set_timing_slot(0);
    pending_task.slot.task_attrs.set_timing_slot(1);
    callback_on(first);
    int callbacks = 0;
    on_doorbell = [&] {
        ++callbacks;
        EXPECT_GT(timing.tail[0].finish_cycle, 0u);
        EXPECT_GT(timing.tail[1].finish_cycle, 0u);
        EXPECT_EQ(running_task.slot.completed_subtasks.load(), 0);
        EXPECT_EQ(pending_task.slot.completed_subtasks.load(), 0);
    };
    auto result = poll();
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(result.physical, 3);
}

TEST_F(CompletionBatch, SingletonPollingAppliesEachCompletionBeforeReadingTheNext) {
    Task first(1), second(1, 1);
    running(0, first);
    running(1, second);
    cond[0] = cond[1] = static_cast<uint32_t>(MAKE_ACK_VALUE(2));
    auto ack_result = poll();
    EXPECT_EQ(ack_result.physical, 0);
    EXPECT_EQ(clock_tick, 100u);  // ACK-only singleton observations do not stamp a FIN.
    cond[0] = cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(2));
    callback_on(first);
    int callbacks = 0, reads = 0;
    on_doorbell = [&] {
        ++callbacks;
    };
    on_read = [&](const volatile uint32_t *ptr) {
        EXPECT_EQ(callbacks, ptr == &cond[0] ? 0 : 1);
        ++reads;
    };
    auto result = poll();
    EXPECT_EQ(reads, 2);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(result.tasks, 2);
}

TEST_F(CompletionBatch, AnErrorKeepsHealthyEarlierAndLaterCredits) {
    Task grouped(3), single(1, 1);
    running(0, grouped);
    running(1, grouped);
    running(2, grouped);
    running(3, single);
    slab(1).error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    auto result = poll();
    EXPECT_EQ(grouped.slot.completed_subtasks.load(), 2);
    EXPECT_FALSE(grouped.slot.is_completion_flag_set());
    EXPECT_TRUE(single.slot.is_completion_flag_set());
    EXPECT_EQ(result.tasks, 1);
    EXPECT_EQ(result.physical, 4);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_F(CompletionBatch, AnEarlierCallbackDoesNotReclassifyThePendingGate) {
    Task first(1), later(2, 1);
    running(0, first);
    running(1, later);
    pending(1, gate);
    cond[1] = static_cast<uint32_t>(MAKE_FIN_VALUE(2));
    callback_on(first, 1);
    int callbacks = 0;
    on_doorbell = [&] {
        ++callbacks;
        EXPECT_EQ(gate.payload.early_dispatch_state.load(), EARLY_DISPATCH_DISPATCHED);
        EXPECT_EQ(later.slot.completed_subtasks.load(), 0);
    };
    auto result = poll();
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(result.physical, 2);
    EXPECT_EQ(later.slot.completed_subtasks.load(), 1);
    EXPECT_EQ(core(1).running_slot_state, &gate.slot);
    EXPECT_EQ(core(1).pending_slot_state, nullptr);
}

#if SIMPLER_DFX
TEST_F(CompletionBatch, BatchTimingPrecedesAckPublicationAndCallbacks) {
    Task first(1), grouped(2, 1);
    running(0, first);
    running(1, grouped);
    running(2, grouped);
    grouped.slot.task_attrs.set_timing_slot(0);
    callback_on(first);
    std::vector<int> order;
    std::vector<uint64_t> finishes;
    const uint64_t acquired_finish = clock_tick + 1;
    on_ack = [&](int core) {
        EXPECT_GT(timing.tail[0].finish_cycle, acquired_finish);
        order.push_back(10 + core);
    };
    on_doorbell = [&] {
        EXPECT_EQ(order, (std::vector<int>{10}));
        order.push_back(20);
        clock_tick += 1000;
    };
    on_finish = [&](int core, uint64_t finish) {
        finishes.push_back(finish);
        order.push_back(30 + core);
    };
    auto result = poll();
    EXPECT_EQ(result.physical, 3);
    EXPECT_EQ(order, (std::vector<int>{10, 20, 30, 11, 31, 12, 32}));
    EXPECT_EQ(finishes, (std::vector<uint64_t>{acquired_finish, acquired_finish, acquired_finish}));
}
#endif
