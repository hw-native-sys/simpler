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
 * Deterministic tests for incremental graph activation.
 *
 * Under incremental activation a sub-task may reach the ready queue before the
 * whole GRAPH task is materialized, so a producer can complete while a later
 * consumer is still being registered. Scene tests hit that interleaving only
 * probabilistically; these host-side tests force it, exercising the exact path
 * (graph_first_unmet_producer re-reading the execution's task_states) that keeps the
 * consumer from being lost on a producer's already-drained wake list.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"

class GraphActivationTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    SchedulerState sched{};
    SchedulerLayout sched_layout{};

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        // Same order the AICPU boots in: the slot arrays are not part of the
        // uploaded image, so nothing can push until they carry their ramp.
        sched.seed_queue_slots();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // One sub-task whose slot is a routable single-block KERNEL/AIC
    // task in the given completion state, with its payload wired the way
    // materialization leaves it for the wake/route path.
    static void
    init_sub_task(ChipTaskStorage &task, std::atomic<ChipTaskState> *states, int32_t task_index, ChipTaskState state) {
        memset(&task, 0, sizeof(ChipTaskStorage));
        states[task_index].store(state);
        task.slot.sub_task_local_id = task_index;
        task.slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        task.slot.task_kind = TaskKind::KERNEL;
        task.slot.total_required_subtasks = 1;
        task.slot.logical_block_num = 1;
    }
};

// A consumer registered after its only producer has completed and drained (head
// == SENTINEL) reaches graph_first_unmet_producer, which reads the state array and
// routes it — it is never lost on the closed wake list.
TEST_F(GraphActivationTest, WakeRoutesConsumerWhenProducerCompletedBeforeRegister) {
    auto tasks = std::make_unique<ChipTaskStorage[]>(2);
    auto states = std::make_unique<std::atomic<ChipTaskState>[]>(2);
    init_sub_task(tasks[0], states.get(), 0, CHIP_TASK_COMPLETED);  // producer, already completed
    init_sub_task(tasks[1], states.get(), 1, CHIP_TASK_PENDING);    // consumer of task 0
    tasks[0].slot.wake_list_head.store(WAKE_LIST_SENTINEL);         // its wake list already drained

    std::vector<int32_t> fanin_offsets{0, 0, 1};  // task 0 is a root; task 1 <- {0}
    std::vector<uint16_t> fanin_indices{0};
    GraphExecution exec{};
    exec.tasks = exec.task_storage = tasks.get();
    exec.task_states = states.get();
    exec.fanin_offsets = fanin_offsets.data();
    exec.fanin_indices = fanin_indices.data();

    sched.register_graph_wake(exec, &tasks[0].slot, &tasks[1].slot);

    ChipTaskSlotState *out[2];
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 2), 1)
        << "consumer must route to ready, not hang on the SENTINEL wake list";
    EXPECT_EQ(out[0], &tasks[1].slot);
}

// graph_incremental_publish routes a sub-task whose producers are all COMPLETED at
// publish time, and wake-chains one with a still-pending producer so it
// routes exactly once that producer completes and drains its wake list.
TEST_F(GraphActivationTest, IncrementalPublishRoutesCompletedDepsAndWakeChainsPending) {
    auto tasks = std::make_unique<ChipTaskStorage[]>(4);
    auto states = std::make_unique<std::atomic<ChipTaskState>[]>(4);
    init_sub_task(tasks[0], states.get(), 0, CHIP_TASK_COMPLETED);  // root, completed
    init_sub_task(tasks[1], states.get(), 1, CHIP_TASK_PENDING);    // root, pending
    init_sub_task(tasks[2], states.get(), 2, CHIP_TASK_PENDING);    // consumer of task 0 (completed)
    init_sub_task(tasks[3], states.get(), 3, CHIP_TASK_PENDING);    // consumer of task 1 (pending)

    std::vector<int32_t> fanin_offsets{0, 0, 0, 1, 2};  // task 2 <- {0}, task 3 <- {1}
    std::vector<uint16_t> fanin_indices{0, 1};
    GraphExecution exec{};
    exec.tasks = exec.task_storage = tasks.get();
    exec.task_states = states.get();
    exec.fanin_offsets = fanin_offsets.data();
    exec.fanin_indices = fanin_indices.data();

    sched.graph_incremental_publish(exec, 0, 4);
    EXPECT_EQ(exec.published_tasks.load(), 4);

    ChipTaskSlotState *out[4];
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 4), 1)
        << "only the consumer whose producers are all COMPLETED routes at publish time";
    EXPECT_EQ(out[0], &tasks[2].slot);

    states[1].store(CHIP_TASK_COMPLETED);
    sched.drain_graph_wake_list(exec, tasks[1].slot);
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, ResourceShape::AIC, out, 4), 1)
        << "the wake-chained consumer must route once its pending producer completes";
    EXPECT_EQ(out[0], &tasks[3].slot);
}

// Incremental activation dispatches a sub-task before the graph reaches ACTIVE, so
// complete_task must accept such a completion while the graph is MATERIALIZING or
// PREPARED, and reject it only for SUBMITTED (not yet bound) or COMPLETED
// (already retired).
TEST_F(GraphActivationTest, CompleteTaskAcceptsCompletionBeforeActive) {
    GraphDefinition definition{};
    auto complete_in_state = [&](GraphExecutionState state) {
        auto task = std::make_unique<ChipTaskStorage[]>(1);
        auto states = std::make_unique<std::atomic<ChipTaskState>[]>(1);
        memset(task.get(), 0, sizeof(ChipTaskStorage));
        task[0].slot.sub_task_local_id = 0;
        task[0].slot.total_required_subtasks = 1;

        GraphExecution exec{};
        exec.definition = &definition;
        exec.tasks = exec.task_storage = task.get();
        exec.task_states = states.get();
        exec.task_count = 1;
        exec.remaining_tasks.store(1);
        exec.outer_slot = nullptr;
        graph_execution_set_state(exec, state);
        task[0].slot.graph_context = &exec;
#if SIMPLER_SCHED_PROFILING
        return sched.complete_task(task[0].slot, 0).error_code;
#else
        return sched.complete_task(task[0].slot).error_code;
#endif
    };

    EXPECT_EQ(complete_in_state(GraphExecutionState::MATERIALIZING), SIMPLER_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::PREPARED), SIMPLER_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::ACTIVE), SIMPLER_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::SUBMITTED), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(complete_in_state(GraphExecutionState::COMPLETED), SIMPLER_ERROR_INVALID_ARGS);
}

// The outer Graph task completes as a task of the run, not into its execution's
// counters. It is the one slot where a non-null graph_context does NOT mean "in a
// Graph body": before localize swaps in the GraphExecution the shell's context is
// the shared GraphDefinition, so the `task_kind == GRAPH` half of complete_task's
// predicate is the only thing keeping the two apart. Drop it and this slot's
// Definition gets read as an execution -- a silent static_cast onto another
// struct's layout, no fault and no error code.
TEST_F(GraphActivationTest, CompleteTaskTakesTheOrdinaryPathForTheOuterGraphTask) {
    GraphDefinition definition{};
    // A whole storage entry, not a bare slot state: a slot reaches its descriptor by
    // ChipTaskStorage's layout, so one on its own would resolve outside itself.
    ChipTaskStorage outer{};
    outer.task.task_id = TaskId::make_global(0);

    ChipTaskSlotState &slot = outer.slot;
    slot.task_kind = TaskKind::GRAPH;
    slot.graph_context = &definition;

#if SIMPLER_SCHED_PROFILING
    const SchedulerState::TaskCompletionOutcome outcome = sched.complete_task(slot, 0);
#else
    const SchedulerState::TaskCompletionOutcome outcome = sched.complete_task(slot);
#endif

    EXPECT_EQ(outcome.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(outcome.stream_tasks_completed, 1) << "the outer Graph task is one completed task of the run";
    EXPECT_TRUE(sm_handle->header->tasks.is_completed(slot.to_descriptor().task_id.local_id()));
}
