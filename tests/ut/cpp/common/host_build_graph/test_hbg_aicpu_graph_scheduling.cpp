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
#include "scheduler/scheduler.h"
#include "graph_execution.h"
#include "hbg_graph_image_test_support.h"

TEST(GraphExecutionReplay, PublishedTaskCompletesBeforeRemainingTasksMaterialize) {
    std::array<uint8_t, 64> boundary{};
    const std::vector<std::byte> definition = make_test_definition(0x1234, reinterpret_cast<uint64_t>(boundary.data()));
    const TestDefinitionObject definition_object(definition);
    OuterHeap heap(definition, 0xAA);
    GraphExecution *execution =
        heap.initialize_execution(definition_object, reinterpret_cast<uint64_t>(boundary.data()), 17);
    ASSERT_NE(execution, nullptr);
    ChipTaskStorage *const tasks = execution->task_storage;
    ASSERT_NE(tasks, nullptr);
    EXPECT_EQ(execution->constructed_tasks, 0);
    EXPECT_EQ(graph_execution_phase(*execution), GraphExecutionPhase::SUBMITTED);

    ChipTaskStorage outer{};
    outer.task.task_id = TaskId::make_global(7);
    outer.task.packed_buffer_base = heap.base();
    outer.task.packed_buffer_end = heap.end();
    outer.slot.task_kind = TaskKind::GRAPH;
    outer.slot.graph_context = execution;
    execution->outer_slot = &outer.slot;

    SharedMemoryHeader header{};
    std::array<std::atomic<ChipTaskState>, 8> stream_states{};
    header.tasks.task_states = stream_states.data();
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    scheduler.task_view.tasks = &header.tasks;
    ChipReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0);
    queue_slots[1].sequence.store(1);
    ChipReadyQueue &queue = scheduler.ready_queues[static_cast<int32_t>(ResourceShape::AIC)];
    queue.slots = queue_slots;
    queue.capacity = 2;
    queue.mask = 1;
    EXPECT_EQ(scheduler.signal_graph_external_ready(outer.slot), 0);

    execution->materialize_busy.store(1);
    EXPECT_EQ(graph_execution_materialize_slice(outer.slot, *execution, heap.image(), 1), GraphMaterializeResult::BUSY);
    EXPECT_EQ(execution->materialized_tasks, 0);
    execution->materialize_busy.store(0);

    ASSERT_EQ(scheduler.prepare_graph_task(outer.slot, heap.image(), 1), GraphMaterializeResult::PENDING);
    EXPECT_EQ(graph_execution_phase(*execution), GraphExecutionPhase::SUBMITTED);
    EXPECT_EQ(execution->materialized_tasks, 1);
    EXPECT_EQ(execution->constructed_tasks, 1);
    EXPECT_EQ(execution->published_tasks.load(), 1);
    ChipTaskSlotState *ready[2]{};
    ASSERT_EQ(queue.pop_batch(ready, 2), 1);
    ASSERT_EQ(ready[0], &tasks[0].slot);
    const auto first = scheduler.complete_task(*ready[0]);
    EXPECT_EQ(first.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(first.stream_tasks_completed, 0);
    EXPECT_EQ(graph_execution_phase(*execution), GraphExecutionPhase::SUBMITTED);
    EXPECT_EQ(execution->remaining_tasks.load(), 1);
    EXPECT_EQ(execution->retired_tasks.load(), 1);
    EXPECT_TRUE(execution->is_completed(0));
    EXPECT_FALSE(header.tasks.is_completed(7));

    ASSERT_EQ(scheduler.prepare_graph_task(outer.slot, heap.image(), 1), GraphMaterializeResult::MATERIALIZED);
    EXPECT_EQ(graph_execution_phase(*execution), GraphExecutionPhase::MATERIALIZED);
    EXPECT_TRUE(graph_execution_external_ready(*execution));
    EXPECT_EQ(execution->task_storage, tasks);
    EXPECT_EQ(execution->materialized_tasks, 2);
    EXPECT_EQ(execution->constructed_tasks, 2);
    EXPECT_TRUE(execution->is_completed(0));
    EXPECT_EQ(tasks[1].payload.scalar_data()[0], 18U);

    EXPECT_EQ(
        graph_execution_materialize_slice(outer.slot, *execution, heap.image(), 1), GraphMaterializeResult::MATERIALIZED
    );
    EXPECT_EQ(execution->constructed_tasks, 2);
    EXPECT_TRUE(execution->is_completed(0));

    ASSERT_EQ(queue.pop_batch(ready, 2), 1);
    ASSERT_EQ(ready[0], &tasks[1].slot);
    const auto last = scheduler.complete_task(*ready[0]);
    EXPECT_EQ(last.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(last.stream_tasks_completed, 1);
    EXPECT_EQ(graph_execution_phase(*execution), GraphExecutionPhase::COMPLETED);
    EXPECT_EQ(execution->remaining_tasks.load(), 0);
    EXPECT_EQ(execution->retired_tasks.load(), 2);
    EXPECT_TRUE(execution->is_completed(1));
    EXPECT_TRUE(header.tasks.is_completed(7));
    EXPECT_EQ(queue.pop_batch(ready, 2), 0);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
}

TEST(GraphExecutionPhase, ShellReadinessOpensTheGateInsteadOfQueueing) {
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    ChipReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.dummy_ready_queue.slots = queue_slots;
    scheduler.dummy_ready_queue.capacity = 2;
    scheduler.dummy_ready_queue.mask = 1;
    scheduler.dummy_ready_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.dummy_ready_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    GraphExecution executions[3]{};
    ChipTaskSlotState graph_slots[3]{};
    for (int32_t i = 0; i < 3; ++i) {
        graph_slots[i].task_kind = TaskKind::GRAPH;
        graph_slots[i].graph_context = &executions[i];
        executions[i].outer_slot = &graph_slots[i];
    }

    scheduler.push_ready_routed(&graph_slots[0]);
    scheduler.push_ready_routed(&graph_slots[1]);
    scheduler.push_ready_routed(&graph_slots[2]);

    for (const GraphExecution &execution : executions) {
        EXPECT_TRUE(graph_execution_external_ready(execution));
    }
    EXPECT_EQ(scheduler.dummy_ready_queue.enqueue_pos.load(std::memory_order_acquire), uint64_t{0});
    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_NONE);
}

TEST(GraphExecutionErrors, GraphPrepareQueueOverflowIsReported) {
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    ChipReadyQueueSlot queue_slots[2]{};
    queue_slots[0].sequence.store(0, std::memory_order_relaxed);
    queue_slots[1].sequence.store(1, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.slots = queue_slots;
    scheduler.graph_prepare_queue.capacity = 2;
    scheduler.graph_prepare_queue.mask = 1;
    scheduler.graph_prepare_queue.enqueue_pos.store(0, std::memory_order_relaxed);
    scheduler.graph_prepare_queue.dequeue_pos.store(0, std::memory_order_relaxed);
    ChipTaskSlotState graph_slots[3]{};

    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[0], TaskId::make_global(10), 3));
    EXPECT_TRUE(scheduler.push_graph_prepare(&graph_slots[1], TaskId::make_global(11), 3));
    EXPECT_FALSE(scheduler.push_graph_prepare(&graph_slots[2], TaskId::make_global(12), 3));

    EXPECT_EQ(header.sched_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(std::memory_order_acquire), 3);
    EXPECT_EQ(header.sched_error_bitmap.load(std::memory_order_acquire), 1U << 3);
}

TEST(GraphExecutionErrors, InvalidSubTaskCompletionIsReported) {
    SchedulerState scheduler{};
    ChipTaskSlotState slot{};
    // Membership without a usable execution: graph_context names one, but neither its
    // definition nor its sub-task array was ever bound.
    GraphExecution execution{};
    slot.graph_context = &execution;

    const SchedulerState::TaskCompletionOutcome outcome = scheduler.complete_task(slot);

    EXPECT_EQ(outcome.error_code, SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(outcome.stream_tasks_completed, 0);
}

TEST(GraphExecutionProgress, SubTaskResolutionIsNotAHostCompletion) {
    SchedulerState scheduler{};
    ChipTaskStorage task{};
    std::atomic<ChipTaskState> states[1]{};
    GraphExecution execution{};
    // Any non-zero offset: the completion path only asks whether this execution
    // names a Definition, not what is at that offset.
    execution.definition_offset = sizeof(GraphImageHeader);
    execution.task_storage = &task;
    execution.task_states = states;
    execution.task_count = 1;
    execution.remaining_tasks.store(1, std::memory_order_relaxed);
    graph_execution_set_phase(execution, GraphExecutionPhase::MATERIALIZED, std::memory_order_relaxed);
    task.slot.graph_context = &execution;
    task.slot.sub_task_local_id = 0;

    AsyncWaitList wait_list{};
    wait_list.entries[0].slot_state = &task.slot;
    wait_list.entries[0].task_token = TaskId::make_global(1);
    wait_list.entries[0].normal_done = true;
    wait_list.count = 1;

    const AsyncPollResult result = wait_list.poll_and_complete<false>(nullptr, &scheduler);

    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(result.resolved, 1);
    EXPECT_EQ(result.completed, 0);
}
