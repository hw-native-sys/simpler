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
 * Under incremental activation a graph node may reach the ready queue before the
 * whole GRAPH task is materialized, so a producer can complete while a later
 * consumer is still being registered. Scene tests hit that interleaving only
 * probabilistically; these host-side tests force it, exercising the exact path
 * (graph_first_unmet_producer re-reading task_state) that keeps the consumer from
 * being lost on a producer's already-drained wake list.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"

class GraphActivationTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2SchedulerState sched{};
    PTO2SchedulerLayout sched_layout{};

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
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

    // One GraphExecution node whose slot is a routable single-block KERNEL/AIC
    // task in the given completion state, with its payload wired the way
    // materialization leaves it for the wake/route path.
    static void init_graph_node(GraphNodeStorage &node, int32_t node_index, PTO2TaskState state) {
        memset(&node, 0, sizeof(GraphNodeStorage));
        node.slot.task_state.store(state);
        node.slot.graph_node_index = node_index;
        node.slot.active_mask = ActiveMask(PTO2_SUBTASK_MASK_AIC);
        node.slot.task_kind = TaskKind::KERNEL;
        node.slot.total_required_subtasks = 1;
        node.slot.logical_block_num = 1;
        node.slot.payload.set(&node.payload);
    }
};

// A consumer registered after its only producer has completed and drained (head
// == SENTINEL) reaches graph_first_unmet_producer, which reads task_state and
// routes it — it is never lost on the closed wake list.
TEST_F(GraphActivationTest, WakeRoutesConsumerWhenProducerCompletedBeforeRegister) {
    auto nodes = std::make_unique<GraphNodeStorage[]>(2);
    init_graph_node(nodes[0], 0, PTO2_TASK_COMPLETED);       // producer, already completed
    init_graph_node(nodes[1], 1, PTO2_TASK_PENDING);         // consumer of node 0
    nodes[0].slot.wake_list_head.store(WAKE_LIST_SENTINEL);  // its wake list already drained

    std::vector<uint32_t> fanin_offsets{0, 0, 1};  // node 0 is a root; node 1 <- {0}
    std::vector<uint16_t> fanin_indices{0};
    GraphExecution exec{};
    exec.nodes = exec.node_storage = nodes.get();
    exec.fanin_offsets = fanin_offsets.data();
    exec.fanin_indices = fanin_indices.data();

    sched.register_graph_wake(exec, &nodes[0].slot, &nodes[1].slot);

    PTO2TaskSlotState *out[2];
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, PTO2ResourceShape::AIC, out, 2), 1)
        << "consumer must route to ready, not hang on the SENTINEL wake list";
    EXPECT_EQ(out[0], &nodes[1].slot);
}

// graph_incremental_publish routes a node whose producers are all COMPLETED at
// publish time, and wake-chains a node with a still-pending producer so it
// routes exactly once that producer completes and drains its wake list.
TEST_F(GraphActivationTest, IncrementalPublishRoutesCompletedDepsAndWakeChainsPending) {
    auto nodes = std::make_unique<GraphNodeStorage[]>(4);
    init_graph_node(nodes[0], 0, PTO2_TASK_COMPLETED);  // root, completed
    init_graph_node(nodes[1], 1, PTO2_TASK_PENDING);    // root, pending
    init_graph_node(nodes[2], 2, PTO2_TASK_PENDING);    // consumer of node 0 (completed)
    init_graph_node(nodes[3], 3, PTO2_TASK_PENDING);    // consumer of node 1 (pending)

    std::vector<uint32_t> fanin_offsets{0, 0, 0, 1, 2};  // node 2 <- {0}, node 3 <- {1}
    std::vector<uint16_t> fanin_indices{0, 1};
    GraphExecution exec{};
    exec.nodes = exec.node_storage = nodes.get();
    exec.fanin_offsets = fanin_offsets.data();
    exec.fanin_indices = fanin_indices.data();

    sched.graph_incremental_publish(exec, 0, 4);
    EXPECT_EQ(exec.published_nodes.load(), 4);

    PTO2TaskSlotState *out[4];
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, PTO2ResourceShape::AIC, out, 4), 1)
        << "only the consumer whose producers are all COMPLETED routes at publish time";
    EXPECT_EQ(out[0], &nodes[2].slot);

    nodes[1].slot.task_state.store(PTO2_TASK_COMPLETED);
    sched.drain_graph_wake_list(exec, nodes[1].slot);
    ASSERT_EQ(sched.get_ready_tasks_batch(sched.ready_queues, PTO2ResourceShape::AIC, out, 4), 1)
        << "the wake-chained consumer must route once its pending producer completes";
    EXPECT_EQ(out[0], &nodes[3].slot);
}

// Incremental activation dispatches a node before the graph reaches ACTIVE, so
// complete_task must accept a node completion while the graph is MATERIALIZING or
// PREPARED, and reject it only for SUBMITTED (not yet bound) or COMPLETED
// (already retired).
TEST_F(GraphActivationTest, CompleteTaskAcceptsCompletionBeforeActive) {
    GraphDefinition definition{};
    auto complete_in_state = [&](GraphExecutionState state) {
        auto node = std::make_unique<GraphNodeStorage[]>(1);
        memset(node.get(), 0, sizeof(GraphNodeStorage));
        node[0].slot.task_kind = TaskKind::GRAPH_NODE;
        node[0].slot.graph_node_index = 0;
        node[0].slot.total_required_subtasks = 1;
        node[0].slot.payload.set(&node[0].payload);

        GraphExecution exec{};
        exec.definition = &definition;
        exec.nodes = exec.node_storage = node.get();
        exec.node_count = 1;
        exec.remaining_nodes.store(1);
        exec.outer_slot = nullptr;
        graph_execution_set_state(exec, state);
        node[0].slot.graph_context = &exec;
#if SIMPLER_SCHED_PROFILING
        return sched.complete_task(node[0].slot, 0).error_code;
#else
        return sched.complete_task(node[0].slot).error_code;
#endif
    };

    EXPECT_EQ(complete_in_state(GraphExecutionState::MATERIALIZING), PTO2_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::PREPARED), PTO2_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::ACTIVE), PTO2_ERROR_NONE);
    EXPECT_EQ(complete_in_state(GraphExecutionState::SUBMITTED), PTO2_ERROR_INVALID_ARGS);
    EXPECT_EQ(complete_in_state(GraphExecutionState::COMPLETED), PTO2_ERROR_INVALID_ARGS);
}
