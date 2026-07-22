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
#include <cstdint>
#include <new>
#include <vector>

#include "utils/device_arena.h"
#include "pto_graph_cache.h"
#include "pto_graph_execution.h"
#include "pto_orchestrator.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"

TEST(ReplayGraphCacheKey, TracksStructureButNotDynamicArgs) {
    uint32_t shape[] = {2, 4};
    Tensor first_tensor = make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 2, DataType::FLOAT16);
    L2TaskArgs first;
    first.add_input(first_tensor);
    first.add_scalar(uint64_t{7});

    Tensor relocated_tensor = first_tensor;
    relocated_tensor.buffer.addr = 0x2000;
    L2TaskArgs relocated;
    relocated.add_input(relocated_tensor);
    relocated.add_scalar(uint64_t{7});
    uint64_t namespace_hash = PTO2_GRAPH_KEY("ut_graph_cache_key_v1");
    uint64_t key = rt_graph_make_key(namespace_hash, first);
    EXPECT_EQ(key, rt_graph_make_key(namespace_hash, relocated));

    L2TaskArgs different_scalar;
    different_scalar.add_input(first_tensor);
    different_scalar.add_scalar(uint64_t{8});
    EXPECT_EQ(key, rt_graph_make_key(namespace_hash, different_scalar));

    Tensor offset_tensor = first_tensor;
    offset_tensor.start_offset = 4;
    offset_tensor.version = 3;
    L2TaskArgs different_offset;
    different_offset.add_input(offset_tensor);
    different_offset.add_scalar(uint64_t{7});
    EXPECT_EQ(key, rt_graph_make_key(namespace_hash, different_offset));

    uint32_t different_shape[] = {4, 4};
    Tensor metadata_tensor =
        make_tensor_external(reinterpret_cast<void *>(0x1000), different_shape, 2, DataType::FLOAT16);
    L2TaskArgs different_metadata;
    different_metadata.add_input(metadata_tensor);
    different_metadata.add_scalar(uint64_t{7});
    EXPECT_NE(key, rt_graph_make_key(namespace_hash, different_metadata));
    L2TaskArgs different_direction;
    different_direction.add_inout(first_tensor);
    different_direction.add_scalar(uint64_t{7});
    EXPECT_NE(key, rt_graph_make_key(namespace_hash, different_direction));
    EXPECT_NE(key, rt_graph_make_key(PTO2_GRAPH_KEY("ut_graph_cache_key_v2"), first));
}

TEST(ReplayGraphAllocator, HeapOverflowIsFatalWithoutWrap) {
    alignas(64) uint8_t heap[256]{};
    int32_t task_slot_map[8]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(8, &task_count, task_slot_map, heap, sizeof(heap), &error);

    auto first = allocator.alloc(128);
    ASSERT_FALSE(first.failed());
    EXPECT_EQ(first.slot, first.task_id);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap) / 2);

    auto overflow = allocator.alloc(64);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_EQ(allocator.heap_top(), sizeof(heap) / 2);
    EXPECT_EQ(task_count.load(), 1);
}

TEST(ReplayGraphAllocator, MapsDenseTaskIdsAcrossPingPongArenas) {
    alignas(64) uint8_t heap[64]{};
    int32_t task_slot_map[4]{};
    std::atomic<int32_t> task_count{0};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator;
    allocator.init(4, &task_count, task_slot_map, heap, sizeof(heap), &error);

    for (int32_t expected = 0; expected < 2; expected++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed());
        EXPECT_EQ(result.task_id, expected);
        EXPECT_EQ(result.slot, expected);
    }
    EXPECT_TRUE(allocator.alloc(0).failed());
    EXPECT_EQ(error.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
    EXPECT_EQ(task_count.load(), 2);

    error.store(PTO2_ERROR_NONE);
    allocator.begin_buffer(1);
    auto third = allocator.alloc(0);
    auto fourth = allocator.alloc(0);
    ASSERT_FALSE(third.failed());
    ASSERT_FALSE(fourth.failed());
    EXPECT_EQ(third.task_id, 2);
    EXPECT_EQ(third.slot, 2);
    EXPECT_EQ(fourth.task_id, 3);
    EXPECT_EQ(fourth.slot, 3);
    EXPECT_EQ(task_slot_map[2], 2);
    EXPECT_EQ(task_slot_map[3], 3);
}

TEST(ReplayGraphDepPool, CapacityMustHoldTheCompleteGraph) {
    PTO2DepListEntry entries[4]{};
    std::atomic<int32_t> error{PTO2_ERROR_NONE};
    PTO2DepListPool pool;
    pool.init(entries, 4, &error);

    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_NE(pool.alloc(), nullptr);
    EXPECT_EQ(pool.used(), 3);
    EXPECT_EQ(pool.alloc(), nullptr);
    EXPECT_EQ(error.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_EQ(pool.used(), 3);
}

TEST(ReplayGraphExecutionPool, ReusesRetiredBlockAndResetsExecutionState) {
    pto2_graph_execution_destroy_all();

    auto *first = pto2_graph_execution_create(4);
    ASSERT_NE(first, nullptr);
    auto *first_storage = first->node_storage;
    std::atomic<int32_t> definition_refs{1};
    first->definition_refcount = &definition_refs;
    first->graph_key = 0x1234;
    first->state.store(PTO2GraphExecutionState::COMPLETED, std::memory_order_relaxed);
    first->retired_nodes.store(first->node_count, std::memory_order_relaxed);
    pto2_graph_execution_publish(first);

    pto2_graph_execution_collect_retired();
    EXPECT_EQ(definition_refs.load(), 0);

    auto *reused = pto2_graph_execution_create(2);
    ASSERT_EQ(reused, first);
    EXPECT_EQ(reused->node_storage, first_storage);
    EXPECT_EQ(reused->node_capacity, 4);
    EXPECT_EQ(reused->node_count, 2);
    EXPECT_EQ(reused->materialized_nodes, 0);
    EXPECT_EQ(reused->constructed_nodes, 0);
    EXPECT_EQ(reused->graph_key, 0);
    EXPECT_EQ(reused->definition_refcount, nullptr);
    EXPECT_EQ(reused->state.load(), PTO2GraphExecutionState::SUBMITTED);
    EXPECT_EQ(reused->remaining_nodes.load(), 2);
    EXPECT_EQ(reused->retired_nodes.load(), 0);

    pto2_graph_execution_discard(reused);
    pto2_graph_execution_destroy_all();
}

TEST(ReplayGraphExecutionPool, PrefersDefinitionAffineBlock) {
    pto2_graph_execution_destroy_all();

    constexpr uint64_t kFirstGraphKey = 0x1234;
    constexpr uint64_t kSecondGraphKey = 0x5678;
    auto *first = pto2_graph_execution_create(8, kFirstGraphKey);
    auto *second = pto2_graph_execution_create(4, kSecondGraphKey);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    for (int32_t i = 0; i < 4; ++i) {
        new (&first->node_storage[i]) PTO2GraphNodeStorage;
        new (&second->node_storage[i]) PTO2GraphNodeStorage;
    }
    first->constructed_nodes = 4;
    second->constructed_nodes = 4;
    // Simulate a capacity-8 block whose most recent execution had four nodes.
    first->node_count = 4;
    first->materialized_graph_key = kFirstGraphKey;
    first->materialized_node_count = 4;
    second->materialized_graph_key = kSecondGraphKey;
    second->materialized_node_count = 4;
    first->state.store(PTO2GraphExecutionState::COMPLETED, std::memory_order_relaxed);
    second->state.store(PTO2GraphExecutionState::COMPLETED, std::memory_order_relaxed);
    first->retired_nodes.store(first->node_count, std::memory_order_relaxed);
    second->retired_nodes.store(second->node_count, std::memory_order_relaxed);
    pto2_graph_execution_publish(first);
    pto2_graph_execution_publish(second);
    pto2_graph_execution_collect_retired();

    auto *reused = pto2_graph_execution_create(4, kFirstGraphKey);
    ASSERT_EQ(reused, first);
    EXPECT_TRUE(reused->definition_affine_reuse);
    EXPECT_EQ(reused->materialized_graph_key, kFirstGraphKey);

    pto2_graph_execution_discard(reused);
    pto2_graph_execution_destroy_all();
}

class ReplayGraphOrchestratorTest : public ::testing::Test {
protected:
    static constexpr uint64_t kTaskWindow = 16;
    static constexpr uint64_t kHeapSize = 4096;
    static constexpr int32_t kDepPoolCapacity = 64;

    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle{nullptr};
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        pto2_graph_execution_destroy_all();
        pto2_graph_definition_destroy_all();
        size_t handle_off = sm_arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
        uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(kTaskWindow);
        size_t buffer_off = sm_arena.reserve(sm_size, PTO2_ALIGN_SIZE);
        ASSERT_NE(sm_arena.commit(), nullptr);
        sm_handle = static_cast<PTO2SharedMemoryHandle *>(sm_arena.region_ptr(handle_off));
        ASSERT_TRUE(sm_handle->init(sm_arena.region_ptr(buffer_off), sm_size, kTaskWindow, kHeapSize));

        gm_heap.resize(kHeapSize);
        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, kTaskWindow, kDepPoolCapacity);
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), kHeapSize, kTaskWindow
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);
    }

    void TearDown() override {
        pto2_graph_execution_destroy_all();
        pto2_graph_definition_destroy_all();
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    void FinalizeGraphDefinition(PTO2GraphCacheStats &stats) {
        orch.graph_end(&stats);
        ASSERT_TRUE(orch.finalize_pending_graph_definition());
    }

    void PrepareGraph(PTO2TaskSlotState &outer) {
        PTO2GraphMaterializeResult result = PTO2GraphMaterializeResult::PENDING;
        int32_t slices = 0;
        while (result == PTO2GraphMaterializeResult::PENDING || result == PTO2GraphMaterializeResult::BUSY) {
            result = sched.prepare_graph_task(outer);
            ASSERT_LT(++slices, 1024);
        }
        ASSERT_EQ(result, PTO2GraphMaterializeResult::PREPARED);
    }
};

TEST_F(ReplayGraphOrchestratorTest, DefinitionStorageTracksRecordedNodesInsteadOfMaximumCapacity) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_compact_definition_v1"), graph_args);
    constexpr uint64_t callable_hash = 0xb4d37c8e5a901f62ULL;

    ASSERT_TRUE(orch.graph_begin(graph_key, graph_args, callable_hash).execute_block);
    L0TaskArgs args;
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());
    PTO2GraphCacheStats stats;
    orch.graph_end(&stats);

    EXPECT_EQ(stats.recorded, 0);
    EXPECT_EQ(pto2_graph_definition_cache_bytes(), 0u);
    ASSERT_TRUE(orch.finalize_pending_graph_definition());
    ASSERT_EQ(stats.recorded, 1);
    EXPECT_GT(pto2_graph_definition_cache_bytes(), 0u);
    // Cache storage is proportional to the live fields of recorded nodes;
    // unused max-sized tensor/scalar/fanin arrays are capture-only.
    EXPECT_LT(pto2_graph_definition_cache_bytes(), 2u * 1024u);
}

TEST_F(ReplayGraphOrchestratorTest, SubmitBuildsFaninWithPublishGate) {
    orch.begin_scope();

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId duplicate_deps[] = {producer.task_id(), producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(duplicate_deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot = sm_handle->header->get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot = sm_handle->header->get_slot_state_by_task_id(consumer.task_id().local());
    EXPECT_EQ(producer_slot.fanin_count, 1);
    EXPECT_EQ(consumer_slot.fanin_count, 2);
    EXPECT_EQ(consumer_slot.fanin_refcount.load(), 0);
    PTO2DepListEntry *fanout = producer_slot.fanout_head.load();
    ASSERT_NE(fanout, nullptr);
    EXPECT_EQ(fanout->slot_state, &consumer_slot);
    EXPECT_EQ(fanout->next, nullptr);
}

TEST_F(ReplayGraphOrchestratorTest, LiveTaskLookupRejectsReusedPhysicalSlot) {
    constexpr int32_t logical_task_id = 0;
    constexpr int32_t reused_task_id = static_cast<int32_t>(kTaskWindow);
    auto &task = sm_handle->header->get_task_by_slot(0);
    auto &slot = sm_handle->header->get_slot_state_by_slot(0);
    slot.task = &task;
    sm_handle->header->task_slot_map[logical_task_id] = 0;

    task.task_id = PTO2TaskId::make(0, logical_task_id);
    EXPECT_EQ(sm_handle->header->find_live_slot_state(task.task_id), &slot);

    task.task_id = PTO2TaskId::make(0, reused_task_id);
    EXPECT_EQ(sm_handle->header->find_live_slot_state(PTO2TaskId::make(0, logical_task_id)), nullptr);
}

TEST_F(ReplayGraphOrchestratorTest, ReplayUsesOneOuterTaskAndExpandsFrozenDag) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_frozen_task_dag_v1"), graph_args);
    constexpr uint64_t callable_hash = 0x8d5e52b41f62d9a3ULL;

    PTO2GraphScopeResult record = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_TRUE(record.execute_block);
    ASSERT_TRUE(record.recording);

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    PTO2TaskId dependency[] = {producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(dependency, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    EXPECT_EQ(stats.recorded, 1);
    ASSERT_EQ(orch.task_allocator.active_count(), 2);

    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    EXPECT_FALSE(replay.execute_block);
    EXPECT_FALSE(replay.recording);
    EXPECT_EQ(replay.task_id.local(), 2);
    ASSERT_FALSE(orch.fatal);
    ASSERT_EQ(orch.task_allocator.active_count(), 3);

    auto &outer = sm_handle->header->get_slot_state_by_task_id(2);
    ASSERT_EQ(outer.task->kind, PTO2TaskKind::GRAPH);
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    ASSERT_EQ(execution->node_count, 2);
    EXPECT_EQ(execution->nodes, nullptr);
    ASSERT_EQ(execution->topology.root_count, 1);
    EXPECT_EQ(execution->topology.root_indices[0], 0);
    ASSERT_EQ(execution->topology.edge_count, 1);
    EXPECT_EQ(execution->topology.fanin_counts[0], 0);
    EXPECT_EQ(execution->topology.fanin_counts[1], 1);
    EXPECT_EQ(execution->topology.fanout_offsets[0], 0);
    EXPECT_EQ(execution->topology.fanout_offsets[1], 1);
    EXPECT_EQ(execution->topology.fanout_offsets[2], 1);
    EXPECT_EQ(execution->topology.fanout_indices[0], 1);

    PrepareGraph(outer);
    EXPECT_EQ(sched.activate_graph_task(outer), 1);
    ASSERT_NE(execution->nodes, nullptr);
    auto &replayed_producer = execution->nodes[0].slot;
    auto &replayed_consumer = execution->nodes[1].slot;
    EXPECT_EQ(replayed_producer.task->kind, PTO2TaskKind::GRAPH_NODE);
    EXPECT_EQ(replayed_consumer.task->kind, PTO2TaskKind::GRAPH_NODE);
    EXPECT_EQ(replayed_producer.fanin_count, 0);
    EXPECT_EQ(replayed_consumer.fanin_count, 1);
    EXPECT_EQ(replayed_producer.fanout_head.load(), nullptr);

    auto *ready_producer = sched.dummy_ready_queue.pop();
    ASSERT_EQ(ready_producer, &replayed_producer);
    auto producer_completion = sched.complete_task(*ready_producer);
    EXPECT_EQ(producer_completion.stream_tasks_completed, 0);
    sched.retire_graph_node(*ready_producer);

    auto *ready_consumer = sched.dummy_ready_queue.pop();
    ASSERT_EQ(ready_consumer, &replayed_consumer);
    auto consumer_completion = sched.complete_task(*ready_consumer);
    EXPECT_EQ(consumer_completion.stream_tasks_completed, 1);
    EXPECT_EQ(outer.task_state.load(), PTO2_TASK_COMPLETED);
    sched.retire_graph_node(*ready_consumer);

    ASSERT_EQ(execution->constructed_nodes, 2);
    PTO2GraphNodeStorage *node_storage = execution->node_storage;
    pto2_graph_execution_collect_retired();
    auto *reused = pto2_graph_execution_create(1);
    ASSERT_EQ(reused, execution);
    EXPECT_EQ(reused->node_storage, node_storage);
    EXPECT_EQ(reused->constructed_nodes, 2);
    EXPECT_EQ(reused->materialized_nodes, 0);
    pto2_graph_execution_discard(reused);
}

TEST_F(ReplayGraphOrchestratorTest, SavedTopologyKeepsRecordedReadyOrder) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_saved_topological_order_v1"), graph_args);
    constexpr uint64_t callable_hash = 0x1c231466085f6d1fULL;

    ASSERT_TRUE(orch.graph_begin(graph_key, graph_args, callable_hash).execute_block);
    L0TaskArgs args;
    TaskOutputTensors first_root = orch.submit_dummy_task(args);
    TaskOutputTensors second_root = orch.submit_dummy_task(args);
    PTO2TaskId first_dependency[] = {first_root.task_id()};
    L0TaskArgs third_args;
    third_args.set_dependencies(first_dependency, 1);
    ASSERT_TRUE(orch.submit_dummy_task(third_args).task_id().is_valid());
    PTO2TaskId join_dependencies[] = {first_root.task_id(), second_root.task_id()};
    L0TaskArgs join_args;
    join_args.set_dependencies(join_dependencies, 2);
    ASSERT_TRUE(orch.submit_dummy_task(join_args).task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    PTO2GraphScopeResult submit = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(submit.execute_block);
    auto &outer = sm_handle->header->get_slot_state_by_task_id(submit.task_id.local());
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    ASSERT_EQ(execution->topology.root_count, 2);
    EXPECT_EQ(execution->topology.root_indices[0], 0);
    EXPECT_EQ(execution->topology.root_indices[1], 1);
    ASSERT_EQ(execution->topology.edge_count, 3);
    EXPECT_EQ(execution->topology.fanout_offsets[0], 0);
    EXPECT_EQ(execution->topology.fanout_offsets[1], 2);
    EXPECT_EQ(execution->topology.fanout_offsets[2], 3);
    EXPECT_EQ(execution->topology.fanout_indices[0], 2);
    EXPECT_EQ(execution->topology.fanout_indices[1], 3);
    EXPECT_EQ(execution->topology.fanout_indices[2], 3);

    PrepareGraph(outer);
    ASSERT_EQ(sched.activate_graph_task(outer), 2);
    auto *ready_first = sched.dummy_ready_queue.pop();
    auto *ready_second = sched.dummy_ready_queue.pop();
    ASSERT_EQ(ready_first, &execution->nodes[0].slot);
    ASSERT_EQ(ready_second, &execution->nodes[1].slot);
    EXPECT_EQ(sched.complete_task(*ready_first).stream_tasks_completed, 0);
    EXPECT_EQ(sched.complete_task(*ready_second).stream_tasks_completed, 0);
    auto *ready_third = sched.dummy_ready_queue.pop();
    auto *ready_join = sched.dummy_ready_queue.pop();
    ASSERT_EQ(ready_third, &execution->nodes[2].slot);
    ASSERT_EQ(ready_join, &execution->nodes[3].slot);
    EXPECT_EQ(sched.complete_task(*ready_third).stream_tasks_completed, 0);
    EXPECT_EQ(sched.complete_task(*ready_join).stream_tasks_completed, 1);

    sched.retire_graph_node(*ready_first);
    sched.retire_graph_node(*ready_second);
    sched.retire_graph_node(*ready_third);
    sched.retire_graph_node(*ready_join);
}

TEST_F(ReplayGraphOrchestratorTest, ConcurrentExecutionsShareFrozenTopologyButNotRuntimeState) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_shared_frozen_topology_v1"), graph_args);
    constexpr uint64_t callable_hash = 0x6f68e0fb5a901a11ULL;

    ASSERT_TRUE(orch.graph_begin(graph_key, graph_args, callable_hash).execute_block);
    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    PTO2TaskId dependency[] = {producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(dependency, 1);
    ASSERT_TRUE(orch.submit_dummy_task(consumer_args).task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    PTO2GraphScopeResult first_submit = orch.graph_begin(graph_key, graph_args, callable_hash);
    PTO2GraphScopeResult second_submit = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(first_submit.execute_block);
    ASSERT_FALSE(second_submit.execute_block);
    auto &first_outer = sm_handle->header->get_slot_state_by_task_id(first_submit.task_id.local());
    auto &second_outer = sm_handle->header->get_slot_state_by_task_id(second_submit.task_id.local());
    auto *first = pto2_graph_execution_from_task(*first_outer.task);
    auto *second = pto2_graph_execution_from_task(*second_outer.task);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);
    EXPECT_EQ(first->topology.fanout_offsets, second->topology.fanout_offsets);
    EXPECT_EQ(first->topology.fanout_indices, second->topology.fanout_indices);

    PrepareGraph(first_outer);
    PrepareGraph(second_outer);
    ASSERT_EQ(sched.activate_graph_task(first_outer), 1);
    ASSERT_EQ(sched.activate_graph_task(second_outer), 1);
    EXPECT_NE(first->nodes, second->nodes);
    auto *first_root = sched.dummy_ready_queue.pop();
    auto *second_root = sched.dummy_ready_queue.pop();
    ASSERT_EQ(first_root, &first->nodes[0].slot);
    ASSERT_EQ(second_root, &second->nodes[0].slot);

    EXPECT_EQ(sched.complete_task(*first_root).stream_tasks_completed, 0);
    EXPECT_EQ(first->nodes[1].slot.fanin_refcount.load(), 1);
    EXPECT_EQ(second->nodes[1].slot.fanin_refcount.load(), 0);
    sched.retire_graph_node(*first_root);
}

TEST_F(ReplayGraphOrchestratorTest, GraphExecutionRoutesCVMixAndDummyNodes) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_graph_resource_shapes_v1"), graph_args);
    constexpr uint64_t callable_hash = 0x104784230925194eULL;

    PTO2GraphScopeResult define = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_TRUE(define.execute_block);

    L0TaskArgs args;
    MixedKernels aic;
    aic.aic_kernel_id = 1;
    ASSERT_TRUE(orch.submit_task(aic, args).task_id().is_valid());
    MixedKernels aiv;
    aiv.aiv0_kernel_id = 2;
    ASSERT_TRUE(orch.submit_task(aiv, args).task_id().is_valid());
    MixedKernels mix;
    mix.aic_kernel_id = 3;
    mix.aiv0_kernel_id = 4;
    ASSERT_TRUE(orch.submit_task(mix, args).task_id().is_valid());
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());

    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    PTO2GraphScopeResult submit = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(submit.execute_block);
    ASSERT_FALSE(orch.fatal);
    auto &outer = sm_handle->header->get_slot_state_by_task_id(submit.task_id.local());
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    ASSERT_EQ(execution->node_count, 4);
    PrepareGraph(outer);
    EXPECT_EQ(sched.activate_graph_task(outer), 4);

    PTO2TaskSlotState *aic_node = sched.ready_queues[static_cast<int>(PTO2ResourceShape::AIC)].pop();
    PTO2TaskSlotState *aiv_node = sched.ready_queues[static_cast<int>(PTO2ResourceShape::AIV)].pop();
    PTO2TaskSlotState *mix_node = sched.ready_queues[static_cast<int>(PTO2ResourceShape::MIX)].pop();
    PTO2TaskSlotState *dummy_node = sched.dummy_ready_queue.pop();
    ASSERT_NE(aic_node, nullptr);
    ASSERT_NE(aiv_node, nullptr);
    ASSERT_NE(mix_node, nullptr);
    ASSERT_NE(dummy_node, nullptr);

    EXPECT_EQ(sched.complete_task(*aic_node).stream_tasks_completed, 0);
    EXPECT_EQ(sched.complete_task(*aiv_node).stream_tasks_completed, 0);
    EXPECT_EQ(sched.complete_task(*mix_node).stream_tasks_completed, 0);
    EXPECT_EQ(sched.complete_task(*dummy_node).stream_tasks_completed, 1);
    EXPECT_EQ(outer.task_state.load(), PTO2_TASK_COMPLETED);

    sched.retire_graph_node(*aic_node);
    sched.retire_graph_node(*aiv_node);
    sched.retire_graph_node(*mix_node);
    sched.retire_graph_node(*dummy_node);
}

TEST_F(ReplayGraphOrchestratorTest, ReplayReadsDynamicScalarFromTaskArgs) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    graph_args.add_scalar(uint64_t{7});
    constexpr uint64_t callable_hash = 0x98fd637ae901204bULL;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_dynamic_scalar_v1"), graph_args);

    PTO2GraphScopeResult record = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_TRUE(record.execute_block);
    const L2TaskArgs &definition_args = graph_args;
    L0TaskArgs args;
    args.add_scalar(definition_args.scalar(0));
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    graph_args.scalar(0) = 19;
    EXPECT_EQ(graph_key, rt_graph_make_key(PTO2_GRAPH_KEY("ut_dynamic_scalar_v1"), graph_args));
    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(replay.execute_block);
    ASSERT_FALSE(orch.fatal);

    auto &outer = sm_handle->header->get_slot_state_by_task_id(1);
    ASSERT_EQ(outer.task->kind, PTO2TaskKind::GRAPH);
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    PrepareGraph(outer);
    ASSERT_EQ(sched.activate_graph_task(outer), 1);
    ASSERT_EQ(execution->nodes[0].payload.scalar_count, 1);
    EXPECT_EQ(execution->nodes[0].payload.scalars[0], 19);

    PTO2TaskSlotState *first_node = sched.dummy_ready_queue.pop();
    ASSERT_EQ(first_node, &execution->nodes[0].slot);
    EXPECT_EQ(sched.complete_task(*first_node).stream_tasks_completed, 1);
    sched.retire_graph_node(*first_node);
    pto2_graph_execution_collect_retired();

    graph_args.scalar(0) = 23;
    PTO2GraphScopeResult second_replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(second_replay.execute_block);
    auto &second_outer = sm_handle->header->get_slot_state_by_task_id(second_replay.task_id.local());
    auto *second_execution = pto2_graph_execution_from_task(*second_outer.task);
    ASSERT_EQ(second_execution, execution);
    EXPECT_TRUE(second_execution->definition_affine_reuse);
    PrepareGraph(second_outer);
    ASSERT_EQ(sched.activate_graph_task(second_outer), 1);
    EXPECT_EQ(second_execution->nodes[0].payload.scalars[0], 23);
}

TEST_F(ReplayGraphOrchestratorTest, ReplayReadsDynamicBoundaryViewFromTaskArgs) {
    orch.begin_scope();
    uint16_t storage[12]{};
    uint32_t full_shape[] = {3, 4};
    Tensor full = make_tensor_external(storage, full_shape, 2, DataType::FLOAT16);
    uint32_t row_shape[] = {1, 4};
    uint32_t row0_offset[] = {0, 0};
    Tensor row0 = full.view(row_shape, row0_offset);

    L2TaskArgs graph_args;
    graph_args.add_input(row0);
    constexpr uint64_t callable_hash = 0x72dbe260691f4c35ULL;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_dynamic_view_v1"), graph_args);
    PTO2GraphScopeResult record = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_TRUE(record.execute_block);
    L0TaskArgs args;
    args.add_input(row0);
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    uint32_t row1_offset[] = {1, 0};
    Tensor row1 = full.view(row_shape, row1_offset);
    L2TaskArgs replay_args;
    replay_args.add_input(row1);
    EXPECT_EQ(graph_key, rt_graph_make_key(PTO2_GRAPH_KEY("ut_dynamic_view_v1"), replay_args));
    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, replay_args, callable_hash);
    ASSERT_FALSE(replay.execute_block);
    ASSERT_FALSE(orch.fatal);

    auto &outer = sm_handle->header->get_slot_state_by_task_id(1);
    ASSERT_EQ(outer.task->kind, PTO2TaskKind::GRAPH);
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    PrepareGraph(outer);
    ASSERT_EQ(sched.activate_graph_task(outer), 1);
    auto &replayed = execution->nodes[0].payload;
    ASSERT_EQ(replayed.tensor_count, 1);
    EXPECT_EQ(replayed.tensors[0].start_offset, row1.start_offset);
    EXPECT_EQ(replayed.tensors[0].shapes[0], 1);
    EXPECT_EQ(replayed.tensors[0].shapes[1], 4);
}

TEST_F(ReplayGraphOrchestratorTest, ReplayReconnectsBoundaryDependencies) {
    orch.begin_scope();
    uint16_t storage[4]{};
    uint32_t shape[] = {4};
    Tensor boundary = make_tensor_external(storage, shape, 1, DataType::FLOAT16);
    L2TaskArgs graph_args;
    graph_args.add_inout(boundary);
    constexpr uint64_t callable_hash = 0xa4620738d8c04f91ULL;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_boundary_dep_v1"), graph_args);

    PTO2GraphScopeResult record = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_TRUE(record.execute_block);
    L0TaskArgs args;
    args.add_inout(boundary);
    TaskOutputTensors recorded = orch.submit_dummy_task(args);
    ASSERT_TRUE(recorded.task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(replay.execute_block);
    ASSERT_FALSE(orch.fatal);

    auto &recorded_slot = sm_handle->header->get_slot_state_by_task_id(recorded.task_id().local());
    auto &replayed_slot = sm_handle->header->get_slot_state_by_task_id(1);
    EXPECT_EQ(replayed_slot.task->kind, PTO2TaskKind::GRAPH);
    EXPECT_EQ(replayed_slot.fanin_count, 2);
    PTO2DepListEntry *fanout = recorded_slot.fanout_head.load();
    ASSERT_NE(fanout, nullptr);
    EXPECT_EQ(fanout->slot_state, &replayed_slot);

    L0TaskArgs downstream_args;
    downstream_args.add_input(boundary);
    TaskOutputTensors downstream = orch.submit_dummy_task(downstream_args);
    ASSERT_TRUE(downstream.task_id().is_valid());
    auto &downstream_slot = sm_handle->header->get_slot_state_by_task_id(downstream.task_id().local());
    EXPECT_EQ(downstream_slot.fanin_count, 2);
    PTO2DepListEntry *outer_fanout = replayed_slot.fanout_head.load();
    ASSERT_NE(outer_fanout, nullptr);
    EXPECT_EQ(outer_fanout->slot_state, &downstream_slot);
}

TEST_F(ReplayGraphOrchestratorTest, SchedulerPreparesGraphBeforeExternalDependenciesAreReady) {
    orch.begin_scope();
    uint16_t storage[4]{};
    uint32_t shape[] = {4};
    Tensor boundary = make_tensor_external(storage, shape, 1, DataType::FLOAT16);
    L2TaskArgs graph_args;
    graph_args.add_inout(boundary);
    constexpr uint64_t callable_hash = 0x72f48cd13e5a910bULL;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_prepare_before_ready_v1"), graph_args);

    ASSERT_TRUE(orch.graph_begin(graph_key, graph_args, callable_hash).execute_block);
    L0TaskArgs args;
    args.add_inout(boundary);
    TaskOutputTensors recorded = orch.submit_dummy_task(args);
    ASSERT_TRUE(recorded.task_id().is_valid());
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);

    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(replay.execute_block);
    auto &outer = sm_handle->header->get_slot_state_by_task_id(replay.task_id.local());
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);
    ASSERT_EQ(outer.fanin_count, 2);

    uint64_t queued_task_id = 0;
    PTO2TaskSlotState *prepare_slot = sched.graph_prepare_queue.pop_tagged(&queued_task_id);
    ASSERT_EQ(prepare_slot, &outer);
    EXPECT_EQ(queued_task_id, outer.task->task_id.raw);
    ASSERT_EQ(sched.prepare_graph_task(outer), PTO2GraphMaterializeResult::PREPARED);
    EXPECT_EQ(execution->state.load(), PTO2GraphExecutionState::PREPARED);
    EXPECT_EQ(sched.dummy_ready_queue.pop(), nullptr);

    EXPECT_FALSE(sched.release_fanin_and_check_ready(outer));
    auto &recorded_slot = sm_handle->header->get_slot_state_by_task_id(recorded.task_id().local());
    sched.complete_task(recorded_slot);
    PTO2TaskSlotState *ready_outer = sched.graph_ready_queue.pop();
    ASSERT_EQ(ready_outer, &outer);
    EXPECT_EQ(sched.activate_graph_task(*ready_outer), 1);
    EXPECT_EQ(execution->state.load(), PTO2GraphExecutionState::ACTIVE);
    EXPECT_EQ(sched.dummy_ready_queue.pop(), &execution->nodes[0].slot);
}

TEST_F(ReplayGraphOrchestratorTest, SchedulerMaterializesGraphInBoundedSlices) {
    orch.begin_scope();
    L2TaskArgs graph_args;
    uint64_t graph_key = rt_graph_make_key(PTO2_GRAPH_KEY("ut_bounded_graph_prepare_v1"), graph_args);
    constexpr uint64_t callable_hash = 0x73c91b5da0284fe1ULL;

    ASSERT_TRUE(orch.graph_begin(graph_key, graph_args, callable_hash).execute_block);
    for (int32_t i = 0; i < 7; ++i) {
        L0TaskArgs args;
        ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());
    }
    PTO2GraphCacheStats stats;
    FinalizeGraphDefinition(stats);
    ASSERT_EQ(stats.recorded, 1);

    PTO2GraphScopeResult replay = orch.graph_begin(graph_key, graph_args, callable_hash);
    ASSERT_FALSE(replay.execute_block);
    auto &outer = sm_handle->header->get_slot_state_by_task_id(replay.task_id.local());
    auto *execution = pto2_graph_execution_from_task(*outer.task);
    ASSERT_NE(execution, nullptr);

    EXPECT_EQ(sched.prepare_graph_task(outer, 3), PTO2GraphMaterializeResult::PENDING);
    EXPECT_EQ(execution->state.load(), PTO2GraphExecutionState::MATERIALIZING);
    EXPECT_EQ(execution->materialized_nodes, 3);
    EXPECT_EQ(execution->nodes, nullptr);
    EXPECT_EQ(sched.activate_graph_task(outer), 0);
    EXPECT_EQ(execution->materialized_nodes, 3);

    EXPECT_EQ(sched.prepare_graph_task(outer, 3), PTO2GraphMaterializeResult::PENDING);
    EXPECT_EQ(execution->materialized_nodes, 6);
    EXPECT_EQ(sched.prepare_graph_task(outer, 3), PTO2GraphMaterializeResult::PREPARED);
    EXPECT_EQ(execution->materialized_nodes, 7);
    EXPECT_EQ(execution->state.load(), PTO2GraphExecutionState::ACTIVE);
    EXPECT_EQ(execution->nodes, execution->node_storage);
    for (int32_t i = 0; i < execution->node_count; ++i) {
        EXPECT_NE(sched.dummy_ready_queue.pop(), nullptr);
    }
}
